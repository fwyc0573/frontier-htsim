from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


CollectiveType = str
TopologyType = str
IntraServerModel = str
CombineRule = str
SUPPORTED_TOPOLOGY_KINDS = ("rail", "rail3", "fattree", "fattree3")


@dataclass(frozen=True)
class ClusterConfig:
    servers: int
    gpus_per_server: int


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    cp: int = 1
    dp: int = 1
    ep: int = 1


@dataclass(frozen=True)
class TopologyConfig:
    kind: TopologyType = "rail"
    spines: int = 8
    servers_per_tor: int = 16
    paths: int = 8
    podsize: int = 0
    tor_down: int = 16
    tor_up: int = 0
    fattree_oversub: int = 1
    latency_ns: int = 800
    switch_latency_ns: int = 800


@dataclass(frozen=True)
class CollectiveConfig:
    kind: CollectiveType = "allreduce"
    # Meaning of tensor_bytes depends on the collective kind:
    #
    #   allreduce:     total tensor size T (bytes). Each rank holds T bytes before and
    #                  after the operation. Ring traffic per rank ≈ 2*(n-1)/n * T.
    #
    #   allgather:     per-rank INPUT size x (bytes). Each rank contributes x bytes;
    #                  after the operation every rank holds x*n bytes.
    #                  Ring traffic per rank = (n-1)*x.
    #
    #   reducescatter: per-rank OUTPUT size x (bytes). Each rank starts with x*n bytes;
    #                  after the operation every rank holds x bytes (its reduced shard).
    #                  Ring traffic per rank = (n-1)*x.
    #
    #   alltoall:      total outgoing data per rank T (bytes), sent evenly to all n peers.
    #                  Per-peer volume = T/n. Traffic per rank = (n-1)/n * T.
    #
    #   p2p / t2g:     direct transfer size (bytes).
    tensor_bytes: int = 1 << 20
    # Must be explicitly provided by users because it defines communication scope.
    domain_dims: Optional[Tuple[str, ...]] = None
    # Must be explicitly provided by users because it defines rank/layout mapping.
    placement_order: Optional[Tuple[str, ...]] = None
    exclude_intra_server: bool = True
    use_triggers: bool = True
    allreduce_model: str = "nccl_ring"
    allgather_model: str = "nccl_ring"
    reducescatter_model: str = "nccl_ring"
    alltoall_model: str = "nccl_pairwise"
    alltoall_channels: int = 8
    alltoall_chunk_bytes: int = 8_000_000   # 8 MB; matches medium-message NCCL tuning
    alltoall_chunk_inflight_per_peer: int = 2
    nchannels: int = 8   # NCCL H800 rail default: 8 independent ring channels (nccl_ring only; ring_steps ignores this)
    # p2p selector within each domain group.
    p2p_src_index: int = 0
    p2p_dst_index: int = 1
    p2p_direction: str = "0->1"
    # Optional override for network/intra combine rule at collective-algorithm level.
    # If unset, predictor chooses by algorithm semantics.
    intra_server_combine_rule: Optional[CombineRule] = None
    # Optional explicit participant ranks in global rank space.
    # When set, runner will simulate this specific communication group instead of
    # enumerating all domain groups.
    participant_ranks: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class NetworkConfig:
    # Per-link bandwidth in Mbps. Default 400_000 = 400 Gbps (one 400G NIC port).
    # Set to 800_000 for 800G, 200_000 for 200G, etc.
    linkspeed_mbps: int = 400_000
    mtu: int = 9216              # MTU in bytes (jumbo frames; used for switch queue sizing)
    q: int = 64                  # switch queue depth in packets (≈ 1.2x BDP at 400G/mtu=9216/RTT≈9.6μs)
    cwnd: Optional[int] = None   # congestion window in packets; None = auto-derive from linkspeed

    def __post_init__(self) -> None:
        if self.cwnd is None:
            # BDP-based: cwnd ≈ linkspeed × RTT / (packet_size × 8), calibrated to
            # mtu=9216, RTT≈9.6μs (2-hop rail).  Gives 100G→16, 200G→32, 400G→64, 800G→128.
            object.__setattr__(self, "cwnd", max(8, self.linkspeed_mbps // 6250))


@dataclass(frozen=True)
class RunnerConfig:
    end_us: int = 20_000_000
    seed: int = 1
    build: bool = False
    progress: bool = False
    print_args: bool = False
    stop_on_finished: bool = False
    heartbeat_s: int = 5
    out_dir: str = ""


@dataclass(frozen=True)
class IntraServerConfig:
    model: IntraServerModel = "legacy_fabric"  # legacy_fabric | ignore | nvlink_analytic
    # Single-direction (one-way) NVLink bandwidth in GB/s.
    nvlink_one_way_bw_GBps: float = 450.0
    nvlink_latency_us: float = 0.5
    nvlink_efficiency: float = 0.8
    nvlink_allreduce_launch_overhead_us: float = 50.0
    nvlink_alltoall_launch_overhead_us: float = 0.0


@dataclass(frozen=True)
class Scenario:
    cluster: ClusterConfig
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    collective: CollectiveConfig = field(default_factory=CollectiveConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    intra_server: IntraServerConfig = field(default_factory=IntraServerConfig)

    def validate(self) -> None:
        nodes = self.parallelism.tp * self.parallelism.cp * self.parallelism.dp * self.parallelism.ep
        expected_nodes = self.cluster.servers * self.cluster.gpus_per_server
        if nodes != expected_nodes:
            raise ValueError(
                f"nodes mismatch: tp*cp*dp*ep={nodes} but servers*gpus_per_server={expected_nodes}"
            )
        if self.cluster.gpus_per_server % self.parallelism.tp != 0:
            raise ValueError("gpus_per_server must be divisible by tp")
        if self.topology.kind not in SUPPORTED_TOPOLOGY_KINDS:
            raise ValueError(
                f"unsupported topology.kind={self.topology.kind!r}; "
                f"supported={SUPPORTED_TOPOLOGY_KINDS}"
            )
        if self.intra_server.model not in ("legacy_fabric", "ignore", "nvlink_analytic"):
            raise ValueError("intra_server.model must be one of legacy_fabric|ignore|nvlink_analytic")
        if self.collective.intra_server_combine_rule is not None and self.collective.intra_server_combine_rule not in (
            "max",
            "sum",
            "phase",
        ):
            raise ValueError("collective.intra_server_combine_rule must be one of max|sum|phase")
        if self.collective.kind == "p2p":
            if self.collective.p2p_direction not in ("0->1", "1->0", "bidir"):
                raise ValueError("collective.p2p_direction must be one of: 0->1 | 1->0 | bidir")
            if self.collective.p2p_src_index < 0:
                raise ValueError("collective.p2p_src_index must be >= 0")
            if self.collective.p2p_dst_index < 0:
                raise ValueError("collective.p2p_dst_index must be >= 0")
            if self.collective.p2p_src_index == self.collective.p2p_dst_index:
                raise ValueError("collective.p2p_src_index and collective.p2p_dst_index must differ")
        if self.collective.participant_ranks is not None:
            participant_ranks = tuple(int(rank) for rank in self.collective.participant_ranks)
            if not participant_ranks:
                raise ValueError("collective.participant_ranks must not be empty when provided")
            if len(participant_ranks) != len(set(participant_ranks)):
                raise ValueError("collective.participant_ranks must not contain duplicates")
            if any(rank < 0 for rank in participant_ranks):
                raise ValueError("collective.participant_ranks must contain non-negative rank ids")
        if (
            self.collective.kind == "allgather"
            and self.collective.allgather_model in ("hierarchical", "hierarchical_ring")
            and self.collective.intra_server_combine_rule not in ("sum", None)
        ):
            raise ValueError(
                "for allgather with allgather_model='hierarchical'/'hierarchical_ring', "
                "collective.intra_server_combine_rule must be 'sum' (or unset)"
            )
        if (
            self.collective.kind == "reducescatter"
            and self.collective.reducescatter_model in ("hierarchical", "hierarchical_ring")
            and self.collective.intra_server_combine_rule not in ("sum", None)
        ):
            raise ValueError(
                "for reducescatter with reducescatter_model='hierarchical'/'hierarchical_ring', "
                "collective.intra_server_combine_rule must be 'sum' (or unset)"
            )
        if not self.collective.domain_dims:
            raise ValueError("collective.domain_dims must be explicitly provided and non-empty")
        if not self.collective.placement_order:
            raise ValueError("collective.placement_order must be explicitly provided and non-empty")
        _VALID_DIMS = {"TP", "CP", "DP", "EP"}
        for d in self.collective.placement_order:
            if d not in _VALID_DIMS:
                raise ValueError(
                    f"invalid dimension {d!r} in placement_order; must be one of {sorted(_VALID_DIMS)}"
                )
        for d in self.collective.domain_dims:
            if d not in _VALID_DIMS:
                raise ValueError(
                    f"invalid dimension {d!r} in domain_dims; must be one of {sorted(_VALID_DIMS)}"
                )
        _dim_sizes = {"TP": self.parallelism.tp, "CP": self.parallelism.cp,
                       "DP": self.parallelism.dp, "EP": self.parallelism.ep}
        _order_set = set(self.collective.placement_order)
        for dim, size in _dim_sizes.items():
            if size > 1 and dim not in _order_set:
                raise ValueError(
                    f"dimension {dim} has size {size} > 1 but is not in placement_order "
                    f"{list(self.collective.placement_order)}; all active dimensions must be included"
                )
        if self.intra_server.model == "nvlink_analytic":
            if not bool(self.collective.exclude_intra_server):
                raise ValueError(
                    "when intra_server.model='nvlink_analytic', collective.exclude_intra_server must be True"
                )
            if self.intra_server.nvlink_one_way_bw_GBps <= 0:
                raise ValueError("intra_server.nvlink_one_way_bw_GBps must be > 0 (one-way bandwidth, GB/s)")
            if self.intra_server.nvlink_latency_us < 0:
                raise ValueError("intra_server.nvlink_latency_us must be >= 0")
            if not (0 < self.intra_server.nvlink_efficiency <= 1):
                raise ValueError("intra_server.nvlink_efficiency must be in (0, 1]")
            if self.intra_server.nvlink_allreduce_launch_overhead_us < 0:
                raise ValueError(
                    "intra_server.nvlink_allreduce_launch_overhead_us must be >= 0"
                )
            if self.intra_server.nvlink_alltoall_launch_overhead_us < 0:
                raise ValueError(
                    "intra_server.nvlink_alltoall_launch_overhead_us must be >= 0"
                )

    def to_runner_spec(self) -> Dict[str, Any]:
        self.validate()
        nodes = self.cluster.servers * self.cluster.gpus_per_server
        # htsim_runner.py reads `seed` from top-level spec["seed"] (primary path).
        return {
            "seed": self.runner.seed,
            "topology": {
                "type": self.topology.kind,
                "spines": self.topology.spines,
                "servers_per_tor": self.topology.servers_per_tor,
                "podsize": self.topology.podsize,
                "tor_down": self.topology.tor_down,
                "tor_up": self.topology.tor_up,
                "oversub": self.topology.fattree_oversub,
                "latency_ns": self.topology.latency_ns,
                "switch_latency_ns": self.topology.switch_latency_ns,
            },
            # htsim_runner reads nodes/servers/gpus_per_server from job.* (primary path).
            "job": {
                "nodes": nodes,
                "servers": self.cluster.servers,
                "gpus_per_server": self.cluster.gpus_per_server,
            },
            # htsim_runner reads tp/cp/dp/ep from parallel.* (primary path).
            "parallel": {
                "tp": self.parallelism.tp,
                "cp": self.parallelism.cp,
                "dp": self.parallelism.dp,
                "ep": self.parallelism.ep,
            },
            # htsim_runner reads collective params from collective.* (primary path).
            "collective": {
                "type": self.collective.kind,
                "tensor_bytes": self.collective.tensor_bytes,
                "domain_dims": list(self.collective.domain_dims or ()),
                "placement_order": list(self.collective.placement_order or ()),
                "exclude_intra_server": self.collective.exclude_intra_server,
                "use_triggers": self.collective.use_triggers,
                "allreduce_model": self.collective.allreduce_model,
                "allgather_model": self.collective.allgather_model,
                "reducescatter_model": self.collective.reducescatter_model,
                "alltoall_model": self.collective.alltoall_model,
                "alltoall_channels": self.collective.alltoall_channels,
                "alltoall_chunk_bytes": self.collective.alltoall_chunk_bytes,
                "alltoall_chunk_inflight_per_peer": self.collective.alltoall_chunk_inflight_per_peer,
                "nchannels": self.collective.nchannels,
                "src_index": self.collective.p2p_src_index,
                "dst_index": self.collective.p2p_dst_index,
                "direction": self.collective.p2p_direction,
                "participant_ranks": list(self.collective.participant_ranks or ()),
            },
            "network": asdict(self.network),
            "sim": {"end_us": self.runner.end_us},
            "runner": {
                "build": self.runner.build,
                "progress": self.runner.progress,
                "print_args": self.runner.print_args,
                "stop_on_finished": self.runner.stop_on_finished,
                "heartbeat_s": self.runner.heartbeat_s,
                "paths": self.topology.paths,
                "out_dir": self.runner.out_dir,
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Scenario":
        collective_raw = dict(data.get("collective", {}))
        # Enforce explicit declaration to avoid accidental semantic drift.
        if "domain_dims" not in collective_raw:
            raise ValueError("collective.domain_dims is required")
        if "placement_order" not in collective_raw:
            raise ValueError("collective.placement_order is required")
        cluster = ClusterConfig(**data["cluster"])
        parallelism = ParallelismConfig(**data.get("parallelism", {}))
        topology = TopologyConfig(**data.get("topology", {}))
        collective = CollectiveConfig(**collective_raw)
        network = NetworkConfig(**data.get("network", {}))
        runner = RunnerConfig(**data.get("runner", {}))
        intra_server = IntraServerConfig(**data.get("intra_server", {}))
        return Scenario(
            cluster=cluster,
            parallelism=parallelism,
            topology=topology,
            collective=collective,
            network=network,
            runner=runner,
            intra_server=intra_server,
        )
