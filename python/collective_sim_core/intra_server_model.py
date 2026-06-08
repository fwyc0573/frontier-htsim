from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from .layout import (
    compute_g_in_group as _layout_compute_g,
    compute_n_group as _layout_compute_n,
)
from .schema import Scenario


def _dim_sizes(scenario: Scenario) -> Dict[str, int]:
    return {
        "TP": scenario.parallelism.tp,
        "CP": scenario.parallelism.cp,
        "DP": scenario.parallelism.dp,
        "EP": scenario.parallelism.ep,
    }


def _compute_g_in_group(scenario: Scenario) -> int:
    """
    GPUs per server that participate in a single domain communication group.

    The placement_order convention: the FIRST dim is fastest-varying (innermost),
    the LAST dim is slowest-varying (outermost, crosses server boundaries).
    "Intra-server dims" are the fastest dims whose cumulative product does not
    exceed gpus_per_server.  g is the product of dim sizes that are BOTH in
    domain_dims AND intra-server.

    Examples with placement_order=["TP","CP","EP","DP"], gpus_per_server=8:
      domain=["DP"]  →  DP is cross-server  →  g=1  (0 intra-server NVLink needed)
      domain=["EP"]  →  EP is intra-server  →  g=8  (full NVLink within server)
    """
    participant_ranks = tuple(int(rank) for rank in (scenario.collective.participant_ranks or ()))
    if participant_ranks:
        gpus_per_server = max(int(scenario.cluster.gpus_per_server), 1)
        counts_per_server: Dict[int, int] = {}
        for rank in participant_ranks:
            server_idx = rank // gpus_per_server
            counts_per_server[server_idx] = counts_per_server.get(server_idx, 0) + 1
        if not counts_per_server:
            return 1
        n = sum(counts_per_server.values())
        s = len(counts_per_server)
        return max(1, int(round(float(n) / float(s))))

    return _layout_compute_g(
        placement_order=tuple(scenario.collective.placement_order or ()),
        domain_dims=tuple(scenario.collective.domain_dims or ()),
        dim_sizes=_dim_sizes(scenario),
        gpus_per_server=scenario.cluster.gpus_per_server,
    )


def _compute_n_group(scenario: Scenario) -> int:
    """Total number of ranks in the domain group."""
    participant_ranks = tuple(int(rank) for rank in (scenario.collective.participant_ranks or ()))
    if participant_ranks:
        return len(participant_ranks)
    return _layout_compute_n(
        placement_order=tuple(scenario.collective.placement_order or ()),
        domain_dims=tuple(scenario.collective.domain_dims or ()),
        dim_sizes=_dim_sizes(scenario),
        gpus_per_server=scenario.cluster.gpus_per_server,
    )


def estimate_intra_server_ms(scenario: Scenario) -> Tuple[float, Dict[str, Any]]:
    cfg = scenario.intra_server

    if cfg.model in ("legacy_fabric", "ignore"):
        return 0.0, {
            "intra_server_model": cfg.model,
            "note": "intra-server time not added; preserves legacy semantics",
        }

    g = _compute_g_in_group(scenario)
    if g <= 1:
        return 0.0, {
            "intra_server_model": cfg.model,
            "note": "domain group has ≤1 GPU per server; no intra-server NVLink transfer",
        }

    steps = _estimate_steps(scenario)
    bytes_per_rank = _estimate_intra_bytes_per_rank(scenario)
    bw_Bps = (cfg.nvlink_one_way_bw_GBps * 1e9) * max(cfg.nvlink_efficiency, 1e-6)
    transfer_us = (bytes_per_rank / bw_Bps) * 1e6
    latency_us = steps * cfg.nvlink_latency_us
    allreduce_launch_overhead_us = 0.0
    alltoall_launch_overhead_us = 0.0
    if scenario.collective.kind == "allreduce":
        allreduce_launch_overhead_us = (
            steps * cfg.nvlink_allreduce_launch_overhead_us
        )
    if scenario.collective.kind == "alltoall":
        alltoall_launch_overhead_us = (
            steps * cfg.nvlink_alltoall_launch_overhead_us
        )
    total_ms = (
        transfer_us
        + latency_us
        + allreduce_launch_overhead_us
        + alltoall_launch_overhead_us
    ) / 1000.0
    return total_ms, {
        "intra_server_model": cfg.model,
        "nvlink_bw_direction": "one_way",
        "nvlink_one_way_bw_GBps": cfg.nvlink_one_way_bw_GBps,
        "nvlink_efficiency": cfg.nvlink_efficiency,
        "nvlink_latency_us": cfg.nvlink_latency_us,
        "nvlink_allreduce_launch_overhead_us": (
            cfg.nvlink_allreduce_launch_overhead_us
        ),
        "nvlink_alltoall_launch_overhead_us": (
            cfg.nvlink_alltoall_launch_overhead_us
        ),
        "effective_nvlink_bw_Bps": bw_Bps,
        "estimated_bytes_per_rank": bytes_per_rank,
        "estimated_steps": steps,
        "estimated_allreduce_launch_overhead_us": allreduce_launch_overhead_us,
        "estimated_alltoall_launch_overhead_us": alltoall_launch_overhead_us,
        "g_in_group": g,
        "num_servers": scenario.cluster.servers,
    }


def _estimate_intra_bytes_per_rank(scenario: Scenario) -> float:
    # g: GPUs per server in the domain group (1 = fully cross-server, no NVLink needed)
    # n: total ranks in the domain group
    # S: servers spanned by the domain group = n / g
    g = float(_compute_g_in_group(scenario))
    n = float(_compute_n_group(scenario))
    S = max(n / g, 1.0)
    x = float(scenario.collective.tensor_bytes)
    collective = scenario.collective.kind

    if collective == "allreduce":
        # Flat ring allreduce over n ranks:
        # total bytes/rank = 2*(n-1)/n*x.
        # Intra-server share depends on whether the ring is fully inside one server.
        if S <= 1.0:
            return (2.0 * (n - 1.0) / n) * x
        return (2.0 * (n - 1.0) / n) * ((g - 1.0) / g) * x

    if collective == "allgather":
        if scenario.collective.allgather_model in ("hierarchical", "hierarchical_ring"):
            # NVLink phase after inter-server NIC allgather:
            # each GPU holds S*x bytes; ring-allgather over g GPUs → (g-1)*S*x.
            return (g - 1.0) * S * x
        # Flat ring allgather across n ranks with g GPUs per server.
        # Each GPU always sends to its fixed right neighbor.
        # When S=1 (fully intra-server): all g ring edges are NVLink (including
        # the wrap-around edge GPU[g-1]→GPU[0]), so all (g-1) steps are NVLink.
        # When S>1: fraction (g-1)/g of GPUs have an intra-server right neighbor,
        # each forwarding all (n-1) steps over NVLink.
        if S <= 1.0:
            return (g - 1.0) * x
        return (g - 1.0) / g * (n - 1.0) * x

    if collective == "reducescatter":
        if scenario.collective.reducescatter_model in ("hierarchical", "hierarchical_ring"):
            # NVLink phase before inter-server NIC reduce-scatter:
            # ring-reducescatter over g GPUs of g*S*x bytes → (g-1)*S*x per GPU.
            return (g - 1.0) * S * x
        # Flat ring: same edge structure as allgather (symmetric traffic).
        if S <= 1.0:
            return (g - 1.0) * x
        return (g - 1.0) / g * (n - 1.0) * x

    if collective == "alltoall":
        # Legacy runner semantics: each (src,dst) pair sends ceil(x/n) bytes, so
        # intra-server bytes/rank ≈ (g-1) * x / n.
        return ((g - 1.0) / n) * x

    if collective == "p2p":
        # p2p: direct transfer between 2 ranks
        if S <= 1.0 and n >= 2:
            return x           # intra-server p2p
        return 0.0             # inter-server, NVLink not needed

    if collective == "t2g":
        return 0.0             # t2g is always cross-pod, no intra-server NVLink

    raise ValueError(f"Unsupported collective: {collective}")


def _estimate_steps(scenario: Scenario) -> int:
    """Number of sequential NVLink steps for latency estimation."""
    g = max(_compute_g_in_group(scenario), 1)
    n = max(_compute_n_group(scenario), 1)
    S = max(float(n) / float(g), 1.0)
    collective = scenario.collective.kind

    if collective == "allreduce":
        if S <= 1.0:
            return 2 * (n - 1)
        return int(math.ceil(2.0 * (n - 1.0) * (g - 1.0) / g))
    if collective in ("allgather", "reducescatter"):
        model = (
            scenario.collective.allgather_model
            if collective == "allgather"
            else scenario.collective.reducescatter_model
        )
        if model in ("hierarchical", "hierarchical_ring"):
            return g - 1
        if S <= 1.0:
            return n - 1
        return int(math.ceil((n - 1.0) * (g - 1.0) / g))
    if collective == "alltoall":
        return max(1, g - 1)          # unified, no longer model-dependent
    return 1                           # p2p, t2g, etc.
