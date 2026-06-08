from __future__ import annotations

from typing import Callable, Dict, Tuple

from .schema import (
    ClusterConfig,
    CollectiveConfig,
    IntraServerConfig,
    NetworkConfig,
    ParallelismConfig,
    RunnerConfig,
    Scenario,
    TopologyConfig,
)

ScenarioProfileBuilder = Callable[..., Scenario]


def cluster_h100_8gpu(servers: int) -> ClusterConfig:
    """Common cluster template: H100 server with 8 GPUs."""
    return ClusterConfig(servers=servers, gpus_per_server=8)


def cluster_h800_8gpu(servers: int) -> ClusterConfig:
    """Common cluster template: H800 server with 8 GPUs."""
    return ClusterConfig(servers=servers, gpus_per_server=8)


def cluster_a100_8gpu(servers: int) -> ClusterConfig:
    """Common cluster template: A100 server with 8 GPUs."""
    return ClusterConfig(servers=servers, gpus_per_server=8)


def cluster_a800_8gpu(servers: int) -> ClusterConfig:
    """Common cluster template: A800 server with 8 GPUs."""
    return ClusterConfig(servers=servers, gpus_per_server=8)


def intra_h100_nvlink() -> IntraServerConfig:
    """
    Starter NVLink profile for H100-class servers.
    nvlink_one_way_bw_GBps is single-direction (one-way) bandwidth in GB/s.
    These are baseline defaults and should be calibrated against real measurements.
    """
    return IntraServerConfig(
        model="nvlink_analytic",
        nvlink_one_way_bw_GBps=450.0,
        nvlink_latency_us=0.5,
        nvlink_efficiency=0.8,
    )


def intra_h800_nvlink() -> IntraServerConfig:
    """
    Starter NVLink profile for H800-class servers.
    nvlink_one_way_bw_GBps is single-direction (one-way) bandwidth in GB/s.
    These are baseline defaults and should be calibrated against real measurements.
    """
    return IntraServerConfig(
        model="nvlink_analytic",
        nvlink_one_way_bw_GBps=200.0,
        nvlink_latency_us=0.6,
        nvlink_efficiency=0.8,
    )


def intra_a100_nvlink() -> IntraServerConfig:
    """
    Starter NVLink profile for A100 SXM servers.
    nvlink_one_way_bw_GBps is single-direction (one-way) bandwidth in GB/s.
    These are baseline defaults and should be calibrated against real measurements.
    """
    return IntraServerConfig(
        model="nvlink_analytic",
        nvlink_one_way_bw_GBps=300.0,
        nvlink_latency_us=0.8,
        nvlink_efficiency=0.78,
    )


def intra_a800_nvlink() -> IntraServerConfig:
    """
    Starter NVLink profile for A800 SXM servers.
    nvlink_one_way_bw_GBps is single-direction (one-way) bandwidth in GB/s.
    These are baseline defaults and should be calibrated against real measurements.
    """
    return IntraServerConfig(
        model="nvlink_analytic",
        nvlink_one_way_bw_GBps=180.0,
        nvlink_latency_us=0.9,
        nvlink_efficiency=0.76,
    )


def topology_rail_optimized(
    *,
    servers: int,
    gpus_per_server: int = 8,
    spines: int | None = None,
    servers_per_tor: int | None = None,
    latency_ns: int = 800,
    switch_latency_ns: int = 800,
) -> TopologyConfig:
    """
    Starter rail-optimized topology template.
    Defaults are conservative and intended for quick integration.
    """
    rail_spines = spines if spines is not None else max(8, gpus_per_server)
    spt = servers_per_tor if servers_per_tor is not None else min(8, max(1, servers))
    return TopologyConfig(
        kind="rail",
        spines=rail_spines,
        servers_per_tor=spt,
        paths=rail_spines,
        latency_ns=latency_ns,
        switch_latency_ns=switch_latency_ns,
    )


def topology_fat_tree(
    *,
    servers: int,
    gpus_per_server: int = 8,
    oversub: int = 2,
    tor_down: int = 16,
    latency_ns: int = 800,
    switch_latency_ns: int = 800,
) -> TopologyConfig:
    """
    Starter 3-tier fat-tree template.
    """
    if oversub <= 0:
        raise ValueError("oversub must be > 0")
    tor_up = max(1, tor_down // oversub)
    nodes = servers * gpus_per_server
    podsize = max(1, nodes // 2)
    return TopologyConfig(
        kind="fattree3",
        spines=0,
        servers_per_tor=0,
        paths=tor_up,
        podsize=podsize,
        tor_down=tor_down,
        tor_up=tor_up,
        fattree_oversub=oversub,
        latency_ns=latency_ns,
        switch_latency_ns=switch_latency_ns,
    )


def scenario_template(
    *,
    cluster: ClusterConfig,
    topology: TopologyConfig,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig | None = None,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
    intra_server: IntraServerConfig | None = None,
) -> Scenario:
    """
    Build a Scenario from reusable config templates.
    """
    return Scenario(
        cluster=cluster,
        parallelism=parallelism or ParallelismConfig(),
        topology=topology,
        collective=collective,
        network=network or NetworkConfig(),
        runner=runner or RunnerConfig(),
        intra_server=intra_server or IntraServerConfig(),
    )


def scenario_profile_h100_rail(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: H100 8-GPU servers + rail-optimized topology + H100 NVLink."""
    cluster = cluster_h100_8gpu(servers)
    topology = topology_rail_optimized(servers=cluster.servers, gpus_per_server=cluster.gpus_per_server)
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_h100_nvlink(),
    )


def scenario_profile_h800_fattree(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    oversub: int = 2,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: H800 8-GPU servers + fat-tree topology + H800 NVLink."""
    cluster = cluster_h800_8gpu(servers)
    topology = topology_fat_tree(
        servers=cluster.servers,
        gpus_per_server=cluster.gpus_per_server,
        oversub=oversub,
    )
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_h800_nvlink(),
    )


def scenario_profile_h800_rail(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: H800 8-GPU servers + rail-optimized topology + H800 NVLink."""
    cluster = cluster_h800_8gpu(servers)
    topology = topology_rail_optimized(servers=cluster.servers, gpus_per_server=cluster.gpus_per_server)
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_h800_nvlink(),
    )


def scenario_profile_h100_fattree(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    oversub: int = 2,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: H100 8-GPU servers + fat-tree topology + H100 NVLink."""
    cluster = cluster_h100_8gpu(servers)
    topology = topology_fat_tree(
        servers=cluster.servers,
        gpus_per_server=cluster.gpus_per_server,
        oversub=oversub,
    )
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_h100_nvlink(),
    )


def scenario_profile_a100_rail(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: A100 8-GPU servers + rail-optimized topology + A100 NVLink."""
    cluster = cluster_a100_8gpu(servers)
    topology = topology_rail_optimized(servers=cluster.servers, gpus_per_server=cluster.gpus_per_server)
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_a100_nvlink(),
    )


def scenario_profile_a100_fattree(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    oversub: int = 2,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: A100 8-GPU servers + fat-tree topology + A100 NVLink."""
    cluster = cluster_a100_8gpu(servers)
    topology = topology_fat_tree(
        servers=cluster.servers,
        gpus_per_server=cluster.gpus_per_server,
        oversub=oversub,
    )
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_a100_nvlink(),
    )


def scenario_profile_a800_rail(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: A800 8-GPU servers + rail-optimized topology + A800 NVLink."""
    cluster = cluster_a800_8gpu(servers)
    topology = topology_rail_optimized(servers=cluster.servers, gpus_per_server=cluster.gpus_per_server)
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_a800_nvlink(),
    )


def scenario_profile_a800_fattree(
    *,
    servers: int,
    collective: CollectiveConfig,
    parallelism: ParallelismConfig,
    oversub: int = 2,
    network: NetworkConfig | None = None,
    runner: RunnerConfig | None = None,
) -> Scenario:
    """Scenario profile: A800 8-GPU servers + fat-tree topology + A800 NVLink."""
    cluster = cluster_a800_8gpu(servers)
    topology = topology_fat_tree(
        servers=cluster.servers,
        gpus_per_server=cluster.gpus_per_server,
        oversub=oversub,
    )
    return scenario_template(
        cluster=cluster,
        topology=topology,
        collective=collective,
        parallelism=parallelism,
        network=network,
        runner=runner,
        intra_server=intra_a800_nvlink(),
    )


_SCENARIO_PROFILE_REGISTRY: Dict[str, ScenarioProfileBuilder] = {
    "h100_rail":    scenario_profile_h100_rail,
    "h100_fattree": scenario_profile_h100_fattree,
    "h800_rail":    scenario_profile_h800_rail,
    "h800_fattree": scenario_profile_h800_fattree,
    "a100_rail":    scenario_profile_a100_rail,
    "a100_fattree": scenario_profile_a100_fattree,
    "a800_rail":    scenario_profile_a800_rail,
    "a800_fattree": scenario_profile_a800_fattree,
}


def list_scenario_profiles() -> Tuple[str, ...]:
    """Return all registered scenario profile names."""
    return tuple(sorted(_SCENARIO_PROFILE_REGISTRY.keys()))


def get_scenario_profile_builder(name: str) -> ScenarioProfileBuilder:
    """Fetch a scenario profile builder by name."""
    if name not in _SCENARIO_PROFILE_REGISTRY:
        raise KeyError(f"unknown scenario profile {name!r}; available={list_scenario_profiles()}")
    return _SCENARIO_PROFILE_REGISTRY[name]


def register_scenario_profile(name: str, builder: ScenarioProfileBuilder, *, overwrite: bool = False) -> None:
    """
    Register a custom scenario profile builder at runtime.
    This allows downstream simulators to inject their own environment-specific profiles.
    """
    if not overwrite and name in _SCENARIO_PROFILE_REGISTRY:
        raise ValueError(f"scenario profile {name!r} already exists; pass overwrite=True to replace")
    _SCENARIO_PROFILE_REGISTRY[name] = builder


