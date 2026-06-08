# collective-sim Integration API

`collective_sim_core` is a compatibility-first Python API layered on top of `htsim_runner.py`.

- Default behavior preserves legacy semantics.
- New intra-server NVLink estimation is opt-in via `intra_server.model=nvlink_analytic`.

## Minimal Usage

```python
from collective_sim_core import predict_collective_time

scenario = {
    "cluster": {"servers": 8, "gpus_per_server": 8},
    "parallelism": {"tp": 8, "cp": 1, "dp": 8, "ep": 1},
    "topology": {"kind": "rail", "spines": 8, "servers_per_tor": 8, "paths": 8},
    "collective": {
        "kind": "allreduce",
        "tensor_bytes": 268435456,
        "domain_dims": ["DP"],
        "placement_order": ["TP", "CP", "DP", "EP"],
    },
    "intra_server": {"model": "legacy_fabric"},
}

res = predict_collective_time(scenario, repo_root="/data/collective-sim")
print(res["predicted_time_ms"])
```

## NVLink Analytic Mode

```python
scenario["intra_server"] = {
    "model": "nvlink_analytic",
    "nvlink_one_way_bw_GBps": 450.0,  # one-way bandwidth (GB/s)
    "nvlink_latency_us": 0.5,
    "nvlink_efficiency": 0.8,
}
scenario["collective"]["exclude_intra_server"] = True
```

NVLink parameter semantics:
- `nvlink_one_way_bw_GBps` is **single-direction (one-way)** bandwidth in GB/s.
- In `nvlink_analytic` mode, all NVLink parameters are used:
  - `nvlink_one_way_bw_GBps`
  - `nvlink_latency_us`
  - `nvlink_efficiency`
- `combine_rule` is selected by collective algorithm, not by `intra_server` config:
  - default `sum` for `allgather/reducescatter + hierarchical`
  - default `max` for others
  - optional override via `collective.intra_server_combine_rule`
- Constraint:
  - when `intra_server.model="nvlink_analytic"`, you must set `collective.exclude_intra_server=True`

## Common Scenario Profiles (H100/H800 + Rail/Fat-Tree)

```python
from collective_sim_core import (
    cluster_h100_8gpu,
    intra_h100_nvlink,
    predict_collective_time,
    scenario_template,
    topology_rail_optimized,
)
from collective_sim_core.schema import CollectiveConfig, ParallelismConfig

cluster = cluster_h100_8gpu(servers=16)
topology = topology_rail_optimized(servers=cluster.servers, gpus_per_server=cluster.gpus_per_server)
collective = CollectiveConfig(
    kind="allreduce",
    tensor_bytes=256 * 1024 * 1024,
    domain_dims=("DP",),
    placement_order=("TP", "CP", "DP", "EP"),
)

scenario = scenario_template(
    cluster=cluster,
    topology=topology,
    collective=collective,
    parallelism=ParallelismConfig(tp=8, cp=1, dp=16, ep=1),
    intra_server=intra_h100_nvlink(),
)

res = predict_collective_time(scenario, repo_root="/data/collective-sim")
print(res["predicted_time_ms"])
```

Scenario profile notes:
- Scenario profiles are starter templates, not calibrated golden values.
- For production use, calibrate NVLink/network params with your measurement data.
- Built-in profiles cover common 8-GPU SXM servers: H100/H800/A100/A800.

## Scenario Profile Registry (Extensible)

You can discover built-in scenario profiles and register custom ones:

```python
from collective_sim_core import (
    get_scenario_profile_builder,
    list_scenario_profiles,
    register_scenario_profile,
)
from collective_sim_core.schema import CollectiveConfig, ParallelismConfig, Scenario

print(list_scenario_profiles())  # ("a100_fattree", "a100_rail", "a800_fattree", "a800_rail", "h100_fattree", "h100_rail", "h800_fattree", "h800_rail")

builder = get_scenario_profile_builder("h100_rail")
scenario = builder(
    servers=16,
    collective=CollectiveConfig(
        kind="allreduce",
        tensor_bytes=256 * 1024 * 1024,
        domain_dims=("DP",),
        placement_order=("TP", "CP", "DP", "EP"),
    ),
    parallelism=ParallelismConfig(tp=8, cp=1, dp=16, ep=1),
)

def my_lab_scenario_profile(*, servers, collective, parallelism, **kwargs) -> Scenario:
    # build your own scenario here...
    return builder(servers=servers, collective=collective, parallelism=parallelism)

register_scenario_profile("my_lab_h100_rail", my_lab_scenario_profile)
```

## Output Contract

- `predicted_time_ms`: final predicted time
- `breakdown.network_ms`: network makespan from legacy runner
- `breakdown.intra_server_ms`: analytic intra-server estimate (0 for legacy mode)
- `breakdown.combine_rule`: max/sum/phase
- `assumptions`: key modeling assumptions

## tensor_bytes Semantics

The meaning of `tensor_bytes` differs by collective kind, following the same convention as
`htsim_runner.py`:

| `kind` | `tensor_bytes` represents | Ring traffic per rank |
|---|---|---|
| `allreduce` | Total tensor size `T` | `2*(n-1)/n * T` |
| `allgather` | **Per-rank input** `x` (output = `x*n`) | `(n-1)*x` |
| `reducescatter` | **Per-rank output** `x` (input = `x*n`) | `(n-1)*x` |
| `alltoall` | Total outgoing data per rank `T` (per-peer = `T/n`) | `(n-1)/n * T` |
| `p2p` / `t2g` | Direct transfer size | — |

> **Note:** `allgather` and `reducescatter` use per-rank (not total) tensor size.
> For a TP=8 allgather over a 1 GiB activation, set `tensor_bytes = 1GiB / 8 = 128 MiB`.

## Network Config

`network` is optional; all fields have sensible defaults for 400G clusters:

```python
scenario["network"] = {
    "linkspeed_mbps": 400000,  # per-link bandwidth in Mbps (default: 400 Gbps)
    "mtu": 9216,               # jumbo frame MTU in bytes
    "q": 256,                  # queue depth in packets
    "cwnd": 64,                # congestion window in packets
}
```

To model 800G NICs, set `linkspeed_mbps=800000`.

## Required Collective Fields

`collective.domain_dims` and `collective.placement_order` are required.
They directly define communication scope and rank-to-layout mapping.

For `kind="p2p"`, you can optionally set:

- `collective.p2p_src_index`
- `collective.p2p_dst_index`
- `collective.p2p_direction` (`0->1` / `1->0` / `bidir`)
