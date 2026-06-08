#!/usr/bin/env python3
"""
quickstart.py — minimal usage examples for collective-sim.

Run from the repo root:
    python examples/quickstart.py

Three progressively richer patterns are shown:
  1. Single one-shot call (dict API)
  2. Fixed cluster/topology, iterate over collectives  (recommended pattern)
  3. Load a built-in scenario profile and iterate
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from collective_sim_core import predict_collective_time


def _print_comm_config(
    *,
    cluster: dict,
    parallelism: dict,
    placement_order: list[str] | tuple[str, ...],
    collective_kind: str,
    domain_dims: list[str] | tuple[str, ...],
    tensor_bytes: int,
    prefix: str = "  ",
) -> None:
    total_gpus = int(cluster["servers"]) * int(cluster["gpus_per_server"])
    po = " -> ".join(str(x) for x in placement_order)
    dd = ", ".join(str(x) for x in domain_dims)
    print(
        f"{prefix}cluster          : servers={cluster['servers']}, "
        f"gpus_per_server={cluster['gpus_per_server']} (total={total_gpus})"
    )
    print(
        f"{prefix}parallelism      : TP={parallelism['tp']}, CP={parallelism['cp']}, "
        f"DP={parallelism['dp']}, EP={parallelism['ep']}"
    )
    print(f"{prefix}placement_order  : {po}")
    print(f"{prefix}collective       : {collective_kind}")
    print(f"{prefix}domain_dims      : {dd}")
    print(f"{prefix}tensor_bytes     : {tensor_bytes} ({tensor_bytes / (1 << 20):.2f} MiB)")


# ---------------------------------------------------------------------------
# Example 1: one-shot call — the simplest possible usage
# ---------------------------------------------------------------------------
def example1_onshot():
    print("=" * 60)
    print("Example 1: one-shot prediction (dict API)")
    print("=" * 60)

    scenario = {
        "cluster": {"servers": 2, "gpus_per_server": 8},
        "parallelism": {"tp": 1, "cp": 1, "dp": 2, "ep": 8},
        "topology": {
            "kind": "rail",
            "spines": 8,
            "servers_per_tor": 2,
            "paths": 8,
        },
        "collective": {
            "kind": "allreduce",
            # tensor_bytes for allreduce = total tensor size T
            "tensor_bytes": 40 * 1024 * 1024,     # 40 MiB
            "domain_dims": ["DP"],                # communicate across DP dimension
            "placement_order": ["TP", "CP", "EP", "DP"],
        },
        # linkspeed_mbps: per-link bandwidth.  400_000 = 400 Gbps (H100/H800 default).
        # Change to 200_000 for 200G NICs, 800_000 for 800G NICs, etc.
        "network": {
            "linkspeed_mbps": 400_000,
            "mtu": 9216,
            "q": 256,
            "cwnd": 64,
        },
        "intra_server": {"model": "legacy_fabric"},
    }
    _print_comm_config(
        cluster=scenario["cluster"],
        parallelism=scenario["parallelism"],
        placement_order=scenario["collective"]["placement_order"],
        collective_kind=scenario["collective"]["kind"],
        domain_dims=scenario["collective"]["domain_dims"],
        tensor_bytes=scenario["collective"]["tensor_bytes"],
    )
    result = predict_collective_time(scenario, repo_root=REPO_ROOT)

    print(f"  predicted_time_ms : {result['predicted_time_ms']:.4f} ms")
    print(f"  network_ms        : {result['breakdown']['network_ms']:.4f} ms")
    print(f"  intra_server_ms   : {result['breakdown']['intra_server_ms']:.4f} ms")
    print(f"  combine_rule      : {result['breakdown']['combine_rule']}")
    print()


# ---------------------------------------------------------------------------
# Example 2: fixed cluster/topology, dynamically iterate over collectives
#            (the recommended pattern when the caller drives the workload)
# ---------------------------------------------------------------------------
def example2_dynamic_collective():
    print("=" * 60)
    print("Example 2: fixed cluster + dynamic collective")
    print("=" * 60)

    # 8 servers × 8 GPUs = 64 ranks.  dp=8 spans all 8 servers.
    # placement_order: last dim (DP) is outermost/slowest → DP crosses server boundaries.
    #   GPU layout:  server s  →  GPUs [8s .. 8s+7]  (ep=0..7, dp=s)
    #   domain=["DP"]: n=8 cross-server ranks (one GPU per server) — NIC traffic ✓
    BASE = {
        "cluster": {"servers": 8, "gpus_per_server": 8},
        "parallelism": {"tp": 1, "cp": 1, "dp": 8, "ep": 8},
        "topology": {"kind": "rail", "spines": 8, "servers_per_tor": 8, "paths": 8},
        # linkspeed_mbps: per-link bandwidth.  400_000 = 400 Gbps.
        "network": {
            "linkspeed_mbps": 400_000,
            "mtu": 9216,
            "q": 256,
            "cwnd": 64,
        },
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 450.0,   # one-way, GB/s (H100 NVLink 4.0)
            "nvlink_latency_us": 0.5,
            "nvlink_efficiency": 0.8,
        },
    }

    # All events communicate across the DP dimension → all involve cross-server NIC traffic.
    PO = ["TP", "CP", "EP", "DP"]   # DP last = slowest = spans servers
    events = [
        # allreduce ring: n=8 cross-server ranks, ring of 8 servers
        {"kind": "allreduce",     "tensor_bytes": 128 << 20, "domain_dims": ["DP"], "placement_order": PO},
        # allgather ring: each server contributes x bytes, gathers to x*8 across all 8 servers
        {"kind": "allgather",     "tensor_bytes": 128 << 20, "domain_dims": ["DP"], "placement_order": PO},
        # reducescatter ring: 8*x total reduced, each server keeps x bytes
        {"kind": "reducescatter", "tensor_bytes": 128 << 20, "domain_dims": ["DP"], "placement_order": PO},
        # alltoall: each of 8 cross-server ranks sends x/8 bytes to each other rank
        {"kind": "alltoall",      "tensor_bytes": 128 << 20, "domain_dims": ["DP"], "placement_order": PO},
    ]

    print(f"  {'collective':<16} {'msg':>6}  {'network_ms':>10}  {'intra_ms':>9}  {'total_ms':>9}  rule")
    print("  " + "-" * 60)
    for ev in events:
        _print_comm_config(
            cluster=BASE["cluster"],
            parallelism=BASE["parallelism"],
            placement_order=ev["placement_order"],
            collective_kind=ev["kind"],
            domain_dims=ev["domain_dims"],
            tensor_bytes=ev["tensor_bytes"],
            prefix="  ",
        )
        scenario = deepcopy(BASE)
        scenario["collective"] = {
            **ev,
            "exclude_intra_server": True,
        }
        r = predict_collective_time(scenario, repo_root=REPO_ROOT)
        bd = r["breakdown"]
        mib = ev["tensor_bytes"] >> 20
        print(
            f"  {ev['kind']:<16} {mib:>5}M"
            f"  {bd['network_ms']:>10.4f}"
            f"  {bd['intra_server_ms']:>9.4f}"
            f"  {bd['combined_ms']:>9.4f}"
            f"  {bd['combine_rule']}"
        )
        print("  " + "-" * 60)
    print()


# ---------------------------------------------------------------------------
# Example 3: load a built-in scenario profile
# ---------------------------------------------------------------------------
def example3_scenario_profile():
    print("=" * 60)
    print("Example 3: built-in scenario profile")
    print("=" * 60)

    from collective_sim_core import (
        get_scenario_profile_builder,
        list_scenario_profiles,
    )
    from collective_sim_core.schema import CollectiveConfig, ParallelismConfig
    from dataclasses import asdict

    print(f"  available profiles: {list_scenario_profiles()}")

    # 8 servers × 8 GPUs.  dp=8 spans all 8 servers; placement_order DP-last.
    # All collectives communicate across DP → all involve cross-server NIC traffic.
    builder = get_scenario_profile_builder("h800_rail")
    base_scenario = builder(
        servers=4,
        parallelism=ParallelismConfig(tp=1, cp=1, dp=32),
        collective=CollectiveConfig(
            kind="allreduce",
            tensor_bytes=1,          # placeholder; overridden per case below
            domain_dims=("DP",),
            # DP last → DP is outermost/slowest → DP spans server boundaries.
            # GPU layout: server s → GPUs [8s .. 8s+7] with ep=0..7, dp=s.
            placement_order=("DP",),
        ),
    )

    BASE = asdict(base_scenario)
    BASE["network"]["linkspeed_mbps"] = 400_000   # 400 Gbps; change for 200G / 800G

    # Each entry: (kind, tensor_bytes, model_overrides)
    # All use domain_dims=["DP"] → n=8 cross-server ranks (one per server, NIC traffic).
    # Hierarchical allgather/reducescatter require intra_server_combine_rule="sum":
    # the NIC inter-server phase and the NVLink intra-server phase are sequential.
    CASES = [
        # ring allreduce — 8-server cross-server ring
        ("allreduce",     100 << 20, {}),
        # ring allgather — each server contributes x bytes, all gather x*8 via ring
        ("allgather",     100 << 20, {"allgather_model": "ring_steps"}),
        # hierarchical allgather — inter-server NIC scatter + intra-server NVLink allgather
        ("allgather",     100 << 20, {
            "allgather_model": "hierarchical",
            "intra_server_combine_rule": "sum",
        }),
        # ring reducescatter — each server keeps x bytes after ring reduce
        ("reducescatter", 100 << 20, {"reducescatter_model": "ring_steps"}),
        # hierarchical reducescatter — intra-server NVLink reduce + inter-server NIC scatter
        ("reducescatter", 100 << 20, {
            "reducescatter_model": "hierarchical",
            "intra_server_combine_rule": "sum",
        }),
        # alltoall — each of 8 cross-server ranks exchanges chunks with all others
        ("alltoall",      100 << 20, {}),
    ]

    print(f"\n  {'collective':<28}  {'msg':>4}  {'network_ms':>10}  {'intra_ms':>8}  {'total_ms':>8}  rule")
    print(f"  {'-'*28}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  ----")
    for kind, tensor_bytes, overrides in CASES:
        _print_comm_config(
            cluster=BASE["cluster"],
            parallelism=BASE["parallelism"],
            placement_order=["TP", "CP", "EP", "DP"],
            collective_kind=kind,
            domain_dims=["DP"],
            tensor_bytes=tensor_bytes,
            prefix="  ",
        )
        scenario = deepcopy(BASE)
        scenario["collective"] = {
            "kind": kind,
            "tensor_bytes": tensor_bytes,
            "domain_dims": ["DP"],                 # all cross-server (n=8 servers, NIC)
            "placement_order": ["TP", "CP", "EP", "DP"],
            "exclude_intra_server": True,
            **overrides,
        }
        r = predict_collective_time(scenario, repo_root=REPO_ROOT)
        bd = r["breakdown"]
        mib = tensor_bytes >> 20
        model = overrides.get("allgather_model") or overrides.get("reducescatter_model") or "ring"
        label = f"{kind}/{model}" if model != "ring" else kind
        print(
            f"  {label:<28}  {mib:>3}M"
            f"  {bd['network_ms']:>10.4f}"
            f"  {bd['intra_server_ms']:>8.4f}"
            f"  {bd['combined_ms']:>8.4f}"
            f"  {bd['combine_rule']}"
        )
        print(f"  {'-'*28}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  ----")
    print()


if __name__ == "__main__":
    example1_onshot()
    example2_dynamic_collective()
    example3_scenario_profile()
