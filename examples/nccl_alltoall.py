#!/usr/bin/env python3
"""
nccl_alltoall.py — NCCL pairwise AllToAll example for collective-sim.

Simulates MoE (Mixture-of-Experts) EP AllToAll traffic on a rail-optimized
topology.  Compares the NCCL pairwise model (chunked, concurrent sends) with
the full-mesh model (single-step all-pairs) for different message sizes.

Scenario:
  - 4 servers × 8 GPUs = 32 ranks
  - TP=4, EP=8, DP=1, CP=1
  - placement_order: [TP, EP] — TP innermost, EP outermost
  - GPU layout: each server holds 2 EP values × 4 TP ranks = 8 GPUs
  - domain_dims=["EP"]: each EP group has 8 ranks (2 per server, 4 servers)
  - AllToAll on EP dimension: each rank exchanges T/8 bytes with every other rank
  - Mix of intra-server (NVLink, g=2) and inter-server (NIC) traffic

Run from the repo root:
    python examples/nccl_alltoall.py
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from collective_sim_core import predict_collective_time


def main() -> None:
    # 4 servers × 8 GPUs = 32 ranks.
    # TP=4, EP=8.  placement_order: [TP, EP] — TP innermost, EP outermost.
    # Rank = tp + 4*ep.  Each server holds 8 consecutive ranks:
    #   server 0: ep=0,1 (ranks 0-7), server 1: ep=2,3 (ranks 8-15), ...
    # domain_dims=["EP"]: each EP group has n=8 ranks, g=2 per server → 4 servers.
    # AllToAll on EP: each rank exchanges T/8 with 7 peers (1 intra + 6 inter).
    BASE = {
        "cluster": {"servers": 4, "gpus_per_server": 8},
        "parallelism": {"tp": 4, "cp": 1, "dp": 1, "ep": 8},
        "topology": {"kind": "rail", "spines": 4, "servers_per_tor": 4, "paths": 4},
        "network": {
            "linkspeed_mbps": 200_000,   # 200 Gbps per NIC
            "mtu": 9216,
            "q": 256,
            "cwnd": 32,
        },
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 200.0,   # H800 NVLink
            "nvlink_latency_us": 0.6,
            "nvlink_efficiency": 0.8,
        },
    }

    # Compare NCCL pairwise vs full-mesh AllToAll at different message sizes.
    # nccl_pairwise: chunked transfers with configurable concurrency (closer to real NCCL).
    # full_mesh:     idealized single-step all-pairs exchange.
    MODELS = [
        ("nccl_pairwise", {}),
        ("full_mesh",     {}),
    ]

    TENSOR_SIZES = [
        (  64 << 20, " 64M"),
        ( 256 << 20, "256M"),
        (1024 << 20, "  1G"),
    ]

    print("=" * 72)
    print("NCCL AllToAll — EP=8, TP=4, 4 servers, 200Gbps rail")
    print("=" * 72)
    print(f"  AllToAll on EP dimension: n=8 ranks per group (g=2 per server)")
    print(f"  Each rank sends T/8 bytes to each of the other 7 ranks")
    print()

    header = f"  {'model':<16} {'msg':>5}  {'network_ms':>10}  {'intra_ms':>9}  {'total_ms':>9}  rule"
    print(header)
    print("  " + "-" * 65)

    for model_name, overrides in MODELS:
        for tensor_bytes, size_label in TENSOR_SIZES:
            scenario = deepcopy(BASE)
            scenario["collective"] = {
                "kind": "alltoall",
                "alltoall_model": model_name,
                "tensor_bytes": tensor_bytes,
                "domain_dims": ["EP"],
                "placement_order": ["TP", "EP"],
                "exclude_intra_server": True,
                **overrides,
            }

            result = predict_collective_time(scenario, repo_root=REPO_ROOT)
            bd = result["breakdown"]
            print(
                f"  {model_name:<16} {size_label:>5}"
                f"  {bd['network_ms']:>10.4f}"
                f"  {bd['intra_server_ms']:>9.4f}"
                f"  {bd['combined_ms']:>9.4f}"
                f"  {bd['combine_rule']}"
            )
        print("  " + "-" * 65)

    print()
    print("Notes:")
    print("  - nccl_pairwise: chunked pairwise sends (8MB chunks, 2 in-flight/peer)")
    print("  - full_mesh:     single-step all-pairs (idealized upper bound)")
    print("  - intra_server_ms from NVLink analytic model (nvlink_analytic)")


if __name__ == "__main__":
    main()
