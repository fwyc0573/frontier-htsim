#!/usr/bin/env python3
"""
Smoke regression: cross-validates predict_collective_time (Python SDK) against
the raw htsim_runner for a matrix of collective types and topologies, and checks
the nvlink_analytic intra-server model produces positive estimates.

Cluster: 2 servers × 8 GPUs = 16 nodes  (gpus_per_server=8 gives meaningful NVLink traffic)
Parallelism: tp=1 cp=1 dp=2 ep=4

Checks per case
  1. legacy_fabric : SDK network_ms == old htsim_runner makespan  (diff ≤ tolerance_us)
  2. nvlink_analytic: intra_server_ms > 0  (model produces a positive estimate)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Case descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopoSpec:
    name: str
    kind: str
    # rail
    servers_per_tor: int = 0
    spines: int = 0
    paths: int = 0
    # fattree
    tor_down: int = 0
    fattree_oversub: int = 1
    latency_ns: int = 800
    switch_latency_ns: int = 800


@dataclass(frozen=True)
class CollSpec:
    kind: str
    domain_dims: tuple
    model_tag: str                          # used in case name
    tensor_bytes: int = 1 << 20
    use_triggers: bool = False
    allreduce_model: str = "ring_steps"
    allgather_model: str = "ring_steps"
    reducescatter_model: str = "ring_steps"
    alltoall_model: str = "pairwise_steps"
    alltoall_channels: int = 4
    alltoall_chunk_bytes: int = 0
    alltoall_chunk_inflight_per_peer: int = 0
    placement_order: tuple = ("TP", "CP", "EP", "DP")


# ---------------------------------------------------------------------------
# Topology definitions  (2 servers × 8 GPUs → 16 nodes)
# ---------------------------------------------------------------------------

RAIL = TopoSpec(
    name="rail",
    kind="rail",
    servers_per_tor=2,
    spines=8,
    paths=8,
)

# fattree: tor_down=8, oversub=1 → paths = tor_down // oversub = 8
FATTREE = TopoSpec(
    name="fattree",
    kind="fattree",
    tor_down=8,
    fattree_oversub=1,
    paths=8,
)

TOPOS: List[TopoSpec] = [RAIL, FATTREE]

# ---------------------------------------------------------------------------
# Collective definitions
# placement_order=[TP,CP,EP,DP]: _iter_coords iterates dims in REVERSE index order,
# so the LAST entry (DP) is the outermost (slowest) dimension and the FIRST entry (TP)
# is the innermost (fastest).  With dp=2, ep=8, servers=2, gpus_per_server=8:
#   Server 0 → GPUs  0-7  (dp=0, ep=0..7)   ← DP spans servers
#   Server 1 → GPUs  8-15 (dp=1, ep=0..7)
#
# domain=["DP"]  → 2 cross-server ranks (one GPU per server)    — inter-server NIC traffic
# domain=["EP"]  → 8 intra-server ranks (all within same server) — intra-server fabric/NVLink
# ---------------------------------------------------------------------------

COLLECTIVES: List[CollSpec] = [
    # allreduce ring  (domain DP: cross-server ring of 2)
    CollSpec(
        kind="allreduce", domain_dims=("DP",), model_tag="ring",
        tensor_bytes=4 << 20, use_triggers=True, allreduce_model="ring_steps",
        placement_order=("TP", "CP", "EP", "DP"),
    ),
    # allgather flat ring  (domain EP: 8 intra-server ranks)
    CollSpec(
        kind="allgather", domain_dims=("EP",), model_tag="ring",
        tensor_bytes=1 << 20, allgather_model="ring_steps",
        placement_order=("TP", "CP", "EP", "DP"),
    ),
    # allgather hierarchical  (domain DP: cross-server, 1 GPU/server — degenerate intra phase)
    CollSpec(
        kind="allgather", domain_dims=("DP",), model_tag="hier",
        tensor_bytes=1 << 20, allgather_model="hierarchical",
        placement_order=("TP", "CP", "EP", "DP"),
    ),
    # reducescatter flat ring  (domain EP: 8 intra-server ranks)
    CollSpec(
        kind="reducescatter", domain_dims=("EP",), model_tag="ring",
        tensor_bytes=1 << 20, reducescatter_model="ring_steps",
        placement_order=("TP", "CP", "EP", "DP"),
    ),
    # reducescatter hierarchical  (domain DP: cross-server)
    CollSpec(
        kind="reducescatter", domain_dims=("DP",), model_tag="hier",
        tensor_bytes=1 << 20, reducescatter_model="hierarchical",
        placement_order=("TP", "CP", "EP", "DP"),
    ),
    # alltoall pairwise  (domain EP: 8 intra-server ranks)
    CollSpec(
        kind="alltoall", domain_dims=("EP",), model_tag="pairwise",
        tensor_bytes=1 << 20,
        alltoall_model="pairwise_steps", alltoall_channels=8,
        alltoall_chunk_bytes=0, alltoall_chunk_inflight_per_peer=0,
        placement_order=("TP", "CP", "EP", "DP"),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_old_runner(repo_root: Path, spec: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    runner = repo_root / "htsim_runner.py"
    spec_path = _write_json_temp(spec)
    cmd = [sys.executable, str(runner), "--spec", str(spec_path), "--out-dir", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[old runner] failed ({out_dir.name})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return _extract_json(proc.stdout)


def _write_json_temp(payload: Dict[str, Any]) -> Path:
    f = tempfile.NamedTemporaryFile(prefix="csc_smoke_", suffix=".json", delete=False)
    p = Path(f.name)
    p.write_text(json.dumps(payload, ensure_ascii=False))
    f.close()
    return p


def _extract_json(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    raise RuntimeError("no JSON output found")


def _topo_dict(t: TopoSpec) -> Dict[str, Any]:
    """Topology dict suitable for both old runner spec and new SDK."""
    if t.kind == "rail":
        return {
            "type": t.kind, "kind": t.kind,
            "servers_per_tor": t.servers_per_tor,
            "spines": t.spines,
            "paths": t.paths,
            "latency_ns": t.latency_ns,
            "switch_latency_ns": t.switch_latency_ns,
        }
    else:
        return {
            "type": t.kind, "kind": t.kind,
            "tor_down": t.tor_down,
            "fattree_oversub": t.fattree_oversub,
            "paths": t.paths,
            "latency_ns": t.latency_ns,
            "switch_latency_ns": t.switch_latency_ns,
        }


def _coll_dict(c: CollSpec, exclude_intra: bool) -> Dict[str, Any]:
    return {
        "type": c.kind, "kind": c.kind,
        "tensor_bytes": c.tensor_bytes,
        "domain_dims": list(c.domain_dims),
        "placement_order": list(c.placement_order),
        "exclude_intra_server": exclude_intra,
        "use_triggers": c.use_triggers,
        "allreduce_model": c.allreduce_model,
        "allgather_model": c.allgather_model,
        "reducescatter_model": c.reducescatter_model,
        "alltoall_model": c.alltoall_model,
        "alltoall_channels": c.alltoall_channels,
        "alltoall_chunk_bytes": c.alltoall_chunk_bytes,
        "alltoall_chunk_inflight_per_peer": c.alltoall_chunk_inflight_per_peer,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))
    from collective_sim_core import predict_collective_time

    # Cluster: 2 servers × 8 GPUs = 16 nodes
    SERVERS, GPUS = 2, 8
    base_parallel = {"tp": 1, "cp": 1, "dp": 2, "ep": 8}
    base_network = {"linkspeed_mbps": 400000, "mtu": 9216, "q": 256, "cwnd": 64}
    base_runner_flags = {
        "build": False, "progress": False, "print_args": False,
        "stop_on_finished": True, "heartbeat_s": 5, "seed": 1,
    }
    END_US = 500_000
    TOLERANCE_US = 1e-3

    nvlink_params = {
        "model": "nvlink_analytic",
        "nvlink_one_way_bw_GBps": 450.0,
        "nvlink_latency_us": 0.5,
        "nvlink_efficiency": 0.8,
    }

    all_ok = True
    results = []
    case_idx = 0

    for topo in TOPOS:
        topo_d = _topo_dict(topo)
        for coll in COLLECTIVES:
            case_name = f"{coll.kind}_{coll.model_tag}__{topo.name}"
            case_idx += 1
            tensor_mib = coll.tensor_bytes / (1024 * 1024)
            print(
                f"[{case_idx:02d}] running {case_name} "
                f"(tensor_bytes={coll.tensor_bytes}, {tensor_mib:.2f} MiB) ...",
                flush=True,
            )

            out_base = Path(tempfile.mkdtemp(prefix=f"csc_smoke_{case_idx}_"))

            # ---- 1. Old runner baseline (exclude_intra_server=True, network-only NIC path) ----
            old_spec = {
                "topology": {k: v for k, v in topo_d.items() if k != "kind"},
                "parallel": dict(base_parallel),
                "job": {"nodes": SERVERS * GPUS, "servers": SERVERS, "gpus_per_server": GPUS},
                "collective": _coll_dict(coll, exclude_intra=True),
                "network": dict(base_network),
                "sim": {"end_us": END_US},
                "runner": {**base_runner_flags, "paths": topo.paths},
            }
            old_payload = run_old_runner(repo_root, old_spec, out_base / "old")
            old_us = float(old_payload.get("makespan_us", 0.0))

            # ---- 2. New SDK — legacy_fabric (should match old runner under exclude_intra_server=True) ----
            scenario_legacy = {
                "cluster": {"servers": SERVERS, "gpus_per_server": GPUS},
                "parallelism": dict(base_parallel),
                "topology": {k: v for k, v in topo_d.items() if k != "type"},
                "collective": {k: v for k, v in _coll_dict(coll, exclude_intra=True).items() if k != "type"},
                "network": dict(base_network),
                "runner": {**base_runner_flags, "end_us": END_US,
                           "out_dir": str(out_base / "new_legacy")},
                "intra_server": {"model": "legacy_fabric"},
            }
            new_legacy = predict_collective_time(scenario_legacy, repo_root=repo_root)
            new_us = float(new_legacy["breakdown"]["network_ms"]) * 1000.0
            diff_us = new_us - old_us
            legacy_ok = abs(diff_us) <= TOLERANCE_US
            if not legacy_ok:
                all_ok = False

            # ---- 3. New SDK — nvlink_analytic (sanity: intra_ms > 0) ----
            scenario_nv = dict(scenario_legacy)
            scenario_nv = {**scenario_legacy}
            scenario_nv["collective"] = {
                **scenario_legacy["collective"], "exclude_intra_server": True,
            }
            scenario_nv["intra_server"] = nvlink_params
            scenario_nv["runner"] = {**base_runner_flags, "end_us": END_US,
                                     "out_dir": str(out_base / "new_nv")}
            new_nv = predict_collective_time(scenario_nv, repo_root=repo_root)
            intra_ms = float(new_nv["breakdown"]["intra_server_ms"])
            nvlink_ok = intra_ms > 0.0

            row = {
                "case": case_name,
                "tensor_bytes": coll.tensor_bytes,
                "tensor_mib": round(tensor_mib, 6),
                "legacy_ok": legacy_ok,
                "old_us": round(old_us, 3),
                "new_us": round(new_us, 3),
                "diff_us": round(diff_us, 6),
                "nvlink_ok": nvlink_ok,
                "intra_ms": round(intra_ms, 6),
                "combine_rule": new_nv["breakdown"]["combine_rule"],
                "combined_ms": round(new_nv["breakdown"]["combined_ms"], 4),
            }
            if coll.model_tag == "hier" and coll.kind in ("allgather", "reducescatter"):
                row["combine_rule_ok"] = (row["combine_rule"] == "sum")
            else:
                row["combine_rule_ok"] = True

            if not row["combine_rule_ok"]:
                all_ok = False
            results.append(row)

            status = ("OK" if (legacy_ok and nvlink_ok and row["combine_rule_ok"]) else "FAIL")
            print(
                f"  msg={coll.tensor_bytes}B ({tensor_mib:.2f} MiB)  "
                f"legacy diff={diff_us:+.3f} us ({'OK' if legacy_ok else 'FAIL'})  "
                f"nvlink intra_ms={intra_ms:.4f} ({'OK' if nvlink_ok else 'FAIL'})  "
                f"combine={row['combine_rule']} ({'OK' if row['combine_rule_ok'] else 'FAIL'})  "
                f"→ {status}",
                flush=True,
            )

    print()
    print(json.dumps({"tolerance_us": TOLERANCE_US, "results": results}, indent=2, ensure_ascii=False))

    passed = sum(1 for r in results if r["legacy_ok"] and r["nvlink_ok"])
    print(f"\n{'ALL PASSED' if all_ok else 'FAILURES DETECTED'}  "
          f"({passed}/{len(results)} cases passed)")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
