#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class Scale:
    name: str
    servers: int
    gpus_per_server: int
    tp: int
    cp: int
    dp: int
    ep: int


@dataclass(frozen=True)
class TopologyCase:
    name: str
    cfg: Dict[str, Any]


@dataclass(frozen=True)
class CollectiveCase:
    name: str
    cfg: Dict[str, Any]
    # Optional per-scale tensor_bytes override: {scale_name: bytes}.
    # Use to reduce simulation cost for large-scale + expensive collective combos
    # without changing the code path under test.
    tensor_bytes_by_scale: Dict[str, int] = None  # type: ignore[assignment]

    def tensor_bytes_for(self, scale: "Scale") -> int:
        if self.tensor_bytes_by_scale and scale.name in self.tensor_bytes_by_scale:
            return self.tensor_bytes_by_scale[scale.name]
        return int(self.cfg["tensor_bytes"])


def _log_progress(message: str) -> None:
    # Print progress to stderr so stdout remains machine-readable JSON.
    print(message, file=sys.stderr, flush=True)


def _write_json_temp(payload: Dict[str, Any]) -> Path:
    f = tempfile.NamedTemporaryFile(prefix="csc_matrix_spec_", suffix=".json", delete=False)
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


def run_old_runner(repo_root: Path, spec: Dict[str, Any], out_dir: Path, timeout_s: int = 0) -> Dict[str, Any]:
    runner = repo_root / "htsim_runner.py"
    cmd = [sys.executable, str(runner), "--spec", str(_write_json_temp(spec)), "--out-dir", str(out_dir)]
    run_kwargs = {"capture_output": True, "text": True}
    if timeout_s and timeout_s > 0:
        run_kwargs["timeout"] = int(timeout_s)
    try:
        proc = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"[old runner] timeout after {timeout_s}s for {out_dir.name}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"partial stderr:\n{(e.stderr or '')[-4000:]}"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"[old runner] failed for {out_dir.name}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return _extract_json(proc.stdout)


def build_matrix() -> List[Dict[str, Any]]:
    scales = [
        Scale(name="s8_g8", servers=8, gpus_per_server=8, tp=8, cp=1, dp=8, ep=1),
        Scale(name="s16_g8", servers=16, gpus_per_server=8, tp=8, cp=1, dp=16, ep=1),
    ]

    topologies = [
        TopologyCase(
            name="rail",
            cfg={
                "type": "rail",
                "latency_ns": 800,
                "switch_latency_ns": 800,
            },
        ),
        TopologyCase(
            name="fattree",
            cfg={
                "type": "fattree",
                "tor_down": 16,
                "fattree_oversub": 1,
                # paths intentionally omitted: auto-computed as tor_down // fattree_oversub = 16,
                # matching htsim_runner.py's own default for topology=fattree.
                "latency_ns": 800,
                "switch_latency_ns": 800,
            },
        ),
    ]

    collectives = [
        CollectiveCase(
            name="allreduce_ring_steps",
            cfg={
                "type": "allreduce",
                "tensor_bytes": 256 * 1024 * 1024,
                "domain_dims": ["DP"],
                "placement_order": ["TP", "CP", "EP", "DP"],
                "exclude_intra_server": False,
                "use_triggers": True,
                "allreduce_model": "ring_steps",
            },
        ),
        CollectiveCase(
            name="allgather_hierarchical",
            cfg={
                "type": "allgather",
                "tensor_bytes": 256 * 1024 * 1024,
                "domain_dims": ["DP"],
                "placement_order": ["TP", "CP", "EP", "DP"],
                # hierarchical treats intra-server as free on the NIC side;
                # matches real experiment configs (rl_net_exp).
                "exclude_intra_server": True,
                "use_triggers": False,
                "allgather_model": "hierarchical",
            },
            # s16_g8 generates 15360 flows at 256MiB which takes >600s to simulate.
            # Use 32MiB for larger scales to keep regression feasible.
            tensor_bytes_by_scale={"s16_g8": 32 * 1024 * 1024},
        ),
        CollectiveCase(
            name="reducescatter_hierarchical",
            cfg={
                "type": "reducescatter",
                "tensor_bytes": 256 * 1024 * 1024,
                "domain_dims": ["DP"],
                "placement_order": ["TP", "CP", "EP", "DP"],
                "exclude_intra_server": True,
                "use_triggers": False,
                "reducescatter_model": "hierarchical",
            },
            tensor_bytes_by_scale={"s16_g8": 32 * 1024 * 1024},
        ),
        CollectiveCase(
            name="alltoall_pairwise_steps",
            cfg={
                "type": "alltoall",
                "tensor_bytes": 64 * 1024 * 1024,
                "domain_dims": ["DP"],
                "placement_order": ["TP", "CP", "EP", "DP"],
                "exclude_intra_server": False,
                "use_triggers": False,
                "alltoall_model": "pairwise_steps",
                "alltoall_channels": 4,
                "alltoall_chunk_bytes": 8 * 1024 * 1024,
                "alltoall_chunk_inflight_per_peer": 2,
            },
        ),
    ]

    matrix: List[Dict[str, Any]] = []
    for s in scales:
        for t in topologies:
            for c in collectives:
                matrix.append(
                    {
                        "name": f"{s.name}__{t.name}__{c.name}",
                        "scale": s,
                        "topology": t,
                        "collective": c,
                    }
                )
    return matrix


def main() -> int:
    ap = argparse.ArgumentParser(description="Large-scale rail/fattree collective regression matrix.")
    ap.add_argument("--max-cases", type=int, default=0, help="Run first N cases only (0 means run all).")
    ap.add_argument(
        "--case-contains",
        type=str,
        default="",
        help="Only run cases whose case name contains this substring.",
    )
    ap.add_argument(
        "--quick-sample",
        action="store_true",
        help="Run a faster sample: smallest scale only, still covers rail/fattree x all collectives.",
    )
    ap.add_argument("--end-us", type=int, default=2_000_000, help="Simulation end time in us.")
    ap.add_argument("--tolerance-us", type=float, default=1e-3, help="Allowed diff between old/new legacy (us).")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-case progress logging.")
    ap.add_argument(
        "--old-runner-timeout-s",
        type=int,
        default=180,
        help="Timeout for each old-runner invocation in seconds (0 means no timeout).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))
    from collective_sim_core import predict_collective_time

    # Aligned with real experiment configs in rl_net_exp/:
    # mtu=9216 (standard jumbo, not over-sized 65000),
    # q=256, cwnd=64 match typical rl_net_exp rail/fattree settings.
    base_network = {"linkspeed_mbps": 400000, "mtu": 9216, "q": 256, "cwnd": 64}
    base_runner = {"build": False, "progress": False, "print_args": False, "stop_on_finished": True, "heartbeat_s": 5, "seed": 1}

    matrix = build_matrix()
    if args.quick_sample:
        # Keep one representative scale while preserving coverage over topology x collective.
        min_servers = min(m["scale"].servers for m in matrix)
        min_gpus = min(m["scale"].gpus_per_server for m in matrix)
        matrix = [
            m
            for m in matrix
            if m["scale"].servers == min_servers and m["scale"].gpus_per_server == min_gpus
        ]
    if args.max_cases and args.max_cases > 0:
        matrix = matrix[: args.max_cases]
    if args.case_contains:
        matrix = [m for m in matrix if args.case_contains in m["name"]]

    all_ok = True
    results: List[Dict[str, Any]] = []
    total = len(matrix)

    if not args.no_progress:
        _log_progress(
            f"[matrix] start total_cases={total} "
            f"(quick_sample={bool(args.quick_sample)}, end_us={int(args.end_us)}, tolerance_us={args.tolerance_us})"
        )

    for i, case in enumerate(matrix):
        s: Scale = case["scale"]
        t: TopologyCase = case["topology"]
        c: CollectiveCase = case["collective"]
        topology_cfg = dict(t.cfg)
        if t.name == "rail":
            # Build rail params from current scale by default.
            topology_cfg["servers_per_tor"] = int(topology_cfg.get("servers_per_tor", s.servers))
            topology_cfg["spines"] = int(topology_cfg.get("spines", s.gpus_per_server))
            topology_cfg["paths"] = int(topology_cfg.get("paths", topology_cfg["spines"]))
        else:
            # fattree: paths = tor_down // fattree_oversub, matching htsim_runner.py's default.
            tor_down = int(topology_cfg.get("tor_down", 16))
            oversub = int(topology_cfg.get("fattree_oversub", 1))
            topology_cfg["paths"] = int(topology_cfg.get("paths", tor_down // max(oversub, 1)))
        tensor_bytes = c.tensor_bytes_for(s)
        if not args.no_progress:
            note = "" if tensor_bytes == c.cfg["tensor_bytes"] else f" [tensor_bytes overridden to {tensor_bytes // (1024*1024)}MiB for this scale]"
            _log_progress(f"[matrix] [{i + 1}/{total}] running {case['name']}{note}")

        collective_cfg = dict(c.cfg)
        collective_cfg["tensor_bytes"] = tensor_bytes

        spec = {
            "topology": dict(topology_cfg),
            "parallel": {"tp": s.tp, "cp": s.cp, "dp": s.dp, "ep": s.ep},
            "job": {"nodes": s.servers * s.gpus_per_server, "servers": s.servers, "gpus_per_server": s.gpus_per_server},
            "collective": collective_cfg,
            "network": dict(base_network),
            "sim": {"end_us": int(args.end_us)},
            "runner": dict(base_runner),
        }

        out_base = Path(tempfile.mkdtemp(prefix=f"csc_matrix_{i}_{case['name']}_"))
        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] old_runner start")
        old_payload = run_old_runner(
            repo_root,
            spec,
            out_base / "out_old",
            timeout_s=int(args.old_runner_timeout_s),
        )
        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] old_runner done")

        scenario = {
            "cluster": {"servers": s.servers, "gpus_per_server": s.gpus_per_server},
            "parallelism": {"tp": s.tp, "cp": s.cp, "dp": s.dp, "ep": s.ep},
            "topology": {
                "kind": topology_cfg["type"],
                "spines": topology_cfg.get("spines", s.gpus_per_server),
                "servers_per_tor": topology_cfg.get("servers_per_tor", s.servers),
                "paths": topology_cfg.get("paths", s.gpus_per_server),
                "tor_down": topology_cfg.get("tor_down", 16),
                "fattree_oversub": topology_cfg.get("fattree_oversub", 1),
                "latency_ns": topology_cfg.get("latency_ns", 800),
                "switch_latency_ns": topology_cfg.get("switch_latency_ns", 800),
            },
            "collective": {
                "kind": c.cfg["type"],
                "tensor_bytes": tensor_bytes,
                "domain_dims": c.cfg["domain_dims"],
                "placement_order": c.cfg["placement_order"],
                "exclude_intra_server": c.cfg.get("exclude_intra_server", False),
                "use_triggers": c.cfg.get("use_triggers", False),
                "allreduce_model": c.cfg.get("allreduce_model", "ring_steps"),
                "allgather_model": c.cfg.get("allgather_model", "ring_steps"),
                "reducescatter_model": c.cfg.get("reducescatter_model", "ring_steps"),
                "alltoall_model": c.cfg.get("alltoall_model", "full_mesh"),
                "alltoall_channels": c.cfg.get("alltoall_channels", 1),
                "alltoall_chunk_bytes": c.cfg.get("alltoall_chunk_bytes", 0),
                "alltoall_chunk_inflight_per_peer": c.cfg.get("alltoall_chunk_inflight_per_peer", 0),
            },
            "network": dict(base_network),
            "runner": {
                "end_us": int(args.end_us),
                "seed": base_runner["seed"],
                "build": base_runner["build"],
                "progress": base_runner["progress"],
                "print_args": base_runner["print_args"],
                "stop_on_finished": base_runner["stop_on_finished"],
                "heartbeat_s": base_runner["heartbeat_s"],
                "out_dir": str(out_base / "out_new_legacy"),
            },
            "intra_server": {"model": "legacy_fabric"},
        }

        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] new_legacy start")
        new_legacy = predict_collective_time(scenario, repo_root=repo_root)
        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] new_legacy done")

        scenario["runner"]["out_dir"] = str(out_base / "out_new_nv")
        scenario["collective"] = dict(scenario["collective"])
        scenario["collective"]["exclude_intra_server"] = True
        scenario["intra_server"] = {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 450.0,
            "nvlink_latency_us": 0.5,
            "nvlink_efficiency": 0.8,
        }
        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] new_nvlink start")
        new_nv = predict_collective_time(scenario, repo_root=repo_root)
        if not args.no_progress:
            _log_progress(f"[matrix] [{i + 1}/{total}] new_nvlink done")

        old_us = float(old_payload.get("makespan_us", 0.0))
        new_legacy_us = float(new_legacy["breakdown"]["network_ms"]) * 1000.0
        diff_us = new_legacy_us - old_us
        ok = abs(diff_us) <= args.tolerance_us
        if not ok:
            all_ok = False

        results.append(
            {
                "case": case["name"],
                "old_makespan_us": old_us,
                "new_legacy_network_us": new_legacy_us,
                "diff_us": diff_us,
                "ok": ok,
                "nvlink_intra_ms": new_nv["breakdown"]["intra_server_ms"],
                "predicted_time_ms_with_nvlink_model": new_nv["breakdown"]["combined_ms"],
            }
        )
        if not args.no_progress:
            _log_progress(
                f"[matrix] [{i + 1}/{total}] done {case['name']} "
                f"(ok={ok}, diff_us={diff_us:.6f}, nvlink_intra_ms={float(new_nv['breakdown']['intra_server_ms']):.6f})"
            )

    if not args.no_progress:
        ok_count = sum(1 for r in results if r["ok"])
        _log_progress(f"[matrix] finished ok={ok_count}/{total}")
    print(json.dumps({"total_cases": len(matrix), "tolerance_us": args.tolerance_us, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

