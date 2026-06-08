from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def run_htsim_runner(
    *,
    repo_root: Path,
    spec: Dict[str, Any],
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    runner_path = repo_root / "htsim_runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"missing runner: {runner_path}")

    # Compatibility guard:
    # if htsim binary is missing in the copied workspace, request build automatically.
    bin_path = repo_root / "sim" / "datacenter" / "htsim_ndp"
    spec_to_run = dict(spec)
    if not bin_path.exists():
        runner_cfg = dict(spec_to_run.get("runner", {}))
        runner_cfg["build"] = True
        spec_to_run["runner"] = runner_cfg

    with tempfile.TemporaryDirectory(prefix="collective_sim_core_spec_") as tmp:
        spec_path = Path(tmp) / "scenario.json"
        spec_path.write_text(json.dumps(spec_to_run, ensure_ascii=False, indent=2))
        cmd = [sys.executable, str(runner_path), "--spec", str(spec_path)]
        if out_dir:
            cmd += ["--out-dir", out_dir]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "htsim_runner failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"returncode: {proc.returncode}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )

        payload = _extract_json_from_stdout(proc.stdout)
        payload["_stdout"] = proc.stdout
        payload["_stderr"] = proc.stderr
        return payload


def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("no JSON payload found in htsim_runner stdout")

