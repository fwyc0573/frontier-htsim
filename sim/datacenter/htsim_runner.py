#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for repo-root /data/csg-htsim/htsim_runner.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "htsim_runner.py"
    if not target.exists():
        raise SystemExit(f"Cannot find repo-root runner: {target}")
    sys.path.insert(0, str(repo_root))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()


