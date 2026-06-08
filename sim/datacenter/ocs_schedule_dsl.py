#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocs_schedule_dsl.py

target: make OCS schedule more readable (introduce DSL), and compile it into the existing .schedule format:
  slot 0
  plane 0 a-b c-d ...

note:
- this is an "offline tool", not called by the runner automatically.
- designed to be minimal: support two types of port DSL used in this document:
    tor(<tor>).up(<up>)
    trunk(<i>)
  and core-ocs:
    pod(<p>).trunk(<i>)
- you can extend more port types as needed.

usage example:
  python ocs_schedule_dsl.py --in agg_ocs_tor_trunk...dsl --out agg_ocs_tor_trunk...schedule
  python ocs_schedule_dsl.py --in core_ocs_ring...dsl --out core_ocs_ring...schedule
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


LINE_COMMENT_RE = re.compile(r"#.*$")


@dataclass(frozen=True)
class Env:
    # For agg-ocs (B1)
    tors: int = 0
    tor_up: int = 0
    trunk_ports: int = 0
    # For core-ocs (B1)
    pods: int = 0
    trunk_ports_per_pod: int = 0


def _strip_comment(line: str) -> str:
    return LINE_COMMENT_RE.sub("", line).strip()


def _parse_int(k: str, v: str) -> int:
    try:
        return int(v.strip())
    except Exception as e:
        raise ValueError(f"bad int for {k}: {v!r}") from e


TOR_UP_RE = re.compile(r"^tor\((\d+)\)\.up\((\d+)\)$")
TRUNK_RE = re.compile(r"^trunk\((\d+)\)$")
POD_TRUNK_RE = re.compile(r"^pod\((\d+)\)\.trunk\((\d+)\)$")


def _port_id(token: str, env: Env, kind: str) -> int:
    """
    kind:
      - "agg"  : Agg-OCS B1 (tor.up + trunk)
      - "core" : Core-OCS B1 (pod.trunk only)
    """
    if kind == "agg":
        m = TOR_UP_RE.match(token)
        if m:
            tor = int(m.group(1))
            up = int(m.group(2))
            if tor < 0 or tor >= env.tors:
                raise ValueError(f"tor index out of range: {token}")
            if up < 0 or up >= env.tor_up:
                raise ValueError(f"up index out of range: {token}")
            return tor * env.tor_up + up
        m = TRUNK_RE.match(token)
        if m:
            i = int(m.group(1))
            if i < 0 or i >= env.trunk_ports:
                raise ValueError(f"trunk index out of range: {token}")
            return env.tors * env.tor_up + i
        raise ValueError(f"unknown port token (agg-ocs): {token!r}")

    if kind == "core":
        m = POD_TRUNK_RE.match(token)
        if m:
            pod = int(m.group(1))
            i = int(m.group(2))
            if pod < 0 or pod >= env.pods:
                raise ValueError(f"pod index out of range: {token}")
            if i < 0 or i >= env.trunk_ports_per_pod:
                raise ValueError(f"trunk index out of range: {token}")
            return pod * env.trunk_ports_per_pod + i
        raise ValueError(f"unknown port token (core-ocs): {token!r}")

    raise ValueError(f"unknown kind: {kind!r}")


def _parse_dsl(text: str) -> Tuple[str, Env, Dict[int, List[Tuple[str, str]]]]:
    """
    Returns: (kind, env, planes)
      kind: "agg" or "core"
      planes: plane_id -> list[(lhs_token, rhs_token)]
    """
    kind = ""
    env = Env()
    planes: Dict[int, List[Tuple[str, str]]] = {}

    cur_plane: int | None = None
    in_header = True

    # Simple key=value header + plane blocks:
    #
    # kind=agg
    # tors=8
    # tor_up=32
    # trunk_ports=256
    #
    # plane 0:
    #   tor(0).up(0) <-> trunk(0)
    #   ...
    #
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line:
            continue

        if in_header and re.match(r"^plane\s+\d+\s*:\s*$", line):
            in_header = False

        if in_header:
            if "=" not in line:
                raise ValueError(f"bad header line (expect key=value): {raw!r}")
            k, v = [x.strip() for x in line.split("=", 1)]
            if k == "kind":
                kind = v
            elif k == "tors":
                env = Env(tors=_parse_int(k, v), tor_up=env.tor_up, trunk_ports=env.trunk_ports, pods=env.pods, trunk_ports_per_pod=env.trunk_ports_per_pod)
            elif k == "tor_up":
                env = Env(tors=env.tors, tor_up=_parse_int(k, v), trunk_ports=env.trunk_ports, pods=env.pods, trunk_ports_per_pod=env.trunk_ports_per_pod)
            elif k == "trunk_ports":
                env = Env(tors=env.tors, tor_up=env.tor_up, trunk_ports=_parse_int(k, v), pods=env.pods, trunk_ports_per_pod=env.trunk_ports_per_pod)
            elif k == "pods":
                env = Env(tors=env.tors, tor_up=env.tor_up, trunk_ports=env.trunk_ports, pods=_parse_int(k, v), trunk_ports_per_pod=env.trunk_ports_per_pod)
            elif k == "trunk_ports_per_pod":
                env = Env(tors=env.tors, tor_up=env.tor_up, trunk_ports=env.trunk_ports, pods=env.pods, trunk_ports_per_pod=_parse_int(k, v))
            else:
                raise ValueError(f"unknown header key: {k!r}")
            continue

        m = re.match(r"^plane\s+(\d+)\s*:\s*$", line)
        if m:
            cur_plane = int(m.group(1))
            planes.setdefault(cur_plane, [])
            continue

        if cur_plane is None:
            raise ValueError(f"connection line outside plane block: {raw!r}")

        # conn format: "<lhs> <-> <rhs>"
        if "<->" not in line:
            raise ValueError(f"bad connection line (expect '<->'): {raw!r}")
        lhs, rhs = [x.strip() for x in line.split("<->", 1)]
        if not lhs or not rhs:
            raise ValueError(f"bad connection line: {raw!r}")
        planes[cur_plane].append((lhs, rhs))

    if kind not in ("agg", "core"):
        raise ValueError("missing/invalid kind= (must be 'agg' or 'core')")
    if kind == "agg":
        if env.tors <= 0 or env.tor_up <= 0 or env.trunk_ports <= 0:
            raise ValueError("agg-ocs requires tors/tor_up/trunk_ports > 0 in header")
        if env.tors * env.tor_up != env.trunk_ports:
            raise ValueError(f"agg-ocs expects trunk_ports == tors*tor_up, got {env.trunk_ports} vs {env.tors}*{env.tor_up}")
    if kind == "core":
        if env.pods <= 0 or env.trunk_ports_per_pod <= 0:
            raise ValueError("core-ocs requires pods/trunk_ports_per_pod > 0 in header")

    return kind, env, planes


def _emit_schedule(kind: str, env: Env, planes: Dict[int, List[Tuple[str, str]]]) -> str:
    lines: List[str] = ["slot 0", ""]
    for p in sorted(planes.keys()):
        used: set[int] = set()
        pairs_out: List[str] = []
        for lhs_t, rhs_t in planes[p]:
            a = _port_id(lhs_t, env, kind)
            b = _port_id(rhs_t, env, kind)
            if a == b:
                raise ValueError(f"self-loop in plane {p}: {lhs_t} <-> {rhs_t}")
            if a in used or b in used:
                raise ValueError(f"endpoint reused in plane {p}: {lhs_t} <-> {rhs_t}")
            used.add(a)
            used.add(b)
            pairs_out.append(f"{a}-{b}")
        lines.append(" ".join(["plane", str(p), *pairs_out]))
    lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .dsl file")
    ap.add_argument("--out", dest="out", required=True, help="Output .schedule file")
    args = ap.parse_args(argv)

    inp = Path(args.inp)
    out = Path(args.out)
    kind, env, planes = _parse_dsl(inp.read_text(errors="ignore"))
    out.write_text(_emit_schedule(kind, env, planes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))


