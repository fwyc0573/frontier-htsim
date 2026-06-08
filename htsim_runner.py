#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
htsim_runner.py (repo root)

be compatible with the old runner's interface（--spec/--out-dir/--progress/--heartbeat-s/--stop-on-finished），and on this basis:
- support 4-dimensional parallelism: TP/CP/DP/EP（domain_dims/placement_order can contain EP）
- after generating tm.cm, calculate onwire throughput based on the "actual tm total bytes" (fix the overestimation caused by exclude_intra_server)
- output makespan (for faster topology selection) as the main metric, and throughput as the secondary metric
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DATACENTER_DIR = REPO_ROOT / "sim" / "datacenter"
HTSIM_BIN = DATACENTER_DIR / "htsim_ndp"
MAKE_DIR = REPO_ROOT / "sim"

DIM_TP = "TP"
DIM_CP = "CP"
DIM_DP = "DP"
DIM_EP = "EP"
KNOWN_DIMS = (DIM_TP, DIM_CP, DIM_DP, DIM_EP)


@dataclass(frozen=True)
class CollectiveSpec:
    collective_type: str  # "alltoall" | "allreduce" | "allgather" | "reducescatter" | "p2p" | "t2g"
    domain_dims: Tuple[str, ...]
    placement_order: Tuple[str, ...]
    exclude_intra_server: bool
    # For collective_type == "allreduce":
    # If true, generate step-by-step ring allreduce using triggers (barriers)
    # instead of starting all steps at time 0.
    use_triggers: bool = False
    # For collective_type == "allreduce":
    # - "ring_steps": unrolled ring model (2*(n-1) steps, optionally barrier-triggered).
    # - "ring_stream": pipelined steady-state (one long stream per rank-pair, 2*(n-1)/n * T bytes).
    # - "nccl_ring": NCCL-accurate multi-channel ring. nchannels independent rings, each rotated
    #   by channel index (channel k anchors at GPU_k -> NIC_k in rail topology). Each channel has
    #   its own independent barrier chain (2*(n-1) steps). Per channel chunk = T/(n*nchannels).
    allreduce_model: str = "ring_steps"
    # For collective_type == "allgather":
    # - "ring_steps": classic ring allgather (n-1 steps, each step sends tensor_bytes).
    # - "hierarchical": NIC-focused: inter-server full-mesh striped across GPUs.
    # - "hierarchical_ring": inter-server ring with S-1 step-by-step barriers.
    # - "nccl_ring": NCCL-accurate multi-channel ring. nchannels independent rings, each rotated
    #   by channel index. Each channel has its own independent barrier chain (n-1 steps).
    #   Per channel chunk = tensor_bytes / nchannels.
    allgather_model: str = "ring_steps"
    # For collective_type == "reducescatter":
    # - "ring_steps": classic ring reduce-scatter (n-1 steps, tensor_bytes is per-rank OUTPUT).
    # - "hierarchical": NIC-focused inter-server full-mesh.
    # - "hierarchical_ring": inter-server ring with S-1 step-by-step barriers.
    # - "nccl_ring": NCCL-accurate multi-channel ring. nchannels independent rings, each rotated
    #   by channel index. Each channel has its own independent barrier chain (n-1 steps).
    #   Per channel chunk = tensor_bytes / nchannels.
    reducescatter_model: str = "ring_steps"
    # For collective_type == "alltoall":
    # - "full_mesh": classic all-to-all where each rank sends to every other rank at time 0.
    # - "pairwise_steps": schedule communication in phases; each phase each rank talks to at most K peers
    #   (K=alltoall_channels), with phases chained by barrier triggers to avoid unrealistic full fanout.
    # - "nccl_pairwise":  NCCL-accurate two-level schedule:
    #                     (S-1) server deltas (zigzag order) x G sub-steps (Latin Square GPU pairing).
    #                     Total (S-1)*G serial micro-rounds; each has G*S concurrent flows.
    alltoall_model: str = "pairwise_steps"
    # For alltoall_model == "pairwise_steps": max peers per phase per rank (>=1).
    alltoall_channels: int = 1
    # Optional for alltoall: override sub-chunk size in bytes (<= chunk_total). When > 0,
    # each (src,dst) transfer of chunk_total bytes is split into multiple flows of size
    # <= alltoall_chunk_bytes. This is useful to approximate NCCL's chunking/pipelining.
    alltoall_chunk_bytes: int = 0
    # Optional for alltoall_model == "pairwise_steps" when chunking is enabled:
    # limit the number of in-flight sub-chunks per (src,dst) peer pair.
    #
    # - 0 (default): keep current behavior (all sub-chunks for a (src,dst) start together within a phase).
    # - >0: schedule sub-chunks as a small pipeline where at most this many chunks are in flight per peer.
    #
    # This tends to be more realistic than "all chunks parallel" because real NCCL has limited per-peer
    # channel depth; recommended values: 2 or 4.
    alltoall_chunk_inflight_per_peer: int = 0
    # Number of parallel ring channels for nccl_ring models (allreduce/allgather/reducescatter).
    # Channel k rotates the ring by k positions within each server; in rail topology
    # anchor GPU_k <-> NIC_k, distributing cross-server flows across all NICs.
    # Each channel independently carries 1/nchannels of the tensor.
    # Only applies to nccl_ring; ring_steps always uses a single ring regardless of this value.
    # Default 1 = backward compatible. NCCL H800 8-NIC rail typical: 8 (or 16 w/ DupChannels).
    nchannels: int = 1
    # For collective_type == "p2p":
    # Within each domain group g, connect exactly one pair chosen by indices.
    # This makes it easy to express “1024 pairs” by using an outer dimension
    # (e.g., EP=1024) so each group has size 2 (e.g., DP=2).
    p2p_src_index: int = 0
    p2p_dst_index: int = 1
    # "0->1" | "1->0" | "bidir"
    p2p_direction: str = "0->1"
    # Optional explicit participant ranks in global rank space.
    # When set, this single group is used for flow generation.
    participant_ranks: Tuple[int, ...] = tuple()

    # For collective_type == "t2g" (Train -> Gen):
    # We model a directional P2P workload from a subset of GPUs in one pod (train) to a subset of GPUs
    # in another pod (gen). The default send rule matches “each train server's 8 GPUs -> one gen GPU”.
    #
    # Pods are identified by DP coordinate when using placement_order like [EP,DP] with dp=2.
    t2g_train_pod: int = 0
    t2g_gen_pod: int = 1
    # Active GPU count within each pod (prefix of EP indices within that pod):
    t2g_train_gpus: int = 0
    t2g_gen_gpus: int = 0
    # Sender grouping:
    # - "server": group senders by physical server (default), then split within each server by group_size.
    # - "global": group senders globally by group_size (ignores server boundaries).
    t2g_group_by: str = "server"
    # Number of senders per destination GPU (k in k->1). Default 8 matches 8 GPUs/server.
    t2g_sender_group_size: int = 8
    # Destination selection policy:
    # - "permute": permute the gen GPU list using seed (default, avoids same-index hot spot).
    # - "round_robin": simple modulo without permutation.
    t2g_dst_policy: str = "permute"
    # If true, each sender group maps to a distinct destination until gen GPUs are exhausted; then wrap.
    # If false, still wraps, but permutation may repeat earlier.
    t2g_unique_until_exhaust: bool = True
    # Traffic mode:
    # - "grouped" (default): k->1 traffic using sender grouping + dst_policy (existing behavior).
    # - "pairwise_same_ep": 1->1 traffic: for e in [0,train_gpus), send (train_pod,e)->(gen_pod,e).
    t2g_mode: str = "grouped"


def _parse_dim_list(s: str) -> Tuple[str, ...]:
    parts = [p.strip().upper() for p in s.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError("empty dim list")
    for p in parts:
        if p not in KNOWN_DIMS:
            raise ValueError(f"unknown dim {p!r}, must be one of {KNOWN_DIMS}")
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out)


def _parse_rank_list(raw: Any) -> Tuple[int, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        text = str(raw).strip()
        if not text:
            return tuple()
        items = [part for part in text.replace(",", " ").split() if part]

    ranks = tuple(int(part) for part in items)
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"participant_ranks must not contain duplicates: {ranks}")
    return ranks


def _parse_p2p_direction(v: Any) -> str:
    s = str(v).strip().lower()
    if s in ("bidir", "bidirectional", "both", "2way", "two-way"):
        return "bidir"
    if s in ("0->1", "0-1", "forward", "fwd"):
        return "0->1"
    if s in ("1->0", "1-0", "reverse", "rev"):
        return "1->0"
    raise ValueError("p2p_direction must be one of: 0->1, 1->0, bidir")


def _as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _as_int_nonneg(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        x = int(v)
        return x if x >= 0 else default
    except Exception:
        return default


def _coords_get_node_id(
    coord_to_node: Dict[Tuple[int, int, int, int], int],
    *,
    tp: int,
    cp: int,
    dp: int,
    ep: int,
    t: int = 0,
    c: int = 0,
    d: int = 0,
    e: int = 0,
) -> int:
    """
    Helper to lookup node_id for a coordinate (t,c,d,e).
    NOTE: _build_layout always stores coord_to_node keys as 4-int tuples:
      (tp_idx, cp_idx, dp_idx, ep_idx)
    even when a dimension size is 1 (index is 0).
    """
    _ = (tp, cp, dp, ep)  # keep signature stable; sizes validated elsewhere
    key = (int(t), int(c), int(d), int(e))
    return int(coord_to_node[key])


def _iter_coords(order: Tuple[str, ...], sizes: Dict[str, int]) -> list[Dict[str, int]]:
    dims = list(order)
    res: list[Dict[str, int]] = []

    def rec_outer(i: int, cur: Dict[str, int]) -> None:
        if i < 0:
            res.append(dict(cur))
            return
        d = dims[i]
        n = int(sizes[d])
        for v in range(n):
            cur[d] = v
            rec_outer(i - 1, cur)

    if not dims:
        return [dict()]
    rec_outer(len(dims) - 1, {})
    return res


def _normalize_dim_order(order: Tuple[str, ...]) -> Tuple[str, ...]:
    dims = [d for d in order if d in KNOWN_DIMS]
    for d in KNOWN_DIMS:
        if d not in dims:
            dims.append(d)
    out: list[str] = []
    for d in dims:
        if d not in out:
            out.append(d)
    return tuple(out)


def _build_layout(
    *,
    nodes: int,
    servers: int,
    gpus_per_server: int,
    tp: int,
    cp: int,
    dp: int,
    ep: int,
    placement_order: Tuple[str, ...],
) -> Tuple[Dict[Tuple[int, int, int, int], int], Dict[int, int]]:
    if nodes != servers * gpus_per_server:
        raise ValueError(f"nodes({nodes}) must equal servers({servers})*gpus_per_server({gpus_per_server})")
    if nodes != tp * cp * dp * ep:
        raise ValueError(f"nodes({nodes}) must equal tp*cp*dp*ep = {tp}*{cp}*{dp}*{ep} = {tp*cp*dp*ep}")

    sizes = {DIM_TP: tp, DIM_CP: cp, DIM_DP: dp, DIM_EP: ep}
    coords = _iter_coords(_normalize_dim_order(placement_order), sizes)
    if len(coords) != nodes:
        raise AssertionError(f"internal error: coord count {len(coords)} != nodes {nodes}")

    coord_to_node: Dict[Tuple[int, int, int, int], int] = {}
    node_to_server: Dict[int, int] = {}
    node_id = 0
    for s in range(servers):
        for _g in range(gpus_per_server):
            c = coords[node_id]
            coord_to_node[(c.get(DIM_TP, 0), c.get(DIM_CP, 0), c.get(DIM_DP, 0), c.get(DIM_EP, 0))] = node_id
            node_to_server[node_id] = s
            node_id += 1
    return coord_to_node, node_to_server


def _groups_for_domain(
    *,
    tp: int,
    cp: int,
    dp: int,
    ep: int,
    domain_dims: Tuple[str, ...],
    coord_to_node: Dict[Tuple[int, int, int, int], int],
) -> list[list[int]]:
    sizes = {DIM_TP: tp, DIM_CP: cp, DIM_DP: dp, DIM_EP: ep}
    dom = tuple(domain_dims)
    compl = tuple(d for d in KNOWN_DIMS if d not in dom)

    fixed_list = _iter_coords(compl, sizes)
    inner_list = _iter_coords(dom, sizes)

    groups: list[list[int]] = []
    for fixed in fixed_list:
        nodes: list[int] = []
        for inner in inner_list:
            tp_i = inner[DIM_TP] if DIM_TP in dom else fixed.get(DIM_TP, 0)
            cp_i = inner[DIM_CP] if DIM_CP in dom else fixed.get(DIM_CP, 0)
            dp_i = inner[DIM_DP] if DIM_DP in dom else fixed.get(DIM_DP, 0)
            ep_i = inner[DIM_EP] if DIM_EP in dom else fixed.get(DIM_EP, 0)
            nodes.append(coord_to_node[(tp_i, cp_i, dp_i, ep_i)])
        groups.append(nodes)
    return groups


@dataclass(frozen=True)
class TriggerDef:
    id: int
    type: str  # "oneshot" | "multishot" | "barrier"
    count: int = 0  # used for barrier


@dataclass(frozen=True)
class FlowDef:
    src: int
    dst: int
    size: int
    # Exactly one of (start, trigger) must be set.
    start: Optional[int] = None
    trigger: Optional[int] = None
    # Optional “done” triggers:
    send_done_trigger: Optional[int] = None
    recv_done_trigger: Optional[int] = None
    prio: Optional[int] = None


def _write_tm(cm_path: Path, *, nodes: int, flows: list[FlowDef], triggers: list[TriggerDef]) -> None:
    lines: list[str] = [
        f"Nodes {nodes}",
        f"Connections {len(flows)}",
        f"Triggers {len(triggers)}",
    ]

    for i, f in enumerate(flows):
        if (f.start is None) == (f.trigger is None):
            raise ValueError("tm writer error: each flow must specify exactly one of start or trigger")

        # IMPORTANT: "id" must appear before "trigger" (connection_matrix.cpp asserts flowid exists
        # when a trigger reference is parsed).
        parts: list[str] = [f"{f.src}->{f.dst}", "id", str(i + 1)]
        if f.start is not None:
            parts += ["start", str(f.start)]
        else:
            parts += ["trigger", str(int(f.trigger))]
        parts += ["size", str(int(f.size))]
        if f.send_done_trigger:
            parts += ["send_done_trigger", str(int(f.send_done_trigger))]
        if f.recv_done_trigger:
            parts += ["recv_done_trigger", str(int(f.recv_done_trigger))]
        if f.prio:
            parts += ["prio", str(int(f.prio))]
        lines.append(" ".join(parts))

    for t in triggers:
        if t.id <= 0:
            raise ValueError("tm writer error: trigger id must be > 0")
        if t.type not in ("oneshot", "multishot", "barrier"):
            raise ValueError(f"tm writer error: unsupported trigger type {t.type!r}")
        if t.type == "barrier":
            if t.count <= 0:
                raise ValueError("tm writer error: barrier trigger requires count > 0")
            lines.append(f"trigger id {t.id} barrier count {t.count}")
        else:
            lines.append(f"trigger id {t.id} {t.type}")

    lines.append("")
    cm_path.write_text("\n".join(lines))


def _read_connections(cm_path: Path) -> Optional[int]:
    try:
        head = cm_path.read_text(errors="ignore").splitlines()[:5]
        for ln in head:
            if ln.startswith("Connections "):
                return int(ln.split()[1])
    except Exception:
        pass
    return None


def _sum_tm_bytes(cm_path: Path) -> int:
    total = 0
    for ln in cm_path.read_text(errors="ignore").splitlines():
        if " size " in ln:
            try:
                # Robust parsing: allow trailing fields after size (e.g., send_done_trigger/recv_done_trigger/prio).
                total += int(ln.rsplit(" size ", 1)[1].split()[0])
            except Exception:
                pass
    return total


def _group_by_server(nodes: list[int], node_to_server: dict[int, int]) -> dict[int, list[int]]:
    """Group node IDs by physical server; each server's list is sorted."""
    by_server: dict[int, list[int]] = {}
    for n in nodes:
        by_server.setdefault(node_to_server[n], []).append(n)
    for srv in by_server:
        by_server[srv].sort()
    return by_server


def _nccl_delta_order(n_servers: int) -> list[int]:
    """
    Inter-server delta visit order following NCCL's zigzag pattern (v2.21 init.cc:1314).
    n_servers=4 -> [2, 1, 3]; n_servers=8 -> [4, 1, 7, 3, 5, 2, 6]
    """
    S = n_servers
    order: list[int] = []
    visited = {0}
    for d in range(S // 4 + 1):
        for cand in (d, S - d, S // 2 - d, S - (S // 2 - d)):
            if cand not in visited and 0 < cand < S:
                order.append(cand)
                visited.add(cand)
    return order


def _make_step_barriers(
    n_steps: int,
    flows_per_step: int,
    triggers: list[TriggerDef],
    *,
    use_triggers: bool,
) -> list[tuple[Optional[int], Optional[int]]]:
    """
    Build barrier triggers for n_steps sequential phases.
    Returns list of (start_trigger_id, done_trigger_id) per step.
    If use_triggers=False or n_steps<=1: [(None,None)]*n_steps.
    """
    if not use_triggers or n_steps <= 1:
        return [(None, None)] * n_steps
    barrier_ids: list[int] = []
    for _ in range(n_steps - 1):
        tid = len(triggers) + 1
        triggers.append(TriggerDef(id=tid, type="barrier", count=flows_per_step))
        barrier_ids.append(tid)
    return [
        (barrier_ids[s - 1] if s > 0 else None,
         barrier_ids[s]     if s < n_steps - 1 else None)
        for s in range(n_steps)
    ]


def _build_channel_rings(
    by_server: dict[int, list[int]],
    server_ids: list[int],
    nchannels: int,
) -> list[list[int]]:
    """
    Build nchannels ring orderings, each a cyclic rotation of the base ring.
    Channel k rotates the intra-server GPU order by k positions:
      GPU_(k%G) becomes the anchor -> NIC_k in rail topology.
    Falls back to base ring for nchannels==1 or non-uniform server membership.
    """
    if nchannels == 1:
        return [[gpu for srv in server_ids for gpu in by_server[srv]]]
    G = len(by_server[server_ids[0]])
    if any(len(by_server[srv]) != G for srv in server_ids):
        base = [gpu for srv in server_ids for gpu in by_server[srv]]
        return [base] * nchannels
    return [
        [by_server[srv][(ch % G + j) % G] for srv in server_ids for j in range(G)]
        for ch in range(nchannels)
    ]


_INTRA_SERVER_BYTES = 1


def _emit_ring_steps(
    ring: list[int],
    chunk_size: int,
    n_steps: int,
    *,
    step_barriers: list[tuple[Optional[int], Optional[int]]],
    node_to_server: dict[int, int],
    exclude_intra_server: bool,
    flows: list[FlowDef],
    intra_size: int = _INTRA_SERVER_BYTES,
) -> None:
    """Emit n_steps of ring flows: ring[i] -> ring[(i+1)%n]."""
    n = len(ring)
    for _s, (start_trig, done_trig) in enumerate(step_barriers):
        start_time_r: Optional[int] = 0 if start_trig is None else None
        for i in range(n):
            src, dst = ring[i], ring[(i + 1) % n]
            size = (intra_size
                    if exclude_intra_server and node_to_server[src] == node_to_server[dst]
                    else chunk_size)
            flows.append(FlowDef(
                src=src, dst=dst, start=start_time_r, trigger=start_trig,
                size=size, send_done_trigger=done_trig,
            ))


def _emit_server_ring_steps(
    server_ids: list[int],
    by_server: dict[int, list[int]],
    chunk_size: int,
    *,
    step_barriers: list[tuple[Optional[int], Optional[int]]],
    flows: list[FlowDef],
) -> None:
    """Emit hierarchical_ring flows: srv[i] -> srv[(i+1)%S], striped by GPU index."""
    S = len(server_ids)
    for _step, (start_trig, done_trig) in enumerate(step_barriers):
        start_time_r: Optional[int] = 0 if start_trig is None else None
        for i in range(S):
            s_nodes = by_server[server_ids[i]]
            t_nodes = by_server[server_ids[(i + 1) % S]]
            for idx in range(min(len(s_nodes), len(t_nodes))):
                flows.append(FlowDef(
                    src=s_nodes[idx], dst=t_nodes[idx],
                    start=start_time_r, trigger=start_trig,
                    size=chunk_size, send_done_trigger=done_trig,
                ))


def generate_tm_from_spec(
    out_dir: Path,
    *,
    spec: CollectiveSpec,
    nodes: int,
    servers: int,
    gpus_per_server: int,
    tp: int,
    cp: int,
    dp: int,
    ep: int,
    tensor_bytes: int,
    seed: int,
) -> Tuple[Path, int]:
    _ = seed
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / "tm.cm"

    # Align with old runner: require TP to divide evenly within a server for packing
    if gpus_per_server % tp != 0:
        raise ValueError("require gpus_per_server % tp == 0 (so TP can be packed within server)")

    coord_to_node, node_to_server = _build_layout(
        nodes=nodes,
        servers=servers,
        gpus_per_server=gpus_per_server,
        tp=tp,
        cp=cp,
        dp=dp,
        ep=ep,
        placement_order=spec.placement_order,
    )

    groups = _groups_for_domain(tp=tp, cp=cp, dp=dp, ep=ep, domain_dims=spec.domain_dims, coord_to_node=coord_to_node)
    if spec.participant_ranks:
        participant_ranks = tuple(int(rank) for rank in spec.participant_ranks)
        if len(participant_ranks) < 2:
            raise ValueError("collective.participant_ranks must contain at least 2 ranks")
        if any(rank < 0 or rank >= nodes for rank in participant_ranks):
            raise ValueError(
                f"collective.participant_ranks must be within [0, {nodes - 1}], got {participant_ranks}"
            )
        groups = [list(participant_ranks)]

    flows: list[FlowDef] = []
    triggers: list[TriggerDef] = []
    if spec.collective_type == "alltoall":
        for g in groups:
            n = len(g)
            if n <= 1:
                continue
            # Legacy runner semantics: chunk_total = ceil(tensor_bytes / n)  (per-peer)
            chunk_total = (tensor_bytes + n - 1) // n
            # Optional override: split each (src,dst) chunk_total into sub-chunks.
            chunk_bytes = int(getattr(spec, "alltoall_chunk_bytes", 0) or 0)
            if chunk_bytes <= 0:
                chunk_bytes = chunk_total
            if chunk_bytes > chunk_total:
                chunk_bytes = chunk_total
            inflight = int(getattr(spec, "alltoall_chunk_inflight_per_peer", 0) or 0)
            if inflight < 0:
                inflight = 0

            def _emit_chunked(
                *,
                src: int,
                dst: int,
                start_time: Optional[int],
                start_trigger: Optional[int],
                send_done_tid: Optional[int],
            ) -> int:
                """
                Emit 1 or more flows for (src,dst) totaling chunk_total bytes.
                Returns number of flows emitted (used to size barrier count).
                """
                remaining = int(chunk_total)
                cnt = 0
                while remaining > 0:
                    sz = chunk_bytes if remaining > chunk_bytes else remaining
                    flows.append(
                        FlowDef(
                            src=src,
                            dst=dst,
                            start=start_time,
                            trigger=start_trigger,
                            size=int(sz),
                            send_done_trigger=send_done_tid,
                        )
                    )
                    remaining -= int(sz)
                    cnt += 1
                return cnt

            def _emit_chunked_pipeline(
                *,
                src: int,
                dst: int,
                start_time: Optional[int],
                start_trigger: Optional[int],
                phase_done_tid: Optional[int],
            ) -> int:
                """
                Emit flows for one (src,dst) transfer totaling chunk_total bytes, with an optional
                per-peer in-flight cap for chunked transfers.

                When inflight > 0 and chunk_total is split into multiple chunks, we schedule a
                credit-based pipeline: chunk i (i>=inflight) starts when chunk (i-inflight) finishes.
                NOTE (important for htsim_ndp):
                - In this simulator, NdpSrc triggers end events (send_done_trigger), but NdpSink does NOT
                  activate end triggers (recv_done_trigger is effectively ignored).
                - Therefore, we MUST drive phase barriers using send_done_trigger, not recv_done_trigger.
                  We do so by attaching phase_done_tid to the *last* chunk of each (src,dst) pair, so each
                  pair contributes exactly one activation to the phase barrier.
                """
                num_chunks = int((chunk_total + chunk_bytes - 1) // chunk_bytes)
                if inflight <= 0 or num_chunks <= 1:
                    # Keep existing behavior: all chunks start together in the phase.
                    return _emit_chunked(
                        src=src,
                        dst=dst,
                        start_time=start_time,
                        start_trigger=start_trigger,
                        send_done_tid=None,
                    )

                # For i>=inflight we create a oneshot trigger to start that chunk; it will be fired by
                # the completion of chunk (i-inflight) via send_done_trigger.
                start_tids: list[Optional[int]] = [None] * num_chunks
                for i in range(num_chunks):
                    if i < inflight:
                        # Start with the phase.
                        start_tids[i] = start_trigger  # may be None when time-based
                    else:
                        tid = len(triggers) + 1
                        triggers.append(TriggerDef(id=tid, type="oneshot"))
                        start_tids[i] = tid

                # credit release: chunk i completion releases chunk (i+inflight)
                release_tid_for_i: list[Optional[int]] = [None] * num_chunks
                for i in range(num_chunks):
                    j = i + inflight
                    if j < num_chunks:
                        release_tid_for_i[i] = start_tids[j]

                cnt = 0
                remaining = int(chunk_total)
                for i in range(num_chunks):
                    sz = chunk_bytes if remaining > chunk_bytes else remaining

                    # Determine this chunk's start method.
                    if i < inflight:
                        st = start_time
                        trig = start_trigger
                    else:
                        st = None
                        trig = int(start_tids[i])  # type: ignore[arg-type]

                    # We can only attach ONE send_done_trigger per flow in tm.cm.
                    # Priority:
                    # - If this chunk releases a later chunk, use send_done_trigger for that (credit pipeline).
                    # - Else, if this is the last chunk for this (src,dst), use send_done_trigger to contribute
                    #   to the phase barrier (phase_done_tid). This makes each pair contribute exactly one
                    #   activation to the barrier.
                    sdt = release_tid_for_i[i]
                    if sdt is None and i == num_chunks - 1:
                        sdt = phase_done_tid

                    flows.append(
                        FlowDef(
                            src=src,
                            dst=dst,
                            start=st,
                            trigger=trig,
                            size=int(sz),
                            send_done_trigger=sdt,
                        )
                    )
                    remaining -= int(sz)
                    cnt += 1
                return cnt
            if getattr(spec, "alltoall_model", "pairwise_steps") == "nccl_pairwise":
                by_server_a2a = _group_by_server(g, node_to_server)
                server_ids_a2a = sorted(by_server_a2a.keys())
                S_a2a = len(server_ids_a2a)
                G_a2a = len(by_server_a2a[server_ids_a2a[0]])
                if not any(len(by_server_a2a[srv]) != G_a2a for srv in server_ids_a2a):
                    inter_deltas = _nccl_delta_order(S_a2a)
                    micro_rounds: list[list[tuple[int, int]]] = []
                    for delta in inter_deltas:
                        for sub_step in range(G_a2a):
                            round_pairs: list[tuple[int, int]] = []
                            for src_s_idx in range(S_a2a):
                                dst_s_idx = (src_s_idx + delta) % S_a2a
                                src_nodes_a2a = by_server_a2a[server_ids_a2a[src_s_idx]]
                                dst_nodes_a2a = by_server_a2a[server_ids_a2a[dst_s_idx]]
                                for local_i in range(G_a2a):
                                    src_node = src_nodes_a2a[local_i]
                                    dst_node = dst_nodes_a2a[(local_i + sub_step) % G_a2a]
                                    if spec.exclude_intra_server and node_to_server[src_node] == node_to_server[dst_node]:
                                        continue
                                    round_pairs.append((src_node, dst_node))
                            if round_pairs:
                                micro_rounds.append(round_pairs)
                    prev_tid_a2a: Optional[int] = None
                    for r_idx, round_pairs in enumerate(micro_rounds):
                        start_time_a2a: Optional[int] = 0 if prev_tid_a2a is None else None
                        start_trigger_a2a: Optional[int] = prev_tid_a2a
                        send_done_tid_a2a: Optional[int] = None
                        if r_idx < len(micro_rounds) - 1:
                            tid = len(triggers) + 1
                            triggers.append(TriggerDef(id=tid, type="barrier", count=len(round_pairs)))
                            send_done_tid_a2a = tid
                        for src, dst in round_pairs:
                            flows.append(FlowDef(
                                src=src, dst=dst, start=start_time_a2a, trigger=start_trigger_a2a,
                                size=int(chunk_total), send_done_trigger=send_done_tid_a2a,
                            ))
                        prev_tid_a2a = send_done_tid_a2a
                    continue
            elif getattr(spec, "alltoall_model", "pairwise_steps") == "pairwise_steps":
                # Phase schedule: partition the (n-1) destination offsets into batches of size K,
                # where K=alltoall_channels. Each batch is one phase, and phases are chained by a
                # barrier trigger so they don't all start at time 0.
                k = int(getattr(spec, "alltoall_channels", 1) or 1)
                if k <= 0:
                    k = 1
                offsets = list(range(1, n))  # exclude self
                phase_batches = [offsets[i : i + k] for i in range(0, len(offsets), k)]

                # Build per-phase pair lists first, so we can:
                # - drop empty phases (possible when exclude_intra_server filters everything), and
                # - set barrier count exactly to the number of flows in that phase (avoid deadlock).
                phase_pairs: list[list[tuple[int, int]]] = []
                for offs in phase_batches:
                    pairs: list[tuple[int, int]] = []
                    for i in range(n):
                        src = g[i]
                        for off in offs:
                            dst = g[(i + off) % n]
                            if spec.exclude_intra_server and node_to_server[src] == node_to_server[dst]:
                                continue
                            pairs.append((src, dst))
                    if pairs:
                        phase_pairs.append(pairs)

                prev_tid: Optional[int] = None
                for p, pairs in enumerate(phase_pairs):
                    start_time: Optional[int] = 0 if prev_tid is None else None
                    start_trigger: Optional[int] = None if prev_tid is None else prev_tid
                    send_done_tid: Optional[int] = None
                    if p < len(phase_pairs) - 1:
                        tid = len(triggers) + 1
                        if inflight > 0 and chunk_bytes < chunk_total:
                            # Pipeline mode: barrier is driven by LAST chunk per pair => one activation per pair.
                            triggers.append(TriggerDef(id=tid, type="barrier", count=int(len(pairs))))
                        else:
                            # Non-pipeline mode: every flow contributes one activation.
                            phase_flow_count = len(pairs) * int((chunk_total + chunk_bytes - 1) // chunk_bytes)
                            triggers.append(TriggerDef(id=tid, type="barrier", count=int(phase_flow_count)))
                        send_done_tid = tid
                    for (src, dst) in pairs:
                        if inflight > 0 and chunk_bytes < chunk_total:
                            # More realistic per-peer chunk pipeline (per-peer inflight cap) + phase barrier.
                            _emit_chunked_pipeline(
                                src=src,
                                dst=dst,
                                start_time=start_time,
                                start_trigger=start_trigger,
                                phase_done_tid=send_done_tid,
                            )
                        else:
                            _emit_chunked(src=src, dst=dst, start_time=start_time, start_trigger=start_trigger, send_done_tid=send_done_tid)
                    prev_tid = send_done_tid
            else:
                # full_mesh: all pairs start at time 0
                for src in g:
                    for dst in g:
                        if dst == src:
                            continue
                        if spec.exclude_intra_server and node_to_server[src] == node_to_server[dst]:
                            continue
                        _emit_chunked(src=src, dst=dst, start_time=0, start_trigger=None, send_done_tid=None)
    elif spec.collective_type == "allreduce":
        for g in groups:
            n = len(g)
            if n <= 1:
                continue
            by_server = _group_by_server(g, node_to_server)
            server_ids = sorted(by_server.keys())

            if spec.allreduce_model == "ring_stream":
                # Pipelined steady-state: 2*(n-1)/n * tensor_bytes per edge, single ring.
                per_edge = (2 * (n - 1) * tensor_bytes + (n - 1)) // n
                for i in range(n):
                    src, dst = g[i], g[(i + 1) % n]
                    size = (_INTRA_SERVER_BYTES
                            if spec.exclude_intra_server and node_to_server[src] == node_to_server[dst]
                            else per_edge)
                    flows.append(FlowDef(src=src, dst=dst, start=0, size=size))

            elif spec.allreduce_model == "nccl_ring":
                # NCCL-accurate: nchannels independent rotated rings, each with 2*(n-1) barrier-chained steps.
                # Per channel chunk = tensor_bytes / (n * nchannels).
                nch = spec.nchannels
                chunk_per_ch = (tensor_bytes + n * nch - 1) // (n * nch)
                n_steps = 2 * (n - 1)
                for ring in _build_channel_rings(by_server, server_ids, nch):
                    # Each channel gets its own independent barrier chain (use_triggers always True).
                    step_barriers = _make_step_barriers(n_steps, n, triggers, use_triggers=True)
                    _emit_ring_steps(ring, chunk_per_ch, n_steps, step_barriers=step_barriers,
                                     node_to_server=node_to_server,
                                     exclude_intra_server=spec.exclude_intra_server, flows=flows)

            else:  # ring_steps (default) — single ring, nchannels does not apply
                chunk = (tensor_bytes + n - 1) // n
                n_steps = 2 * (n - 1)
                for ring in _build_channel_rings(by_server, server_ids, 1):
                    step_barriers = _make_step_barriers(n_steps, n, triggers,
                                                        use_triggers=spec.use_triggers)
                    _emit_ring_steps(ring, chunk, n_steps, step_barriers=step_barriers,
                                     node_to_server=node_to_server,
                                     exclude_intra_server=spec.exclude_intra_server, flows=flows)
    elif spec.collective_type in ("allgather", "reducescatter"):
        model = spec.allgather_model if spec.collective_type == "allgather" else spec.reducescatter_model

        for g in groups:
            n = len(g)
            if n <= 1:
                continue
            by_server = _group_by_server(g, node_to_server)
            server_ids = sorted(by_server.keys())

            # --- Hierarchical models (inter-server NIC phase only) ---
            if model in ("hierarchical", "hierarchical_ring") and len(server_ids) > 1:
                if model == "hierarchical":
                    for s in server_ids:
                        for t in server_ids:
                            if t == s:
                                continue
                            s_nodes, t_nodes = by_server[s], by_server[t]
                            for idx in range(min(len(s_nodes), len(t_nodes))):
                                flows.append(FlowDef(src=s_nodes[idx], dst=t_nodes[idx],
                                                     start=0, size=int(tensor_bytes)))
                else:  # hierarchical_ring
                    S = len(server_ids)
                    flows_per_step = sum(
                        min(len(by_server[server_ids[i]]), len(by_server[server_ids[(i + 1) % S]]))
                        for i in range(S)
                    )
                    step_barriers = _make_step_barriers(S - 1, flows_per_step, triggers,
                                                        use_triggers=spec.use_triggers)
                    _emit_server_ring_steps(server_ids, by_server, int(tensor_bytes),
                                            step_barriers=step_barriers, flows=flows)
                continue

            # --- nccl_ring: multi-channel ring with per-channel step barriers ---
            if model == "nccl_ring":
                # Each channel: (n-1) barrier-chained steps, chunk = tensor_bytes / nchannels.
                nch = spec.nchannels
                chunk_per_ch = (tensor_bytes + nch - 1) // nch
                n_steps = n - 1
                for ring in _build_channel_rings(by_server, server_ids, nch):
                    # Each channel gets its own independent barrier chain (use_triggers always True).
                    step_barriers = _make_step_barriers(n_steps, n, triggers, use_triggers=True)
                    _emit_ring_steps(ring, chunk_per_ch, n_steps, step_barriers=step_barriers,
                                     node_to_server=node_to_server,
                                     exclude_intra_server=spec.exclude_intra_server, flows=flows)
                continue

            # --- ring_steps: (n-1) serial steps, single ring, nchannels does not apply ---
            chunk = int(tensor_bytes)
            n_steps = n - 1
            for ring in _build_channel_rings(by_server, server_ids, 1):
                step_barriers = _make_step_barriers(n_steps, n, triggers,
                                                    use_triggers=spec.use_triggers)
                _emit_ring_steps(ring, chunk, n_steps, step_barriers=step_barriers,
                                 node_to_server=node_to_server,
                                 exclude_intra_server=spec.exclude_intra_server, flows=flows)
    elif spec.collective_type == "p2p":
        for g in groups:
            n = len(g)
            if n <= 1:
                continue
            if spec.p2p_src_index < 0 or spec.p2p_src_index >= n:
                raise ValueError(f"p2p_src_index={spec.p2p_src_index} out of range for group size {n}")
            if spec.p2p_dst_index < 0 or spec.p2p_dst_index >= n:
                raise ValueError(f"p2p_dst_index={spec.p2p_dst_index} out of range for group size {n}")
            if spec.p2p_src_index == spec.p2p_dst_index:
                continue

            def emit(src: int, dst: int) -> None:
                if spec.exclude_intra_server and node_to_server[src] == node_to_server[dst]:
                    return
                flows.append(FlowDef(src=src, dst=dst, start=0, size=tensor_bytes))

            a = g[spec.p2p_src_index]
            b = g[spec.p2p_dst_index]
            if spec.p2p_direction == "0->1":
                emit(a, b)
            elif spec.p2p_direction == "1->0":
                emit(b, a)
            elif spec.p2p_direction == "bidir":
                emit(a, b)
                emit(b, a)
            else:
                raise ValueError(f"Unsupported p2p_direction: {spec.p2p_direction}")
    elif spec.collective_type == "t2g":
        # Train -> Gen traffic generator, designed for 2-pod fattree3 experiments.
        # We interpret pods via DP coordinate (requires dp>=2 and EP dimension for within-pod index).
        train_pod = int(getattr(spec, "t2g_train_pod", 0) or 0)
        gen_pod = int(getattr(spec, "t2g_gen_pod", 1) or 1)
        train_gpus = int(getattr(spec, "t2g_train_gpus", 0) or 0)
        gen_gpus = int(getattr(spec, "t2g_gen_gpus", 0) or 0)
        mode = str(getattr(spec, "t2g_mode", "grouped") or "grouped").strip().lower()
        if dp <= 1:
            raise ValueError("collective=t2g requires dp>=2 so pods can be represented (DP dimension).")
        if ep <= 1:
            raise ValueError("collective=t2g requires ep>=2 so GPUs within a pod can be indexed (EP dimension).")
        if train_pod < 0 or train_pod >= dp:
            raise ValueError(f"t2g_train_pod={train_pod} out of range for dp={dp}")
        if gen_pod < 0 or gen_pod >= dp:
            raise ValueError(f"t2g_gen_pod={gen_pod} out of range for dp={dp}")
        if train_pod == gen_pod:
            raise ValueError("t2g requires train_pod != gen_pod")
        if train_gpus <= 0 or train_gpus > ep:
            raise ValueError(f"t2g_train_gpus must be in [1, ep]; got {train_gpus}, ep={ep}")
        if gen_gpus <= 0 or gen_gpus > ep:
            raise ValueError(f"t2g_gen_gpus must be in [1, ep]; got {gen_gpus}, ep={ep}")

        if mode == "pairwise_same_ep":
            if gen_gpus < train_gpus:
                raise ValueError(f"t2g.mode=pairwise_same_ep requires gen_gpus>=train_gpus; got gen_gpus={gen_gpus}, train_gpus={train_gpus}")
            for eidx in range(train_gpus):
                src = _coords_get_node_id(coord_to_node, tp=tp, cp=cp, dp=dp, ep=ep, t=0, c=0, d=train_pod, e=eidx)
                dst = _coords_get_node_id(coord_to_node, tp=tp, cp=cp, dp=dp, ep=ep, t=0, c=0, d=gen_pod, e=eidx)
                flows.append(FlowDef(src=int(src), dst=int(dst), start=0, size=int(tensor_bytes)))
            # Done.
            _write_tm(cm_path, nodes=nodes, flows=flows, triggers=triggers)
            return cm_path, len(flows)
        if mode != "grouped":
            raise ValueError("t2g.mode must be one of: grouped | pairwise_same_ep")

        # Enumerate active node IDs in each pod (EP index prefix).
        train_nodes: list[int] = []
        gen_nodes: list[int] = []
        for eidx in range(ep):
            if eidx < train_gpus:
                train_nodes.append(
                    _coords_get_node_id(coord_to_node, tp=tp, cp=cp, dp=dp, ep=ep, t=0, c=0, d=train_pod, e=eidx)
                )
            if eidx < gen_gpus:
                gen_nodes.append(
                    _coords_get_node_id(coord_to_node, tp=tp, cp=cp, dp=dp, ep=ep, t=0, c=0, d=gen_pod, e=eidx)
                )

        # Group senders.
        group_by = str(getattr(spec, "t2g_group_by", "server") or "server").strip().lower()
        k = int(getattr(spec, "t2g_sender_group_size", gpus_per_server) or gpus_per_server)
        if k <= 0:
            k = 1

        sender_groups: list[list[int]] = []
        if group_by == "server":
            # group within each server to model “each server’s GPUs send together”.
            by_server: Dict[int, list[int]] = {}
            for n0 in train_nodes:
                s = int(node_to_server[n0])
                by_server.setdefault(s, []).append(int(n0))
            for s in sorted(by_server.keys()):
                xs = sorted(by_server[s])
                # chunk within server
                for i in range(0, len(xs), k):
                    sender_groups.append(xs[i : i + k])
        elif group_by == "global":
            xs = sorted(train_nodes)
            for i in range(0, len(xs), k):
                sender_groups.append(xs[i : i + k])
        else:
            raise ValueError("t2g_group_by must be one of: server | global")

        if not sender_groups:
            raise ValueError("t2g produced 0 sender groups (check t2g_train_gpus).")
        if not gen_nodes:
            raise ValueError("t2g produced 0 gen nodes (check t2g_gen_gpus).")

        # Build destination order.
        dst_policy = str(getattr(spec, "t2g_dst_policy", "permute") or "permute").strip().lower()
        unique_until_exhaust = bool(getattr(spec, "t2g_unique_until_exhaust", True))
        dst_list: list[int]
        if dst_policy == "permute":
            # "permute" with spread: try to scatter destinations across different gen servers AND
            # different local GPU ids, to avoid rail-like “same-index hot spot”.
            rng = random.Random(int(seed) ^ 0xC0FFEE)

            # Group gen nodes by physical server.
            gen_by_srv: Dict[int, list[int]] = {}
            for n0 in gen_nodes:
                gen_by_srv.setdefault(int(node_to_server[n0]), []).append(int(n0))
            gen_srvs = sorted(gen_by_srv.keys())
            rng.shuffle(gen_srvs)

            # Choose local GPU ids in a permuted cycle.
            # NOTE: under our node_id assignment, local gpu id == node_id % gpus_per_server.
            local_ids = list(range(int(gpus_per_server)))
            rng.shuffle(local_ids)

            # Per-server shuffled preference order of node_ids, keyed by local id.
            srv_local_to_node: Dict[int, Dict[int, int]] = {}
            for srv in gen_srvs:
                srv_nodes = list(gen_by_srv[srv])
                # map local_id -> node_id (only for active gen_gpus prefix)
                m: Dict[int, int] = {}
                for nid in srv_nodes:
                    m[int(nid % gpus_per_server)] = int(nid)
                srv_local_to_node[srv] = m

            # Build dst_list by iterating servers and assigning a rotating local_id,
            # falling back to any available local_id on that server if the chosen one is absent.
            dst_list = []
            for i, srv in enumerate(gen_srvs):
                want_lid = int(local_ids[i % len(local_ids)])
                m = srv_local_to_node[srv]
                if want_lid in m:
                    dst_list.append(int(m[want_lid]))
                else:
                    # fallback: pick the first available local id (stable given shuffled server order)
                    dst_list.append(int(m[sorted(m.keys())[0]]))

            # If we have more sender groups than servers, continue filling by cycling servers and
            # walking local_ids; this spreads across both dimensions as much as possible.
            if len(sender_groups) > len(dst_list):
                per_srv_used: Dict[int, int] = {srv: 1 for srv in gen_srvs}
                j = len(dst_list)
                while len(dst_list) < len(gen_nodes):
                    srv = gen_srvs[j % len(gen_srvs)]
                    m = srv_local_to_node[srv]
                    lid = local_ids[(j // len(gen_srvs)) % len(local_ids)]
                    if lid in m:
                        nid = m[lid]
                    else:
                        nid = m[sorted(m.keys())[per_srv_used[srv] % len(m)]]
                    # Avoid duplicates if possible.
                    if nid not in dst_list:
                        dst_list.append(int(nid))
                    per_srv_used[srv] += 1
                    j += 1
        elif dst_policy == "round_robin":
            dst_list = list(gen_nodes)
        else:
            raise ValueError("t2g_dst_policy must be one of: permute | round_robin")

        # Emit flows: every sender in a group sends tensor_bytes to one chosen dst GPU.
        for gid, senders in enumerate(sender_groups):
            if unique_until_exhaust:
                dst = dst_list[gid % len(dst_list)]
            else:
                dst = dst_list[(gid * 1315423911) % len(dst_list)]
            for src in senders:
                if spec.exclude_intra_server and node_to_server[src] == node_to_server[dst]:
                    continue
                flows.append(FlowDef(src=int(src), dst=int(dst), start=0, size=int(tensor_bytes)))
    else:
        raise ValueError(f"Unsupported collective_type: {spec.collective_type}")

    _write_tm(cm_path, nodes=nodes, flows=flows, triggers=triggers)
    return cm_path, len(flows)


# Match both decimal and scientific notation timestamps, e.g. "26232.8" or "1.03015e+06".
FINISHED_RE = re.compile(r" finished at ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) ")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


def _green(s: str) -> str:
    return _color(s, "32")


def _red(s: str) -> str:
    return _color(s, "31")


def _spec_get(spec: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = spec
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_pairs(v: Any) -> list[tuple[int, int]]:
    """
    Parse list-like pairs, e.g.:
      - [[0,4],[1,5]] or ["0-4","1-5"] or ["0,4", ...]
    """
    pairs: list[tuple[int, int]] = []
    if v is None:
        return pairs
    if isinstance(v, list):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a = int(item[0])
                b = int(item[1])
                pairs.append((a, b))
            else:
                s = str(item).strip()
                if "-" in s:
                    aa, bb = s.split("-", 1)
                elif "," in s:
                    aa, bb = s.split(",", 1)
                else:
                    raise ValueError(f"invalid pair token {item!r}; expect [a,b] or 'a-b'")
                pairs.append((int(aa.strip()), int(bb.strip())))
    else:
        raise ValueError("pairs must be a list (e.g. [[0,4],[1,5],...])")
    return pairs


def _validate_matching_pairs(*, tors: int, pairs: list[tuple[int, int]]) -> None:
    seen: set[int] = set()
    for a, b in pairs:
        if a < 0 or b < 0 or a >= tors or b >= tors:
            raise ValueError(f"schedule pair out of range for tors={tors}: {a}-{b}")
        if a == b:
            raise ValueError(f"schedule self-loop not allowed: {a}-{b}")
        if a in seen or b in seen:
            raise ValueError(f"schedule ToR appears twice in a plane: {a}-{b}")
        seen.add(a)
        seen.add(b)


def validate_rail_ocs_schedule_file(
    schedule_path: Path,
    *,
    tors: int,
    planes: int,
    require_all_planes: bool = True,
    slot: int = 0,
) -> None:
    """
    Lightweight validation for schedule file compatibility with inferred (tors, planes).
    It follows the same token format as C++ loader:
      slot 0
      plane 0 0-1 2-3 ...
    """
    if tors <= 0:
        raise ValueError("rail_ocs schedule validate: tors must be > 0")
    if planes <= 0:
        raise ValueError("rail_ocs schedule validate: planes must be > 0")
    if slot != 0:
        raise ValueError("rail_ocs schedule validate currently supports slot=0 only")
    if not schedule_path.exists():
        raise ValueError(f"rail_ocs schedule file not found: {schedule_path}")

    in_slot0 = False
    seen_planes: set[int] = set()
    for raw in schedule_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "slot":
            if len(toks) < 2:
                raise ValueError(f"rail_ocs schedule: bad slot line: {raw!r}")
            in_slot0 = (toks[1] == "0")
            continue
        if not in_slot0:
            continue
        if toks[0] != "plane":
            continue
        if len(toks) < 2:
            raise ValueError(f"rail_ocs schedule: bad plane line: {raw!r}")
        p = int(toks[1])
        if p < 0 or p >= planes:
            raise ValueError(f"rail_ocs schedule: plane out of range (planes={planes}): {p}")
        seen_planes.add(p)
        pairs: list[tuple[int, int]] = []
        for tok in toks[2:]:
            if "-" not in tok:
                raise ValueError(f"rail_ocs schedule: bad pair token {tok!r} in line: {raw!r}")
            aa, bb = tok.split("-", 1)
            pairs.append((int(aa), int(bb)))
        if pairs:
            _validate_matching_pairs(tors=tors, pairs=pairs)
        # Allow empty plane line (meaning no circuits) if user really wants it.
        # But in that case, matching validation is skipped.

    if require_all_planes and len(seen_planes) != planes:
        missing = [p for p in range(planes) if p not in seen_planes]
        raise ValueError(f"rail_ocs schedule: slot0 missing plane lines: {missing[:16]}{'...' if len(missing) > 16 else ''}")


def generate_rail_ocs_schedule(
    schedule_path: Path,
    *,
    tors: int,
    planes: int,
    pairs: list[tuple[int, int]],
    slot: int = 0,
    repeat_across_planes: bool = True,
) -> Path:
    if tors <= 0:
        raise ValueError("rail_ocs schedule: tors must be > 0")
    if planes <= 0:
        raise ValueError("rail_ocs schedule: planes must be > 0")
    if slot != 0:
        raise ValueError("rail_ocs schedule generator currently supports slot=0 only")
    if not pairs:
        raise ValueError("rail_ocs schedule: missing pairs (e.g. [[0,4],[1,5],...])")

    _validate_matching_pairs(tors=tors, pairs=pairs)
    toks = [f"{a}-{b}" for a, b in pairs]

    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Auto-generated by {__file__}",
        f"# RailOCS static schedule (slot 0 only) for {planes} planes, tors={tors}.",
        "",
        "slot 0",
        "",
    ]
    if not repeat_across_planes:
        raise ValueError("rail_ocs schedule generator currently supports repeat_across_planes=true only")
    for p in range(planes):
        lines.append(" ".join(["plane", str(p), *toks]))
    lines.append("")
    schedule_path.write_text("\n".join(lines))
    return schedule_path


def generate_rail_ocs_topo(
    topo_path: Path,
    *,
    servers: int,
    gpus_per_server: int,
    servers_per_tor: int,
    planes: int,
    link_speed_gbps: float,
    link_latency_ns: int,
    switch_latency_ns: int,
    ocs_queue_pkts: int,
    ocs_no_drop: bool,
    ocs_queue_type: str,
    ocs_cut_through: bool,
    ocs_link_latency_ns: int,
    ocs_switch_latency_ns: int,
    slot_us: float,
    reconfig_us: float,
    schedule_file: Path,
) -> Path:
    if servers <= 0:
        raise ValueError("rail_ocs topo: servers must be > 0")
    if gpus_per_server <= 0:
        raise ValueError("rail_ocs topo: gpus_per_server must be > 0")
    if servers_per_tor <= 0 or servers_per_tor > servers:
        raise ValueError("rail_ocs topo: invalid servers_per_tor")
    if planes <= 0:
        raise ValueError("rail_ocs topo: planes must be > 0")
    if link_speed_gbps <= 0:
        raise ValueError("rail_ocs topo: link_speed_gbps must be > 0")
    if slot_us <= 0:
        raise ValueError("rail_ocs topo: slot_us must be > 0")
    if reconfig_us < 0:
        raise ValueError("rail_ocs topo: reconfig_us must be >= 0")

    topo_path.parent.mkdir(parents=True, exist_ok=True)
    topo_path.write_text(
        "\n".join(
            [
                "Type RailOCS",
                "",
                f"# Auto-generated by {__file__}",
                "",
                f"Servers {servers}",
                f"GpusPerServer {gpus_per_server}",
                f"ServersPerTor {servers_per_tor}",
                "",
                f"Planes {planes}",
                "",
                f"LinkSpeedGbps {link_speed_gbps:g}",
                "",
                f"OcsQueuePkts {ocs_queue_pkts}",
                f"OcsNoDrop {1 if ocs_no_drop else 0}",
                f"OcsQueueType {str(ocs_queue_type or 'fifo')}",
                f"OcsCutThrough {1 if ocs_cut_through else 0}",
                f"OcsLinkLatencyNs {ocs_link_latency_ns}",
                f"OcsSwitchLatencyNs {ocs_switch_latency_ns}",
                "",
                f"LinkLatencyNs {link_latency_ns}",
                f"SwitchLatencyNs {switch_latency_ns}",
                "",
                f"SlotUs {slot_us:g}",
                f"ReconfigUs {reconfig_us:g}",
                "",
                f"ScheduleFile {str(schedule_file)}",
                "",
            ]
        )
    )
    return topo_path


def _load_spec_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"spec file not found: {path}")
    text = p.read_text(errors="ignore")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("spec root must be a mapping/dict")
    return data


def _merge_spec_into_args(args: argparse.Namespace, spec: Dict[str, Any]) -> None:
    def set_if_none(attr: str, value: Any) -> None:
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    def set_if_none_or_zero(attr: str, value: Any) -> None:
        # Treat "missing" values as absent: many spec fields are optional, and we
        # want CLI/default logic to fill them in. In YAML, missing numeric fields
        # are represented here via our default (0); skip those.
        if value in (None, 0, ""):
            return
        v = getattr(args, attr)
        if v is None or v == 0:
            setattr(args, attr, value)

    def set_if_none_or_empty(attr: str, value: Any) -> None:
        v = getattr(args, attr)
        if v is None or v == "":
            setattr(args, attr, value)

    set_if_none_or_empty("collective_type", _spec_get(spec, "collective.type", _spec_get(spec, "collective_type", "")))
    dom = _spec_get(spec, "collective.domain_dims", _spec_get(spec, "domain_dims", ""))
    if isinstance(dom, list):
        dom = ",".join(str(x) for x in dom)
    set_if_none_or_empty("domain_dims", dom)
    po = _spec_get(spec, "collective.placement_order", _spec_get(spec, "placement_order", ""))
    if isinstance(po, list):
        po = ",".join(str(x) for x in po)
    set_if_none_or_empty("placement_order", po)
    set_if_none("exclude_intra_server", _spec_get(spec, "collective.exclude_intra_server", None))
    # allreduce triggers (optional)
    set_if_none("use_triggers", _spec_get(spec, "collective.use_triggers", _spec_get(spec, "use_triggers", None)))
    # allreduce model (optional)
    set_if_none_or_empty("allreduce_model", _spec_get(spec, "collective.allreduce_model", _spec_get(spec, "allreduce_model", "")))
    # alltoall model + channels (optional)
    set_if_none_or_empty("alltoall_model", _spec_get(spec, "collective.alltoall_model", _spec_get(spec, "alltoall_model", "")))
    set_if_none_or_zero("alltoall_channels", _spec_get(spec, "collective.alltoall_channels", _spec_get(spec, "alltoall_channels", None)))
    set_if_none_or_zero("alltoall_chunk_bytes", _spec_get(spec, "collective.alltoall_chunk_bytes", _spec_get(spec, "alltoall_chunk_bytes", None)))
    set_if_none_or_zero(
        "alltoall_chunk_inflight_per_peer",
        _spec_get(spec, "collective.alltoall_chunk_inflight_per_peer", _spec_get(spec, "alltoall_chunk_inflight_per_peer", None)),
    )
    # allgather model (optional)
    set_if_none_or_empty("allgather_model", _spec_get(spec, "collective.allgather_model", _spec_get(spec, "allgather_model", "")))
    # reducescatter model (optional)
    set_if_none_or_empty("reducescatter_model", _spec_get(spec, "collective.reducescatter_model", _spec_get(spec, "reducescatter_model", "")))
    # nchannels (optional)
    set_if_none_or_zero("nchannels", _spec_get(spec, "collective.nchannels", _spec_get(spec, "nchannels", None)))
    # p2p (optional)
    set_if_none("p2p_src_index", _spec_get(spec, "collective.src_index", _spec_get(spec, "p2p_src_index", None)))
    set_if_none("p2p_dst_index", _spec_get(spec, "collective.dst_index", _spec_get(spec, "p2p_dst_index", None)))
    set_if_none("p2p_direction", _spec_get(spec, "collective.direction", _spec_get(spec, "p2p_direction", None)))
    participant_ranks = _spec_get(spec, "collective.participant_ranks", _spec_get(spec, "participant_ranks", None))
    if isinstance(participant_ranks, (list, tuple)):
        participant_ranks = ",".join(str(rank) for rank in participant_ranks)
    set_if_none_or_empty("participant_ranks", participant_ranks)

    # t2g (optional)
    # Nested under collective.t2g.* (preferred) or top-level t2g_* keys for convenience.
    set_if_none_or_zero("t2g_train_pod", _spec_get(spec, "collective.t2g.train_pod", _spec_get(spec, "t2g_train_pod", None)))
    set_if_none_or_zero("t2g_gen_pod", _spec_get(spec, "collective.t2g.gen_pod", _spec_get(spec, "t2g_gen_pod", None)))
    set_if_none_or_zero("t2g_train_gpus", _spec_get(spec, "collective.t2g.train_gpus", _spec_get(spec, "t2g_train_gpus", None)))
    set_if_none_or_zero("t2g_gen_gpus", _spec_get(spec, "collective.t2g.gen_gpus", _spec_get(spec, "t2g_gen_gpus", None)))
    set_if_none_or_zero("t2g_sender_group_size", _spec_get(spec, "collective.t2g.sender_group_size", _spec_get(spec, "t2g_sender_group_size", None)))
    set_if_none_or_empty("t2g_group_by", _spec_get(spec, "collective.t2g.group_by", _spec_get(spec, "t2g_group_by", "")))
    set_if_none_or_empty("t2g_dst_policy", _spec_get(spec, "collective.t2g.dst_policy", _spec_get(spec, "t2g_dst_policy", "")))
    set_if_none("t2g_unique_until_exhaust", _spec_get(spec, "collective.t2g.unique_until_exhaust", _spec_get(spec, "t2g_unique_until_exhaust", None)))
    set_if_none_or_empty("t2g_mode", _spec_get(spec, "collective.t2g.mode", _spec_get(spec, "t2g_mode", "")))

    set_if_none_or_zero("nodes", _spec_get(spec, "job.nodes", _spec_get(spec, "nodes", None)))
    set_if_none_or_zero("servers", _spec_get(spec, "job.servers", _spec_get(spec, "servers", None)))
    set_if_none_or_zero("gpus_per_server", _spec_get(spec, "job.gpus_per_server", _spec_get(spec, "gpus_per_server", None)))
    set_if_none_or_zero("tp", _spec_get(spec, "parallel.tp", _spec_get(spec, "tp", None)))
    set_if_none_or_zero("cp", _spec_get(spec, "parallel.cp", _spec_get(spec, "cp", None)))
    set_if_none_or_zero("dp", _spec_get(spec, "parallel.dp", _spec_get(spec, "dp", None)))
    set_if_none_or_zero("ep", _spec_get(spec, "parallel.ep", _spec_get(spec, "ep", None)))
    set_if_none_or_zero("tensor_bytes", _spec_get(spec, "collective.tensor_bytes", _spec_get(spec, "tensor_bytes", None)))
    set_if_none_or_zero("seed", _spec_get(spec, "seed", _spec_get(spec, "runner.seed", None)))

    set_if_none_or_empty("topology", _spec_get(spec, "topology.type", _spec_get(spec, "topology", "")))
    set_if_none_or_zero("spines", _spec_get(spec, "topology.spines", _spec_get(spec, "spines", None)))
    set_if_none_or_zero("servers_per_tor", _spec_get(spec, "topology.servers_per_tor", _spec_get(spec, "servers_per_tor", None)))
    set_if_none_or_empty("ocs_topo_file", _spec_get(spec, "topology.ocs_topo_file", _spec_get(spec, "ocs_topo_file", "")))
    set_if_none_or_empty("rail3_topo_file", _spec_get(spec, "topology.rail3_topo_file", _spec_get(spec, "rail3_topo_file", "")))
    # RailOCS3
    set_if_none_or_empty("ocs3_topo_file", _spec_get(spec, "topology.ocs3_topo_file", _spec_get(spec, "ocs3_topo_file", "")))
    set_if_none_or_empty("ocs3_dump_reach_file", _spec_get(spec, "runner.ocs3_dump_reach_file", _spec_get(spec, "ocs3_dump_reach_file", "")))
    # MixNet
    set_if_none_or_empty("mixnet_eps_topo_file", _spec_get(spec, "topology.mixnet.eps_topo_file", _spec_get(spec, "mixnet_eps_topo_file", "")))
    set_if_none_or_empty("mixnet_ocs_schedule_file", _spec_get(spec, "topology.mixnet.ocs_schedule_file", _spec_get(spec, "mixnet_ocs_schedule_file", "")))
    set_if_none_or_zero("mixnet_ocs_planes", _spec_get(spec, "topology.mixnet.ocs_planes", _spec_get(spec, "mixnet_ocs_planes", None)))
    set_if_none("mixnet_ocs_link_speed_gbps", _spec_get(spec, "topology.mixnet.ocs_link_speed_gbps", _spec_get(spec, "mixnet_ocs_link_speed_gbps", None)))
    set_if_none_or_zero("mixnet_ocs_queue_pkts", _spec_get(spec, "topology.mixnet.ocs_queue_pkts", _spec_get(spec, "mixnet_ocs_queue_pkts", None)))
    set_if_none("mixnet_ocs_no_drop", _spec_get(spec, "topology.mixnet.ocs_no_drop", _spec_get(spec, "mixnet_ocs_no_drop", None)))
    set_if_none_or_zero("mixnet_ocs_link_latency_ns", _spec_get(spec, "topology.mixnet.ocs_link_latency_ns", _spec_get(spec, "mixnet_ocs_link_latency_ns", None)))
    set_if_none_or_zero("mixnet_ocs_switch_latency_ns", _spec_get(spec, "topology.mixnet.ocs_switch_latency_ns", _spec_get(spec, "mixnet_ocs_switch_latency_ns", None)))
    set_if_none_or_zero("podsize", _spec_get(spec, "topology.podsize", _spec_get(spec, "podsize", None)))
    set_if_none_or_zero("tor_down", _spec_get(spec, "topology.tor_down", _spec_get(spec, "tor_down", None)))
    set_if_none_or_zero("tor_up", _spec_get(spec, "topology.tor_up", _spec_get(spec, "tor_up", None)))
    set_if_none_or_zero("fattree_oversub", _spec_get(spec, "topology.oversub", _spec_get(spec, "fattree_oversub", None)))
    set_if_none_or_zero("latency_ns", _spec_get(spec, "topology.latency_ns", _spec_get(spec, "latency_ns", None)))
    set_if_none_or_zero("switch_latency_ns", _spec_get(spec, "topology.switch_latency_ns", _spec_get(spec, "switch_latency_ns", None)))
    set_if_none("hop_latency_us", _spec_get(spec, "topology.hop_latency_us", _spec_get(spec, "hop_latency_us", None)))
    set_if_none("switch_latency_us", _spec_get(spec, "topology.switch_latency_us", _spec_get(spec, "switch_latency_us", None)))

    set_if_none_or_zero("linkspeed_mbps", _spec_get(spec, "network.linkspeed_mbps", _spec_get(spec, "linkspeed_mbps", None)))
    set_if_none_or_zero("mtu", _spec_get(spec, "network.mtu", _spec_get(spec, "mtu", None)))
    set_if_none_or_zero("q", _spec_get(spec, "network.q", _spec_get(spec, "q", None)))
    set_if_none_or_zero("cwnd", _spec_get(spec, "network.cwnd", _spec_get(spec, "cwnd", None)))
    set_if_none_or_empty("queue_type", _spec_get(spec, "network.queue_type", _spec_get(spec, "queue_type", "")))
    set_if_none_or_zero("end_us", _spec_get(spec, "sim.end_us", _spec_get(spec, "end_us", None)))

    set_if_none("progress", _spec_get(spec, "runner.progress", None))
    set_if_none("print_args", _spec_get(spec, "runner.print_args", _spec_get(spec, "print_args", None)))
    set_if_none_or_zero("heartbeat_s", _spec_get(spec, "runner.heartbeat_s", 0))
    set_if_none("stop_on_finished", _spec_get(spec, "runner.stop_on_finished", None))
    set_if_none("build", _spec_get(spec, "runner.build", None))
    set_if_none_or_empty("out_dir", _spec_get(spec, "runner.out_dir", ""))
    set_if_none_or_zero("paths", _spec_get(spec, "runner.paths", _spec_get(spec, "paths", None)))


def _clean_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    return env


def _run_with_heartbeat(
    cmd: list[str],
    *,
    cwd: Optional[Path],
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_s: int,
    progress: bool,
    stop_on_finished: bool,
    expected_flows: Optional[int],
) -> None:
    env = _clean_env()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    scan_pos = 0
    finished_cnt = 0
    requested_terminate = False
    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=out_f, stderr=err_f)
        last = 0.0
        while True:
            rc = p.poll()
            now = time.time()
            if rc is not None:
                break
            def _scan_finished_incremental() -> None:
                nonlocal scan_pos, finished_cnt
                try:
                    out_f.flush()
                    with stdout_path.open("rb") as rf:
                        rf.seek(scan_pos)
                        chunk = rf.read()
                    if chunk:
                        scan_pos += len(chunk)
                        finished_cnt += chunk.count(b" finished at ")
                except Exception:
                    pass

            # Scan stdout at most once per loop iteration, then reuse the updated
            # finished_cnt for both stop_on_finished and heartbeat display.
            need_scan = (stop_on_finished and expected_flows is not None and expected_flows > 0) or (
                progress and heartbeat_s > 0 and (now - last) >= heartbeat_s
            )
            if need_scan:
                _scan_finished_incremental()

            if stop_on_finished and expected_flows is not None and expected_flows > 0:
                if finished_cnt >= expected_flows:
                    if progress:
                        _eprint(f"[htsim_runner] all flows finished ({finished_cnt}/{expected_flows}); terminating early")
                    requested_terminate = True
                    p.terminate()
            if progress and heartbeat_s > 0 and (now - last) >= heartbeat_s:
                last = now
                try:
                    sz = stdout_path.stat().st_size
                except Exception:
                    sz = -1
                if expected_flows is not None and expected_flows > 0:
                    _eprint(
                        f"[htsim_runner] running... elapsed={now-start:.1f}s "
                        f"out_log_size={sz} bytes finished_flows={finished_cnt}/{expected_flows}"
                    )
                else:
                    _eprint(
                        f"[htsim_runner] running... elapsed={now-start:.1f}s "
                        f"out_log_size={sz} bytes finished_flows={finished_cnt}"
                    )
            time.sleep(0.2)
        if rc != 0:
            if requested_terminate and (rc == -15 or rc == 143):
                return
            raise subprocess.CalledProcessError(rc, cmd)


def _build_htsim_if_needed(build: bool) -> None:
    if not build:
        return
    if HTSIM_BIN.exists():
        return
    subprocess.run(["make", "-C", str(MAKE_DIR), "-j", str(os.cpu_count() or 8)], check=True, env=_clean_env())


def _parse_finished_times(out_path: Path) -> Tuple[int, float]:
    finished = 0
    makespan_us = 0.0
    for line in out_path.read_text(errors="ignore").splitlines():
        m = FINISHED_RE.search(line)
        if not m:
            continue
        finished += 1
        try:
            t = float(m.group(1))
            if t > makespan_us:
                makespan_us = t
        except Exception:
            pass
    return finished, makespan_us


def generate_leaf_spine_topo(
    topo_path: Path,
    *,
    nodes: int,
    link_gbps: int,
    tor_down: int,
    tor_up: int,
    spines: int,
    latency_ns: int,
    switch_latency_ns: int,
    oversub: int,
) -> Path:
    if nodes % tor_down != 0:
        raise ValueError(f"nodes({nodes}) must be multiple of tor_down({tor_down})")
    tors = nodes // tor_down
    if tor_up != spines:
        raise ValueError(f"full-mesh requires tor_up({tor_up}) == spines({spines})")
    if oversub <= 0:
        raise ValueError("oversub must be >= 1")
    if tor_down // tor_up != oversub:
        raise ValueError(f"oversub mismatch: tor_down/tor_up={tor_down}/{tor_up} != {oversub}")

    topo_path.parent.mkdir(parents=True, exist_ok=True)
    topo_path.write_text(
        "\n".join(
            [
                f"Nodes {nodes}",
                "Tiers 2",
                f"Podsize {nodes}",
                "",
                "# Auto-generated by /data/csg-htsim/htsim_runner.py",
                f"# Leaf(ToR): down={tor_down} up={tor_up} (total ports={tor_down + tor_up})",
                f"# Spines: {spines}, each connects to all {tors} ToRs (Tier1 Radix_Down={tors})",
                "",
                "Tier 0",
                f"Downlink_speed_Gbps {link_gbps}",
                f"Radix_Down {tor_down}",
                f"Radix_Up {tor_up}",
                f"Downlink_Latency_ns {latency_ns}",
                f"Switch_Latency_ns {switch_latency_ns}",
                f"Oversubscribed {oversub}",
                "",
                "Tier 1",
                f"Downlink_speed_Gbps {link_gbps}",
                f"Radix_Down {tors}",
                f"Downlink_Latency_ns {latency_ns}",
                f"Switch_Latency_ns {switch_latency_ns}",
                "",
            ]
        )
        + "\n"
    )
    return topo_path


def generate_fattree3_topo(
    topo_path: Path,
    *,
    nodes: int,
    podsize: int,
    link_gbps: int,
    tor_down: int,
    tor_up: int,
    tor_oversub: int,
    latency_ns: int,
    switch_latency_ns: int,
) -> Path:
    """
    Generate a 3-tier fat-tree config file consumable by FatTreeTopology::load().

    This is a "fat-tree style" 3-tier with explicit Podsize. For the common case
    where nodes = 2 * podsize (two pods), we set the core switch down-radix to
    number_of_pods so that pod-to-pod traffic has meaningful multi-path.
    """
    if podsize <= 0:
        raise ValueError("podsize must be > 0")
    if nodes % podsize != 0:
        raise ValueError(f"nodes({nodes}) must be a multiple of podsize({podsize})")
    if tor_down <= 0 or tor_up <= 0:
        raise ValueError("tor_down/tor_up must be > 0")
    if tor_oversub <= 0:
        raise ValueError("tor_oversub must be >= 1")
    if tor_down % tor_oversub != 0:
        raise ValueError(f"tor_down({tor_down}) must be divisible by tor_oversub({tor_oversub})")
    if tor_up != tor_down // tor_oversub:
        raise ValueError(f"tor_up({tor_up}) must equal tor_down/tor_oversub = {tor_down}//{tor_oversub} = {tor_down // tor_oversub}")
    if podsize % tor_down != 0:
        raise ValueError(f"podsize({podsize}) must be multiple of tor_down({tor_down})")
    pods = nodes // podsize
    tors_per_pod = podsize // tor_down
    if tors_per_pod <= 0:
        raise ValueError("invalid tors_per_pod")

    # 3-tier fat-tree with:
    # - ToR oversub only (tor_oversub at Tier0): tor_up = tor_down / tor_oversub
    # - Above ToR, keep 1:1 (Tier1 oversub=1): agg_down == agg_up == tors_per_pod
    # - Core down-radix equals number of pods; number of cores scales accordingly.
    agg_down = tors_per_pod
    agg_up = tors_per_pod
    core_down = pods

    topo_path.parent.mkdir(parents=True, exist_ok=True)
    topo_path.write_text(
        "\n".join(
            [
                f"Nodes {nodes}",
                "Tiers 3",
                f"Podsize {podsize}",
                "",
                "# Auto-generated by /data/csg-htsim/htsim_runner.py",
                f"# Pods={pods}, tors_per_pod={tors_per_pod}, tor_down={tor_down}, tor_up={tor_up}",
                f"# Agg: down={agg_down}, up={agg_up}; Core: down={core_down}",
                "",
                "Tier 0",
                f"Downlink_speed_Gbps {link_gbps}",
                f"Radix_Down {tor_down}",
                f"Radix_Up {tor_up}",
                f"Downlink_Latency_ns {latency_ns}",
                f"Switch_Latency_ns {switch_latency_ns}",
                f"Oversubscribed {tor_oversub}",
                "",
                "Tier 1",
                f"Downlink_speed_Gbps {link_gbps}",
                f"Radix_Down {agg_down}",
                f"Radix_Up {agg_up}",
                f"Downlink_Latency_ns {latency_ns}",
                f"Switch_Latency_ns {switch_latency_ns}",
                "Oversubscribed 1",
                "",
                "Tier 2",
                f"Downlink_speed_Gbps {link_gbps}",
                f"Radix_Down {core_down}",
                f"Downlink_Latency_ns {latency_ns}",
                f"Switch_Latency_ns {switch_latency_ns}",
                "",
            ]
        )
        + "\n"
    )
    return topo_path


@dataclass(frozen=True)
class RunResult:
    finished_flows: int
    expected_flows: Optional[int]
    makespan_us: float
    makespan_s: float
    tm_total_bytes: int
    metrics: Dict[str, float]
    out_dir: str
    out_log: str
    err_log: str
    queue_type: str


def run_htsim(
    out_dir: Path,
    *,
    topology: str,
    collective_spec: CollectiveSpec,
    nodes: int,
    servers: int,
    gpus_per_server: int,
    tp: int,
    cp: int,
    dp: int,
    ep: int,
    tensor_bytes: int,
    linkspeed_mbps: int,
    mtu: int,
    q_pkts: int,
    cwnd_pkts: int,
    end_us: int,
    seed: int,
    spines: int,
    servers_per_tor: int,
    ocs_topo_file: str,
    rail3_topo_file: str,
    ocs3_topo_file: str,
    ocs3_dump_reach_file: str,
    mixnet_eps_topo_file: str,
    mixnet_ocs_schedule_file: str,
    mixnet_ocs_planes: int,
    mixnet_ocs_link_speed_gbps: float,
    mixnet_ocs_queue_pkts: int,
    mixnet_ocs_no_drop: bool,
    mixnet_ocs_link_latency_ns: int,
    mixnet_ocs_switch_latency_ns: int,
    podsize: int,
    tor_down: int,
    tor_up: int,
    oversub: int,
    latency_ns: int,
    switch_latency_ns: int,
    hop_latency_us: Optional[float],
    switch_latency_us: Optional[float],
    paths: int,
    queue_type: str,
    progress: bool,
    print_args: bool,
    heartbeat_s: int,
    stop_on_finished: bool,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        _eprint("[htsim_runner] generating traffic matrix...")
    cm_path, expected_flows_int = generate_tm_from_spec(
        out_dir,
        spec=collective_spec,
        nodes=nodes,
        servers=servers,
        gpus_per_server=gpus_per_server,
        tp=tp,
        cp=cp,
        dp=dp,
        ep=ep,
        tensor_bytes=tensor_bytes,
        seed=seed,
    )
    expected_flows: Optional[int] = expected_flows_int

    tm_connections = _read_connections(cm_path)
    if tm_connections is None:
        _eprint(_red(f"[htsim_runner] WARNING: cannot read Connections from tm header: {cm_path}"))
    elif tm_connections != expected_flows_int:
        _eprint(_red(f"[htsim_runner] WARNING: expected_flows({expected_flows_int}) != tm Connections({tm_connections}) for {cm_path}"))
    else:
        _eprint(_green(f"[htsim_runner] tm check OK: expected_flows == Connections == {tm_connections}"))

    tm_total_bytes = _sum_tm_bytes(cm_path)

    topo_args: list[str] = []
    if topology == "rail":
        topo_args = ["-rail", str(servers), str(spines), str(gpus_per_server), str(servers_per_tor)]
    elif topology == "rail3":
        if not rail3_topo_file:
            raise ValueError("topology=rail3 requires --rail3-topo-file (Rail3 .topo file)")
        topo_args = ["-rail3_topo", rail3_topo_file]
    elif topology == "rail_ocs":
        if not ocs_topo_file:
            raise ValueError("topology=rail_ocs requires --ocs-topo-file")
        topo_args = ["-ocs_topo", ocs_topo_file]
    elif topology == "rail_ocs3":
        if not ocs3_topo_file:
            raise ValueError("topology=rail_ocs3 requires --ocs3-topo-file (RailOCS3 .topo file)")
        topo_args = ["-ocs3_topo", ocs3_topo_file]
        if ocs3_dump_reach_file:
            topo_args += ["-ocs3_dump_reach", ocs3_dump_reach_file]
    elif topology == "mixnet":
        if not mixnet_eps_topo_file:
            raise ValueError("topology=mixnet requires --mixnet-eps-topo-file")
        if not mixnet_ocs_schedule_file:
            raise ValueError("topology=mixnet requires --mixnet-ocs-schedule-file")
        if mixnet_ocs_planes <= 0:
            raise ValueError("topology=mixnet requires --mixnet-ocs-planes (>0)")

        # Generate EP-group map (rank -> group id) under the user's EP-group definition:
        # EP group = fixed (TP,CP,DP) and varying EP.
        #
        # group_id = flatten(tp,cp,dp) with CP as slowest-changing (arbitrary but consistent).
        coord_to_node, _node_to_server = _build_layout(
            nodes=nodes,
            servers=servers,
            gpus_per_server=gpus_per_server,
            tp=tp,
            cp=cp,
            dp=dp,
            ep=ep,
            placement_order=collective_spec.placement_order,
        )
        # invert coord_to_node -> node_to_coord
        node_to_coord: dict[int, tuple[int, int, int, int]] = {}
        for (tp_i, cp_i, dp_i, ep_i), nid in coord_to_node.items():
            node_to_coord[int(nid)] = (int(tp_i), int(cp_i), int(dp_i), int(ep_i))
        if len(node_to_coord) != nodes:
            raise ValueError("internal error: node_to_coord size mismatch")

        ep_group_map_path = out_dir / "ep_group.map"
        lines: list[str] = []
        for nid in range(nodes):
            tp_i, cp_i, dp_i, ep_i = node_to_coord[nid]
            gid = (cp_i * (tp * dp)) + (tp_i * dp) + dp_i
            # third column is local port index within EP group (ep_idx)
            lines.append(f"{nid} {gid} {ep_i}")
        ep_group_map_path.write_text("\n".join(lines) + "\n")

        mixnet_topo_path = out_dir / "mixnet.topo"
        mixnet_topo_path.write_text(
            "\n".join(
                [
                    "Type MixNet",
                    f"GpusPerServer {int(gpus_per_server)}",
                    "EpsGatewayLocal 0 1",
                    f"EpsTopoFile {mixnet_eps_topo_file}",
                    f"OcsScheduleFile {mixnet_ocs_schedule_file}",
                    f"OcsPlanes {int(mixnet_ocs_planes)}",
                    f"OcsLinkSpeedGbps {float(mixnet_ocs_link_speed_gbps)}",
                    f"OcsQueuePkts {int(mixnet_ocs_queue_pkts)}",
                    f"OcsNoDrop {1 if mixnet_ocs_no_drop else 0}",
                    f"OcsLinkLatencyNs {int(mixnet_ocs_link_latency_ns)}",
                    f"OcsSwitchLatencyNs {int(mixnet_ocs_switch_latency_ns)}",
                    f"EpGroupMapFile {str(ep_group_map_path)}",
                    "",
                ]
            )
        )
        topo_args = ["-mixnet_topo", str(mixnet_topo_path)]
    elif topology == "fattree":
        tor_up = tor_down // oversub
        topo_path = out_dir / f"leaf_spine_{nodes}n_{tor_down}d_{tor_up}u_os{oversub}_{linkspeed_mbps}mbps.topo"
        link_gbps = linkspeed_mbps // 1000
        if link_gbps * 1000 != linkspeed_mbps:
            raise ValueError("linkspeed_mbps must be multiple of 1000")
        generate_leaf_spine_topo(
            topo_path,
            nodes=nodes,
            link_gbps=link_gbps,
            tor_down=tor_down,
            tor_up=tor_up,
            spines=tor_up,
            latency_ns=latency_ns,
            switch_latency_ns=switch_latency_ns,
            oversub=oversub,
        )
        topo_args = ["-topo", str(topo_path), "-tiers", "2"]
    elif topology == "fattree3":
        # Real 3-tier fat-tree with explicit Podsize
        link_gbps = linkspeed_mbps // 1000
        if link_gbps * 1000 != linkspeed_mbps:
            raise ValueError("linkspeed_mbps must be multiple of 1000")
        if podsize <= 0:
            raise ValueError("topology=fattree3 requires explicit podsize; set topology.podsize in spec or pass --podsize.")
        # fattree3 supports ToR-only oversubscription:
        # - oversub == 1 => NB within a pod => tor_up == tor_down
        # - oversub > 1 => tor_up == tor_down/oversub
        if oversub <= 0:
            raise ValueError("fattree3 oversub must be >= 1")
        if tor_up <= 0:
            if tor_down % oversub != 0:
                raise ValueError(f"tor_down({tor_down}) must be divisible by oversub({oversub}) for fattree3")
            tor_up = tor_down // oversub
        else:
            if tor_up != tor_down // oversub:
                raise ValueError(f"fattree3 requires tor_up == tor_down/oversub, but tor_up={tor_up}, tor_down={tor_down}, oversub={oversub}")
        topo_path = out_dir / f"fattree3_{nodes}n_pod{podsize}_td{tor_down}_tu{tor_up}_nb_{linkspeed_mbps}mbps.topo"
        generate_fattree3_topo(
            topo_path,
            nodes=nodes,
            podsize=podsize,
            link_gbps=link_gbps,
            tor_down=tor_down,
            tor_up=tor_up,
            tor_oversub=oversub,
            latency_ns=latency_ns,
            switch_latency_ns=switch_latency_ns,
        )
        topo_args = ["-topo", str(topo_path)]
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    out_prefix = out_dir / "run"
    out_log = out_prefix.with_suffix(".out")
    err_log = out_prefix.with_suffix(".err")

    cmd = [
        str(HTSIM_BIN),
        *topo_args,
        "-nodes",
        str(nodes),
        "-tm",
        str(cm_path),
        "-o",
        str(out_prefix),
        "-linkspeed",
        str(linkspeed_mbps),
        "-mtu",
        str(mtu),
        "-q",
        str(q_pkts),
        "-cwnd",
        str(cwnd_pkts),
        "-queue_type",
        str(queue_type),
        "-strat",
        "ecmp",
        "-paths",
        str(paths),
        "-end",
        str(end_us),
    ]
    if hop_latency_us is not None:
        cmd += ["-hop_latency", str(hop_latency_us)]
    if switch_latency_us is not None:
        cmd += ["-switch_latency", str(switch_latency_us)]

    if progress or print_args:
        _eprint(f"[htsim_runner] running htsim_ndp... out_dir={out_dir}")
        _eprint(f"[htsim_runner] cmd: {' '.join(cmd)}")
    _run_with_heartbeat(
        cmd,
        cwd=None,
        stdout_path=out_log,
        stderr_path=err_log,
        heartbeat_s=heartbeat_s,
        progress=progress,
        stop_on_finished=stop_on_finished,
        expected_flows=expected_flows,
    )

    if progress:
        _eprint("[htsim_runner] parsing results...")
    finished, makespan_us = _parse_finished_times(out_log)
    makespan_s = makespan_us / 1e6 if makespan_us else 0.0

    metrics: Dict[str, float] = {}
    if makespan_s > 0:
        # Per-rank “effective” (legacy): tensor_bytes per rank / makespan
        metrics["effective_GBps"] = (tensor_bytes / makespan_s) / 1e9

        # Correct onwire throughput computed from tm.cm:
        # - total bytes injected into network across all ranks / makespan
        metrics["onwire_total_GBps"] = (tm_total_bytes / makespan_s) / 1e9
        metrics["onwire_total_Gbps"] = metrics["onwire_total_GBps"] * 8.0
        metrics["onwire_per_rank_GBps"] = metrics["onwire_total_GBps"] / float(nodes)
        # Backward-compatible key: per-rank onwire send GB/s
        metrics["onwire_send_GBps"] = metrics["onwire_per_rank_GBps"]

    return RunResult(
        finished_flows=finished,
        expected_flows=expected_flows,
        makespan_us=makespan_us,
        makespan_s=makespan_s,
        tm_total_bytes=tm_total_bytes,
        metrics=metrics,
        out_dir=str(out_dir),
        out_log=str(out_log),
        err_log=str(err_log),
        queue_type=str(queue_type),
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run htsim_ndp with a simple Python interface.")
    ap.add_argument("--spec", type=str, default="", help="Spec file (JSON/YAML). CLI args override spec fields.")
    ap.add_argument("--collective-type", choices=["alltoall", "allreduce", "allgather", "reducescatter", "p2p", "t2g"], default=None)
    ap.add_argument("--domain-dims", type=str, default=None)
    ap.add_argument("--placement-order", type=str, default=None)
    ap.add_argument("--exclude-intra-server", default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument(
        "--use-triggers",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="For collective=allreduce: generate step-by-step ring using triggers (barriers). Default: true for allreduce, false otherwise.",
    )
    ap.add_argument(
        "--allreduce-model",
        type=str,
        default=None,
        choices=["ring_steps", "ring_stream", "nccl_ring"],
        help="For collective=allreduce: choose TM model. ring_steps=unrolled steps (default). ring_stream=pipelined steady-state approximation. nccl_ring=NCCL-accurate multi-channel ring.",
    )
    ap.add_argument(
        "--alltoall-model",
        type=str,
        default=None,
        choices=["full_mesh", "pairwise_steps", "nccl_pairwise"],
        help="For collective=alltoall: full_mesh=all pairs at time 0. pairwise_steps=phased schedule with barriers. nccl_pairwise=NCCL-accurate (S-1)*G micro-rounds.",
    )
    ap.add_argument(
        "--alltoall-channels",
        type=int,
        default=None,
        help="For collective=alltoall and alltoall_model=pairwise_steps: max peers per phase per rank (>=1). Default: 1.",
    )
    ap.add_argument(
        "--alltoall-chunk-bytes",
        type=int,
        default=None,
        help="Optional for collective=alltoall: split each per-peer chunk into sub-chunks of this size (bytes). 0/None keeps default chunk=ceil(tensor_bytes/n).",
    )
    ap.add_argument(
        "--alltoall-chunk-inflight-per-peer",
        type=int,
        default=None,
        help=(
            "Optional for collective=alltoall with alltoall_model=pairwise_steps when chunking is enabled: "
            "cap the number of in-flight sub-chunks per (src,dst) peer pair. "
            "0/None keeps current behavior (all sub-chunks start together). Suggested: 2 or 4."
        ),
    )
    ap.add_argument(
        "--allgather-model",
        type=str,
        default=None,
        choices=["ring_steps", "nccl_ring", "hierarchical", "hierarchical_ring"],
        help="For collective=allgather: ring_steps (default), nccl_ring (NCCL-accurate multi-channel), hierarchical (NIC-focused inter-server full-mesh), or hierarchical_ring (inter-server ring with barriers).",
    )
    ap.add_argument(
        "--reducescatter-model",
        type=str,
        default=None,
        choices=["ring_steps", "nccl_ring", "hierarchical", "hierarchical_ring"],
        help="For collective=reducescatter: ring_steps (default), nccl_ring (NCCL-accurate multi-channel), hierarchical (NIC-focused inter-server full-mesh), or hierarchical_ring (inter-server ring with barriers).",
    )
    ap.add_argument(
        "--nchannels", type=int, default=None,
        help=(
            "Parallel ring channels for nccl_ring models (allreduce/allgather/reducescatter). "
            "Channel k rotates the ring start by k GPU positions within each server "
            "(anchor = GPU_k = NIC_k in rail topology). "
            "Each channel independently carries 1/nchannels of the data. "
            "Ignored by ring_steps (always single ring). "
            "Default 1. NCCL H800 rail typical: 8 (or 16 w/ DupChannels)."
        ),
    )
    ap.add_argument("--topology", default=None, choices=["rail", "rail3", "rail_ocs", "rail_ocs3", "mixnet", "fattree", "fattree3"])
    ap.add_argument("--p2p-src-index", type=int, default=None, help="For collective=p2p: src index within each domain group (default: 0)")
    ap.add_argument("--p2p-dst-index", type=int, default=None, help="For collective=p2p: dst index within each domain group (default: 1)")
    ap.add_argument("--p2p-direction", type=str, default=None, help="For collective=p2p: 0->1 | 1->0 | bidir (default: 0->1)")
    ap.add_argument(
        "--participant-ranks",
        type=str,
        default=None,
        help="Optional explicit participant ranks (comma-separated global ids), e.g. 0,8,16,24",
    )

    # t2g args (optional; spec is preferred)
    ap.add_argument("--t2g-train-pod", type=int, default=None, help="For collective=t2g: train pod id (default: 0)")
    ap.add_argument("--t2g-gen-pod", type=int, default=None, help="For collective=t2g: gen pod id (default: 1)")
    ap.add_argument("--t2g-train-gpus", type=int, default=None, help="For collective=t2g: active train GPU count within train pod (<=ep)")
    ap.add_argument("--t2g-gen-gpus", type=int, default=None, help="For collective=t2g: active gen GPU count within gen pod (<=ep)")
    ap.add_argument("--t2g-group-by", type=str, default=None, help="For collective=t2g: sender grouping: server|global (default: server)")
    ap.add_argument("--t2g-sender-group-size", type=int, default=None, help="For collective=t2g: k in k->1 (default: 8)")
    ap.add_argument("--t2g-dst-policy", type=str, default=None, help="For collective=t2g: dst policy: permute|round_robin (default: permute)")
    ap.add_argument("--t2g-unique-until-exhaust", default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument("--t2g-mode", type=str, default=None, help="For collective=t2g: grouped|pairwise_same_ep (default: grouped)")

    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--servers", type=int, default=None)
    ap.add_argument("--gpus-per-server", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--cp", type=int, default=None)
    ap.add_argument("--dp", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--tensor-bytes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--linkspeed-mbps", type=int, default=None)
    ap.add_argument("--mtu", type=int, default=None)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--cwnd", type=int, default=None)
    ap.add_argument("--end-us", type=int, default=None)
    ap.add_argument("--paths", type=int, default=None, help="Path entropy size for ECMP in htsim_ndp (-paths). If not set, inferred from topology.")
    ap.add_argument("--queue-type", type=str, default=None, help="htsim_ndp -queue_type (composite|composite_ecn|lossless|lossless_input). Default: composite.")

    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--build", default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument("--progress", default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument("--print-args", default=None, action=argparse.BooleanOptionalAction, help="Print merged/resolved args and htsim_ndp cmd to stderr before running.")
    ap.add_argument("--heartbeat-s", type=int, default=None)
    ap.add_argument("--stop-on-finished", default=None, action=argparse.BooleanOptionalAction)

    ap.add_argument("--spines", type=int, default=None)
    ap.add_argument("--servers-per-tor", type=int, default=None)
    ap.add_argument("--ocs-topo-file", type=str, default=None)
    ap.add_argument("--rail3-topo-file", type=str, default=None, help="For topology=rail3: Rail3 .topo file (packet-switched 3-tier baseline with rail host mapping).")
    ap.add_argument("--ocs3-topo-file", type=str, default=None, help="For topology=rail_ocs3: RailOCS3 .topo file (points to Agg/Core schedules).")
    ap.add_argument("--ocs3-dump-reach-file", type=str, default=None, help="For topology=rail_ocs3: dump reachability debug info to this file (optional).")
    ap.add_argument("--mixnet-eps-topo-file", type=str, default=None, help="For topology=mixnet: EPS fat-tree topo file (FatTreeTopology .topo).")
    ap.add_argument("--mixnet-ocs-schedule-file", type=str, default=None, help="For topology=mixnet: OCS static schedule file (slot0, plane matchings within EP group).")
    ap.add_argument("--mixnet-ocs-planes", type=int, default=None, help="For topology=mixnet: number of OCS planes in the schedule (>=1).")
    ap.add_argument("--mixnet-ocs-link-speed-gbps", type=float, default=None, help="For topology=mixnet: OCS circuit link speed (Gbps). 0/None => reuse -linkspeed.")
    ap.add_argument("--mixnet-ocs-queue-pkts", type=int, default=None, help="For topology=mixnet: OCS circuit queue size (pkts). 0/None => reuse -q.")
    ap.add_argument("--mixnet-ocs-no-drop", default=None, action=argparse.BooleanOptionalAction, help="For topology=mixnet: if set, OCS circuit queues never drop on overflow.")
    ap.add_argument("--mixnet-ocs-link-latency-ns", type=int, default=None, help="For topology=mixnet: OCS circuit link latency override (ns). 0/None => reuse hop latency.")
    ap.add_argument("--mixnet-ocs-switch-latency-ns", type=int, default=None, help="For topology=mixnet: OCS circuit switch latency override (ns). 0/None => reuse switch latency.")
    ap.add_argument("--podsize", type=int, default=None, help="For topology=fattree3: hosts per pod (Podsize in topo). Default: nodes/2 (two pods).")
    ap.add_argument("--tor-down", type=int, default=None)
    ap.add_argument("--tor-up", type=int, default=None, help="For topology=fattree3: ToR uplinks within a pod. Default: tor_down (NB).")
    ap.add_argument("--fattree-oversub", type=int, default=None)
    ap.add_argument("--latency-ns", type=int, default=None)
    ap.add_argument("--switch-latency-ns", type=int, default=None)
    ap.add_argument("--hop-latency-us", type=float, default=None)
    ap.add_argument("--switch-latency-us", type=float, default=None)

    args = ap.parse_args(argv)

    spec: Dict[str, Any] = {}
    if args.spec:
        spec = _load_spec_file(args.spec)
        _merge_spec_into_args(args, spec)

    # defaults
    if args.placement_order is None:
        args.placement_order = "TP,CP,DP,EP"
    if args.cp is None:
        args.cp = 1
    if args.dp is None:
        args.dp = 1
    if args.ep is None:
        args.ep = 1
    if args.seed is None:
        args.seed = 1
    if args.linkspeed_mbps is None:
        args.linkspeed_mbps = 400000
    if args.mtu is None:
        args.mtu = 9216
    if args.q is None:
        args.q = 64
    if args.cwnd is None:
        args.cwnd = max(8, args.linkspeed_mbps // 6250)
    if args.end_us is None:
        args.end_us = 20_000_000

    if args.queue_type is None or args.queue_type == "":
        # main_ndp default is COMPOSITE; be explicit for reproducibility + alignment.
        args.queue_type = "composite"
    if args.heartbeat_s is None:
        args.heartbeat_s = 5
    # Avoid guessing topology-critical defaults (can silently change experiment semantics).
    # Require explicit values for topologies that need them.
    if getattr(args, "podsize", None) is None:
        args.podsize = 0
    if args.tor_down is None:
        args.tor_down = 16
    if getattr(args, "tor_up", None) is None:
        args.tor_up = 0
    if args.fattree_oversub is None:
        args.fattree_oversub = 1

    # For topology=rail_ocs: allow inferring spines/servers_per_tor from spec (YAML is source of truth).
    if args.topology == "rail_ocs":
        if args.spines in (None, 0):
            inferred_planes = _spec_get(spec, "topology.rail_ocs.planes", _spec_get(spec, "topology.planes", 0))
            if inferred_planes:
                args.spines = int(inferred_planes)
        if args.servers_per_tor in (None, 0):
            inferred_spt = _spec_get(spec, "topology.rail_ocs.servers_per_tor", _spec_get(spec, "topology.servers_per_tor", 0))
            if inferred_spt:
                args.servers_per_tor = int(inferred_spt)
    # paths default (after spec merge + after topology-related defaults)
    if args.paths is None:
        if args.topology == "rail":
            # If user explicitly provided spines/servers_per_tor, infer paths from spines.
            if args.spines in (None, 0) or args.servers_per_tor in (None, 0):
                raise ValueError(
                    "topology=rail requires explicit spines and servers_per_tor; then paths defaults to spines. "
                    "Provide via spec(topology.spines/topology.servers_per_tor) or CLI (--spines/--servers-per-tor)."
                )
            args.paths = int(args.spines)
        elif args.topology == "rail_ocs":
            # If user explicitly provided spines (used as planes/path-count proxy), infer paths from spines.
            if args.spines in (None, 0):
                raise ValueError(
                    "topology=rail_ocs requires explicit spines (used as planes/path-count proxy); then paths defaults to spines. "
                    "Provide via spec(topology.spines) or CLI --spines."
                )
            args.paths = int(args.spines)
        elif args.topology == "rail_ocs3":
            # For RailOCS3, path count depends on the schedules and the src/dst pair.
            # Let htsim_ndp clamp ECMP to the actual number of routes; pick a large entropy size by default.
            args.paths = 2048
        elif args.topology == "fattree":
            tor_up = int(args.tor_down) // int(args.fattree_oversub)
            args.paths = int(tor_up)
        elif args.topology == "fattree3":
            raise ValueError("topology=fattree3 requires explicit --paths (or runner.paths in spec); no default inference to avoid silent misconfig.")
        else:
            # conservative fallback
            # For -strat ecmp (SCATTER_ECMP), HTSim will clamp to the actual number
            # of available paths: no_of_paths = min(_path_entropy_size, rt_list->size()).
            # Use a large value (same as main_ndp default) to avoid accidentally
            # collapsing ECMP to a single path when topology/pathcount is unknown.
            args.paths = 2048
    if args.latency_ns is None:
        args.latency_ns = 800
    if args.switch_latency_ns is None:
        args.switch_latency_ns = 800
    if args.exclude_intra_server is None:
        args.exclude_intra_server = False
    # Trigger model default (after spec merge): enable for allreduce unless explicitly disabled.
    if getattr(args, "use_triggers", None) is None:
        args.use_triggers = bool(args.collective_type == "allreduce")
    if args.progress is None:
        args.progress = False
    # Default: always print resolved args/cmd to stderr for easy verification.
    # Can be disabled with --no-print-args or runner.print_args: false in spec.
    if getattr(args, "print_args", None) is None:
        args.print_args = True
    if args.stop_on_finished is None:
        args.stop_on_finished = False
    if args.build is None:
        args.build = False

    # p2p defaults (after spec merge)
    if args.p2p_src_index is None:
        args.p2p_src_index = 0
    if args.p2p_dst_index is None:
        args.p2p_dst_index = 1
    if args.p2p_direction is None:
        args.p2p_direction = "0->1"
    if getattr(args, "participant_ranks", None) is None:
        args.participant_ranks = ""

    # t2g defaults (after spec merge)
    if getattr(args, "t2g_train_pod", None) is None:
        args.t2g_train_pod = 0
    if getattr(args, "t2g_gen_pod", None) is None:
        args.t2g_gen_pod = 1
    if getattr(args, "t2g_train_gpus", None) is None:
        args.t2g_train_gpus = 0
    if getattr(args, "t2g_gen_gpus", None) is None:
        args.t2g_gen_gpus = 0
    if getattr(args, "t2g_group_by", None) is None:
        args.t2g_group_by = "server"
    if getattr(args, "t2g_sender_group_size", None) is None:
        args.t2g_sender_group_size = 8
    if getattr(args, "t2g_dst_policy", None) is None:
        args.t2g_dst_policy = "permute"
    if getattr(args, "t2g_unique_until_exhaust", None) is None:
        args.t2g_unique_until_exhaust = True
    if getattr(args, "t2g_mode", None) is None:
        args.t2g_mode = "grouped"

    missing = []
    for k in ("collective_type", "domain_dims", "topology", "nodes", "gpus_per_server", "tp", "tensor_bytes"):
        if getattr(args, k) in (None, "", 0):
            missing.append(k)
    if missing:
        print(f"Error: missing required fields: {missing}. Provide via CLI or --spec.", file=sys.stderr)
        return 2

    if args.servers is None or args.servers <= 0:
        if args.nodes % args.gpus_per_server != 0:
            print("Error: cannot infer --servers: nodes must be multiple of gpus_per_server.", file=sys.stderr)
            return 2
        args.servers = args.nodes // args.gpus_per_server

    if args.nodes != args.tp * args.cp * args.dp * args.ep:
        print("Error: requires nodes == tp*cp*dp*ep.", file=sys.stderr)
        return 2

    # Topology-specific required args (avoid silent defaults)
    if args.topology == "rail":
        missing_topo = []
        if args.spines in (None, 0):
            missing_topo.append("spines")
        if args.servers_per_tor in (None, 0):
            missing_topo.append("servers_per_tor")
        if missing_topo:
            print(f"Error: topology=rail requires: {missing_topo}. Provide via spec(topology.*) or CLI.", file=sys.stderr)
            return 2
    if args.topology == "rail_ocs":
        if args.spines in (None, 0):
            print("Error: topology=rail_ocs requires spines (used as planes/path count). Provide via spec(topology.spines) or --spines.", file=sys.stderr)
            return 2
    if args.topology == "rail_ocs3":
        if not getattr(args, "ocs3_topo_file", None):
            print(
                "Error: topology=rail_ocs3 requires ocs3_topo_file. Provide via spec(topology.ocs3_topo_file) or --ocs3-topo-file.",
                file=sys.stderr,
            )
            return 2
    if args.topology == "rail3":
        if not getattr(args, "rail3_topo_file", None):
            print(
                "Error: topology=rail3 requires rail3_topo_file. Provide via spec(topology.rail3_topo_file) or --rail3-topo-file.",
                file=sys.stderr,
            )
            return 2
    if args.topology == "mixnet":
        missing_topo = []
        if not getattr(args, "mixnet_eps_topo_file", None):
            missing_topo.append("mixnet_eps_topo_file")
        if not getattr(args, "mixnet_ocs_schedule_file", None):
            missing_topo.append("mixnet_ocs_schedule_file")
        if getattr(args, "mixnet_ocs_planes", 0) in (None, 0):
            missing_topo.append("mixnet_ocs_planes")
        if missing_topo:
            print(f"Error: topology=mixnet requires: {missing_topo}. Provide via spec(topology.mixnet.*) or CLI.", file=sys.stderr)
            return 2
    if args.topology == "fattree3":
        if getattr(args, "podsize", 0) in (None, 0):
            print("Error: topology=fattree3 requires podsize. Provide via spec(topology.podsize) or --podsize.", file=sys.stderr)
            return 2

    collective_spec = CollectiveSpec(
        collective_type=str(args.collective_type),
        domain_dims=_parse_dim_list(args.domain_dims),
        placement_order=_parse_dim_list(args.placement_order),
        exclude_intra_server=bool(args.exclude_intra_server),
        use_triggers=bool(getattr(args, "use_triggers", False)),
        allreduce_model=str(getattr(args, "allreduce_model", "ring_steps") or "ring_steps"),
        alltoall_model=str(getattr(args, "alltoall_model", "pairwise_steps") or "pairwise_steps"),
        alltoall_channels=int(getattr(args, "alltoall_channels", 1) or 1),
        alltoall_chunk_bytes=int(getattr(args, "alltoall_chunk_bytes", 0) or 0),
        alltoall_chunk_inflight_per_peer=int(getattr(args, "alltoall_chunk_inflight_per_peer", 0) or 0),
        nchannels=int(getattr(args, "nchannels", None) or 1),
        allgather_model=str(getattr(args, "allgather_model", "ring_steps") or "ring_steps"),
        reducescatter_model=str(getattr(args, "reducescatter_model", "ring_steps") or "ring_steps"),
        p2p_src_index=int(args.p2p_src_index),
        p2p_dst_index=int(args.p2p_dst_index),
        p2p_direction=_parse_p2p_direction(args.p2p_direction),
        participant_ranks=_parse_rank_list(getattr(args, "participant_ranks", "")),
        t2g_train_pod=int(getattr(args, "t2g_train_pod", 0) or 0),
        t2g_gen_pod=int(getattr(args, "t2g_gen_pod", 1) or 1),
        t2g_train_gpus=int(getattr(args, "t2g_train_gpus", 0) or 0),
        t2g_gen_gpus=int(getattr(args, "t2g_gen_gpus", 0) or 0),
        t2g_group_by=str(getattr(args, "t2g_group_by", "server") or "server"),
        t2g_sender_group_size=int(getattr(args, "t2g_sender_group_size", 8) or 8),
        t2g_dst_policy=str(getattr(args, "t2g_dst_policy", "permute") or "permute"),
        t2g_unique_until_exhaust=bool(getattr(args, "t2g_unique_until_exhaust", True)),
        t2g_mode=str(getattr(args, "t2g_mode", "grouped") or "grouped"),
    )

    _build_htsim_if_needed(args.build)
    if not HTSIM_BIN.exists():
        print(f"Error: htsim_ndp not found at {HTSIM_BIN}.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path("/data/tmp") / f"htsim_runner_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # YAML-as-truth: auto-generate RailOCS topo/schedule into out_dir when enabled.
    if args.topology == "rail_ocs":
        rail_ocs_cfg = _spec_get(spec, "topology.rail_ocs", {})
        if not isinstance(rail_ocs_cfg, dict):
            rail_ocs_cfg = {}
        gen_enabled = _as_bool(rail_ocs_cfg.get("generate", False), default=False)
        if gen_enabled:
            planes = int(args.spines)
            servers_per_tor = int(args.servers_per_tor) if args.servers_per_tor not in (None, 0) else int(args.servers)

            # Derive rail ToR count (must match C++ logic in main_ndp.cpp).
            rail_splits = (int(args.servers) + (servers_per_tor - 1)) // servers_per_tor
            tors = int(args.gpus_per_server) * rail_splits

            # Schedule spec
            sched = rail_ocs_cfg.get("schedule", {})
            if not isinstance(sched, dict):
                sched = {}
            schedule_file = sched.get("file", _spec_get(spec, "topology.rail_ocs.schedule_file", ""))
            schedule_path: Path
            if schedule_file:
                schedule_path = Path(str(schedule_file)).expanduser()
                if not schedule_path.is_absolute():
                    # Interpret relative to repo root for convenience.
                    schedule_path = (REPO_ROOT / schedule_path).resolve()
                strict = _as_bool(sched.get("validate", True), default=True)
                if strict:
                    validate_rail_ocs_schedule_file(
                        schedule_path,
                        tors=tors,
                        planes=planes,
                        require_all_planes=_as_bool(sched.get("require_all_planes", True), default=True),
                        slot=_as_int(sched.get("slot", 0), default=0),
                    )
            else:
                pairs = _parse_pairs(sched.get("pairs", _spec_get(spec, "topology.rail_ocs.schedule_pairs", None)))
                repeat_across_planes = _as_bool(sched.get("repeat_across_planes", True), default=True)
                schedule_path = out_dir / "rail_ocs.schedule"
                generate_rail_ocs_schedule(
                    schedule_path,
                    tors=tors,
                    planes=planes,
                    pairs=pairs,
                    slot=_as_int(sched.get("slot", 0), default=0),
                    repeat_across_planes=repeat_across_planes,
                )

            # Topo params (defaults chosen to match existing hand-written topo files)
            link_speed_gbps = _as_float(rail_ocs_cfg.get("link_speed_gbps", 0.0), default=0.0)
            if link_speed_gbps <= 0:
                # fall back to CLI linkspeed_mbps
                link_speed_gbps = float(int(args.linkspeed_mbps)) / 1000.0

            topo_path = out_dir / "rail_ocs.topo"
            generate_rail_ocs_topo(
                topo_path,
                servers=int(args.servers),
                gpus_per_server=int(args.gpus_per_server),
                servers_per_tor=int(servers_per_tor),
                planes=int(planes),
                link_speed_gbps=float(link_speed_gbps),
                link_latency_ns=_as_int(rail_ocs_cfg.get("link_latency_ns", args.latency_ns), default=int(args.latency_ns)),
                switch_latency_ns=_as_int(rail_ocs_cfg.get("switch_latency_ns", args.switch_latency_ns), default=int(args.switch_latency_ns)),
                ocs_queue_pkts=_as_int(rail_ocs_cfg.get("ocs_queue_pkts", 1), default=1),
                ocs_no_drop=_as_bool(rail_ocs_cfg.get("ocs_no_drop", True), default=True),
                ocs_queue_type=str(rail_ocs_cfg.get("ocs_queue_type", "fifo") or "fifo"),
                ocs_cut_through=_as_bool(rail_ocs_cfg.get("ocs_cut_through", False), default=False),
                ocs_link_latency_ns=_as_int(rail_ocs_cfg.get("ocs_link_latency_ns", 50), default=50),
                ocs_switch_latency_ns=_as_int(rail_ocs_cfg.get("ocs_switch_latency_ns", 0), default=0),
                slot_us=_as_float(rail_ocs_cfg.get("slot_us", 5), default=5.0),
                reconfig_us=_as_float(rail_ocs_cfg.get("reconfig_us", 0), default=0.0),
                schedule_file=schedule_path.resolve(),
            )

            # Override args.ocs_topo_file to point to generated topo.
            if args.ocs_topo_file:
                _eprint(_red(f"[htsim_runner] WARNING: ignoring spec(topology.ocs_topo_file)={args.ocs_topo_file!r} because topology.rail_ocs.generate=true"))
            args.ocs_topo_file = str(topo_path.resolve())

    if args.print_args:
        # For printing only: fill in derived topology params to avoid confusing "0"/None defaults.
        topo_tor_up = getattr(args, "tor_up", None)
        if (topo_tor_up is None or topo_tor_up == 0) and args.topology == "fattree":
            try:
                topo_tor_up = int(args.tor_down) // int(args.fattree_oversub)
            except Exception:
                topo_tor_up = getattr(args, "tor_up", None)
        snap: Dict[str, Any] = {
            "spec": args.spec,
            "topology": args.topology,
            "job": {"nodes": args.nodes, "servers": args.servers, "gpus_per_server": args.gpus_per_server},
            "parallel": {"tp": args.tp, "cp": args.cp, "dp": args.dp, "ep": args.ep},
            "collective": {
                "type": collective_spec.collective_type,
                "domain_dims": list(collective_spec.domain_dims),
                "placement_order": list(collective_spec.placement_order),
                "exclude_intra_server": bool(collective_spec.exclude_intra_server),
                "participant_ranks": list(getattr(collective_spec, "participant_ranks", ())),
                "use_triggers": bool(collective_spec.use_triggers),
                "allreduce_model": getattr(collective_spec, "allreduce_model", ""),
                "alltoall_model": getattr(collective_spec, "alltoall_model", ""),
                "alltoall_channels": getattr(collective_spec, "alltoall_channels", 0),
                "alltoall_chunk_bytes": getattr(collective_spec, "alltoall_chunk_bytes", 0),
                "alltoall_chunk_inflight_per_peer": getattr(collective_spec, "alltoall_chunk_inflight_per_peer", 0),
                "allgather_model": getattr(collective_spec, "allgather_model", ""),
                "reducescatter_model": getattr(collective_spec, "reducescatter_model", ""),
                "p2p": {
                    "src_index": collective_spec.p2p_src_index,
                    "dst_index": collective_spec.p2p_dst_index,
                    "direction": collective_spec.p2p_direction,
                },
            },
            "topology_params": {
                "spines": args.spines,
                "servers_per_tor": args.servers_per_tor,
                "ocs_topo_file": args.ocs_topo_file,
                "rail3_topo_file": getattr(args, "rail3_topo_file", None),
                "ocs3_topo_file": getattr(args, "ocs3_topo_file", None),
                "ocs3_dump_reach_file": getattr(args, "ocs3_dump_reach_file", None),
                "mixnet_eps_topo_file": getattr(args, "mixnet_eps_topo_file", None),
                "mixnet_ocs_schedule_file": getattr(args, "mixnet_ocs_schedule_file", None),
                "mixnet_ocs_planes": getattr(args, "mixnet_ocs_planes", None),
                "mixnet_ocs_link_speed_gbps": getattr(args, "mixnet_ocs_link_speed_gbps", None),
                "mixnet_ocs_queue_pkts": getattr(args, "mixnet_ocs_queue_pkts", None),
                "mixnet_ocs_no_drop": getattr(args, "mixnet_ocs_no_drop", None),
                "mixnet_ocs_link_latency_ns": getattr(args, "mixnet_ocs_link_latency_ns", None),
                "mixnet_ocs_switch_latency_ns": getattr(args, "mixnet_ocs_switch_latency_ns", None),
                "podsize": getattr(args, "podsize", None),
                "tor_down": args.tor_down,
                "tor_up": topo_tor_up,
                "oversub": args.fattree_oversub,
                "latency_ns": args.latency_ns,
                "switch_latency_ns": args.switch_latency_ns,
            },
            "network": {"linkspeed_mbps": args.linkspeed_mbps, "mtu": args.mtu, "q": args.q, "cwnd": args.cwnd, "queue_type": args.queue_type},
            "sim": {"end_us": args.end_us, "seed": args.seed},
            "runner": {
                "out_dir": str(out_dir),
                "paths": args.paths,
                "build": args.build,
                "progress": args.progress,
                "heartbeat_s": args.heartbeat_s,
                "stop_on_finished": args.stop_on_finished,
            },
        }
        _eprint("[htsim_runner] resolved_args_json=" + json.dumps(snap, sort_keys=True))

    res = run_htsim(
        out_dir,
        topology=args.topology,
        collective_spec=collective_spec,
        nodes=args.nodes,
        servers=args.servers,
        gpus_per_server=args.gpus_per_server,
        tp=args.tp,
        cp=args.cp,
        dp=args.dp,
        ep=args.ep,
        tensor_bytes=args.tensor_bytes,
        linkspeed_mbps=args.linkspeed_mbps,
        mtu=args.mtu,
        q_pkts=args.q,
        cwnd_pkts=args.cwnd,
        end_us=args.end_us,
        seed=args.seed,
        spines=args.spines,
        servers_per_tor=args.servers_per_tor,
        ocs_topo_file=args.ocs_topo_file or "",
        rail3_topo_file=str(getattr(args, "rail3_topo_file", "") or ""),
        ocs3_topo_file=str(getattr(args, "ocs3_topo_file", "") or ""),
        ocs3_dump_reach_file=str(getattr(args, "ocs3_dump_reach_file", "") or ""),
        mixnet_eps_topo_file=str(getattr(args, "mixnet_eps_topo_file", "") or ""),
        mixnet_ocs_schedule_file=str(getattr(args, "mixnet_ocs_schedule_file", "") or ""),
        mixnet_ocs_planes=int(getattr(args, "mixnet_ocs_planes", 0) or 0),
        mixnet_ocs_link_speed_gbps=float(getattr(args, "mixnet_ocs_link_speed_gbps", 0.0) or 0.0),
        mixnet_ocs_queue_pkts=int(getattr(args, "mixnet_ocs_queue_pkts", 0) or 0),
        mixnet_ocs_no_drop=bool(getattr(args, "mixnet_ocs_no_drop", False)),
        mixnet_ocs_link_latency_ns=int(getattr(args, "mixnet_ocs_link_latency_ns", 0) or 0),
        mixnet_ocs_switch_latency_ns=int(getattr(args, "mixnet_ocs_switch_latency_ns", 0) or 0),
        podsize=int(args.podsize),
        tor_down=args.tor_down,
        tor_up=int(args.tor_up),
        oversub=args.fattree_oversub,
        latency_ns=args.latency_ns,
        switch_latency_ns=args.switch_latency_ns,
        hop_latency_us=args.hop_latency_us,
        switch_latency_us=args.switch_latency_us,
        paths=int(args.paths),
        queue_type=str(args.queue_type),
        progress=args.progress,
        print_args=bool(args.print_args),
        heartbeat_s=args.heartbeat_s,
        stop_on_finished=args.stop_on_finished,
    )

    payload: Dict[str, Any] = {
        "collective_type": collective_spec.collective_type,
        "domain_dims": list(collective_spec.domain_dims),
        "placement_order": list(collective_spec.placement_order),
        "exclude_intra_server": bool(collective_spec.exclude_intra_server),
        "participant_ranks": list(getattr(collective_spec, "participant_ranks", ())),
        "use_triggers": bool(collective_spec.use_triggers),
        "allreduce_model": getattr(collective_spec, "allreduce_model", ""),
        "alltoall_model": getattr(collective_spec, "alltoall_model", ""),
        "alltoall_channels": int(getattr(collective_spec, "alltoall_channels", 0) or 0),
        "alltoall_chunk_bytes": int(getattr(collective_spec, "alltoall_chunk_bytes", 0) or 0),
        "alltoall_chunk_inflight_per_peer": int(getattr(collective_spec, "alltoall_chunk_inflight_per_peer", 0) or 0),
        "allgather_model": getattr(collective_spec, "allgather_model", ""),
        "reducescatter_model": getattr(collective_spec, "reducescatter_model", ""),
        "p2p": {
            "src_index": int(getattr(collective_spec, "p2p_src_index", 0) or 0),
            "dst_index": int(getattr(collective_spec, "p2p_dst_index", 0) or 0),
            "direction": str(getattr(collective_spec, "p2p_direction", "") or ""),
        },
        "topology": args.topology,
        "nodes": args.nodes,
        "servers": args.servers,
        "gpus_per_server": args.gpus_per_server,
        "tp": args.tp,
        "cp": args.cp,
        "dp": args.dp,
        "ep": args.ep,
        "tensor_bytes": args.tensor_bytes,
        "linkspeed_mbps": args.linkspeed_mbps,
        "mtu_bytes": args.mtu,
        "q_pkts": args.q,
        "cwnd_pkts": args.cwnd,
        "queue_type": str(args.queue_type),
        "end_us": args.end_us,
        "tm_total_bytes": res.tm_total_bytes,
        "finished_flows": res.finished_flows,
        "expected_flows": res.expected_flows,
        "makespan_us": res.makespan_us,
        "makespan_s": res.makespan_s,
        "metrics": res.metrics,
        "out_dir": res.out_dir,
        "out_log": res.out_log,
        "err_log": res.err_log,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
