"""Canonical rank-to-server layout computation.

This module is the single source of truth for mapping parallelism coordinates
to linear rank IDs and physical servers.  The same algorithm is independently
implemented in htsim_runner.py (_build_layout / _groups_for_domain) for the
C++ simulation path; if you change this module, verify htsim_runner.py stays
in sync.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Set, Tuple


def domain_group_server_counts(
    *,
    placement_order: Tuple[str, ...],
    domain_dims: Tuple[str, ...],
    dim_sizes: Dict[str, int],
    gpus_per_server: int,
) -> Dict[int, int]:
    """Count how many ranks from one domain group land on each server.

    We materialise a representative domain group by fixing all non-domain dims
    to 0 and enumerating all coordinates on domain dims, then map each
    coordinate to its linear rank id using *placement_order* (first dim =
    fastest-varying).
    """
    if not placement_order:
        return {}
    domain_set: Set[str] = set(domain_dims)
    domain_dims_ordered = [d for d in placement_order if d in domain_set]
    if not domain_dims_ordered:
        return {}

    # Stride with first dim as fastest-varying.
    strides: Dict[str, int] = {}
    stride = 1
    for dim in placement_order:
        strides[dim] = stride
        stride *= int(dim_sizes.get(dim, 1))

    base_coord: Dict[str, int] = {d: 0 for d in placement_order}
    counts: Dict[int, int] = {}
    gps = max(int(gpus_per_server), 1)

    for vals in product(*(range(int(dim_sizes[d])) for d in domain_dims_ordered)):
        coord = dict(base_coord)
        for d, v in zip(domain_dims_ordered, vals):
            coord[d] = int(v)
        linear = 0
        for dim in placement_order:
            linear += int(coord[dim]) * int(strides[dim])
        server = linear // gps
        counts[server] = counts.get(server, 0) + 1

    return counts


def compute_g_in_group(
    *,
    placement_order: Tuple[str, ...],
    domain_dims: Tuple[str, ...],
    dim_sizes: Dict[str, int],
    gpus_per_server: int,
) -> int:
    """GPUs per server participating in one domain group."""
    counts = domain_group_server_counts(
        placement_order=placement_order,
        domain_dims=domain_dims,
        dim_sizes=dim_sizes,
        gpus_per_server=gpus_per_server,
    )
    if not counts:
        return 1
    n = sum(counts.values())
    s = len(counts)
    return max(1, int(round(float(n) / float(s))))


def compute_n_group(
    *,
    placement_order: Tuple[str, ...],
    domain_dims: Tuple[str, ...],
    dim_sizes: Dict[str, int],
    gpus_per_server: int,
) -> int:
    """Total ranks in one domain group."""
    counts = domain_group_server_counts(
        placement_order=placement_order,
        domain_dims=domain_dims,
        dim_sizes=dim_sizes,
        gpus_per_server=gpus_per_server,
    )
    return max(1, sum(counts.values()))
