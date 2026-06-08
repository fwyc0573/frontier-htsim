from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from .intra_server_model import (
    _compute_g_in_group,
    _compute_n_group,
    estimate_intra_server_ms,
)
from .runner import run_htsim_runner
from .schema import Scenario


def predict_collective_time(
    scenario: Union[Scenario, Dict[str, Any]],
    *,
    repo_root: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    sc = scenario if isinstance(scenario, Scenario) else Scenario.from_dict(scenario)
    sc.validate()

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    payload = run_htsim_runner(repo_root=root, spec=sc.to_runner_spec(), out_dir=sc.runner.out_dir)

    raw_network_ms = float(payload.get("makespan_us", 0.0)) / 1000.0
    intra_ms, intra_assumptions = estimate_intra_server_ms(sc)
    network_ms = _effective_network_ms(
        raw_network_ms=raw_network_ms,
        scenario=sc,
    )
    combine_rule = _resolve_combine_rule(sc)
    combined_ms = _combine_time_ms(
        network_ms=network_ms,
        intra_ms=intra_ms,
        scenario=sc,
        combine_rule=combine_rule,
    )

    warnings: list[str] = []
    if (
        sc.intra_server.model in ("legacy_fabric", "ignore")
        and sc.collective.exclude_intra_server
        and raw_network_ms < 0.001
    ):
        g = _compute_g_in_group(sc)
        n = _compute_n_group(sc)
        if g > 1 and g == n:
            warnings.append(
                f"Domain group is fully intra-server (g={g}, n={n}) but "
                f"intra_server.model='{sc.intra_server.model}' returns 0ms. "
                f"Consider setting intra_server.model='nvlink_analytic' for "
                f"meaningful intra-server time estimation."
            )
    if raw_network_ms != network_ms:
        warnings.append(
            "Ignored surrogate network makespan for a fully intra-server collective "
            "group because exclude_intra_server=True and intra_server.model is active."
        )

    result: Dict[str, Any] = {
        "predicted_time_ms": combined_ms,
        "breakdown": {
            "network_ms": network_ms,
            "raw_network_ms": raw_network_ms,
            "intra_server_ms": intra_ms,
            "combined_ms": combined_ms,
            "combine_rule": combine_rule,
        },
        "assumptions": {**intra_assumptions, "combine_rule_source": "collective_algorithm"},
        "raw_runner_payload": payload,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _effective_network_ms(*, raw_network_ms: float, scenario: Scenario) -> float:
    if _should_ignore_surrogate_network_ms(scenario):
        return 0.0
    return raw_network_ms


def _should_ignore_surrogate_network_ms(scenario: Scenario) -> bool:
    if scenario.intra_server.model in ("legacy_fabric", "ignore"):
        return False
    if not scenario.collective.exclude_intra_server:
        return False
    g = _compute_g_in_group(scenario)
    n = _compute_n_group(scenario)
    return g > 1 and g == n


def _combine_time_ms(*, network_ms: float, intra_ms: float, scenario: Scenario, combine_rule: str) -> float:
    model = scenario.intra_server.model
    if model in ("legacy_fabric", "ignore"):
        return network_ms
    if combine_rule == "sum":
        return network_ms + intra_ms
    if combine_rule == "phase":
        return network_ms + intra_ms
    return max(network_ms, intra_ms)


def _resolve_combine_rule(scenario: Scenario) -> str:
    # Allow explicit override at collective level.
    if scenario.collective.intra_server_combine_rule in ("max", "sum", "phase"):
        return str(scenario.collective.intra_server_combine_rule)

    ctype = scenario.collective.kind
    # Hierarchical allgather/reducescatter are staged with inter-server NIC communication,
    # so sum is a safer default than max.
    if ctype == "allgather" and scenario.collective.allgather_model in ("hierarchical", "hierarchical_ring"):
        return "sum"
    if ctype == "reducescatter" and scenario.collective.reducescatter_model in ("hierarchical", "hierarchical_ring"):
        return "sum"
    # Default heuristic for non-hierarchical collectives: overlap-friendly max.
    return "max"
