"""Zero, missing, negative and positive collective payloads at the real boundary.

An empty all-to-all is a legitimate collective. An expert-parallel all-to-all
whose local lane received no tokens in a step carries zero bytes and still pays
the synchronization latency of the fabric. It is the only kind with a defined
empty payload; the runner rejects zero for every other kind. These tests run the
real scenario serialization and the real runner, so a change that starts
treating zero as a missing field fails here.

The cases that need a prediction skip unless the CPU simulator has been built
(`cd sim && make -j"$(nproc)"`). The input-handling cases run without it.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SIMULATOR = REPO / "sim/datacenter/htsim_ndp"
sys.path.insert(0, str(REPO / "python"))

from collective_sim_core.predictor import predict_collective_time  # noqa: E402
from collective_sim_core.schema import Scenario  # noqa: E402


NVLINK_BW_GBPS = 450
NVLINK_LATENCY_US = 0.5
NVLINK_EFFICIENCY = 0.8
GPUS = 8


def scenario(payload, out_dir, kind="alltoall"):
    return Scenario.from_dict({
        "cluster": {"servers": 1, "gpus_per_server": GPUS},
        "parallelism": {"tp": 1, "cp": 1, "dp": 1, "ep": GPUS},
        "collective": {
            "kind": kind,
            "tensor_bytes": payload,
            "domain_dims": ["EP"],
            "placement_order": ["TP", "CP", "DP", "EP"],
            "participant_ranks": list(range(GPUS)),
            "exclude_intra_server": True,
            "alltoall_model": "pairwise_steps",
        },
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": NVLINK_BW_GBPS,
            "nvlink_latency_us": NVLINK_LATENCY_US,
            "nvlink_efficiency": NVLINK_EFFICIENCY,
        },
        "runner": {"out_dir": str(out_dir)},
    })


def run_runner(tmp_path, *, spec_payload, cli_payload=None, drop_field=False,
               raw_payload=None, kind="alltoall"):
    """Serialize a scenario and run the real runner over it.

    `raw_payload` edits the serialized file after the schema has accepted it,
    which is the only way to hand the runner a value the schema rejects.
    """

    spec = scenario(spec_payload, tmp_path, kind).to_runner_spec()
    if drop_field:
        del spec["collective"]["tensor_bytes"]
    elif raw_payload is not None:
        spec["collective"]["tensor_bytes"] = raw_payload
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(spec))
    command = [
        sys.executable, str(REPO / "htsim_runner.py"),
        "--spec", str(source),
        "--out-dir", str(tmp_path / "runner"),
    ]
    if cli_payload is not None:
        command.extend(["--tensor-bytes", str(cli_payload)])
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


needs_simulator = pytest.mark.skipif(
    not SIMULATOR.is_file(),
    reason="build the CPU simulator first: cd sim && make -j\"$(nproc)\"",
)


def test_a_missing_payload_is_still_an_error(tmp_path):
    result = run_runner(tmp_path, spec_payload=32768, drop_field=True)
    assert result.returncode == 2
    assert "missing required fields: ['tensor_bytes']" in result.stderr


def test_an_explicit_zero_is_not_a_missing_payload(tmp_path):
    """The distinction this change exists for.

    Before it, this case produced the identical error as the test above, so a
    caller could not tell a real empty transfer from a field it forgot to set.
    """

    result = run_runner(tmp_path, spec_payload=0)
    assert "missing required fields" not in result.stderr
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["allreduce", "allgather", "reducescatter", "p2p"])
def test_only_an_all_to_all_accepts_an_empty_payload(kind, tmp_path):
    """These kinds size their cross-server flows from the payload.

    A zero payload gives zero-size flows there, which htsim never finishes, so a
    cross-server all-reduce ran until it was killed. The runner refuses zero for
    these kinds before building any flow, on one server as well as across
    servers. The scenario here is one server, where each kind runs at a positive
    payload.
    """

    result = run_runner(tmp_path, spec_payload=0, kind=kind)
    assert result.returncode == 2
    assert f"tensor_bytes=0 is defined only for alltoall, not {kind}" in result.stderr


def test_the_runner_rejects_a_negative_payload(tmp_path):
    result = run_runner(tmp_path, spec_payload=32768, raw_payload=-1)
    assert result.returncode == 2
    assert "tensor_bytes must be >= 0" in result.stderr


def test_the_runner_rejects_a_negative_payload_from_the_command_line(tmp_path):
    result = run_runner(tmp_path, spec_payload=32768, cli_payload=-1)
    assert result.returncode == 2
    assert "tensor_bytes must be >= 0" in result.stderr


def test_the_scenario_rejects_a_negative_payload_when_it_is_serialized(tmp_path):
    with pytest.raises(ValueError, match=r"tensor_bytes must be >= 0"):
        scenario(-1, tmp_path).to_runner_spec()


def test_an_explicit_command_line_zero_overrides_the_scenario_file(tmp_path):
    result = run_runner(tmp_path, spec_payload=32768, cli_payload=0)
    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout.splitlines()[-1])
    assert emitted["tensor_bytes"] == 0


def test_the_scenario_file_still_supplies_an_unset_payload(tmp_path):
    result = run_runner(tmp_path, spec_payload=32768)
    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout.splitlines()[-1])
    assert emitted["tensor_bytes"] == 32768


@needs_simulator
@pytest.mark.parametrize("payload", [0, 32768])
def test_an_empty_transfer_keeps_its_synchronization_latency(payload, tmp_path):
    """Zero payload is not zero time.

    The intra-server term is `(gpus - 1)` pairwise steps of NVLink latency plus
    the transfer time of this rank's outgoing share. At zero bytes the transfer
    term vanishes and the latency term does not.
    """

    result = predict_collective_time(scenario(payload, tmp_path), repo_root=REPO)

    peers = GPUS - 1
    bytes_per_rank = peers / GPUS * payload
    transfer_us = bytes_per_rank / (NVLINK_BW_GBPS * 1e9 * NVLINK_EFFICIENCY) * 1e6
    expected_ms = (peers * NVLINK_LATENCY_US + transfer_us) / 1000

    assert result["predicted_time_ms"] == pytest.approx(expected_ms)
    assert result["breakdown"]["network_ms"] == 0
    assert result["assumptions"]["estimated_bytes_per_rank"] == pytest.approx(bytes_per_rank)
    if payload == 0:
        assert result["predicted_time_ms"] > 0


def two_server_scenario(payload, alltoall_model, out_dir):
    """An EP=16 group on two 8-GPU servers, so every cross-server pair is a network flow."""

    return Scenario.from_dict({
        "cluster": {"servers": 2, "gpus_per_server": GPUS},
        "parallelism": {"tp": 1, "cp": 1, "dp": 1, "ep": 2 * GPUS},
        "collective": {
            "kind": "alltoall",
            "tensor_bytes": payload,
            "domain_dims": ["EP"],
            "placement_order": ["TP", "CP", "DP", "EP"],
            "participant_ranks": list(range(2 * GPUS)),
            "alltoall_model": alltoall_model,
        },
        "runner": {"out_dir": str(out_dir)},
    })


@needs_simulator
@pytest.mark.parametrize("alltoall_model", ["pairwise_steps", "nccl_pairwise", "full_mesh"])
def test_an_empty_transfer_across_servers_synchronizes_every_pair(alltoall_model, tmp_path):
    """Each cross-server pair still exchanges one message when no token moves.

    Any payload of at most one byte per peer is priced as that message, so an
    empty all-to-all costs the same as a one-byte one, through the network.
    """

    empty = predict_collective_time(
        two_server_scenario(0, alltoall_model, tmp_path / "empty"), repo_root=REPO
    )
    one_byte = predict_collective_time(
        two_server_scenario(1, alltoall_model, tmp_path / "one_byte"), repo_root=REPO
    )

    flows = empty["raw_runner_payload"]
    assert flows["expected_flows"] > 0
    assert flows["finished_flows"] == flows["expected_flows"]
    assert empty["breakdown"]["network_ms"] > 0
    assert empty["predicted_time_ms"] == pytest.approx(one_byte["predicted_time_ms"])
