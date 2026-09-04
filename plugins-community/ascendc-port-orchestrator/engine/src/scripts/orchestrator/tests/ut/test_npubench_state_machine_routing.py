# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression coverage for provider-owned NPUBench O4 handoff routing.

An NPUBench worker is intentionally forbidden from writing precision or
performance evidence.  Its ``done`` handoff means only that the candidate build
is ready; Phase O5 in ``finalize`` must invoke the provider-owned evaluator.
Legacy providers must retain the existing missing-precision escalation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable test path setup)
import pytest

sys.path.insert(0, str(_reorg_paths.WORKFLOW_DIR))
sys.path.insert(0, str(_reorg_paths.ORCH_DIR))

import state_machine as sm  # noqa: E402
from state_machine import _extract_handoff_from_progress  # noqa: E402
import reference_source  # noqa: E402
import state_executor  # noqa: E402


def _npubench_binding() -> dict[str, object]:
    """Minimal complete binding accepted by reference_source's registry."""
    return {
        "schema_version": 3,
        "source": "npubench",
        "semantic_binding": "npubench_old_format_task_bundle",
        "runner_contract_version": "npubench/v1",
        "bundle_manifest_path": "reference_inputs/npubench/digest/bundle_manifest.json",
        "bundle_manifest_sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
        "task_relative_path": "level1/3_Add.py",
        "task_sha256": "c" * 64,
        "sidecar_relative_path": "level1/3_Add.json",
        "sidecar_sha256": "d" * 64,
        "sidecar_encoding": "json",
    }


def _write_workspace(workspace: Path, reference: dict[str, object]) -> None:
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "port_a3_to_a5", "reference": reference}),
        encoding="utf-8",
    )
    # Deliberately omit precision/performance: those fields are worker-owned
    # for legacy routes but must be absent before NPUBench O5 runs.
    (workspace / "verification.json").write_text(
        json.dumps({"build_evidence": {"compiled_provenance": {"source": "workspace"}}}),
        encoding="utf-8",
    )


def _extract_handoff(raw: str) -> str:
    """Load the production handoff parser only when a test exercises it."""
    import handoff_audit

    return handoff_audit.extract_canonical_handoff(raw)


def _is_npubench_transition(transition) -> bool:
    """Exit transition whose rationale names the NPUBench contract."""
    return "NPUBench contract" in transition.get("rationale", "")


def _is_precision_probe_transition(transition) -> bool:
    """await_probe transition gated on the precision-status clause."""
    if transition.get("goto") != "await_probe":
        return False
    for clause in transition.get("condition", {}).get("all_of", []):
        if isinstance(clause, dict) and "verification_precision_status_not_in" in clause:
            return True
    return False


def _first_transition_index(transitions, predicate) -> int:
    """Index of the first exit transition accepted by ``predicate``."""
    for index, transition in enumerate(transitions):
        if predicate(transition):
            return index
    raise AssertionError("no exit transition matched the expected shape")


def test_npubench_build_ready_done_goes_directly_to_finalize(tmp_path):
    workspace = tmp_path / "npubench-op"
    _write_workspace(workspace, _npubench_binding())

    result = sm.next_state(
        workspace,
        "await_worker",
        "→ orchestrator: done — candidate build is ready for NPUKernelBench harness evaluation",
    )

    assert result["next_state"] == "finalize"
    transitions = sm.get_state_spec(sm.load_state_machine(), "await_worker")[
        "exit_transitions"
    ]
    npubench_index = _first_transition_index(transitions, _is_npubench_transition)
    precision_probe_index = _first_transition_index(
        transitions, _is_precision_probe_transition
    )
    assert result["matched_transition_index"] == npubench_index
    assert npubench_index < precision_probe_index
    assert state_executor.next_agent(result["next_state"]) == "aog-finalize-pipeline"
    assert state_executor.next_agent(result["next_state"]) != "aog-precision-probe"
    assert "NPUBench" in result["rationale"]


def test_npubench_at_orchestrator_build_ready_goes_directly_to_finalize(tmp_path):
    """The worker's @orchestrator build-ready form must not pause the run."""
    workspace = tmp_path / "npubench-at-build-ready-op"
    _write_workspace(workspace, _npubench_binding())

    extracted = _extract_handoff(
        "@orchestrator: build-ready candidate produced; static check PASS; continue to A5 target runner"
    )
    assert extracted.startswith("→ orchestrator: build-ready")
    result = sm.next_state(
        workspace,
        "await_worker",
        extracted,
    )

    assert result["next_state"] == "finalize"
    assert "build-ready" in result["rationale"]


def test_npubench_build_ready_with_nested_aog_text_stays_build_ready(tmp_path):
    """A descriptive @aog mention must not replace the provider handoff."""
    workspace = tmp_path / "npubench-build-ready-with-aog-op"
    _write_workspace(workspace, _npubench_binding())

    extracted = _extract_handoff(
        "→ orchestrator: build-ready — static check PASS; if a later build fails, "
        "escalate to @aog-kernel-optimizer"
    )
    result = sm.next_state(workspace, "await_worker", extracted)

    assert extracted.startswith("→ orchestrator: build-ready")
    assert result["next_state"] == "finalize"


def test_progress_restore_recognizes_at_orchestrator_build_ready(tmp_path):
    workspace = tmp_path / "progress-restore-op"
    workspace.mkdir()
    (workspace / "PROGRESS.md").write_text(
        "### EXIT\n"
        "@orchestrator: build-ready candidate produced; static check PASS\n",
        encoding="utf-8",
    )

    extracted = _extract_handoff_from_progress(workspace)

    assert extracted is not None
    assert extracted.startswith("→ orchestrator: build-ready")


@pytest.mark.parametrize(
    "raw_handoff",
    [
        "@orchestrator:build-ready candidate produced; static check PASS",
        "@orchestrator:   build-ready candidate produced; static check PASS",
        "→ orchestrator:build-ready candidate produced; static check PASS",
        "→ orchestrator:   build-ready candidate produced; static check PASS",
    ],
)
def test_progress_restore_build_ready_bootstraps_to_finalize(tmp_path, raw_handoff):
    workspace = tmp_path / "progress-restore-npubench-op"
    _write_workspace(workspace, _npubench_binding())
    (workspace / "PROGRESS.md").write_text(
        "### EXIT\n"
        f"{raw_handoff}\n",
        encoding="utf-8",
    )

    current = sm.get_current_state(workspace, sm.load_state_machine())

    assert current == "finalize"
    assert sm.read_log(workspace)[-1]["to_state"] == "finalize"


def test_partial_npubench_build_ready_pauses_on_direct_route(tmp_path):
    workspace = tmp_path / "partial-npubench-build-ready-op"
    _write_workspace(workspace, {"source": "npubench"})

    extracted = _extract_handoff(
        "@orchestrator: build-ready candidate produced; static check PASS"
    )
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"
    assert "binding" in result["rationale"].lower()


@pytest.mark.parametrize(
    "raw_handoff",
    [
        "@orchestrator:    build-ready candidate produced; static check PASS",
        "→ orchestrator:   build-ready candidate produced; static check PASS",
    ],
)
def test_partial_npubench_build_ready_spacing_still_pauses(tmp_path, raw_handoff):
    workspace = tmp_path / "partial-npubench-build-ready-spacing-op"
    _write_workspace(workspace, {"source": "npubench"})

    extracted = _extract_handoff(raw_handoff)
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"
    assert "binding" in result["rationale"].lower()


def test_partial_npubench_build_ready_bootstrap_persists_pause(tmp_path):
    workspace = tmp_path / "partial-npubench-build-ready-restore-op"
    _write_workspace(workspace, {"source": "npubench"})
    (workspace / "PROGRESS.md").write_text(
        "### EXIT\n"
        "@orchestrator: build-ready candidate produced; static check PASS\n",
        encoding="utf-8",
    )

    current = sm.get_current_state(workspace, sm.load_state_machine())

    assert current == "await_user_decision"
    log = sm.read_log(workspace)
    assert log[-1]["to_state"] == "await_user_decision"


def test_build_ready_unexpected_condition_error_is_not_paused(tmp_path, monkeypatch):
    workspace = tmp_path / "npubench-build-ready-unexpected-error-op"
    _write_workspace(workspace, _npubench_binding())

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("condition evaluator bug")

    monkeypatch.setattr(sm, "eval_condition", raise_unexpected)
    with pytest.raises(RuntimeError, match="condition evaluator bug"):
        sm.next_state(
            workspace,
            "await_worker",
            "→ orchestrator: build-ready candidate produced",
        )


def test_partial_npubench_done_binding_error_pauses(tmp_path):
    """Provider binding failures on done are fail-closed, not process crashes."""
    workspace = tmp_path / "partial-npubench-done-binding-error-op"
    _write_workspace(workspace, {"source": "npubench"})

    extracted = _extract_handoff("@orchestrator: done")
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"
    assert "binding" in result["rationale"].lower()


@pytest.mark.parametrize(
    "raw_handoff",
    [
        "@orchestrator: build-ready-v2 candidate produced",
        "→ orchestrator: build-readyx candidate produced",
        "@orchestrator: done-typo",
        "→ orchestrator: done_v2",
    ],
)
def test_npubench_unknown_handoff_suffix_does_not_finalize(tmp_path, raw_handoff):
    """Known tokens require a boundary; malformed suffixes stay diagnostic."""
    workspace = tmp_path / "npubench-unknown-handoff-suffix-op"
    _write_workspace(workspace, _npubench_binding())

    extracted = _extract_handoff(raw_handoff)
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"
    assert result["next_state"] != "finalize"


@pytest.mark.parametrize(
    "raw_handoff",
    [
        "@orchestrator: await_user_decision",
        "@orchestrator:   await_user_decision please inspect",
        "→ orchestrator:await_user_decision",
        "→ orchestrator:   await_user_decision please inspect",
    ],
)
def test_explicit_user_decision_uses_parser_then_state_machine(tmp_path, raw_handoff):
    """Explicit pause requests remain pauses through the production parser."""
    workspace = tmp_path / "npubench-explicit-user-decision-op"
    _write_workspace(workspace, {"source": "npubench"})

    extracted = _extract_handoff(raw_handoff)
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"
    assert result["next_state"] != "finalize"


def test_legacy_build_ready_after_parser_normalization_still_pauses(tmp_path):
    """Only NPUBench owns automatic build-ready dispatch."""
    workspace = tmp_path / "legacy-at-build-ready-op"
    _write_workspace(workspace, reference_source.explicit_a3_live_binding())

    extracted = _extract_handoff(
        "@orchestrator: build-ready candidate produced; static check PASS"
    )
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"


def test_legacy_build_stuck_after_parser_normalization_keeps_abort_route(tmp_path):
    """The dual-form diagnostic fix must not change build-stuck semantics."""
    workspace = tmp_path / "legacy-build-stuck-op"
    _write_workspace(workspace, reference_source.explicit_a3_live_binding())

    extracted = _extract_handoff(
        "@orchestrator: build stuck — compiler signature is unchanged"
    )
    result = sm.next_state(workspace, "await_worker", extracted)

    assert extracted.startswith("→ orchestrator: build stuck")
    assert result["next_state"] == "abort"


@pytest.mark.parametrize(
    "raw_handoff",
    [
        "@orchestrator: infra unreachable — target container is unavailable",
        "@orchestrator: BLOCKED — target runner capacity is unavailable",
    ],
)
def test_diagnostic_handoffs_after_parser_normalization_pause(tmp_path, raw_handoff):
    workspace = tmp_path / "legacy-diagnostic-op"
    _write_workspace(workspace, reference_source.explicit_a3_live_binding())

    extracted = _extract_handoff(raw_handoff)
    result = sm.next_state(workspace, "await_worker", extracted)

    assert extracted.startswith("→ orchestrator:")
    assert result["next_state"] == "await_user_decision"


def test_partial_npubench_diagnostic_catchall_does_not_raise(tmp_path):
    """Generic diagnostics must not invoke the provider binding resolver."""
    workspace = tmp_path / "partial-npubench-diagnostic-op"
    _write_workspace(workspace, {"source": "npubench"})

    extracted = _extract_handoff(
        "@orchestrator: unknown diagnostic; inspect the preserved workspace"
    )
    result = sm.next_state(workspace, "await_worker", extracted)

    assert result["next_state"] == "await_user_decision"


def test_legacy_done_without_precision_still_escalates_to_probe(tmp_path):
    workspace = tmp_path / "legacy-op"
    _write_workspace(workspace, reference_source.explicit_a3_live_binding())

    result = sm.next_state(workspace, "await_worker", "→ orchestrator: done")

    assert result["next_state"] == "await_probe"
    assert result["matched_transition_index"] != 0


def test_reference_source_condition_fails_closed_for_partial_binding(tmp_path):
    workspace = tmp_path / "malformed-op"
    _write_workspace(workspace, {"source": "npubench"})
    ctx = {
        "handoff": "→ orchestrator: done",
        "snapshot": sm.snapshot(workspace),
        "iter_counts": {},
        "ws": workspace,
        "sm": sm.load_state_machine(),
    }

    with pytest.raises(reference_source.ReferenceSourceError):
        sm.eval_condition({"reference_source_is": "npubench"}, ctx)


def test_reference_source_condition_keeps_legacy_fallback_for_valid_non_npubench(
    tmp_path,
):
    workspace = tmp_path / "legacy-op"
    _write_workspace(workspace, reference_source.explicit_a3_live_binding())
    ctx = {
        "handoff": "→ orchestrator: done",
        "snapshot": sm.snapshot(workspace),
        "iter_counts": {},
        "ws": workspace,
        "sm": sm.load_state_machine(),
    }

    assert sm.eval_condition({"reference_source_is": "npubench"}, ctx) is False


def test_reference_source_condition_does_not_downgrade_tilelang_route_on_bad_binding(
    tmp_path,
):
    workspace = tmp_path / "tilelang2ascendc-op"
    _write_workspace(workspace, {"source": "npubench"})
    state_path = workspace / ".opgen_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["port_source"] = {"kind": "port-aclnn-tilelang2ascendc"}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    ctx = {
        "handoff": "→ orchestrator: done",
        "snapshot": sm.snapshot(workspace),
        "iter_counts": {},
        "ws": workspace,
        "sm": sm.load_state_machine(),
    }

    with pytest.raises(reference_source.ReferenceSourceError):
        sm.eval_condition({"reference_source_is": "npubench"}, ctx)


def test_reference_source_condition_requires_workspace_evidence():
    with pytest.raises(RuntimeError, match="requires a workspace"):
        sm.eval_condition(
            {"reference_source_is": "npubench"},
            {"snapshot": {}, "iter_counts": {}, "ws": None, "sm": {}},
        )
