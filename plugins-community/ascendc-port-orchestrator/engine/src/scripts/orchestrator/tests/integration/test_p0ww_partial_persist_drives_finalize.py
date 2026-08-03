# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Drive legitimate pipeline exhaustion through finalize_pipeline.finalize_op.

Caught 2026-05-06 via 9_topktopp run: orchestrator hit iter_cap on
await_optimizer with full pipeline evidence (V3.8.8 PARTIAL_PERSIST),
called _record_partial_persist_finalize (which writes a state-log
entry transitioning to `finalize` and tags
verification.json.persist_verdict=PARTIAL_PERSIST), then `return 0`
exited the run. The next loop iteration (which would re-snapshot,
see current_state=finalize, and run finalize_pipeline.finalize_op)
never happened. Result: workspace tagged PARTIAL_PERSIST but never
promoted to canonical archive — no .finalized marker, no done state.

Fix: replace `return 0` with `last_handoff = "..." ; continue` so
the loop drives one more iter through the finalize handler.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

import orchestrator  # noqa: E402


def _seed_workspace_at_iter_cap_optimizer(ws: Path):
    """Seed a workspace at the optimizer iteration cap.

    The fixture represents a workspace where:
      - Pipeline ran fully: probe with verdict=requirement,
        researcher with cann_strategy_inference.md, optimizer at iter_cap
      - Worker emitted PARTIAL precision (legitimate pipeline-exhaustion)
    """
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": ws.name,
        "opgen_mode": "backward",
    }))
    # State log shows we got to await_optimizer
    log_lines = [
        json.dumps({
            "from_state": "init", "to_state": "await_worker",
            "ts": "t1", "handoff": "", "matched_transition_index": 0,
            "rationale": "", "iter_counts_snapshot": {},
        }),
        json.dumps({
            "from_state": "await_worker", "to_state": "await_probe",
            "ts": "t2", "handoff": "@aog-precision-probe",
            "matched_transition_index": 0, "rationale": "",
            "iter_counts_snapshot": {"worker": 1},
        }),
        json.dumps({
            "from_state": "await_probe", "to_state": "await_researcher",
            "ts": "t3", "handoff": "verdict=requirement",
            "matched_transition_index": 0, "rationale": "",
            "iter_counts_snapshot": {"worker": 1, "probe": 1},
        }),
        json.dumps({
            "from_state": "await_researcher", "to_state": "await_optimizer",
            "ts": "t4", "handoff": "cann_strategy_inference.md ready",
            "matched_transition_index": 0, "rationale": "",
            "iter_counts_snapshot": {"worker": 1, "probe": 1, "researcher": 1},
        }),
    ]
    # Add 5 optimizer iters → hit iter_cap=5
    for i in range(5):
        log_lines.append(json.dumps({
            "from_state": "await_optimizer", "to_state": "await_optimizer",
            "ts": f"t{5+i}", "handoff": "→ orchestrator: done — KO_PERF_PLATEAU",
            "matched_transition_index": 4, "rationale": "keep iterating within budget",
            "iter_counts_snapshot": {
                "worker": 1, "probe": 1, "researcher": 1, "optimizer": i + 1,
            },
        }))
    (ws / "state_transitions.jsonl").write_text("\n".join(log_lines) + "\n")

    # Pipeline evidence
    (ws / "probe_report.md").write_text("# probe report\n" + "x" * 200)
    (ws / "probe_result.json").write_text(json.dumps({
        "verdict": "requirement",
        "classification": "requirement",
    }))
    (ws / "cann_strategy_inference.md").write_text("# strategy\n" + "x" * 200)
    (ws / "optimization_log.md").write_text("# optimization log\n" + "x" * 200)

    # Verification with PARTIAL precision + perf below threshold (P0aa-eligible)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PARTIAL",
            "pass_a": {"status": "PARTIAL", "tier1_pass": 46, "total": 50},
        },
        "performance": {"status": "PASS", "ratio": 0.34},
        "determinism": {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
    }))

    # PROGRESS.md with self-introspection (P0qq prereq for terminal handoff)
    (ws / "PROGRESS.md").write_text(
        "# 9_topktopp\n\n## Self-introspection (kw-1)\n\n"
        "### Pressure modes I felt\nP1.\n\n"
        "### Decisions I almost rationalized\nnone\n\n"
        "### Verifications I might have skipped\nnone\n\n"
        "### Confidence calibration\nprecision: HIGH\nperf: HIGH\narchitectural fit: HIGH\n"
    )

    # Knowledge update (finalize gate requires)
    (ws / "knowledge_update.md").write_text(
        "## Context\nStub.\n\n## Findings\n- Stub.\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\n- STUB-0\n\n## Anti-patterns avoided\nNone\n"
    )

    # Phase O2.5 artifacts present
    (ws / "input_gen.py").write_text("# stub")
    (ws / "edge_inputs.pt").write_bytes(b"\x80\x02tensor")
    (ws / "manifest.json").write_text(json.dumps({"op": "x"}))
    (ws / "ref_runnable.json").write_text(json.dumps({"verdict": "RUNNABLE"}))
    (ws / "edge_dataset.pt").write_bytes(b"\x80\x02ds")

    # PB-33 (2026-05-14): op_host/ completeness — finalize gate requires
    # complete PR4778 mirror. Test fixture seeds the minimum 3 files
    # (_def.cpp, _tiling.cpp, _tiling.h) so the OP_HOST_COMPLETENESS gate
    # is satisfied. Real workers populate these via kw_brief Phase B.4.
    op_host = ws / "op_host"
    op_host.mkdir(exist_ok=True)
    (op_host / "test_op_def.cpp").write_text("// stub op_def\n")
    (op_host / "test_op_tiling.cpp").write_text("// stub tiling impl\n")
    (op_host / "test_op_tiling.h").write_text("// stub tiling header\n")

    # .ascendc_env at workspace parent
    (ws.parent / ".ascendc_env").write_text(
        "TARGET=a5\nA5_HOST=test\nA5_USER=root\nA5_PASSWORD=t\n"
        "A5_CONTAINER=t\nCANN_PATH=/test\nSOC_VERSION=Ascend950PR_9579\n"
    )


def test_iter_cap_p0aa_drives_finalize_pipeline(tmp_path, monkeypatch):
    """Run finalize_pipeline.finalize_op after legitimate iter-cap exhaustion.

    The orchestrator must continue the loop instead of returning immediately.
    """
    ws = tmp_path / "test_op"
    _seed_workspace_at_iter_cap_optimizer(ws)

    # Track that finalize_pipeline.finalize_op was actually invoked
    import finalize_pipeline as fp
    finalize_called = [False]
    real_finalize_op = fp.finalize_op

    def wrapped_finalize_op(op, workspace, **kwargs):
        finalize_called[0] = True
        return real_finalize_op(op, workspace, **kwargs)
    monkeypatch.setattr(fp, "finalize_op", wrapped_finalize_op)

    # Skip O5 post-verify (no real SSH)
    import phase_o5_runner
    # P0aba (2026-05-07) RUNNER_FAILED now blocks finalize, so return a
    # VERIFIED measurement that matches the PARTIAL claim.
    monkeypatch.setattr(phase_o5_runner, "ssh_runner",
                        lambda ws, op, *args, **kwargs: phase_o5_runner.MeasuredResult(
                            pass_a={"tier1_pass": 46, "total": 50}))
    import importlib as _il
    monkeypatch.setattr(_il, "reload", lambda mod: mod)

    # Skip critic to avoid spawning subprocesses
    import critic_invoke
    import kb_invoke
    monkeypatch.setattr(critic_invoke, "should_fire_iter_cap_warning",
                        lambda ws, st: False)
    monkeypatch.setattr(critic_invoke, "fire_critic", lambda ws, trigger: None)
    monkeypatch.setattr(
        kb_invoke,
        "merge_one",
        lambda ws: {"success": True, "merged": "integration-test stub"},
    )

    exit_code = orchestrator.run_single_op(
        "test_op", workspace=ws, lane=0, plan_only=False,
    )

    # Critical assertion: finalize_pipeline.finalize_op MUST have been called
    assert finalize_called[0], (
        "REGRESSION: orchestrator hit iter_cap with full-pipeline evidence "
        "but did NOT invoke finalize_pipeline.finalize_op. The op would be "
        "tagged PARTIAL_PERSIST but never archive-promoted (P0ww)."
    )

    # And the .finalized-* marker should exist (proof of full pipeline run)
    finalized_markers = list(ws.glob(".finalized-*"))
    assert finalized_markers, (
        f"No .finalized-* marker in workspace after PARTIAL_PERSIST + finalize "
        f"(P0ww). Workspace contents: {sorted(p.name for p in ws.iterdir())}"
    )

    # Exit code should be 0 (terminal `done` reached cleanly)
    assert exit_code == 0
