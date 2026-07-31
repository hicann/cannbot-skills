# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Mode 5 entry — orchestration that gates + spawns + validates aog-cann-learner.

Invocation:
    python3 -m cann_learn.mode5_runner --op 10_LayerNorm \
            --workspace workspace/10_layernorm \
            --module-path /data/cann_b103/cann-9.0.0/include/.../normalize \
            --kb-root src/skills/references \
            --api-catalog src/skills/references/target/ascendc/API_CATALOG.md

Steps (per v2 design):
  0. Hook preflight (verify G11/G12/SC10 enforced; refuse if not)
  1. Gate preconditions:
     - workspace/{op}/cann_strategy_inference.md exists (researcher ran)
     - state log shows researcher iter ≥ 1
     - arch22 -> arch35 migration op (fused or non-fused; without learning
       fused-op patterns the migration workflow
       cannot scale to FlashAttention / MoE / fused-norm class ops)
     - workspace/{op}/ref_runnable.json: verdict=RUNNABLE
     - CANN reference is BETTER than our kernel (per user 2026-05-05:
       "CANN should be learned only if it has better precision OR same
       precision but better performance")
  2. Acquire .cann_learn_active lease
  3. Spawn aog-cann-learner sub-agent (fresh context, restricted tools)
  4. Validate returned artifacts:
     - sealed/ exists, public summary.json schema-valid
     - re-run scanners independently against same files (cross-check)
     - apply policy: leak/compile/copy_shape FAIL → reject
                     KB-overlap → metadata-fix proposal not new entry
                     all clean → keep candidates with .kb_promotion_pending marker
                     (kb_manager auto-promote picks them up; no user gate)
  5. Release lease

The agent itself only writes to:
  - workspace/{op}/.cann_learn_sealed_<run_id>/  (private, sealed)
  - workspace/{op}/cann_learn_summary.json       (public, JSON-only)
  - patterns/unverified/candidates.md            (candidate KB entries)

This module is INVOKED BY /aog-knowledge-maintain Mode 5. It IS the gate logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from cann_learn import identifier_scanner, copy_shape, kb_overlap, summary_schema  # noqa: E402


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]


@dataclass
class Mode5Result:
    run_id: str
    op: str
    gate_passed: bool
    gate_reasons: list[str]
    summary_path: Optional[str] = None
    candidates_appended: int = 0
    metadata_fix_proposals: int = 0
    self_review_passed: bool = False
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Gate preconditions
# ---------------------------------------------------------------------------
def gate_check_preconditions(
    workspace: Path,
    op: str,
    *,
    skip_compare: bool = False,
    allow_finalized_without_researcher_iter: bool = False,
) -> GateResult:
    """Verify all preconditions for invoking aog-cann-learner.

    Per user 2026-05-05: CANN must have better precision OR same precision +
    better performance. Without that comparison, we'd be importing a
    potentially-worse strategy.
    """
    reasons: list[str] = []

    if not workspace.exists():
        reasons.append(f"workspace {workspace} does not exist")
        return GateResult(False, reasons)

    # Researcher actually ran with output
    csi = workspace / "cann_strategy_inference.md"
    if not csi.exists():
        reasons.append("cann_strategy_inference.md absent — researcher hasn't run")
    else:
        try:
            if csi.stat().st_size < 100:
                reasons.append("cann_strategy_inference.md too short (<100 bytes)")
        except OSError:
            reasons.append("cann_strategy_inference.md unreadable")

    # State log shows researcher iter ≥ 1
    log = workspace / "state_transitions.jsonl"
    if log.exists():
        researcher_count = 0
        try:
            for line in log.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("to_state") == "await_researcher":
                    researcher_count += 1
        except Exception:
            researcher_count = -1
        if researcher_count < 1:
            # P0aau-c35 followup (2026-05-09): when op is finalized AND
            # cann_strategy_inference.md has substantial content, the spirit of
            # "pipeline exhausted" is satisfied even if the V3.8.7 P0j routing
            # path took kw → finalize directly without explicit researcher
            # iter (which is common for PARTIAL_PERSIST scope-gap finalizes).
            # The carve-out's intent is "don't pre-empt the full pipeline";
            # a finalized + Phase-O5-verified op clearly didn't pre-empt.
            # Finalized signal: orchestrator_events.jsonl has
            # event=orchestrator.terminal with data.state=done
            events_log = workspace / "orchestrator_events.jsonl"
            finalized = False
            if events_log.exists():
                try:
                    for ev_line in events_log.read_text().splitlines():
                        if not ev_line.strip():
                            continue
                        ev = json.loads(ev_line)
                        if (ev.get("event") == "orchestrator.terminal"
                                and ev.get("data", {}).get("state") == "done"):
                            finalized = True
                            break
                except Exception:
                    finalized = False
            csi_substantial = csi.exists() and csi.stat().st_size >= 1000
            if allow_finalized_without_researcher_iter and finalized and csi_substantial:
                pass  # spirit satisfied; gate's proxy is over-strict here
            else:
                reasons.append(f"state_transitions.jsonl shows researcher iter < 1 ({researcher_count})")
    else:
        reasons.append("state_transitions.jsonl missing")

    # Reference runnable
    rr = workspace / "ref_runnable.json"
    if rr.exists():
        try:
            d = json.loads(rr.read_text())
            if d.get("verdict") != "RUNNABLE":
                reasons.append(f"ref_runnable.json verdict={d.get('verdict')!r} (need RUNNABLE)")
        except Exception:
            reasons.append("ref_runnable.json unparseable")
    else:
        reasons.append("ref_runnable.json missing — preflight not run for this op")

    # CANN-better-than-ours precondition
    if not skip_compare:
        verdict = workspace / "verification.json"
        if verdict.exists():
            try:
                vj = json.loads(verdict.read_text())
                perf = vj.get("performance", {})
                ratio = perf.get("ratio")
                if ratio is None:
                    reasons.append(
                        "verification.json.performance.ratio missing — cannot establish "
                        "CANN-vs-ours comparison; skipping carve-out"
                    )
                elif isinstance(ratio, (int, float)) and ratio >= 1.0:
                    reasons.append(
                        f"our kernel ratio={ratio} ≥ 1.0× CANN; learning from CANN "
                        "would import a worse strategy"
                    )
            except Exception:
                reasons.append("verification.json unparseable")
        else:
            reasons.append("verification.json missing — cannot establish CANN-vs-ours comparison")

    return GateResult(passed=len(reasons) == 0, reasons=reasons)


# ---------------------------------------------------------------------------
# Lease management (G11)
# ---------------------------------------------------------------------------
def acquire_lease(workspace: Path, run_id: str) -> Path:
    lease = workspace / ".cann_learn_active"
    lease.write_text(json.dumps({
        "run_id": run_id,
        "acquired_at": time.time(),
    }))
    return lease


def release_lease(workspace: Path) -> None:
    lease = workspace / ".cann_learn_active"
    if lease.exists():
        lease.unlink()


# ---------------------------------------------------------------------------
# Hook preflight
# ---------------------------------------------------------------------------
def hook_preflight() -> GateResult:
    """Run preflight_install_hooks.py --check + verify G11/G12/SC10 active.

    Returns GateResult — passed=False if hooks not installed or stale.
    """
    reasons: list[str] = []

    # This file lives under src/scripts/cann_learn. Walking up four parent
    # levels reaches cann_learn, scripts, src, and finally the repository root.
    repo_root = _HERE.parents[3]
    preflight = repo_root / "src" / "scripts" / "preflight_install_hooks.py"
    if not preflight.exists():
        reasons.append(f"preflight_install_hooks.py missing at {preflight}")
        return GateResult(False, reasons)

    try:
        result = subprocess.run(
            ["python3", str(preflight), "--check"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        reasons.append(f"preflight invocation failed: {e}")
        return GateResult(False, reasons)

    verdict = (result.stdout or "").split()
    verdict_word = verdict[0] if verdict else ""

    if verdict_word == "STAMP_OK":
        pass  # hooks installed + up to date
    elif verdict_word == "STAMP_DRIFT_ONLY":
        # Auto-heal — let user know but don't block
        pass
    elif verdict_word == "CONTENT_DRIFT":
        reasons.append(
            "Hook content drift: workflow_critic settings stale; re-install + restart "
            "Claude Code, then re-invoke Mode 5"
        )
    else:
        reasons.append(f"preflight returned unknown verdict: {verdict_word!r}")

    return GateResult(passed=len(reasons) == 0, reasons=reasons)


# ---------------------------------------------------------------------------
# Independent re-validation (post-agent)
# ---------------------------------------------------------------------------
# Mode 6 (build_system) — relaxed copy_shape threshold per main agent
# review 2026-05-21T16:48Z msg DISCORD_ID_REDACTED Q3: "C34c copy_shape (5%)
# will likely drop all candidates because CMake boilerplate
# (cmake_minimum_required, project(), common target_link_libraries) trivially
# hits >5%. False-positive-everything = wasted $5-10."
# Empirically validated threshold subject to first-run adjustment.
_COPY_SHAPE_THRESHOLD_BY_MODE = {
    "kernel_structural": 0.05,
    "build_system": 0.30,
}


def revalidate_post_agent(
    workspace: Path,
    summary_path: Path,
    *,
    cann_files_read: list[Path],
    candidate_paths: list[Path],
    api_catalog_path: Path,
    extraction_mode: str = "kernel_structural",
) -> tuple[bool, list[str]]:
    """Independently run scanners after agent return; cross-check vs agent's
    self_review claims. Agent could lie about own scores; skill caller can re-scan.

    extraction_mode: "kernel_structural" (Mode 5, default) | "build_system" (Mode 6).
    Determines the copy_shape threshold used for C34c re-scan.
    """
    failures: list[str] = []

    # Schema check
    schema_res = summary_schema.validate_file(summary_path)
    if not schema_res.valid:
        failures.extend(f"schema:{e}" for e in schema_res.errors)
        return False, failures

    summary = json.loads(summary_path.read_text())

    # C34a: re-run identifier scan
    id_res = identifier_scanner.scan(
        cann_files_read=cann_files_read,
        candidate_output_paths=candidate_paths,
        api_catalog_path=api_catalog_path,
    )
    if not id_res.passed:
        failures.append(
            f"C34a leak detected by re-scan: {id_res.leak_count} leaks; "
            f"agent's self-report: {summary.get('checks', {}).get('C34a', {}).get('passed')}"
        )

    # C34c: re-run copy-shape per candidate — threshold depends on extraction_mode
    cs_threshold = _COPY_SHAPE_THRESHOLD_BY_MODE.get(extraction_mode, 0.05)
    dropped_for_visibility: list[dict] = []  # log dropped candidates per main Q3
    for cand in candidate_paths:
        try:
            text = cand.read_text(errors="replace")
            cs_res = copy_shape.check(
                cand.name, text, cann_files_read,
                threshold=cs_threshold,
            )
            if not cs_res.passed:
                failures.append(
                    f"C34c copy-shape detected for {cand.name}: "
                    f"score={cs_res.score:.3f} threshold={cs_threshold:.3f} "
                    f"(extraction_mode={extraction_mode})"
                )
                dropped_for_visibility.append({
                    "candidate": cand.name,
                    "score": round(cs_res.score, 4),
                    "threshold": cs_threshold,
                    "extraction_mode": extraction_mode,
                    "reason": "C34c_copy_shape_exceeded_threshold",
                })
        except Exception as e:
            failures.append(f"C34c re-scan errored: {e}")

    # Self-review verdict mismatch
    if summary.get("self_review_verdict") == "PASS" and failures:
        failures.append("agent self-review = PASS but independent re-scan found failures")

    # Mode 6 (build_system) — surface dropped candidates to Mode 2 reviewer per main
    # agent Q3 directive: "log every DROPPED candidate to summary.json (don't silently
    # filter — give Mode 2 reviewer the visibility)". Write side-file rather than
    # mutate summary.json (which is schema-pinned).
    if dropped_for_visibility:
        side_file = workspace / "cann_learn_dropped_candidates.json"
        side_file.write_text(json.dumps({
            "extraction_mode": extraction_mode,
            "copy_shape_threshold": cs_threshold,
            "dropped": dropped_for_visibility,
        }, indent=2))

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Sealed dir lifecycle
# ---------------------------------------------------------------------------
def setup_sealed_dir(workspace: Path, run_id: str) -> Path:
    sealed = workspace / f".cann_learn_sealed_{run_id}"
    sealed.mkdir(mode=0o700, exist_ok=True)
    return sealed


def archive_sealed_dir(workspace: Path, run_id: str) -> Optional[Path]:
    """Move sealed dir to repo-wide archive (cold storage). Returns archive path."""
    sealed = workspace / f".cann_learn_sealed_{run_id}"
    if not sealed.exists():
        return None
    repo_root = _HERE.parent.parent.parent
    archive_root = repo_root / "archive" / "sealed_cann_learn"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{run_id}.tar.gz"
    # Tar + gzip
    subprocess.run(
        ["tar", "czf", str(archive_path), "-C", str(sealed.parent), sealed.name],
        check=True,
    )
    shutil.rmtree(sealed)
    return archive_path


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------
def run_mode5(
    *,
    op: str,
    workspace: Path,
    module_path: Path,
    kb_root: Path,
    api_catalog_path: Path,
    skip_hook_preflight: bool = False,
    skip_compare: bool = False,
    allow_finalized_without_researcher_iter: bool = False,
    spawn_agent_func: Optional[callable] = None,
    extraction_mode: str = "kernel_structural",
) -> Mode5Result:
    """Run Mode 5 end-to-end.

    spawn_agent_func: callable(brief) → (sealed_files: list[Path], summary_path: Path,
        candidate_paths: list[Path], cann_files_read: list[Path]).
        Pluggable for testing (unit tests use a mock; production uses the real
        agent_dispatch invocation).

    extraction_mode: "kernel_structural" (default, Mode 5 historical behavior) or
        "build_system" (Mode 6, 2026-05-21 — extracts CMakeLists.txt + register_*.cpp
        + op_proto*.cpp + apt.cpp build-system patterns, separate KB destination).
    """
    run_id = uuid.uuid4().hex[:12]

    # Step 0: hook preflight
    if not skip_hook_preflight:
        hp = hook_preflight()
        if not hp.passed:
            return Mode5Result(
                run_id=run_id, op=op,
                gate_passed=False,
                gate_reasons=hp.reasons,
                failure_reason="hook_preflight_failed",
            )

    # Step 1: gate preconditions
    gate = gate_check_preconditions(
        workspace, op,
        skip_compare=skip_compare,
        allow_finalized_without_researcher_iter=allow_finalized_without_researcher_iter,
    )
    if not gate.passed:
        return Mode5Result(
            run_id=run_id, op=op,
            gate_passed=False,
            gate_reasons=gate.reasons,
            failure_reason="gate_preconditions_failed",
        )

    # Step 2: acquire lease (G11 hook now blocks all writes to sensitive paths)
    _lease = acquire_lease(workspace, run_id)

    try:
        # Step 3: spawn agent (or mock for tests)
        if spawn_agent_func is None:
            return Mode5Result(
                run_id=run_id, op=op,
                gate_passed=True, gate_reasons=[],
                failure_reason="spawn_agent_func not provided — Mode 5 is a no-op skeleton "
                               "until aog-cann-learner agent is wired into orchestrator",
            )

        sealed_dir = setup_sealed_dir(workspace, run_id)
        try:
            # Pass extraction_mode if the spawn function supports it. Graceful
            # fallback for legacy callers (Mode 5 test mocks) that don't accept
            # the kwarg — they continue to behave as kernel_structural.
            spawn_kwargs = dict(
                op=op,
                workspace=workspace,
                module_path=module_path,
                sealed_dir=sealed_dir,
                run_id=run_id,
                kb_root=kb_root,
                api_catalog_path=api_catalog_path,
            )
            try:
                spawn_result = spawn_agent_func(
                    extraction_mode=extraction_mode,
                    **spawn_kwargs,
                )
            except TypeError as e:
                if "extraction_mode" not in str(e):
                    raise
                # Legacy mock without extraction_mode kwarg — call without it.
                if extraction_mode != "kernel_structural":
                    raise RuntimeError(
                        f"extraction_mode={extraction_mode!r} requested but "
                        f"spawn_agent_func={spawn_agent_func!r} does not accept "
                        f"the kwarg; cannot run Mode 6"
                    ) from e
                spawn_result = spawn_agent_func(**spawn_kwargs)
        except Exception as e:
            return Mode5Result(
                run_id=run_id, op=op,
                gate_passed=True, gate_reasons=[],
                failure_reason=f"agent spawn failed: {e}",
            )

        _sealed_files = spawn_result.get("sealed_files", [])
        summary_path = spawn_result.get("summary_path")
        candidate_paths = spawn_result.get("candidate_paths", [])
        cann_files_read = spawn_result.get("cann_files_read", [])

        # Step 4a: validate schema + re-run scanners
        if summary_path is None or not Path(summary_path).exists():
            return Mode5Result(
                run_id=run_id, op=op,
                gate_passed=True, gate_reasons=[],
                failure_reason="agent did not produce summary.json",
            )

        valid, failures = revalidate_post_agent(
            workspace,
            Path(summary_path),
            cann_files_read=[Path(p) for p in cann_files_read],
            candidate_paths=[Path(p) for p in candidate_paths],
            api_catalog_path=api_catalog_path,
            extraction_mode=extraction_mode,
        )
        if not valid:
            return Mode5Result(
                run_id=run_id, op=op,
                gate_passed=True, gate_reasons=[],
                summary_path=str(summary_path),
                self_review_passed=False,
                failure_reason="re-validation failed: " + " | ".join(failures),
            )

        # Step 4b: archive sealed (cold storage), keep summary + candidates
        archive_sealed_dir(workspace, run_id)

        # Step 4c: write .kb_promotion_pending markers for each new candidate.
        # P0acl 2026-05-10: renamed from .kb_review_required (which implied
        # user-review-as-gate, violating 0-interaction product design per
        # self-critic C40). The new marker signals "ready for kb_manager
        # auto-promote pipeline" — kb_manager picks these up, then applies
        # the C36-C39 generalization, deduplication, conflict, and
        # transferability gates plus the Codex review hook, then promotes (or
        # BLOCKs with reason). NO user sign-off
        # required.
        kb_review_dir = kb_root / "patterns" / "unverified"
        kb_review_dir.mkdir(parents=True, exist_ok=True)
        for cand in candidate_paths:
            cand_basename = Path(cand).stem
            marker = kb_review_dir / f".kb_promotion_pending-{run_id}-{cand_basename}"
            marker.write_text(json.dumps({
                "run_id": run_id,
                "op": op,
                "candidate_id": cand_basename,
                "ts": time.time(),
                "next_action": "kb_manager auto-promote pipeline (C36-C39 gates + codex hook)",
            }))

        return Mode5Result(
            run_id=run_id, op=op,
            gate_passed=True, gate_reasons=[],
            summary_path=str(summary_path),
            candidates_appended=len(candidate_paths),
            metadata_fix_proposals=spawn_result.get("metadata_fix_proposals_count", 0),
            self_review_passed=True,
        )

    finally:
        # Step 5: release lease
        release_lease(workspace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--module-path", required=True, type=Path)
    ap.add_argument("--kb-root", required=True, type=Path)
    ap.add_argument("--api-catalog", required=True, type=Path)
    ap.add_argument("--skip-hook-preflight", action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument(
        "--allow-finalized-without-researcher-iter",
        action="store_true",
        help="P0aau-c35 followup: allow mode 5 on a finalized op (state log "
             "shows 'done') with substantial cann_strategy_inference.md even "
             "if state log shows researcher iter < 1. Spirit of carve-out is "
             "'pipeline exhausted'; a finalized op clearly satisfies that.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the live agent spawn step (returns failure_reason "
             "indicating skeleton). Default behavior wires production "
             "spawn_cann_learner_agent — set this flag for unit-test or "
             "gate-only runs.",
    )
    ap.add_argument(
        "--extraction-mode",
        choices=["kernel_structural", "build_system"],
        default="kernel_structural",
        help="kernel_structural (default, Mode 5 historical) — read 2-5 "
             "kernel files; extract algorithm-structural patterns; "
             "candidates → patterns/unverified/candidates.md. "
             "build_system (Mode 6, 2026-05-21) — read CMakeLists.txt + "
             "register_*.cpp + op_proto*.cpp + apt.cpp; extract per-source-"
             "file flag isolation + register glue + launch macro routing; "
             "candidates → target/ascendc/build_system/candidates.md.",
    )
    args = ap.parse_args()

    # Wire production agent spawn unless --dry-run.
    spawn_func = None
    if not args.dry_run:
        from cann_learn.agent_spawn import spawn_cann_learner_agent
        spawn_func = spawn_cann_learner_agent

    result = run_mode5(
        op=args.op,
        workspace=args.workspace,
        module_path=args.module_path,
        kb_root=args.kb_root,
        api_catalog_path=args.api_catalog,
        skip_hook_preflight=args.skip_hook_preflight,
        skip_compare=args.skip_compare,
        allow_finalized_without_researcher_iter=args.allow_finalized_without_researcher_iter,
        spawn_agent_func=spawn_func,
        extraction_mode=args.extraction_mode,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0 if (result.gate_passed and result.self_review_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
