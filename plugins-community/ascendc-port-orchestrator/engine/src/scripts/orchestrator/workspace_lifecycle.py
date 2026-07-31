# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Workspace lifecycle — agent-died marker, stale-output archival, re-entry reset, partial-persist
finalize record.

Mechanically extracted from orchestrator.py (god-file decomposition 2026-06-30, per
ORCHESTRATOR_REFACTOR_AND_UT_SPEC §1). Behavior unchanged — function bodies are VERBATIM.
Re-imported into orchestrator's namespace so existing call-sites + `orchestrator.<name>` external
access are preserved.

DAG: imports only stdlib + the `logging_config` sibling (no orchestrator import) — acyclic, imports
standalone without pulling orchestrator. The _STALE_OUTPUTS_BY_STATE constant is MOVED here (only
consumer is _archive_stale_outputs_before_spawn) and re-imported into orchestrator.

Not moved (kept in core, see decompose_log.md):
- _cold_start_reset_workspace: carries a [baselined] GOD_FUNCTION (217 lines) — relocating trips the
  arch-lint changed-file ratchet (baseline-regen out of scope).
- _sync_benchmark_jsonl_to_workspace: reads PROJECT_ROOT, which tests monkeypatch on the
  `orchestrator` module (test_p0abf_phase_o2_jsonl_sync). Re-deriving PROJECT_ROOT here would make the
  function read this module's copy and the orch-level monkeypatch would silently no-op (2 UT failures
  observed). Importing PROJECT_ROOT from orchestrator would create a cycle. So it stays in core.

MONKEYPATCH NOTE (durable — OL-160-class latent-coupling guard): the functions and module-level
constants/logger here are re-imported into orchestrator's namespace, which preserves
`orchestrator.<name>` attribute LOOKUP only — it does NOT rebind THIS module's own globals. A test
that overrides a symbol one of these functions reads must `monkeypatch.setattr(<this_module>,
'<name>', ...)` on THIS module, NOT on `orchestrator` (patching orchestrator silently misses the
binding used here). No current test patches these on orchestrator; this note prevents a future one
from a silent no-op.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)


def _mark_agent_died(workspace: Path, state: str, reason: str) -> None:
    """Codex C4: agent crash → mark workspace, surface to resume.py."""
    marker = workspace / f".agent_died_at_{state}"
    marker.write_text(json.dumps({
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": state,
        "reason": reason,
    }, indent=2))


# P0v: state-specific stale outputs that the next agent MAY write and that
# the state machine evaluates via path_exists in exit_transitions. If a
# prior session left these files behind, the path_exists check false-matches
# and routes incorrectly. Archive them BEFORE spawn so only fresh writes count.
_STALE_OUTPUTS_BY_STATE: dict[str, tuple[str, ...]] = {
    "await_researcher": ("optimization_directive.md", "research_report.md"),
    "await_optimizer": ("optimization_directive.md",),
    "await_fused_optimizer": ("optimization_directive.md",),
    "await_probe": ("probe_report.md", "probe_result.json"),
}


def _archive_stale_outputs_before_spawn(
    workspace: Path, state: str, spawn_index: int
) -> None:
    """P0v (2026-05-05): rename stale outputs to .pre-{state}-{spawn_idx}-* so
    the upcoming agent has a clean slate. The state machine's path_exists
    checks then evaluate against this spawn's writes only.

    History: op#9 TopKTopP 2026-05-05 had a stale optimization_directive.md
    from prior pp-3/ko-1 sessions. State machine matched path_exists in
    await_researcher.exit_transitions → await_worker, falsely treating it
    as ar-2's directive. Worker correctly diagnosed false-match, exited
    with prose summary → abort.
    """
    stale_paths = _STALE_OUTPUTS_BY_STATE.get(state, ())
    # --optimize re-entry (.optimize_active marker): optimization_directive.md is the
    # driver's PERSISTENT step-2 input consumed across the whole ko→researcher→kw chain,
    # NOT a stale prior-session output. Suppress archiving it for the duration of the
    # re-entry (across await_optimizer / await_researcher / await_fused_optimizer) so the
    # optimizer/researcher/worker brief is never rendered directive-less. Normal flow has
    # no marker → archiving (load-bearing for the FSM's directive path_exists transitions)
    # is unchanged.
    if (workspace / ".optimize_active").exists():
        stale_paths = tuple(p for p in stale_paths if p != "optimization_directive.md")
    if not stale_paths:
        return
    ts_suffix = f"pre-{state}-{spawn_index}-{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}"
    for fname in stale_paths:
        src = workspace / fname
        if not src.exists():
            continue
        dst = workspace / f".{ts_suffix}-{fname}"
        try:
            src.rename(dst)
            log.info(f"P0v: archived stale {fname} → {dst.name}")
        except OSError as e:
            log.warning(f"P0v archive failed for {fname}: {e}")


def _optimize_built_kernel_present(workspace: Path, plugin) -> bool:
    """Return whether a supported AscendC plugin has a built kernel to tune.

    Every supported mode declares its C++ surface via ``kernel_cpp_dirs()``.
    All declared directories must exist and contain at least one artifact.  A
    legacy workspace without plugin state keeps the historical ``kernel/``
    fallback.
    """
    if plugin is None:
        kd = workspace / "kernel"
        return kd.is_dir() and any(kd.iterdir())
    cpp_dirs = plugin.kernel_cpp_dirs()
    return bool(cpp_dirs) and all(
        (workspace / d).is_dir() and any((workspace / d).iterdir())
        for d in cpp_dirs
    )


def _optimize_reentry_workspace(workspace: Path) -> tuple[bool, str]:
    """`--optimize` re-entry — FSM-owned done→optimize (the `optimize` MODE the state
    machine spec lists but the CLI never wired). Re-opens an already-verified op into
    the optimizer loop WITHOUT re-generating.

    Contract (back/main 2026-06-21):
      - PRESERVE the built kernel/ + worker outputs (the verified step-1 kernel) + the
        driver's EXISTING optimization_directive.md — the optimize-loop tunes/rewrites
        ON it, not from scratch. (Opposite of _cold_start_reset_workspace, which wipes
        worker outputs.)
      - REQUIRE a verified kernel AND an existing optimization_directive.md — we do NOT
        create/clobber the directive; the caller/driver writes it. Missing either →
        return (False, ...) so the caller hard-rejects (can't optimize a non-existent /
        un-directed kernel).
      - Enter at await_optimizer with a FRESH state log → fresh per-state iter-cap, and
        reset lifetime_spawn_count=0 → fresh spawn budget (a new phase gets its own cap;
        no owner `--bump-cap`).
      - Escalation: the optimizer may @handoff researcher→worker for a structural lever
        (e.g. stage-fusion) — that is the existing O4 agent-scheduling chain.

    Returns (ok, reason). Only the terminal state-machine markers are archived (outside
    the workspace); kernel/ + verification.json + optimization_directive.md stay put."""
    vj = workspace / "verification.json"
    directive = workspace / "optimization_directive.md"
    # Ask the mode plugin WHERE its built kernel lives — do NOT hard-code `kernel/`.
    # Ask the plugin for the supported mode's declared AscendC C++ surface;
    # migration and backward generation use different directory layouts.
    from plugins import detect_plugin  # deferred (keep workspace_lifecycle stdlib-acyclic)
    _plugin = detect_plugin(workspace)
    if not _optimize_built_kernel_present(workspace, _plugin):
        _mode = getattr(_plugin, "name", "none")
        return False, (
            f"--optimize requires a built kernel in {workspace} (none found for mode "
            f"'{_mode}': its kernel_cpp_dirs() are absent/empty and no kernel_logic_files present)")
    if not vj.is_file():
        return False, f"--optimize requires verification.json (a verified step-1 kernel) in {workspace}"
    if not directive.is_file():
        return False, ("--optimize requires an existing optimization_directive.md — the "
                       "optimize-loop reads the driver's directive (we do NOT create it). Absent.")
    # Archive ONLY terminal state-machine markers (preserve kernel/ + all worker outputs).
    ts = int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    op_name = workspace.name
    backup_root_env = os.environ.get("COLD_START_BACKUP_ROOT")
    backup_root = (Path(backup_root_env) / op_name if backup_root_env
                   else Path.home() / ".opgen_backups" / op_name)
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / f"pre-optimize-{ts}"
    backup_dir.mkdir(exist_ok=True)
    for name in ("state_transitions.jsonl", ".opgen_state.json", "PROGRESS.md"):
        src = workspace / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    # Remove terminal/finalized markers so the op is no longer "done".
    for marker in list(workspace.glob(".finalized*")) + list(workspace.glob(".finalize_*")):
        try:
            marker.unlink()
        except OSError:
            pass
    # Fresh state log entering at await_optimizer (get_current_state = tail to_state;
    # iter_counts derived per to_state → a fresh log gives the optimize phase a fresh cap).
    iso = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    entry = {
        "ts": iso,
        "from_state": "done",
        "to_state": "await_optimizer",
        "handoff": "→ orchestrator: --optimize re-entry (done→optimize; verified kernel preserved)",
        "matched_transition_index": -1,
        "rationale": "--optimize FSM-owned done→optimize re-entry; fresh spawn-cap; kernel + directive preserved",
        "iter_counts_snapshot": {},
    }
    (workspace / "state_transitions.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    # Reset spawn-cap budget for the new optimize phase (no --bump-cap needed).
    state_json = workspace / ".opgen_state.json"
    st = {}
    if state_json.is_file():
        try:
            st = json.loads(state_json.read_text())
        except Exception:
            st = {}
    st["lifetime_spawn_count"] = 0
    st["optimize_reentry_ts"] = ts
    state_json.write_text(json.dumps(st, indent=2), encoding="utf-8")
    # Marker: while an --optimize re-entry is active, optimization_directive.md is the
    # driver's PERSISTENT step-2 input that the whole ko→researcher→kw chain consumes —
    # NOT a stale prior-session output. _archive_stale_outputs_before_spawn reads this
    # marker to suppress archiving the directive (else the await_optimizer/await_researcher
    # pre-spawn archive renders the optimizer/worker brief directive-less). Normal flow
    # has no marker → archiving unchanged.
    (workspace / ".optimize_active").write_text(
        f"--optimize re-entry active since {ts}; protects optimization_directive.md from "
        f"P0v pre-spawn archiving for the duration of the optimize phase.\n", encoding="utf-8")
    # scan 2026-07-24 (harness-gap #1 cpu_pytorch variant; main's `.opgen_state.json`
    # `reference_regen` schema; OL-282 / DEBT-158). ADDITIVE re-entry hydrate-reference step:
    # the O2.5 CPU-truth reference (edge_dataset.pt) is transient/gitignored and NOT hydrated
    # on re-entry (see this fn's contract — no archive copy) → re-entry then hard-blocks at
    # O2.5 (surfaced on selective_scan_fwd_simd re-optimize). If the reference is MISSING and a
    # committed `reference_regen` recipe is present, regenerate it deterministically. NO-OP when
    # there is no `reference_regen` block → existing re-entry behavior is UNCHANGED (the guard).
    if not (workspace / "edge_dataset.pt").is_file():
        import sys  # local (module keeps a minimal stdlib import set; cf. `import shutil` above)
        _rp = str(Path(__file__).resolve().parent.parent / "reference_provider")
        if _rp not in sys.path:
            sys.path.insert(0, _rp)
        try:
            from reference_regen import regen_reference  # stdlib-only; acyclic (no orchestrator import)
        except Exception:
            regen_reference = None
        if regen_reference is not None:
            try:
                regen_reference(workspace)  # True = reference regenerated; False = no recipe (no-op)
            except Exception as e:
                return False, (
                    f"--optimize re-entry: reference_regen could not reprovide the O2.5 CPU-truth "
                    f"reference for '{op_name}': {e}")
    return True, (f"--optimize: '{op_name}' re-opened at await_optimizer — kernel + "
                  f"optimization_directive.md preserved, spawn-cap reset. State backup: {backup_dir}")


def _record_partial_persist_finalize(
    workspace: Path, state: str, count: int, cap: int
) -> None:
    """P0y: record the legitimate PARTIAL_PERSIST terminal state in both
    state_transitions.jsonl AND verification.json.persist_verdict so REPORT
    generators and audit tools see the cleanly-recorded outcome."""
    # Append a synthetic transition: state → finalize with rationale
    log_path = workspace / "state_transitions.jsonl"
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "from_state": state,
        "to_state": "finalize",
        "handoff": "[P0y orchestrator] iter_cap exhausted with full-pipeline evidence",
        "matched_transition_index": -1,  # synthetic, not from YAML
        "rationale": f"P0y: {state} iter_cap hit (count={count}/{cap}); "
                     f"researcher + probe both ran with requirement verdict → "
                     f"finalize PARTIAL_PERSIST (per V3.8.8 'never let PARTIAL "
                     f"pass' policy: full pipeline exhausted is the terminal state)",
        "iter_counts_snapshot": {},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Tag verification.json with persist_verdict so REPORT.md picks it up.
    # If verification.json is absent (e.g. workspace was cleaned and worker
    # never re-wrote it before exhaustion), create a minimal one carrying
    # ONLY the persist_verdict info so REPORT generation has a usable record.
    # Backup files (.batch*-bak / .pre-*) are intentionally NOT consulted —
    # we don't want to silently restore stale precision/perf numbers.
    vj_path = workspace / "verification.json"
    persist_block = {
        "persist_verdict": "PARTIAL_PERSIST",
        "persist_classification": "requirement",
        "persist_evidence": (
            f"Full V3.8.8 pipeline exhausted: probe verdict=requirement, "
            f"researcher iter_cap hit ({count}/{cap}). Evidence chain: "
            f"workspace/probe_report.md + workspace/cann_strategy_inference.md."
        ),
        "persist_signed_off_by": "P0y_orchestrator_pipeline_exhaustion",
    }
    try:
        if vj_path.exists():
            vj = json.loads(vj_path.read_text())
        else:
            vj = {
                "precision": {"status": "PARTIAL"},
                "performance": {},
                "determinism": {},
                "_note": "verification.json was absent at iter_cap finalize; "
                         "P0y created minimal record with persist_verdict only. "
                         "Per-case precision/perf data unavailable.",
            }
            log.info("P0y: verification.json absent — creating minimal record")
        prec = vj.setdefault("precision", {})
        prec.update(persist_block)
        vj_path.write_text(json.dumps(vj, indent=2))
        log.info("P0y: tagged verification.json.precision.persist_verdict=PARTIAL_PERSIST")
    except Exception as e:
        log.warning(f"P0y persist_verdict tag failed: {e}")
