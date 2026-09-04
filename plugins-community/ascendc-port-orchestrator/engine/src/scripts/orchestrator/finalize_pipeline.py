# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Finalize pipeline — runs when an op reaches the `finalize` state.

Gate identifiers (`GateID`) name every reason a rollback can fire. The id is
the structural contract between `check_finalize_eligibility` (which emits)
and the orchestrator + loop-break detector (which consume). The reason
string is a human-readable diagnostic; the gate id is the stable handle.
Add a new gate → add a `GateID` enum entry → use it in the rollback return.
Never match reason text by substring; always read `elig["gate"]`.


Per old SKILL.md migration map, finalize was supposed to do:
  Phase O5: independent post-verification
  Phase O6: KB merge + archive promotion + commit

Reality (caught 2026-05-05): NEITHER orchestrator.run_single_op NOR resume.execute
ever ran any of these steps. Reaching `finalize` just emitted a state-log
event and exited 0. Result: 20+ ops at finalize state with NO archive in
output/<project>/src/kernels/, stale REPORT.md, no per-op .finalized marker.

This module implements the missing pipeline. Steps (in order):

  1. Idempotency check — if `.finalized-<verification-hash>` marker exists,
     skip (already finalized with same outcome).
  2. Promote workspace artifacts → archive_root/<op_archive_name>/ :
     verification.json, PROGRESS.md, knowledge_update.md, analysis.md,
     kernel/, probe_report.md, optimization_log.md, etc.
  3. KB merge guard — record .kb_merged marker if knowledge_update.md
     was non-trivial (already triggered by orchestrator main loop per-spawn,
     this just drops the marker for audit).
  4. Drop .finalized-<verification-hash> marker.
  5. Return FinalizeReport with steps performed.

Out of scope (future):
  - Independent post-verify rerun (requires live A5 SSH — costly + flaky)
  - Auto-commit + push (CLAUDE.md says don't auto-commit)
  - Auto-Discord (CLAUDE.md says explicit user direction)
  - REPORT.md regeneration (separate /aog-report-gen skill, can run manually)
"""
from __future__ import annotations

import enum
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class GateID(str, enum.Enum):
    """Stable identifiers for every finalize-rollback gate.

    These are the contract `check_finalize_eligibility` returns under the
    `gate` key on rollback. The orchestrator + loop-break detector key off
    `gate`, never reason-text matching. New gate = new enum entry; otherwise
    reuse one of the existing categories.
    """
    VERIFICATION_FILE_MISSING = "verification_file_missing"
    VERIFICATION_MALFORMED = "verification_malformed"
    MODEL_PY_SHAPE = "model_py_shape"          # P0abg
    PASS_A_COVERAGE = "pass_a_coverage"         # P0abd
    KB_WRITEUP = "kb_writeup"              # P0aax
    POST_WORKER_AUDIT = "post_worker_audit"       # P0aba
    PASS_COUNT = "pass_count"              # P0ee
    PERSIST_EVIDENCE = "persist_evidence"        # P0ff
    SIGMOID_FORM_REMEDIATION = "sigmoid_form_remediation"  # P0abi (P-P88)
    OP_HOST_COMPLETENESS = "op_host_completeness"    # PB-33 (2026-05-14)
    ACLNN_VERIFY_PATH_FRAUD = "aclnn_verify_path_fraud"  # DEBT-NEW (2026-05-14)
    BINARY_PROVENANCE = "binary_provenance"       # DEBT-091 (2026-05-15)
    PLATFORM_BLAME_UNBACKED = "platform_blame_unbacked"  # P94 (2026-05-15) — DS PATTERN-1
    INFRA_BASELINE_PAPER_OVER = "infra_baseline_paper_over"  # P96 (2026-05-15) — C-INFRA-BASELINE-PAPER-OVER
    INFRA_RETRY_WITHOUT_CAP = "infra_retry_without_cap"  # P96 (2026-05-15) — C-INFRA-RETRY-WITHOUT-CAP
    PORT_A3_PASS_B_SCHEMA = "port_a3_pass_b_schema"  # P96 (2026-05-15) — C-PORT-A3-PASS-B-SCHEMA
    # P96 (2026-05-15) — caught when gates updated after orchestrator startup
    STALE_ORCHESTRATOR = "stale_orchestrator"
    # P96 follow-up (2026-05-15) — OL-160 USAGE-gap (filename aligned but ModelNew bypassed)
    VERIFIER_USES_MODELNEW = "verifier_uses_modelnew"
    # P97 (2026-05-16) — baseline + candidate use different measurement paths
    PERF_METHODOLOGY_ASYMMETRY = "perf_methodology_asymmetry"
    # P149 (2026-05-18) — host-side business logic (output alignment, CPU offload) in pybind11.cpp
    PYBIND_HOST_BUSINESS_LOGIC = "pybind_host_business_logic"
    # P135.VC (2026-05-18 task #21) — pass_b_runner.py + edge_dataset.pt exist but pass_b never ran
    PASS_B_COVERAGE_SILENT_SKIP = "pass_b_coverage_silent_skip"
    # P0dd 2026-05-23 (owner directive) — kernel TU / pybind shim `#include`s
    # upstream arch35 V351 source (not a V220→V351 port).
    ARCH35_WRAP_CHEAT = "arch35_wrap_cheat"
    # 2026-08-26 (npubench provider contract): provider evaluation pending
    # (verification.json without npubench_evidence) - clear verdict instead of
    # the legacy UNKNOWN_PRECISION_STATUS misroute.
    NPUBENCH_EVALUATION_PENDING = "npubench_evaluation_pending"
    # flash_attention_score-pbh-1 2026-06-11 (owner mandate) — port_a3 FA GE
    # op_host (def/infershape/tiling.cpp) byte-copied from CANN source OR
    # tiling.cpp doesn't use the KB shared `wfh::`/`wp_fa_host::` layer
    # (raw arch35 copy, not recipe-assembled).
    GE_OPHOST_RAW_CANN_COPY = "ge_ophost_raw_cann_copy"
    # OL-188 2026-05-25 (owner directive 02:16Z) — pure-VEC kernel for
    # cube-required CANN reference op (same anti-cheat tier as CPU fallback).
    ARCHITECTURAL_HACK = "architectural_hack"
    # P0ee 2026-05-26 (ROADMAP §86) — default-deny gate: ratio>1.0× claim
    # without explicit symmetric-method declaration is REJECTED.
    METHODOLOGY_DECLARATION = "methodology_declaration"
    UNKNOWN_PRECISION_STATUS = "unknown_precision_status"
    # DEBT-092 2026-05-27 (DS) — PROJECT.json missing required fields
    PROJECT_JSON_METADATA = "project_json_metadata"


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent  # repo root (engine/)

# cannbot KB-relocation adaptation (KB lives at <plugin_root>/kb/, not
# src/skills/references). Re-exported here so callers + tests that reference
# finalize_pipeline._kb_root keep resolving to the relocated KB after the
# v3.13.0 finalize decomposition displaced the original definition.
try:
    from kb_paths import kb_root as _kb_root
except ImportError:  # pragma: no cover — fallback if orchestrator/ not on sys.path
    def _kb_root():  # type: ignore
        return _PROJECT_ROOT.parent / "kb"

# P96 (2026-05-15): capture the mtime of this module at process startup.
# Used to detect "orchestrator running against stale finalize_pipeline" —
# if the on-disk file's mtime > the cached value, new gates landed AFTER
# this process started + the cached module won't enforce them.
# Caught: 2026-05-15 gather_elements_v2 finalize accepted at 20:35Z even
# though P96 _check_infra_paper_over (merged 20:21Z) would have rejected.
# Orchestrator process was started 17:18Z, so cached pre-P96 module.
try:
    _ORCH_MODULE_STARTUP_MTIME = _HERE.stat().st_mtime
except Exception:
    _ORCH_MODULE_STARTUP_MTIME = None


def _check_stale_orchestrator() -> Optional[str]:
    """P96 stale-orchestrator gate: detect when on-disk finalize_pipeline.py
    is newer than the cached module in this Python process.

    Caught 2026-05-15: P96 finalize gates landed mid-flight; running
    orchestrators (started before the merge) accepted ops that the new
    gates would reject. Worker subprocess output is fine (claude --print
    re-execs disk Python), but orchestrator's own check_finalize_eligibility
    runs in the long-lived Python process.

    Returns None if module is current, error string if stale.
    """
    if _ORCH_MODULE_STARTUP_MTIME is None:
        return None  # couldn't capture startup mtime; don't block
    try:
        current_mtime = _HERE.stat().st_mtime
    except Exception:
        return None
    # Permit small drift (<1s) for FS quirks; only flag genuine post-startup edits
    if current_mtime > _ORCH_MODULE_STARTUP_MTIME + 1.0:
        from datetime import datetime as _datetime, timezone as _timezone
        return (
            f"STALE_ORCHESTRATOR (P96): finalize_pipeline.py on disk mtime "
            f"({_datetime.fromtimestamp(current_mtime, _timezone.utc).isoformat()}) "
            f"is newer than process-startup mtime "
            f"({_datetime.fromtimestamp(_ORCH_MODULE_STARTUP_MTIME, _timezone.utc).isoformat()}). "
            f"Cached module does NOT include the new gates. KILL + RESTART this "
            f"orchestrator process to pick up the new code; finalize is BLOCKED "
            f"until restart. See feedback_merge_gates_requires_orchestrator_restart.md."
        )
    return None


# ---------------------------------------------------------------------------
# Promotion policy: archive the WORKSPACE except clear scratch/internal files.
#
# Rationale (P0dd 2026-05-05): a fixed PROMOTE_FILES list is guesswork that
# silently drops artifacts whoever wrote the kernel needed for reproduction
# (input_gen.py, edge_dataset.pt, ref_runnable.json, run_*.py scripts, state
# logs, etc.). Default to archive-everything; explicit exclude set covers
# scratch + per-session backups + agent-internal markers.
#
# To exclude a new pattern: add to EXCLUDE_FILES_RE or EXCLUDE_DIR_NAMES.
# ---------------------------------------------------------------------------
import re

EXCLUDE_DIR_NAMES = frozenset({
    ".cann_learn_sealed",  # sealed research inputs never enter customer archives
    "__pycache__",
    ".cache",
    ".pytest_cache",
})

# Patterns matched against any path component (file or dir):
EXCLUDE_PATTERNS_RE = (
    # dotfiles + dotdirs (.agent_died_at_*, .cc_*, .kernel_worker_active, .opgen_state.json, .recover_log.jsonl, etc.)
    re.compile(r"^\..*"),
    re.compile(r".*\.bak(\..+)?$"),  # *.bak, *.bak_2026-04-15
    re.compile(r".*_bak$"),  # *_bak, *.opt0_bak, *.opt3_bak (kernel snapshots)
    re.compile(r".*\.day\d+-bak.*"),  # *.day4-bak-*, *.day4-bak3-*
    re.compile(r".*\.batch\d+-bak.*"),  # *.batch6-bak-*
    re.compile(r".*\.pre-cold-start.*"),
    re.compile(r".*\.pre-p0[a-z]-recovery.*"),  # P0t recovery archives
    re.compile(r".*\.pyc$"),
    re.compile(r".*~$"),
)


def _merge_copy_dir(src: Path, dst: Path) -> None:
    """Recursively copy src into dst, OVERWRITING files that exist in both
    but PRESERVING files in dst that aren't in src. Skips scratch patterns
    via _should_skip on each child.

    This is the merge-not-replace semantic the finalize pipeline needs:
    workspace's kernel/ might have just (kernel.h, kernels.cpp), but
    archive's kernel/ might also have (msprof_opt_0.json, msprof_opt_1.json)
    from prior runs — those should be PRESERVED, not wiped.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if _should_skip(child.name):
            continue
        target = dst / child.name
        if child.is_dir():
            _merge_copy_dir(child, target)
        else:
            shutil.copy2(child, target)


def _should_skip(name: str) -> bool:
    """True if this top-level name should not be promoted."""
    if name in EXCLUDE_DIR_NAMES:
        return True
    for p in EXCLUDE_PATTERNS_RE:
        if p.match(name):
            return True
    return False


# ---------------------------------------------------------------------------
# Op-name → archive-dir mapping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def is_finalized(workspace: Path) -> bool:
    """True if a .finalized-* marker exists matching current verification hash."""
    h = _verification_hash(workspace)
    if h is None:
        return False
    return (workspace / f".finalized-{h}").exists()


# ---------------------------------------------------------------------------
# P0abe (2026-05-07): rollback-history loop break.
#
# Background: P0abd (and other finalize-rollback gates: P0aax KB, P0aba audit,
# P0ee pass-count) correctly REJECT a bad workspace and route back to
# await_worker. But the rollback REASON is recorded only in
# state_transitions.jsonl. The next kw spawn does NOT see it: kw_brief is
# rebuilt from scratch each time and reads only the workspace's static state,
# not the prior-rollback signature. Result: kw-N+1 emits the SAME shape kw-N
# was rejected for → infinite loop until TOTAL_SPAWN_CAP_PER_OP is hit.
#
# Fix: when we issue a rollback, write a structured signature to
# `.rollback_history.jsonl`. On the NEXT rollback, if the signature matches
# the prior one, route to `await_user_decision` instead of `await_worker`.
# This caps the pathological loop at 2 spawns: kw-N rejected → kw-N+1
# rejected with same signature → user_decision (not infinite).
#
# Companion: kw_brief.py's `rollback_context_block` reads this same file
# and prepends the signature + remediation hint to the next worker's brief
# so kw-N+1 has CONCRETE knowledge of what was rejected, not just same
# stale-context respawn.
# ---------------------------------------------------------------------------
from datetime import datetime as _datetime, timezone as _timezone


# ---------------------------------------------------------------------------
# DEBT-192 (2026-07-03): finalize rollback-loop CONVERGENCE guard.
#
# P0abe (detect_loop_break above) caps a rollback loop at 2 spawns by routing
# to await_user_decision — BUT only when the last TWO *consecutive* rollback
# signatures are byte-identical. The perf-methodology gate loop observed live
# 2026-07-03 (gelu gate #2, run 2684792) evaded it: a precision-clean port_a3
# op declared performance.status=PASS while perf was never validly measured
# (no A3 host, phase_o5 device path stubbed, host-composition-dominated). Each
# respawn oscillated between TWO different-but-equivalent perf rejects —
# `P141 PERF_METHODOLOGY_ASYMMETRY` and the `independent_re_measure`-missing
# POST_WORKER_AUDIT reject — so no two *consecutive* signatures matched,
# detect_loop_break stayed blind, and the op looped to the spawn cap
# (8 spawns / 6 iterations / $28.77, no convergence). See ROADMAP DEBT-192.
#
# This guard is STRICTLY loop-control. It does NOT change WHAT any perf gate
# judges (the gate is RIGHT — the perf claim IS invalid); it only recognizes
# that K rollbacks drawn entirely from the perf-methodology family = no forward
# progress on the perf claim, and stops the loop. The action is chosen by
# classify_loop_break_action (fail-fast halt by default; coerce-N/A recommended
# only for a precision-PASS port_a3 op — that coercion path is owned by the
# port_a3 perf-N/A contract seam, NOT applied here).
# ---------------------------------------------------------------------------

# K: number of rollbacks drawn from the perf-methodology family (in the tail
# window) with no forward progress after which the loop is declared
# non-convergent. K=3 gives the worker three genuine attempts (initial + 2
# respawns) before the engine stops looping — enough for a transient worker
# mistake to self-correct, far below the observed 6-iteration runaway, and one
# more than P0abe's cap-at-2 for the byte-identical case (which still fires via
# detect_loop_break, preserved below).
LOOP_BREAK_K = 3

# Perf-methodology gate family. A worker oscillating among these makes no real
# forward progress on the perf claim, so they collapse to ONE non-convergent
# loop even when two *consecutive* (gate, rollback_state) signatures differ.
# PERF_METHODOLOGY_ASYMMETRY (P97/P141) and METHODOLOGY_DECLARATION (P0ee) are
# unambiguously perf-methodology gates. POST_WORKER_AUDIT (P0aba) is broader
# than perf, so it counts as perf-family ONLY when its recorded reason is about
# performance / independent_re_measure — a read-only inspection of the reason
# text already stored in .rollback_history.jsonl, never a gate-criteria change.
_PERF_METHODOLOGY_GATES = frozenset({
    GateID.PERF_METHODOLOGY_ASYMMETRY.value,
    GateID.METHODOLOGY_DECLARATION.value,
})


def _entry_is_perf_family(entry: dict) -> bool:
    """True if a rollback-history entry belongs to the perf-methodology family.

    Read-only classification for loop-convergence detection (DEBT-192). Does
    NOT influence any gate verdict.
    """
    gate = entry.get("gate")
    if gate in _PERF_METHODOLOGY_GATES:
        return True
    if gate == GateID.POST_WORKER_AUDIT.value:
        reason = (entry.get("reason") or "").lower()
        return (
            "independent_re_measure" in reason
            or "performance" in reason
            or "perf" in reason
        )
    return False


# ---------------------------------------------------------------------------
# P0abd (2026-05-07): benchmark Pass A coverage gate.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# P0abg (2026-05-08): model.py shape lint — `get_input_groups()` mandatory.
#
# The multi-case source interface (`model.py` + `<op>.json`) uses
# `get_input_groups()`. When
# kw writes model_cpu_truth.py for Path-A CPU-truth (P0xxx 2026-05-15),
# the source interface contract carries: BOTH model.py AND model_cpu_truth.py
# MUST define `get_input_groups()` returning a list of input lists (even if
# it's a list of one element). model.py is the preserved benchmark source
# (CANN NPU reference); model_cpu_truth.py is the Path-A CPU oracle.
#
# `verification_ascendc.py` has a legacy fallback that wraps `get_inputs()`
# in `[inputs]` — silently produces `pass_a.total=1` regardless of how many
# cases the JSONL spec declared. This is silent coverage fraud (caught
# 3_FusionAttention 2026-05-07: 50 cases declared, 1 case verified).
#
# Fix is structural at the orchestrator layer (we control kw, vendor verifier
# is fine as-is): refuse to finalize when either model.py or model_cpu_truth.py
# only defines `get_inputs`. kw must rewrite. Companion: kw_brief Phase D
# mandates the rule explicitly.
# ---------------------------------------------------------------------------
import ast as _ast


# ---------------------------------------------------------------------------
# P96 (2026-05-15) — infra-baseline + retry-vs-paper-over gates
# Catalog entries C-INFRA-RETRY-WITHOUT-CAP + C-INFRA-BASELINE-PAPER-OVER +
# C-PORT-A3-PASS-B-SCHEMA in aog-self-critic. See:
# - kb/shared/ANTI_PRESSURE_PROTOCOLS.md §P9
# - docs/baseline/environment_baseline.yaml
# ---------------------------------------------------------------------------

# Keyword catalog for C-INFRA-BASELINE-PAPER-OVER. Workspace docs containing
# any of these phrases indicate worker did STRUCTURAL env workaround without
# escalating to preflight/specialized skill.
# DEBT-98 (2026-05-24): negative-assertion markers that, if found in the
# 80-char window before a trigger keyword match, indicate the match is a P9
# negative-assertion (worker documenting they DID NOT perform paper-over)
# and should be skipped — not a real paper-over to block.


# NPU error codes that are baseline violations, NOT transient


# P96 self-defeat fix (2026-06-16, abs_nocase backward incident). Markers that,
# when found in the window around an NPU baseline error-code citation, indicate
# the error was KERNEL-CAUSED (the worker's own kernel UB bug, env proven
# healthy, fixed in-kernel) and therefore NOT an infra baseline violation that
# needs an INFRA_BASELINE_VIOLATED escalation. Without this, the bare
# `if code in text` check forced honest workers to SCRUB the error-code token
# out of docs to pass P96 — the gate INDUCING documentation fraud, the inverse
# of its purpose (witnessed: worker deleted the `507035` token while keeping the
# full kernel-cause mechanism + env-health proof, so the scanner went blind but
# the meaning was unchanged). Mirrors the DEBT-98 negative-assertion window the
# paper-over phrase-check already honors — same fail-honest principle.


# `_check_op_host_completeness` (PB-33 op_host completeness dispatcher) moved to
# finalize_dispatch.py (DEBT-201, 2026-07-06) with the rest of the plugin-dispatch
# cluster so that its bare-name `_get_active_plugin` call keeps seeing the
# monkeypatched plugin. Re-exported at the bottom of this module.


# fa_class GE op_host template dir (the KB-authored, compile-verified, non-CANN
# generative source for def/infershape/tiling.cpp + the shared-layer headers).
# Canonical FA template op-name token. The template files are named/coded for
# this op; the assembler substitutes it with the actual op name so the step
# generalizes to other FA-class ops (identity for flash_attention_score).
# The 3 GE units to assemble (suffixes appended to the op-name token).
# Op-name-specific GE headers (suffix + ".h"). `_tiling_common` carries the
# KB-authored CompileInfo POD — re-authored from the installed platform_ascendc
# API (NOT the CANN-source-only `<op>_tiling_common.h`), per owner rule
# 2026-06-13: CANN *install* headers are usable AscendC API; CANN *source-only*
# headers are not. Emitting it from KB makes the GE op_host 100% KB-generated
# (logic + this POD) + installed-CANN-API — no target implementation access.
# Shared-layer headers the GE units consume (op-name-agnostic — copied verbatim).


# ---------------------------------------------------------------------------
# The plugin-dispatch + finalize-eligibility + promote cluster
# (`_get_active_plugin`, `_run_plugin_extra_finalize_checks`,
# `_check_op_host_completeness`, `_pass_branch_gate_specs`,
# `check_finalize_eligibility`, `batch_precheck`, `_precheck_blocked`,
# `format_batch_precheck_report`, `_finalize_with_plugin_layout`, `finalize_op`,
# `FinalizeReport`, `_PLUGIN_EXTRAS_SENTINEL`) moved to finalize_dispatch.py
# (DEBT-201, 2026-07-06). The two monkeypatched entry points
# (`_get_active_plugin`, `finalize_op`) are called by BARE name from their
# sibling functions in that cluster, so they had to relocate TOGETHER for
# monkeypatch.setattr(<module>, ...) to keep biting the intra-cluster callers.
# finalize_pipeline re-exports the same objects (bottom import) so
# `finalize_pipeline.finalize_op` etc. stay valid; tests that patch these two
# now target finalize_dispatch (see test_finalize_decomposition.py).
# ---------------------------------------------------------------------------

# --- pure leaf helpers extracted to finalize_shared.py (DEBT-201, 2026-07-06) ---
# These are pure (stdlib-only) helpers + marker constants that were previously
# defined here AND cross-imported by the finalize_checks_* siblings. Housing them
# in finalize_shared breaks the finalize_pipeline<->finalize_checks_* cycle for
# these symbols. Re-imported here so the remaining functions in this module call
# them by bare name unchanged AND `finalize_pipeline.<name>` stays valid for
# callers/tests. None are monkeypatched.
from finalize_shared import (  # noqa: E402,F401  re-export: keep call sites + import paths stable
    _verification_hash, _is_negative_assertion, _is_negative_assertion_window,
    _is_kernel_caused_context_window, _benchmark_case_count, _kb_writeup_body_len,
    _is_harness_internal, _is_v220_ec41_output_pad_exempt, _has_profiler_csv_method,
    _HARNESS_INTERNAL_FILES,
    _NEGATIVE_ASSERTION_MARKERS, _KERNEL_CAUSED_CONTEXT_MARKERS,
    _INFRA_PAPER_OVER_PHRASES, _NPU_BASELINE_ERROR_CODES,
    _PROFILER_CSV_TOKENS, _DEVICE_DURATION_TOKENS,
)

# --- archive README rendering extracted to finalize_readme.py (DEBT-201, 2026-07-06) ---
from finalize_readme import (  # noqa: E402,F401  re-export: keep call sites + import paths stable
    _render_verification_conclusion, _assemble_readme, _write_archive_readme,
)

# --- GE op_host assembly extracted to finalize_ge_ophost.py (DEBT-201, 2026-07-06) ---
from finalize_ge_ophost import (  # noqa: E402,F401  re-export: keep call sites + import paths stable
    assemble_ge_ophost,
    _FA_GE_OPHOST_TEMPLATE_DIR, _FA_GE_TEMPLATE_OP_TOKEN, _FA_GE_CPP_SUFFIXES,
    _FA_GE_HDR_SUFFIXES, _FA_GE_SHARED_HEADERS,
)

# --- KB-candidate verified-on tracking + archive-op-name resolver extracted to
#     finalize_candidates.py (DEBT-201, 2026-07-06) ---
from finalize_candidates import (  # noqa: E402,F401  re-export: keep call sites + import paths stable
    _resolve_archive_op_name, _scan_workspace_for_candidate_refs, _patch_evidence_prose,
    _append_verified_on, update_verified_on_for_consumed_candidates,
    _CAND_TOKEN_RE, _PROSE_PATCH_PATTERNS,
)

# --- rollback/loop-break extracted to finalize_rollback.py (behavior-neutral, 2026-07-05) ---
from finalize_rollback import (  # re-export: keep existing import paths stable
    _rollback_signature, _rollback_history_path, record_rollback, detect_loop_break,
    _read_rollback_history, detect_nonconvergent_loop, classify_loop_break_action,
)

# --- plugin-dispatch + finalize-eligibility + promote cluster extracted to
#     finalize_dispatch.py (DEBT-201, 2026-07-06). Re-export so
#     `finalize_pipeline.finalize_op`, `.check_finalize_eligibility`,
#     `.batch_precheck`, `._get_active_plugin`, `.FinalizeReport`, etc. all keep
#     resolving. The two monkeypatched names (`_get_active_plugin`,
#     `finalize_op`) now LIVE in finalize_dispatch; monkeypatch tests target
#     that module (patching this module's re-exported attribute would NOT reach
#     the intra-cluster bare-name callers). This import MUST precede the
#     finalize_checks import below: finalize_checks_provenance does
#     `from finalize_pipeline import _get_active_plugin`, so the name has to be
#     re-exported here first. finalize_dispatch is acyclic w.r.t. finalize_checks
#     because it imports the `_check_*` gate functions LAZILY (inside its bodies).
from finalize_dispatch import (  # re-export: keep call sites + import paths stable
    _get_active_plugin, _run_plugin_extra_finalize_checks, _check_op_host_completeness,
    _pass_branch_gate_specs, _PLUGIN_EXTRAS_SENTINEL, check_finalize_eligibility,
    batch_precheck, _precheck_blocked, format_batch_precheck_report,
    _finalize_with_plugin_layout, finalize_op, FinalizeReport,
)

# --- gate checks extracted to finalize_checks.py (behavior-neutral, 2026-07-05) ---
from finalize_checks import (  # re-export: keep call sites + import paths stable
    _check_pass_a_coverage, _check_model_py_shape, _check_binary_provenance,
    _check_universal_entrypoints, _check_platform_blame_backed, _check_infra_paper_over,
    _check_infra_retry_budget, _check_port_a3_pass_b_schema, _check_pass_b_coverage,
    _check_a5_verify_path_provenance, _check_arch35_wrap_cheat,
    _check_ge_ophost_raw_cann_copy, _check_architecture_class, _check_project_json_metadata,
    _check_pp88_compliance, _check_kb_writeup,
    _check_verifier_uses_modelnew, _check_post_worker_audit, _check_pass_count_concrete,
    _check_pybind_host_logic, _check_perf_methodology,
    _check_methodology_declaration,
)
