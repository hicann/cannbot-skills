# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O5 independent post-verify (P0kk 2026-05-06).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 2.

Scope:
    Before finalize_pipeline runs, re-measure the kernel against
    edge_dataset.pt CPU truth on A5 NPU, compare to the worker's
    claimed counts in verification.json. If mismatch >tolerance,
    refuse to finalize and route back to await_worker.

Closes the loophole: workers can write status="PASS" with pass counts
they didn't actually achieve. P0ee added schema-level cross-check
(counts must match status); P0kk adds physical cross-check (counts
must match real measurement).

Architecture (per user direction 2026-05-06): model-agnostic. The
runner that executes verifier scripts on A5 is a pluggable callable.
Empirical A/B can later decide whether to use sonnet for the runner
agent (env setup + parse) vs pure-Python.

Default runner is pure-Python: subprocess to run verifier locally
(if verifier produces structured output) OR SSH-tunnel to A5 if
needed. No LLM call in the default path.
"""
from __future__ import annotations
import logging

import json
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


def _mfu_op_kind(workspace: Path, op: str) -> Optional[str]:
    """Derive MFU op_kind from op-class (NOT hardcoded). None if unmapped -> hook skips honestly.
    Sources: op name heuristics + workspace/op_classification.json. Extend as FLOPs formulas grow."""
    name = (op or "").lower()
    if "mm_grad" in name or ("matmul" in name and "grad" in name):
        return "mm_grad"
    if "matmul" in name or "gemm" in name or "linear" in name:
        return "matmul"
    try:
        oc = json.loads((workspace / "op_classification.json").read_text())
        cls = str(oc.get("class") or oc.get("op_class") or oc.get("category") or "").lower()
        if "matmul" in cls or "gemm" in cls or "cube" in cls:
            return "matmul"
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    return None


def _inject_mfu_ceiling_safe(vp: Path, workspace: Path, op: str, v: dict) -> None:
    """机械注入 MFU 天花板到 verification.json(perf 后), ko 据此做 MFU-gated done.
    Fail-safe: 绝不因这个可选信号打断 perf pipeline. op_kind 从 op-class 取(非写死 matmul);
    hw 从 target/soc 取. op-class 无 FLOPs 公式 -> hook 诚实记 skipped, 不造假."""
    try:
        mfu_dir = str(Path(__file__).resolve().parents[3] / "mfu")
        if mfu_dir not in _sys.path:
            _sys.path.insert(0, mfu_dir)
        from verification_hook import inject_mfu_ceiling
        tgt = str(v.get("target") or "").lower()
        soc = str(v.get("soc_version") or "").lower()
        if "950" in tgt or "950" in soc or tgt == "a5":
            hw = "950PR"
        else:
            hw = "910C_die"  # a3/910 default
        op_kind = _mfu_op_kind(workspace, op)
        inject_mfu_ceiling(str(vp), op_kind=op_kind, hw_name=hw, write=True)
    except Exception:
        pass  # optional signal — never break perf finalize


# Tolerance for re-measurement vs claim. tier1_pass should match exactly;
# total may legitimately differ if benchmark grew between runs but worker
# was using older edge_dataset.pt.
EXACT_FIELDS = ("tier1_pass", "total")
COMPARABLE_PASSES = ("pass_a", "pass_b")

# P0cc (2026-05-23, flat_quant follow-up to EC-59): when worker tags
# precision.<pass>.status as PASS_WITHIN_TOLERANCE OR PARTIAL_PASS, the
# strict tier1_pass count alone over-rejects — bf16 / fp8 / int4 quant
# ops legitimately produce T2 (within v2.1 §4.5.1 tolerance) verdicts
# that are NOT bit-exact but ARE accepted as PARTIAL_PERSIST. The
# inclusive count (tier1_pass + tier2_pass_within_tolerance) is the
# truth for those statuses. Runner template MUST emit BOTH counts so
# O5 can compare inclusively when the worker-declared status warrants.
INCLUSIVE_FIELDS = ("tier1_pass_inclusive", "total")
INCLUSIVE_STATUSES = (
    "PASS_WITHIN_TOLERANCE",
    "PARTIAL_PASS_WITHIN_TOLERANCE",
    "PARTIAL_PASS",
)

# O5-BESTEFFORT-DET-COUNT-TOLERANCE (2026-07-21): a best-effort NONDETERMINISTIC
# op whose tier1_pass count varies run-to-run (e.g. 13-16 of 16) stochastically
# TRIPS the exact-equality count reconciliation below → O5 MISMATCH → the FSM
# rolls back to await_worker and respawns, burning the entire worker budget on
# an op that never actually regressed. The fix: for `tier1_pass` ONLY, when the
# op is best_effort AND the author declared an immutable `det_floor`, MISMATCH
# only when the MEASURED count drops BELOW the declared floor; tolerate any
# measured count at/above the floor (that is exactly the benign stochastic
# variance). Deterministic ops (det_policy in {required, n_a}) and best_effort
# ops with NO declared floor keep today's exact-equality check — no silent
# tolerance.
#
# CRITICAL anti-launder invariant: this tolerance ONLY stops the stochastic
# count-MISMATCH→respawn churn. It does NOT mark the op deterministic/clean and
# it NEVER touches the determinism sub-block (`determinism.policy_satisfied` /
# the O5.inv1 gate in workflow_critic_validators.py) — that stays a SEPARATE
# required gate which still honestly reports policy_satisfied=false when
# determinism was not achieved. The overall op outcome for a tolerated
# best-effort op is therefore a stable PARTIAL_PERSIST that STILL records the
# determinism failure, never a laundered VERIFIED-deterministic. det_floor is a
# DECLARED INPUT (Phase O1.5, immutable) — never derived from the measurement
# being judged, so author≠measurer stays intact.
_DET_TOLERANCE_FIELD = "tier1_pass"
_BEST_EFFORT_POLICY = "best_effort"


def _read_det_policy_and_floor(workspace: Path, v: dict) -> tuple[Optional[str], Optional[int]]:
    """Return (det_policy, det_floor) for the count-reconciliation tolerance.

    Both are DECLARED INPUTS, read (never derived from this run's measurement):
      - primary source: verification.json `determinism` block
        (`policy`, `det_floor`) — the block the worker/determinism-analyzer
        authors and where `determinism.policy` already lives.
      - fallback: the durable per-op state file (`.opgen_state.json`,
        `det_policy` / `det_floor`) written at Phase O1.5.

    det_floor is normalized to a non-negative int; anything unparseable → None
    (= no tolerance declared → exact equality preserved, no silent tolerance).
    """
    det = v.get("determinism") if isinstance(v.get("determinism"), dict) else {}
    policy = det.get("policy")
    floor = det.get("det_floor")
    if policy is None or floor is None:
        stored_policy, stored_floor = _read_stored_det_policy_and_floor(workspace)
        policy = policy if policy is not None else stored_policy
        floor = floor if floor is not None else stored_floor
    return policy, _normalize_det_floor(floor)


def _read_stored_det_policy_and_floor(workspace: Path) -> tuple[Optional[str], object]:
    """Read the optional Phase O1.5 determinism declaration without failing O5."""
    try:
        import phase_o05
        state = json.loads((workspace / phase_o05.STATE_FILE).read_text())
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
        return None, None
    if not isinstance(state, dict):
        return None, None
    return state.get("det_policy"), state.get("det_floor")


def _normalize_det_floor(floor: object) -> Optional[int]:
    """Return a valid non-negative declared floor, or None when it is unusable."""
    try:
        normalized_floor = int(floor) if floor is not None else None
    except (TypeError, ValueError):
        return None
    return normalized_floor if normalized_floor is None or normalized_floor >= 0 else None


@dataclass
class MeasuredResult:
    """A runner produces this — concrete pass counts re-measured against
    the same edge_dataset that produced the claim.
    """
    pass_a: Optional[dict] = None  # {tier1_pass, total, ...}
    pass_b: Optional[dict] = None
    determinism: Optional[dict] = None  # {n_identical_cases, n_cases_checked}
    # B3.3c (2026-06-17): orchestrator-side independent perf re-measure captured by
    # a runner that parses the verify script's perf block. Backward mode:
    # backward_verify_runner runs verify_<op>.py on the NPU and parses its perf
    # summary — independent of the worker's verification.json self-report. Shape:
    # {ratio, median_ratio, rows, method, ...} as emitted by verify_<op>.py. None
    # when the runner doesn't capture perf (benchmark/port_a3 use the separate
    # phase_o5_perf_capture path; only backward populates this so far). Mapped to
    # performance.independent_re_measure downstream (gate logic unchanged).
    perf: Optional[dict] = None
    runner_error: Optional[str] = None  # set if runner couldn't execute
    # NODE-5 (2026-05-28): when `runner_error` is an infrastructure-class
    # failure (SCP timeout, oversized .pt, verifier env issue, JSON parse
    # fail on stdout, ssh/cat unreachable), set to "infra" so the downstream
    # rollback transition can be tagged and `iter_below_cap` can skip the
    # re-entry from consuming the algorithm-iter budget. Empirical anchor:
    # FA arch22 reference run `bjfp4fi3e` 12:18-16:03Z — kernel verified 10/10 on
    # probe-1, probe-2, AND probe-3 (same algorithm, no kernel change) but
    # each finalize was rolled back by a different infra gate (pass_b_runner
    # missing → P0kk O5 MISMATCH → P0ff knowledge_update.md missing), each
    # consuming a probe-iter slot. NODE-4 raised cap 4→8 as a stopgap; this
    # field enables the cleaner long-term fix.
    #
    # Auto-classified in __post_init__ when runner_error is set and rollback_kind
    # is None — call sites can stay terse (`MeasuredResult(runner_error=msg)`)
    # and still get the right tag downstream.
    rollback_kind: Optional[str] = None  # None / "infra" / "algorithm"

    def __post_init__(self) -> None:
        if self.runner_error and self.rollback_kind is None:
            self.rollback_kind = classify_runner_error(self.runner_error)


# NODE-5 (2026-05-28): infrastructure-class runner_error fragments. When a
# `MeasuredResult.runner_error` message contains ANY of these substrings,
# the failure is classified as "infra" — the kernel didn't fail; some
# external dependency did. Rollback re-entry triggered by an infra failure
# SHOULD NOT consume the algorithm iter_below_cap budget (handled in
# state_machine.iter_counts_from_log via the rollback_kind tag in
# state_transitions.jsonl entries).
#
# Conservative classification: when in doubt, default to "algorithm"
# (consumes iter), NOT "infra" (free re-entry). False-positive "infra" would
# let real algorithm bugs loop forever; false-negative ("algorithm" when
# actually infra) just costs one iter slot — recoverable. Pattern set is
# substring-match (cheap + readable); regex was considered but the empirical
# error strings are stable enough that substring is adequate.
_INFRA_ERROR_PATTERNS: tuple = (
    # SCP / network
    "scp aborted",
    "scp failed",
    "ssh timeout",
    "ssh: connect",
    "ConnectTimeout",
    # Oversized payload (DEBT-23 / Gap A)
    "oversized payload",
    "exceed 100 MiB threshold",
    "exceeds the size limit",
    # Verifier subprocess / env
    "tool missing",
    "verifier stdout had no parseable JSON",  # NODE-2 (Gap B) class
    "verifier exit",
    "timeout after",
    "SSH+verifier timeout",
    # Container / docker
    "docker: Error",
    "docker exec",
    "container not found",
    # Workspace sync
    "pre-O5 workspace sync failed",
    # Env / config
    ".ascendc_env",
    "missing A5_HOST",
    "missing A3_HOST",
    "CANN env",
)


# DEBT-O5-INFRA-0-MISCLASSIFY (2026-07-20): interpreter-resolution failure.
# A stale/missing {TARGET}_NPU_PYTHON_BIN makes the O5 re-verify run
# `<stale_bin>/python3 <script>`, which the shell cannot exec → exit 127 with
# `<bin>/python3: No such file or directory`. That is an INFRASTRUCTURE
# failure — the interpreter path is wrong, the kernel ARTIFACT is intact — so
# it should be re-attempted in place, NOT rolled back to await_worker + a
# pointless base-worker respawn (there is nothing for the author to re-emit).
#
# TIGHT SCOPE (deliberate): we must NOT match the bare "No such file" / "127"
# tokens — a legit kernel/source file-not-found (e.g. `mykernel.h: No such
# file or directory`) or the run_pass_b.py-discoverability case
# (`python3: can't open file '.../run_pass_b.py'`) would falsely match and
# get a free infra re-entry. The signature below fires ONLY when the
# not-found error names a PYTHON INTERPRETER PATH itself (contiguous, so the
# interpreter path + the not-found error necessarily co-occur), or when the
# error explicitly names the NPU_PYTHON_BIN token; and it NEVER fires on a
# script-open failure ("can't open file"), which is the discoverability class
# that legitimately rolls to await_worker.
_INTERPRETER_RESOLUTION_FAILURE_PATTERNS: tuple = (
    "/python3: no such file or directory",
    "/python: no such file or directory",
    "/python3: cannot execute: required file not found",
    "/python: cannot execute: required file not found",
)


def _is_interpreter_resolution_failure(lower: str) -> bool:
    """DEBT-O5-INFRA-0-MISCLASSIFY: True iff `lower` (a lowercased runner_error)
    is a python-interpreter-resolution failure (stale/missing NPU_PYTHON_BIN
    → exit-127 `<bin>/python3: No such file or directory`), NOT a kernel/source
    file-not-found nor a script-open failure. See the pattern-block comment for
    the tight-scope rationale."""
    if "can't open file" in lower:  # `python3: can't open file 'X.py'` — script
        return False  # discoverability, NOT interpreter-missing
    if any(p in lower for p in _INTERPRETER_RESOLUTION_FAILURE_PATTERNS):
        return True
    # A form that explicitly names the resolution var alongside not-found.
    return "npu_python_bin" in lower and "no such file or directory" in lower


def classify_runner_error(msg: str) -> str:
    """NODE-5 (2026-05-28): classify a runner_error string as "infra" vs
    "algorithm". See `_INFRA_ERROR_PATTERNS` for the substring catalog +
    rationale for conservative bias toward "algorithm".

    Returns "infra" iff the message contains ANY infra pattern fragment;
    "algorithm" otherwise (default — conservative). The classification is
    used downstream by `iter_counts_from_log` to skip infra re-entries from
    the iter_below_cap budget, and (DEBT-O5-INFRA-0-MISCLASSIFY) by
    `fsm_phase_finalize._o5_post_verify` to re-attempt O5 in place instead of
    respawning the base worker on an intact artifact.
    """
    if not msg:
        return "algorithm"
    lower = msg.lower()
    for pat in _INFRA_ERROR_PATTERNS:
        if pat.lower() in lower:
            return "infra"
    # DEBT-O5-INFRA-0-MISCLASSIFY: interpreter-resolution failure (exit-127
    # stale NPU_PYTHON_BIN) — tightly scoped so a legit kernel-file "No such
    # file" does NOT match. Checked after the substring catalog so the catalog
    # keeps precedence for any overlapping fragment.
    if _is_interpreter_resolution_failure(lower):
        return "infra"
    return "algorithm"


@dataclass
class O5Report:
    """Result of Phase O5 re-verification."""
    verdict: str  # "VERIFIED" | "PROVISIONAL" | "MISMATCH" | "RUNNER_FAILED" | "SKIPPED"
    claimed: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    # O5-BESTEFFORT-DET-COUNT-TOLERANCE (2026-07-21): tier1_pass discrepancies
    # that were TOLERATED (best_effort + measured >= declared det_floor) instead
    # of raised as mismatches. Recorded for honesty — the count DID differ from
    # the claim, we just did not treat the benign stochastic drop as a MISMATCH.
    # A non-empty list means the op is nondeterministic-but-above-floor; it does
    # NOT upgrade the op to deterministic/clean (the separate determinism gate
    # still owns policy_satisfied).
    det_tolerated: list[str] = field(default_factory=list)
    summary: str = ""
    # W6 (2026-05-12, ROADMAP §1.5): observability — emit which truth source
    # the verifier scripts SHOULD have used for the claimed comparison.
    # "a3_cann"     — port_a3_to_a5 mode (truth = edge_dataset.pt["a3_outputs"]
    #                 captured by phase_o25_a3_ref)
    # "backward_autograd" — backward generation's fp64 gradient oracle
    # The actual truth comparison lives in the worker-authored verifier
    # scripts (run_pass_a.py / run_pass_b.py); phase_o5 only re-measures
    # counts. This field surfaces the expected source so the finalize gate
    # / report can audit the worker's verifier matched the mode.
    truth_source: str = "unresolved"
    # NODE-5 (2026-05-28): when verdict=RUNNER_FAILED OR MISMATCH triggers
    # a state-machine rollback, this tag classifies the rollback for
    # iter-cap accounting:
    #   - "infra"     — infrastructure failure (SCP timeout, oversized .pt
    #                   buffer, verifier env issue, JSON parse fail on
    #                   stdout). Re-entry SHOULD NOT consume the algorithm
    #                   iter_below_cap budget (the kernel didn't change).
    #   - "algorithm" — real algorithm-level rollback (count mismatch,
    #                   precision regression). Re-entry SHOULD consume
    #                   iter_below_cap budget (the kernel needs rework).
    #   - None        — no rollback applicable (verdict=VERIFIED/SKIPPED).
    # Threaded into TransitionDecision.rollback_kind which `record_transition`
    # writes to state_transitions.jsonl; `iter_counts_from_log` skips
    # entries with rollback_kind=="infra".
    rollback_kind: Optional[str] = None
    # DEBT-213(b) (2026-07-17): was the code that MEASURES this run
    # unmodified while it measured? O5 is the step that mints VERIFIED, so
    # the harness state at O5 time is the state of the instrument that
    # produced this verdict.
    #   harness_git_state — "CLEAN" | "DIRTY" | "UNKNOWN" (see
    #       harness_pristine.py; UNKNOWN = not a git checkout, e.g. a
    #       customer's unpacked bundle — recorded, but does NOT downgrade).
    #   harness_dirty — the uncommitted harness paths, named so the reason
    #       is auditable later rather than a bare "something was dirty".
    # DIRTY downgrades verdict VERIFIED -> PROVISIONAL. It deliberately does
    # NOT block: per ROADMAP DEBT-213 the op still finalizes, its verdict
    # just does not count as verified until part (c) re-verifies on clean
    # main.
    harness_git_state: str = "UNKNOWN"
    harness_dirty: list[str] = field(default_factory=list)


def _apply_harness_pristine(rep: O5Report) -> O5Report:
    """DEBT-213(b): stamp the harness git state onto `rep`; downgrade a
    VERIFIED verdict to PROVISIONAL when the measuring code is dirty.

    Only VERIFIED is downgraded. MISMATCH / RUNNER_FAILED already refuse
    finalize, and SKIPPED never claimed anything — a dirty harness makes
    none of them weaker.

    Never raises: a failure to determine the state is UNKNOWN, which is
    recorded but does not downgrade (see harness_pristine.__doc__ for why
    UNKNOWN must not fire — a non-git bundle would otherwise flag every run).
    """
    try:
        import harness_pristine
        state = harness_pristine.harness_state()
    except Exception:  # defensive: never let the audit break finalize
        rep.harness_git_state = "UNKNOWN"
        rep.harness_dirty = []
        return rep

    rep.harness_git_state = state.state
    rep.harness_dirty = list(state.dirty_paths)
    if state.is_dirty and rep.verdict == "VERIFIED":
        rep.verdict = "PROVISIONAL"
        rep.summary = (
            f"O5 PROVISIONAL (DEBT-213): claim matches measurement, but the "
            f"harness was NOT pristine for this run — {len(state.dirty_paths)} "
            f"uncommitted harness file(s): {', '.join(state.dirty_paths)}. "
            f"The subject may have modified the instrument that measured it; "
            f"this verdict does not count as VERIFIED until the op re-verifies "
            f"on a clean harness."
        )
    return rep


def expected_truth_source(workspace: Path) -> str:
    """W6: plugin-method dispatch (DEBT-161 paradigm). Each plugin declares
    its O5 truth source via BasePlugin.truth_source(); a plugin returning a
    truthy value wins. Missing, malformed, or unsupported workspace state is a
    hard error: it must never fall back to a synthetic CPU truth source.
    """
    try:
        from plugins import detect_plugin as _detect_plugin
        _active_plugin = _detect_plugin(workspace)
        if _active_plugin is not None:
            _ts = _active_plugin.truth_source()
            if _ts:
                return _ts
    except Exception as exc:
        raise RuntimeError(
            f"cannot resolve supported truth source for {workspace}: {exc}"
        ) from exc
    raise RuntimeError(
        f"cannot resolve supported truth source for {workspace}: "
        "workspace is not owned by port_a3_to_a5 or backward"
    )


@dataclass
class _PerfReMeasureContext:
    """Inputs needed to persist an O5 independent performance measurement."""

    verification: dict
    verification_path: Path
    measured: MeasuredResult
    workspace: Path
    op: str
    report: O5Report


def _build_independent_re_measure(perf: dict, worker_ratio: object) -> dict:
    """Build the honest independent-measurement block for a runner result."""
    if perf.get("_remeasure_na"):
        return {
            "ran": False,
            "status": "N/A",
            "reason": perf.get("reason")
            or "orchestrator could not independently re-measure perf",
            "source": (
                "phase_o5 post_verify: orchestrator attempted an independent "
                "perf re-measure and it was structurally unmeasurable "
                "(DEBT-192 worker-perf-report contract)"
            ),
        }

    measured_ratio = perf.get("median_ratio")
    if measured_ratio is None:
        measured_ratio = perf.get("ratio")
    result = {
        "ran": True,
        "ratio": measured_ratio,
        "median_ratio": perf.get("median_ratio"),
        "rows": perf.get("rows"),
        "method": perf.get("method"),
        "source": (
            "phase_o5 post_verify: orchestrator re-ran verify_<op>.py on NPU and "
            "parsed its perf block (independent of worker verification.json self-report)"
        ),
    }
    if worker_ratio is not None and measured_ratio is not None:
        try:
            result["delta_vs_kw_self_report"] = round(
                float(measured_ratio) - float(worker_ratio), 4
            )
        except (TypeError, ValueError):
            pass
    return result


def _write_perf_independent_re_measure(context: _PerfReMeasureContext) -> None:
    """Write performance.independent_re_measure from the orchestrator's perf
    re-measure (B3.3c 2026-06-17 + DEBT-192 2026-07-04).

    measured.perf comes from a runner re-measuring on the NPU (orchestrator-
    driven), NOT the worker's verification.json self-report — that
    orchestrator-drive is what makes it an *independent* re-measure
    (author≠measurer preserved). MODE-AGNOSTIC: any runner that sets
    measured.perf flows through here (backward_verify_runner; port_a3 A3-golden
    via _maybe_port_a3_perf_remeasure). Gate logic in finalize_pipeline is
    UNCHANGED — this only supplies the field it already reads.

    DEBT-192: when the runner signals `_remeasure_na` (a port_a3 op whose perf
    was structurally unmeasurable — no perf_runner.py / no A3 baseline / runner
    errored), record an HONEST {ran:false, status:N/A, reason} irm — never a
    ran:true with a fabricated ratio, and never leaving the worker's
    self-reported PASS unchallenged. The worker-perf-report contract is that
    unmeasured perf is N/A-with-reason, not a bare self-report PASS.
    """
    try:
        perf_block = context.verification.get("performance")
        if not isinstance(perf_block, dict):
            perf_block = {}
        kw_ratio = perf_block.get("ratio")  # worker self-reported ratio (for delta)
        irm = _build_independent_re_measure(context.measured.perf, kw_ratio)
        perf_block["independent_re_measure"] = irm
        context.verification["performance"] = perf_block
        context.verification_path.write_text(json.dumps(context.verification, indent=2))
        # MFU 天花板机械注入(2026-07-01, main gate 要求闭合"真链入"): perf 写完即注入
        # mfu_ceiling 块 -> ko/fo 机械读取做 MFU-gated done. Fail-safe, 非写死 op_kind.
        _inject_mfu_ceiling_safe(
            context.verification_path,
            context.workspace,
            context.op,
            context.verification,
        )
    except Exception as _e:  # non-fatal: gate will flag missing irm → rollback
        context.report.summary = (
            context.report.summary or ""
        ) + f" [perf independent_re_measure write skipped: {_e!r}]"


def _persist_measured_perf(context: _PerfReMeasureContext) -> None:
    """Persist an independent measurement only when the runner supplied one."""
    if context.measured.perf and isinstance(context.measured.perf, dict):
        _write_perf_independent_re_measure(context)


def _load_verification(workspace: Path) -> tuple[Path, dict] | O5Report:
    """Load verification.json or return its existing fail-closed O5 result."""
    verification_path = workspace / "verification.json"
    if not verification_path.exists():
        return O5Report(
            verdict="RUNNER_FAILED",
            summary="verification.json missing — nothing to verify against",
            truth_source=expected_truth_source(workspace),
        )
    try:
        return verification_path, json.loads(verification_path.read_text())
    except Exception as error:
        return O5Report(
            verdict="RUNNER_FAILED",
            summary=f"verification.json malformed: {error}",
            truth_source=expected_truth_source(workspace),
        )


def _build_claim_report(workspace: Path, verification: dict) -> tuple[O5Report, dict[str, str]]:
    """Build the claimed-count subset and statuses used for reconciliation."""
    report = O5Report(verdict="VERIFIED", truth_source=expected_truth_source(workspace))
    precision = verification.get("precision", {}) or {}
    pass_statuses: dict[str, str] = {}
    for pass_name in COMPARABLE_PASSES:
        pass_result = precision.get(pass_name)
        if not isinstance(pass_result, dict):
            continue
        status = pass_result.get("status")
        if status in ("N/A", "SKIPPED", None):
            continue
        pass_statuses[pass_name] = status or ""
        fields = (
            INCLUSIVE_FIELDS
            if status in INCLUSIVE_STATUSES and "tier1_pass_inclusive" in pass_result
            else EXACT_FIELDS
        )
        report.claimed[pass_name] = {
            field_name: pass_result.get(field_name)
            for field_name in fields
            if field_name in pass_result
        }
    return report, pass_statuses


def _report_without_claims(report: O5Report) -> O5Report:
    """Preserve the distinct backward fail-closed and explicit N/A paths."""
    if report.truth_source == "backward_autograd":
        return O5Report(
            verdict="RUNNER_FAILED",
            summary=(
                "backward re-measure requires a non-empty pass_a claim; "
                "missing/N/A pass_a cannot be independently verified"
            ),
            truth_source=report.truth_source,
        )
    report.summary = "no verifiable passes claimed (all N/A); skipping re-measurement"
    return _apply_harness_pristine(report)


def _run_remeasurement(
    workspace: Path,
    op: str,
    lane: int,
    runner: Optional[Callable[[Path, str, int], MeasuredResult]],
    report: O5Report,
) -> MeasuredResult | O5Report:
    """Run the injected/default verifier and retain its existing error contract."""
    active_runner = runner or _default_runner
    try:
        measured = active_runner(workspace, op, lane)
    except Exception as error:
        return O5Report(
            verdict="RUNNER_FAILED",
            claimed=report.claimed,
            summary=f"runner raised: {error}",
            truth_source=report.truth_source,
        )
    if not measured.runner_error:
        return measured
    return O5Report(
        verdict="RUNNER_FAILED",
        claimed=report.claimed,
        summary=f"runner reported error: {measured.runner_error}",
        truth_source=report.truth_source,
        rollback_kind=measured.rollback_kind,
    )


def _record_measured_passes(report: O5Report, measured: MeasuredResult) -> None:
    """Copy runner count fields needed by the exact/inclusive compare paths."""
    all_fields = tuple(set(EXACT_FIELDS) | set(INCLUSIVE_FIELDS))
    for pass_name in COMPARABLE_PASSES:
        pass_result = getattr(measured, pass_name, None)
        if pass_result is not None:
            report.measured[pass_name] = {
                field_name: pass_result.get(field_name)
                for field_name in all_fields
                if field_name in pass_result
            }


def post_verify_for_finalize(
    workspace: Path,
    op: str,
    *,
    lane: int = 0,
    runner: Optional[Callable[[Path, str, int], MeasuredResult]] = None,
    skip: bool = False,
) -> O5Report:
    """Re-measure kernel + compare against claimed verification.json.

    Args:
        workspace: workspace dir
        op: op name
        runner: callable(workspace, op, lane) -> MeasuredResult. Default uses
            the SSH-based runner (requires A5 reachable). Tests pass mocks.
        skip: if True, return SKIPPED immediately (test-env shortcut).

    Returns O5Report with verdict:
        VERIFIED: measured matches claimed within EXACT_FIELDS for present passes
        MISMATCH: counts differ — caller should refuse finalize, route to worker
        RUNNER_FAILED: runner couldn't execute (A5 down, kernel build missing,
            etc.) — caller decides whether to allow finalize anyway
        SKIPPED: caller asked us not to run (test mode)
    """
    if skip:
        return O5Report(
            verdict="SKIPPED",
            summary="post-verify skipped by caller",
            truth_source=expected_truth_source(workspace),  # W6
        )

    loaded_verification = _load_verification(workspace)
    if isinstance(loaded_verification, O5Report):
        return loaded_verification
    verification_path, verification = loaded_verification
    rep, pass_statuses = _build_claim_report(workspace, verification)

    if not rep.claimed:
        return _report_without_claims(rep)

    measured = _run_remeasurement(workspace, op, lane, runner, rep)
    if isinstance(measured, O5Report):
        return measured

    _persist_measured_perf(
        _PerfReMeasureContext(verification, verification_path, measured, workspace, op, rep)
    )
    _record_measured_passes(rep, measured)
    det_policy, det_floor = _read_det_policy_and_floor(workspace, verification)
    _reconcile_counts(rep, pass_statuses, det_policy, det_floor)
    return _apply_harness_pristine(rep)


def _comparison_fields(claimed: dict, status: str) -> tuple[str, ...]:
    """Choose strict or inclusive fields from the worker-declared pass status."""
    if status in INCLUSIVE_STATUSES and "tier1_pass_inclusive" in claimed:
        return INCLUSIVE_FIELDS
    return EXACT_FIELDS


@dataclass
class _CountComparison:
    """One field-level O5 count comparison and its declared tolerance context."""

    pass_name: str
    field_name: str
    claimed_value: object
    measured_value: object
    status: str
    compare_fields: tuple[str, ...]
    det_floor: Optional[int]


def _record_count_comparison(report: O5Report, comparison: _CountComparison) -> None:
    """Record one exact or declared-best-effort count comparison."""
    if comparison.claimed_value is None or comparison.measured_value is None:
        return
    if comparison.det_floor is not None and comparison.field_name == _DET_TOLERANCE_FIELD:
        if comparison.measured_value < comparison.det_floor:
            report.mismatches.append(
                f"{comparison.pass_name}.{comparison.field_name}: "
                f"measured={comparison.measured_value} BELOW declared "
                f"det_floor={comparison.det_floor} (best_effort; "
                f"claimed={comparison.claimed_value}, status={comparison.status})"
            )
        elif comparison.claimed_value != comparison.measured_value:
            report.det_tolerated.append(
                f"{comparison.pass_name}.{comparison.field_name}: "
                f"claimed={comparison.claimed_value} measured={comparison.measured_value} "
                f"TOLERATED (>= det_floor={comparison.det_floor}, best_effort)"
            )
        return
    if comparison.claimed_value != comparison.measured_value:
        report.mismatches.append(
            f"{comparison.pass_name}.{comparison.field_name}: "
            f"claimed={comparison.claimed_value} measured={comparison.measured_value} "
            f"(status={comparison.status}, compare_set={comparison.compare_fields})"
        )


def _reconcile_claimed_pass(
    report: O5Report,
    pass_name: str,
    status: str,
    det_floor: Optional[int],
) -> None:
    """Reconcile one claimed pass while retaining the original message ordering."""
    if pass_name not in report.measured:
        report.mismatches.append(
            f"{pass_name}: claimed but not measured (runner didn't return it)"
        )
        return
    claimed = report.claimed[pass_name]
    measured = report.measured[pass_name]
    compare_fields = _comparison_fields(claimed, status)
    for field_name in compare_fields:
        _record_count_comparison(
            report,
            _CountComparison(
                pass_name,
                field_name,
                claimed.get(field_name),
                measured.get(field_name),
                status,
                compare_fields,
                det_floor,
            ),
        )


def _reconcile_counts(
    rep: O5Report,
    pass_statuses: dict,
    det_policy: Optional[str],
    det_floor: Optional[int],
) -> None:
    """Compare claimed vs measured counts per pass, set rep.mismatches /
    rep.det_tolerated, and stamp rep.verdict + rep.summary. Mutates `rep`.

    Exact equality is the default. The ONLY relaxation: for `tier1_pass` on a
    best_effort op with a declared det_floor, a measured count AT/ABOVE the
    floor is tolerated (recorded in rep.det_tolerated for honesty) and only a
    measured count BELOW the floor MISMATCHes. `total`, every other field, and
    det_policy in {required, n_a} (or best_effort with no floor) keep exact
    equality. This ONLY stops the stochastic count-MISMATCH→respawn churn — it
    never touches the determinism sub-block (policy_satisfied), so a tolerated
    op still finalizes as a PARTIAL_PERSIST recording the determinism failure.
    """
    active_floor = (
        det_floor if det_policy == _BEST_EFFORT_POLICY and det_floor is not None else None
    )
    for pass_name in rep.claimed:
        _reconcile_claimed_pass(
            rep,
            pass_name,
            pass_statuses.get(pass_name, ""),
            active_floor,
        )

    if rep.mismatches:
        rep.verdict = "MISMATCH"
        rep.summary = (
            f"O5 MISMATCH ({len(rep.mismatches)} discrepancies). "
            f"Worker claimed pass counts that don't match re-measurement. "
            f"Refuse finalize."
        )
    elif rep.det_tolerated:
        # Count reconciliation cleared UNDER the best_effort det_floor
        # tolerance: no respawn churn, but this is NOT a clean/deterministic
        # verdict. Determinism remains the separate gate's call
        # (policy_satisfied); the op finalizes as PARTIAL_PERSIST with the
        # determinism failure still recorded — never a laundered VERIFIED.
        rep.summary = (
            f"O5 count reconciliation within best_effort det_floor tolerance "
            f"({len(rep.det_tolerated)} tolerated tier1_pass discrepanc(ies); "
            f"det_floor={det_floor}). NOT a determinism pass — the determinism "
            f"gate (policy_satisfied) remains authoritative; op finalizes as "
            f"PARTIAL_PERSIST if determinism was not achieved."
        )
    else:
        rep.summary = (
            f"O5 VERIFIED: claim matches measurement for "
            f"{len(rep.claimed)} pass(es)."
        )


def _default_runner(workspace: Path, op: str, lane: int = 0) -> MeasuredResult:
    """Default runner: stub that returns runner_error. Real implementation
    comes in Step 2.1 (SSH integration). For now, the framework accepts
    pluggable runners — tests mock; real ops can wire up later.

    Until the real runner lands, orchestrator should call
    post_verify_for_finalize(skip=True) OR pass an explicit runner.
    """
    return MeasuredResult(
        runner_error=(
            "default runner not yet implemented (Step 2.1 — SSH-based "
            "verifier integration). Call with skip=True or pass an explicit "
            "runner."
        )
    )


def record_harness_state(workspace: Path, rep: O5Report) -> bool:
    """DEBT-213(b): persist the harness git state into verification.json so
    the fact outlives the process that observed it.

    Written for CLEAN as well as DIRTY on purpose: a recorded CLEAN is the
    evidence the assertion actually ran for this op. Absence of the block
    then means "no check", not "check passed" — a silent pass is as bad as
    no check at all.

    Additive + fail-open, matching the `_inject_*` verdict-stamp convention
    in finalize_dispatch.py: never raise, never block finalize on the stamp.
    Returns True if the block was written.
    """
    vp = workspace / "verification.json"
    try:
        import harness_pristine
        v = json.loads(vp.read_text())
        if not isinstance(v, dict):
            return False
        v["harness_pristine"] = {
            "state": rep.harness_git_state,
            "o5_verdict": rep.verdict,
            "harness_dirty": list(rep.harness_dirty),
            "scope": list(harness_pristine.HARNESS_PATHS),
            "sampled_at": "o5_post_verify",
            "note": (
                "DEBT-213(b): git state of the code that MEASURES this run, "
                "sampled when O5 re-measured it. DIRTY => o5_verdict is "
                "PROVISIONAL, never VERIFIED. UNKNOWN => not a git checkout "
                "(recorded, does not downgrade)."
            ),
        }
        vp.write_text(json.dumps(v, indent=2))
        return True
    except Exception:
        return False  # fail-open: the audit stamp never blocks finalize


def format_block_message(op: str, rep: O5Report) -> str:
    """Format a block message when O5 MISMATCH refuses finalize."""
    lines = [
        f"[orchestrator] O5 post-verify: {rep.summary}",
        f"[orchestrator] Op: {op}",
    ]
    if rep.claimed:
        lines.append("[orchestrator] Claimed (worker self-report):")
        for pname, fields in rep.claimed.items():
            lines.append(f"[orchestrator]   {pname}: {fields}")
    if rep.measured:
        lines.append("[orchestrator] Measured (re-run):")
        for pname, fields in rep.measured.items():
            lines.append(f"[orchestrator]   {pname}: {fields}")
    if rep.mismatches:
        lines.append(f"[orchestrator] Mismatches ({len(rep.mismatches)}):")
        lines.extend(f"[orchestrator]   - {m}" for m in rep.mismatches)
    return "\n".join(lines)
