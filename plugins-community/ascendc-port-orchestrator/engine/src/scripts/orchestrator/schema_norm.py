# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Schema normalization — auto-fix worker output to canonical form.

DEBT-074 (worker output schema non-conformance) made this necessary: workers
write `to_state: done`/`partial_persist`/`from`/`to` instead of canonical
YAML state names; verification.json uses `overall_speedup` instead of `ratio`.

CODEX REVIEW E.2 (2026-05-04): "schema normalization can hide real bugs.
Treat alias normalization as compatibility, not success. Log every
normalization, fail closed on state-changing aliases like
`partial_persist → finalize` unless accompanied by valid precision/perf
evidence."

Implementation:
- Two alias categories: SAFE (always rewrite, e.g. `from → from_state`) and
  TERMINAL (only rewrite with evidence verification, e.g. `partial_persist →
  finalize`)
- Every normalization logged to workspace/.schema_normalizations.log with
  reason + before/after snapshot
- Same alias appearing twice for same agent type after a session marked
  "post-prompt-update" is HARD ERROR
"""
from __future__ import annotations
import logging

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Alias maps
# ---------------------------------------------------------------------------

# SAFE aliases: pure key renames or value-form changes that don't affect
# routing decisions. Auto-fix without evidence check.
SAFE_KEY_ALIASES = {
    "state_transitions.jsonl": {
        # Workers write non-canonical key names
        "from": "from_state",
        "to": "to_state",
        # `verdict` is informational, route on `handoff`
    },
    "verification.json::performance": {
        # Workers wrote various perf-ratio field names
        "overall_speedup": "ratio",
        "ratio_mean": "ratio",            # common in DS/sonnet-written ops (13_Cat, 14_Split, 27_MMA)
        "median": "ratio_median",
        "profiler_median_ratio": "ratio",  # deprecated; bogus profiler artifact per OL-95
    },
}

# TERMINAL aliases: free-form handoff strings agents sometimes emit that
# the YAML doesn't accept directly. Require evidence to auto-fix; otherwise
# warn-only.
#
# P0dd (2026-05-05): `done` was previously aliased to `finalize`. After
# refactoring `done` to be a REAL terminal state (finalize is now an
# in-process pipeline; done is the truly-terminal post-pipeline state),
# `done` must NOT be aliased — log entries with to_state="done" are
# canonical. The alias caused false-positive evidence rejection on legitimate
# finalize→done transitions in workspaces that ran P0dd's pipeline.
TERMINAL_STATE_ALIASES = {
    "partial_persist": "finalize",     # Tier-2 evidence required
    "await_orchestrator": None,         # Drop entirely (orchestrator implicit)
    "await_orchestrator_decision": None,
}


@dataclass
class NormalizationEvent:
    ts: str
    workspace: str
    file: str
    field_path: str
    before: str
    after: Optional[str]      # None = entry dropped
    category: str             # "SAFE" | "TERMINAL_AUTO" | "TERMINAL_REJECT" | "DROP"
    reason: str
    evidence: dict = field(default_factory=dict)


@dataclass
class NormalizationReport:
    events: list[NormalizationEvent]
    rejected_terminal_aliases: list[NormalizationEvent]   # require user action
    files_modified: list[str]


class SchemaNormalizationError(RuntimeError):
    """Raised when a TERMINAL alias is rejected (requires manual resolution)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_workspace(workspace: Path, *, fail_strict: bool = True) -> NormalizationReport:
    """Run all auto-fix passes on a workspace.

    Args:
        workspace: workspace directory (e.g. workspace/22_Nonzero/)
        fail_strict: if True, raise SchemaNormalizationError on rejected
                     TERMINAL aliases. Default True (codex E.2 — fail-strict
                     for terminal transitions).

    Returns:
        NormalizationReport summarizing changes.

    Raises:
        SchemaNormalizationError if fail_strict=True and any TERMINAL alias
        was rejected (no valid evidence). Caller must resolve manually.
    """
    events: list[NormalizationEvent] = []
    rejected: list[NormalizationEvent] = []
    modified: list[str] = []

    # Pass 1: state_transitions.jsonl
    state_log = workspace / "state_transitions.jsonl"
    if state_log.exists():
        events_pass1, modified_pass1, rejected_pass1 = _normalize_state_log(workspace, state_log)
        events.extend(events_pass1)
        rejected.extend(rejected_pass1)
        if modified_pass1:
            modified.append(str(state_log.relative_to(workspace.parent)))

    # Pass 2: verification.json
    vj_path = workspace / "verification.json"
    if vj_path.exists():
        events_pass2, modified_pass2 = _normalize_verification_json(workspace, vj_path)
        events.extend(events_pass2)
        if modified_pass2:
            modified.append(str(vj_path.relative_to(workspace.parent)))

    # Persist event log
    if events:
        _append_event_log(workspace, events)

    # Fail-strict on rejected terminal aliases
    if rejected and fail_strict:
        report = NormalizationReport(events=events, rejected_terminal_aliases=rejected, files_modified=modified)
        msg = "TERMINAL alias REJECTED (requires manual evidence verification):\n"
        for r in rejected:
            msg += f"  - {r.file}: {r.before!r} → {r.after!r} ({r.reason})\n"
        raise SchemaNormalizationError(msg)

    return NormalizationReport(events=events, rejected_terminal_aliases=rejected, files_modified=modified)


def _normalize_state_log(
    workspace: Path, log_file: Path
) -> tuple[list[NormalizationEvent], bool, list[NormalizationEvent]]:
    """Apply alias rewrites to state_transitions.jsonl. Returns (events, modified, rejected)."""
    events: list[NormalizationEvent] = []
    rejected: list[NormalizationEvent] = []
    raw = log_file.read_text()
    if not raw.strip():
        return events, False, rejected

    new_lines = []
    modified = False
    key_aliases = SAFE_KEY_ALIASES.get("state_transitions.jsonl", {})

    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            entry = json.loads(line)
        except Exception:
            new_lines.append(line)  # malformed, leave alone
            continue

        # SAFE key aliases: from→from_state, to→to_state
        for old_key, new_key in key_aliases.items():
            if old_key in entry and new_key not in entry:
                events.append(NormalizationEvent(
                    ts=_now_z(),
                    workspace=str(workspace),
                    file=str(log_file.name) + f":{line_no}",
                    field_path=f"key:{old_key}",
                    before=old_key, after=new_key,
                    category="SAFE",
                    reason="key alias",
                ))
                entry[new_key] = entry.pop(old_key)
                modified = True

        # TERMINAL state aliases: done/partial_persist → finalize
        for state_field in ("from_state", "to_state"):
            v = entry.get(state_field)
            if isinstance(v, str) and v in TERMINAL_STATE_ALIASES:
                target = TERMINAL_STATE_ALIASES[v]
                if target is None:
                    # DROP this entry entirely (e.g. await_orchestrator is invalid)
                    events.append(NormalizationEvent(
                        ts=_now_z(),
                        workspace=str(workspace),
                        file=str(log_file.name) + f":{line_no}",
                        field_path=state_field,
                        before=v, after=None,
                        category="DROP",
                        reason=f"{v!r} not a valid YAML state; entry dropped",
                    ))
                    entry = None  # signal drop
                    modified = True
                    break
                # Verify evidence before applying TERMINAL_AUTO
                ev = _check_evidence_for_terminal(workspace, v, target, entry)
                if ev["passes"]:
                    events.append(NormalizationEvent(
                        ts=_now_z(),
                        workspace=str(workspace),
                        file=str(log_file.name) + f":{line_no}",
                        field_path=state_field,
                        before=v, after=target,
                        category="TERMINAL_AUTO",
                        reason=f"evidence verified: {ev['reason']}",
                        evidence=ev,
                    ))
                    entry[state_field] = target
                    modified = True
                else:
                    rejected.append(NormalizationEvent(
                        ts=_now_z(),
                        workspace=str(workspace),
                        file=str(log_file.name) + f":{line_no}",
                        field_path=state_field,
                        before=v, after=target,
                        category="TERMINAL_REJECT",
                        reason=f"evidence missing: {ev['reason']}",
                        evidence=ev,
                    ))
                    # Don't auto-fix; keep raw alias in log so caller can resolve

        if entry is not None:
            new_lines.append(json.dumps(entry))

    if modified:
        log_file.write_text("\n".join(new_lines) + "\n")

    return events, modified, rejected


def _normalize_verification_json(workspace: Path, vj_path: Path) -> tuple[list[NormalizationEvent], bool]:
    """Apply key aliases to verification.json. Returns (events, modified)."""
    events: list[NormalizationEvent] = []
    try:
        data = json.loads(vj_path.read_text())
    except Exception:
        return events, False

    perf = data.get("performance")
    if not isinstance(perf, dict):
        return events, False

    perf_aliases = SAFE_KEY_ALIASES.get("verification.json::performance", {})
    modified = False

    for old_key, new_key in perf_aliases.items():
        if old_key in perf and new_key not in perf:
            events.append(NormalizationEvent(
                ts=_now_z(),
                workspace=str(workspace),
                file="verification.json",
                field_path=f"performance.{old_key}",
                before=old_key, after=new_key,
                category="SAFE",
                reason="performance key alias",
            ))
            # Don't pop — keep both for backward compat
            perf[new_key] = perf[old_key]
            modified = True

    if modified:
        vj_path.write_text(json.dumps(data, indent=2))

    return events, modified


_SELF_INTROSPECT_HEADING_RE = re.compile(
    r"^##\s+Self-introspection\b",
    re.MULTILINE | re.IGNORECASE,
)
_SELF_INTROSPECT_REQUIRED_SUBSECTIONS = (
    "Pressure modes I felt",
    "Decisions I almost rationalized",
    "Verifications I might have skipped",
    "Confidence calibration",
)


def _check_self_introspection(workspace: Path) -> dict:
    """P0qq (2026-05-06): require ## Self-introspection block in PROGRESS.md
    for terminal handoffs. The block must list pressure modes felt, decisions
    almost rationalized, verifications possibly skipped, and confidence
    calibration. See briefs/_common.self_introspection_block().

    Returns {"passes": bool, "reason": str}.
    """
    progress = workspace / "PROGRESS.md"
    if not progress.exists():
        return {"passes": False, "reason": "PROGRESS.md missing — self-introspection cannot be verified (P0qq)"}
    try:
        text = progress.read_text(errors="replace")
    except Exception as e:
        return {"passes": False, "reason": f"PROGRESS.md unreadable: {e}"}
    if not _SELF_INTROSPECT_HEADING_RE.search(text):
        return {
            "passes": False,
            "reason": (
                "PROGRESS.md missing `## Self-introspection` section (P0qq) — "
                "agent must write in-context self-introspection BEFORE terminal "
                "handoff. See briefs/_common.self_introspection_block()."
            ),
        }
    missing_subs = [
        sub for sub in _SELF_INTROSPECT_REQUIRED_SUBSECTIONS
        if sub.lower() not in text.lower()
    ]
    if missing_subs:
        return {
            "passes": False,
            "reason": (
                f"## Self-introspection present but missing required subsections "
                f"(P0qq): {', '.join(missing_subs)}"
            ),
        }
    return {"passes": True, "reason": "self-introspection present"}


def _check_evidence_for_terminal(
    workspace: Path, alias: str, target: str, entry: dict
) -> dict:
    """For a TERMINAL alias (e.g. `done` or `partial_persist`), check if there's
    valid precision/perf evidence to safely route to `target`. Returns:
    {passes: bool, reason: str, ...}
    """
    # P0qq (2026-05-06): require `## Self-introspection` section in PROGRESS.md
    # before any terminal handoff. This is the in-context self-critic gate —
    # agents must reflect on their own reasoning trace BEFORE emitting a
    # terminal verdict, not as a separate spawn.
    introspect = _check_self_introspection(workspace)
    if not introspect["passes"]:
        return introspect

    vj_path = workspace / "verification.json"
    if not vj_path.exists():
        return {"passes": False, "reason": "verification.json missing"}

    try:
        vj = json.loads(vj_path.read_text())
    except Exception as e:
        return {"passes": False, "reason": f"verification.json malformed: {e}"}

    prec = vj.get("precision", {}) or {}
    perf = vj.get("performance", {}) or {}

    # `done` → `finalize`: requires precision PASS (or PASS_WITHIN_TOLERANCE)
    # AND (perf ≥ threshold OR perf check N/A)
    if alias == "done":
        prec_status = prec.get("status") or _resolve_pass_status(prec)
        if prec_status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return {"passes": False, "reason": f"precision.status={prec_status} (need PASS or PASS_WITHIN_TOLERANCE)"}

        # P0ee (2026-05-05): cross-check that the self-declared status is
        # actually backed by pass counts. Worker can write status="PASS"
        # while pass_a.tier1_pass=0/31 — must catch that here. The previous
        # gate trusted the literal status field. Reported by DS agent
        # 30_NMS scenario: kernel verified 0/31 but worker emitted
        # done handoff and orchestrator routed to finalize.
        consistency = _check_pass_count_consistency(prec)
        if not consistency["consistent"]:
            return {
                "passes": False,
                "reason": (
                    f"precision.status={prec_status} but pass counts contradict: "
                    f"{consistency['reason']}"
                ),
            }

        # P0uu (2026-05-06): perf gate
        # Worker must produce a numeric ratio OR explicit N/A justification.
        # The previous gate accepted `ratio: null` as "no perf data → OK"
        # which lets a worker that skipped performance.py claim done.
        # Now we require either:
        #   - numeric ratio ≥ 0.6× (V3.8.4 threshold), OR
        #   - explicit `performance.status == "N/A"` (Path A / OL-68 case A
        #     pattern — reference is unrunnable so perf is meaningless), OR
        #   - explicit `performance.skipped` truthy with `performance.skip_reason`
        #     (caller documented why perf wasn't measured)
        # Reported by DS agent 2026-05-06: workers were emitting done with
        # performance.ratio=null and orchestrator routed to finalize.
        ratio = (perf.get("ratio")
                 or perf.get("ratio_mean")
                 or perf.get("overall_speedup")
                 or perf.get("ratio_median"))
        if ratio is not None:
            try:
                # The threshold is plugin-aware so each scoped workflow owns
                # its optimization target without core-side mode branches.
                threshold = _resolve_perf_threshold(workspace, vj)
                if float(ratio) < threshold:
                    return {
                        "passes": False,
                        "reason": (
                            f"perf.ratio={ratio} < {threshold}× threshold "
                            f"(plugin-aware per V3.8.4)"
                        ),
                    }
            except (TypeError, ValueError):
                ratio = None  # malformed → treat as missing for gate logic

        if ratio is None:
            perf_status = (perf.get("status") or "").upper()
            if perf_status == "N/A":
                return {"passes": True, "reason": "perf.status=N/A (Path A / OL-68 case A)"}
            if perf.get("skipped") and perf.get("skip_reason"):
                return {
                    "passes": True,
                    "reason": f"perf skipped with reason: {perf.get('skip_reason')}",
                }
            # Phase B5 (PERF_GATE_PROFILE_DESIGN_2026_05_20): under
            # PRECISION_ONLY profile (--perf-threshold=0) the user has
            # explicitly opted out of perf measurement. profile.require_
            # ratio_in_verification=False means worker doesn't need to run
            # performance.py — accept ratio=null without N/A or skip_reason
            # justification. This closes the finalize-side hole left after
            # B4 YAML wire-up: B4 suppresses kw→ko routing, but without B5
            # the worker still gets rejected here when it emits ratio=null.
            try:
                from perf_gate import resolve_profile
                _profile = resolve_profile(workspace)
                if not _profile.require_ratio_in_verification:
                    return {
                        "passes": True,
                        "reason": (
                            f"perf.ratio=null accepted under "
                            f"perf_gate profile '{_profile.name}' "
                            f"(require_ratio_in_verification=False)"
                        ),
                    }
            except Exception:
                pass  # fall through to legacy reject
            # No numeric ratio + no N/A flag + no skip justification + no
            # profile opt-out → reject; worker must run performance.py
            # before claiming done. (P0uu)
            return {
                "passes": False,
                "reason": (
                    "performance.ratio missing/null AND no explicit N/A "
                    "(performance.status=='N/A') or skipped justification "
                    "(performance.skipped + performance.skip_reason). "
                    "Worker must run performance.py before claiming done — "
                    "V3.8.4 cannot route to ko without ratio data. (P0uu)"
                ),
            }
        # P0aay (2026-05-11): pre-handoff schema validation — check
        # knowledge_update.md structure AND pass_b.status BEFORE the worker
        # emits "done". Catches format gaps in the same spawn (avoids the
        # finalize→ROLLBACK→respawn cycle that wasted ~70min on 2_SwiGLU).
        ku_path = workspace / "knowledge_update.md"
        if not ku_path.exists():
            return {"passes": False, "reason": (
                "precision PASS but knowledge_update.md missing — "
                "every done handoff must include KB writeup. "
                "Write knowledge_update.md per Phase E template before "
                "emitting done. Required sections: ## Context, ## Findings, "
                "## KB-promotable patterns, ## Cited KB items, "
                "## Anti-patterns avoided. (P0aay gate)"
            )}
        try:
            ku_body = ku_path.read_text(errors="replace")
        except Exception as e:
            return {"passes": False, "reason": f"knowledge_update.md unreadable: {e}"}
        if len(ku_body) < 100:
            return {"passes": False, "reason": (
                f"knowledge_update.md is {len(ku_body)} bytes (<100) — "
                "writeup must be non-trivial. Expand per Phase E template. "
                "(P0aay gate)"
            )}
        if "## Findings" not in ku_body and "## findings" not in ku_body.lower():
            return {"passes": False, "reason": (
                "knowledge_update.md lacks `## Findings` section — "
                "file present but does not follow Phase E structure. "
                "Rewrite per template: ## Context / ## Findings / "
                "## KB-promotable patterns / ## Cited KB items / "
                "## Anti-patterns avoided. (P0aay gate)"
            )}

        # pass_b.status must be explicit. The finalize gate requires one of:
        # PASS / PASS_WITHIN_TOLERANCE / N/A / SKIPPED. Silent-skip rejected.
        pass_b = prec.get("pass_b", {}) or {}
        pb_status = pass_b.get("status")
        if pb_status not in ("PASS", "PASS_WITHIN_TOLERANCE", "N/A", "SKIPPED"):
            return {"passes": False, "reason": (
                f"precision.pass_b.status={pb_status!r} — must be explicit: "
                "PASS, PASS_WITHIN_TOLERANCE, N/A (with reason), or "
                "SKIPPED (with reason). Set this field in verification.json "
                "before emitting done. (P0aay gate)"
            )}
        if pb_status in ("N/A", "SKIPPED") and not pass_b.get("reason", "").strip():
            return {"passes": False, "reason": (
                f"precision.pass_b.status={pb_status} but no reason given — "
                "N/A or SKIPPED requires explicit reason. (P0aay gate)"
            )}

        return {"passes": True, "reason": f"precision={prec_status}, perf={ratio}"}

    # `partial_persist` → `finalize`: requires Tier-2-with-evidence
    # (precision.status=PARTIAL with explicit Tier-2 evidence pointers)
    if alias == "partial_persist":
        prec_status = prec.get("status")
        if prec_status != "PARTIAL":
            return {"passes": False, "reason": f"precision.status={prec_status} (need PARTIAL for partial_persist)"}
        # Evidence required: probe_report.md exists OR pass_b two-tier with PASS_T2
        probe = workspace / "probe_report.md"
        if probe.exists() and probe.stat().st_size > 100:
            return {"passes": True, "reason": "probe_report.md present"}
        pb = prec.get("pass_b", {})
        if isinstance(pb, dict) and pb.get("op_verdict") in ("PASS_T2", "PARTIAL"):
            return {"passes": True, "reason": f"two-tier evidence in pass_b: {pb.get('op_verdict')}"}
        return {"passes": False, "reason": "no probe_report.md and no Tier-2 evidence in pass_b"}

    return {"passes": False, "reason": f"alias {alias!r} requires manual review"}


def _resolve_perf_threshold(workspace: Path, vj: dict) -> float:
    """Resolve perf-ratio escalation threshold for the finalize gate.

    Resolution (first non-None hit wins):
    1. PerfGateProfile.finalize_threshold from workspace marker
       (Phase B2 wire-up of PERF_GATE_PROFILE_DESIGN — Zheng 2026-05-20).
       - PRECISION_ONLY profile → 0.0 (accept any ratio)
       - HERO_OP_STRICT → 0.9 (strict global)
       - custom_t<N> → N (synthesized)
       - DEFAULT → None (falls through to step 2)
    2. plugin.ko_escalation_threshold(op_class) — band-aware
    3. 1.0 — parity default (safe fallback when plugin missing; owner-directed
       2026-07-21, was 0.6 legacy AscendC default)

    The profile mechanism is opt-in via --perf-threshold CLI arg (Phase B1).
    Workspaces without the marker fall through to plugin band-aware unchanged.
    """
    # Step 1: PerfGateProfile override (B2 wire-up)
    try:
        from perf_gate import resolve_profile  # type: ignore
        profile = resolve_profile(workspace)
        if profile.finalize_threshold is not None:
            return float(profile.finalize_threshold)
    except Exception:
        pass  # profile module missing → fall through (safe)

    # Step 2: plugin band-aware (existing PR #21 behavior)
    from plugins import detect_plugin  # type: ignore
    plug = detect_plugin(workspace)
    if plug is None:
        return 1.0  # parity default (owner-directed 2026-07-21, was 0.6)

    op_class = _detect_op_class(workspace, vj)
    try:
        # task#31: pass workspace so the FA-class threshold can fall back to the
        # deterministic op-name backstop (is_attention_named) when the classifier
        # misses the `attention` tag. Additive kwarg — overrides that don't accept
        # it raise TypeError → retry positional-only.
        try:
            return float(plug.ko_escalation_threshold(op_class, workspace=workspace))
        except TypeError:
            return float(plug.ko_escalation_threshold(op_class))
    except Exception:
        return 1.0  # parity default (owner-directed 2026-07-21, was 0.6)


def _is_backward_named(op_name_upper: str) -> bool:
    """True when an op name signals a backward/gradient operator.

    Matches the conventional suffixes seen across the suite: `*_grad`
    (lightning_indexer_grad), `*Backward` (KvCacheUpdateWithRopeBackward),
    `*-Bwd` / `*_bwd` (AdaIN2D-Bwd). Used by _detect_op_class to append a
    GRADIENT taxonomy token so is_backward_class() fires.
    """
    if not op_name_upper:
        return False
    u = op_name_upper.replace("-", "_")
    return (
        u.endswith("GRAD") or u.endswith("_GRAD")
        or u.endswith("BWD") or u.endswith("_BWD")
        or "BACKWARD" in u
        or "_GRAD_" in u
    )


def _detect_op_class(workspace: Path, vj: dict) -> str:
    """Op-class detection + additive GRADIENT token for backward ops.

    Delegates to _detect_op_class_core for the base category, then appends
    a `GRADIENT` token when the op is backward-named (so is_backward_class()
    drives the C2 OL-200 brief block cross-mode). Additive: the base category
    is preserved, so substring predicates (is_fa_class / is_l4_fused) and any
    consumer of the base category are unaffected.
    """
    base = _detect_op_class_core(workspace, vj)
    base_u = base.upper()
    if "GRADIENT" in base_u or "BACKWARD" in base_u:
        return base  # already carries a backward token (classifier-emitted)
    op_name = (workspace.name if workspace is not None else "").upper()
    if _is_backward_named(op_name):
        return f"{base} GRADIENT"
    return base


def _detect_op_class_core(workspace: Path, vj: dict) -> str:
    """Best-effort op-class detection for Phase 5 perf threshold.

    Order:
    1. workspace/op_classification.json — explicit classifier output
    2. vj.precision.op_class / vj.op_class
    3. Heuristic from op name (REDUCTION / ELEMENTWISE / SCAN / LAYOUT_IDENTITY)
    4. "unknown" — plugin returns its safe default
    """
    op_cls_path = workspace / "op_classification.json"
    if op_cls_path.is_file():
        try:
            cls = json.loads(op_cls_path.read_text())
            tags = cls.get("op_class_tags") or cls.get("op_class") or []
            if isinstance(tags, list):
                return " ".join(str(t) for t in tags).upper()
            if isinstance(tags, str):
                return tags.upper()
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    for src in (vj.get("precision") or {}, vj):
        oc = src.get("op_class") if isinstance(src, dict) else None
        if oc:
            return str(oc).upper()
    # Heuristic from workspace name.
    op_name = workspace.name.upper()
    reduction_keywords = (
        "SUM", "MEAN", "NORM", "NLLLOSS", "VAR", "PROD", "MAX_REDUCE",
        "ARGMAX", "ARGMIN", "TOPK",
    )
    for keyword in reduction_keywords:
        if keyword in op_name:
            return "REDUCTION"
    if any(k in op_name for k in ("SORT", "CUMSUM", "CUMPROD", "SCAN")):
        return "SCAN"
    if any(k in op_name for k in ("PERMUTE", "TRANSPOSE", "RESHAPE", "VIEW", "CONTIGUOUS")):
        return "LAYOUT_IDENTITY"
    if any(k in op_name for k in ("ADAMW", "FUSED", "SWIGLU")):
        return "ELEMENTWISE_FUSED"
    elementwise_keywords = (
        "ABS", "GELU", "RELU", "ELU", "SIGMOID", "TANH", "ADD", "MUL",
        "SUB", "DIV", "NEG", "SQRT", "EXP", "LOG",
    )
    for keyword in elementwise_keywords:
        if keyword in op_name:
            return "ELEMENTWISE_SMALL"
    return "unknown"


# Public alias for _detect_op_class. The underscore-prefixed function
# is stable-private-to-module; the alias makes it discoverable + linkable from
# state_machine.py / plugin code without grepping for the underscore form.
detect_op_class = _detect_op_class


def _check_pass_count_consistency(prec: dict) -> dict:
    """P0ee (2026-05-05): when precision.status is PASS or PASS_WITHIN_TOLERANCE,
    verify pass_a / pass_b actually have non-empty passing counts.

    Rejects:
    - pass_a.tier1_pass < pass_a.total when pass_a.status is not N/A
    - pass_a.tier1_pass == 0 AND pass_a.total > 0 (worker claimed PASS but
      0 cases actually passed)
    - same for pass_b

    Allowed:
    - pass_a.status == "N/A" (the pass didn't run — e.g. Path A / OL-68 case A)
    - pass_a.tier1_pass == pass_a.total > 0 (genuine pass)
    - pass_a missing entirely (no counts to check)

    Returns {"consistent": bool, "reason": str}.
    """
    for pass_name in ("pass_a", "pass_b"):
        sub = prec.get(pass_name)
        if not isinstance(sub, dict):
            continue  # missing — skip
        sub_status = sub.get("status")
        if sub_status in ("N/A", "SKIPPED", None):
            continue  # genuinely didn't run

        # Resolve canonical (V3.8.x) and legacy (V3.7.x) field names.
        tier1 = sub.get("tier1_pass")
        if tier1 is None:
            tier1 = sub.get("n_pass")
        total = sub.get("total")
        if total is None:
            total = sub.get("n_total")

        # If both missing → can't verify, skip.
        if tier1 is None or total is None:
            continue

        try:
            t1 = int(tier1)
            tot = int(total)
        except (TypeError, ValueError):
            return {
                "consistent": False,
                "reason": f"{pass_name} counts unparseable: tier1_pass={tier1!r} total={total!r}",
            }

        if tot == 0:
            # No cases ran — can't claim PASS unless explicitly N/A above.
            return {
                "consistent": False,
                "reason": f"{pass_name}.total=0 with status={sub_status!r} (no cases ran but PASS claimed)",
            }

        if t1 == 0:
            return {
                "consistent": False,
                "reason": (
                    f"{pass_name}.tier1_pass=0/{tot} with status={sub_status!r} "
                    "(zero cases passed but PASS claimed)"
                ),
            }

        if t1 < tot:
            # Less-than-total passing is OK only if pass_name's own status
            # is PASS_WITHIN_TOLERANCE (acknowledges some cases were waived).
            # If sub_status is "PASS" but counts are short, reject.
            if sub_status == "PASS":
                return {
                    "consistent": False,
                    "reason": (
                        f"{pass_name}.tier1_pass={t1}/{tot} with status='PASS' "
                        f"(should be PASS_WITHIN_TOLERANCE or PARTIAL)"
                    ),
                }

    return {"consistent": True, "reason": "counts consistent with status"}


def _resolve_pass_status(prec: dict) -> Optional[str]:
    """Mirror state_machine._resolved_precision_status for offline use."""
    if isinstance(prec.get("status"), str):
        return prec["status"]
    pa = (prec.get("pass_a") or {}).get("status")
    pb = (prec.get("pass_b") or {}).get("status")
    if pa in ("PASS", "PASS_WITHIN_TOLERANCE", "N/A", None) and pb == "PASS":
        return "PASS"
    if pb == "PASS_WITHIN_TOLERANCE":
        return "PASS_WITHIN_TOLERANCE"
    if pa == "FAIL" or pb == "FAIL":
        return "FAIL"
    if pa == "PARTIAL" or pb == "PARTIAL":
        return "PARTIAL"
    return None


def _now_z() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _append_event_log(workspace: Path, events: list[NormalizationEvent]) -> None:
    """Append events to workspace/.schema_normalizations.log (JSONL)."""
    log = workspace / ".schema_normalizations.log"
    with open(log, "a") as f:
        for ev in events:
            f.write(json.dumps({
                "ts": ev.ts,
                "file": ev.file,
                "field_path": ev.field_path,
                "before": ev.before,
                "after": ev.after,
                "category": ev.category,
                "reason": ev.reason,
                **({"evidence": ev.evidence} if ev.evidence else {}),
            }) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="schema_norm.py — auto-fix worker output to canonical form")
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--no-fail-strict", action="store_true",
                    help="downgrade TERMINAL_REJECT to warning (default: fail-strict per codex E.2)")
    args = ap.parse_args()

    try:
        report = normalize_workspace(args.workspace, fail_strict=not args.no_fail_strict)
        print(json.dumps({
            "modified_files": report.files_modified,
            "n_events": len(report.events),
            "n_rejected": len(report.rejected_terminal_aliases),
            "events": [
                {"file": e.file, "field": e.field_path, "before": e.before, "after": e.after,
                 "category": e.category, "reason": e.reason}
                for e in report.events
            ],
        }, indent=2))
    except SchemaNormalizationError as e:
        print(f"REJECTED: {e}", file=sys.stderr)
        sys.exit(2)
