# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""FSM condition-DSL evaluator (extracted from state_machine.py, DEBT-201).

This is the cohesive condition-evaluation cluster: `eval_condition` (the YAML
transition-guard DSL interpreter) plus its four private, exclusively-used
helpers (`_resolved_precision_status`, `_perf_ratio_and_threshold`,
`_trajectory_stats`, `_parse_worker_signal`).

Extracted VERBATIM (byte-identical AST source-segment relocation) to keep
state_machine.py <1000 lines while preserving behaviour. The parent
`state_machine` re-imports `eval_condition` (used by `next_state`) so
`import state_machine as sm; sm.eval_condition(...)` and
`from state_machine import eval_condition` keep working for every caller
(state_executor, resume, and the characterization/UT suite).

Dependency edge is strictly one-way: state_machine -> fsm_conditions. This
module depends only on the stdlib (re / pathlib / sys / typing.Any) plus the
orchestrator siblings `schema_norm` / `perf_gate`, imported inline exactly as
in the original (the `__file__`-relative sys.path bootstrap resolves the same
from workflow/ since both files share that parent). No import of state_machine
-> no cycle. Characterization-locked by test_eval_condition_characterization.py.
"""
from __future__ import annotations

import pathlib
import re  # noqa: F401  (used by the relocated helpers' regexes)
import sys  # noqa: F401
from collections.abc import Mapping
from typing import Any


def _handoff_match(handoff: str, arg: Any) -> bool:
    """Prefix-match a handoff, with exact token boundaries for terminal routes."""
    if not isinstance(arg, str) or not handoff.startswith(arg):
        return False
    if not arg.endswith((" done", " build-ready")):
        return True
    suffix = handoff[len(arg):]
    return not suffix or bool(re.match(r"^[^\w-]", suffix))


# ---------------------------------------------------------------------------
# Condition DSL evaluator
# ---------------------------------------------------------------------------
def _resolved_precision_status(prec: dict | None) -> str | None:
    """V3.8.3 (DEBT-073, 2026-05-04): roll up per-pass status to a single
    `precision.status` when worker writes only `precision.pass_a.status` +
    `precision.pass_b.status` (Path A / OL-68 case A pattern).

    Resolution table:
      - top-level prec.status set       → use as-is (worker explicit)
      - pass_a + pass_b both PASS       → PASS
      - pass_a in {PASS, N/A} + pass_b PASS → PASS  (Path A / OL-68 case A)
      - pass_b PASS_WITHIN_TOLERANCE    → PASS_WITHIN_TOLERANCE
      - any FAIL / PARTIAL              → FAIL or PARTIAL respectively
      - else                            → None (caller treats as "not done")

    The fallback ONLY fires when the explicit top-level field is missing.
    Workers that write top-level status keep current behavior.
    """
    if not prec:
        return None
    explicit = prec.get("status")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
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


def _perf_ratio_and_threshold(vj: dict | None) -> tuple[float | None, float]:
    if not vj:
        return None, 1.0  # parity default (owner-directed 2026-07-21, was 0.6)
    perf = vj.get("performance", {}) or {}
    ratio = (
        perf.get("ratio")
        or perf.get("ratio_mean")
        or perf.get("overall_speedup")
        or perf.get("sum_ratio")
        or perf.get("ratio_median")
        or perf.get("median_ratio")
        or perf.get("profiler_median_ratio")  # deprecated; bogus profiler artifact per OL-95
    )
    try:
        ratio_f = float(ratio) if ratio is not None else None
    except Exception:
        ratio_f = None
    try:
        thresh = float(perf.get("threshold", 1.0))  # parity default (owner-directed 2026-07-21, was 0.6)
    except Exception:
        thresh = 1.0
    return ratio_f, thresh


def _trajectory_stats(ws: pathlib.Path) -> dict:
    """Parse last 3 Phase-D iter DIAG sections from PROGRESS.md.
    Conservative default: returns empty dict if < 3 iters found."""
    prog = ws / "PROGRESS.md"
    if not prog.exists():
        return {}
    text = prog.read_text()
    sigs = re.findall(r"error_signature[:\s]+([^\n,]+)", text, re.IGNORECASE)
    mads = [float(m) for m in re.findall(r"max_abs_diff[:\s]+([0-9.eE+-]+)", text, re.IGNORECASE)]
    if len(sigs) < 3 and len(mads) < 3:
        return {}
    return {
        "sigs_last_3": sigs[-3:],
        "mads_last_3": mads[-3:] if len(mads) >= 3 else mads,
    }


def _parse_worker_signal(handoff: str) -> str:
    """Parse a worker handoff line into a structured signal enum.

    Added 2026-05-20 for the structural-rewrite escalation route. Existing handoffs
    map to legacy enum values; new `structural_rewrite_needed` sentinel
    surfaces from worker briefs (kw_brief.py — added in a follow-up PR).

    Recognized signals (extend here as new ones land):
      - "structural_rewrite_needed" — worker emits when 2-axis-scope rewrite
        is needed (algorithm + tile + sync); IL escalation entry signal per
        design doc §4.3.
      - "done" / "partial_persist" / "abort" — legacy worker outcomes
      - "unknown" — fallback when no recognized prefix found

    Pure-function; no I/O. Plugin method dispatch in eval_condition uses this
    so the plugin doesn't re-parse freeform text (per codex review MUST-FIX
    #3 — deterministic orchestrator-side resolution).

    gap(c) 2026-06-16 (CELU arch22→arch35 migration): workers commonly emit the
    verdict token in markdown bold — `→ orchestrator: **done**, precision ...`.
    Inline `**` then sits between the space and the token (`: **done`), so
    `endswith("done")` / `" done" in h` / `startswith("→ orchestrator: done")`
    all miss → "unknown" → spurious abort BEFORE O5 runs. Strip inline bold so
    verdict recognition is robust to bolded handoffs. Only `**` (paired bold) is
    stripped — single `*` is left intact (could be a legit `5*8` shape note).
    """
    h = handoff.strip().replace("**", "")
    if "structural_rewrite_needed" in h:
        return "structural_rewrite_needed"
    if h.endswith("done") or " done" in h.lower():
        return "done"
    if "PARTIAL_PERSIST" in h or "partial_persist" in h.lower():
        return "partial_persist"
    if h.endswith("abort") or "→ orchestrator: abort" in h.lower():
        return "abort"
    return "unknown"


def _orchestrator_module(name: str):
    """Lazily import an orchestrator-sibling module by name (schema_norm,
    ko_variant_ledger, ...). fsm_conditions lives in workflow/ but shares the
    src/scripts parent with orchestrator/; this mirrors the existing inline
    sys.path insert used by the pass_count + perf_gate primitives. Returns the
    module, or None if it cannot be imported (caller decides the fail-safe)."""
    import importlib
    import sys as _sys
    import pathlib as _pl
    _orch = _pl.Path(__file__).resolve().parent.parent / "orchestrator"
    if str(_orch) not in _sys.path:
        _sys.path.insert(0, str(_orch))
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _perf_at_or_above_finalize_floor(vj: dict | None, ws) -> bool:
    """iter_cap await_optimizer graceful-finalize fix (2026-07-24). Answers "is
    this perf already shippable?" — DISTINCT from verification_perf_below_threshold,
    which reads vj.perf.threshold, the PARITY optimization target (1.0× by owner
    default 2026-07-21). This resolves the FINALIZE floor (the shippable bar, e.g.
    0.6× set via --perf-threshold / a perf-gate profile) with the SAME resolver
    the finalize gate + iter_cap legitimate-exhaustion check use
    (schema_norm._resolve_perf_threshold), so a converged optimizer whose perf sits
    in the [finalize-floor, parity) band is recognized as shippable instead of
    looping await_optimizer to iter_cap. FACT-based (verification.json + resolved
    floor), never a handoff-string match. Fail-safe: on any resolution error the
    floor defaults to parity (1.0), so it reports above-floor only at true parity —
    never a false shippable."""
    ratio, _thresh = _perf_ratio_and_threshold(vj)
    if ratio is None:
        return False
    floor = 1.0
    _sn = _orchestrator_module("schema_norm")
    if _sn is not None and ws is not None:
        try:
            floor = getattr(_sn, "_resolve_perf_threshold")(ws, vj)
        except Exception:
            floor = 1.0
    return ratio >= floor


def _optimizer_kernel_converged(ws) -> bool:
    """iter_cap await_optimizer graceful-finalize fix (2026-07-24). True when the
    optimizer re-ran and produced a BYTE-IDENTICAL kernel across the last >=2
    consecutive spawns (0 net edits — nothing left to improve). Reads the
    deterministic orchestrator-written signature ledger via ko_variant_ledger
    (which reuses compute_kernel_md5, the SAME kernel-identity function the variant
    gate uses). Fail-CLOSED → not-converged, so an optimizer still making edits
    (kernel md5 changes each spawn) is never treated as converged."""
    if ws is None:
        return False
    _kvl = _orchestrator_module("ko_variant_ledger")
    if _kvl is None:
        return False
    try:
        return bool(_kvl.optimizer_kernel_converged(ws))
    except Exception:
        return False


def eval_condition(cond: Any, ctx: dict) -> bool:
    """Evaluate a condition DSL entry. Supports nested all_of/any_of/single-primitive.

    ctx keys:
        handoff: str — worker handoff line
        snapshot: dict — verification.json + related per-op state
        iter_counts: dict — per-counter spawn counts (probe, optimizer, etc.)
        ws: pathlib.Path — workspace dir
        sm: dict — loaded state machine YAML
        plugin: optional[object] — active scoped op-generation plugin. When
            present, the generic `plugin_method`
            primitive (added 2026-05-20) can dispatch to plugin methods.
            Backwards-compatible: callers
            that don't pass plugin (or pass None) cause `plugin_method`
            conditions to evaluate False — the legacy behavior pre-2026-05-20.
    """
    # Canonical form: { all_of: [ {primitive: arg}, ... ] } OR single {primitive: arg}
    if isinstance(cond, bool):
        return cond
    if not isinstance(cond, dict):
        return False

    if "all_of" in cond:
        return all(eval_condition(c, ctx) for c in cond["all_of"])
    if "any_of" in cond:
        return any(eval_condition(c, ctx) for c in cond["any_of"])

    # single primitive
    (kind, arg), = cond.items()
    handoff = ctx.get("handoff", "") or ""
    snap = ctx.get("snapshot", {}) or {}
    vj = snap.get("verification") or {}
    prec = vj.get("precision", {}) or {}
    det = vj.get("determinism", {}) or {}
    iter_counts = ctx.get("iter_counts", {}) or {}
    ws: pathlib.Path = ctx.get("ws")  # type: ignore
    sm = ctx.get("sm", {})

    # gap(c) 2026-06-16: strip inline markdown bold (`**`) so a bolded verdict
    # (`→ orchestrator: **done**`) is recognized by prefix/substring matching —
    # same robustness as _parse_worker_signal. Without this, handoff_match
    # `→ orchestrator: done` (prefix) misses `: **done`.
    handoff_norm = handoff.replace("**", "")
    if kind == "handoff_match":
        return _handoff_match(handoff_norm, arg)
    if kind == "handoff_contains":
        # Substring (not prefix). For verdict tokens that ride as a SUFFIX of the
        # canonical arrow handoff, e.g. `→ orchestrator: done — KO_PERF_PLATEAU`.
        # handoff_match (prefix-only) silently never matches those; this does.
        return arg in handoff_norm
    if kind == "reference_source_is":
        # Provider-owned routes use the complete durable binding rather than an
        # opgen mode or a handoff token.  The resolver validates the binding and
        # rejects missing, partial, or symlinked state.  A malformed provider
        # binding must not become ``False`` for an NPUBench workspace:
        # that would let the following legacy transition win.  The only
        # compatibility fallback is a resolver failure for a state that still
        # exposes a valid, non-NPUBench provider (the intentional legacy route).
        if not isinstance(arg, str):
            return False
        if ws is None:
            raise RuntimeError("reference source condition requires a workspace")
        reference_source = _orchestrator_module("reference_source")
        if reference_source is None:
            raise RuntimeError("reference source resolver is unavailable")
        try:
            return reference_source.resolve_reference_source(ws) == arg
        except Exception:
            state = reference_source.load_durable_state(ws)
            if not isinstance(state, Mapping):
                raise
            reference = state.get("reference")
            source = reference.get("source") if isinstance(reference, Mapping) else None
            valid_sources = getattr(reference_source, "VALID_REFERENCE_SOURCES", ())
            if (
                source in valid_sources
                and source != "npubench"
            ):
                return False
            raise
    if kind == "plugin_method":
        # Added 2026-05-20 as a generic paradigm-dispatch primitive.
        # Dispatches to a method on the active plugin (passed via
        # ctx["plugin"]) — keeps paradigm-specific decisions inside the
        # plugin layer rather than in YAML mode-uniform conditionals.
        #
        # `arg` is either:
        #   (a) a string method name; the
        #       method is called with resolved orchestrator-side values
        #       (op_class, op_complexity, worker_signal) passed positionally
        #       in that order. Resolution happens here from the snapshot.
        #   (b) a dict { method: "...", args: [...], forward_kwargs: [...] }
        #       — `args` optional (auto-resolves orchestrator-side values
        #       when absent, same as form (a)); `forward_kwargs` (added
        #       2026-05-27 for force-switch generalization) is a list of
        #       kwarg names that the evaluator pulls from
        #       `ctx["runtime_kwargs"]` and forwards as kwargs to the
        #       plugin method. The evaluator is feature-agnostic — adding a
        #       new force-X switch is purely YAML + runtime_kwargs wiring,
        #       no evaluator branch.
        #
        # Backwards-compatible: when ctx["plugin"] is None or absent (e.g.
        # legacy callers that don't pass plugin), the condition evaluates
        # False — pre-2026-05-20 behavior preserved for any YAML stanza
        # that doesn't reference plugin_method.
        plugin = ctx.get("plugin")
        if plugin is None:
            return False
        runtime_kwargs = ctx.get("runtime_kwargs") or {}
        forward_kwargs_names: list[str] = []

        # Helper for the default-resolved (op_class, op_complexity, worker_signal)
        # tuple — shared by string form and dict form when `args` is omitted.
        def _resolve_default_args():
            op_class = (snap.get("op_taxonomy") or {}).get("class", "unknown")
            op_complexity = (snap.get("op_taxonomy") or {}).get("complexity", "unknown")
            worker_signal = _parse_worker_signal(handoff)
            # Review P3 (2026-05-20): safe default — when orchestrator
            # CANNOT classify the op (op_taxonomy missing or detection failed),
            # do NOT trigger any paradigm-specific routing. Returning False
            # here (via sentinel) is more conservative than delegating to
            # plugin logic.
            if op_class == "unknown":
                return None
            return (op_class, op_complexity, worker_signal)

        if isinstance(arg, str):
            method_name = arg
            resolved = _resolve_default_args()
            if resolved is None:
                return False
            args = resolved
        elif isinstance(arg, dict):
            method_name = arg.get("method", "")
            if "args" in arg:
                args = tuple(arg.get("args", []))
            else:
                resolved = _resolve_default_args()
                if resolved is None:
                    return False
                args = resolved
            forward_kwargs_names = list(arg.get("forward_kwargs", []) or [])
        else:
            sys.stderr.write(
                f"state_machine: plugin_method arg must be str or dict, got {type(arg).__name__}\n"
            )
            return False
        method = getattr(plugin, method_name, None)
        if not callable(method):
            sys.stderr.write(
                f"state_machine: plugin_method '{method_name}' not callable on plugin "
                f"{getattr(plugin, 'name', plugin.__class__.__name__)!r}\n"
            )
            return False
        # Generic mechanism — pull declared forward_kwargs by name from
        # runtime_kwargs. Missing names default to absent (caller may have
        # a default on the plugin method's signature). No route-specific knowledge
        # in this evaluator.
        forwarded = {k: runtime_kwargs[k] for k in forward_kwargs_names if k in runtime_kwargs}
        try:
            # Pass workspace path as a kwarg so plugins that need scoped local
            # state can access it. Existing
            # plugin methods that don't accept **kwargs would break, so
            # we try with workspace= first and fall back to args-only if
            # the method signature doesn't accept it. This is the
            # "graceful kwarg extension" pattern.
            try:
                return bool(method(*args, workspace=ws, **forwarded))
            except TypeError as te:
                msg = str(te)
                if "workspace" in msg or "unexpected keyword" in msg:
                    return bool(method(*args, **forwarded))
                raise
        except Exception as e:
            sys.stderr.write(
                f"state_machine: plugin_method '{method_name}' raised "
                f"{type(e).__name__}: {e}\n"
            )
            return False
    if kind == "verification_precision_status_in":
        return _resolved_precision_status(prec) in arg
    if kind == "verification_precision_status_not_in":
        return _resolved_precision_status(prec) not in arg
    if kind == "verification_det_policy_satisfied":
        return bool(det.get("policy_satisfied")) == bool(arg)
    if kind == "pass_count_consistent":
        # P0xx (2026-05-06): YAML-evaluable wrapper around
        # schema_norm._check_pass_count_consistency. Returns True iff
        # `precision.status` is consistent with the per-pass counts. A PASS or
        # PASS_WITHIN_TOLERANCE claim requires every non-skipped, applicable
        # pass to report a positive total with every case passing. Partial,
        # failed, or absent status remains valid here because this primitive
        # only catches contradictions in PASS claims.
        #
        # Reported by DS agent 2026-05-06 30_NMS scenario: worker wrote
        # canonical Pass A 0/31 status=FAIL_EXPECTED, Pass B 31/31
        # status=PASS, top-level status=PASS. The await_worker
        # `done → finalize` transition matched on `handoff_match: done`
        # alone; P0ee schema_norm caught it at finalize-time but the
        # routing was already committed. Now use this primitive in YAML
        # to catch BEFORE the transition.
        try:
            import sys as _sys
            import pathlib as _pl
            _orch = _pl.Path(__file__).resolve().parent.parent / "orchestrator"
            if str(_orch) not in _sys.path:
                _sys.path.insert(0, str(_orch))
            import schema_norm as _sn  # type: ignore
            consistent = getattr(_sn, "_check_pass_count_consistency")(prec).get("consistent", True)
        except Exception:
            consistent = True  # fail open if schema_norm not importable
        return consistent == bool(arg)
    if kind == "verification_perf_below_threshold":
        ratio, thresh = _perf_ratio_and_threshold(vj)
        below = ratio is not None and ratio < thresh
        return below == bool(arg)
    if kind == "perf_ratio_below":
        # V3.7.11 (DEBT-077 Day 4 implementation): explicit threshold version
        # of verification_perf_below_threshold. Lets YAML transitions use a
        # different threshold than the canonical 0.6× (e.g. 0.5× for vendor-
        # strategy escalation). Returns True iff perf.ratio is present AND
        # < arg. arg is a float threshold.
        ratio, _thresh = _perf_ratio_and_threshold(vj)
        try:
            arg_f = float(arg)
        except (TypeError, ValueError):
            return False
        return ratio is not None and ratio < arg_f
    if kind == "verification_perf_at_or_above_finalize_floor":
        return _perf_at_or_above_finalize_floor(vj, ws) == bool(arg)
    if kind == "optimizer_kernel_converged":
        return _optimizer_kernel_converged(ws) == bool(arg)
    if kind in ("perf_gate_profile_allows",
                "perf_gate_profile_requires",
                "perf_gate_profile_measures"):
        # Phase B3 (PERF_GATE_PROFILE_DESIGN_2026_05_20 §8): typed primitives
        # that bridge YAML transitions to PerfGateProfile fields. resolve_profile
        # honors escalation_overrides[current_state] so per-state profile
        # switches happen without YAML having to know about the profiles.
        #
        # Mapping:
        #   perf_gate_profile_allows: <name>   → profile.allow_<name>
        #   perf_gate_profile_requires: <name> → profile.require_<name>
        #   perf_gate_profile_measures: <name> → profile.measure_<name>
        #                                        OR profile.include_<name>
        # Safe default if field missing: True (preserves legacy behavior — a
        # primitive with no matching field acts like an unconditional pass).
        try:
            import sys as _sys
            import pathlib as _pl
            _orch = _pl.Path(__file__).resolve().parent.parent / "orchestrator"
            if str(_orch) not in _sys.path:
                _sys.path.insert(0, str(_orch))
            from perf_gate import resolve_profile  # type: ignore
        except Exception:
            return True  # fail open if perf_gate not importable
        if ws is None:
            return True
        profile = resolve_profile(ws, current_state=ctx.get("current_state"))
        if kind == "perf_gate_profile_allows":
            field_name = f"allow_{arg}"
        elif kind == "perf_gate_profile_requires":
            field_name = f"require_{arg}"
        else:  # perf_gate_profile_measures
            # Two prefixes used in PerfGateProfile: measure_<name> for action
            # gates (measure_reference_perf) and include_<name> for content
            # gates (include_perf_in_brief). Try measure_ first, fall back to
            # include_.
            field_name = f"measure_{arg}"
            if not hasattr(profile, field_name):
                field_name = f"include_{arg}"
        return bool(getattr(profile, field_name, True))
    if kind == "iter_below_cap":
        counter_name = arg
        # Find cap from YAML — look up the state that uses this counter
        cap = None
        for s in sm.get("phase_o4_states", []):
            if s.get("iter_counter") == counter_name:
                cap = s.get("iter_cap", 999)
                break
        if cap is None:
            return True  # unknown cap → permissive
        return iter_counts.get(counter_name, 0) < int(cap)
    if kind == "path_exists":
        relpath = arg.replace("workspace/{op}/", "")
        return (ws / relpath).exists() if ws else False
    if kind == "file_absent":
        relpath = arg.replace("workspace/{op}/", "")
        if not ws:
            return True
        if relpath.endswith("*") or "*" in relpath:
            import fnmatch
            parent = ws / pathlib.Path(relpath).parent
            pat = pathlib.Path(relpath).name
            if not parent.exists():
                return True
            return not any(fnmatch.fnmatch(p.name, pat) for p in parent.iterdir())
        return not (ws / relpath).exists()
    if kind == "fused_analysis_contains_tuning_candidate":
        # V3.7.12: detect CB-N source-level bottleneck or RECOMMEND_KO routing hint.
        fa = ws / "fused_analysis.md" if ws else None
        if not fa or not fa.exists():
            return False
        text = fa.read_text(errors="replace")
        patterns = [
            r"\bRECOMMEND_KO\b",
            r"\bKIND2_DIRECTIVE\b",
            r"\btuning candidate\b",
            r"CB-\d+\s+\(highest impact\)",
            r"CB-\d+:\s+(?:vectorize|Reg-based|scalar|Launch fusion)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns) == bool(arg)
    if kind == "fused_analysis_lacks_strategy_citation":
        # V3.7.11: a strategy citation names the vendor's algorithmic approach,
        # e.g. "vendor uses MrgSort hierarchy", "CANN uses radix sort + ...".
        fa = ws / "fused_analysis.md" if ws else None
        if not fa or not fa.exists():
            return bool(arg)  # absent file ⇒ lacks citation
        text = fa.read_text(errors="replace").lower()
        markers = [
            "vendor uses", "cann uses", "vendor strategy", "algorithmic strategy",
            "vendor algorithm", "reference strategy", "cann strategy",
            "vendor reference algorithm",
        ]
        has = any(m in text for m in markers)
        lacks = not has
        return lacks == bool(arg)
    if kind == "optimization_log_lacks_strategy_citation":
        # V3.7.11 (DEBT-077 Day 4 implementation): mirror of
        # fused_analysis_lacks_strategy_citation but on optimization_log.md
        # (which ko/fo write). Used in await_fused_optimizer's vendor-
        # strategy researcher escalation gate. Citation = same markers as
        # fused-analysis variant.
        ol = ws / "optimization_log.md" if ws else None
        if not ol or not ol.exists():
            return bool(arg)  # absent ⇒ lacks
        text = ol.read_text(errors="replace").lower()
        markers = [
            "vendor uses", "cann uses", "vendor strategy", "algorithmic strategy",
            "vendor algorithm", "reference strategy", "cann strategy",
            "vendor reference algorithm",
        ]
        has = any(m in text for m in markers)
        lacks = not has
        return lacks == bool(arg)
    if kind == "analysis_md_contains_any":
        text = snap.get("analysis_md_lower", "")
        return any(kw.lower() in text for kw in arg)
    if kind == "probe_report_has_actionable_fix":
        return bool(snap.get("probe_report_has_actionable_fix")) == bool(arg)
    if kind == "probe_classification_in":
        # V3.3.1 (2026-04-25): match probe_report.md §Classification verdict
        # against an allowed-list. arg is a list of strings (e.g. ["requirement"]).
        return snap.get("probe_classification") in (arg or [])
    if kind == "probe_infra_block_streak_ge":
        # P0aay (2026-08-25): consecutive infra-blocked probe verdicts
        # (deferred / INFRA-BLOCKED untested-cluster). arg is an int threshold.
        try:
            return int(snap.get("probe_infra_block_streak") or 0) >= int(arg)
        except (TypeError, ValueError):
            return False
    if kind == "det_report_decision_in":
        return snap.get("det_report_decision") in arg
    if kind == "trajectory_sigs_stable":
        traj = _trajectory_stats(ws) if ws else {}
        sigs = traj.get("sigs_last_3", [])
        stable = len(sigs) == 3 and len(set(sigs)) == 1
        return stable == bool(arg)
    if kind == "trajectory_mad_tight":
        traj = _trajectory_stats(ws) if ws else {}
        mads = traj.get("mads_last_3", [])
        tight = bool(mads) and max(mads) < 1e-3
        return tight == bool(arg)
    if kind == "trajectory_mad_loose":
        traj = _trajectory_stats(ws) if ws else {}
        mads = traj.get("mads_last_3", [])
        loose = bool(mads) and min(mads) > 1
        return loose == bool(arg)
    if kind == "user_decision_target_in":
        # V3.8.5 / DEBT-077 #59: user_decision.md `next_state:` value must
        # be in the allowed-list. arg is list of valid state ids.
        target = snap.get("user_decision_target")
        return target in (arg or [])
    if kind == "always":
        return bool(arg)
    # Unknown primitive — fail closed (returns False, doesn't match)
    sys.stderr.write(f"state_machine: unknown condition primitive '{kind}'\n")
    return False
