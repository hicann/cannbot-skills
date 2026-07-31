#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""finalize_checks_precision — precision / pass-count / performance-methodology finalize gate CHECK functions.

Behavior-neutral extraction from finalize_checks.py (DEBT-201 god-file
sub-split, 2026-07-06). Byte-identical function bodies; only relocated.
finalize_checks re-imports these (bottom import) so call sites + import
paths (`from finalize_checks import ...`) are unaffected."""
from __future__ import annotations
import ast as _ast
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path
from typing import Optional

from finalize_shared import (  # DEBT-201: shared pure leaves (breaks the finalize_pipeline cycle)
    _benchmark_case_count, _has_profiler_csv_method)


def _check_pass_a_coverage(workspace: Path, prec: dict) -> Optional[str]:
    """P0abd gate: Pass A `total` must match the benchmark JSON case count.

    Returns None if coverage OK (or unverifiable), error string otherwise.

    Worker may legitimately skip cases ONLY by listing them in
    precision.pass_a.skipped_cases with explicit per-case reason. Bare
    `total < benchmark_count` is silent coverage fraud.
    """
    bench_count = _benchmark_case_count(workspace)
    if bench_count is None:
        return None  # no benchmark JSON to compare against — can't enforce

    pass_a = prec.get("pass_a", {}) or {}
    total = pass_a.get("total")
    if not isinstance(total, int) or total <= 0:
        # If pass_a is N/A / SKIPPED with reason, allow (other gate checks
        # the reason). If just missing, the regular pass_a check will block.
        return None

    if total >= bench_count:
        return None  # coverage met or exceeded

    # Allow explicit skip-with-reason: pass_a.skipped_cases must list every
    # case that's missing from total, each with non-empty reason.
    expected_skipped = bench_count - total
    skipped_cases = pass_a.get("skipped_cases", []) or []
    if (
        isinstance(skipped_cases, list)
        and len(skipped_cases) >= expected_skipped
        and all(
            isinstance(s, dict) and s.get("case_idx") is not None
            and s.get("reason")
            for s in skipped_cases
        )
    ):
        return None  # explicit skip log present + populated → allowed

    return (
        f"P0abd coverage gate: precision.pass_a.total={total} but "
        f"benchmark <op>.json has {bench_count} cases. Worker ran only "
        f"{total}/{bench_count} of the spec — {expected_skipped} cases "
        f"silently skipped. CLAUDE.md feedback rule: input requirements "
        f"are immutable. To allow legitimate skips, populate "
        f"precision.pass_a.skipped_cases as a list of "
        f"{{case_idx: <int>, reason: '<explicit reason>'}} entries "
        f"(one per skipped case). Otherwise re-run worker with full "
        f"benchmark coverage."
    )


def _check_port_a3_pass_b_schema(workspace: Path, vj: dict) -> Optional[str]:
    """C-PORT-A3-PASS-B-SCHEMA (P96, 2026-05-15): port_a3 mode pass_b is
    degenerate by design — edge_dataset IS truth, pass_a IS the dispatch
    test. Worker writing run_pass_b.py OR claiming pass_b PASS without
    canonical N/A reason is mode-schema drift.

    Caught 2026-05-15: gather_elements_v2 kw-2 produced real PASS
    (8/8 T1, 12.91× perf) but also wrote run_pass_b.py that self-cited
    verification.json → P94 cycle gate trip. Root cause: shared Phase E
    checklist forced benchmark-style pass_b template into port_a3 mode.

    Returns None on pass, error string on fail.
    """
    mode = vj.get("mode") or vj.get("port_mode")
    if not mode or "port_a3" not in str(mode).lower():
        return None  # only applies to port_a3_to_a5 mode

    # Violation 1: run_pass_b.py exists at workspace root
    rpb = workspace / "run_pass_b.py"
    if rpb.is_file():
        return (
            "C-PORT-A3-PASS-B-SCHEMA (P96): port_a3_to_a5 mode op has "
            "workspace/run_pass_b.py — this file is a legacy generic "
            "verifier; port_a3 mode does NOT use it (edge_dataset IS "
            "truth, pass_a IS the runner-vs-edge_dataset comparison). "
            "Required: delete the file + set verification.json."
            "precision.pass_b.status='N/A' with canonical port_a3 reason. "
            "See kw_brief.py Phase D.6 (port_a3 block)."
        )

    # Violation 2: pass_b.status=PASS without canonical N/A reason
    pb = (vj.get("precision") or {}).get("pass_b") or {}
    pb_status = pb.get("status")
    if pb_status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return (
            f"C-PORT-A3-PASS-B-SCHEMA (P96): port_a3_to_a5 mode op has "
            f"verification.json.precision.pass_b.status='{pb_status}' but "
            f"pass_b is degenerate by design in port_a3 mode "
            f"(edge_dataset IS truth source, pass_a IS the dispatch test). "
            f"Required canonical N/A shape: "
            f'{{"status": "N/A", "reason": "port_a3_to_a5 mode: pass_b is '
            f'subsumed by pass_a — edge_dataset.pt[\'a3_outputs\'] IS the truth '
            f'source per ROADMAP §1.5 Path-B contract...", '
            f'"method": "n/a — port_a3 mode pass_b not applicable"}}'
        )

    # Violation 3: pass_b.method references the legacy generic schema
    pb_method = (pb.get("method") or "").lower()
    bench_method_signals = (
        "precision_eval_two_tier",
        "model.forward vs modelnew",
        "model_forward vs modelnew_forward",
        "tier1_pass + tier2_pass = total",
    )
    for signal in bench_method_signals:
        if signal in pb_method:
            return (
                f"C-PORT-A3-PASS-B-SCHEMA (P96): port_a3_to_a5 mode op has "
                f"verification.json.precision.pass_b.method='{pb_method[:80]}' "
                f"which references a legacy generic template ('{signal}'). "
                f"port_a3 mode pass_b must use the canonical N/A reason; "
                f"see kw_brief.py Phase D.6."
            )

    return None


def _check_pass_b_coverage(workspace: Path, vj: dict) -> Optional[str]:
    """P135.VC (2026-05-18 task #21): hard-block finalize when pass_b
    artifacts EXIST but verification.json.precision.pass_b is None/missing.

    A prior 6_Histc workspace had pass_b_runner.py + edge_dataset.pt with 138
    cases, but verification.json.pass_b was None — 138 edge cases NEVER
    RAN. orchestrator events showed phase_o5_runner: 'no Pass B verifier
    found' on early attempts (before kw-9 wrote pass_b_runner.py), then
    silent skip on re-finalize. Gate was too permissive.

    SCOPE: applies when the active scoped plugin requires pass_b. port_a3
    pass_b is degenerate by design (edge_dataset IS truth; pass_a runs against
    it) — handled by sibling _check_port_a3_pass_b_schema.

    Conditions to FIRE the rejection (all must hold):
    1. mode != port_a3_to_a5
    2. workspace/pass_b_runner.py exists (worker authored the runner)
    3. workspace/edge_dataset.pt exists (input cases to verify against)
    4. precision.pass_b is None / missing / has no usable status

    Conditions that BYPASS (gate doesn't fire):
    - precision.pass_b.status == 'PASS' / 'PASS_WITHIN_TOLERANCE' / 'PARTIAL' /
      'SKIPPED' / 'N/A' — any populated status (the worker DID make a
      coverage claim, even if it's N/A; that's the verdict to audit later)
    - precision.pass_b.tier1_pass / total present — counts are evidence
      that the runner ran
    - Missing pass_b_runner.py — nothing to invoke (different gap)
    - Missing edge_dataset.pt — no cases to run (different gap)

    Returns None on pass, error string on fail.
    """
    mode = vj.get("mode") or ""
    if mode == "port_a3_to_a5":
        return None  # port_a3 pass_b is degenerate; handled by separate gate

    pb_runner = workspace / "pass_b_runner.py"
    edge_ds = workspace / "edge_dataset.pt"
    if not pb_runner.is_file() or not edge_ds.is_file():
        return None  # no pass_b artifacts to enforce

    precision = vj.get("precision") or {}
    pb = precision.get("pass_b")
    # pb is None / missing / not-a-dict → fire
    # pb is dict with no usable signal → fire
    if pb is None:
        # missing entirely
        pass_b_signal = None
    elif not isinstance(pb, dict):
        pass_b_signal = None
    else:
        # Any of these counts as a usable signal
        pass_b_signal = (
            pb.get("status")
            or (pb.get("tier1_pass") is not None and "_counts_present")
            or pb.get("method")
        )

    if pass_b_signal:
        return None  # worker made a pass_b claim (good — let other gates audit)

    # Hard-block: artifacts exist but no pass_b claim
    return (
        f"P135.VC PASS_B_COVERAGE_SILENT_SKIP: "
        f"workspace/pass_b_runner.py + workspace/edge_dataset.pt exist "
        f"(both are non-empty pass_b artifacts the worker authored), "
        f"but verification.json.precision.pass_b is "
        f"{pb!r} (None / no status / no counts). The 138-case edge "
        f"dataset verifier was never invoked or its results never landed "
        f"in verification.json — silent coverage skip. Worker MUST: (a) "
        f"invoke pass_b_runner.py against edge_dataset.pt, (b) write the "
        f"per-case PASS/FAIL into precision.pass_b.status + tier1_pass + "
        f"total + (if applicable) failing_cases. If pass_b_runner.py "
        f"fundamentally cannot run (env issue, missing dependency), "
        f"DELETE it from workspace + write precision.pass_b.status='N/A' "
        f"+ reason citing the infeasibility — do NOT leave the runner "
        f"as an unused artifact while claiming PASS on pass_a only. "
        f"6_Histc 2026-05-18 incident: archive shipped with 15/15 pass_a "
        f"clean but 138 edge cases never tested; this gate prevents the "
        f"recurrence. Task #21 / P135.VC."
    )


def _check_pass_count_concrete(workspace: Path, vj: dict) -> Optional[str]:
    """PASS_COUNT gate (P0ee). When pass_b is PASS, tier1_pass + total MUST
    be concrete ints (written by an independent verifier, not a self-claim
    string). Extracted verbatim from the PASS branch."""
    prec = vj.get("precision", {}) or {}
    pass_b = prec.get("pass_b", {}) or {}
    pb_status = pass_b.get("status")
    if pb_status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        if not isinstance(pass_b.get("tier1_pass"), int) or not isinstance(pass_b.get("total"), int):
            return (
                f"precision.pass_b.status={pb_status} but tier1_pass/"
                "total counts missing — PASS requires "
                "concrete denominator (an independent verifier wrote "
                "the numbers, not a self-claim string)."
            )
    return None


def _check_perf_methodology(workspace: Path, vj: dict) -> Optional[str]:
    """P97 (2026-05-16) — cross-artifact methodology symmetry check.

    Catches the case where A3 baseline and A5 candidate use DIFFERENT
    measurement methods. Outcome-based reward-hacking: worker accepts an
    existing A3 baseline (torch_npu Python dispatch) + measures A5 with
    C++ aclnn-direct → ratio inflated by dispatch-overhead delta, NOT
    real hardware speedup. User catch 2026-05-16 02:30Z: "Worker 没作弊
    我认为这就是作弊, 因为可以让数值高, 生成更快".

    Per PR #103 `ascendc-operator-performance-eval` skill, both baseline
    and candidate MUST be measured by `torch_npu.profiler` with
    `warmup=5, active=5`, on NPU.

    Gate fires when precision.status is PASS / PASS_WITHIN_TOLERANCE
    (meaning perf claim is being made):
    - performance.method (or method_note) MUST mention "torch_npu.profiler"
    - method MUST mention "warmup=5" AND "active=5"
    - performance.ratio MUST be numeric
    - status outside the measured precision states makes no perf claim here;
      finalize eligibility rejects unsupported terminal states
    """
    prec = vj.get("precision", {}) or {}
    status = prec.get("status")
    if status not in ("PASS", "PASS_WITHIN_TOLERANCE", "PARTIAL"):
        return None  # no perf claim being made

    perf = vj.get("performance", {}) or {}
    perf_status = perf.get("status")

    # P146 gate extension (2026-05-17): in port_a3 mode with precision=PASS,
    # ALL retraction-equivalent perf statuses (N/A, SKIPPED, NA, None, AND
    # NOT_VERIFIED_SAME_METHOD) require per-option infeasibility evidence.
    # Without this, worker can dodge P143 by picking "N/A" instead of
    # NOT_VERIFIED_SAME_METHOD — clipped_swiglu 2026-05-17T23:13Z incident.
    # The gate applies uniformly: any "I'm not making a perf claim because
    # I can't" requires evidence of why I can't.
    is_port_a3_local = (vj.get("mode") == "port_a3_to_a5")
    retraction_equivalent = perf_status in ("N/A", "SKIPPED", "NA", None, "NOT_VERIFIED_SAME_METHOD")

    if retraction_equivalent and not is_port_a3_local:
        return None  # non-port-a3 compatibility modes: N/A still permissive
    if retraction_equivalent and is_port_a3_local:
        # port_a3 + precision=PASS + perf=retraction-equivalent → require evidence
        if True:
            retraction = perf.get("retraction") or {}
            reason_raw = retraction.get("reason") or perf.get("reason") or ""
            reason_low = reason_raw.lower()
            has_opt1_clause = (
                "option_1_infeasible_because" in reason_low
                or "option 1 infeasible" in reason_low
                or (
                    "aclrtevent" in reason_low
                    and (
                        "infeasible" in reason_low
                        or "cannot" in reason_low
                        or "no stream" in reason_low
                    )
                )
            )
            has_opt2_clause = (
                "option_2_infeasible_because" in reason_low
                or "option 2 infeasible" in reason_low
                or (
                    "perf_counter" in reason_low
                    and (
                        "infeasible" in reason_low
                        or "cannot wrap" in reason_low
                        or "no python entry" in reason_low
                    )
                )
            )
            if not (has_opt1_clause and has_opt2_clause):
                return (
                    f"P146 PERF_METHODOLOGY_LAZY_ESCAPE (port_a3): "
                    f"performance.status={perf_status!r} with precision.status=PASS "
                    "requires retraction.reason to cite per-option infeasibility "
                    "evidence. MUST include: (a) `option_1_infeasible_because` "
                    "explaining why aclrtEvent on A3 + torch.npu.Event on A5 "
                    "can't be applied (cite CANN-doc / aclnn header line) and "
                    "(b) `option_2_infeasible_because` explaining why "
                    "perf_counter wrap of each side's natural call shape "
                    "can't be done (cite code/runner constraint). 'API differs' "
                    "or 'A5 doesn't ship aclnn' is NOT evidence — that's "
                    "literally the case Option 1 is designed for. See OL-163 + "
                    "docs/design/PORT_A3_PERF_METHODOLOGY.md. Worker MAY NOT "
                    "dodge P143 by picking N/A / SKIPPED / NA / NOT_VERIFIED_SAME_METHOD "
                    "interchangeably — all retraction-equivalent statuses require "
                    "the same per-option evidence. clipped_swiglu 2026-05-17T23:13Z "
                    "incident: worker shipped N/A + empty method to bypass P143's "
                    "NOT_VERIFIED_SAME_METHOD-only check. "
                    f"reason={reason_raw[:300]!r}"
                )
        return None  # accepted retraction with evidence

    # P141 PORT_A3 PERF METHODOLOGY GATE (2026-05-17)
    # Independent audit confirmed port_a3 archives with `perf_counter` + sync
    # wrapping around `torch_npu.npu_<op>` (a3) vs `ACLRT_LAUNCH_KERNEL` (a5)
    # produce ratios that conflate aclnn-host-overhead with kernel speedup.
    # 4.5x (clipped_swiglu) + 3.77x (expand_into_jagged_permute) retracted
    # 2026-05-17. For port_a3 mode, perf claim requires method to declare
    # ONE OF:
    #   Option 1 device-event: "aclrtEventElapsedTime" (A3) + "torch.npu.Event" (A5)
    #   Option 2 same-callable: declare "same_wrapper" or "symmetric=true" + name both wrappers
    #   Option 3 retract: status=NOT_VERIFIED_SAME_METHOD (handled above)
    is_port_a3 = (vj.get("mode") == "port_a3_to_a5")
    if is_port_a3:
        method_low = (perf.get("method") or "").lower()
        # P135.PR (2026-05-18): profiler-CSV is now PRIMARY accepted method.
        # `_measure_single_with_profiler` excludes
        # ALL host runtime + handles foreach-list ops correctly (no stream
        # stall leakage that breaks torch.npu.Event wrap on V220).
        # See `_has_profiler_csv_method` for accepted CSV tokens (DEBT-128
        # extended to `kernel_details` for aog-perf-eval custom-AscendC).
        has_profiler_csv = _has_profiler_csv_method(method_low)
        has_event_pair = ("aclrtevent" in method_low or "device-event" in method_low) and (
            "torch.npu.event" in method_low or "torch_npu.event" in method_low
            or "elapsed_time" in method_low
        )
        has_symmetric_decl = (
            "same_wrapper" in method_low or "symmetric=true" in method_low
            or "method_symmetric" in method_low
        )
        # Common-but-WRONG signature: perf_counter (Python wall-clock) on both
        # sides + non-equivalent callables (a3=torch_npu wrapper, a5=pybind+ACLRT_LAUNCH_KERNEL).
        # The user-visible symptom is the retracted 4.5×/3.77× pattern.
        has_perf_counter = "perf_counter" in method_low
        a3_aclnn_signal = False
        for signal in (
            "torch_npu.npu_", "aclnn", "torch_npu wrapper", "npu_clipped", "npu_fatrelu",
        ):
            if signal in method_low:
                a3_aclnn_signal = True
                break
        a5_macro_signal = False
        for signal in (
            "aclrt_launch_kernel", "aclrtlaunch", "macro launch", "pybind",
        ):
            if signal in method_low:
                a5_macro_signal = True
                break
        if (
            has_perf_counter
            and a3_aclnn_signal
            and a5_macro_signal
            and not (has_profiler_csv or has_event_pair or has_symmetric_decl)
        ):
            return (
                f"P141 PERF_METHODOLOGY_ASYMMETRY (port_a3): performance.method "
                f"declares `perf_counter` wrap around a3=aclnn-pipeline + "
                f"a5=ACLRT_LAUNCH_KERNEL macro. These are NOT byte-equivalent "
                f"callables — aclnn host overhead 30-80µs inflates the A3 "
                f"side. Per P141 contract, perf claim requires ONE OF: "
                f"(1) device-event timing (aclrtEvent on A3 + torch.npu.Event "
                f"on A5), (2) symmetric callable wrapper declared via "
                f"`method_symmetric=true`, or (3) status=NOT_VERIFIED_SAME_METHOD "
                f"with ratio removed. See P141 retraction precedent: "
                f"clipped_swiglu 4.5× + expand_into_jagged_permute 3.77×, "
                f"both retracted 2026-05-17. method={method_low[:300]!r}"
            )
        # If port_a3 + method missing entirely + ratio present, ALSO reject —
        # bare ratio without method declaration cannot be audited.
        if perf.get("ratio") is not None and not method_low:
            return (
                f"P141 PERF_METHODOLOGY_ASYMMETRY (port_a3): performance.ratio "
                f"is set ({perf.get('ratio')}) but performance.method is empty. "
                f"port_a3 archives MUST declare method including both A3 + A5 "
                f"timed-window contents. Bare ratio = unauditable = retraction-eligible."
            )

        # P135.DS (2026-05-18, Event-vs-perf_counter direction sanity check):
        # If worker declares Option 1 device-event AND populates Option 2
        # wrapper-inclusive cross-check, the Option 1 ratio MUST be ≤
        # Option 2 ratio (modulo 10% noise). Reason: Option 1 excludes
        # host overhead on both sides; A3 has heavier host overhead than
        # A5 (aclnn dispatch ~30-80µs vs pybind ~5-10µs); so excluding
        # host LOWERS the A3 number more than the A5 number → Option 1
        # ratio (a3_event/a5_event) should be LOWER than Option 2 ratio
        # (a3_wallclock/a5_wallclock).
        # If Option 1 ratio > Option 2 ratio × 1.10, Event is INFLATING
        # numbers instead of excluding host. This catches the V220
        # foreach-list stream-stall leakage observed on foreach_sqrt
        # 2026-05-18 (Option 1 Event ratio 2.70× > Option 2 perf_counter
        # ratio 2.42× — wrong direction, indicates Event captured stream
        # stall as device time).
        if has_event_pair and not has_profiler_csv:
            opt2 = perf.get("option2_wrapper_inclusive") or {}
            opt1_ratio = perf.get("ratio")
            opt2_ratio = opt2.get("ratio")
            if (
                isinstance(opt1_ratio, (int, float))
                and isinstance(opt2_ratio, (int, float))
                and opt2_ratio > 0
            ):
                inflation = opt1_ratio / opt2_ratio
                if inflation > 1.10:
                    return (
                        f"P135.DS PERF_METHODOLOGY_EVENT_LEAKAGE (port_a3): "
                        f"performance declares Option 1 device-event timing with "
                        f"ratio={opt1_ratio} but option2_wrapper_inclusive shows "
                        f"perf_counter ratio={opt2_ratio} — Option 1 / Option 2 "
                        f"= {inflation:.3f} > 1.10. Device-event should EXCLUDE "
                        f"host overhead → ratio should be ≤ wall-clock ratio "
                        f"(A3 host overhead heavier than A5; excluding it lowers "
                        f"A3 number more than A5 number, lowering ratio). "
                        f"Option 1 > Option 2 indicates Event captured stream "
                        f"stall as device time (V220 foreach-list regime "
                        f"leakage — empirically confirmed foreach_sqrt 2026-05-18: "
                        f"A3 Event 0.081-0.150ms > A3 perf_counter 0.048-0.068ms, "
                        f"wrong direction). Switch to Option 1 profiler-CSV "
                        f"(torch_npu.profiler operator_details.csv) which avoids "
                        f"the regime split, OR retract Option 1 and ship only "
                        f"Option 2 wrapper-inclusive (relabel as primary)."
                    )

    method = method_low if is_port_a3 else (perf.get("method") or perf.get("method_note") or "").lower()
    # P97 transition rule (2026-05-16): only REJECT when method is
    # explicitly wrong (e.g., std::chrono / aclrtEvent without
    # torch_npu.profiler). Missing method = legacy fixture / pre-P97
    # archive, treat as not-yet-enforced (warn-via-comment only) so
    # existing test fixtures + legacy archives don't all break at once.
    # Future strictening: once all archives + tests declare method,
    # tighten to reject empty too.
    if not method:
        return None  # transition allowance
    # Definitely-asymmetric measurement: contains std::chrono / aclrtEvent
    # but NOT torch_npu.profiler. This is the exact 2026-05-16 incident.
    explicit_asymmetric_signals = (
        "std::chrono", "high_resolution_clock", "aclrtevent",
        "aclrtsynchronizestream",
    )
    has_asymmetric = any(s in method for s in explicit_asymmetric_signals)
    has_profiler = "torch_npu.profiler" in method
    # P141 carve-out: port_a3 Option 1 device-event pattern uses
    # aclrtEvent (A3) + torch.npu.Event (A5) symmetrically. This is the
    # CORRECT cross-arch pattern (PR #103's torch_npu.profiler is a same-side
    # only — it can't run on A3 hardware to time aclnn). Allow it when
    # method explicitly declares both event types + symmetric intent.
    p141_device_event_symmetric = (
        is_port_a3 and "aclrtevent" in method
        and ("torch.npu.event" in method or "torch_npu.event" in method
             or "elapsed_time" in method)
        and ("method_symmetric=true" in method or "symmetric=true" in method
             or "device-event" in method)
    )
    if has_asymmetric and not has_profiler and not p141_device_event_symmetric:
        return (
            f"P97 PERF_METHODOLOGY_ASYMMETRY: performance.method uses C++/"
            f"aclrtEvent timing without `torch_npu.profiler` declared. "
            f"Per PR #103 skill, BOTH baseline + candidate must use same "
            f"`torch_npu.profiler` (warmup=5, active=5). C++ timing on "
            f"candidate + torch_npu on baseline = dispatch-overhead delta "
            f"inflates ratio = outcome-based cheating. Method: {method[:200]!r}"
        )
    # If torch_npu.profiler declared, also require schedule
    if has_profiler:
        compact = method.replace(" ", "")
        if "warmup=5" not in compact or "active=5" not in compact:
            return (
                f"P97 PERF_METHODOLOGY_ASYMMETRY: `torch_npu.profiler` "
                f"declared but schedule values not `warmup=5, active=5` "
                f"(PR #103 skill mandate). Method: {method[:200]!r}"
            )
    return None


def _check_methodology_declaration(workspace: Path, vj: dict) -> Optional[str]:
    """P0ee (2026-05-26, ROADMAP §86) — methodology declaration gate.

    Default-deny: any perf ratio > 1.0× MUST positively declare symmetry
    via one of the canonical positive signals:
      - 'torch_npu.profiler' + (any `_PROFILER_CSV_TOKENS`) + (any `_DEVICE_DURATION_TOKENS`)
        (see `_has_profiler_csv_method`; DEBT-128 extended tokens to include
        `kernel_details` for aog-perf-eval custom-AscendC profiling)
      - 'aclrtEvent' + ('torch.npu.Event' OR 'torch_npu.Event' OR 'elapsed_time')
      - 'same_wrapper' / 'symmetric=true' / 'method_symmetric'

    Complements P97 + P141 which catch SPECIFIC asymmetric patterns
    (perf_counter + aclnn-vs-ACLRT_LAUNCH). P0ee is strict default: silent
    or unmarked ratio > 1.0 is unauditable, retraction-eligible.

    Fires only when:
      - precision.status PASS / PASS_WITHIN_TOLERANCE / PARTIAL (perf claim active)
      - performance.status not in retraction-equivalent set (handled by P146)
      - performance.ratio_median or .ratio is numeric AND > 1.0

    Both port_a3 and backward are subject to the same default-deny — speedup claims uniformly require
    symmetric-method declaration.

    Skipped (returns None) when:
      - precision.status not making a claim (FAIL / PARTIAL_PERSIST / N/A)
      - perf.status in retraction-equivalent set (per P146 handling)
      - ratio_median/ratio ≤ 1.0× (no speedup claim → no symmetry burden)
      - explicit positive signal present in method
    """
    prec = vj.get("precision", {}) or {}
    if prec.get("status") not in ("PASS", "PASS_WITHIN_TOLERANCE", "PARTIAL"):
        return None

    perf = vj.get("performance", {}) or {}
    perf_status = perf.get("status")
    if perf_status in ("N/A", "SKIPPED", "NA", None, "NOT_VERIFIED_SAME_METHOD"):
        return None  # P146 owns retraction-equivalent validation

    # Numeric ratio extraction — try ratio_median first, then ratio
    ratio = perf.get("ratio_median")
    if ratio is None:
        ratio = perf.get("ratio")
    if not isinstance(ratio, (int, float)):
        return None  # no numeric speedup claim
    if ratio <= 1.0:
        return None  # no speedup claim, no symmetry burden

    method_low = (perf.get("method") or "").lower()
    if not method_low:
        return (
            f"P0ee METHODOLOGY_DECLARATION: performance.ratio={ratio} > 1.0× "
            f"claim WITHOUT performance.method declaration. Default-deny: "
            f"any speedup claim MUST explicitly declare symmetric measurement "
            f"via ONE OF:\n"
            f"  (a) `torch_npu.profiler` + (`operator_details` OR "
            f"`kernel_details`) + `device_self_duration` "
            f"(PR #103 skill mandate; `kernel_details` accepted DEBT-128 "
            f"for aog-perf-eval custom-AscendC profiling)\n"
            f"  (b) `aclrtEvent` + `torch.npu.Event` / `elapsed_time` device-event pair\n"
            f"  (c) `same_wrapper` / `symmetric=true` / `method_symmetric` explicit declaration\n"
            f"Bare ratio = unauditable = retraction-eligible. See P97/P141 "
            f"precedents (clipped_swiglu 4.5× / expand_into_jagged_permute 3.77× both retracted)."
        )

    # Positive signals — any one suffices.
    # See `_has_profiler_csv_method` for accepted CSV tokens. Token set
    # is single-source-of-truth at `_PROFILER_CSV_TOKENS` — extending a
    # new profiler-CSV source = 1-line append, NOT N edits across sites.
    has_profiler_csv = _has_profiler_csv_method(method_low)
    has_event_pair = ("aclrtevent" in method_low or "device-event" in method_low) and (
        "torch.npu.event" in method_low or "torch_npu.event" in method_low
        or "elapsed_time" in method_low
    )
    has_symmetric_decl = (
        "same_wrapper" in method_low
        or "symmetric=true" in method_low
        or "method_symmetric" in method_low
    )

    if has_profiler_csv or has_event_pair or has_symmetric_decl:
        return None  # at least one positive declaration present

    return (
        f"P0ee METHODOLOGY_DECLARATION: performance.ratio={ratio} > 1.0× claim "
        f"WITHOUT positive symmetric-method declaration. method={method_low[:200]!r} "
        f"contains none of: (a) torch_npu.profiler (operator_details OR kernel_details) device_self_duration, "
        f"(b) aclrtEvent + torch.npu.Event device-event pair, "
        f"(c) same_wrapper / symmetric=true / method_symmetric explicit flag. "
        f"Default-deny — speedup claim unauditable without one of these signals. "
        f"P0ee is the strict default complementing P97/P141 (which catch SPECIFIC "
        f"asymmetric patterns); P0ee catches the silent / unmarked case where "
        f"worker omitted methodology declaration entirely."
    )
