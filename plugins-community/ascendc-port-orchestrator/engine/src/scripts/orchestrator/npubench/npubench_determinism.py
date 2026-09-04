# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Precision repeat fingerprinting and determinism history for the runner.

This module owns the runner's precision-stability semantics: how many times a
failing case is re-observed (``NPUBENCH_PRECISION_REPEATS``), the
per-observation fingerprint (verdict + non-finite quadruple + reference mask
digest) and its classification into the P2-4 repeat-fingerprint vocabulary,
the aggregated precision report, and the per-binding verdict history that
raises the determinism alert when the same binding drifts across evaluations.

It imports only ``npubench_core`` and ``npubench_precision``.
``npubench_runner`` re-exports its public surface, so importers keep using
the runner module path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from npubench_core import (
    EVIDENCE_DIRNAME,
    NpuBenchRunnerError,
    _atomic_json,
    _base_report,
    _canonical_sha256,
    _json_safe_non_finite,
    _workspace_runtime_directory,
)

from npubench_precision import (
    _clone_value,
    _detach_to_cpu,
    _dtype_name,
    _infer_input_type,
    _invoke_model,
    _is_floating_tensor,
    _move_value,
    compare_outputs,
)


def _precision_history_payload(
    report: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Extract the binding history fields from a completed precision report."""
    if report.get("schema") != "cannbot.npubench.precision/v1":
        return None
    if report.get("status") not in {"PASS", "FAIL"}:
        return None
    cases = report.get("cases")
    binding_sha = report.get("binding_sha256")
    if not isinstance(cases, list) or not isinstance(binding_sha, str) or not binding_sha:
        return None
    verdicts = {
        str(case.get("case")): case.get("status")
        for case in cases
        if isinstance(case, Mapping)
    }
    classes = {
        str(case.get("case")): case.get("repeat_fingerprint", {}).get("class")
        for case in cases
        if isinstance(case, Mapping) and isinstance(case.get("repeat_fingerprint"), Mapping)
    }
    return binding_sha, verdicts, classes


def _read_precision_binding_history(history_path: Path) -> dict[str, Any]:
    """Read a valid binding history or return an empty history container."""
    history: dict[str, Any] = {"schema": _PRECISION_BINDING_HISTORY_SCHEMA, "bindings": {}}
    if not history_path.is_file() or history_path.is_symlink():
        return history
    try:
        loaded = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return history
    if (
        isinstance(loaded, Mapping)
        and loaded.get("schema") == _PRECISION_BINDING_HISTORY_SCHEMA
        and isinstance(loaded.get("bindings"), Mapping)
    ):
        return {"schema": _PRECISION_BINDING_HISTORY_SCHEMA, "bindings": dict(loaded["bindings"])}
    return history


def _append_precision_binding_history(
    history: dict[str, Any],
    binding_sha: str,
    report: Mapping[str, Any],
    verdicts: Mapping[str, Any],
    classes: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, bool]:
    """Append one report and return its previous entry plus the drift flag."""
    entries = history["bindings"].setdefault(binding_sha, [])
    if not isinstance(entries, list):
        entries = []
        history["bindings"][binding_sha] = entries
    previous = entries[-1] if entries and isinstance(entries[-1], Mapping) else None
    alert = previous is not None and previous.get("case_verdicts") != verdicts
    entries.append(
        {
            "run_id": report.get("run_id"),
            "timestamp": report.get("timestamp"),
            "case_verdicts": verdicts,
            "case_classes": classes,
        }
    )
    del entries[:-_PRECISION_BINDING_HISTORY_MAX_ENTRIES]
    return previous, bool(alert)


def _set_precision_determinism_alert(
    workspace: Path,
    report: dict[str, Any],
    *,
    binding_sha: str,
    verdicts: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    alert: bool,
) -> None:
    """Add the drift detail and emit its event after history is persisted."""
    report["determinism_alert"] = alert
    if not alert or previous is None:
        return
    previous_verdicts = previous.get("case_verdicts", {})
    changed = [
        case
        for case in set(verdicts) | set(previous_verdicts)
        if verdicts.get(case) != previous_verdicts.get(case)
    ]
    changed_cases = sorted(int(case) for case in changed if str(case).lstrip("-").isdigit())
    report["determinism_alert_detail"] = {
        "binding_sha256": binding_sha,
        "previous_run_id": previous.get("run_id"),
        "changed_cases": changed_cases,
    }
    _emit_precision_determinism_event(workspace, report, previous, changed_cases)


def _update_precision_binding_history(workspace: Path, report: dict[str, Any]) -> None:
    """Record precision verdicts and alert when one binding drifts across evals."""
    payload = _precision_history_payload(report)
    if payload is None:
        return
    binding_sha, verdicts, classes = payload
    try:
        evidence_root = _workspace_runtime_directory(
            workspace, EVIDENCE_DIRNAME, "NPUKernelBench evidence root"
        )
        history_path = evidence_root / PRECISION_BINDING_HISTORY_FILENAME
        history = _read_precision_binding_history(history_path)
        previous, alert = _append_precision_binding_history(
            history, binding_sha, report, verdicts, classes
        )
        _atomic_json(history_path, history)
    except (NpuBenchRunnerError, OSError, ValueError):
        return
    _set_precision_determinism_alert(
        workspace,
        report,
        binding_sha=binding_sha,
        verdicts=verdicts,
        previous=previous,
        alert=alert,
    )


def _emit_precision_determinism_event(
    workspace: Path,
    report: Mapping[str, Any],
    previous: Mapping[str, Any],
    changed_cases: Sequence[int],
) -> None:
    """Emit the NEW-P2 determinism event; best-effort, never fatal."""
    try:
        import events
    except ImportError:
        return
    try:
        events.emit(
            Path(workspace),
            "npubench.precision_determinism_alert",
            data={
                "binding_sha256": report.get("binding_sha256"),
                "run_id": report.get("run_id"),
                "previous_run_id": previous.get("run_id"),
                "changed_cases": list(changed_cases),
                "status": report.get("status"),
            },
        )
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "precision determinism event emission failed; continuing without the event",
            exc_info=exc,
        )


PRECISION_REPEATS_ENV = "NPUBENCH_PRECISION_REPEATS"
PRECISION_REPEATS_DEFAULT = 3
REFERENCE_OVERFLOW_INPUT_ABS_ENV = "NPUBENCH_REFERENCE_OVERFLOW_INPUT_ABS"
REFERENCE_OVERFLOW_INPUT_ABS_DEFAULT = 1.0e4
PRECISION_BINDING_HISTORY_FILENAME = "precision_binding_history.json"
_PRECISION_BINDING_HISTORY_SCHEMA = "cannbot.npubench.precision_binding_history/v1"
_PRECISION_BINDING_HISTORY_MAX_ENTRIES = 32
REPEAT_FINGERPRINT_CLASSES = (
    "stable-pass",
    "deterministic-fail",
    "bimodal",
    "reference-unstable",
)


def _resolve_precision_repeats() -> int:
    """Total observations for a FAIL case: 1 initial pass + (N-1) FAIL reruns.

    Passing cases are always observed exactly once, so the total number of
    case executions stays bounded by ``case_count + (N-1) * failed_count``.
    An invalid env value is a configuration error and fails fast, matching
    ``_resolve_task_execution_timeout``.
    """
    raw = os.environ.get(PRECISION_REPEATS_ENV)
    if raw is None:
        return PRECISION_REPEATS_DEFAULT
    value = raw.strip()
    try:
        repeats = int(value)
    except ValueError as exc:
        raise NpuBenchRunnerError(
            f"{PRECISION_REPEATS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if not value or repeats < 1:
        raise NpuBenchRunnerError(
            f"{PRECISION_REPEATS_ENV} must be a positive integer, got {raw!r}"
        )
    return repeats


def _resolve_reference_overflow_threshold() -> float:
    """max(|input|) above which a case is annotated as reference overflow risk."""
    raw = os.environ.get(REFERENCE_OVERFLOW_INPUT_ABS_ENV)
    if raw is None:
        return REFERENCE_OVERFLOW_INPUT_ABS_DEFAULT
    try:
        threshold = float(raw.strip())
    except ValueError as exc:
        raise NpuBenchRunnerError(
            f"{REFERENCE_OVERFLOW_INPUT_ABS_ENV} must be a positive float, got {raw!r}"
        ) from exc
    if not threshold > 0:
        raise NpuBenchRunnerError(
            f"{REFERENCE_OVERFLOW_INPUT_ABS_ENV} must be a positive float, got {raw!r}"
        )
    return threshold


def _iter_output_tensors(value: Any, torch: Any) -> Any:
    """Yield every tensor inside one model output or input group tree."""
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(value, tensor_type):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_output_tensors(item, torch)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_output_tensors(item, torch)


def _nonfinite_counts(output: Any, torch: Any) -> tuple[int, int] | None:
    """Return ``(inf_count, nan_count)`` over all floating output tensors.

    NaN and Inf are counted separately on purpose: the P2-4 fingerprint
    contract requires it (a NaN mask jump and an Inf mask jump have different
    root causes).  ``None`` means the output was never produced (the case
    raised before invocation finished).
    """
    if output is None:
        return None
    inf_count = 0
    nan_count = 0
    for tensor in _iter_output_tensors(output, torch):
        if not _is_floating_tensor(tensor):
            continue
        cpu = _detach_to_cpu(tensor)
        nan_count += int(torch.isnan(cpu).sum().item())
        inf_count += int(torch.isinf(cpu).sum().item())
    return inf_count, nan_count


def _clone_and_move_precision_input(group: Any, torch: Any, device_value: Any) -> Any:
    """Clone one task-owned input group before moving it to the execution device."""
    return _move_value(_clone_value(group, torch), device_value, torch)


def _mask_bytes(mask: Any, torch: Any) -> bytes:
    """Serialize a boolean mask tensor to bytes for digesting."""
    as_byte = mask.to(torch.uint8).contiguous()
    numpy_view = getattr(as_byte, "numpy", None)
    if callable(numpy_view):
        try:
            return numpy_view().tobytes()
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "numpy mask serialization failed; using the tensor-list fallback",
                exc_info=exc,
            )
    return bytes(bytearray(as_byte.reshape(-1).tolist()))


def _reference_nonfinite_mask_sha256(output: Any, torch: Any) -> str | None:
    """Digest the reference output's non-finite mask for cross-eval comparison.

    The mask (not the values) is hashed so two evaluations can be compared for
    "the reference went non-finite in the same places" without persisting
    tensors.  Shape and dtype per tensor are mixed into the digest so a mask
    byte string is never read against the wrong layout.
    """
    if output is None:
        return None
    digest = hashlib.sha256()
    for tensor in _iter_output_tensors(output, torch):
        if not _is_floating_tensor(tensor):
            continue
        cpu = _detach_to_cpu(tensor)
        mask = ~torch.isfinite(cpu)
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(_dtype_name(cpu.dtype).encode("ascii"))
        digest.update(_mask_bytes(mask, torch))
    return digest.hexdigest()


def _input_max_abs(group: Any, torch: Any) -> float | None:
    """Largest |input| over the case's floating tensors, or None when tensorless."""
    peak: float | None = None
    for tensor in _iter_output_tensors(group, torch):
        if not _is_floating_tensor(tensor) or not int(tensor.numel()):
            continue
        value = float(tensor.detach().abs().max().item())
        peak = value if peak is None else max(peak, value)
    return peak


def _precision_observation(
    passed: bool,
    metrics: Mapping[str, Any],
    reference_output: Any,
    candidate_output: Any,
    torch: Any,
) -> dict[str, Any]:
    """One fingerprint observation: verdict + the P2-4 quadruple + ref mask sha."""
    reference_counts = _nonfinite_counts(reference_output, torch)
    candidate_counts = _nonfinite_counts(candidate_output, torch)
    return {
        "status": "PASS" if passed else "FAIL",
        "mere": metrics.get("MERE"),
        "matched_ratio": metrics.get("matched_ratio"),
        "candidate_inf_count": candidate_counts[0] if candidate_counts is not None else None,
        "candidate_nan_count": candidate_counts[1] if candidate_counts is not None else None,
        "reference_inf_count": reference_counts[0] if reference_counts is not None else None,
        "reference_nan_count": reference_counts[1] if reference_counts is not None else None,
        "reference_nonfinite_mask_sha256": _reference_nonfinite_mask_sha256(reference_output, torch),
    }


def _observation_key(observation: Mapping[str, Any]) -> str:
    """Bit-identity key for one observation, safe against inf/nan metric floats."""
    return _canonical_sha256(
        _json_safe_non_finite(
            {
                "status": observation.get("status"),
                "mere": observation.get("mere"),
                "matched_ratio": observation.get("matched_ratio"),
                "candidate_inf_count": observation.get("candidate_inf_count"),
                "candidate_nan_count": observation.get("candidate_nan_count"),
                "reference_inf_count": observation.get("reference_inf_count"),
                "reference_nan_count": observation.get("reference_nan_count"),
            }
        )
    )


def _classify_repeat_fingerprint(observations: Sequence[Mapping[str, Any]]) -> str:
    """Classify one case's repeat observations into the P2-4 vocabulary.

    Precedence matters: ``reference-unstable`` wins over everything (a case
    whose reference went non-finite or whose reference non-finite count moved
    across repeats cannot adjudicate the candidate at all); ``stable-pass``
    only requires every observation to PASS (floating metrics may legitimately
    drift between passing observations); ``deterministic-fail`` requires all
    observations to FAIL with bit-identical quadruples; anything else --
    verdict flips or diverging failure values -- is ``bimodal``.
    """
    if not observations:
        raise NpuBenchRunnerError("repeat fingerprint requires at least one observation")
    ref_nonfinite: list[int | None] = []
    for observation in observations:
        inf_count = observation.get("reference_inf_count")
        nan_count = observation.get("reference_nan_count")
        if inf_count is None and nan_count is None:
            ref_nonfinite.append(None)
        else:
            ref_nonfinite.append(int(inf_count or 0) + int(nan_count or 0))
    if any(value is not None and value > 0 for value in ref_nonfinite):
        return "reference-unstable"
    known_counts = [value for value in ref_nonfinite if value is not None]
    if len(set(known_counts)) > 1:
        return "reference-unstable"
    if all(observation.get("status") == "PASS" for observation in observations):
        return "stable-pass"
    if len({_observation_key(observation) for observation in observations}) == 1:
        return "deterministic-fail"
    return "bimodal"


@dataclass(frozen=True)
class _PrecisionCaseContext:
    """Runtime dependencies shared by one tracked precision case."""

    torch: Any
    reference_model: Any
    candidate_model: Any
    device_value: Any


def _precision_case_context(
    case_args: Sequence[Any], case_kwargs: Mapping[str, Any]
) -> _PrecisionCaseContext:
    """Package legacy positional or keyword calls without changing callers."""
    names = ("torch", "reference_model", "candidate_model", "device_value")
    if len(case_args) > len(names):
        raise TypeError("tracked precision case received too many runtime context arguments")
    context_values = dict(zip(names, case_args))
    unknown = set(case_kwargs) - set(names)
    if unknown:
        raise TypeError(f"unexpected tracked precision context: {sorted(unknown)!r}")
    duplicate = set(context_values) & set(case_kwargs)
    if duplicate:
        raise TypeError(f"multiple values for tracked precision context: {sorted(duplicate)!r}")
    context_values.update(case_kwargs)
    if set(context_values) != set(names):
        missing = sorted(set(names) - set(context_values))
        raise TypeError(f"missing tracked precision context: {missing!r}")
    return _PrecisionCaseContext(
        torch=context_values["torch"],
        reference_model=context_values["reference_model"],
        candidate_model=context_values["candidate_model"],
        device_value=context_values["device_value"],
    )


def _run_precision_case_comparison(
    index: int,
    group: Any,
    context: _PrecisionCaseContext,
) -> tuple[list[Any], list[Any], bool, dict[str, Any], str]:
    """Invoke both models and compare one case using a compact execution loop."""
    inputs = [
        _clone_and_move_precision_input(group, context.torch, context.device_value)
        for _ in range(2)
    ]
    outputs: list[Any] = [None, None]
    try:
        for output_index, (model, role) in enumerate(
            (
                (context.reference_model, "reference"),
                (context.candidate_model, "candidate"),
            )
        ):
            outputs[output_index] = _invoke_model(
                model, inputs[output_index], role, index
            )
        input_type, input_dtype = _infer_input_type(group, context.torch)
        passed, metrics, reason = compare_outputs(
            outputs[0],
            outputs[1],
            context.torch,
            input_type=input_type,
            input_dtype=input_dtype,
        )
    except Exception as exc:
        passed, metrics, reason = (
            False,
            {},
            f"case execution failed: {type(exc).__name__}: {exc}",
        )
    return inputs, outputs, passed, metrics, reason


def _run_precision_case_tracked(
    index: int,
    group: Any,
    *case_args: Any,
    overflow_threshold: float,
    annotate_inputs: bool,
    **case_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one precision case once, returning its report and one observation.

    Same execution semantics as ``_run_precision_cases`` (one task-owned input
    generation, independently cloned copies per model); the additions are the
    fingerprint observation and the one-time extreme-input annotation.
    """
    context = _precision_case_context(case_args, case_kwargs)
    torch = context.torch
    reference_model = context.reference_model
    candidate_model = context.candidate_model
    device_value = context.device_value
    input_max_abs = _input_max_abs(group, torch) if annotate_inputs else None
    case_inputs, case_outputs, passed, metrics, reason = _run_precision_case_comparison(
        index, group, context
    )
    reference_inputs, candidate_inputs = case_inputs
    reference_output, candidate_output = case_outputs
    observation = _precision_observation(passed, metrics, reference_output, candidate_output, torch)
    case_report = {
        "case": index,
        "status": "PASS" if passed else "FAIL",
        "metrics": metrics,
        "reason": reason if not passed else "",
    }
    if input_max_abs is not None:
        case_report["input_max_abs"] = input_max_abs
        if input_max_abs > overflow_threshold:
            # NEW-P1 half-measure: the reference is computed on-device, so an
            # extreme-magnitude input (fp32 softmax overflow zone) makes the
            # reference itself suspect.  Annotate only; the compute path is
            # deliberately not changed here.
            case_report["reference_overflow_risk"] = True
    # Release CPU and device copies before the next potentially large case.
    del reference_inputs, candidate_inputs, reference_output, candidate_output
    return case_report, observation


def _run_precision_cases_with_fingerprint(
    groups: Any,
    torch: Any,
    reference_model: Any,
    candidate_model: Any,
    device_value: Any,
    *,
    repeats: int,
    overflow_threshold: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Evaluate every case once, then re-observe FAIL cases ``repeats - 1`` times.

    The initial pass keeps the historical report fields (status/metrics/reason
    of the first observation); reruns only feed ``repeat_fingerprint``.  Input
    groups are re-cloned from the same task-owned generation for every
    observation, so a repeat measures execution/reference stability, not input
    resampling.
    """
    case_reports: list[dict[str, Any]] = []
    observations: list[list[dict[str, Any]]] = []
    failed_indices: list[int] = []
    # P0abb (2026-08-25): a fault OUTSIDE the per-case try (input `_move_value`
    # H2D after an earlier case's kernel wedged the device — 45_CrossformerAttention
    # 507035 signature) used to crash the whole child, losing every case report.
    # Publish the partial evidence with status=ERROR instead: the worker needs
    # case N-1's "case execution failed: ...507035" reason to distinguish a
    # kernel-wedged device from a genuine host-transient H2D fault.
    abort_reason: str | None = None
    for index, group in enumerate(groups):
        try:
            case_report, observation = _run_precision_case_tracked(
                index,
                group,
                torch,
                reference_model,
                candidate_model,
                device_value,
                overflow_threshold=overflow_threshold,
                annotate_inputs=True,
            )
        except Exception as exc:
            abort_reason = f"{type(exc).__name__}: {exc}"
            break
        case_reports.append(case_report)
        observations.append([observation])
        if case_report["status"] != "PASS":
            failed_indices.append(index)
    for _ in range(repeats - 1):
        for index in failed_indices:
            _, observation = _run_precision_case_tracked(
                index,
                groups[index],
                torch,
                reference_model,
                candidate_model,
                device_value,
                overflow_threshold=overflow_threshold,
                annotate_inputs=False,
            )
            observations[index].append(observation)
    for case_report, case_observations in zip(case_reports, observations):
        case_report["repeat_fingerprint"] = {
            "repeats": len(case_observations),
            "class": _classify_repeat_fingerprint(case_observations),
            "observations": case_observations,
        }
    return case_reports, abort_reason


def _precision_report(
    case_reports: list[dict[str, Any]],
    binding: Mapping[str, Any],
    *,
    seed: int,
    seed_events: Any,
    input_adapter: Mapping[str, Any],
    device_value: Any,
    case_count: int,
    repeats: int,
    overflow_threshold: float,
    abort_reason: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-case verdicts into the published precision report."""
    passed_count = sum(case["status"] == "PASS" for case in case_reports)
    status = "PASS" if passed_count == len(case_reports) else "FAIL"
    if abort_reason is not None:
        # P0abb: partial evidence after an out-of-case abort (wedged device).
        status = "ERROR"
    # Reference-stability gate (P2-4 layer 2): reference-unstable cases cannot
    # adjudicate the candidate, so they are isolated from the scored
    # denominator and reported separately.  The historical raw counts and
    # ``pass_a``/top-level ``status`` keep their all-cases meaning; the scored
    # view is purely additive for finalize-side consumers.
    reference_unstable_cases = [
        case["case"]
        for case in case_reports
        if case.get("repeat_fingerprint", {}).get("class") == "reference-unstable"
    ]
    reference_unstable_set = set(reference_unstable_cases)
    scored_reports = [case for case in case_reports if case["case"] not in reference_unstable_set]
    scored_passed = sum(case["status"] == "PASS" for case in scored_reports)
    result = _base_report("precision", status=status, binding=binding)
    result.update(
        {
            "seed": seed,
            "seed_events": seed_events,
            "input_adapter": input_adapter,
            "device": str(device_value),
            "case_count": case_count,
            "passed_case_count": passed_count,
            "failed_case_count": len(case_reports) - passed_count,
            "pass_a": {"status": status, "tier1_pass": passed_count, "total": case_count},
            "cases": case_reports,
            "precision_repeats": repeats,
            # NEW-P1 annotation: the reference model is constructed on the
            # evaluation device, so its outputs are device-side computations.
            "reference_compute_side": (
                "cpu" if str(device_value).startswith("cpu") else "device"
            ),
            "reference_overflow_threshold": overflow_threshold,
            "reference_unstable_cases": reference_unstable_cases,
            "scored_case_count": len(scored_reports),
            "scored_passed_case_count": scored_passed,
            "scored_failed_case_count": len(scored_reports) - scored_passed,
            "reason": abort_reason if abort_reason is not None else "",
        }
    )
    return result
