# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Native runner for the old NPUKernelBench task-pair format.

The runner deliberately treats a NPUKernelBench task as a Python program plus
its neighbouring ``.json``/``.jsonl`` sidecar.  It does not project the task
onto ``model.py``/``test.py``.  A short-lived adapter view is made only where
the existing profiler has that fixed-filename interface; the reference module
is still loaded from the frozen task's *real* path, retaining ``__file__`` and
package-relative imports.

This is a narrow runner, rather than an O2.5/O5 implementation.  Its public
entry points intentionally have no dependency on an A3 module:

* :func:`preflight_workspace` verifies a staged immutable binding and task API;
* :func:`run_precision_workspace` performs the old benchmark's three checks;
* :func:`run_performance_workspace` invokes the bundled quick profiler;
* :func:`evaluate_workspace` combines two already-assigned lanes.

The staging implementation lives in ``npubench_inputs``.  It is imported only
at the staging/preflight boundary so a malformed or unsupported workspace
fails closed without making this runner import provider or A3 machinery.
"""
from __future__ import annotations

__all__ = [
    "ALLCLOSE_TOLERANCES", "CANDIDATE_DIGEST_SCHEME", "DEFAULT_SEED", "EVALUATE_REPORT_FILENAME",
    "EVIDENCE_DIRNAME", "EXECUTION_DIRNAME", "INPUT_ADAPTER_CONTRACT_VERSION", "INT_LSB_TOLERANCE",
    "NATIVE_PERF_CASE_DIRNAME", "NATIVE_PERF_CASE_SCHEMA", "NATIVE_PERF_FIXTURE_FILENAME",
    "NATIVE_PERF_FIXTURE_SCHEMA", "NATIVE_PERF_MANIFEST_FILENAME", "NATIVE_PERF_MANIFEST_SCHEMA", "NPUBENCH_SOURCE",
    "NPU_LIMITS", "NpuBenchRunnerError", "PERFORMANCE_CONTRACT_VERSION", "PERFORMANCE_REPORT_FILENAME",
    "PERF_SKIP_ENV", "PRECISION_BINDING_HISTORY_FILENAME", "PRECISION_CONTRACT_VERSION",
    "PRECISION_REPORT_FILENAME", "PRECISION_REPEATS_DEFAULT", "PRECISION_REPEATS_ENV",
    "PRECISION_SEMANTICS_SOURCE", "REFERENCE_OVERFLOW_INPUT_ABS_DEFAULT", "REFERENCE_OVERFLOW_INPUT_ABS_ENV",
    "PREFLIGHT_REPORT_FILENAME", "REPEATS", "REPEAT_FINGERPRINT_CLASSES", "REQUIRED_MATCHED_RATIO",
    "RUNNER_CONTRACT_VERSION",
    "RUNNER_MODULE_FILENAMES", "SIDECAR_DESCRIPTOR_ADAPTER", "SIDECAR_DESCRIPTOR_SCHEMA", "SNAPSHOT_DIRNAME",
    "StagedBundle", "TASK_EXECUTION_TIMEOUT_ENV", "TASK_EXECUTION_TIMEOUT_SECONDS", "WARM_UP",
    "_CANDIDATE_BUILD_RELATIVE", "_CANDIDATE_RUNTIME_SUFFIXES", "_CANDIDATE_RUNTIME_TOP_LEVEL", "_ExecutionContext",
    "_NATIVE_PRIMITIVE_TYPES", "_NATIVE_QUICK_SHIM_HELPERS",
    "_NATIVE_QUICK_SHIM_INPUTS", "_PerformanceLanes", "_SIDECAR_DTYPE_ALIASES", "_SIDECAR_DTYPE_ITEMSIZE",
    "_SIDECAR_FLOAT_DTYPES", "_SIDECAR_INT_DTYPES", "_SIDECAR_MAX_CASE_BYTES", "_SIDECAR_MAX_DIM",
    "_SIDECAR_MAX_ELEMENTS", "_SIDECAR_MAX_TENSOR_BYTES", "_SIDECAR_MAX_TOTAL_BYTES", "_SidecarInputGroups",
    "_apply_assigned_device", "_archive_retained_profiles", "_assert_evaluate_devices", "_assert_input_adapter_api",
    "_assert_input_adapter_binding", "_assert_repository_profiler_script", "_assert_request_binding",
    "_assert_safe_child_name", "_assert_task_relative_imports", "_atomic_json", "_atomic_torch_fixture",
    "_base_report", "_benchmark_accuracy", "_build_execution_request", "_candidate_entry", "_candidate_excluded",
    "_candidate_root", "_candidate_tree_sha256", "_canonical_sha256", "_case_raw_profile_paths", "_check_nan_inf",
    "_child_profiler_report", "_classify_repeat_fingerprint", "_cleanup_execution_context",
    "_cleanup_frozen_native_fixture",
    "_cleanup_performance_lanes", "_cli_parser", "_clone_pythonish", "_clone_value", "_compare_complex_tensor",
    "_compare_finite_tensor", "_compare_float_tensor", "_compare_integer_tensor", "_compare_output_leaf",
    "_compare_output_mapping", "_compare_output_sequence", "_compare_tensor", "_configured_target_python_values",
    "_construct_model", "_construct_precision_models", "_copy_candidate_build_directory", "_copy_candidate_files",
    "_copy_candidate_scope", "_copy_frozen_native_fixture", "_copy_raw_profiles", "_copy_regular_tree",
    "_create_execution_context", "_create_real_child_directory", "_decorate_adapter_manifest",
    "_default_profiler_summary", "_deferred_performance_report", "_detach_to_cpu", "_dispatch_execution_verb",
    "_drain_npu_stream", "_dtype_flag", "_dtype_name", "_dtype_rank", "_ensure_real_child_directory",
    "_evaluate_aggregate_report", "_evaluate_lane_verdict", "_exec_task_module", "_execute_fixture_verb",
    "_execute_performance_verb", "_execute_precision_verb", "_execute_preflight_verb", "_expected_profile_dir",
    "_failed_native_profiler_report", "_file_sha256", "_freeze_native_case_chunks", "_freeze_native_perf_fixture",
    "_frozen_native_fixture_slots", "_get_init_args", "_get_input_groups", "_identity_path", "_import_torch",
    "_in_process_base_report", "_in_process_measurement_fields", "_indexed_profile_cases",
    "_indexed_quick_profiler_rows", "_infer_input_type", "_input_adapter_binding", "_input_adapter_contract",
    "_input_adapter_identity", "_input_max_abs", "_input_provider_names_bound_by", "_install_package",
    "_install_synthetic_packages",
    "_internal_execute_request", "_invoke_model", "_is_a3_routing_variable", "_is_bool_dtype",
    "_is_floating_tensor", "_is_frozen_candidate_snapshot", "_is_int_like_dtype", "_is_integer_dtype",
    "_is_torch_descriptor_type", "_iter_output_tensors", "_json_safe_non_finite", "_kept_candidate_subdirectories",
    "_load_inputs_provider",
    "_load_native_common_fixture", "_load_native_perf_manifest", "_load_quick_profiler_summary",
    "_make_tree_read_only", "_mask_any", "_materialize_native_perf_fixture", "_materialize_sidecar_case",
    "_move_model", "_move_value", "_native_case_records", "_native_fixture_realization",
    "_native_perf_manifest_document", "_native_quick_shim_header", "_native_quick_shim_model",
    "_native_quick_shim_targets", "_native_quick_sidecar_name", "_native_reject_tensor_aliases",
    "_native_restricted_mapping", "_native_restricted_tensor", "_native_restricted_tree",
    "_native_tensor_storage_key", "_native_value_has_empty_tensor", "_nonfinite_counts", "_observation_key",
    "_output_tail", "_package_layout",
    "_parent_performance_command", "_parent_performance_report_fields", "_parse_child_report", "_parse_jsonl",
    "_parse_task_syntax", "_passed_native_profiler_report", "_perf_below_threshold", "_performance_adapter_path",
    "_performance_child_context", "_performance_contract", "_performance_error_report", "_positive_case_count",
    "_precision_contract", "_precision_observation", "_precision_report", "_preflight_workspace_in_process",
    "_prepare_execution_root",
    "_prepare_native_quick_adapter", "_profile_archive_slots", "_publish_execution_request",
    "_quick_profiler_timings", "_raw_profile_paths", "_read_execution_request_payload", "_read_lease_manifest",
    "_read_native_perf_manifest", "_read_native_quick_adapter_manifest", "_read_quick_profiler_report",
    "_read_state", "_reference_nonfinite_mask_sha256", "_request_bundle_document", "_request_candidate_document",
    "_request_fixture_document",
    "_request_scratch", "_require_allowed_profile_dir", "_require_parent_frozen_native_fixture",
    "_require_real_directory", "_require_real_read_only_tree", "_require_regular", "_require_snapshot",
    "_required_relative_path", "_resolve_child_run_id", "_resolve_configured_python", "_resolve_device",
    "_resolve_execution_request", "_resolve_input_groups", "_resolve_manifest_path", "_resolve_model_constructor",
    "_resolve_precision_repeats", "_resolve_reference_overflow_threshold",
    "_resolve_request_bundle", "_resolve_request_candidate", "_resolve_request_fixture",
    "_resolve_request_manifest", "_resolve_request_task_paths", "_resolve_target_python",
    "_resolve_task_execution_timeout", "_resolve_under", "_run_child_process_group", "_run_evaluation_lanes",
    "_run_evaluation_lanes_in_parallel", "_run_evaluation_lanes_in_sequence", "_run_isolated_context",
    "_run_isolated_performance", "_run_native_fixture_phase", "_run_native_profiler_phase",
    "_run_performance_workspace_in_process", "_run_precision", "_run_precision_case_tracked", "_run_precision_cases",
    "_run_precision_cases_with_fingerprint", "_run_precision_workspace_in_process", "_run_runner_child",
    "_runner_ascendc_env_path",
    "_safe_child_failure_reason", "_safe_prof_tag", "_sanitized_quick_profiler_row", "_scrubbed_task_environment",
    "_set_eval", "_sidecar_attr_descriptor", "_sidecar_attr_value_ok", "_sidecar_case_generator",
    "_sidecar_descriptor_fields_ok", "_sidecar_range_ok", "_sidecar_shape_ok", "_sidecar_tensor_descriptor",
    "_sidecar_tensor_extent", "_sidecar_torch_dtype", "_skipped_quick_profiler_row", "_stage_execution_inputs",
    "_stage_execution_runner_copy", "_stage_isolated_native_fixture", "_subreport_failure_reason",
    "_precision_host_error_only", "_synthesized_skipped_performance",
    "_task_module_name", "_tensor_is_complex", "_tensor_sign", "_terminate_child_process_group",
    "_try_write_report", "_update_precision_binding_history", "_valid_profile_case_indices",
    "_validate_native_case_fixture_payload",
    "_validate_parallel_leases", "_validate_parent_published_child_report", "_validate_reference_api",
    "_validate_reported_profile_dir", "_validate_requested_lease", "_validate_sidecar_case",
    "_validate_sidecar_descriptors", "_validated_case_indices", "_validated_expected_valid_cases",
    "_validated_native_adapter_identity", "_validated_native_case_fixtures", "_validated_native_case_partition",
    "_validated_native_case_record", "_verify_parent_binding_unchanged", "_workspace_runtime_directory",
    "_write_adapter_proxy", "_write_native_common_fixture", "_write_native_quick_case_index",
    "_write_native_quick_model_shim", "build_evaluation_binding", "build_performance_command",
    "candidate_tree_sha256", "compare_outputs", "evaluate_workspace", "load_task_module", "main",
    "materialize_candidate_snapshot", "parse_sidecar", "preflight_workspace", "prepare_adapter_view",
    "profile_tree_sha256", "resolve_staged_bundle", "run_performance_workspace", "run_precision_workspace",
    "runner_module_path", "seed_everything", "stage_workspace", "tree_sha256", "verify_evidence_report",
]

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The runner is split across sibling modules for size, but its module path
# is the public one: transports, plugins and the UT suite all reach these
# names through ``npubench_runner``.  Re-export them here so no importer
# changes, and so ``monkeypatch.setattr(npubench_runner, ...)`` keeps
# rebinding the name that this module's own functions resolve.
from npubench_core import (  # noqa: F401  re-exported runner surface
    ALLCLOSE_TOLERANCES,
    CANDIDATE_DIGEST_SCHEME,
    DEFAULT_SEED,
    EVALUATE_REPORT_FILENAME,
    EVIDENCE_DIRNAME,
    EXECUTION_DIRNAME,
    INPUT_ADAPTER_CONTRACT_VERSION,
    INT_LSB_TOLERANCE,
    NATIVE_PERF_CASE_DIRNAME,
    NATIVE_PERF_CASE_SCHEMA,
    NATIVE_PERF_FIXTURE_FILENAME,
    NATIVE_PERF_FIXTURE_SCHEMA,
    NATIVE_PERF_MANIFEST_FILENAME,
    NATIVE_PERF_MANIFEST_SCHEMA,
    NPUBENCH_SOURCE,
    NPU_LIMITS,
    NpuBenchRunnerError,
    PERFORMANCE_CONTRACT_VERSION,
    PERFORMANCE_REPORT_FILENAME,
    PERF_SKIP_ENV,
    PRECISION_CONTRACT_VERSION,
    PRECISION_REPORT_FILENAME,
    PRECISION_SEMANTICS_SOURCE,
    PREFLIGHT_REPORT_FILENAME,
    REPEATS,
    REQUIRED_MATCHED_RATIO,
    RUNNER_CONTRACT_VERSION,
    SIDECAR_DESCRIPTOR_ADAPTER,
    SIDECAR_DESCRIPTOR_SCHEMA,
    SNAPSHOT_DIRNAME,
    StagedBundle,
    TASK_EXECUTION_TIMEOUT_ENV,
    TASK_EXECUTION_TIMEOUT_SECONDS,
    WARM_UP,
    _CANDIDATE_BUILD_RELATIVE,
    _CANDIDATE_RUNTIME_SUFFIXES,
    _CANDIDATE_RUNTIME_TOP_LEVEL,
    _ExecutionContext,
    _SIDECAR_DTYPE_ALIASES,
    _SIDECAR_DTYPE_ITEMSIZE,
    _SIDECAR_FLOAT_DTYPES,
    _SIDECAR_INT_DTYPES,
    _SIDECAR_MAX_CASE_BYTES,
    _SIDECAR_MAX_DIM,
    _SIDECAR_MAX_ELEMENTS,
    _SIDECAR_MAX_TENSOR_BYTES,
    _SIDECAR_MAX_TOTAL_BYTES,
    _assert_request_binding,
    _assert_safe_child_name,
    _atomic_json,
    _base_report,
    _candidate_entry,
    _candidate_excluded,
    _candidate_root,
    _candidate_tree_sha256,
    _canonical_sha256,
    _cleanup_execution_context,
    _cli_parser,
    _configured_target_python_values,
    _copy_candidate_build_directory,
    _copy_candidate_files,
    _copy_candidate_scope,
    _copy_regular_tree,
    _create_real_child_directory,
    _deferred_performance_report,
    _ensure_real_child_directory,
    _evaluate_lane_verdict,
    _file_sha256,
    _import_torch,
    _input_adapter_binding,
    _input_adapter_contract,
    _input_adapter_identity,
    _input_provider_names_bound_by,
    _is_a3_routing_variable,
    _is_frozen_candidate_snapshot,
    _json_safe_non_finite,
    _kept_candidate_subdirectories,
    _load_inputs_provider,
    _make_tree_read_only,
    _output_tail,
    _parent_performance_command,
    _parent_performance_report_fields,
    _parse_child_report,
    _parse_jsonl,
    _perf_below_threshold,
    _performance_adapter_path,
    _performance_contract,
    _positive_case_count,
    _precision_contract,
    _read_execution_request_payload,
    _read_state,
    _request_scratch,
    _require_real_directory,
    _require_real_read_only_tree,
    _require_regular,
    _require_snapshot,
    _required_relative_path,
    _resolve_configured_python,
    _resolve_execution_request,
    _resolve_manifest_path,
    _resolve_request_bundle,
    _resolve_request_candidate,
    _resolve_request_fixture,
    _resolve_request_manifest,
    _resolve_request_task_paths,
    _resolve_target_python,
    _resolve_task_execution_timeout,
    _resolve_under,
    _run_child_process_group,
    _run_isolated_context,
    _run_runner_child,
    _runner_ascendc_env_path,
    _safe_child_failure_reason,
    _safe_prof_tag,
    _precision_host_error_only,
    _scrubbed_task_environment,
    _subreport_failure_reason,
    _synthesized_skipped_performance,
    _terminate_child_process_group,
    _try_write_report,
    _validate_parent_published_child_report,
    _verify_parent_binding_unchanged,
    _workspace_runtime_directory,
    build_evaluation_binding,
    candidate_tree_sha256,
    materialize_candidate_snapshot,
    parse_sidecar,
    profile_tree_sha256,
    resolve_staged_bundle,
    runner_module_path,
    stage_workspace,
    tree_sha256,
    verify_evidence_report,
)

from npubench_precision import (  # noqa: F401  re-exported runner surface
    _assert_task_relative_imports,
    _benchmark_accuracy,
    _check_nan_inf,
    _clone_pythonish,
    _clone_value,
    _compare_complex_tensor,
    _compare_finite_tensor,
    _compare_float_tensor,
    _compare_integer_tensor,
    _compare_output_leaf,
    _compare_output_mapping,
    _compare_output_sequence,
    _compare_tensor,
    _construct_model,
    _detach_to_cpu,
    _drain_npu_stream,
    _dtype_flag,
    _dtype_name,
    _dtype_rank,
    _exec_task_module,
    _get_init_args,
    _get_input_groups,
    _infer_input_type,
    _install_package,
    _install_synthetic_packages,
    _invoke_model,
    _is_bool_dtype,
    _is_floating_tensor,
    _is_int_like_dtype,
    _is_integer_dtype,
    _mask_any,
    _move_model,
    _move_value,
    _package_layout,
    _parse_task_syntax,
    _resolve_device,
    _resolve_model_constructor,
    _run_precision_cases,
    _set_eval,
    _task_module_name,
    _tensor_is_complex,
    _tensor_sign,
    _validate_reference_api,
    compare_outputs,
    load_task_module,
    seed_everything,
)

from npubench_fixture import (  # noqa: F401  re-exported runner surface
    _NATIVE_PRIMITIVE_TYPES,
    _NATIVE_QUICK_SHIM_HELPERS,
    _NATIVE_QUICK_SHIM_INPUTS,
    _SidecarInputGroups,
    _assert_input_adapter_api,
    _assert_input_adapter_binding,
    _atomic_torch_fixture,
    _cleanup_frozen_native_fixture,
    _copy_frozen_native_fixture,
    _decorate_adapter_manifest,
    _freeze_native_case_chunks,
    _freeze_native_perf_fixture,
    _frozen_native_fixture_slots,
    _is_torch_descriptor_type,
    _load_native_common_fixture,
    _load_native_perf_manifest,
    _materialize_sidecar_case,
    _native_quick_shim_header,
    _native_quick_shim_model,
    _native_quick_shim_targets,
    _native_quick_sidecar_name,
    _native_reject_tensor_aliases,
    _native_restricted_mapping,
    _native_restricted_tensor,
    _native_restricted_tree,
    _native_tensor_storage_key,
    _native_value_has_empty_tensor,
    _prepare_native_quick_adapter,
    _read_native_perf_manifest,
    _read_native_quick_adapter_manifest,
    _require_parent_frozen_native_fixture,
    _sidecar_attr_descriptor,
    _sidecar_attr_value_ok,
    _sidecar_case_generator,
    _sidecar_descriptor_fields_ok,
    _sidecar_range_ok,
    _sidecar_shape_ok,
    _sidecar_tensor_descriptor,
    _sidecar_tensor_extent,
    _sidecar_torch_dtype,
    _validate_native_case_fixture_payload,
    _validate_sidecar_case,
    _validate_sidecar_descriptors,
    _validated_case_indices,
    _validated_native_adapter_identity,
    _validated_native_case_fixtures,
    _validated_native_case_partition,
    _validated_native_case_record,
    _write_adapter_proxy,
    _write_native_common_fixture,
    _write_native_quick_case_index,
    _write_native_quick_model_shim,
    prepare_adapter_view,
)

from npubench_profile import (  # noqa: F401  re-exported runner surface
    _archive_retained_profiles,
    _case_raw_profile_paths,
    _copy_raw_profiles,
    _default_profiler_summary,
    _expected_profile_dir,
    _identity_path,
    _indexed_profile_cases,
    _indexed_quick_profiler_rows,
    _load_quick_profiler_summary,
    _native_case_records,
    _profile_archive_slots,
    _quick_profiler_timings,
    _raw_profile_paths,
    _read_lease_manifest,
    _read_quick_profiler_report,
    _require_allowed_profile_dir,
    _sanitized_quick_profiler_row,
    _skipped_quick_profiler_row,
    _valid_profile_case_indices,
    _validate_parallel_leases,
    _validate_reported_profile_dir,
    _validate_requested_lease,
    _validated_expected_valid_cases,
)


from npubench_determinism import (  # re-exported runner surface
    PRECISION_BINDING_HISTORY_FILENAME,
    PRECISION_REPEATS_DEFAULT,
    PRECISION_REPEATS_ENV,
    REFERENCE_OVERFLOW_INPUT_ABS_DEFAULT,
    REFERENCE_OVERFLOW_INPUT_ABS_ENV,
    REPEAT_FINGERPRINT_CLASSES,
    _PRECISION_BINDING_HISTORY_MAX_ENTRIES,
    _PRECISION_BINDING_HISTORY_SCHEMA,
    _PrecisionCaseContext,
    _append_precision_binding_history,
    _classify_repeat_fingerprint,
    _clone_and_move_precision_input,
    _emit_precision_determinism_event,
    _input_max_abs,
    _iter_output_tensors,
    _mask_bytes,
    _nonfinite_counts,
    _observation_key,
    _precision_case_context,
    _precision_history_payload,
    _precision_observation,
    _precision_report,
    _read_precision_binding_history,
    _reference_nonfinite_mask_sha256,
    _resolve_precision_repeats,
    _resolve_reference_overflow_threshold,
    _run_precision_case_comparison,
    _run_precision_case_tracked,
    _run_precision_cases_with_fingerprint,
    _set_precision_determinism_alert,
    _update_precision_binding_history,
)


# Every file the evaluator needs when it is copied out of the checkout: this
# runner, the modules it is split across, and the immutable-stage verifier it
# imports lazily.  ``_stage_execution_runner_copy`` copies exactly this list;
# any transport that ships the runner to another host must ship all of it.
RUNNER_MODULE_FILENAMES: tuple[str, ...] = (
    "npubench_runner.py",
    "npubench_core.py",
    "npubench_determinism.py",
    "npubench_fixture.py",
    "npubench_precision.py",
    "npubench_profile.py",
    "npubench_inputs.py",
)


def preflight_workspace(
    workspace: Path,
    *,
    isolated: bool = True,
    subprocess_run: Callable[..., Any] = subprocess.run,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Validate only staged NPUKernelBench input, never a candidate or A3.

    Expected malformed input returns a JSON-safe ``ERROR`` report rather than
    throwing into O2.5.  A successful report contains the binding digest which
    O5/finalize can bind to later candidate evidence.
    """
    timeout_seconds = _resolve_task_execution_timeout(timeout_seconds)
    context: _ExecutionContext | None = None
    try:
        if isolated:
            bundle = resolve_staged_bundle(Path(workspace))
            binding = build_evaluation_binding(Path(workspace), bundle=bundle)
            context = _create_execution_context(
                Path(workspace), bundle=bundle, candidate_dir=None, binding=binding, verb="preflight"
            )
            result = _run_isolated_context(
                context, subprocess_run=subprocess_run, timeout_seconds=timeout_seconds
            )
            _validate_parent_published_child_report(
                result, binding, verb="preflight", expected_run_id=context.run_id
            )
        else:
            result = _preflight_workspace_in_process(Path(workspace))
            result["execution_isolation"] = "in_process_test_only"
            result["tamper_protection"] = "none"
    except (NpuBenchRunnerError, OSError, ValueError, subprocess.SubprocessError) as exc:
        result = _base_report("preflight", status="ERROR")
        result["reason"] = str(exc)
    if isinstance(result, Mapping) and result.get("status") == "READY":
        # READY describes staging, not an executable task preflight.  Keep the
        # runner contract stricter than any transport/finalizer compatibility
        # shim so staging-only evidence cannot enter O2.5 as a success.
        result = dict(result)
        result["status"] = "ERROR"
        result["reason"] = "preflight returned staging-only READY status"
    _try_write_report(Path(workspace), PREFLIGHT_REPORT_FILENAME, result)
    if context is not None:
        _cleanup_execution_context(context)
    return result


def _preflight_workspace_in_process(workspace: Path) -> dict[str, Any]:
    """Internal child-only implementation that imports the untrusted task."""
    try:
        bundle = resolve_staged_bundle(workspace)
        module = load_task_module(bundle.task_path, bundle.root, role="reference")
        api = _validate_reference_api(module)
        binding = build_evaluation_binding(workspace, bundle=bundle)
        _assert_input_adapter_api(api, binding)
        if api.get("input_provider") == "sidecar_descriptor":
            _validate_sidecar_descriptors(bundle.sidecar_cases)
        result = _base_report("preflight", status="PASS", binding=binding)
        result.update(
            {
                "task_path": str(bundle.task_path),
                "sidecar_path": str(bundle.sidecar_path),
                "sidecar_encoding": bundle.sidecar_encoding,
                "case_count": len(bundle.sidecar_cases),
                "task_api": api,
            }
        )
    except (NpuBenchRunnerError, OSError, ValueError, SyntaxError, ImportError) as exc:
        result = _base_report("preflight", status="ERROR")
        result["reason"] = str(exc)
    return result


def _apply_assigned_device(request: dict[str, Any]) -> None:
    """Pin the child to its ONE assigned NPU before any torch/torch_npu import.

    Full 8-card visibility makes torch_npu negotiate an HCCL collective world
    at first device use; on shared-mode cards that init faults with acl error
    507035 (2026-08-22 BAM/SDPA/CoT on lanes 1-3, while single-card
    ASCEND_RT_VISIBLE_DEVICES runs work fine — stack traceback shows
    libhccl.so during the first tensor move).
    """
    assigned_device = request.get("device")
    if isinstance(assigned_device, int) and assigned_device >= 0:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(assigned_device)
        request["device"] = 0


def _execute_preflight_verb(
    bundle: StagedBundle, binding: Mapping[str, Any], run_id: str | None
) -> dict[str, Any]:
    """Child-side ``preflight``: import the task and validate its input API."""
    module = load_task_module(bundle.task_path, bundle.root, role="reference")
    api = _validate_reference_api(module)
    _assert_input_adapter_api(api, binding)
    if api.get("input_provider") == "sidecar_descriptor":
        _validate_sidecar_descriptors(bundle.sidecar_cases)
    result = _base_report("preflight", status="PASS", binding=binding, run_id=run_id)
    result.update(
        {
            "sidecar_encoding": bundle.sidecar_encoding,
            "case_count": len(bundle.sidecar_cases),
            "task_api": api,
        }
    )
    return result


def _execute_precision_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Child-side ``precision``: run the three checks against the candidate."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated precision execution requires a candidate snapshot")
    result = _run_precision(
        bundle,
        candidate,
        device=request.get("device"),
        seed=int(request.get("seed", DEFAULT_SEED)),
        binding=binding,
    )
    result["run_id"] = str(request.get("run_id") or result.get("run_id"))
    return result


def _execute_fixture_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Child-side ``fixture``: materialize the shared native perf fixture."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated native fixture requires a candidate snapshot")
    fixture_dir = _request_scratch(request) / "native_fixture"
    if fixture_dir.exists():
        raise NpuBenchRunnerError("isolated native fixture output already exists")
    fixture_dir.mkdir(mode=0o700)
    native_manifest = _materialize_native_perf_fixture(
        fixture_dir,
        bundle,
        candidate,
        binding=binding,
        seed=int(request.get("seed", DEFAULT_SEED)),
        write_adapter_manifest=False,
    )
    result = _base_report("fixture", status="PASS", binding=binding, run_id=run_id)
    result.update(
        {
            "fixture_relative": "native_fixture",
            "fixture_sha256": native_manifest["fixture_sha256"],
            "case_count": native_manifest["case_count"],
        }
    )
    return result


def _execute_performance_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    fixture_root: Path | None,
    run_id: str,
) -> dict[str, Any]:
    """Child-side ``performance``: drive the bundled quick profiler once."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated performance execution requires a candidate snapshot")
    if fixture_root is None:
        raise NpuBenchRunnerError("isolated performance requires a parent-frozen native fixture")
    scratch = _request_scratch(request)
    adapter = prepare_adapter_view(
        scratch,
        candidate,
        bundle=bundle,
        binding=binding,
        run_id=run_id,
        allow_existing_native_fixture=True,
    )
    device = request.get("device")
    if isinstance(device, bool) or not isinstance(device, int):
        raise NpuBenchRunnerError("isolated performance request has invalid device")
    native = _prepare_native_quick_adapter(
        adapter,
        fixture_root,
        binding=binding,
    )
    return _child_profiler_report(
        adapter,
        scratch,
        binding,
        native,
        device=device,
        run_id=run_id,
        profiler_script=Path(str(request.get("profiler_script", ""))),
    )


def _child_profiler_report(
    adapter: Path,
    scratch: Path,
    binding: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    device: int,
    run_id: str,
    profiler_script: Path,
) -> dict[str, Any]:
    """Invoke the repository quick profiler once and report what it returned."""
    command = build_performance_command(
        adapter,
        device=device,
        run_id=run_id,
        profiler_script=profiler_script,
    )
    completed = subprocess.run(command, cwd=str(adapter), text=True, capture_output=True, check=False)
    result = _base_report(
        "performance",
        status="PASS" if completed.returncode == 0 else "FAIL",
        binding=binding,
        run_id=run_id,
    )
    result.update(
        {
            "returncode": int(completed.returncode),
            "command": command,
            "adapter_relative": str(adapter.relative_to(scratch)),
            "native_fixture_sha256": native["fixture_sha256"],
            "stdout_tail": _output_tail(completed.stdout),
            "stderr_tail": _output_tail(completed.stderr),
        }
    )
    return result


def _resolve_child_run_id(request: Mapping[str, Any], verb: str) -> str:
    """Return the run id this child reports under, generating one when allowed."""
    if verb in {"fixture", "performance"}:
        return str(request.get("run_id") or uuid.uuid4().hex)
    return str(request.get("run_id") or None)


def _dispatch_execution_verb(
    verb: str,
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: dict[str, Any],
    *,
    fixture_root: Path | None,
    run_id: str,
) -> dict[str, Any]:
    """Route one already-verified request to its child-side verb handler."""
    _apply_assigned_device(request)
    if verb == "preflight":
        return _execute_preflight_verb(bundle, binding, run_id)
    if verb == "precision":
        return _execute_precision_verb(bundle, candidate, binding, request)
    if verb == "fixture":
        return _execute_fixture_verb(bundle, candidate, binding, request, run_id)
    if verb == "performance":
        return _execute_performance_verb(
            bundle, candidate, binding, request, fixture_root=fixture_root, run_id=run_id
        )
    raise NpuBenchRunnerError(f"unsupported internal execution verb: {verb}")


def _internal_execute_request(request_path: Path, *, verb: str) -> dict[str, Any]:
    """Child-only task/candidate executor; it never opens state or evidence."""
    binding: Mapping[str, Any] | None = None
    run_id: str | None = None
    try:
        bundle, candidate, binding, request, fixture_root = _resolve_execution_request(request_path)
        run_id = _resolve_child_run_id(request, verb)
        return _dispatch_execution_verb(
            verb, bundle, candidate, binding, dict(request), fixture_root=fixture_root, run_id=run_id
        )
    except Exception as exc:
        # A task import/fixture error is still a response to one verified
        # parent request.  Preserve that binding when it is already available
        # so the parent can surface the actionable child reason rather than
        # reporting a misleading "binding differs" infrastructure failure.
        # The catch is deliberately ``Exception`` and not ``BaseException``:
        # the interpreter's own shutdown signals are not task failures and
        # must escape this catch-all unchanged, which they do by not being
        # ``Exception`` subclasses.
        # 2026-08-22 (BAM 507035): candidate kernels can FAULT the NPU device
        # (acl error 507035) and torch surfaces that as RuntimeError — outside
        # the historical catch tuple — so the child died with a bare traceback
        # and the parent reported "no machine-readable report", classifying a
        # worker-fixable kernel fault as infra-terminal.  Catch EVERYTHING and
        # emit an ERROR report: the parent then records a measured FAIL and
        # the FSM routes the fix back to the worker.
        result = _base_report(verb, status="ERROR", binding=binding, run_id=run_id)
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


def _materialize_native_perf_fixture(
    adapter: Path,
    bundle: StagedBundle,
    candidate_dir: Path,
    *,
    binding: Mapping[str, Any],
    seed: int,
    write_adapter_manifest: bool = True,
) -> Mapping[str, Any]:
    """Freeze one model/input realization for the native profiler bridge.

    The legacy profiler used to import both task variants and invoke their
    input generators independently.  That is observably wrong for old
    NPUKernelBench tasks such as ``3_Add`` whose inputs contain random or
    otherwise non-``model.py`` state.  This adapter-side fixture is generated
    exactly once from the frozen reference task, then the profiler consumer
    constructs both models from the same candidate-preferred initialization
    arguments and invokes them with those exact input groups.

    The common payload plus one per-case ``torch.save`` chunk are validated
    restricted trees.  Consumers load the common payload and only the selected
    case chunk with ``weights_only=True`` and ``map_location='cpu'``; they must
    never fall back to the executable pickle loader.  This retains tensor
    dtype/storage/view aliasing within each fixture payload while avoiding a
    monolithic multi-gigabyte load for every profiler process.  See
    ``native_perf_manifest.json`` for the public handoff schema.
    """
    adapter = Path(adapter)
    torch = _import_torch()
    seed_events, init_args, groups, input_adapter = _native_fixture_realization(
        bundle, candidate_dir, torch, seed=seed
    )
    _assert_input_adapter_binding(input_adapter, binding)
    case_records, empty_case_indices, case_count = _freeze_native_case_chunks(
        adapter, groups, torch, binding
    )
    empty_case_set = set(empty_case_indices)
    valid_case_indices = [index for index in range(case_count) if index not in empty_case_set]
    fixture_sha256 = _write_native_common_fixture(
        adapter,
        torch,
        binding=binding,
        seed=seed,
        seed_events=seed_events,
        input_adapter=input_adapter,
        init_args=init_args,
        case_count=case_count,
        empty_case_indices=empty_case_indices,
        valid_case_indices=valid_case_indices,
    )
    manifest = _native_perf_manifest_document(
        binding,
        fixture_sha256=fixture_sha256,
        input_adapter=input_adapter,
        case_records=case_records,
        case_count=case_count,
        valid_case_indices=valid_case_indices,
        empty_case_indices=empty_case_indices,
    )
    manifest_path = adapter / NATIVE_PERF_MANIFEST_FILENAME
    _atomic_json(manifest_path, manifest)
    if write_adapter_manifest:
        _decorate_adapter_manifest(adapter, manifest_path, fixture_sha256)
    return manifest


def _native_perf_manifest_document(
    binding: Mapping[str, Any],
    *,
    fixture_sha256: str,
    input_adapter: Mapping[str, Any],
    case_records: list[dict[str, Any]],
    case_count: int,
    valid_case_indices: list[int],
    empty_case_indices: list[int],
) -> dict[str, Any]:
    """Return the public native-bridge handoff manifest for one fixture."""
    return {
        "schema": NATIVE_PERF_MANIFEST_SCHEMA,
        "fixture_path": NATIVE_PERF_FIXTURE_FILENAME,
        "fixture_sha256": fixture_sha256,
        "fixture_schema": NATIVE_PERF_FIXTURE_SCHEMA,
        "binding_sha256": binding["binding_sha256"],
        "task_sha256": binding["task_sha256"],
        "candidate_tree_sha256": binding.get("candidate_tree_sha256"),
        "candidate_entry_sha256": binding.get("candidate_entry_sha256"),
        "case_count": case_count,
        "valid_case_indices": valid_case_indices,
        "empty_case_indices": empty_case_indices,
        "init_call_style": "args",
        "input_adapter": input_adapter,
        "case_fixtures": case_records,
    }


def _native_fixture_realization(
    bundle: StagedBundle, candidate_dir: Path, torch: Any, *, seed: int
) -> tuple[Any, Sequence[Any], Any, Mapping[str, Any]]:
    """Seed once, then produce the init arguments and reference input groups.

    Match the selected NPUKernelBench verifier: candidate construction
    arguments take precedence, then reference input groups are generated
    before either model is constructed.  Do not transfer reference state to
    the candidate: optimized wrappers legitimately have a different state
    layout, and the verifier does not perform such a transfer.
    """
    reference_module = load_task_module(bundle.task_path, bundle.root, role="reference")
    candidate_root = _candidate_root(candidate_dir)
    candidate_entry = _candidate_entry(candidate_root)
    candidate_module = load_task_module(candidate_entry, candidate_root, role="candidate")
    _validate_reference_api(reference_module)
    seed_events = seed_everything(seed, torch_module=torch)
    init_args = _get_init_args(candidate_module, fallback_module=reference_module)
    groups, input_adapter = _resolve_input_groups(
        reference_module,
        bundle=bundle,
        torch=torch,
        seed=seed,
    )
    return seed_events, init_args, groups, input_adapter


def run_precision_workspace(
    workspace: Path,
    candidate_dir: Path,
    *,
    device: int | str | None = None,
    seed: int = DEFAULT_SEED,
    isolated: bool = True,
    subprocess_run: Callable[..., Any] = subprocess.run,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run precision in a scrubbed child process and write durable evidence."""
    workspace = Path(workspace)
    timeout_seconds = _resolve_task_execution_timeout(timeout_seconds)
    context: _ExecutionContext | None = None
    try:
        if isolated:
            bundle = resolve_staged_bundle(workspace)
            binding = build_evaluation_binding(workspace, candidate_dir, bundle=bundle)
            context = _create_execution_context(
                workspace,
                bundle=bundle,
                candidate_dir=Path(candidate_dir),
                binding=binding,
                verb="precision",
                device=device,
                seed=seed,
            )
            result = _run_isolated_context(
                context, subprocess_run=subprocess_run, timeout_seconds=timeout_seconds
            )
            _validate_parent_published_child_report(
                result, binding, verb="precision", expected_run_id=context.run_id
            )
            _verify_parent_binding_unchanged(workspace, Path(candidate_dir), binding)
        else:
            result = _run_precision_workspace_in_process(
                workspace, Path(candidate_dir), device=device, seed=seed
            )
            result["execution_isolation"] = "in_process_test_only"
            result["tamper_protection"] = "none"
    except (NpuBenchRunnerError, OSError, ValueError, subprocess.SubprocessError) as exc:
        result = _base_report("precision", status="ERROR")
        result["reason"] = str(exc)
    # Normalize non-finite metric floats (inf/nan from a diverged candidate)
    # before the report leaves this function: persistence uses allow_nan=False,
    # and finalize compares the returned report against the on-disk copy, so
    # the in-memory and persisted forms must already be identical here.
    _update_precision_binding_history(workspace, result)
    result = _json_safe_non_finite(result)
    _try_write_report(workspace, PRECISION_REPORT_FILENAME, result)
    if context is not None:
        _cleanup_execution_context(context)
    return result


def _run_precision_workspace_in_process(
    workspace: Path,
    candidate_dir: Path,
    *,
    device: int | str | None,
    seed: int,
) -> dict[str, Any]:
    """Internal child-only precision implementation that imports task code."""
    workspace = Path(workspace)
    run_id = uuid.uuid4().hex
    try:
        bundle = resolve_staged_bundle(workspace)
        binding = build_evaluation_binding(workspace, candidate_dir, bundle=bundle)
        result = _run_precision(bundle, Path(candidate_dir), device=device, seed=seed, binding=binding)
    except (NpuBenchRunnerError, OSError, ValueError, ImportError) as exc:
        result = _base_report("precision", status="ERROR", run_id=run_id)
        result["reason"] = str(exc)
    result.setdefault("run_id", run_id)
    return result


def _run_precision(
    bundle: StagedBundle,
    candidate_dir: Path,
    *,
    device: int | str | None,
    seed: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    torch = _import_torch()
    before_candidate_digest = str(binding["candidate_tree_sha256"])
    reference_module = load_task_module(bundle.task_path, bundle.root, role="reference")
    candidate_root = _candidate_root(candidate_dir)
    candidate_path = _candidate_entry(candidate_root)
    candidate_module = load_task_module(candidate_path, candidate_root, role="candidate")
    _validate_reference_api(reference_module)
    reference_ctor = _resolve_model_constructor(reference_module, preferred="Model", role="reference")
    candidate_ctor = _resolve_model_constructor(candidate_module, preferred="ModelNew", role="candidate")

    seed_events = seed_everything(seed, torch_module=torch)
    # Match the selected NPUKernelBench verifier: candidate construction
    # arguments take precedence, with the reference provider as fallback.
    # Both variants receive independently cloned copies in _construct_model.
    init_args = _get_init_args(candidate_module, fallback_module=reference_module)
    # The source verifier generates the reference-owned inputs before model
    # construction.  That ordering matters for old tasks whose generator
    # consumes PyTorch RNG, including level1/3_Add.
    groups, input_adapter = _resolve_input_groups(
        reference_module,
        bundle=bundle,
        torch=torch,
        seed=seed,
    )
    _assert_input_adapter_binding(input_adapter, binding)
    reference_model, candidate_model, device_value = _construct_precision_models(
        torch, reference_ctor, candidate_ctor, init_args, device
    )
    case_count = len(groups)
    repeats = _resolve_precision_repeats()
    overflow_threshold = _resolve_reference_overflow_threshold()
    case_reports, abort_reason = _run_precision_cases_with_fingerprint(
        groups,
        torch,
        reference_model,
        candidate_model,
        device_value,
        repeats=repeats,
        overflow_threshold=overflow_threshold,
    )

    if _candidate_tree_sha256(candidate_root) != before_candidate_digest:
        raise NpuBenchRunnerError("candidate tree changed during precision evaluation")
    return _precision_report(
        case_reports,
        binding,
        seed=seed,
        seed_events=seed_events,
        input_adapter=input_adapter,
        device_value=device_value,
        case_count=case_count,
        repeats=repeats,
        overflow_threshold=overflow_threshold,
        abort_reason=abort_reason,
    )


def _construct_precision_models(
    torch: Any,
    reference_ctor: Callable[..., Any],
    candidate_ctor: Callable[..., Any],
    init_args: Sequence[Any],
    device: int | str | None,
) -> tuple[Any, Any, Any]:
    """Build both models from one init payload and place them on one device."""
    reference_model = _construct_model(reference_ctor, init_args, "reference")
    candidate_model = _construct_model(candidate_ctor, init_args, "candidate")

    device_value = _resolve_device(torch, device)
    _move_model(reference_model, device_value, "reference")
    _move_model(candidate_model, device_value, "candidate")
    _set_eval(reference_model)
    _set_eval(candidate_model)
    return reference_model, candidate_model, device_value


def build_performance_command(
    adapter_dir: Path,
    *,
    device: int,
    run_id: str,
    profiler_script: Path | None = None,
) -> list[str]:
    """Build the fixed quick W3/R5/keep-profile command without executing it.

    The NPUKernelBench adapter supplies frozen inputs through generated local
    shims, then delegates timing and CSV parsing to the existing repository
    quick profiler engine.  The generic shell launcher is intentionally not
    used here: its current quick branch drops ``--keep-prof`` before it
    reaches that engine.
    """
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise NpuBenchRunnerError("performance device must be a non-negative integer")
    if not _safe_prof_tag(run_id):
        raise NpuBenchRunnerError("run_id is not safe for --prof-tag")
    script = Path(profiler_script) if profiler_script is not None else _default_profiler_summary()
    _require_regular(script, "msprof profiler summary script")
    command = [
        str(sys.executable),
        str(script),
        "--quick",
        "--warmup",
        str(WARM_UP),
        "--device",
        str(device),
        "--keep-prof",
        "--repeats",
        str(REPEATS),
        "--output-dir",
        str(Path(adapter_dir)),
        "--prof-tag",
        run_id,
    ]
    return command


def run_performance_workspace(
    workspace: Path,
    candidate_dir: Path,
    *,
    device: int,
    lease_manifest: Path | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
    profiler_script: Path | None = None,
    timeout_seconds: int | None = None,
    isolated: bool = True,
) -> dict[str, Any]:
    """Run the fixed quick profiler through two independent child processes.

    The public default never imports a benchmark task in the controller
    process.  It resolves/freeze-binds inputs outside the child, then lets the
    child generate a native fixture and invoke the repository profiler.  Only
    after it exits does this parent validate fixture/candidate hashes and
    complete coverage, archive the exact tagged raw directories and publish
    evidence.  This is a practical process-boundary workflow, not an
    adversarial same-UID sandbox guarantee.
    """
    workspace = Path(workspace)
    timeout_seconds = _resolve_task_execution_timeout(timeout_seconds)
    lanes = _PerformanceLanes(run_id=uuid.uuid4().hex)
    try:
        bundle = resolve_staged_bundle(workspace)
        binding = build_evaluation_binding(workspace, candidate_dir, bundle=bundle)
        _validate_requested_lease(lease_manifest, role="performance", device=device)
        if isolated:
            report = _run_isolated_performance(
                workspace,
                Path(candidate_dir),
                bundle,
                binding,
                lanes,
                device=device,
                lease_manifest=lease_manifest,
                subprocess_run=subprocess_run,
                profiler_script=profiler_script,
                timeout_seconds=timeout_seconds,
            )
        else:
            report = _run_performance_workspace_in_process(
                workspace,
                Path(candidate_dir),
                binding=binding,
                bundle=bundle,
                device=device,
                lease_manifest=lease_manifest,
                subprocess_run=subprocess_run,
                profiler_script=profiler_script,
                timeout_seconds=timeout_seconds,
                run_id=lanes.run_id,
            )
    except (NpuBenchRunnerError, OSError, ValueError, subprocess.SubprocessError) as exc:
        report = _performance_error_report(lanes.run_id, exc)
    # Same non-finite normalization as the precision lane: the returned report
    # must be byte-identical in content to the persisted evidence file.
    report = _json_safe_non_finite(report)
    _try_write_report(workspace, PERFORMANCE_REPORT_FILENAME, report)
    _cleanup_performance_lanes(workspace, lanes)
    return report


@dataclass
class _PerformanceLanes:
    """Parent-owned resources one performance run has to release afterwards."""

    run_id: str
    fixture_context: _ExecutionContext | None = None
    context: _ExecutionContext | None = None
    frozen_fixture: Path | None = None


def _run_isolated_performance(
    workspace: Path,
    candidate_dir: Path,
    bundle: StagedBundle,
    binding: Mapping[str, Any],
    lanes: _PerformanceLanes,
    *,
    device: int,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    profiler_script: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Drive the two-child fixture/profiler workflow, recording what to clean."""
    _assert_repository_profiler_script(profiler_script)
    # Phase A reads the frozen task/candidate once to materialize the
    # source-verifier initialization and input fixture.  Its writable
    # scratch is never reused by the profiler phase.
    lanes.fixture_context = _performance_child_context(
        workspace, bundle, binding, candidate_dir, verb="fixture"
    )
    lanes.frozen_fixture, native = _run_native_fixture_phase(
        workspace,
        candidate_dir,
        binding,
        lanes.fixture_context,
        subprocess_run=subprocess_run,
        timeout_seconds=timeout_seconds,
    )
    # Phase B gets a fresh adapter/tmp root and an adapter-local
    # byte-identical fixture copy.  This is process isolation plus
    # post-run hash verification, not a same-UID mount boundary.
    lanes.context = _performance_child_context(
        workspace, bundle, binding, candidate_dir, verb="performance",
        device=device, native_fixture_root=lanes.frozen_fixture,
    )
    lanes.run_id = lanes.context.run_id
    return _run_native_profiler_phase(
        workspace,
        candidate_dir,
        binding,
        lanes.context,
        native=native,
        frozen_fixture=lanes.frozen_fixture,
        device=device,
        lease_manifest=lease_manifest,
        subprocess_run=subprocess_run,
        timeout_seconds=timeout_seconds,
    )


def _performance_child_context(
    workspace: Path,
    bundle: StagedBundle,
    binding: Mapping[str, Any],
    candidate_dir: Path,
    *,
    verb: str,
    device: int | None = None,
    native_fixture_root: Path | None = None,
) -> _ExecutionContext:
    """Create one performance-lane child context at the shared default seed."""
    return _create_execution_context(
        workspace,
        bundle=bundle,
        candidate_dir=candidate_dir,
        binding=binding,
        verb=verb,
        device=device,
        seed=DEFAULT_SEED,
        native_fixture_root=native_fixture_root,
    )


def _performance_error_report(run_id: str, exc: BaseException) -> dict[str, Any]:
    """Build the lane report for a run that never reached a measurement."""
    report = _base_report("performance", status="ERROR", run_id=run_id)
    report.update(
        {
            "reason": str(exc),
            "warm_up": WARM_UP,
            "repeats": REPEATS,
            "keep_prof": True,
            "profiling_mode": "quick",
            "profile_archive": None,
            "profile_tree_sha256": None,
        }
    )
    return report


def _cleanup_performance_lanes(workspace: Path, lanes: _PerformanceLanes) -> None:
    """Release both child execution contexts and the parent-frozen fixture."""
    if lanes.context is not None:
        _cleanup_execution_context(lanes.context)
    if lanes.fixture_context is not None:
        _cleanup_execution_context(lanes.fixture_context)
    if lanes.frozen_fixture is not None:
        _cleanup_frozen_native_fixture(workspace, lanes.frozen_fixture)


def _assert_repository_profiler_script(profiler_script: Path | None) -> None:
    """Reject any profiler engine other than the repository msprof summary."""
    if profiler_script is None:
        return
    if Path(profiler_script).resolve() != _default_profiler_summary().resolve():
        raise NpuBenchRunnerError(
            "strict NPUKernelBench performance requires the repository msprof summary engine"
        )


def _run_native_fixture_phase(
    workspace: Path,
    candidate_dir: Path,
    binding: Mapping[str, Any],
    fixture_context: _ExecutionContext,
    *,
    subprocess_run: Callable[..., Any],
    timeout_seconds: int,
) -> tuple[Path, Mapping[str, Any]]:
    """Run phase A and publish its output as a parent-owned read-only fixture."""
    fixture_child = _run_isolated_context(
        fixture_context,
        subprocess_run=subprocess_run,
        timeout_seconds=timeout_seconds,
    )
    _validate_parent_published_child_report(
        fixture_child,
        binding,
        verb="fixture",
        expected_run_id=fixture_context.run_id,
    )
    if int(fixture_child.get("child_returncode", 1)) != 0 or fixture_child.get("status") != "PASS":
        raise NpuBenchRunnerError(
            "isolated native fixture stage did not complete successfully: "
            + _safe_child_failure_reason(fixture_child)
        )
    _verify_parent_binding_unchanged(workspace, candidate_dir, binding)
    return _freeze_native_perf_fixture(
        workspace,
        fixture_context.scratch / "native_fixture",
        binding=binding,
        token=f"fixture-{fixture_context.run_id}",
    )


def _failed_native_profiler_report(
    child: Mapping[str, Any], binding: Mapping[str, Any], run_id: str, common: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the performance report for a child that did not measure."""
    status = "ERROR" if child.get("status") == "ERROR" else "FAIL"
    report = _base_report("performance", status=status, binding=binding, run_id=run_id)
    report.update(common)
    report["reason"] = _safe_child_failure_reason(child)
    report["profile_archive"] = None
    report["profile_tree_sha256"] = None
    report["profiler_summary"] = None
    return report


def _passed_native_profiler_report(
    workspace: Path,
    context: _ExecutionContext,
    binding: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    run_id: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the adapter fixture, archive profiles and publish the result."""
    adapter = _performance_adapter_path(context)
    adapter_native = _load_native_perf_manifest(
        adapter / "native_fixture", binding=binding, verify_case_payloads=True
    )
    if adapter_native != native:
        raise NpuBenchRunnerError("adapter native fixture changed during performance execution")
    summary = _load_quick_profiler_summary(
        adapter,
        expected_case_count=int(native["case_count"]),
        expected_valid_case_indices=list(native["valid_case_indices"]),
        run_id=run_id,
        profile_root=Path("/tmp"),
        native=native,
    )
    profile_archive, profile_digest = _archive_retained_profiles(
        workspace,
        adapter,
        run_id,
        summary,
        expected_valid_case_indices=list(native["valid_case_indices"]),
        profile_root=Path("/tmp"),
        source_profile_root=Path("/tmp"),
    )
    report = _base_report("performance", status="PASS", binding=binding, run_id=run_id)
    report.update(common)
    below_threshold = _perf_below_threshold(summary)
    report.update(
        {
            "profile_archive": profile_archive,
            "profile_tree_sha256": profile_digest,
            "profiler_summary": dict(summary),
            "native_fixture": dict(native),
            "measurement_completed": True,
            "perf_ratio_below_threshold": below_threshold,
            "perf_pending_optimization": below_threshold,
        }
    )
    return report


def _run_native_profiler_phase(
    workspace: Path,
    candidate_dir: Path,
    binding: Mapping[str, Any],
    context: _ExecutionContext,
    *,
    native: Mapping[str, Any],
    frozen_fixture: Path,
    device: int,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run phase B and re-verify every frozen input before publishing evidence."""
    run_id = context.run_id
    child = _run_isolated_context(
        context,
        subprocess_run=subprocess_run,
        timeout_seconds=timeout_seconds,
    )
    _validate_parent_published_child_report(
        child, binding, verb="performance", expected_run_id=run_id
    )
    _verify_parent_binding_unchanged(workspace, candidate_dir, binding)
    frozen_after = _load_native_perf_manifest(
        _require_parent_frozen_native_fixture(workspace, frozen_fixture),
        binding=binding,
        verify_case_payloads=True,
    )
    if frozen_after != native:
        raise NpuBenchRunnerError(
            "parent-frozen native fixture changed during performance execution"
        )
    returncode = int(child.get("child_returncode", 1))
    command = _parent_performance_command(device=device, run_id=run_id)
    common = _parent_performance_report_fields(
        device=device,
        lease_manifest=lease_manifest,
        command=command,
        returncode=returncode,
    )
    if returncode != 0 or child.get("status") != "PASS":
        return _failed_native_profiler_report(child, binding, run_id, common)
    return _passed_native_profiler_report(
        workspace, context, binding, native, run_id=run_id, common=common
    )


def _run_performance_workspace_in_process(
    workspace: Path,
    candidate_dir: Path,
    *,
    binding: Mapping[str, Any],
    bundle: StagedBundle,
    device: int,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    profiler_script: Path | None,
    timeout_seconds: int,
    run_id: str,
) -> dict[str, Any]:
    """Test-only non-process-boundary path; finalization rejects its marker."""
    if timeout_seconds <= 0:
        raise NpuBenchRunnerError("task execution timeout must be positive")
    adapter = prepare_adapter_view(workspace, candidate_dir, bundle=bundle, binding=binding, run_id=run_id)
    native = _materialize_native_perf_fixture(
        adapter, bundle, candidate_dir, binding=binding, seed=DEFAULT_SEED
    )
    native = _prepare_native_quick_adapter(adapter, adapter, binding=binding)
    command = build_performance_command(
        adapter,
        device=device,
        run_id=run_id,
        profiler_script=profiler_script,
    )
    completed = subprocess_run(
        command,
        cwd=str(adapter),
        text=True,
        capture_output=True,
        check=False,
        env=_scrubbed_task_environment(),
        timeout=timeout_seconds,
    )
    returncode = int(getattr(completed, "returncode", 1))
    report = _in_process_base_report(
        binding, run_id, device=device, lease_manifest=lease_manifest,
        command=command, returncode=returncode,
    )
    if returncode != 0:
        report["reason"] = "test-only profiler subprocess failed"
        return report
    report.update(_in_process_measurement_fields(workspace, adapter, binding, run_id))
    return report


def _in_process_base_report(
    binding: Mapping[str, Any],
    run_id: str,
    *,
    device: int,
    lease_manifest: Path | None,
    command: Sequence[str],
    returncode: int,
) -> dict[str, Any]:
    """Start the test-only lane's report; it is FAIL until a measurement lands."""
    report = _base_report("performance", status="FAIL", binding=binding, run_id=run_id)
    report.update(
        {
            "device": device,
            "lease_manifest": str(lease_manifest) if lease_manifest else None,
            "profiling_mode": "quick",
            "warm_up": WARM_UP,
            "repeats": REPEATS,
            "keep_prof": True,
            "command": command,
            "returncode": returncode,
            "child_returncode": returncode,
            "execution_isolation": "in_process_test_only",
            "tamper_protection": "none",
            "profile_archive": None,
            "profile_tree_sha256": None,
            "profiler_summary": None,
        }
    )
    return report


def _in_process_measurement_fields(
    workspace: Path, adapter: Path, binding: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    """Summarize and archive the test-only lane's completed measurement."""
    native_info = _load_native_perf_manifest(adapter, binding=binding)
    summary = _load_quick_profiler_summary(
        adapter,
        expected_case_count=int(native_info["case_count"]),
        expected_valid_case_indices=list(native_info["valid_case_indices"]),
        run_id=run_id,
        profile_root=Path("/tmp"),
        native=native_info,
    )
    archive, digest = _archive_retained_profiles(
        workspace,
        adapter,
        run_id,
        summary,
        expected_valid_case_indices=list(native_info["valid_case_indices"]),
        profile_root=Path("/tmp"),
        source_profile_root=Path("/tmp"),
    )
    below_threshold = _perf_below_threshold(summary)
    return {
        "status": "PASS",
        "profile_archive": archive,
        "profile_tree_sha256": digest,
        "profiler_summary": dict(summary),
        "native_fixture": dict(native_info),
        "measurement_completed": True,
        "perf_ratio_below_threshold": below_threshold,
        "perf_pending_optimization": below_threshold,
    }


def evaluate_workspace(
    workspace: Path,
    candidate_dir: Path,
    *,
    precision_device: int | str | None,
    performance_device: int | None,
    lease_manifest: Path | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
    profiler_script: Path | None = None,
) -> dict[str, Any]:
    """Evaluate on assigned lanes, parallel only with two valid distinct leases."""
    workspace = Path(workspace)
    run_id = uuid.uuid4().hex
    try:
        _assert_evaluate_devices(precision_device, performance_device)
        lease_ready = _validate_parallel_leases(lease_manifest, precision_device, performance_device)
        # Read skip-perf before the lane fork (2026-08-25): the deferred
        # precision-first path must win even when two valid leases would
        # otherwise take the parallel branch.
        skip_perf = os.environ.get(PERF_SKIP_ENV, "0") == "1"
        parallelism, precision, performance = _run_evaluation_lanes(
            workspace,
            candidate_dir,
            precision_device=precision_device,
            performance_device=performance_device,
            lease_manifest=lease_manifest,
            subprocess_run=subprocess_run,
            profiler_script=profiler_script,
            lease_ready=lease_ready,
            skip_perf=skip_perf,
            run_id=run_id,
        )
        report = _evaluate_aggregate_report(
            precision, performance, run_id=run_id, parallelism=parallelism,
            lease_manifest=lease_manifest,
        )
    except (NpuBenchRunnerError, OSError, ValueError) as exc:
        report = _base_report("evaluate", status="ERROR", run_id=run_id)
        report.update({"reason": str(exc), "parallelism": "not_started"})
    # Keep the returned aggregate identical to the persisted file even when a
    # lane report carries non-finite metrics (finalize compares both forms).
    report = _json_safe_non_finite(report)
    _try_write_report(workspace, EVALUATE_REPORT_FILENAME, report)
    return report


def _run_evaluation_lanes(
    workspace: Path,
    candidate_dir: Path,
    *,
    precision_device: int,
    performance_device: int | None,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    profiler_script: Path | None,
    lease_ready: bool,
    skip_perf: bool,
    run_id: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Pick the lane strategy and run both lanes, returning what was used."""
    run_parallel = lease_ready and precision_device != performance_device and not skip_perf
    if run_parallel:
        precision, performance = _run_evaluation_lanes_in_parallel(
            workspace,
            candidate_dir,
            precision_device=precision_device,
            performance_device=performance_device,
            lease_manifest=lease_manifest,
            subprocess_run=subprocess_run,
            profiler_script=profiler_script,
        )
        return "parallel_two_lane", precision, performance
    precision, performance = _run_evaluation_lanes_in_sequence(
        workspace,
        candidate_dir,
        precision_device=precision_device,
        performance_device=performance_device,
        lease_manifest=lease_manifest,
        subprocess_run=subprocess_run,
        profiler_script=profiler_script,
        skip_perf=skip_perf,
        run_id=run_id,
    )
    return ("precision_only" if skip_perf else "degraded_single_lane"), precision, performance


def _assert_evaluate_devices(precision_device: Any, performance_device: Any) -> None:
    """Reject an evaluate call that did not explicitly assign both lanes."""
    if precision_device is None or performance_device is None:
        raise NpuBenchRunnerError("evaluate requires explicitly assigned precision and performance devices")
    if not isinstance(precision_device, int) or isinstance(precision_device, bool):
        raise NpuBenchRunnerError("precision_device must be an integer in evaluate")


def _evaluate_aggregate_report(
    precision: Mapping[str, Any],
    performance: Mapping[str, Any],
    *,
    run_id: str,
    parallelism: str,
    lease_manifest: Path | None,
) -> dict[str, Any]:
    """Combine both lane reports into the published evaluate aggregate."""
    binding_sha = precision.get("binding_sha256")
    status, reason = _evaluate_lane_verdict(precision, performance, binding_sha)
    report = _base_report("evaluate", status=status, run_id=run_id)
    report.update(
        {
            "binding_sha256": binding_sha,
            "evaluation_binding": precision.get("evaluation_binding"),
            "parallelism": parallelism,
            "lease_manifest": str(lease_manifest) if lease_manifest else None,
            "precision": precision,
            "performance": performance,
        }
    )
    if reason:
        report["reason"] = reason
    return report


def _run_evaluation_lanes_in_parallel(
    workspace: Path,
    candidate_dir: Path,
    *,
    precision_device: int,
    performance_device: int,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    profiler_script: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run both lanes concurrently; only two valid distinct leases allow this."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        precision_future = executor.submit(
            run_precision_workspace, workspace, candidate_dir, device=precision_device
        )
        performance_future = executor.submit(
            run_performance_workspace,
            workspace,
            candidate_dir,
            device=performance_device,
            lease_manifest=lease_manifest,
            subprocess_run=subprocess_run,
            profiler_script=profiler_script,
        )
        precision = precision_future.result()
        performance = performance_future.result()
    return precision, performance


def _run_evaluation_lanes_in_sequence(
    workspace: Path,
    candidate_dir: Path,
    *,
    precision_device: int,
    performance_device: int | None,
    lease_manifest: Path | None,
    subprocess_run: Callable[..., Any],
    profiler_script: Path | None,
    skip_perf: bool,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run precision first, then either defer or run the performance lane."""
    precision = run_precision_workspace(workspace, candidate_dir, device=precision_device)
    if not skip_perf and _precision_host_error_only(precision):
        # Fail-fast (source-migration flow 2026-08-22): a candidate whose
        # op is unregistered / import-broken fails EVERY precision case
        # with a host-side error.  The msprof perf sweep on such a
        # candidate hangs for tens of minutes without producing
        # evidence — skip it and surface a synthetic performance
        # report so finalize can route back to await_worker promptly.
        performance = _synthesized_skipped_performance(
            precision, workspace=workspace, run_id=run_id,
            device=performance_device,
        )
        return precision, performance
    if not skip_perf:
        performance = run_performance_workspace(
            workspace,
            candidate_dir,
            device=performance_device,
            lease_manifest=lease_manifest,
            subprocess_run=subprocess_run,
            profiler_script=profiler_script,
        )
        return precision, performance
    performance = _deferred_performance_report(
        binding=precision.get("evaluation_binding") or {},
        run_id=run_id,
        device=performance_device,
    )
    # run_performance_workspace persists its own report file; the deferred
    # placeholder must do the same, otherwise the finalize contract reads a
    # STALE performance_report.json from a previous attempt (2026-08-22: O5
    # kept reporting the old wave's perf timeout even though perf was
    # skipped).
    _try_write_report(workspace, PERFORMANCE_REPORT_FILENAME, performance)
    return precision, performance


def _resolve_input_groups(
    module: types.ModuleType,
    *,
    bundle: StagedBundle,
    torch: Any,
    seed: int,
) -> tuple[Sequence[Any], Mapping[str, Any]]:
    """Resolve task-owned inputs or the strict native sidecar descriptor.

    Validation is deliberately completed for every case before allocating any
    tensor.  This prevents a malformed late case from leaving a large partial
    fixture in the evaluator scratch directory.
    """
    api = _validate_reference_api(module)
    provider_kind = api.get("input_provider")
    # Narrow unit doubles may intentionally return a reduced API mapping;
    # real task modules are always classified by _validate_reference_api.
    if provider_kind is None:
        provider_kind = "get_input_groups"
    if provider_kind != "sidecar_descriptor":
        groups = _get_input_groups(module)
        return groups, _input_adapter_identity(provider_kind, case_count=None)
    descriptors = _validate_sidecar_descriptors(bundle.sidecar_cases)
    groups = _SidecarInputGroups(descriptors, torch, seed)
    return groups, _input_adapter_identity("sidecar_descriptor", case_count=len(groups))


def _create_execution_context(
    workspace: Path,
    *,
    bundle: StagedBundle,
    candidate_dir: Path | None,
    binding: Mapping[str, Any],
    verb: str,
    device: int | str | None = None,
    seed: int | None = None,
    native_fixture_root: Path | None = None,
) -> _ExecutionContext:
    """Create a parent-owned request/scratch area without mounting workspace."""
    workspace, run_id, root, scratch, temp_root = _prepare_execution_root(workspace)
    # Resolve before copying this runner.  The copied child purposely has no
    # engine-tree dependency and therefore cannot discover workspace/.ascendc_env
    # by walking up from its temporary execution directory.
    target_python = _resolve_target_python()
    runner_root = _stage_execution_runner_copy(root)
    candidate_root = _candidate_root(candidate_dir) if candidate_dir is not None else None
    if candidate_root is not None:
        _candidate_entry(candidate_root)
    profiler_script, native_fixture_source, native_fixture_mount = _stage_execution_inputs(
        workspace, scratch, native_fixture_root, binding=binding, verb=verb, run_id=run_id
    )
    request = _build_execution_request(
        bundle, binding, candidate_root,
        verb=verb, run_id=run_id, device=device, seed=seed, scratch=scratch,
        profiler_script=profiler_script,
        native_fixture_source=native_fixture_source,
        native_fixture_mount=native_fixture_mount,
    )
    return _ExecutionContext(
        request_path=_publish_execution_request(root, request),
        root=root,
        runner_root=runner_root,
        scratch=scratch,
        tmp=temp_root,
        bundle=bundle,
        candidate_root=candidate_root,
        binding=dict(binding),
        run_id=run_id,
        verb=verb,
        device=device,
        seed=seed,
        target_python=target_python,
        native_fixture_source=native_fixture_source,
        native_fixture_mount=native_fixture_mount,
    )


def _stage_execution_inputs(
    workspace: Path,
    scratch: Path,
    native_fixture_root: Path | None,
    *,
    binding: Mapping[str, Any],
    verb: str,
    run_id: str,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve the profiler engine and stage the child's fixture copy.

    The shared profiler engine is needed only by the performance child.  Do
    not resolve it while preparing preflight/precision/fixture contexts: those
    phases intentionally work from the self-contained plugin bundle and must
    remain usable when the optional root-repository profiler tree is absent
    (notably for local O2.5 preflight).
    """
    profiler_script = _default_profiler_summary() if verb == "performance" else None
    if native_fixture_root is None:
        return profiler_script, None, None
    native_fixture_source, native_fixture_mount = _stage_isolated_native_fixture(
        workspace,
        scratch,
        native_fixture_root,
        binding=binding,
        run_id=run_id,
    )
    return profiler_script, native_fixture_source, native_fixture_mount


def _publish_execution_request(root: Path, request: Mapping[str, Any]) -> Path:
    """Write the child's request document and freeze it read-only."""
    request_path = root / "request.json"
    _atomic_json(request_path, request)
    os.chmod(request_path, 0o400)
    return request_path


def _prepare_execution_root(workspace: Path) -> tuple[Path, str, Path, Path, Path]:
    """Create the parent-owned execution root, scratch and tmp for one child."""
    supplied_workspace = Path(workspace)
    if supplied_workspace.is_symlink():
        raise NpuBenchRunnerError("workspace must be a real non-symlink directory")
    workspace = supplied_workspace.resolve()
    _require_real_directory(workspace, "workspace")
    run_id = uuid.uuid4().hex
    execution_parent = _workspace_runtime_directory(
        workspace, EXECUTION_DIRNAME, "execution root"
    )
    root = _create_real_child_directory(execution_parent, run_id, "execution context")
    scratch = _create_real_child_directory(root, "scratch", "execution scratch")
    temp_root = _create_real_child_directory(root, "tmp", "execution tmp")
    return workspace, run_id, root, scratch, temp_root


def _stage_execution_runner_copy(root: Path) -> Path:
    """Copy the evaluator modules and the stage verifier next to the request.

    The evaluator, the modules it is split across and its immutable-stage
    verifier are copied together so a child can validate the frozen bundle
    without depending on the controller checkout or its ``PYTHONPATH``.  Keep
    those bytes with the execution record, independent of concurrent edits to
    the engine checkout.
    """
    runner_root = _create_real_child_directory(root, "runner", "execution runner")
    package_dir = Path(__file__).resolve().parent
    for name in RUNNER_MODULE_FILENAMES:
        source = package_dir / name
        _require_regular(source, f"npubench runner module {name}")
        destination = runner_root / name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o400)
    os.chmod(runner_root, 0o500)
    return runner_root


def _stage_isolated_native_fixture(
    workspace: Path,
    scratch: Path,
    native_fixture_root: Path,
    *,
    binding: Mapping[str, Any],
    run_id: str,
) -> tuple[Path, Path]:
    """Give the child an adapter-local read-only copy of the frozen fixture.

    The plugin-local shim reads an adapter-local frozen fixture.  This copy is
    byte-identical and read-only; it is *not* a same-UID anti-tamper boundary,
    so the parent later revalidates both this copy and the parent-frozen
    source by hash.
    """
    native_fixture_source = _require_parent_frozen_native_fixture(workspace, native_fixture_root)
    adapter_root = _ensure_real_child_directory(
        scratch, ".npubench_adapter", "isolated adapter root"
    )
    binding_root = _ensure_real_child_directory(
        adapter_root, str(binding["binding_sha256"]), "isolated adapter binding root"
    )
    adapter_mount = _create_real_child_directory(
        binding_root, run_id, "isolated adapter root"
    )
    native_fixture_mount = _create_real_child_directory(
        adapter_mount, "native_fixture", "isolated native fixture root"
    )
    _copy_frozen_native_fixture(native_fixture_source, native_fixture_mount)
    return native_fixture_source, native_fixture_mount


def _build_execution_request(
    bundle: StagedBundle,
    binding: Mapping[str, Any],
    candidate_root: Path | None,
    *,
    verb: str,
    run_id: str,
    device: int | str | None,
    seed: int | None,
    scratch: Path,
    profiler_script: Path | None,
    native_fixture_source: Path | None,
    native_fixture_mount: Path | None,
) -> dict[str, Any]:
    """Author the child's state-free request document.

    Practical anti-cheat deliberately uses a normal independent child process
    rather than requiring a host-specific mount namespace.  The request
    therefore names exact parent-authored host paths; their input and
    candidate digests are checked before/after the child returns.
    """
    return {
        "schema": "cannbot.npubench.execution_request/v1",
        "verb": verb,
        "run_id": run_id,
        "bundle": _request_bundle_document(bundle),
        "candidate": _request_candidate_document(binding, candidate_root),
        "binding": dict(binding),
        "device": device,
        "seed": seed,
        "scratch": str(scratch),
        "profiler_script": (
            str(profiler_script) if profiler_script is not None else None
        ),
        "native_fixture": _request_fixture_document(native_fixture_source, native_fixture_mount),
    }


def _request_bundle_document(bundle: StagedBundle) -> dict[str, Any]:
    """Name the frozen task/sidecar the child must reproduce byte for byte."""
    return {
        "root": str(bundle.root),
        "manifest_path": str(bundle.root / "bundle_manifest.json"),
        "manifest_sha256": _file_sha256(bundle.manifest_path),
        "task_relative_path": str(bundle.task_path.relative_to(bundle.root)),
        "task_sha256": _file_sha256(bundle.task_path),
        "sidecar_relative_path": str(bundle.sidecar_path.relative_to(bundle.root)),
        "sidecar_sha256": _file_sha256(bundle.sidecar_path),
        "sidecar_encoding": bundle.sidecar_encoding,
    }


def _request_candidate_document(
    binding: Mapping[str, Any], candidate_root: Path | None
) -> dict[str, Any] | None:
    """Name the candidate snapshot, or ``None`` for a candidate-free verb."""
    if candidate_root is None:
        return None
    return {
        "root": str(candidate_root),
        "tree_sha256": binding.get("candidate_tree_sha256"),
        "entry": "model_new_ascendc.py",
        "entry_sha256": binding.get("candidate_entry_sha256"),
    }


def _request_fixture_document(
    native_fixture_source: Path | None, native_fixture_mount: Path | None
) -> dict[str, Any] | None:
    """Name the parent-frozen fixture copy, or ``None`` when there is none."""
    if native_fixture_source is None or native_fixture_mount is None:
        return None
    return {
        "root": str(native_fixture_mount),
        "tree_sha256": tree_sha256(native_fixture_source),
        "manifest_sha256": _file_sha256(native_fixture_source / NATIVE_PERF_MANIFEST_FILENAME),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.verb == "internal-preflight":
        report = _preflight_workspace_in_process(args.workspace)
    elif args.verb == "internal-precision":
        device: int | str
        device = int(args.device) if str(args.device).isdigit() else str(args.device)
        report = _run_precision_workspace_in_process(
            args.workspace, args.candidate_dir, device=device, seed=args.seed
        )
    elif args.verb.startswith("internal-exec-"):
        report = _internal_execute_request(
            args.execution_request,
            verb=args.verb.removeprefix("internal-exec-"),
        )
    elif args.verb == "stage":
        report = stage_workspace(args.workspace, task_path=args.task, root=args.root)
    elif args.verb == "preflight":
        report = preflight_workspace(args.workspace)
    elif args.verb == "precision":
        report = run_precision_workspace(args.workspace, args.candidate_dir, device=args.device)
    elif args.verb == "performance":
        report = run_performance_workspace(
            args.workspace,
            args.candidate_dir,
            device=args.device,
            lease_manifest=args.lease_manifest,
        )
    else:
        report = evaluate_workspace(
            args.workspace,
            args.candidate_dir,
            precision_device=args.precision_device,
            performance_device=args.performance_device,
            lease_manifest=args.lease_manifest,
        )
    # This is the child/parent wire protocol, not a log line: the parent reads
    # the single JSON document this process writes to stdout
    # (``npubench_core._parse_child_report``).  Routing it through the logging
    # module would add framing and default to stderr, breaking that contract,
    # so the document is written to stdout's binary buffer directly -- which
    # also says at the call site that this is a data channel, not output meant
    # for a human.
    document = json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"
    sys.stdout.buffer.write(document.encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
