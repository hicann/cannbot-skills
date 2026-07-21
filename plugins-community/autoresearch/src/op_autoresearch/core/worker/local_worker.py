# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import math
import os
import sys
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from op_autoresearch.op.utils import triton_ascend_api_docs
from op_autoresearch.op.verifier import aggregate, profiler_utils, roofline_utils
from op_autoresearch.utils import process_utils

from ..async_pool.device_pool import DevicePool
from .eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
from .interface import WorkerInterface
from .interface import empty_profile_result as _empty_profile_result

logger = logging.getLogger(__name__)

# Stable module-level seams used by the worker tests and downstream embedders.
compute_roofline_profile = roofline_utils.compute_roofline_profile
write_roofline_profile_result = roofline_utils.write_roofline_profile_result
run_profile_scripts_and_collect_results = (
    profiler_utils.run_profile_scripts_and_collect_results
)


# 信号编号到名称的映射
_SIGNAL_NAMES = {
    1: "SIGHUP",   # Hangup
    2: "SIGINT",   # Interrupt
    3: "SIGQUIT",  # Quit
    6: "SIGABRT",  # Abort
    9: "SIGKILL",  # Kill
    11: "SIGSEGV", # Segmentation fault
    13: "SIGPIPE", # Broken pipe
    15: "SIGTERM", # Termination
}


def _get_signal_name(signum: int) -> str:
    """将信号编号转换为可读名称"""
    return _SIGNAL_NAMES.get(signum, f"SIG({signum})")


def _read_text_artifact(file_path: str, rel_path: str) -> Optional[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as artifact_file:
            return artifact_file.read()
    except Exception as exc:
        logger.warning('Failed to read file %s: %s', rel_path, exc)
        return None


def collect_json_artifacts(directory: str, trace_names: tuple = ()) -> Dict[str, str]:
    """{rel_path: content} for json/jsonl, plus any basename in ``trace_names``
    (the --trace timeline + per-op CSVs); skips other ``*_ascend_pt`` internals.
    """
    artifacts = {}
    if not os.path.exists(directory):
        return artifacts

    for root, _dirs, files in os.walk(directory):
        in_trace = "_ascend_pt" in root
        for filename in files:
            if not (filename in trace_names or
                    (not in_trace and filename.endswith((".json", ".jsonl")))):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, directory)
            content = _read_text_artifact(file_path, rel_path)
            if content is not None:
                artifacts[rel_path] = content
    return artifacts


class PackageExtractError(Exception):
    """A worker package could not be materialized into a directory."""


def _validate_archive_member(extract_root: str, member) -> None:
    target = os.path.realpath(os.path.join(extract_root, member.name))
    try:
        contained = os.path.commonpath((extract_root, target)) == extract_root
    except ValueError:
        contained = False
    special_file = any((
        member.issym(), member.islnk(), member.isdev(), member.isfifo(),
    ))
    if not contained or special_file:
        raise PackageExtractError(
            f"Unsafe archive member rejected: {member.name!r}")


def _extract_archive(tar_path: str, extract_dir: str) -> None:
    try:
        with tarfile.open(tar_path, "r") as archive:
            extract_root = os.path.realpath(extract_dir)
            for member in archive.getmembers():
                _validate_archive_member(extract_root, member)
            if hasattr(tarfile, "data_filter"):
                archive.extractall(extract_dir, filter="data")
            else:
                archive.extractall(extract_dir)
    except (OSError, tarfile.TarError) as exc:
        raise PackageExtractError(
            f"Failed to extract package: {exc}") from exc


@contextmanager
def _extract_package(package_data: Union[bytes, str]):
    """Yield a directory holding the package contents. ``bytes`` → extracted
    into a TemporaryDirectory (auto-cleaned on exit); ``str`` → an
    already-extracted dir path, used as-is. Raises PackageExtractError on a
    bad type or corrupt archive so callers map it to their own error shape.
    Single owner of the tempdir + untar boilerplate the entry points share.
    """
    if isinstance(package_data, str):
        yield package_data
        return
    if not isinstance(package_data, (bytes, bytearray)):
        raise PackageExtractError("Unsupported package_data type for LocalWorker")
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, "package.tar")
        with open(tar_path, "wb") as f:
            f.write(package_data)
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        _extract_archive(tar_path, extract_dir)
        yield extract_dir


@dataclass(frozen=True)
class ScriptRunRequest:
    extract_dir: str
    script_name: str
    timeout: Optional[int]
    task_id: str
    action: str
    keep_res: bool = False


@dataclass(frozen=True)
class TraceProfileRequest:
    extract_dir: str
    op_name: str
    task_id: str
    run: Any
    analyze: Any
    method: str


@dataclass(frozen=True)
class MsprofRequest:
    extract_dir: str
    op_name: str
    task_id: str
    warmup_times: int
    run_times: int
    timeout: Optional[int] = None


@dataclass(frozen=True)
class ProfileOptions:
    backend: str
    dsl: str
    run_times: int
    warmup_times: int
    timeout: int
    keep_res: bool
    override_base_section: Optional[dict]


@dataclass(frozen=True)
class ProfileContext:
    extract_dir: str
    op_name: str
    task_id: str
    settings: Dict[str, Any]
    options: ProfileOptions
    base_requested: bool
    generation_requested: bool


@dataclass(frozen=True)
class ProfileSections:
    base: Optional[dict]
    generation: Optional[dict]
    error: Optional[str] = None
    failure: Optional[dict] = None


@dataclass(frozen=True)
class ProfileMeasurements:
    generation_time: float
    base_time: float
    generation_per_shape: list[float]
    base_per_shape: list[float]
    speedup: float


def _valid_base_override(section) -> bool:
    return (
        isinstance(section, dict)
        and isinstance(section.get("avg_us"), (int, float))
        and 0 < section["avg_us"] < float("inf")
    )


def _make_profile_context(extract_dir: str, op_name: str, task_id: str,
                          settings: Dict[str, Any], worker_backend: str,
                          ) -> ProfileContext:
    options = ProfileOptions(
        backend=settings.get("backend", worker_backend),
        dsl=settings.get("dsl", ""),
        run_times=resolve_run_times(settings.get("run_times")),
        warmup_times=resolve_warmup_times(settings.get("warmup_times")),
        timeout=resolve_eval_timeout(settings.get("timeout")),
        keep_res=bool(settings.get("keep_res")),
        override_base_section=settings.get("override_base_section"),
    )
    base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
    generation_script = os.path.join(
        extract_dir, f"profile_{op_name}_generation.py")
    return ProfileContext(
        extract_dir=extract_dir,
        op_name=op_name,
        task_id=task_id,
        settings=settings,
        options=options,
        base_requested=(os.path.exists(base_script)
                        or _valid_base_override(options.override_base_section)),
        generation_requested=os.path.exists(generation_script),
    )


def _validate_profile_sections(context: ProfileContext,
                               sections: dict) -> ProfileSections:
    base_section = sections.get("base")
    generation_section = sections.get("gen")
    if context.generation_requested and generation_section is None:
        logger.error("[%s] Generation profile produced no result",
                     context.task_id)
        return ProfileSections(
            base_section, generation_section,
            failure=_empty_profile_result(error="generation profile failed"),
        )
    profile_error = None
    if context.base_requested and base_section is None:
        logger.error("[%s] Base profile produced no result", context.task_id)
        if generation_section is None:
            return ProfileSections(
                base_section, generation_section,
                failure=_empty_profile_result(error="base profile failed"),
            )
        profile_error = "base profile failed"
    if not context.base_requested and not context.generation_requested:
        logger.error("[%s] No profile scripts or base override found",
                     context.task_id)
        return ProfileSections(
            base_section, generation_section,
            failure=_empty_profile_result(
                error="no profile section requested"),
        )
    return ProfileSections(base_section, generation_section, profile_error)


def _profile_measurements(sections: ProfileSections) -> ProfileMeasurements:
    generation_time = (
        sections.generation["avg_us"]
        if sections.generation else float("inf"))
    base_time = sections.base["avg_us"] if sections.base else float("inf")
    generation_per_shape = (
        list(sections.generation["per_case_us"])
        if sections.generation else [])
    base_per_shape = (
        list(sections.base["per_case_us"]) if sections.base else [])
    speedup = 0.0
    if sections.base and sections.generation:
        speedup = aggregate.geomean_ratio(
            base_per_shape, generation_per_shape) or 0.0
    return ProfileMeasurements(
        generation_time=generation_time,
        base_time=base_time,
        generation_per_shape=generation_per_shape,
        base_per_shape=base_per_shape,
        speedup=speedup,
    )


def _profile_roofline(context: ProfileContext, sections: ProfileSections,
                      measurements: ProfileMeasurements) -> Optional[dict]:
    if sections.generation is None:
        return None
    roofline = compute_roofline_profile(
        verify_dir=context.extract_dir,
        op_name=context.op_name,
        task_id=context.task_id,
        profile_settings=context.settings,
    )
    roofline = roofline_utils.augment_roofline_metrics(
        roofline,
        gen_time_us=measurements.generation_time,
        base_time_us=(measurements.base_time if sections.base else None),
        gen_per_shape_us=measurements.generation_per_shape,
        base_per_shape_us=(measurements.base_per_shape
                           if sections.base else None),
    )
    write_roofline_profile_result(context.extract_dir, roofline)
    return roofline


def _profile_time(section: Optional[Dict[str, Any]], value: float) -> Optional[float]:
    if not section or value >= float("inf"):
        return None
    return value


def _profile_payload(context: ProfileContext,
                     sections: ProfileSections) -> Dict[str, Any]:
    if sections.failure is not None:
        return sections.failure
    measurements = _profile_measurements(sections)
    roofline = _profile_roofline(context, sections, measurements)
    trace_names = (
        ("trace_view.json", "op_statistic.csv", "kernel_details.csv")
        if context.options.keep_res else ())
    artifacts = collect_json_artifacts(context.extract_dir, trace_names)
    if artifacts:
        logger.info("[%s] Collected %s artifact files: %s", context.task_id,
                    len(artifacts), list(artifacts.keys()))
    result = {
        "gen_time": _profile_time(sections.generation, measurements.generation_time),
        "base_time": _profile_time(sections.base, measurements.base_time),
        "speedup": measurements.speedup,
        "per_shape_gen_us": measurements.generation_per_shape,
        "per_shape_base_us": measurements.base_per_shape,
        "gen_method": (sections.generation.get("method")
                       if sections.generation else None),
        "base_method": sections.base.get("method") if sections.base else None,
        "roofline_time": (roofline.get("time_us")
                          if roofline and roofline.get("success") else None),
        "roofline_speedup": (
            roofline.get("speedup_vs_generated", 0.0) if roofline else 0.0),
        "roofline": roofline,
        "artifacts": artifacts,
    }
    if sections.error is not None:
        result["error"] = sections.error
    return result


def _single_profile_result(time_us=None, log: str = "") -> Dict[str, Any]:
    try:
        value = float(time_us)
    except (TypeError, ValueError):
        value = None
    if value is not None and (not math.isfinite(value) or value <= 0):
        value = None
    return {"time_us": value, "success": value is not None, "log": log}


def _time_from_profile_log(output_log: str):
    for line in output_log.splitlines():
        if "PROFILE_RESULT:" not in line:
            continue
        try:
            return float(line.split("PROFILE_RESULT:", 1)[1].strip())
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _read_single_profile_time(extract_dir: str, output_log: str):
    result_file = os.path.join(extract_dir, "profile_single_result.json")
    if not os.path.exists(result_file):
        return _time_from_profile_log(output_log)
    try:
        with open(result_file, "r", encoding="utf-8") as stream:
            return json.load(stream).get("avg_time_us")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", result_file, exc)
        return None


class LocalWorker(WorkerInterface):
    """Execute prepared projects in isolated local subprocesses."""

    def __init__(self, device_pool: DevicePool, backend: str = "ascend"):
        self.device_pool = device_pool
        self.backend = backend

    async def acquire_device(
        self, task_id: str = "unknown", timeout: Optional[float] = None
    ) -> Tuple[int, int]:
        return await self.device_pool.acquire_device(owner=task_id, timeout=timeout)

    async def release_device(
        self, device_id: int, lease_id: int, task_id: str = "unknown"
    ) -> None:
        await self.device_pool.release_device(device_id, lease_id)

    async def verify(
        self,
        package_data: Union[bytes, str],
        task_id: str,
        op_name: str,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Execute one generated verification script."""
        budget = resolve_eval_timeout(timeout)
        try:
            with _extract_package(package_data) as extract_dir:
                returncode, output_log = await self._execute_operator_script(
                    ScriptRunRequest(
                        extract_dir=extract_dir,
                        script_name=f"verify_{op_name}.py",
                        timeout=budget,
                        task_id=task_id,
                        action="Verification",
                    )
                )
                if returncode is None:
                    return False, output_log, {}

                artifacts = collect_json_artifacts(extract_dir)
                if artifacts:
                    logger.info(
                        "[%s] Collected %s artifact files: %s",
                        task_id,
                        len(artifacts),
                        list(artifacts),
                    )
                success = returncode == 0
                if success:
                    logger.info("[%s] Verification passed.", task_id)
                else:
                    logger.error(
                        "[%s] Verification failed with log:\n%s",
                        task_id,
                        output_log,
                    )
                return success, output_log, artifacts
        except Exception as exc:
            logger.error(
                "[%s] LocalWorker verification failed: %s",
                task_id,
                exc,
                exc_info=True,
            )
            return False, str(exc), {}

    async def get_doc(self, doc_name: str) -> str:
        providers = {
            "triton_ascend_api": triton_ascend_api_docs.load_triton_ascend_api_docs,
        }
        try:
            provider = providers[doc_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported doc name: {doc_name}") from exc
        return provider()

    async def profile(
        self, package_data: bytes, task_id: str, op_name: str,
        profile_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the base and candidate profiling scripts."""
        try:
            with _extract_package(package_data) as extract_dir:
                return await self._profile_in_directory(
                    extract_dir, task_id, op_name, profile_settings
                )
        except PackageExtractError as exc:
            return _empty_profile_result(error=str(exc))
        except Exception as exc:
            logger.error("[%s] LocalWorker profiling failed: %s",
                         task_id, exc, exc_info=True)
            return _empty_profile_result(error=str(exc))

    async def run_profile_script_async(self, request: ScriptRunRequest) -> bool:
        """Run one profile script (killable; self-times + writes its result
        JSON, which the caller reads). True on rc==0.
        """
        rc, stdout, stderr, timed_out = await self._run_script(request)
        if timed_out:
            logger.error("[%s] %s timed out after %s seconds",
                         request.task_id, request.action, request.timeout)
            return False
        if rc != 0:
            log = (stdout.decode(errors="replace") + "\n"
                   + stderr.decode(errors="replace"))
            logger.error("[%s] %s failed (rc=%s): %s", request.task_id,
                         request.action, rc, log.strip())
            return False
        return True

    def run_trace_profiling(
            self, request: TraceProfileRequest,
            ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Shared base+generation loop for Ascend msprof: run each script,
        analyze the trace,
        build a canonical single-element section (the profiler attributes the
        kernel time to the whole run). ``run(script) -> (ok, err, path)``;
        ``analyze(path, kind) -> (ok, err, avg_us)``; ``method`` tags the
        section + log lines. Single owner of this loop for both backends.
        """
        sections: Dict[str, Optional[Dict[str, Any]]] = {"base": None, "gen": None}
        try:
            for kind, json_key in (("base", "base"), ("generation", "gen")):
                script = os.path.join(
                    request.extract_dir,
                    f"profile_{request.op_name}_{kind}.py",
                )
                if not os.path.exists(script):
                    logger.info("[%s] %s profile script not found; skipping",
                                request.task_id, kind)
                    continue
                ok, err, path = request.run(script)
                if not ok or not path:
                    logger.error("[%s] %s %s failed: %s", request.task_id,
                                 kind, request.method, err)
                    continue
                ok, err, avg_us = request.analyze(path, kind)
                if not ok:
                    logger.error("[%s] %s %s analysis failed: %s",
                                 request.task_id, kind, request.method, err)
                    continue
                sections[json_key] = profiler_utils.make_profile_section(
                    avg_us, method=request.method)
        except Exception as exc:
            logger.error("[%s] %s profiling failed: %s", request.task_id,
                         request.method, exc, exc_info=True)
        return sections

    async def run_external_profiler(self, func, request=None):
        """Run a blocking profiler without leaking it on coroutine cancel.

        Executor futures do not stop their underlying thread.  Propagate a
        cooperative event into every profiler subprocess, then wait until the
        thread has killed/drained that process tree before re-raising cancel.
        This preserves the device-lease teardown ordering used by async script
        profiling and verification.
        """
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        if request is None:
            future = loop.run_in_executor(None, func, cancel_event)
        else:
            future = loop.run_in_executor(None, func, request, cancel_event)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                await asyncio.shield(future)
            except Exception as e:
                logger.warning("External profiler cleanup failed: %s", e)
            raise

    async def profile_single_task(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        profile_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one profile script without a baseline."""
        try:
            with _extract_package(package_data) as extract_dir:
                return await self._profile_single_in_directory(
                    extract_dir, task_id, op_name, profile_settings)
        except PackageExtractError as exc:
            return _single_profile_result(log=str(exc))
        except Exception as exc:
            logger.error("[%s] LocalWorker profile_single_task failed: %s",
                         task_id, exc, exc_info=True)
            return _single_profile_result(log=str(exc))

    async def generate_reference(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, bytes]:
        """Run the reference producer and read its serialized tensor file."""
        budget = resolve_reference_timeout(timeout)
        try:
            with _extract_package(package_data) as extract_dir:
                returncode, output_log = await self._execute_operator_script(
                    ScriptRunRequest(
                        extract_dir=extract_dir,
                        script_name=f"verify_{op_name}.py",
                        timeout=budget,
                        task_id=task_id,
                        action="Reference generation",
                    )
                )
                if returncode is None:
                    return False, output_log, b""
                if returncode != 0:
                    logger.error(
                        "[%s] Reference generation failed with log:\n%s",
                        task_id,
                        output_log,
                    )
                    return False, output_log, b""
                if "REFERENCE_GENERATION_SUCCESS" not in output_log:
                    message = (
                        "Reference generation did not complete successfully:\n"
                        f"{output_log}"
                    )
                    return False, message, b""

                ref_file = os.path.join(extract_dir, f"{op_name}_reference.pt")
                if not os.path.isfile(ref_file):
                    return (
                        False,
                        f"Reference file {ref_file} not found after generation.",
                        b"",
                    )
                with open(ref_file, "rb") as stream:
                    ref_bytes = stream.read()
                logger.info(
                    "[%s] Reference generation succeeded, .pt file size: %s bytes",
                    task_id,
                    len(ref_bytes),
                )
                return True, output_log, ref_bytes
        except Exception as exc:
            logger.error(
                "[%s] LocalWorker generate_reference failed: %s",
                task_id,
                exc,
                exc_info=True,
            )
            return False, str(exc), b""

    async def _execute_operator_script(
        self, request: ScriptRunRequest
    ) -> tuple[Optional[int], str]:
        script_path = os.path.join(request.extract_dir, request.script_name)
        if not os.path.isfile(script_path):
            return None, f"Verification script {request.script_name} not found."

        logger.info(
            "[%s] Running %s", request.task_id, request.action.lower()
        )
        returncode, stdout, stderr, timed_out = await self._run_script(request)
        if timed_out:
            return (
                None,
                f"{request.action} timed out after {request.timeout} seconds.",
            )

        output = "\n".join(
            stream.decode(errors="replace") for stream in (stdout, stderr)
        )
        if returncode < 0 and not output.strip():
            signal_number = -returncode
            output = (
                f"Process terminated by signal {signal_number} "
                f"({_get_signal_name(signal_number)}).\n"
                "No output captured (process died before writing to "
                "stdout/stderr).\n"
            )
        return returncode, output

    async def _collect_profile_sections(
            self, context: ProfileContext,
            ) -> Optional[Dict[str, Any]]:
        from op_autoresearch.op.verifier.adapters.factory import get_dsl_adapter
        options = context.options
        adapter = get_dsl_adapter(options.dsl)
        if adapter.profile_via_python_script or options.backend == "cpu":
            return await run_profile_scripts_and_collect_results(
                context.extract_dir,
                context.op_name,
                lambda name, label: self.run_profile_script_async(
                    ScriptRunRequest(
                        extract_dir=context.extract_dir,
                        script_name=name,
                        timeout=options.timeout,
                        task_id=context.task_id,
                        action=label,
                        keep_res=options.keep_res,
                    )),
                task_id=context.task_id,
                override_base_section=options.override_base_section,
            )
        if options.backend == "ascend":
            return await self.run_external_profiler(
                self._run_msprof_profiling,
                MsprofRequest(
                    extract_dir=context.extract_dir,
                    op_name=context.op_name,
                    task_id=context.task_id,
                    warmup_times=options.warmup_times,
                    run_times=options.run_times,
                    timeout=options.timeout,
                ),
            )
        logger.warning("[%s] Unsupported backend for profiling: %s",
                       context.task_id, options.backend)
        return None

    async def _profile_in_directory(
            self, extract_dir: str, task_id: str, op_name: str,
            profile_settings: Dict[str, Any],
            ) -> Dict[str, Any]:
        context = _make_profile_context(
            extract_dir, op_name, task_id, profile_settings, self.backend)
        try:
            raw_sections = await self._collect_profile_sections(context)
            if raw_sections is None:
                return _empty_profile_result()
            sections = _validate_profile_sections(context, raw_sections)
            return _profile_payload(context, sections)
        except Exception as exc:
            logger.error("[%s] Profiling execution failed: %s", task_id, exc,
                         exc_info=True)
            return _empty_profile_result(error=str(exc))

    async def _profile_single_in_directory(
            self, extract_dir: str, task_id: str, op_name: str,
            profile_settings: Dict[str, Any],
            ) -> Dict[str, Any]:
        script_name = f"profile_single_{op_name}.py"
        if not os.path.exists(os.path.join(extract_dir, script_name)):
            return _single_profile_result(
                log=f"Profile script {script_name} not found")
        logger.info("[%s] Running single task profiling for %s",
                    task_id, op_name)
        timeout = resolve_eval_timeout(profile_settings.get("timeout"))
        returncode, stdout, stderr, timed_out = await self._run_script(
            ScriptRunRequest(
                extract_dir=extract_dir,
                script_name=script_name,
                timeout=timeout,
                task_id=task_id,
                action="Profile single task",
            ))
        if timed_out:
            return _single_profile_result(
                log=f"Timed out after {timeout} seconds")
        output_log = (stdout.decode(errors="replace") + "\n"
                      + stderr.decode(errors="replace"))
        if returncode != 0:
            logger.error("[%s] Profile single task failed with log:\n%s",
                         task_id, output_log)
            return _single_profile_result(log=output_log)
        time_us = _read_single_profile_time(extract_dir, output_log)
        result = _single_profile_result(time_us, output_log)
        logger.info("[%s] Profile single task result: %s us",
                    task_id, result["time_us"])
        return result

    async def _run_script(
            self, request: ScriptRunRequest,
            ) -> Tuple[Optional[int], bytes, bytes, bool]:
        """Spawn ``python script_name`` in extract_dir as a killable async
        subprocess (own process group, torn down on timeout/cancel). Returns
        ``(returncode, stdout, stderr, timed_out)``; callers own the decode +
        result shaping. Single owner of the worker's subprocess spawn+kill.
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if os.name == "posix":
            env["PWD"] = os.path.abspath(request.extract_dir)
        # --trace: per-subprocess only — never mutate the daemon's global env.
        if request.keep_res:
            env["OP_AUTORESEARCH_PROF_KEEP_RES"] = "1"
        process = await asyncio.create_subprocess_exec(
            sys.executable, request.script_name, cwd=request.extract_dir, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **process_utils.popen_process_group_kwargs(),
        )
        stdout, stderr, timed_out = await process_utils.communicate_or_kill(
            process, request.timeout, request.task_id, request.action)
        return process.returncode, stdout, stderr, timed_out

    def _run_msprof_profiling(self, request: MsprofRequest,
                              cancel_event=None):
        timeout = resolve_eval_timeout(request.timeout)
        return self.run_trace_profiling(TraceProfileRequest(
            extract_dir=request.extract_dir,
            op_name=request.op_name,
            task_id=request.task_id,
            run=lambda s: profiler_utils.run_msprof(
                s, request.op_name, request.task_id,
                timeout=timeout, cancel_event=cancel_event),
            analyze=lambda path, _k: profiler_utils.analyze_prof_data(
                path, request.warmup_times, request.run_times,
                request.op_name, request.task_id),
            method="msprof",
        ))
