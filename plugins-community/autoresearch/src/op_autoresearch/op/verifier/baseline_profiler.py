# Copyright 2026 Huawei Technologies Co., Ltd
#
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

"""
Baseline Profiler: 预先测量 baseline 性能

用于 evolve/adaptive_search 场景，在开始前单独 profile baseline 一次，
避免所有任务重复测量。支持 KernelBench、SOL-ExecBench 和 CANN-Bench 三种 bench_type。
"""

import io
import json
import logging
import os
import shutil
import tarfile
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier.data_cache import (
    CacheEntry,
    CacheIdentity,
    build_baseline_cache_key,
    build_baseline_cache_payload,
    build_sol_problem_cache_identity,
    delete_baseline_result_from_cache,
    extract_baseline_time_us,
    get_baseline_cache_file_path,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineProfileRequest:
    """Unresolved inputs for one baseline measurement."""

    op_name: str
    task_desc: str
    dsl: str
    framework: str
    backend: str
    arch: str
    config: Dict[str, Any]
    warmup_times: Optional[int] = None
    run_times: Optional[int] = None
    timeout: Optional[int] = None

    def resolve(self) -> "BaselineContext":
        return BaselineContext(
            op_name=self.op_name,
            dsl=self.dsl,
            framework=self.framework,
            backend=self.backend,
            arch=self.arch,
            config=self.config,
            warmup_times=resolve_warmup_times(self.warmup_times),
            run_times=resolve_run_times(self.run_times),
            timeout=resolve_eval_timeout(self.timeout),
        )


@dataclass(frozen=True)
class BaselineContext:
    """Resolved execution settings shared by all baseline formats."""

    op_name: str
    dsl: str
    framework: str
    backend: str
    arch: str
    config: Dict[str, Any]
    warmup_times: int
    run_times: int
    timeout: int


@dataclass(frozen=True)
class ProfilePreparation:
    """Worker resources assigned to a baseline profile preparation."""

    worker: Any
    verifier: Any
    profile_dir: str
    device_id: int
    context: BaselineContext
    spec: "CachedProfileSpec"


PrepareProfile = Callable[[ProfilePreparation], Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class CachedProfileSpec:
    """Bench-specific hooks and cache identity for the shared runner."""

    bench_type: str
    bench_label: str
    cache_framework_code: str
    prepare: PrepareProfile
    cache_method: str
    times_label: str


@dataclass(frozen=True)
class BaselineCacheState:
    """Resolved cache location and identity for one run."""

    config: Any
    key_id: str
    key: Optional[str]
    file: Optional[str]

    @property
    def enabled(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.cache_baseline_result
            and self.key
        )


@dataclass(frozen=True)
class ProfileExecution:
    """Artifacts needed to validate and persist a completed profile."""

    context: BaselineContext
    spec: CachedProfileSpec
    cache: BaselineCacheState
    profile_dir: str
    times_us: list


async def profile_baseline_once(
    request: BaselineProfileRequest,
) -> Optional[float]:
    """
    预先 profile baseline 一次（只测量框架实现的性能）

    根据 config["bench_type"] 自动选择 KernelBench 或 SOL-ExecBench 的 baseline profiling 流程。

    设备分配通过 worker 的 device_pool acquire/release 管理，
    与正常 verify/profile 流程保持一致。

    Args:
        request: Operator source, target, and optional profiling overrides.

    Returns:
        float: baseline 时间（微秒），失败返回 None
    """
    context = request.resolve()
    bench_type = context.config.get("bench_type", "kernelbench")

    if bench_type == "sol":
        return await _profile_sol_baseline(context)
    if bench_type == "cann":
        return await _profile_cann_baseline(context)
    return await _profile_kernelbench_baseline(context, request.task_desc)


async def _profile_kernelbench_baseline(
    context: BaselineContext,
    task_desc: str,
) -> Optional[float]:
    spec = CachedProfileSpec(
        bench_type="kernelbench",
        bench_label="KernelBench",
        cache_framework_code=task_desc,
        prepare=_prepare_kernelbench_profile,
        cache_method="profile_single_task",
        times_label="case",
    )
    return await _run_cached_baseline_profile(context, spec)


async def _prepare_kernelbench_profile(
    preparation: ProfilePreparation,
) -> Dict[str, Any]:
    from op_autoresearch.op.verifier.kernel_verifier import (
        SingleTaskProfileRequest,
    )

    context = preparation.context
    result = await preparation.verifier.profile_single_task(
        SingleTaskProfileRequest(
            task_desc=preparation.spec.cache_framework_code,
            warmup_times=context.warmup_times,
            run_times=context.run_times,
            timeout=context.timeout,
            device_id=preparation.device_id,
        )
    )
    if result.get("success"):
        _save_baseline_profile_scripts(preparation)
    return result


async def _try_read_baseline_cache(
    cache_cfg, op_name: str, cache_key: str, cache_file: str, bench_label: str,
) -> Optional[float]:
    """Try to read baseline time from cache. Returns cached time_us or None."""
    cached_entry = read_baseline_result_from_cache(
        cache_cfg, op_name=op_name, cache_key=cache_key,
    )
    cached_time_us = extract_baseline_time_us(cached_entry)
    if cached_time_us is not None:
        logger.info(
            "[%s] ✅ 命中本地 %s baseline cache: %.2fus, "
            "cache_file=%s, cache_key=%s",
            op_name,
            bench_label,
            cached_time_us,
            cache_file,
            cache_key,
        )
        return cached_time_us
    if cached_entry:
        logger.warning(
            "[%s] %s baseline cache 内容无效，删除旧缓存并重新测量: "
            "cache_file=%s, cache_key=%s",
            op_name,
            bench_label,
            cache_file,
            cache_key,
        )
        delete_baseline_result_from_cache(
            cache_cfg, op_name=op_name, cache_key=cache_key,
        )
    return None


def _profile_time_from_line(line: str) -> Optional[float]:
    if "base time:" not in line or "us" not in line or "Geometric mean" in line:
        return None
    try:
        time_str = line.split("base time:")[1].strip().replace("us", "").strip()
        return float(time_str)
    except (ValueError, IndexError) as exc:
        logger.debug("ignored malformed baseline timing %r: %s", line, exc)
        return None


def _parse_profile_log_times(output_log: str, op_name: str) -> list:
    """Parse per-item times from profile output log."""
    times_us = []
    for line in output_log.splitlines():
        stripped = line.strip()
        if not stripped or op_name not in stripped:
            continue
        logger.info('[%s] %s', op_name, stripped)
        time_us = _profile_time_from_line(stripped)
        if time_us is not None:
            times_us.append(time_us)
    return times_us


def _handle_profile_result(
    result: Dict[str, Any],
    execution: ProfileExecution,
) -> Optional[float]:
    """Handle profile result: validate, save, cache. Returns baseline_time_us or None."""
    context = execution.context
    spec = execution.spec
    if not result.get('success', False):
        error_log = result.get('log', 'Unknown error')
        logger.warning(
            '[%s] %s Baseline profile 失败: %s', context.op_name, spec.bench_label, error_log
        )
        return None

    baseline_time_us = result.get('time_us')
    if not baseline_time_us or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        logger.warning(
            '[%s] %s Baseline profile 结果无效: %s', context.op_name, spec.bench_label, baseline_time_us
        )
        return None

    logger.info(
        '[%s] ✅ %s Baseline profile 完成（几何平均）: %.2fus', context.op_name, spec.bench_label, baseline_time_us
    )
    _save_baseline_result_json(execution, baseline_time_us)
    _cache_profile_result(execution, baseline_time_us)
    logger.info(
        '[%s] %s Baseline profile 脚本及结果已保存到: %s', context.op_name, spec.bench_label, execution.profile_dir
    )
    return baseline_time_us


def _cache_profile_result(
    execution: ProfileExecution,
    baseline_time_us: float,
) -> None:
    if not execution.cache.enabled:
        return
    context = execution.context
    spec = execution.spec
    write_baseline_result_to_cache(
        execution.cache.config,
        op_name=context.op_name,
        cache_key=execution.cache.key,
        result_data=build_baseline_cache_payload(
            base_time_us=baseline_time_us,
            warmup_times=context.warmup_times,
            run_times=context.run_times,
            method=spec.cache_method,
            extra={
                f"{spec.times_label}_count": len(execution.times_us),
                f"{spec.times_label}_times_us": execution.times_us,
            },
        ),
        metadata={
            "framework": context.framework,
            "dsl": context.dsl,
            "cache_key_id": execution.cache.key_id,
            "backend": context.backend,
            "arch": context.arch,
            "bench_type": spec.bench_type,
        },
    )


def _build_cache_key(
    context: BaselineContext,
    spec: CachedProfileSpec,
    cache_cfg: Any,
    cache_key_id: str,
):
    """Build baseline cache key and file path. Returns (cache_key, cache_file) or (None, None)."""
    if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
        return None, None
    try:
        identity = CacheIdentity(
            op_name=context.op_name,
            framework=context.framework,
            backend=context.backend,
            arch=context.arch,
            bench_type=spec.bench_type,
            dsl=context.dsl,
            task_id=cache_key_id,
        )
        cache_key = build_baseline_cache_key(
            identity,
            spec.cache_framework_code,
            context.warmup_times,
            context.run_times,
        )
        cache_file = get_baseline_cache_file_path(
            cache_cfg,
            op_name=context.op_name,
            cache_key=cache_key,
        )
        return cache_key, cache_file
    except Exception as exc:
        logger.info(
            '[%s] baseline cache key 构建失败，跳过 cache: %s', context.op_name, exc
        )
        return None, None


def _build_cache_state(
    context: BaselineContext,
    spec: CachedProfileSpec,
) -> BaselineCacheState:
    cache_cfg = load_verifier_data_cache_config(context.config)
    cache_key_id = get_verifier_data_cache_key_id(
        context.config,
        "baseline_profile",
    )
    cache_key, cache_file = _build_cache_key(
        context,
        spec,
        cache_cfg,
        cache_key_id,
    )
    return BaselineCacheState(cache_cfg, cache_key_id, cache_key, cache_file)


async def _read_cached_value(
    context: BaselineContext,
    spec: CachedProfileSpec,
    cache: BaselineCacheState,
) -> Optional[float]:
    if not cache.key:
        return None
    return await _try_read_baseline_cache(
        cache.config,
        context.op_name,
        cache.key,
        cache.file,
        spec.bench_label,
    )


async def _profile_with_cache_lock(
    context: BaselineContext,
    spec: CachedProfileSpec,
    cache: BaselineCacheState,
) -> Optional[float]:
    async with AsyncExitStack() as stack:
        if cache.enabled:
            await stack.enter_async_context(
                verifier_data_cache_lock(
                    cache.config,
                    CacheEntry("baseline", context.op_name, cache.key),
                )
            )
            cached = await _read_cached_value(context, spec, cache)
            if cached is not None:
                return cached
        return await _execute_baseline_profile(stack, context, spec, cache)


async def _run_cached_baseline_profile(
    context: BaselineContext,
    spec: CachedProfileSpec,
) -> Optional[float]:
    """Run one baseline profile with shared cache and device-lease handling."""
    cache = _build_cache_state(context, spec)
    try:
        cached = await _read_cached_value(context, spec, cache)
        if cached is not None:
            return cached

        logger.info(
            '[%s] 🚀 开始预先 %s baseline profile（只测一次）...', context.op_name, spec.bench_label
        )

        return await _profile_with_cache_lock(context, spec, cache)
    except TimeoutError as exc:
        logger.warning(
            "[%s] 等待 %s baseline cache lock 超时，"
            "跳过预先 profile baseline: %s",
            context.op_name,
            spec.bench_label,
            exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            '[%s] %s baseline profile 失败: %s', context.op_name, spec.bench_label, exc
        )
        import traceback
        logger.debug(traceback.format_exc())
        return None


async def _execute_baseline_profile(
    stack: AsyncExitStack,
    context: BaselineContext,
    spec: CachedProfileSpec,
    cache: BaselineCacheState,
) -> Optional[float]:
    from op_autoresearch.core.worker.manager import get_worker_manager

    worker_manager = get_worker_manager()
    worker = await worker_manager.select(
        backend=context.backend,
        arch=context.arch,
    )
    if not worker:
        logger.warning(
            '[%s] 无法获取 worker，跳过预先 %s baseline profile', context.op_name, spec.bench_label
        )
        return None
    stack.push_async_callback(worker_manager.release, worker)
    device_id = await stack.enter_async_context(
        worker.device_lease("baseline_profile")
    )
    preparation = _make_profile_preparation(worker, device_id, context, spec)
    result = await spec.prepare(preparation)
    execution = ProfileExecution(
        context,
        spec,
        cache,
        preparation.profile_dir,
        _parse_profile_log_times(result.get("log", ""), context.op_name),
    )
    return _handle_profile_result(result, execution)


def _make_profile_preparation(
    worker: Any,
    device_id: int,
    context: BaselineContext,
    spec: CachedProfileSpec,
) -> ProfilePreparation:
    from op_autoresearch.op.verifier.kernel_verifier import KernelVerifier

    verifier = KernelVerifier(
        op_name=context.op_name,
        framework_code=spec.cache_framework_code,
        task_id="baseline_profile",
        framework=context.framework,
        dsl=context.dsl,
        backend=context.backend,
        arch=context.arch,
        config=context.config,
        bench_type=spec.bench_type,
        worker=worker,
    )
    profile_dir = os.path.join(
        os.path.expanduser(verifier.log_dir),
        f"{context.op_name}_profile_single_baseline_profile",
    )
    os.makedirs(profile_dir, exist_ok=True)
    return ProfilePreparation(
        worker,
        verifier,
        profile_dir,
        device_id,
        context,
        spec,
    )


def _require_problem_dir(config: Dict[str, Any], key: str, label: str) -> str:
    problem_dir = config.get(key)
    if not problem_dir:
        raise ValueError(f"config[{key!r}] 未配置")
    resolved = os.path.expandvars(os.path.expanduser(str(problem_dir)))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{label} case 目录不存在: {resolved}")
    return resolved


def _stage_required_files(
    source_dir: str,
    profile_dir: str,
    file_names: list[str],
    label: str,
) -> None:
    for file_name in file_names:
        source = os.path.join(source_dir, file_name)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Missing required {label} file: {source}")
        shutil.copy2(source, os.path.join(profile_dir, file_name))


def _device_setup_code(preparation: ProfilePreparation) -> list[str]:
    from op_autoresearch.op.verifier.adapters.factory import (
        get_backend_adapter,
        get_framework_adapter,
    )

    verifier = preparation.verifier
    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(preparation.device_id, verifier.arch)
    return verifier.prepare_code_lines(
        framework_adapter.get_device_setup_code(
            verifier.backend,
            verifier.arch,
            preparation.device_id,
        )
    )


def _render_profile_template(template_path: str, **values: Any) -> str:
    from jinja2 import Template

    with open(template_path, "r", encoding="utf-8") as file:
        return Template(file.read()).render(**values)


async def _run_prepared_profile(
    preparation: ProfilePreparation,
    base_script: str,
) -> Dict[str, Any]:
    wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
    script_path = os.path.join(
        preparation.profile_dir,
        f"profile_single_{preparation.context.op_name}.py",
    )
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(wrapper)
    context = preparation.context
    profile_settings = {
        "warmup_times": context.warmup_times,
        "run_times": context.run_times,
        "timeout": context.timeout,
    }
    return await preparation.worker.profile_single_task(
        _pack_directory(preparation.profile_dir),
        "baseline_profile_profile_single",
        context.op_name,
        profile_settings,
    )


async def _prepare_sol_profile(
    preparation: ProfilePreparation,
) -> Dict[str, Any]:
    """Prepare and execute SOL baseline profile project."""
    from op_autoresearch import get_project_root
    from op_autoresearch.op.verifier.sol_verifier import PROF_SOL_BASE_TEMPLATE_PATH

    verifier = preparation.verifier
    profile_dir = preparation.profile_dir
    device_id = preparation.device_id
    context = preparation.context
    sol_problem_dir = _require_problem_dir(
        verifier.config,
        "sol_problem_dir",
        "SOL",
    )
    _stage_required_files(
        sol_problem_dir,
        profile_dir,
        ["definition.json", "workload.jsonl", "reference.py"],
        "SOL",
    )

    sol_correctness_src = os.path.join(
        get_project_root(), "op", "resources", "utils", "sol_correctness.py",
    )
    shutil.copy2(sol_correctness_src, os.path.join(profile_dir, "sol_correctness.py"))

    device_setup_code = _device_setup_code(preparation)
    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src"),
    )

    base_script = _render_profile_template(
        PROF_SOL_BASE_TEMPLATE_PATH,
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        device_id=device_id,
        warmup_times=context.warmup_times,
        run_times=context.run_times,
        device_setup_code=device_setup_code,
        sol_execbench_src_dir=sol_execbench_src_dir,
    )
    return await _run_prepared_profile(preparation, base_script)


async def _prepare_cann_profile(
    preparation: ProfilePreparation,
) -> Dict[str, Any]:
    """Prepare and execute CANN baseline profile project."""
    import yaml

    from op_autoresearch.op.cann_correctness import CANN_BENCH_SRC_DIR, stage_core_into
    from op_autoresearch.op.cann_correctness.verifier import (
        PROF_CANN_BASE_TEMPLATE_PATH,
    )

    verifier = preparation.verifier
    profile_dir = preparation.profile_dir
    device_id = preparation.device_id
    context = preparation.context
    cann_problem_dir = _require_problem_dir(
        verifier.config,
        "cann_problem_dir",
        "CANN",
    )
    _stage_required_files(
        cann_problem_dir,
        profile_dir,
        ["proto.yaml", "golden.py", "cases.yaml"],
        "CANN",
    )

    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        shutil.copy2(desc_src, os.path.join(profile_dir, "desc.md"))

    stage_core_into(profile_dir)

    device_setup_code = _device_setup_code(preparation)

    proto_path = os.path.join(profile_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    schema = proto.get("operator", {}).get("schema", "")

    base_script = _render_profile_template(
        PROF_CANN_BASE_TEMPLATE_PATH,
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        dsl=verifier.dsl,
        device_id=device_id,
        warmup_times=context.warmup_times,
        run_times=context.run_times,
        device_setup_code=device_setup_code,
        schema=schema,
        cann_bench_src_dir=CANN_BENCH_SRC_DIR,
    )
    return await _run_prepared_profile(preparation, base_script)


async def _profile_sol_baseline(
    context: BaselineContext,
) -> Optional[float]:
    """SOL-ExecBench baseline profiling."""
    sol_problem_dir_for_cache = context.config.get("sol_problem_dir", "")
    if not sol_problem_dir_for_cache:
        logger.warning(
            "[%s] config['sol_problem_dir'] 未配置，跳过预先 SOL baseline profile", context.op_name
        )
        return None
    sol_cache_identity = build_sol_problem_cache_identity(sol_problem_dir_for_cache)
    return await _run_cached_baseline_profile(
        context,
        CachedProfileSpec(
            bench_type="sol",
            bench_label="SOL",
            cache_framework_code=sol_cache_identity,
            prepare=_prepare_sol_profile,
            cache_method="sol_profile_single_task",
            times_label="workload",
        ),
    )


async def _profile_cann_baseline(
    context: BaselineContext,
) -> Optional[float]:
    """CANN-Bench baseline profiling."""
    return await _run_cached_baseline_profile(
        context,
        CachedProfileSpec(
            bench_type="cann",
            bench_label="CANN",
            cache_framework_code=context.config.get("cann_problem_dir", ""),
            prepare=_prepare_cann_profile,
            cache_method="cann_profile_single_task",
            times_label="case",
        ),
    )


def _save_baseline_result_json(
    execution: ProfileExecution,
    baseline_time_us: float,
) -> None:
    """将 baseline profile 结果写为 base_profile_result.json，保存到 profile_dir"""
    context = execution.context
    spec = execution.spec
    try:
        method_prefix = f"{spec.bench_type}_base"
        method = (
            f"{method_prefix}_profiler_npu"
            if context.backend == "ascend"
            else f"{method_prefix}_loop_timer"
        )
        if spec.bench_type == "sol":
            method = (
                "sol_base_profiler_npu"
                if context.backend == "ascend"
                else "sol_base_do_bench"
            )
        result_data = {
            "execution_time_ms": baseline_time_us / 1000.0,
            "execution_time_us": baseline_time_us,
            "avg_time_us": baseline_time_us,
            "warmup_times": context.warmup_times,
            "run_times": context.run_times,
            f"{spec.times_label}_count": len(execution.times_us),
            f"{spec.times_label}_times_us": execution.times_us,
            "method": method,
            "bench_type": spec.bench_type,
        }
        result_file = os.path.join(execution.profile_dir, "base_profile_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        logger.info(
            '[%s] base_profile_result.json 已写入: %s', context.op_name, result_file
        )
    except Exception as exc:
        logger.warning(
            '[%s] 写入 base_profile_result.json 失败: %s', context.op_name, exc
        )


def _pack_directory(dir_path: str) -> bytes:
    """将目录打包为 tar 字节流"""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        for root, _dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                tar_file.add(file_path, arcname=arcname)
    return tar_buffer.getvalue()


def _save_baseline_profile_scripts(preparation: ProfilePreparation) -> None:
    """
    保存 KernelBench baseline profile 脚本到 log 目录
    """
    verifier = preparation.verifier
    context = preparation.context
    try:
        baseline_dir = os.path.join(
            os.path.expanduser(verifier.log_dir),
            f"{context.op_name}_baseline_profile"
        )
        os.makedirs(baseline_dir, exist_ok=True)

        verifier.materialize_framework_bundle(
            baseline_dir,
            preparation.spec.cache_framework_code,
        )

        script_file = os.path.join(
            baseline_dir,
            f"profile_baseline_{context.op_name}.py",
        )
        verifier.gen_profile_single_task_file(
            script_file,
            device_id=preparation.device_id,
            warmup_times=context.warmup_times,
            run_times=context.run_times,
        )

        logger.info(
            '[%s] Baseline profile 脚本已保存到: %s', context.op_name, baseline_dir
        )

    except Exception as exc:
        logger.warning(
            '[%s] 保存 baseline profile 脚本失败: %s', context.op_name, exc
        )


def set_baseline_in_config(config: Dict[str, Any], baseline_time_us: float) -> None:
    """
    将缓存的 baseline 时间设置到 config 中

    Args:
        config: 配置字典
        baseline_time_us: baseline 时间（微秒）
    """
    # 只有当 baseline_time_us 是有效值时才设置
    if baseline_time_us is None or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        return

    if 'profile_settings' not in config:
        config['profile_settings'] = {}

    from op_autoresearch.op.verifier.profiler_utils import make_profile_section
    config['profile_settings']['override_base_section'] = make_profile_section(
        baseline_time_us, method="override")
    config['profile_settings']['skip_base_profile'] = True
