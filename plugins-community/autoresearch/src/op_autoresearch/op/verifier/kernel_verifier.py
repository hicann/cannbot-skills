# Copyright 2025 Huawei Technologies Co., Ltd
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

import ast
import io
import json
import logging
import os
import shutil
import tarfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from jinja2 import Template

from op_autoresearch import get_project_root
from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.core.worker.interface import WorkerInterface, empty_profile_result
from op_autoresearch.op.utils.config_utils import normalize_dsl
from op_autoresearch.op.utils.task_layout import REF_FILE_DEFAULT
from op_autoresearch.op.verifier.adapters.dsl.base import (
    BenchmarkSpec,
    MaterializeSpec,
)
from op_autoresearch.op.verifier.adapters.factory import (
    get_backend_adapter,
    get_dsl_adapter,
    get_framework_adapter,
)
from op_autoresearch.op.verifier.config_verifier import (
    ConfigVerificationRequest,
    has_autotune_configs,
    verify_autotune_configs,
)
from op_autoresearch.op.verifier.data_cache import (
    CacheEntry,
    CacheIdentity,
    build_baseline_cache_key,
    build_baseline_cache_payload,
    build_reference_cache_key,
    build_sol_problem_cache_identity,
    delete_baseline_result_from_cache,
    delete_reference_data_from_cache,
    extract_baseline_time_us,
    get_baseline_cache_file_path,
    get_reference_cache_file_path,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    read_reference_data_from_cache,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
    write_reference_data_to_cache,
)
from op_autoresearch.op.verifier.profile_project import ProfileProjectSpec
from op_autoresearch.op.verifier.profiler_utils import make_profile_section

# 模板路径
TEMPLATE_PATH = os.path.join(get_project_root(), "op", "resources", "templates",
                             "kernel_verify_template_refactored.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "op", "resources",
                                          "templates", "prof_base_template_refactored.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_generation_template_refactored.j2")
PROFILE_SINGLE_TASK_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_single_task_template.j2")
RUNTIME_CHECK_TEMPLATE_PATHS = {
    "torch": os.path.join(
        get_project_root(), "op", "resources", "templates", "check_reference_torch.py.j2"
    ),
    "mindspore": os.path.join(
        get_project_root(), "op", "resources", "templates", "check_reference_mindspore.py.j2"
    ),
}
REFERENCE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "generate_reference.py.j2"
)
# 生成CMakeLists.txt和运行脚本的路径

# 类型定义
FrameworkType = Literal["torch", "mindspore"]
ImplType = Literal["triton_ascend", "triton-russia", "ascendc", "ascendc_catlass"]
BackendType = Literal["ascend"]
ArchType = str

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _VerifyReferenceMode:
    use_data: bool = False
    use_inputs: bool = False
    reference_file: Optional[str] = None


@dataclass(frozen=True)
class _SpeedupRecord:
    speedup: float
    base_time: float
    gen_time: float
    unique_dir: str
    roofline_time: Optional[float] = None
    roofline_speedup: Optional[float] = None


@dataclass(frozen=True)
class _VerificationRecord:
    verify_dir: str
    current_step: int
    passed: bool
    logs: str
    total_configs: int = 0
    valid_configs: int = 0


@dataclass
class _ProfileSession:
    unique_dir: str
    settings: Dict[str, Any]
    warmup_times: int
    run_times: int
    base_only: bool
    actual_device_id: int = 0
    acquired_device: Optional[int] = None
    acquired_lease: Any = None
    verify_dir: str = ""
    skip_base: bool = False


@dataclass(frozen=True)
class SingleTaskProfileRequest:
    """Code and execution limits for a framework-only profile run."""

    task_desc: str
    warmup_times: Optional[int] = None
    run_times: Optional[int] = None
    timeout: Optional[int] = None
    device_id: int = 0


def _render_template_file(template_path: str, **context: Any) -> str:
    with open(template_path, "r", encoding="utf-8") as file:
        template = Template(file.read(), keep_trailing_newline=True)
    return template.render(**context)


def _resource_template_path(template_name: str) -> str:
    return os.path.join(
        get_project_root(), "op", "resources", "templates", template_name
    )


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def _write_content(path: str, content: Any) -> None:
    if isinstance(content, bytes):
        with open(path, "wb") as file:
            file.write(content)
        return
    _write_text(path, content)


def _format_autotune_case(case_idx: str, data: Dict[str, Any]) -> List[str]:
    lines = [f"Case {case_idx}:"]
    for kernel_name, configs in data.items():
        lines.append(f"All config timings for {kernel_name}:")
        for config_info in sorted(configs, key=lambda item: item['rank']):
            status = " (BEST)" if config_info['is_best'] else ""
            lines.append(
                f"  Config {config_info['rank']}: {config_info['config']} -> "
                f"{config_info['timing_us']:.4f}us{status}"
            )
    lines.append("")
    return lines


def _runtime_check_device_id(worker) -> int:
    if worker is None:
        return 0
    from op_autoresearch.core.worker.local_worker import LocalWorker

    if isinstance(worker, LocalWorker) and worker.device_pool:
        return worker.device_pool.device_list[0]
    return 0


def _runtime_check_script(framework: str, device_id: int) -> str:
    template_path = RUNTIME_CHECK_TEMPLATE_PATHS[framework]
    return _render_template_file(template_path, device_id=device_id)


def _benchmark_sync_code(framework: str, backend: str) -> str:
    if framework == "torch" and backend == "ascend":
        return "torch.npu.synchronize()"
    if framework == "mindspore" and backend == "ascend":
        return "ms.runtime.synchronize()"
    return ""


def _missing_task_desc_symbols(tree: ast.Module) -> List[str]:
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = (
        ("Model", class_names, "class Model"),
        ("get_inputs", function_names, "function get_inputs"),
        ("get_init_inputs", function_names, "function get_init_inputs"),
    )
    return [label for name, names, label in required if name not in names]


def sync_artifacts_to_directory(artifacts: Dict[str, str], target_dir: str, task_id: str = "0") -> None:
    """
    将 artifacts 同步到目标目录。

    Args:
        artifacts: 从 Worker 返回的 artifacts 字典，格式为 {relative_path: file_content}
                   例如: {"autotune_info_case_0.json": "{...}", "subdir/result.jsonl": "..."}
        target_dir: 目标目录路径（通常是 verify_dir）
        task_id: 任务ID（用于日志）
    """
    if not artifacts:
        return

    logger.info('[%s] Syncing %s artifact files to %s', task_id, len(artifacts), target_dir)

    target_root = os.path.realpath(target_dir)
    for rel_path, content in artifacts.items():
        if not isinstance(rel_path, str) or not rel_path:
            logger.warning('[%s] Ignoring invalid artifact path: %r', task_id, rel_path)
            continue
        full_path = os.path.realpath(os.path.join(target_root, rel_path))
        try:
            contained = os.path.commonpath(
                (target_root, full_path)) == target_root
        except ValueError:
            contained = False
        if not contained:
            logger.warning(
                '[%s] Ignoring artifact path outside verify dir: %r', task_id, rel_path)
            continue

        # 确保目录存在
        dir_path = os.path.dirname(full_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.debug("[%s] Created directory: %s", task_id, dir_path)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug("[%s] Synced artifact: %s", task_id, rel_path)
        except Exception as e:
            logger.warning('[%s] Failed to sync artifact %s: %s', task_id, rel_path, e)


class KernelVerifier:

    # Accepted multi-shape factory names (any one triggers dynamic mode).
    # ``get_inputs_dyn_list`` — legacy (209 internal benchmark refs);
    # ``get_input_groups`` — NPUKernelBench + WA new convention. The
    # framework adapter aliases whichever the ref defines back to
    # ``get_inputs_dyn_list`` (the template's internal local name).
    _DYN_FACTORY_NAMES = ("get_inputs_dyn_list", "get_input_groups")

    def __init__(
        self,
        op_name: str,
        framework_code: str,
        task_id: str = "0",
        framework: FrameworkType = "torch",
        dsl: ImplType = "triton_ascend",
        backend: BackendType = "ascend",
        arch: ArchType = "ascend910b3",
        impl_func_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        worker: Optional[WorkerInterface] = None,
        bench_type: Literal["kernelbench", "sol", "cann"] = "kernelbench",
    ):
        """Initialize a verifier from one task and execution configuration."""
        if not config:
            raise ValueError("config is required for KernelVerifier")
        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        self.dsl = normalize_dsl(dsl, backend)
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.task_id = task_id
        self.bench_type = bench_type
        self.config = config
        self.log_dir = config.get("log_dir")
        self.config["bench_type"] = bench_type
        self.config["arch"] = self.arch
        self.dsl_adapter = get_dsl_adapter(self.dsl)
        self._configure_framework_bundle()
        default_impl_name = (
            self.dsl_adapter.impl_func_name_template.format(
                op_name=op_name,
                dsl=dsl,
                framework=framework,
            )
        )
        self.impl_func_name = impl_func_name or default_impl_name
        from op_autoresearch.op.utils.config_utils import check_backend_arch

        check_backend_arch(self.backend, self.arch)
        self.worker = worker
        self._profile_generation_enabled = True
        self.last_verify_ok: Optional[bool] = None
        self.last_verify_sidecar: Optional[Dict[str, Any]] = None
        self.last_verify_dir: Optional[str] = None

    @property
    def profile_generation_enabled(self) -> bool:
        """Whether the current project should include candidate profiling."""
        return self._profile_generation_enabled

    @staticmethod
    def check_task_desc_static(code: str) -> Tuple[bool, str]:
        """
        静态检查 task_desc 代码是否符合规范

        Args:
            code: task_desc 代码字符串

        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        try:
            tree = ast.parse(code)
            missing = _missing_task_desc_symbols(tree)
            if missing:
                return False, f"Missing required components in task_desc: {', '.join(missing)}"
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error in task_desc: {e}"
        except Exception as e:
            return False, f"Static check failed: {e}"

    @staticmethod
    def prepare_code_lines(code_snippet: Any) -> List[str]:
        """将多行代码片段规范化为按行列表，方便模板渲染时控制缩进。"""
        if not code_snippet:
            return []
        if isinstance(code_snippet, (list, tuple)):
            lines: List[str] = []
            for snippet in code_snippet:
                lines.extend(KernelVerifier.prepare_code_lines(snippet))
            return lines
        if isinstance(code_snippet, str):
            normalized = textwrap.dedent(code_snippet).strip("\n")
            if not normalized:
                return []
            return normalized.split("\n")
        raise TypeError(f"Unsupported code snippet type: {type(code_snippet)}")

    @staticmethod
    def pack_directory(dir_path: str) -> bytes:
        """将目录打包为tar字节流"""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
            for root, _dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    tar_file.add(file_path, arcname=arcname)
        return tar_buffer.getvalue()

    @staticmethod
    def _should_skip_base_profile(settings: Dict[str, Any]) -> bool:
        if settings.get("skip_base_profile", False):
            return True
        override = settings.get("override_base_section")
        if not isinstance(override, dict):
            return False
        average = override.get("avg_us")
        return (
            isinstance(average, (int, float))
            and 0 < average < float("inf")
        )

    @staticmethod
    def _baseline_cache_payload(
        base_time_us: float,
        warmup_times: int,
        run_times: int,
        artifacts: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        raw_json = (artifacts or {}).get("base_profile_result.json")
        if raw_json:
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError as exc:
                logger.debug("invalid base_profile_result.json artifact: %s", exc)
        return build_baseline_cache_payload(
            base_time_us=base_time_us,
            warmup_times=warmup_times,
            run_times=run_times,
        )

    # ------------------------------------------------------------------
    # Autotune 逐 config 验证辅助方法（OP_AUTORESEARCH_VERIFY_PER_CONFIG=1 时启用）
    # ------------------------------------------------------------------

    def materialize_framework_bundle(self, target_dir: str,
                                      framework_code: str,
                                      target_filename: Optional[str] = None
                                      ) -> str:
        """Write the framework reference module and its sidecars to
        ``target_dir`` as a single unit.

        ``target_filename`` is the .py basename to land at. Sidecars in
        ``self.framework_aux_files`` are written verbatim by their declared
        task-relative names. Callers that want
        ``reference.py/reference.json`` should set the framework filename and
        import module to ``reference`` instead of relying on implicit sidecar
        renames.

        Returns the absolute path of the .py file written.
        """
        target_filename = target_filename or self.framework_filename

        py_path = os.path.join(target_dir, target_filename)
        try:
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(framework_code)
            logger.debug("[%s] framework 文件已写入: %s", self.op_name, py_path)
        except Exception as e:
            logger.error('[%s] framework 文件写入失败: %s, 错误: %s', self.op_name, py_path, e)
            raise

        if not self.framework_aux_files:
            return py_path

        for rel_name, content in self.framework_aux_files.items():
            rel_parts = rel_name.replace("\\", "/").split("/")
            if os.path.isabs(rel_name) or ".." in rel_parts:
                logger.warning(
                    '[%s] 跳过非法 sidecar 路径: %r', self.op_name, rel_name
                )
                continue
            aux_path = os.path.join(target_dir, rel_name)
            os.makedirs(os.path.dirname(aux_path) or target_dir, exist_ok=True)
            try:
                _write_content(aux_path, content)
                logger.debug(
                    "[%s] sidecar 文件已写入: %s", self.op_name, aux_path
                )
            except Exception as e:
                logger.error(
                    '[%s] sidecar 文件写入失败: %s, 错误: %s', self.op_name, aux_path, e
                )
                raise

        return py_path

    async def check_task_desc_runtime(
        self, task_desc: str, timeout: int = 60
    ) -> Tuple[bool, str]:
        """Execute a minimal reference forward pass on the configured worker."""
        check_dir = os.path.join(
            os.path.expanduser(self.log_dir),
            f"{self.op_name}_check_desc_{self.task_id}",
        )
        os.makedirs(check_dir, exist_ok=True)
        try:
            device_id = _runtime_check_device_id(self.worker)
            self.materialize_framework_bundle(
                check_dir, task_desc, target_filename=REF_FILE_DEFAULT
            )
            script = _runtime_check_script(self.framework, device_id)
            verify_file = os.path.join(check_dir, f"verify_{self.op_name}.py")
            _write_text(verify_file, script)
            if not self.worker:
                raise RuntimeError("Worker not set for runtime check")
            package_data = self.pack_directory(check_dir)
            success, log, _ = await self.worker.verify(
                package_data,
                f"{self.task_id}_check",
                self.op_name,
                timeout,
            )
            if success and "REFERENCE_CHECK_SUCCESS" in log:
                return True, ""
            return False, f"Runtime check failed:\n{log}"
        except Exception as error:
            return False, f"Runtime check exception: {error}"
        finally:
            shutil.rmtree(check_dir, ignore_errors=True)

    async def generate_reference_data(
        self,
        task_desc: str,
        timeout: Optional[int] = None,
        save_inputs: bool = False,
        device_id: Optional[int] = None,
    ) -> Tuple[bool, str, bytes]:
        """Generate serialized reference outputs on the configured worker."""
        timeout = resolve_reference_timeout(timeout)
        ref_dir = os.path.join(
            os.path.expanduser(self.log_dir),
            f"{self.op_name}_gen_ref_{self.task_id}",
        )
        os.makedirs(ref_dir, exist_ok=True)
        try:
            self.materialize_framework_bundle(
                ref_dir, task_desc, target_filename=REF_FILE_DEFAULT
            )
            target_device_id = (
                0 if device_id is None or device_id < 0 else int(device_id)
            )
            script = _render_template_file(
                REFERENCE_GENERATION_TEMPLATE_PATH,
                save_inputs=save_inputs,
                backend=self.backend,
                device_id=target_device_id,
                op_name=self.op_name,
            )
            script_file = os.path.join(ref_dir, f"verify_{self.op_name}.py")
            _write_text(script_file, script)
            if not self.worker:
                raise RuntimeError("Worker not set for reference generation")
            package_data = self.pack_directory(ref_dir)
            success, log, ref_bytes = await self.worker.generate_reference(
                package_data,
                f"{self.task_id}_gen_ref",
                self.op_name,
                timeout,
            )
            if not success:
                return False, f"Reference generation failed:\n{log}", b""
            return True, log, ref_bytes
        except Exception as error:
            return False, f"Reference generation exception: {error}", b""
        finally:
            shutil.rmtree(ref_dir, ignore_errors=True)

    async def profile_single_task(
        self,
        request: SingleTaskProfileRequest,
    ) -> Dict[str, Any]:
        """
        执行单个任务的性能测试（只测量 task_desc 的性能，不进行 base vs generation 对比）

        此功能用于单独测量某段代码（包含 Model 类）的执行性能，会临时创建目录并生成 profile 脚本。

        Args:
            task_desc: 包含 Model, get_inputs, get_init_inputs 的代码字符串
            warmup_times: 预热次数
            run_times: 实际运行次数
            timeout: 超时时间
            device_id: 设备ID

        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        warmup_times = resolve_warmup_times(request.warmup_times)
        run_times = resolve_run_times(request.run_times)
        timeout = resolve_eval_timeout(request.timeout)
        # 1. 创建临时目录
        profile_dir = os.path.join(os.path.expanduser(self.log_dir),
                                   f"{self.op_name}_profile_single_{self.task_id}")
        os.makedirs(profile_dir, exist_ok=True)

        try:
            # framework code + sidecar 一起落盘（bundle 内部决定 .py 名 +
            # sidecar 跟 stem 改名）。
            self.materialize_framework_bundle(
                profile_dir, request.task_desc)

            # 3. 使用模板生成性能测试脚本
            script_file = os.path.join(profile_dir, f"profile_single_{self.op_name}.py")
            self.gen_profile_single_task_file(
                script_file,
                request.device_id,
                warmup_times,
                run_times,
            )

            # 4. 打包目录
            package_data = self.pack_directory(profile_dir)

            # 5. 使用 Worker.profile_single_task 执行
            if not self.worker:
                raise RuntimeError("Worker not set for profile_single_task")

            profile_settings = {
                'warmup_times': warmup_times,
                'run_times': run_times,
                'timeout': timeout
            }

            result = await self.worker.profile_single_task(
                package_data, f"{self.task_id}_profile_single", self.op_name, profile_settings
            )

            return result

        except Exception as e:
            logger.error('[%s] profile_single_task exception: %s', self.op_name, e, exc_info=True)
            return {'time_us': float('inf'), 'success': False, 'log': f"Profile single task exception: {str(e)}"}
        finally:
            # 清理临时目录
            shutil.rmtree(profile_dir, ignore_errors=True)

    def gen_profile_single_task_file(
        self, profile_file: str, device_id: int,
        warmup_times: int, run_times: int,
    ) -> None:
        """Render the framework-only single-task profiling script."""
        profile_spec = ProfileProjectSpec(
            device_id=device_id,
            warmup_times=warmup_times,
            run_times=run_times,
        )
        context = self._single_profile_template_context(profile_spec)
        rendered = _render_template_file(
            PROFILE_SINGLE_TASK_TEMPLATE_PATH, **context
        )
        _write_text(profile_file, rendered)
        logger.info(
            "[%s] wrote single-task profile script: %s",
            self.op_name,
            profile_file,
        )

    def detect_dynamic_shape(self) -> bool:
        """True iff the ref module exposes a multi-shape factory (auto-
        detected) or ``framework_factory_names.is_dynamic_shape=True``
        is explicitly declared.
        """
        declared = (self.framework_factory_names or {}).get("is_dynamic_shape")
        if isinstance(declared, bool):
            return declared
        return self._resolve_dyn_factory() is not None

    def get_data_cache_config(self):
        return load_verifier_data_cache_config(self.config)

    def get_reference_cache_key(self) -> str:
        return build_reference_cache_key(
            CacheIdentity(
                op_name=self.op_name,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                bench_type=self.bench_type,
                task_id=self._get_data_cache_key_id(),
            ),
            self.framework_code,
        )

    def get_baseline_cache_key(self, warmup_times: int, run_times: int) -> Optional[str]:
        cache_source = self._get_baseline_cache_source()
        if cache_source is None:
            return None
        return build_baseline_cache_key(
            CacheIdentity(
                op_name=self.op_name,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                bench_type=self.bench_type,
                dsl=self.dsl,
                task_id=self._get_data_cache_key_id(),
            ),
            cache_source,
            warmup_times,
            run_times,
        )

    def is_valid_cached_reference_data(self, reference_data: bytes) -> bool:
        if not reference_data:
            return False
        if self.framework != "torch":
            return True
        payload = self._load_cached_torch_reference_payload(reference_data)
        if payload is None:
            return False

        if not isinstance(payload, dict):
            logger.warning('[%s] Verifier Data Cache reference data 格式无效，准备重新生成', self.op_name)
            return False
        if "outputs" not in payload:
            logger.warning('[%s] Verifier Data Cache reference data 缺少 outputs，准备重新生成', self.op_name)
            return False
        if not payload.get("save_inputs") or payload.get("inputs") is None:
            logger.warning(
                '[%s] Verifier Data Cache reference data 缺少可复用 inputs，准备重新生成', self.op_name
            )
            return False
        return True

    async def prepare_cached_reference_data(
        self, device_id: int
    ) -> Optional[bytes]:
        """Load or generate reusable single-shape reference data."""
        if self.bench_type != "kernelbench":
            self._clear_managed_reference_data(
                "reference cache only supports kernelbench"
            )
            return None
        if self.detect_dynamic_shape():
            self._clear_managed_reference_data(
                "dynamic shapes do not reuse static reference data"
            )
            logger.info(
                "[%s] skipping reference cache for dynamic shapes",
                self.op_name,
            )
            return None

        cache_config = self.get_data_cache_config()
        cache_key = self.get_reference_cache_key()
        if self._caller_reference_takes_precedence(
            cache_config, cache_key
        ):
            return None
        if (
            not cache_config.enabled
            or not cache_config.cache_reference_data
        ):
            return None
        try:
            return await self._load_or_generate_cached_reference(
                cache_config, cache_key, device_id
            )
        except TimeoutError as error:
            logger.warning(
                "[%s] reference cache lock timed out; using live reference: %s",
                self.op_name,
                error,
            )
            return None

    def apply_cached_reference_data(self, reference_data: bytes, cache_key: Optional[str] = None) -> None:
        if not reference_data:
            return
        self.config["use_reference_data"] = True
        self.config["use_reference_inputs"] = True
        self.config["reference_data"] = reference_data
        self.config["_data_cache_reference_key"] = cache_key or self.get_reference_cache_key()

    def gen_verify_project(
        self, impl_code: str, verify_dir: str, device_id: int = 0
    ) -> None:
        """Generate the selected benchmark's verification project."""
        if self.bench_type == "sol":
            from op_autoresearch.op.verifier.sol_verifier import (
                generate_sol_verify_project,
            )
            generate_sol_verify_project(self, impl_code, verify_dir, device_id)
            return
        if self.bench_type == "cann":
            from op_autoresearch.op.cann_correctness import (
                generate_cann_verify_project,
            )
            generate_cann_verify_project(self, impl_code, verify_dir, device_id)
            return

        logger.info(
            "[%s] generating verification project in %s on device %s",
            self.op_name,
            verify_dir,
            device_id,
        )
        reference_mode = self._prepare_verify_reference_mode(verify_dir)
        self._stage_verify_sources(impl_code, verify_dir)
        context = self._verify_template_context(device_id, reference_mode)
        rendered = _render_template_file(TEMPLATE_PATH, **context)
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")
        _write_text(verify_file, rendered)
        logger.info("[%s] wrote verification script: %s", self.op_name, verify_file)

    async def run_verify(self, verify_dir: str,
                         timeout: Optional[int] = None,
                         device_id: int = 0):
        """Run an already generated verification project."""
        timeout = resolve_eval_timeout(timeout)
        verify_script = os.path.join(
            verify_dir, f"verify_{self.op_name}.py"
        )
        logger.info(
            "[%s] running verification script %s (timeout=%ss)",
            self.op_name,
            verify_script,
            timeout,
        )
        try:
            worker = self._ensure_execution_worker(device_id)
            package_data = self._verify_package(worker, verify_dir)
            success, log, artifacts = await worker.verify(
                package_data,
                self.task_id,
                self.op_name,
                timeout + 30,
            )
            sync_artifacts_to_directory(
                artifacts, verify_dir, self.task_id
            )
            log_method = logger.info if success else logger.error
            log_method(
                "[%s] verification %s",
                self.op_name,
                "passed" if success else "failed",
            )
            return success, log
        except Exception as error:
            logger.error(
                "[%s] verification raised: %s",
                self.op_name,
                error,
                exc_info=True,
            )
            return False, str(error)

    def gen_profile_project(
        self,
        verify_dir: str,
        requested_spec: ProfileProjectSpec,
    ) -> None:
        """生成profile项目文件到指定目录

        Args:
            verify_dir: 验证目录
            device_id: 设备ID
            warmup_times: 预热次数
            run_times: 运行次数
            skip_base: 是否跳过 base profile（跨后端场景下设为 True）
        """
        warmup_times = resolve_warmup_times(requested_spec.warmup_times)
        run_times = resolve_run_times(requested_spec.run_times)
        profile_spec = ProfileProjectSpec(
            device_id=requested_spec.device_id,
            warmup_times=warmup_times,
            run_times=run_times,
            skip_base=requested_spec.skip_base,
        )
        if self.bench_type == "sol":
            from op_autoresearch.op.verifier.sol_verifier import (
                generate_sol_profile_project,
            )
            generate_sol_profile_project(self, verify_dir, profile_spec)
            return
        if self.bench_type == "cann":
            from op_autoresearch.op.cann_correctness import (
                generate_cann_profile_project,
            )
            generate_cann_profile_project(self, verify_dir, profile_spec)
            return

        profile_generation_enabled = self.profile_generation_enabled

        # 生成基准性能测试脚本（如果不跳过）
        if not profile_spec.skip_base:
            profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_base.py")
            self.gen_profile_file_from_template(
                PROFILE_BASE_TEMPLATE_PATH, profile_file, profile_spec
            )
        else:
            logger.info('[%s] 跳过 base profile 生成（使用缓存 baseline 或跨后端场景）', self.op_name)

        # 生成性能测试脚本
        if profile_generation_enabled:
            profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_generation.py")
            self.gen_profile_file_from_template(
                PROFILE_GENERATION_TEMPLATE_PATH, profile_file, profile_spec
            )
        else:
            logger.info('[%s] 跳过 generation profile 生成（上一轮 verify 未通过）', self.op_name)

    def gen_profile_file_from_template(
        self,
        template_path: str,
        profile_file: str,
        profile_spec: ProfileProjectSpec,
    ) -> None:
        """Render one profile script from its resource template."""
        is_base = "base" in os.path.basename(template_path).lower()
        logger.info(
            "[%s] rendering %s profile script from %s",
            self.op_name,
            "base" if is_base else "generation",
            os.path.basename(template_path),
        )
        context = self._profile_template_context(profile_spec, is_base)
        rendered = _render_template_file(template_path, **context)
        _write_text(profile_file, rendered)
        logger.info(
            "[%s] wrote profile script: %s", self.op_name, profile_file
        )

    def save_speedup_result(self, record: _SpeedupRecord) -> None:
        """Append one profiling result to the speedup history."""
        try:
            profiling_dir = os.path.join(
                os.path.expanduser(self.log_dir),
                self.op_name,
                "profiling",
            )
            os.makedirs(profiling_dir, exist_ok=True)
            filepath = os.path.join(profiling_dir, "speed_up_record.txt")
            line = (
                f"op_name: {self.op_name}, task_id: {self.task_id}, "
                f"unique_dir: {record.unique_dir}, "
                f"base_time: {record.base_time:.6f} us, "
                f"generation_time: {record.gen_time:.6f} us, "
                f"speedup: {record.speedup:.6f}x"
            )
            if (
                record.roofline_time is not None
                and record.roofline_speedup is not None
            ):
                line += (
                    f", roofline_time: {record.roofline_time:.6f} us, "
                    f"roofline_speedup: {record.roofline_speedup:.6f}x"
                )
            with open(filepath, "a", encoding="utf-8") as file:
                file.write(line + "\n")
            logger.debug(
                "[%s:%s] saved speedup result",
                self.task_id,
                self.op_name,
            )
        except OSError as error:
            logger.warning(
                "[%s:%s] failed to save speedup result: %s",
                self.task_id,
                self.op_name,
                error,
            )

    async def run_profile(
        self,
        task_info: Dict[str, Any],
        current_step: int = 0,
        device_id: int = -1,
        profile_settings: Optional[dict] = None,
    ) -> dict:
        """Generate and execute the profiling project for one iteration."""
        unique_dir = f"Iteration{self.task_id}_Step{current_step}_verify"
        session = None
        try:
            session = self._create_profile_session(
                task_info,
                unique_dir,
                profile_settings,
            )
            await self._acquire_profile_device(session, device_id)
            self._prepare_profile_project(task_info, session)
            result = await self._dispatch_profile(session, device_id)
            return self._finalize_profile_result(session, result)
        except Exception as error:
            logger.warning(
                "[%s:%s] profiling failed: %s",
                self.task_id,
                self.op_name,
                error,
            )
            result = empty_profile_result(error=str(error))
            result.update(case_descs=[], unique_dir=unique_dir)
            return result
        finally:
            if session is not None:
                await self._release_profile_device(session)

    def read_autotune_results_from_directory(self, verify_dir: str) -> str:
        """从验证目录读取所有autotune结果并格式化输出

        读取指定目录下的所有 autotune_info_case_*.json 文件，
        并以类似 TRITON_PRINT_AUTOTUNING=1 的格式输出。

        Args:
            verify_dir: 验证目录路径

        Returns:
            格式化的autotune结果字符串，格式如下：

            Case 0:
            All config timings for kernel_name:
              Config 1: BLOCK_M=128, BLOCK_N=256 -> 145.2300us (BEST)
              Config 2: BLOCK_M=64, BLOCK_N=128 -> 178.5600us
              ...
        """

        result_lines = []

        # 查找所有autotune文件
        verify_path = Path(verify_dir)
        autotune_files = sorted(verify_path.glob("autotune_info_case_*.json"))

        if not autotune_files:
            return ""

        # 逐个读取并格式化
        for autotune_file in autotune_files:
            # 提取case索引
            case_idx = autotune_file.stem.split('_')[-1]

            try:
                with open(autotune_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result_lines.extend(_format_autotune_case(case_idx, data))

            except Exception as e:
                logger.warning(
                    "[%s: %s] 读取autotune文件失败 %s: %s",
                    self.op_name,
                    self.task_id,
                    autotune_file.name,
                    e,
                )

        return "\n".join(result_lines)

    async def run(
        self,
        task_info: Dict[str, Any],
        current_step: int = 0,
        device_id: int = -1,
    ):
        """Verify the generated implementation for one workflow step."""
        logger.info("Verifier Run - Step: %s", current_step)
        self.last_verify_ok = None
        target_code = task_info.get("coder_code", "")
        if not target_code:
            logger.error("No target code found for verification")
            self.last_verify_ok = False
            return False, "No target code found for verification"

        verify_dir = self._create_verify_dir(current_step)
        self.dsl_adapter.prepare_config(self.config, task_info=task_info)
        actual_device, acquired_device, lease = (
            await self._acquire_verify_device(device_id)
        )
        try:
            config_result = await self._run_config_verification_if_enabled(
                task_info,
                ConfigVerificationRequest(
                    target_code=target_code,
                    verify_dir=verify_dir,
                    device_id=actual_device,
                    timeout=resolve_eval_timeout(
                        self.config.get("verify_timeout")
                    ),
                    current_step=current_step,
                ),
            )
            if config_result is not None:
                return config_result
            await self._prepare_verify_reference(actual_device)
            verify_result, verify_logs = await self._execute_verify_project(
                target_code, verify_dir, actual_device
            )
            self._load_verify_sidecar(verify_dir)
            self._save_verification_result_to_jsonl(
                _VerificationRecord(
                    verify_dir=verify_dir,
                    current_step=current_step,
                    passed=verify_result,
                    logs=verify_logs,
                )
            )
            self.last_verify_ok = bool(verify_result)
            return verify_result, verify_logs
        finally:
            await self._release_verify_device(acquired_device, lease)

    def _configure_framework_bundle(self) -> None:
        auxiliary_files = self.config.get("framework_aux_files") or {}
        factory_names = self.config.get("framework_factory_names") or {}
        if not isinstance(auxiliary_files, dict):
            raise TypeError("config['framework_aux_files'] must be a dict")
        if not isinstance(factory_names, dict):
            raise TypeError(
                "config['framework_factory_names'] must be a dict"
            )
        self.framework_aux_files = auxiliary_files
        self.framework_factory_names = factory_names
        self.framework_module_name = (
            self.config.get("framework_module_name")
            or f"{self.op_name}_{self.framework}"
        )
        self.framework_filename = (
            self.config.get("framework_filename")
            or f"{self.framework_module_name}.py"
        )

    def _ensure_execution_worker(self, device_id: int):
        if self.worker is None:
            if device_id == -1:
                raise RuntimeError(
                    f"[{self.op_name}] worker is required when device_id=-1"
                )
            import warnings

            warnings.warn(
                "Implicit LocalWorker creation is deprecated; register a "
                "worker before running the verifier.",
                DeprecationWarning,
                stacklevel=3,
            )
            from op_autoresearch.core.async_pool.device_pool import DevicePool
            from op_autoresearch.core.worker.local_worker import LocalWorker

            logger.warning(
                "[%s] creating a temporary LocalWorker for device %s",
                self.op_name,
                device_id,
            )
            self.worker = LocalWorker(
                device_pool=DevicePool([device_id]),
                backend=self.backend,
            )
        self._validate_local_worker(self.worker)
        return self.worker

    def _validate_local_worker(self, worker) -> None:
        from op_autoresearch.core.worker.local_worker import LocalWorker

        if isinstance(worker, LocalWorker) and worker.device_pool is None:
            raise RuntimeError(
                f"[{self.op_name}] LocalWorker must have a device_pool"
            )

    def _verify_package(self, worker, verify_dir: str) -> Union[str, bytes]:
        from op_autoresearch.core.worker.local_worker import LocalWorker

        if isinstance(worker, LocalWorker):
            return verify_dir
        logger.info("[%s] packing verification project", self.op_name)
        return self.pack_directory(verify_dir)

    def _profile_common_context(
        self, profile_spec: ProfileProjectSpec
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        framework_adapter = get_framework_adapter(self.framework)
        dsl_adapter = get_dsl_adapter(self.dsl)
        backend_adapter = get_backend_adapter(self.backend)
        warmup_times = resolve_warmup_times(profile_spec.warmup_times)
        run_times = resolve_run_times(profile_spec.run_times)
        is_dynamic_shape = self.detect_dynamic_shape()
        backend_adapter.setup_environment(profile_spec.device_id, self.arch)
        context = {
            "op_name": self.op_name,
            "framework": self.framework,
            "dsl": self.dsl,
            "device_id": profile_spec.device_id,
            "impl_func_name": self.impl_func_name,
            "backend": self.backend,
            "arch": self.arch,
            "warmup_times": warmup_times,
            "run_times": run_times,
            "total_count": warmup_times + run_times,
            "is_dynamic_shape": is_dynamic_shape,
            "framework_imports": self.prepare_code_lines(
                framework_adapter.get_import_statements()
            ),
            "framework_model_import": self.prepare_code_lines(
                framework_adapter.get_framework_import(
                    self.op_name,
                    is_dynamic_shape,
                    inputs_factory_name=self._resolve_dyn_factory(),
                    module_name=self.framework_module_name,
                )
            ),
            "device_setup_code": self.prepare_code_lines(
                framework_adapter.get_device_setup_code(
                    self.backend, self.arch, profile_spec.device_id
                )
            ),
            "process_input_code": self.prepare_code_lines(
                framework_adapter.get_process_input_code(
                    self.backend, self.dsl
                )
            ),
            "set_seed_code": self.prepare_code_lines(
                framework_adapter.get_set_seed_code(self.backend)
            ),
            "tensor_type_name": framework_adapter.get_tensor_type_name(),
        }
        return framework_adapter, dsl_adapter, context

    def _generation_profile_context(
        self, framework_adapter, dsl_adapter,
        profile_spec: ProfileProjectSpec,
    ) -> Dict[str, Any]:
        dsl_adapter.prepare_config(self.config, task_info=None)
        needs_binary_io = dsl_adapter.needs_binary_io
        binary_io = ""
        if needs_binary_io:
            binary_io = framework_adapter.get_binary_io_functions(
                self.op_name
            )
        benchmark = dsl_adapter.benchmark_impl(
            BenchmarkSpec(
                inputs="inputs",
                warmup=resolve_warmup_times(profile_spec.warmup_times),
                runs=resolve_run_times(profile_spec.run_times),
                backend=self.backend,
                op_name=self.op_name,
                clear_l2_cache=dsl_adapter.benchmark_requires_l2_clear,
                framework=self.framework,
            )
        )
        snippets = {
            "dsl_imports": dsl_adapter.get_import_statements(self.framework),
            "dsl_impl_import": dsl_adapter.get_impl_import(
                self.op_name, self.impl_func_name
            ),
            "special_setup_code": dsl_adapter.get_special_setup_code(
                framework=self.framework
            ),
            "create_impl_code": dsl_adapter.create_impl_module(
                self.framework, framework_adapter
            ),
            "binary_io_functions": binary_io,
            "benchmark_code": benchmark,
        }
        return {
            key: self.prepare_code_lines(value)
            for key, value in snippets.items()
        } | {"needs_binary_io": needs_binary_io}

    def _profile_template_context(
        self, profile_spec: ProfileProjectSpec, is_base: bool
    ) -> Dict[str, Any]:
        framework_adapter, dsl_adapter, context = (
            self._profile_common_context(profile_spec)
        )
        context.update({
            "dsl_imports": [],
            "dsl_impl_import": [],
            "special_setup_code": [],
            "create_impl_code": [],
            "binary_io_functions": [],
            "needs_binary_io": False,
        })
        if is_base:
            benchmark = self._generate_base_benchmark_code(
                dsl_adapter,
                resolve_warmup_times(profile_spec.warmup_times),
                resolve_run_times(profile_spec.run_times),
                clear_l2_cache=dsl_adapter.benchmark_requires_l2_clear,
            )
            context["benchmark_code"] = self.prepare_code_lines(benchmark)
        else:
            context.update(
                self._generation_profile_context(
                    framework_adapter, dsl_adapter, profile_spec
                )
            )
        return context

    def _single_profile_template_context(
        self, profile_spec: ProfileProjectSpec
    ) -> Dict[str, Any]:
        _framework_adapter, dsl_adapter, context = (
            self._profile_common_context(profile_spec)
        )
        benchmark = self._generate_base_benchmark_code(
            dsl_adapter,
            resolve_warmup_times(profile_spec.warmup_times),
            resolve_run_times(profile_spec.run_times),
        )
        context["benchmark_code"] = self.prepare_code_lines(benchmark)
        return context

    def _create_profile_session(
        self,
        task_info: Dict[str, Any],
        unique_dir: str,
        profile_settings: Optional[dict],
    ) -> _ProfileSession:
        logger.info("[%s] preparing profile config", self.op_name)
        self.dsl_adapter.prepare_config(self.config, task_info=task_info)
        settings = dict(profile_settings or {})
        run_times = resolve_run_times(settings.get("run_times"))
        warmup_times = resolve_warmup_times(settings.get("warmup_times"))
        if settings.get("override_base_section") is None:
            cached_time = self._get_cached_baseline_time_us(
                warmup_times, run_times
            )
            if cached_time is not None:
                settings["override_base_section"] = make_profile_section(
                    cached_time, method="override"
                )
                settings["skip_base_profile"] = True
        return _ProfileSession(
            unique_dir=unique_dir,
            settings=settings,
            warmup_times=warmup_times,
            run_times=run_times,
            base_only=getattr(self, "last_verify_ok", None) is False,
        )

    async def _acquire_profile_device(
        self, session: _ProfileSession, requested_device: int
    ) -> None:
        if self.worker is None:
            session.actual_device_id = (
                requested_device if requested_device != -1 else 0
            )
            logger.info(
                "[%s] using device %s without a registered worker",
                self.op_name,
                session.actual_device_id,
            )
            return
        self._validate_local_worker(self.worker)
        device, lease = await self.worker.acquire_device(self.task_id)
        session.actual_device_id = device
        session.acquired_device = device
        session.acquired_lease = lease
        logger.info(
            "[%s] acquired device %s for profiling",
            self.op_name,
            device,
        )

    def _remove_generation_profile_artifacts(
        self, verify_dir: str
    ) -> None:
        filenames = (
            f"profile_{self.op_name}_generation.py",
            "generation_profile_result.json",
            "roofline_profile_result.json",
        )
        for filename in filenames:
            path = os.path.join(verify_dir, filename)
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                logger.warning(
                    "[%s] failed to remove stale artifact %s: %s",
                    self.op_name,
                    path,
                    error,
                )

    def _prepare_profile_project(
        self, task_info: Dict[str, Any], session: _ProfileSession
    ) -> None:
        session.verify_dir = os.path.join(
            os.path.expanduser(self.log_dir),
            self.op_name,
            session.unique_dir,
        )
        os.makedirs(session.verify_dir, exist_ok=True)
        if session.base_only:
            self.materialize_framework_bundle(
                session.verify_dir, self.framework_code
            )
            self._remove_generation_profile_artifacts(session.verify_dir)
        elif not self._verify_impl_artifacts_ready(session.verify_dir):
            impl_code = task_info.get("coder_code", "")
            if not impl_code:
                raise ValueError("task_info is missing coder_code")
            self.gen_verify_project(
                impl_code, session.verify_dir, session.actual_device_id
            )
        session.skip_base = self._should_skip_base_profile(session.settings)
        generation_enabled = self.profile_generation_enabled
        self._profile_generation_enabled = not session.base_only
        try:
            self.gen_profile_project(
                session.verify_dir,
                ProfileProjectSpec(
                    device_id=session.actual_device_id,
                    warmup_times=session.warmup_times,
                    run_times=session.run_times,
                    skip_base=session.skip_base,
                ),
            )
        finally:
            self._profile_generation_enabled = generation_enabled

    def _profile_worker_settings(
        self, session: _ProfileSession
    ) -> Dict[str, Any]:
        settings = {
            **session.settings,
            "backend": self.backend,
            "dsl": self.dsl,
            "op_name": self.op_name,
            "framework": self.framework,
            "arch": self.arch,
            "bench_type": self.bench_type,
            "enable_roofline": session.settings.get(
                "enable_roofline", True
            ),
            "roofline_arch_config": session.settings.get(
                "roofline_arch_config",
                self.config.get("roofline_arch_config"),
            ),
        }
        return settings

    async def _dispatch_profile(
        self, session: _ProfileSession, requested_device: int
    ) -> Dict[str, Any]:
        logger.info("[%s] packing profile project", self.op_name)
        package_data = self.pack_directory(session.verify_dir)
        worker = self._ensure_execution_worker(requested_device)
        result = await worker.profile(
            package_data,
            self.task_id,
            self.op_name,
            self._profile_worker_settings(session),
        )
        sync_artifacts_to_directory(
            result.get("artifacts", {}),
            session.verify_dir,
            self.task_id,
        )
        return result

    def _profile_case_descriptions(self) -> List[str]:
        sidecar = getattr(self, "last_verify_sidecar", None)
        if not isinstance(sidecar, dict):
            return []
        return [
            case.get("case_desc", "")
            for case in (sidecar.get("per_case") or [])
            if isinstance(case, dict)
        ]

    def _record_profile_metrics(
        self, session: _ProfileSession, result: Dict[str, Any]
    ) -> None:
        gen_time = result.get("gen_time")
        base_time = result.get("base_time")
        artifacts = result.get("artifacts", {})
        valid_base = (
            base_time is not None
            and 0 < base_time < float("inf")
        )
        if not session.skip_base and valid_base:
            self._store_baseline_result_in_data_cache(
                base_time_us=base_time,
                warmup_times=session.warmup_times,
                run_times=session.run_times,
                artifacts=artifacts,
            )
        base_display = (
            base_time if base_time is not None else float("inf")
        )
        gen_display = gen_time if gen_time is not None else float("inf")
        if gen_time is not None:
            roofline_time = result.get("roofline_time")
            self.save_speedup_result(
                _SpeedupRecord(
                    speedup=result.get("speedup", 0.0),
                    base_time=base_display,
                    gen_time=gen_display,
                    unique_dir=session.unique_dir,
                    roofline_time=roofline_time,
                    roofline_speedup=(
                        result.get("roofline_speedup", 0.0)
                        if roofline_time is not None
                        else None
                    ),
                )
            )
        self._log_profile_metrics(result, base_display, gen_display)

    def _log_profile_metrics(
        self,
        result: Dict[str, Any],
        base_time: float,
        gen_time: float,
    ) -> None:
        logger.info("orig performance is %.2f us", base_time)
        if result.get("gen_time") is None:
            logger.info("generated profile skipped after failed verification")
        else:
            logger.info("op_autoresearch performance is %.2f us", gen_time)
        roofline_time = result.get("roofline_time")
        if roofline_time is not None:
            logger.info("solar roofline performance is %.2f us", roofline_time)
            logger.info(
                "roofline speedup is %.4fx",
                result.get("roofline_speedup", 0.0),
            )
        logger.info(
            "[%s:%s] profiling complete: %.2f%% of baseline",
            self.task_id,
            self.op_name,
            result.get("speedup", 0.0) * 100.0,
        )

    def _finalize_profile_result(
        self, session: _ProfileSession, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._record_profile_metrics(session, result)
        profile_result = empty_profile_result(result.get("error"))
        profile_result.update({
            "gen_time": result.get("gen_time"),
            "base_time": result.get("base_time"),
            "speedup": result.get("speedup", 0.0),
            "per_shape_gen_us": list(
                result.get("per_shape_gen_us") or []
            ),
            "per_shape_base_us": list(
                result.get("per_shape_base_us") or []
            ),
            "case_descs": self._profile_case_descriptions(),
            "gen_method": result.get("gen_method"),
            "base_method": result.get("base_method"),
            "roofline_time": result.get("roofline_time"),
            "roofline_speedup": result.get("roofline_speedup", 0.0),
            "roofline": result.get("roofline"),
            "artifacts": result.get("artifacts", {}),
            "unique_dir": session.unique_dir,
        })
        if self.dsl_adapter.emits_autotune_artifacts:
            summary = self.read_autotune_results_from_directory(
                session.verify_dir
            )
            if summary:
                profile_result["autotune_summary"] = summary
                logger.info(
                    "[%s:%s] autotune configs:\n%s",
                    self.op_name,
                    self.task_id,
                    summary,
                )
        return profile_result

    async def _release_profile_device(
        self, session: _ProfileSession
    ) -> None:
        if session.acquired_device is None:
            return
        await self.worker.release_device(
            session.acquired_device,
            session.acquired_lease,
            self.task_id,
        )
        logger.info(
            "[%s] released device %s",
            self.op_name,
            session.acquired_device,
        )

    async def _acquire_verify_device(
        self, requested_device: int
    ) -> Tuple[int, Optional[int], Any]:
        if self.worker is None:
            actual_device = (
                requested_device if requested_device != -1 else 0
            )
            logger.info(
                "[%s] using device %s without a registered worker",
                self.op_name,
                actual_device,
            )
            return actual_device, None, None
        self._validate_local_worker(self.worker)
        device, lease = await self.worker.acquire_device(self.task_id)
        logger.info(
            "[%s] acquired device %s for verification",
            self.op_name,
            device,
        )
        return device, device, lease

    async def _run_config_verification_if_enabled(
        self,
        task_info: Dict[str, Any],
        request: ConfigVerificationRequest,
    ) -> Optional[Tuple[bool, str]]:
        enabled = (
            os.environ.get("OP_AUTORESEARCH_VERIFY_PER_CONFIG", "0") == "1"
            or self.config.get("verify_per_config", False)
        )
        is_autotune = (
            self.dsl_adapter.supports_autotune_configs
            and has_autotune_configs(request.target_code)
        )
        if not enabled or not is_autotune:
            return None
        passed, logs, final_code = await self._verify_configs_separately(request)
        if passed is None:
            return None
        if passed:
            task_info["coder_code"] = final_code
        self.last_verify_ok = bool(passed)
        return passed, logs

    async def _prepare_verify_reference(self, device_id: int) -> None:
        if getattr(self.dsl_adapter, "uses_cannbench_precision", False):
            logger.info(
                "[%s] using live dual-reference precision checks",
                self.op_name,
            )
            return
        reference_data = await self.prepare_cached_reference_data(device_id)
        if reference_data:
            self.apply_cached_reference_data(reference_data)
            logger.info("[%s] applied cached reference data", self.op_name)

    async def _execute_verify_project(
        self, target_code: str, verify_dir: str, device_id: int
    ) -> Tuple[bool, str]:
        generation_log = ""
        try:
            self.gen_verify_project(target_code, verify_dir, device_id)
        except Exception as error:
            logger.error(
                "[%s] failed to generate verification project: %s",
                self.op_name,
                error,
            )
            generation_log = f"Project generation failed: {error}\n"
        result, logs = await self.run_verify(
            verify_dir,
            timeout=resolve_eval_timeout(
                self.config.get("verify_timeout")
            ),
            device_id=device_id,
        )
        return result, generation_log + logs

    def _load_verify_sidecar(self, verify_dir: str) -> None:
        self.last_verify_sidecar = None
        self.last_verify_dir = verify_dir
        sidecar_path = os.path.join(verify_dir, "verify_result.json")
        if not os.path.isfile(sidecar_path):
            return
        try:
            with open(sidecar_path, "r", encoding="utf-8") as file:
                self.last_verify_sidecar = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(
                "[%s] failed to read verify_result.json: %s",
                self.op_name,
                error,
            )

    async def _release_verify_device(
        self, acquired_device: Optional[int], lease: Any
    ) -> None:
        if acquired_device is None:
            return
        await self.worker.release_device(
            acquired_device, lease, self.task_id
        )
        logger.info(
            "[%s] released device %s", self.op_name, acquired_device
        )

    def _caller_reference_takes_precedence(
        self, cache_config: Any, cache_key: str
    ) -> bool:
        has_reference = (
            self.config.get("use_reference_data")
            and self.config.get("reference_data")
        )
        if not has_reference:
            return False
        managed_key = self.config.get("_data_cache_reference_key")
        if not managed_key:
            logger.info(
                "[%s] using caller-provided reference data",
                self.op_name,
            )
            return True
        cache_enabled = (
            cache_config.enabled
            and cache_config.cache_reference_data
        )
        if not cache_enabled:
            self._clear_managed_reference_data("reference cache disabled")
            return False
        if managed_key == cache_key:
            logger.info(
                "[%s] reusing reference data already injected by cache",
                self.op_name,
            )
            return True
        self._clear_managed_reference_data("reference cache key changed")
        return False

    def _read_cached_reference(
        self, cache_config: Any, cache_key: str
    ) -> Optional[bytes]:
        reference = read_reference_data_from_cache(
            cache_config,
            op_name=self.op_name,
            cache_key=cache_key,
        )
        if reference and self.is_valid_cached_reference_data(reference):
            cache_file = get_reference_cache_file_path(
                cache_config,
                op_name=self.op_name,
                cache_key=cache_key,
            )
            logger.info(
                "[%s] reference cache hit: %s",
                self.op_name,
                cache_file,
            )
            return reference
        if reference:
            delete_reference_data_from_cache(
                cache_config,
                op_name=self.op_name,
                cache_key=cache_key,
            )
            logger.info(
                "[%s] removed invalid cached reference data",
                self.op_name,
            )
        return None

    async def _generate_reference_bytes(
        self, device_id: int
    ) -> Optional[bytes]:
        if self.worker is None:
            logger.info(
                "[%s] no worker available to populate reference cache",
                self.op_name,
            )
            return None
        timeout = resolve_reference_timeout(
            self.config.get(
                "reference_data_timeout",
                self.config.get("verify_timeout"),
            )
        )
        try:
            success, logs, reference = await self.generate_reference_data(
                self.framework_code,
                timeout=timeout,
                save_inputs=True,
                device_id=device_id,
            )
        except Exception as error:
            logger.warning(
                "[%s] reference generation raised: %s",
                self.op_name,
                error,
            )
            return None
        if success and reference:
            return reference
        logger.warning(
            "[%s] reference generation failed: %.500s",
            self.op_name,
            logs or "",
        )
        return None

    def _write_cached_reference(
        self, cache_config: Any, cache_key: str, reference: bytes
    ) -> None:
        path = write_reference_data_to_cache(
            cache_config,
            op_name=self.op_name,
            cache_key=cache_key,
            reference_data=reference,
            metadata={
                "framework": self.framework,
                "task_id": self.task_id,
                "cache_key_id": self._get_data_cache_key_id(),
                "backend": self.backend,
                "arch": self.arch,
                "bench_type": self.bench_type,
                "save_inputs": True,
            },
        )
        if path:
            logger.info(
                "[%s] stored reference data in cache: %s",
                self.op_name,
                path,
            )

    async def _load_or_generate_cached_reference(
        self, cache_config: Any, cache_key: str, device_id: int
    ) -> Optional[bytes]:
        async with verifier_data_cache_lock(
            cache_config,
            CacheEntry("reference", self.op_name, cache_key),
        ):
            cached = self._read_cached_reference(
                cache_config, cache_key
            )
            if cached is not None:
                return cached
            reference = await self._generate_reference_bytes(device_id)
            if reference is None:
                return None
            self._write_cached_reference(
                cache_config, cache_key, reference
            )
            return reference

    def _prepare_verify_reference_mode(self, verify_dir: str) -> _VerifyReferenceMode:
        if not self.config.get("use_reference_data", False):
            return _VerifyReferenceMode()
        reference_data = self.config.get("reference_data")
        if not reference_data:
            logger.warning(
                "[%s] use_reference_data is enabled but reference_data is empty",
                self.op_name,
            )
            return _VerifyReferenceMode()
        file_name = f"{self.op_name}_reference.pt"
        destination = os.path.join(verify_dir, file_name)
        try:
            with open(destination, "wb") as file:
                file.write(reference_data)
        except OSError as error:
            logger.error(
                "[%s] failed to write reference data: %s",
                self.op_name,
                error,
            )
            return _VerifyReferenceMode()
        logger.info(
            "[%s] wrote %s bytes of reference data to %s",
            self.op_name,
            len(reference_data),
            destination,
        )
        return _VerifyReferenceMode(
            use_data=True,
            use_inputs=bool(self.config.get("use_reference_inputs", False)),
            reference_file=file_name,
        )

    def _stage_verify_sources(self, impl_code: str, verify_dir: str) -> None:
        self.materialize_framework_bundle(verify_dir, self.framework_code)
        self.dsl_adapter.materialize_impl(
            MaterializeSpec(
                impl_code=impl_code,
                verify_dir=verify_dir,
                op_name=self.op_name,
                framework=self.framework,
                dsl_name=self.dsl,
                config=self.config,
            )
        )
        if getattr(self.dsl_adapter, "uses_cannbench_precision", False):
            from op_autoresearch.op.cann_correctness import CORE_PY_PATH

            shutil.copy2(
                CORE_PY_PATH,
                os.path.join(verify_dir, "cann_correctness.py"),
            )

    def _verify_adapter_snippets(self, device_id: int,
                                 is_dynamic_shape: bool) -> Dict[str, Any]:
        framework_adapter = get_framework_adapter(self.framework)
        dsl_adapter = get_dsl_adapter(self.dsl)
        backend_adapter = get_backend_adapter(self.backend)
        dsl_adapter.prepare_config(self.config, task_info=None)
        backend_adapter.setup_environment(device_id, self.arch)
        needs_binary_io = dsl_adapter.needs_binary_io
        snippets = {
            "framework_imports": framework_adapter.get_import_statements(),
            "framework_model_import": framework_adapter.get_framework_import(
                self.op_name,
                is_dynamic_shape,
                inputs_factory_name=self._resolve_dyn_factory(),
                module_name=self.framework_module_name,
            ),
            "dsl_imports": dsl_adapter.get_import_statements(self.framework),
            "dsl_impl_import": dsl_adapter.get_impl_import(
                self.op_name, self.impl_func_name
            ),
            "special_setup_code": dsl_adapter.get_special_setup_code(
                framework=self.framework
            ),
            "device_setup_code": framework_adapter.get_device_setup_code(
                self.backend, self.arch, device_id
            ),
            "process_input_code": framework_adapter.get_process_input_code(
                self.backend, self.dsl
            ),
            "create_impl_code": dsl_adapter.create_impl_module(
                self.framework, framework_adapter
            ),
            "call_impl_code": dsl_adapter.call_impl("inputs_for_impl"),
            "set_seed_code": framework_adapter.get_set_seed_code(self.backend),
            "reference_sync_code": _benchmark_sync_code(
                self.framework, self.backend
            ),
            "binary_io_functions": (
                framework_adapter.get_binary_io_functions(self.op_name)
                if needs_binary_io else ""
            ),
            "needs_binary_io": needs_binary_io,
            "tensor_type_name": framework_adapter.get_tensor_type_name(),
        }
        snippets.update(self._verify_compare_snippets(framework_adapter))
        return snippets

    def _verify_compare_snippets(self, framework_adapter) -> Dict[str, str]:
        result = {"compare_code": framework_adapter.get_compare_code()}
        use_cannbench = (
            getattr(self.dsl_adapter, "uses_cannbench_precision", False)
            and self.framework == "torch"
        )
        if use_cannbench:
            from op_autoresearch.op import cann_correctness

            result["reference_call_code"] = cann_correctness.reference_call_snippet()
            result["compare_outputs_code"] = cann_correctness.compare_snippet()
        else:
            result["reference_call_code"] = (
                "framework_output = framework_model(*inputs_for_framework)"
            )
            result["compare_outputs_code"] = (
                framework_adapter.get_compare_outputs_code()
            )
        return result

    def _verify_template_context(self, device_id: int,
                                 reference_mode: _VerifyReferenceMode) -> Dict[str, Any]:
        dynamic_shape = self.detect_dynamic_shape()
        snippets = self._verify_adapter_snippets(device_id, dynamic_shape)
        context = {
            "op_name": self.op_name,
            "framework": self.framework,
            "dsl": self.dsl,
            "device_id": device_id,
            "impl_func_name": self.impl_func_name,
            "backend": self.backend,
            "arch": self.arch,
            "is_dynamic_shape": dynamic_shape,
            "timeout": resolve_eval_timeout(self.config.get("verify_timeout")),
            "use_reference_data": reference_mode.use_data,
            "use_reference_inputs": reference_mode.use_inputs,
            "reference_file": reference_mode.reference_file,
        }
        raw_keys = {"needs_binary_io", "tensor_type_name"}
        context.update({
            key: value if key in raw_keys else self.prepare_code_lines(value)
            for key, value in snippets.items()
        })
        return context

    def _verify_impl_artifacts_ready(self, verify_dir: str) -> bool:
        """Return True when generated verify/profile artifacts exist.

        bench_type variants (sol / cann) override the default ``framework
        file + <op>_<dsl>_impl.py`` shape. Per-DSL artifact shape (e.g.
        catlass needs kernel.py + CMakeLists.txt) is delegated to the
        adapter via ``expected_artifacts``.
        """
        # bench_type variants are not per-DSL — handle at this layer.
        impl_file = os.path.join(verify_dir, f"{self.op_name}_{self.dsl}_impl.py")
        if self.bench_type == "sol":
            return (os.path.isfile(os.path.join(verify_dir, "definition.json"))
                    and os.path.isfile(impl_file))
        if self.bench_type == "cann":
            return (os.path.isfile(os.path.join(verify_dir, "proto.yaml"))
                    and os.path.isfile(impl_file))
        artifacts = self.dsl_adapter.expected_artifacts(
            verify_dir, self.op_name, self.framework, self.dsl,
        )
        return all(os.path.isfile(p) for p in artifacts)

    def _create_verify_dir(self, step_counter) -> str:
        """创建验证目录并返回目录路径"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        unique_dir = f"Iteration{self.task_id}_Step{step_counter}_verify"

        target_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _resolve_dyn_factory(self) -> Optional[str]:
        """Name of the ref's multi-shape factory, or None for single-shape.
        Explicit ``framework_factory_names.inputs_factory`` wins; else AST-
        scan the ref source for one of :attr:`_DYN_FACTORY_NAMES`.
        """
        explicit = (self.framework_factory_names or {}).get("inputs_factory")
        if explicit:
            return explicit
        code = self.framework_code or ""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            for n in self._DYN_FACTORY_NAMES:
                if n in code:
                    return n
            return None
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in self._DYN_FACTORY_NAMES):
                return node.name
        return None

    def _get_data_cache_key_id(self) -> str:
        return get_verifier_data_cache_key_id(self.config, self.task_id)

    def _get_baseline_cache_source(self) -> Optional[str]:
        if self.bench_type == "cann":
            try:
                cann_problem_dir = self.config.get("cann_problem_dir")
                if not cann_problem_dir:
                    logger.info(
                        "[%s] config['cann_problem_dir'] 未配置，跳过 CANN baseline cache", self.op_name
                    )
                    return None
                return cann_problem_dir
            except Exception as exc:
                logger.info(
                    '[%s] CANN baseline cache key 构建失败，跳过: %s', self.op_name, exc
                )
                return None
        if self.bench_type != "sol":
            return self.framework_code
        try:
            sol_problem_dir = self.config.get("sol_problem_dir")
            if not sol_problem_dir:
                logger.info(
                    "[%s] config['sol_problem_dir'] 未配置，跳过 SOL baseline cache", self.op_name
                )
                return None
            return build_sol_problem_cache_identity(sol_problem_dir)
        except Exception as exc:
            logger.info(
                '[%s] SOL baseline cache key 构建失败，跳过 baseline cache: %s', self.op_name, exc
            )
            return None

    def _load_cached_torch_reference_payload(self, reference_data: bytes) -> Optional[Any]:
        try:
            import torch

            return torch.load(io.BytesIO(reference_data), map_location="cpu", weights_only=True)
        except TypeError as exc:
            if "weights_only" in str(exc):
                logger.warning(
                    "[%s] 当前 PyTorch 不支持 weights_only=True，"
                    "禁用该 reference data cache 以避免不安全反序列化",
                    self.op_name,
                )
            else:
                logger.warning(
                    '[%s] Verifier Data Cache reference data 无法解析，准备重新生成: %s', self.op_name, exc
                )
            return None
        except Exception as exc:
            logger.warning(
                '[%s] Verifier Data Cache reference data 无法解析，准备重新生成: %s', self.op_name, exc
            )
            return None

    def _clear_managed_reference_data(self, reason: str = "") -> None:
        if not self.config.get("_data_cache_reference_key"):
            return
        self.config.pop("reference_data", None)
        self.config.pop("use_reference_data", None)
        self.config.pop("use_reference_inputs", None)
        self.config.pop("_data_cache_reference_key", None)
        if reason:
            logger.info('[%s] 清理本地 Data Cache 注入的 reference data: %s', self.op_name, reason)

    def _get_cached_baseline_time_us(self, warmup_times: int, run_times: int) -> Optional[float]:
        if self.bench_type not in {"kernelbench", "sol", "cann"}:
            return None

        cache_cfg = self.get_data_cache_config()
        if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
            return None

        cache_key = self.get_baseline_cache_key(warmup_times, run_times)
        if not cache_key:
            return None
        cache_file = get_baseline_cache_file_path(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
        )
        cache_entry = read_baseline_result_from_cache(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
        )
        baseline_time_us = extract_baseline_time_us(cache_entry)
        if baseline_time_us is not None:
            logger.info(
                "[%s] Verifier Data Cache 命中：baseline=%.2f us, "
                "cache_file=%s, cache_key=%s",
                self.op_name,
                baseline_time_us,
                cache_file,
                cache_key,
            )
        elif cache_entry:
            logger.warning(
                "[%s] Verifier Data Cache baseline 结果无效，删除旧缓存: "
                "cache_file=%s, cache_key=%s",
                self.op_name,
                cache_file,
                cache_key,
            )
            delete_baseline_result_from_cache(
                cache_cfg,
                op_name=self.op_name,
                cache_key=cache_key,
            )
        return baseline_time_us

    def _store_baseline_result_in_data_cache(
        self,
        *,
        base_time_us: Optional[float],
        warmup_times: int,
        run_times: int,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        valid_time = (
            base_time_us is not None
            and 0 < base_time_us < float("inf")
        )
        if self.bench_type not in {"kernelbench", "sol", "cann"}:
            return
        if not valid_time:
            return
        cache_config = self.get_data_cache_config()
        if (
            not cache_config.enabled
            or not cache_config.cache_baseline_result
        ):
            return
        cache_key = self.get_baseline_cache_key(
            warmup_times, run_times
        )
        if not cache_key:
            return
        path = write_baseline_result_to_cache(
            cache_config,
            op_name=self.op_name,
            cache_key=cache_key,
            result_data=self._baseline_cache_payload(
                base_time_us,
                warmup_times,
                run_times,
                artifacts,
            ),
            metadata=self._baseline_cache_metadata(),
        )
        if path:
            logger.info(
                "[%s] stored baseline result in cache: %s",
                self.op_name,
                path,
            )

    def _baseline_cache_metadata(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "task_id": self.task_id,
            "cache_key_id": self._get_data_cache_key_id(),
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch,
            "bench_type": self.bench_type,
        }

    def _generate_base_benchmark_code(
        self, dsl_adapter, warmup, runs,
        clear_l2_cache: bool = True,
    ) -> str:
        """Render the framework-model benchmark used by profile scripts."""
        sync_code = _benchmark_sync_code(self.framework, self.backend)
        if self.backend != "ascend":
            return _render_template_file(
                _resource_template_path("benchmark_base_loop.py.j2"),
                warmup=warmup,
                runs=runs,
                sync_code=sync_code,
                backend=self.backend,
            )
        profiler_dsl = getattr(dsl_adapter, "profiler_dsl", "other")
        set_framework_code = ""
        framework_arg = ""
        if self.framework == "mindspore":
            set_framework_code = """        import os
        os.environ["TRITON_BACKEND"] = "mindspore"
        try:
            from op_autoresearch.op.utils.triton_autotune_patch import set_framework
            set_framework("mindspore")
        except ImportError:
            pass
"""
            framework_arg = ', framework="mindspore"'
        return _render_template_file(
            _resource_template_path("benchmark_base_ascend.py.j2"),
            set_framework_code=set_framework_code,
            warmup=warmup,
            runs=runs,
            sync_code=sync_code,
            clear_l2_cache=clear_l2_cache,
            profiler_dsl=profiler_dsl,
            framework_arg=framework_arg,
        )

    def _save_verification_result_to_jsonl(
        self, record: _VerificationRecord
    ) -> None:
        """Append one verification result to the workflow JSONL log."""
        result_path = os.path.join(
            os.path.expanduser(self.log_dir),
            "verification_results.jsonl",
        )
        result_info = {
            "task_name": self.op_name,
            "task_id": self.task_id,
            "step": record.current_step,
            "verify_dir": record.verify_dir,
            "passed": record.passed,
            "error_log": record.logs,
            "timestamp": datetime.now(timezone.utc)
            .astimezone()
            .replace(tzinfo=None)
            .isoformat(),
            "framework": self.framework,
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch,
        }
        if record.total_configs > 0:
            result_info["autotune_configs"] = {
                "total": record.total_configs,
                "passed": record.valid_configs,
            }
        with open(result_path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(result_info, ensure_ascii=False, indent=2)
                + "\n\n"
            )

    async def _verify_configs_separately(
        self,
        request: ConfigVerificationRequest,
    ) -> Tuple[Optional[bool], str, str]:
        result = await verify_autotune_configs(
            self,
            request,
        )
        if result.should_record:
            self._save_verification_result_to_jsonl(
                _VerificationRecord(
                    verify_dir=request.verify_dir,
                    current_step=request.current_step,
                    passed=bool(result.passed),
                    logs=result.logs,
                    total_configs=result.total_configs,
                    valid_configs=result.valid_configs,
                )
            )
        return result.passed, result.logs, result.final_code
