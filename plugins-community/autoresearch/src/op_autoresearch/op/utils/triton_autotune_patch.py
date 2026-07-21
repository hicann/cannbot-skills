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

import logging
import os

from op_autoresearch.utils.console import emit

logger = logging.getLogger(__name__)

_collected_config_timings = {}
_current_framework = "torch"


def set_framework(framework: str):
    """设置当前框架类型，影响 op_autoresearch_restore_copy / benchmarker 的行为。"""
    global _current_framework
    _current_framework = framework
    if framework == "mindspore":
        os.environ["TRITON_BACKEND"] = "mindspore"


def get_framework() -> str:
    return _current_framework

# ============================================================================
# OP_AUTORESEARCH_restore_copy Triton kernel
# 参考 l2_cache_clear.py 的设计：使用带 OP_AUTORESEARCH_ 前缀的专用 kernel，
# 便于在 profiler 的 op_statistic.csv 中按名字精确过滤。
# ============================================================================


OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME = "op_autoresearch_restore_copy_kernel"

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:
    @triton.jit
    def op_autoresearch_restore_copy_kernel(
        dst_ptr, src_ptr, n_elements,
        block_size: tl.constexpr, core_num: tl.constexpr,
    ):
        """
        restore_value 专用 copy kernel。

        kernel 名称带 OP_AUTORESEARCH_ 前缀，在 profiler 中显示为 OP_AUTORESEARCH_restore_copy，
        可精确过滤，不会误删用户代码中的 TensorMove 等同名操作。
        """
        pid = tl.program_id(0)
        num_blocks = tl.cdiv(n_elements, block_size)
        for block_idx in range(pid, num_blocks, core_num):
            block_start = block_idx * block_size
            offsets = block_start + tl.arange(0, block_size)
            mask = offsets < n_elements
            data = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, data, mask=mask)


def _get_core_nums(vec_default=40, cube_default=20):
    if get_framework() == "mindspore":
        import mindspore as ms
        limits = ms.runtime.get_device_limit(0)
        vec = limits.get("vector_core_num", vec_default)
        cube = limits.get("cube_core_num", cube_default)
        return (vec, cube)
    vec, cube = vec_default, cube_default
    try:
        import torch
        device = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        vec = properties.get("num_vectorcore", vec_default)
        cube = properties.get("num_aicore", cube_default)
    except Exception:
        logger.debug("Could not query Triton core counts", exc_info=True)
    return (vec, cube)


def op_autoresearch_restore_copy_torch(dst, src):
    """用 OP_AUTORESEARCH_restore_copy kernel 执行 tensor copy（PyTorch 版）。"""
    import torch
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num, _ = _get_core_nums()
    block_size = 1024
    grid = (core_num,)
    op_autoresearch_restore_copy_kernel[grid](
        dst_flat, src_flat, n, block_size=block_size, core_num=core_num
    )
    torch.npu.synchronize()


def op_autoresearch_restore_copy_mindspore(dst, src):
    """用 OP_AUTORESEARCH_restore_copy kernel 执行 tensor copy（MindSpore 版）。"""
    import mindspore as ms
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num, _ = _get_core_nums()
    block_size = 1024
    grid = (core_num,)
    op_autoresearch_restore_copy_kernel[grid](
        dst_flat, src_flat, n, block_size=block_size, core_num=core_num
    )
    ms.runtime.synchronize()


def op_autoresearch_restore_copy(dst, src):
    """用 OP_AUTORESEARCH_restore_copy kernel 执行 tensor copy，替代 tensor.copy_()。"""
    if get_framework() == "mindspore":
        op_autoresearch_restore_copy_mindspore(dst, src)
    else:
        op_autoresearch_restore_copy_torch(dst, src)


def _restore_saved_tensors(saved, args):
    """Restore saved output tensors back to the live kernel arguments."""
    for idx, saved_val in saved.items():
        op_autoresearch_restore_copy(args[idx], saved_val)


def _wrap_kernel_call_with_restore(kernel_call, restore_info):
    """Wrap benchmark calls with Triton-like pre/post restore semantics."""
    if restore_info is None:
        return kernel_call

    saved = restore_info['saved']
    args = restore_info['args']

    def wrapped_call():
        _restore_saved_tensors(saved, args)
        try:
            return kernel_call()
        finally:
            # Leave every benchmark iteration with the original output state
            # so a later config cannot inherit stale values from an earlier one.
            _restore_saved_tensors(saved, args)

    return wrapped_call


# ============================================================================
# _bench patch: 禁用原生 restore_value 的 copy_()，
# 让 kernel_call 只包含纯 kernel，restore 交给 benchmarker 用命名 kernel 做。
# ============================================================================

_restore_info = None


def _patch_autotuner_bench(autotuner_module):
    """Patch Autotuner._bench，在 restore_value 场景下接管 pre_hook。"""
    original_bench = getattr(autotuner_module.Autotuner, '_bench', None)
    if original_bench is None:
        return
    if getattr(original_bench, 'op_autoresearch_bench_patched', False):
        return

    def _noop(*_args, **_kwargs):
        return None

    def patched_bench(self, *args, config, **meta):
        global _restore_info

        if not (_TRITON_AVAILABLE and hasattr(self, 'restore_value') and self.restore_value):
            _restore_info = None
            return original_bench(self, *args, config=config, **meta)

        saved = {}
        for name in self.restore_value:
            idx = self.fn.arg_names.index(name)
            saved[idx] = args[idx].clone()
        _restore_info = {'saved': saved, 'args': list(args)}

        orig_rv = self.restore_value
        orig_ph = getattr(self, 'pre_hook', None)
        orig_posth = getattr(self, 'post_hook', None)
        self.restore_value = None
        self.pre_hook = _noop
        self.post_hook = _noop

        try:
            result = original_bench(self, *args, config=config, **meta)
        finally:
            self.restore_value = orig_rv
            self.pre_hook = orig_ph
            self.post_hook = orig_posth
            _restore_info = None

        return result

    patched_bench.op_autoresearch_bench_patched = True
    # Triton exposes no public setter for its benchmark callback.
    setattr(autotuner_module.Autotuner, '_bench', patched_bench)


# ============================================================================
# 需要过滤的底层实现参数
# ============================================================================

_FILTERED_CONFIG_PARAMS = {
    'num_warps',
    'num_ctas',
    'num_stages',
    'num_buffers_warp_spec',
    'num_consumer_groups',
    'reg_dec_producer',
    'reg_inc_consumer',
    'maxnreg'
}


def _filter_config_string(config_str: str) -> str:
    """过滤配置字符串，移除底层实现参数"""
    params = []
    for param in config_str.split(','):
        param = param.strip()
        if not param:
            continue
        if ':' in param:
            param_name = param.split(':', 1)[0].strip()
        elif '=' in param:
            param_name = param.split('=', 1)[0].strip()
        else:
            params.append(param)
            continue
        if param_name not in _FILTERED_CONFIG_PARAMS:
            params.append(param)
    return ', '.join(params)


def _tuner_function_name(tuner) -> str:
    for attribute in ("base_fn", "fn"):
        function = getattr(tuner, attribute, None)
        name = getattr(function, "__name__", None)
        if name:
            return name
    return "unknown_function"


def _timing_row(tuner, config, timing, rank: int):
    try:
        timing_value = timing[0] if isinstance(timing, list) else timing
        return {
            "config": _filter_config_string(str(config)),
            "timing_us": float(timing_value),
            "is_best": config == tuner.best_config,
            "rank": rank,
        }
    except (TypeError, ValueError, AttributeError):
        return None


def _sorted_config_timings(tuner):
    timings = getattr(tuner, "configs_timings", None)
    if not isinstance(timings, dict) or not timings:
        return []
    try:
        return sorted(timings.items(), key=lambda item: item[1])
    except (TypeError, ValueError, AttributeError):
        return []


def _print_config_timings(function_name: str, tuner, timings) -> None:
    emit(f"All config timings for {function_name}:")
    for rank, (config, timing) in enumerate(timings, 1):
        row = _timing_row(tuner, config, timing, rank)
        if row is None:
            continue
        status = " (BEST)" if row["is_best"] else ""
        emit(
            f"  Config {rank}: {row['config']} -> "
            f"{row['timing_us']:.4f}us{status}"
        )


def _process_config_timings(tuner) -> None:
    if not hasattr(tuner, "best_config"):
        return
    timings = _sorted_config_timings(tuner)
    rows = []
    for rank, (config, timing) in enumerate(timings, 1):
        row = _timing_row(tuner, config, timing, rank)
        if row is not None:
            rows.append(row)
    if not rows:
        return
    function_name = _tuner_function_name(tuner)
    if function_name in _collected_config_timings:
        return
    _collected_config_timings[function_name] = rows
    if os.getenv("TRITON_PRINT_AUTOTUNING") == "1":
        _print_config_timings(function_name, tuner, timings)


def patch_triton_autotuner():
    """动态补丁 triton autotuner，添加配置信息收集 + _bench restore_value 接管。"""
    try:
        import triton.runtime.autotuner as autotuner_module
    except ImportError:
        return True

    try:
        import triton.runtime.autotiling_tuner as autotiling_module
    except ImportError:
        autotiling_module = None

    if not hasattr(autotuner_module, 'Autotuner'):
        return True

    original_autotuner_run = getattr(autotuner_module.Autotuner, 'run', None)
    if original_autotuner_run is None:
        return True
    if getattr(original_autotuner_run, 'op_autoresearch_run_patched', False):
        return True

    original_autotiling_run = None
    if autotiling_module and hasattr(autotiling_module, 'AutoTilingTuner'):
        original_autotiling_run = getattr(autotiling_module.AutoTilingTuner, 'run', None)

    # Patch _bench 接管 restore_value
    _patch_autotuner_bench(autotuner_module)

    def patched_autotuner_run(self, *args, **kwargs):
        result = original_autotuner_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            logger.debug("Could not collect autotuner timings", exc_info=True)
        return result

    def patched_autotiling_run(self, *args, **kwargs):
        result = original_autotiling_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            logger.debug("Could not collect autotiling timings", exc_info=True)
        return result

    try:
        patched_autotuner_run.op_autoresearch_run_patched = True
        autotuner_module.Autotuner.run = patched_autotuner_run
    except (AttributeError, TypeError) as exc:
        logger.debug("unable to patch Triton Autotuner.run: %s", exc)

    if original_autotiling_run is not None:
        try:
            patched_autotiling_run.op_autoresearch_run_patched = True
            autotiling_module.AutoTilingTuner.run = patched_autotiling_run
        except (AttributeError, TypeError) as exc:
            logger.debug("unable to patch Triton AutoTilingTuner.run: %s", exc)

    return True


def get_collected_config_timings():
    global _collected_config_timings
    return _collected_config_timings.copy()


def clear_collected_config_timings():
    global _collected_config_timings
    _collected_config_timings = {}


def patch_driver_benchmarker():
    """补丁 driver.active.get_benchmarker()，让 autotune 使用 profiler_npu。

    当 _restore_info 不为空时（即 _bench 禁用了原生 restore_value），
    benchmarker 自动用 OP_AUTORESEARCH_restore_copy kernel 包装 kernel_call，
    profiler 按 kernel 名字精确过滤，不会误删用户的 TensorMove 操作。
    """
    try:
        from triton.runtime import driver

        if hasattr(driver.active.get_benchmarker, 'op_autoresearch_patched'):
            return True

        original_get_benchmarker = driver.active.get_benchmarker

        def patched_get_benchmarker():
            def custom_benchmarker(kernel_call, quantiles=(0.5, 0.2, 0.8)):
                fn_to_profile = _wrap_kernel_call_with_restore(kernel_call, _restore_info)

                try:
                    from op_autoresearch.op.verifier.profiler import profiler_npu

                    time_us = profiler_npu(
                        fn_to_profile,
                        warmup=5,
                        active=30,
                        suppress_warnings=True,
                        clear_l2_cache=True,
                        dsl="triton_ascend",
                        filter_restore_copy=(_restore_info is not None),
                        framework=get_framework(),
                    )
                    return [time_us] * 3

                except ImportError:
                    original_benchmarker = original_get_benchmarker()
                    return original_benchmarker(fn_to_profile, quantiles)

            return custom_benchmarker

        driver.active.get_benchmarker = patched_get_benchmarker
        driver.active.get_benchmarker.op_autoresearch_patched = True
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.warning("Failed to patch driver benchmarker: %s", e)
        return False


def apply_triton_patches():
    """应用所有triton补丁"""
    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()
    return success1 or success2


if __name__ != "__main__":
    apply_triton_patches()

if __name__ == "__main__":
    emit("Testing Triton patches...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()

    if success1:
        emit("Autotuner patch applied successfully!")
    if success2:
        emit("Driver benchmarker patch applied successfully!")

    if not any([success1, success2]):
        emit("Failed to apply patches")
