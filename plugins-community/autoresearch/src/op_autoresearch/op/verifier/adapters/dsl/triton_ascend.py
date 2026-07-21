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

"""Triton Ascend DSL adapter - 支持 ModelNew (KernelBench) 格式."""

from .base import (
    BenchmarkSpec,
    DSLAdapter,
    render_code_template,
)


def _runtime_setup_code(framework: str) -> str:
    """Return the framework selection and idempotent Triton patch setup."""
    blocks = []
    if framework == "mindspore":
        blocks.append(
            '''import os
os.environ["TRITON_BACKEND"] = "mindspore"
try:
    from op_autoresearch.op.utils.triton_autotune_patch import set_framework
    set_framework("mindspore")
except ImportError:
    pass
'''
        )
    blocks.append(
        '''try:
    from op_autoresearch.op.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
'''
    )
    return "".join(blocks)


class DSLAdapterTritonAscend(DSLAdapter):
    """Adapter for Triton Ascend DSL."""

    profile_via_python_script = True
    impl_func_name_template = "ModelNew"
    model_wrapper_eval_mode = False
    profiler_dsl = "triton_ascend"
    supports_autotune_configs = True
    emits_autotune_artifacts = True

    def get_import_statements(self, framework: str) -> str:
        """Return Triton Ascend import statements."""
        code = _runtime_setup_code(framework)
        if framework == "numpy":
            code += "import numpy as np\n"
        code += "import triton\nimport triton.language as tl\n"
        return code

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.
        
        统一使用 ModelNew 类格式（KernelBench 风格）。
        """
        return f"from {op_name}_triton_ascend_impl import ModelNew\n"

    def call_impl(self, inputs: str) -> str:
        """Return code string to call Triton Ascend implementation function.
        
        调用已经实例化好的 impl_model（可以多次调用）。
        """
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, spec: BenchmarkSpec) -> str:
        """Return generated code that benchmarks a Triton Ascend implementation."""
        set_framework_code = ""
        framework_arg = ""
        if spec.framework == "mindspore":
            set_framework_code = """        import os
        os.environ["TRITON_BACKEND"] = "mindspore"
        try:
            from op_autoresearch.op.utils.triton_autotune_patch import set_framework
            set_framework("mindspore")
        except ImportError:
            pass
"""
            framework_arg = ', framework="mindspore"'
        return render_code_template(
            "triton_ascend_benchmark.py.j2",
            set_framework_code=set_framework_code,
            inputs=spec.inputs,
            case_idx=spec.case_idx,
            op_name=spec.op_name,
            warmup=spec.warmup,
            runs=spec.runs,
            clear_l2_cache=spec.clear_l2_cache,
            framework_arg=framework_arg,
            backend_repr=repr(spec.backend),
        )

    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code for triton_ascend."""
        return _runtime_setup_code(framework)
