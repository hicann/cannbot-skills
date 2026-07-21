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

"""Integration tests for KernelVerifier using Adapters."""

import importlib.util
import os
import tempfile

import pytest
from op_autoresearch.op.verifier.kernel_verifier import (
    KernelVerifier,
    sync_artifacts_to_directory,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is an accelerator-runtime dependency",
)

_OP_NAME = "test_op"
_FRAMEWORK = "torch"
_DSL = "triton_ascend"
_IMPL_CODE = """
def test_op_triton_ascend_torch(x):
    import torch
    return x * 2
"""


def _make_verifier(tmpdir: str, framework_code: str) -> KernelVerifier:
    return KernelVerifier(
        op_name=_OP_NAME,
        framework=_FRAMEWORK,
        dsl=_DSL,
        backend="ascend",
        arch="ascend910b3",
        framework_code=framework_code,
        impl_func_name="test_op_triton_ascend_torch",
        config={"log_dir": tmpdir},
    )


def test_sync_artifacts_rejects_path_escape(tmp_path):
    verify_dir = tmp_path / "verify"
    verify_dir.mkdir()
    outside = tmp_path / "escaped.json"

    sync_artifacts_to_directory(
        {"ok/result.json": "{}", "../escaped.json": "owned"},
        str(verify_dir),
        "unit",
    )

    assert (verify_dir / "ok" / "result.json").read_text() == "{}"
    assert not outside.exists()


def _assert_generated_project(
    verify_dir: str, op_name: str, framework: str, dsl: str
) -> None:
    assert os.path.exists(
        os.path.join(verify_dir, f"{op_name}_{framework}.py")
    )
    assert os.path.exists(
        os.path.join(verify_dir, f"{op_name}_{dsl}_impl.py")
    )
    verify_script = os.path.join(verify_dir, f"verify_{op_name}.py")
    assert os.path.exists(verify_script)
    with open(verify_script, "r", encoding="utf-8") as file:
        content = file.read()
    assert "import torch" in content
    assert f"from {op_name}_torch import Model as FrameworkModel" in content
    assert "import triton" in content or "from" in content
    assert f"from {op_name}_triton_ascend_impl import" in content
    assert "torch.device(\"npu\")" in content
    assert "def process_input" in content


class TestKernelVerifierWithAdapters:
    """Test KernelVerifier integration with adapters."""

    @requires_torch
    def test_gen_verify_project_torch_triton_ascend(self):
        """Test generating verify project for torch + triton_ascend."""
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import torch
        return torch.tensor([1.0, 2.0, 3.0])

def get_inputs():
    import torch
    return [torch.tensor([1.0, 2.0, 3.0])]
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = _make_verifier(tmpdir, framework_code)

            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)

            # Generate verify project
            verifier.gen_verify_project(_IMPL_CODE, verify_dir, device_id=0)

            _assert_generated_project(
                verify_dir, _OP_NAME, _FRAMEWORK, _DSL
            )

    @requires_torch
    def test_dynamic_shape_detection(self):
        """Test dynamic shape detection."""
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import torch
        return torch.tensor([1.0, 2.0, 3.0])

def get_inputs_dyn_list():
    import torch
    return [[torch.tensor([1.0])], [torch.tensor([2.0])]]
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = _make_verifier(tmpdir, framework_code)

            # Check dynamic shape detection
            assert verifier.detect_dynamic_shape() is True

            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)

            # Generate verify project
            verifier.gen_verify_project(_IMPL_CODE, verify_dir, device_id=0)

            # Check verify script uses dynamic shape
            verify_script = os.path.join(verify_dir, f"verify_{_OP_NAME}.py")
            with open(verify_script, "r", encoding="utf-8") as f:
                content = f.read()
                assert "get_inputs_dyn_list" in content
