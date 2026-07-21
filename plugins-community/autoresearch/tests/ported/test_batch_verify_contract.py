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

"""Contracts shared by batch Tier 2 and the workspace eval bridge."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]


def _load_batch_verify():
    path = REPO / "scripts" / "batch" / "verify.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("batch_verify_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inspect_reference(
    monkeypatch,
    tmp_path,
    source_lines,
    runtime_modules,
):
    batch_verify = _load_batch_verify()
    reference = tmp_path / "reference.py"
    reference.write_text("\n".join(source_lines), encoding="utf-8")
    monkeypatch.setattr(
        batch_verify,
        "_worker_runtime_modules",
        lambda: frozenset(runtime_modules),
    )
    return batch_verify.inspect_tier1(
        reference,
        batch_verify.REF_REQUIRED,
    )


def test_tier2_calls_eval_bridge_with_request_object(monkeypatch, tmp_path):
    batch_verify = _load_batch_verify()
    config = object()
    captured = {}

    monkeypatch.setattr(
        batch_verify, "_scaffold_tier2_task", lambda _setup: tmp_path
    )

    from task_config import loader
    from utils import eval_bridge

    monkeypatch.setattr(loader, "load_task_config", lambda _path: config)

    def fake_eval(request):
        captured["request"] = request
        return {"outcome": "ok"}

    monkeypatch.setattr(eval_bridge, "eval_kernel", fake_eval)
    setup = SimpleNamespace(
        device_ids=(2,),
        request=SimpleNamespace(worker_url="127.0.0.1:19111"),
    )

    assert batch_verify.execute_tier2(setup) == {"outcome": "ok"}
    request = captured.get("request")
    assert request is not None
    assert isinstance(request, eval_bridge.EvalRequest)
    assert request.task_dir == str(tmp_path)
    assert request.config is config
    assert request.device_id == [2]
    assert request.worker_url == "127.0.0.1:19111"
    assert request.current_step == 0
    assert request.verify_only is True


def test_ascendc_example_declares_bisheng_stub_with_cpp_linkage():
    source = (
        REPO
        / "ar_examples"
        / "ascendc"
        / "add_custom"
        / "ascendc_op"
        / "op_extension"
        / "add_custom_torch.cpp"
    ).read_text(encoding="utf-8")

    assert "void add_custom_kernel(uint32_t blockDim" in source
    assert 'extern "C" void add_custom_kernel(uint32_t blockDim' not in source


def test_tier1_defers_worker_runtime_import_and_checks_static_exports(
    monkeypatch, tmp_path
):
    result = _inspect_reference(
        monkeypatch,
        tmp_path,
        (
            "import missing_npu_runtime",
            "class Model: pass",
            "def get_inputs(): return []",
            "def get_init_inputs(): return []",
        ),
        {"missing_npu_runtime"},
    )

    assert result["compile"] == "PASS"
    assert result["import"] == "SKIP"
    assert result["exports"] == "PASS"
    assert result["missing"] == []


def test_tier1_does_not_defer_unknown_missing_import(monkeypatch, tmp_path):
    result = _inspect_reference(
        monkeypatch,
        tmp_path,
        (
            "import misspelled_dependency",
            "class Model: pass",
            "def get_inputs(): return []",
            "def get_init_inputs(): return []",
        ),
        {"torch"},
    )

    assert result["compile"] == "PASS"
    assert result["import"] == "FAIL"
    assert result["exports"] == "skip"
    assert "misspelled_dependency" in result["msg"]


def test_tier1_static_fallback_still_rejects_missing_exports(
    monkeypatch, tmp_path
):
    result = _inspect_reference(
        monkeypatch,
        tmp_path,
        (
            "import missing_npu_runtime",
            "class Model: pass",
            "def get_init_inputs(): return []",
        ),
        {"missing_npu_runtime"},
    )

    assert result["import"] == "SKIP"
    assert result["exports"] == "FAIL"
    assert result["missing"] == ["get_inputs or get_input_groups"]


def test_worker_runtime_policy_is_shared_with_batch_tier1():
    from op_autoresearch.op.utils.code_checker import CodeChecker

    triton_modules = CodeChecker.worker_runtime_modules("TRITON_ASCEND")
    ascendc_modules = CodeChecker.worker_runtime_modules("ascendc")

    assert {"torch", "torch_npu", "triton"} <= triton_modules
    assert "torch" in ascendc_modules
    assert "triton" not in ascendc_modules


def test_codegen_adapters_import_without_local_torch(monkeypatch):
    from op_autoresearch.op.verifier.adapters import factory

    framework_module = (
        "op_autoresearch.op.verifier.adapters.framework.torch"
    )
    backend_module = "op_autoresearch.op.verifier.adapters.backend.ascend"
    monkeypatch.delitem(sys.modules, framework_module, raising=False)
    monkeypatch.delitem(sys.modules, backend_module, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    framework = factory.get_framework_adapter("torch")
    backend = factory.get_backend_adapter("ascend")

    assert framework.get_import_statements() == "import torch\n"
    assert framework.get_tensor_type_name() == "torch.Tensor"
    monkeypatch.setenv("DEVICE_ID", "")
    backend.setup_environment(2, "ascend910b3")
    with pytest.raises(ModuleNotFoundError):
        framework.get_tensor_type()


def test_cann_correctness_codegen_imports_without_local_torch(monkeypatch):
    package_name = "op_autoresearch.op.cann_correctness"
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(package_name + "."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    package = importlib.import_module(package_name)

    assert package.CORE_PY_PATH.endswith("core.py")
    assert package.compare_snippet()
    assert f"{package_name}.core" not in sys.modules
