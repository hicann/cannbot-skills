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

"""PyTorch framework adapter."""

import os
from typing import Any, Optional

import numpy as np

from .base import FrameworkAdapter, load_compare_code


def _torch_runtime():
    """Import torch only when executing framework operations."""
    import torch

    return torch


class FrameworkAdapterTorch(FrameworkAdapter):
    """Adapter for PyTorch framework."""

    def get_import_statements(self) -> str:
        """Return PyTorch import statements."""
        return "import torch\n"

    def get_framework_import(
        self,
        op_name: str,
        is_dynamic_shape: bool,
        inputs_factory_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ) -> str:
        # Template's local name; ref's name may differ (e.g. get_input_groups
        # for NPUKB → aliased to get_inputs_dyn_list). `import X as X` is
        # a Python no-op, so always emit the alias form.
        local = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
        factory = inputs_factory_name or local
        module = module_name or f"{op_name}_torch"
        return (f"from {module} import Model as FrameworkModel, "
                f"get_init_inputs, {factory} as {local}\n")

    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup PyTorch device."""
        torch = _torch_runtime()
        if backend == "ascend":
            if "ascend910" in arch or "ascend950" in arch:
                os.environ['DEVICE_ID'] = str(device_id)
                device = torch.device("npu")
                torch.npu.set_device(device_id)
                return device
            elif "ascend310" in arch:
                os.environ['DEVICE_ID'] = str(device_id)
                return torch.device("cpu")
            else:
                raise ValueError(f"不支持的ascend架构: {arch}")
        raise ValueError(f"仅支持 Ascend，收到 backend={backend}, arch={arch}")

    def process_input(self, x: Any, device: Any) -> Any:
        """Process input and move to device."""
        torch = _torch_runtime()
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.process_input(item, device) for item in x)
        if isinstance(x, (int, float, bool, type(None))):
            return x
        try:
            return x.to(device)
        except (AttributeError, TypeError):
            return x

    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert PyTorch tensor to numpy."""
        torch = _torch_runtime()
        if isinstance(tensor, torch.Tensor):
            return tensor.flatten().detach().cpu().numpy()
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor

    def get_limit(self, dtype: Any) -> float:
        """Get precision rtol for dtype (backward compatibility)."""
        torch = _torch_runtime()
        if dtype == torch.float32:
            return 1.22e-4
        elif dtype == torch.float16:
            return 9.77e-4
        elif dtype == torch.bfloat16:
            return 7.81e-3
        else:
            return 1.22e-4

    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save PyTorch tensor to binary file."""
        torch = _torch_runtime()
        tensor_contiguous = tensor.contiguous().cpu()
        uint8_view = tensor_contiguous.view(torch.uint8)
        with open(bin_path, 'wb') as f:
            f.write(uint8_view.numpy().tobytes())

    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load PyTorch tensor from binary file."""
        torch = _torch_runtime()
        with open(bin_path, 'rb') as f:
            data = f.read()
            uint8_tensor = torch.frombuffer(data, dtype=torch.uint8)
            return uint8_tensor.view(reference_tensor.dtype).reshape(reference_tensor.shape)

    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        torch = _torch_runtime()
        torch.manual_seed(0)
        if backend == "ascend":
            torch.npu.manual_seed(0)

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device."""
        return model.to(device)

    def get_tensor_type(self) -> type:
        """Get PyTorch tensor type."""
        torch = _torch_runtime()
        return torch.Tensor

    def get_tensor_type_name(self) -> str:
        """Get PyTorch tensor type name as string (full path)."""
        return "torch.Tensor"

    def get_device_setup_code(self, backend: str, arch: str, device_id: int) -> str:
        """Get device setup code for PyTorch."""
        if backend == "ascend":
            if "ascend910" in arch or "ascend950" in arch:
                return f"""    import torch_npu
    os.environ['DEVICE_ID'] = str({device_id})
    device = torch.device("npu")
    torch.npu.set_device({device_id})
"""
            elif "ascend310" in arch:
                return f"""    os.environ['DEVICE_ID'] = str({device_id})
    device = torch.device("cpu")
"""
        raise ValueError(f"仅支持 Ascend，收到 backend={backend}, arch={arch}")

    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for PyTorch."""
        return """    def process_input(x):
        \"\"\"Move inputs to the selected torch device.\"\"\"
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(process_input(item) for item in x)
        elif isinstance(x, (int, float, bool, type(None))):
            return x
        else:
            try:
                return x.to(device)
            except (AttributeError, TypeError):
                return x
"""

    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for PyTorch.

        Note: Returns code without indentation, template will handle indentation.
        """
        if backend == "ascend":
            return """torch.manual_seed(0)
torch.npu.manual_seed(0)
"""
        else:
            return """torch.manual_seed(0)
"""

    def get_compare_code(self) -> str:
        """Load the framework-specific generated comparison helpers."""
        return load_compare_code("torch")

    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            compare(fw_out, impl_out, data_type)
'''

    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for PyTorch."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"将PyTorch张量保存为二进制文件\"\"\"
    tensor_contiguous = tensor.contiguous().cpu()
    uint8_view = tensor_contiguous.view(torch.uint8)
    with open(bin_path, 'wb') as f:
        f.write(uint8_view.numpy().tobytes())

"""

    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for PyTorch."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"从二进制文件加载PyTorch张量\"\"\"
    with open(bin_path, 'rb') as f:
        data = f.read()
        uint8_tensor = torch.frombuffer(data, dtype=torch.uint8)
        return uint8_tensor.view(expect_tensor.dtype).reshape(expect_tensor.shape)

"""
