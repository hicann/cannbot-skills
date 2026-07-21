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

"""MindSpore framework adapter."""

import os
from typing import Any, Optional

import mindspore as ms
import numpy as np
from mindspore.common import np_dtype

from op_autoresearch.op.utils.config_utils import check_backend_arch

from .base import FrameworkAdapter, load_compare_code


class FrameworkAdapterMindSpore(FrameworkAdapter):
    """Adapter for MindSpore framework."""

    def get_import_statements(self) -> str:
        """Return MindSpore import statements."""
        return "import mindspore as ms\nfrom mindspore.common import np_dtype\n"

    def get_framework_import(
        self,
        op_name: str,
        is_dynamic_shape: bool,
        inputs_factory_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ) -> str:
        local = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
        factory = inputs_factory_name or local
        module = module_name or f"{op_name}_mindspore"
        return (f"from {module} import Model as FrameworkModel, "
                f"get_init_inputs, {factory} as {local}\n")

    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup MindSpore device."""
        os.environ['DEVICE_ID'] = str(device_id)
        if backend == "ascend":
            check_backend_arch(backend, arch)
            return "Ascend"
        elif backend == "cpu":
            return "CPU"
        else:
            raise ValueError(f"MindSpore不支持的后端: {backend}")

    def process_input(self, x: Any, device: Any) -> Any:
        """Process input (MindSpore doesn't need device movement)."""
        return x

    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert MindSpore tensor to numpy."""
        if isinstance(tensor, ms.Tensor):
            return tensor.flatten().asnumpy()
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor

    def get_limit(self, dtype: Any) -> float:
        """Get precision rtol for dtype (backward compatibility)."""
        if dtype == ms.float32:
            return 1.22e-4
        elif dtype == ms.float16:
            return 9.77e-4
        elif dtype == ms.bfloat16:
            return 7.81e-3
        else:
            return 1.22e-4

    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save MindSpore tensor to binary file."""
        tensor_np = tensor.asnumpy()
        uint8_view = tensor_np.view(np.uint8)
        with open(bin_path, 'wb') as f:
            f.write(uint8_view.tobytes())

    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load MindSpore tensor from binary file."""
        with open(bin_path, 'rb') as f:
            data = f.read()
            uint8_array = np.frombuffer(data, dtype=np.uint8)
            numpy_dtype = self.get_dtype_mapping().get(reference_tensor.dtype)
            if numpy_dtype is None:
                raise ValueError(f"不支持的数据类型: {reference_tensor.dtype}")
            numpy_tensor = uint8_array.view(numpy_dtype).reshape(reference_tensor.shape)
            return ms.Tensor(numpy_tensor, dtype=reference_tensor.dtype)

    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        ms.manual_seed(0)

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device (MindSpore doesn't need explicit move)."""
        return model

    def get_tensor_type(self) -> type:
        """Get MindSpore tensor type."""
        return ms.Tensor

    def get_tensor_type_name(self) -> str:
        """Get MindSpore tensor type name as string (full path)."""
        return "ms.Tensor"

    def get_dtype_mapping(self) -> dict:
        """Get MindSpore to NumPy dtype mapping."""
        return {
            ms.float32: np.float32,
            ms.float16: np.float16,
            ms.bfloat16: np_dtype.bfloat16,
            ms.int8: np.int8,
            ms.int16: np.int16,
            ms.int32: np.int32,
            ms.int64: np.int64,
            ms.uint8: np.uint8,
            ms.uint16: np.uint16,
            ms.uint32: np.uint32,
            ms.uint64: np.uint64,
            ms.bool_: np.bool_,
        }

    def get_device_setup_code(self, backend: str, arch: str, device_id: int) -> str:
        """Get device setup code for MindSpore."""
        code = f"""    os.environ['DEVICE_ID'] = str({device_id})
"""
        if backend == "ascend":
            check_backend_arch(backend, arch)
            code += """    device = "Ascend"
"""
        elif backend == "cpu":
            code += """    device = "CPU"
"""
        return code

    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for MindSpore."""
        return """    def process_input(x):
        \"\"\"处理输入数据\"\"\"
        return x
"""

    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for MindSpore.

        Note: Returns code without indentation, template will handle indentation.
        """
        return """ms.manual_seed(0)
"""

    def get_compare_code(self) -> str:
        """Load the framework-specific generated comparison helpers."""
        return load_compare_code("mindspore")

    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            compare(fw_out, impl_out, data_type)
'''

    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for MindSpore."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"将MindSpore张量保存为二进制文件\"\"\"
    tensor_np = tensor.asnumpy()
    uint8_view = tensor_np.view(np.uint8)
    with open(bin_path, 'wb') as f:
        f.write(uint8_view.tobytes())

"""

    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for MindSpore."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"从二进制文件加载MindSpore张量\"\"\"
    with open(bin_path, 'rb') as f:
        data = f.read()
        uint8_array = np.frombuffer(data, dtype=np.uint8)
        numpy_dtype = MS_TO_NP_DTYPE_MAP.get(expect_tensor.dtype)
        if numpy_dtype is None:
            raise ValueError(f"不支持的数据类型: {expect_tensor.dtype}")
        numpy_tensor = uint8_array.view(numpy_dtype).reshape(expect_tensor.shape)
        return ms.Tensor(numpy_tensor, dtype=expect_tensor.dtype)

"""
