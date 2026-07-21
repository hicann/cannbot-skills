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

import importlib

import torch
import triton
import triton.language as tl

importlib.import_module("torch_npu")


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr,
                       block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n_elements = out.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel[grid](x, y, out, n_elements,
                                 block_size=block_size)
        return out
