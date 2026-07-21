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

importlib.import_module("torch_npu")


class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def get_inputs():
    device = "npu"
    x = torch.randn(128, 128, device=device, dtype=torch.float16)
    y = torch.randn(128, 128, device=device, dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []
