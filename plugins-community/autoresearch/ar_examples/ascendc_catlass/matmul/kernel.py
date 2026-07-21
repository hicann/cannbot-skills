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

from __future__ import annotations

import importlib
from pathlib import Path

import torch

importlib.import_module("torch_npu")


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        library = Path(__file__).with_name("catlass_op") / "build" / "libcatlass.so"
        if not library.is_file():
            raise RuntimeError(f"CATLASS library not found: {library}")
        torch.ops.load_library(str(library))

    def forward(self, x, y):
        return torch.ops.catlass.basic_matmul(x, y)
