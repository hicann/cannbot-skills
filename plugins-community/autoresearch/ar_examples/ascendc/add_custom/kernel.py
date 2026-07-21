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
        self._loaded = False

    def forward(self, x, y):
        self._load()
        return torch.ops.npu.add_custom(x, y)

    def _load(self):
        if self._loaded:
            return
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        preferred = build_dir / "libadd_custom_ops.so"
        so_path = preferred if preferred.is_file() else None
        if so_path is None:
            matches = sorted(build_dir.rglob("*.so"))
            if matches:
                so_path = matches[0]
        if so_path is None:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_path))
        self._loaded = True
