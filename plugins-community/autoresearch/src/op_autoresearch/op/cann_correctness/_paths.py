# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Paths and staging helpers shared during package initialization."""

import os
import shutil

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_PY_PATH = os.path.join(_PKG_DIR, "core.py")
TEMPLATES_DIR = os.path.join(_PKG_DIR, "templates")


def stage_core_into(dest_dir: str) -> str:
    """Copy core.py into the destination as cann_correctness.py."""
    dst = os.path.join(dest_dir, "cann_correctness.py")
    shutil.copy2(CORE_PY_PATH, dst)
    return dst
