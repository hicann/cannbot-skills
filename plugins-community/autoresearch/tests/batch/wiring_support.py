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

"""Shared imports for standalone batch wiring tests."""

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

for project_path in (REPO / "src", REPO / "scripts", REPO / "scripts" / "batch"):
    sys.path.insert(0, str(project_path))

emit = importlib.import_module("op_autoresearch.utils.console").emit
phase_machine = importlib.import_module("phase_machine")
R = importlib.import_module("run")
task_handle = importlib.import_module("task_handle")
