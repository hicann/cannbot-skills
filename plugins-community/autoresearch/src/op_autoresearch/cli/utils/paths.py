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

from pathlib import Path

from op_autoresearch import DEFAULT_LOG_DIR


def get_log_dir() -> Path:
    path = Path(DEFAULT_LOG_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_process_log_dir() -> Path:
    path = get_log_dir() / "processes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_package_dir() -> Path:
    import op_autoresearch
    return Path(op_autoresearch.__file__).resolve().parent

