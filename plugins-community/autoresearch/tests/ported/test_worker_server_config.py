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

import pytest
from op_autoresearch.worker.server import get_worker_config


def test_worker_config_requires_explicit_identity(monkeypatch):
    for name in ("WORKER_BACKEND", "WORKER_ARCH", "WORKER_DEVICES"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="WORKER_BACKEND"):
        get_worker_config()


def test_worker_config_rejects_invalid_devices(monkeypatch):
    monkeypatch.setenv("WORKER_BACKEND", "ascend")
    monkeypatch.setenv("WORKER_ARCH", "ascend910b4")
    monkeypatch.setenv("WORKER_DEVICES", "0,bad")

    with pytest.raises(ValueError, match="non-negative integers"):
        get_worker_config()


def test_worker_config_reads_cli_environment(monkeypatch):
    monkeypatch.setenv("WORKER_BACKEND", "ascend")
    monkeypatch.setenv("WORKER_ARCH", "ascend910b4")
    monkeypatch.setenv("WORKER_DEVICES", "2,3")

    assert get_worker_config() == ("ascend", "ascend910b4", [2, 3])
