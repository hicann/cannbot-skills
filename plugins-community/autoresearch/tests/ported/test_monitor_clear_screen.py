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

"""clear_screen 在子进程缺失时不得让刷新循环崩溃。"""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_monitor_test",
    ROOT / "scripts" / "batch" / "monitor.py",
)
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_clear_screen_survives_missing_executable(monkeypatch, capsys):
    def raise_file_not_found(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(MONITOR.subprocess, "run", raise_file_not_found)
    MONITOR.clear_screen()  # 不得抛异常
    assert "\x1b[2J\x1b[H" in capsys.readouterr().out


def test_clear_screen_normal_path(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(MONITOR.subprocess, "run", fake_run)
    MONITOR.clear_screen()
    assert calls == [["clear"]]  # POSIX 正常路径行为不变
