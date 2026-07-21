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

"""Regression coverage for dependency-free worker diagnostics."""

from op_autoresearch.cli.service.diagnostics import Finding, render_findings


def test_render_findings_uses_plain_stderr(capsys):
    render_findings(
        [
            Finding("fatal", "ssh", "timed out", "check VPN"),
            Finding("ok", "remote :19112", "free", ""),
        ],
        "worker stopped",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Remote diagnostics:" in captured.err
    assert "[FATAL] ssh: timed out | check VPN" in captured.err
    assert "[OK] remote :19112: free" in captured.err
    assert "daemon log tail:\nworker stopped" in captured.err


def test_render_findings_ignores_missing_log_placeholder(capsys):
    render_findings([], "(no log file)")

    captured = capsys.readouterr()
    assert captured.err == "Remote diagnostics:\n"
