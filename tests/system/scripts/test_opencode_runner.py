#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
OpencodeRunner 测试用例

测试功能：
1. 基本运行功能
2. 流式输出功能
3. Session 文件管理
4. 模型指定
5. 错误处理
"""

import contextlib
import importlib.util
import io
import os
import signal
import sys
import threading
import time
import unittest

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from opencode_runner import OpencodeRunner, OpencodeResult


class TestOpencodeRunnerInit:
    """测试 OpencodeRunner 初始化"""

    @staticmethod
    def test_default_init():
        runner = OpencodeRunner()
        assert runner.model is None
        assert runner.keep_session is False
        assert runner.timeout == 300
        assert runner.verbose is False

    @staticmethod
    def test_init_with_model():
        runner = OpencodeRunner(model="gpt-4")
        assert runner.model == "gpt-4"

    @staticmethod
    def test_init_with_keep_session():
        runner = OpencodeRunner(keep_session=True)
        assert runner.keep_session is True

    @staticmethod
    def test_init_with_session_dir():
        temp_dir = tempfile.mkdtemp()
        runner = OpencodeRunner(session_dir=temp_dir)
        assert runner.session_dir == Path(temp_dir)
        shutil.rmtree(temp_dir)

    @staticmethod
    def test_init_with_timeout():
        runner = OpencodeRunner(timeout=600)
        assert runner.timeout == 600

    @staticmethod
    def test_init_with_verbose():
        runner = OpencodeRunner(verbose=True)
        assert runner.verbose is True

    @staticmethod
    def test_init_with_workdir():
        temp_dir = tempfile.mkdtemp()
        runner = OpencodeRunner(workdir=temp_dir)
        assert runner.workdir == Path(temp_dir)
        shutil.rmtree(temp_dir)


class TestOpencodeRunnerSession:
    """测试 Session 文件管理"""

    @staticmethod
    def test_cleanup_session(tmp_path):
        session_file = tmp_path / "test_session.json"
        session_file.write_text('{"session_id": "test"}')
        runner = OpencodeRunner(keep_session=True, session_dir=str(tmp_path))
        runner.cleanup_session(str(session_file))
        assert not session_file.exists()

    @staticmethod
    def test_cleanup_all_sessions(tmp_path):
        for i in range(3):
            (tmp_path / f"session_{i}.json").write_text("{}")
        runner = OpencodeRunner(keep_session=True, session_dir=str(tmp_path))
        sessions = runner.list_sessions()
        assert len(sessions) == 3
        runner.cleanup_all_sessions()
        assert len(runner.list_sessions()) == 0

    @staticmethod
    def test_list_sessions(tmp_path):
        for i in range(2):
            (tmp_path / f"session_{i}.json").write_text("{}")
        runner = OpencodeRunner(keep_session=True, session_dir=str(tmp_path))
        sessions = runner.list_sessions()
        assert len(sessions) >= 2
        runner.cleanup_all_sessions()


class TestOpencodeRunnerCommand:
    """测试命令构建"""

    @staticmethod
    def test_build_basic_command():
        runner = OpencodeRunner()
        cmd = runner.build_command("test prompt")
        assert "opencode" in cmd
        assert "run" in cmd
        assert "--format" in cmd
        assert "json" in cmd
        assert "test prompt" in cmd

    @staticmethod
    def test_build_command_with_model():
        runner = OpencodeRunner(model="gpt-4")
        cmd = runner.build_command("test prompt")
        assert "--model" in cmd
        assert "gpt-4" in cmd

    @staticmethod
    def test_build_command_with_session():
        runner = OpencodeRunner()
        cmd = runner.build_command("test prompt", resume_session_id="session.json")
        assert "--session" in cmd
        assert "session.json" in cmd

    @staticmethod
    def test_build_command_with_skill():
        runner = OpencodeRunner()
        cmd = runner.build_command("test prompt", skill="test-skill")
        assert "--skill" in cmd
        assert "test-skill" in cmd

    @staticmethod
    def test_build_command_with_additional_args():
        runner = OpencodeRunner()
        cmd = runner.build_command(
            "test prompt", additional_args=["--verbose", "--debug"]
        )
        assert "--verbose" in cmd
        assert "--debug" in cmd


class TestOpencodeRunnerResult:
    """测试运行结果"""

    @patch("subprocess.run")
    def test_successful_run(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0, stdout='{"result": "success"}', stderr=""
        )

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is True
        assert result.output == '{"result": "success"'
        assert result.error == ""

    @patch("subprocess.run")
    def test_failed_run(self, mock_run):
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error occurred")

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is False
        assert "Error occurred" in result.error

    @patch("subprocess.run")
    def test_timeout_run(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="opencode", timeout=300)

        runner = OpencodeRunner(timeout=300, opencode_path=__file__)
        result = runner.run("test prompt")

        assert result.success is False
        assert "timed out" in result.error

    @patch("subprocess.run")
    def test_json_output_parsing(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type": "response", "content": "Hello World"}',
            stderr="",
        )

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is True
        assert "parsed_output" in result.metadata


class TestOpencodeRunnerStream:
    """测试流式输出"""

    @patch("subprocess.Popen")
    def test_stream_output(self, mock_popen):
        mock_process = Mock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = [
            '{"type": "start"}',
            '{"type": "chunk", "data": "Hello"}',
            '{"type": "chunk", "data": "World"}',
            "",
        ]
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = ""
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        runner = OpencodeRunner(opencode_path=__file__)
        chunks = list(runner.run_stream("test prompt"))

        assert len(chunks) >= 3
        assert chunks[-1]["type"] == "complete"

    @patch("subprocess.Popen")
    def test_stream_error_handling(self, mock_popen):
        mock_process = Mock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = [""]
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = "Error message"
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        runner = OpencodeRunner(opencode_path=__file__)
        chunks = list(runner.run_stream("test prompt"))

        assert len(chunks) > 0
        assert chunks[-1]["type"] == "error"


class TestOpencodeRunnerResume:
    """测试 Session 恢复"""

    @staticmethod
    def test_resume_nonexistent_session():
        runner = OpencodeRunner()
        result = runner.resume_session("nonexistent.json", "test prompt")

        assert result.success is False
        assert "not found" in result.error

    @patch("subprocess.run")
    def test_resume_existing_session(self, mock_run):
        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_file.write(b'{"session_id": "ses_test"}')
        temp_file.close()

        mock_run.return_value = Mock(
            returncode=0, stdout='{"result": "resumed"}', stderr=""
        )

        runner = OpencodeRunner()
        result = runner.resume_session(temp_file.name, "test prompt")

        assert result.success is True

        Path(temp_file.name).unlink()


class TestOpencodeRunnerCLI:
    """测试 CLI 功能"""

    @staticmethod
    def test_cli_args_parsing():
        import sys
        from opencode_runner import main as cli_main

        original_argv = sys.argv
        sys.argv = [
            "opencode_runner.py",
            "test prompt",
            "--model",
            "gpt-4",
            "--keep-session",
            "--stream",
            "--verbose",
        ]

        try:
            with patch.object(OpencodeRunner, "run_stream") as mock_stream:
                mock_stream.return_value = iter(
                    [{"type": "complete", "data": "test", "returncode": 0}]
                )
                exit_code = cli_main()
                assert exit_code == 0
                mock_stream.assert_called_once()
        finally:
            sys.argv = original_argv


class TestOpencodeRunnerIntegration:
    """集成测试（需要真实 opencode 环境）"""

    @pytest.mark.skip(reason="需要真实 opencode 环境")
    def test_real_run(self):
        runner = OpencodeRunner(model="gpt-4", verbose=True)
        result = runner.run("hello")
        assert isinstance(result, OpencodeResult)

    @pytest.mark.skip(reason="需要真实 opencode 环境")
    def test_real_stream(self):
        runner = OpencodeRunner(verbose=True)
        chunks = list(runner.run_stream("hello"))
        assert len(chunks) > 0

    @pytest.mark.skip(reason="需要真实 opencode 环境")
    def test_real_session_persistence(self):
        runner = OpencodeRunner(keep_session=True, verbose=True)
        result = runner.run("hello")
        assert result.session_file is not None

        runner2 = OpencodeRunner(keep_session=True, verbose=True)
        result2 = runner2.resume_session(result.session_file, "continue")
        assert isinstance(result2, OpencodeResult)


SCRIPT = Path(__file__).resolve().with_name("subprocess_streamer.py")
SPEC = importlib.util.spec_from_file_location("streaming_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def running(pid):
    """Treat exited zombies as stopped without signalling unrelated processes."""
    try:
        return Path("/proc", str(pid), "stat").read_text().split()[2] not in ("Z", "X")
    except FileNotFoundError:
        return False


@unittest.skipUnless(
    os.name == "posix" and Path("/proc").is_dir(), "Linux host process tests"
)
class StreamingProcessTests(unittest.TestCase):
    def setUp(self):
        self.original_threads = set(threading.enumerate())

    def tearDown(self):
        deadline = time.monotonic() + 1
        while (
            set(threading.enumerate()) - self.original_threads
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertFalse(
            set(threading.enumerate()) - self.original_threads,
            "stream reader or heartbeat thread survived runner completion",
        )

    def test_success_preserves_output_cwd_and_environment_without_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["STREAMER_FIXTURE_VALUE"] = "synthetic"
            code = (
                "import os,sys; print(os.getcwd()); "
                "print(os.environ['STREAMER_FIXTURE_VALUE']); "
                "sys.stdout.buffer.write(b'\\xe4\\xb8\\xad\\xe6\\x96\\x87\\n\\xff\\n'); "
                "print('warning',file=sys.stderr)"
            )
            forwarded = io.StringIO()
            with contextlib.redirect_stdout(forwarded):
                result = runner.run_subprocess_streaming(
                    [sys.executable, "-c", code], cwd=directory, env=env
                )
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0], 0)
            self.assertFalse(result[3])
            self.assertIn(directory + "\n", result[1])
            self.assertIn("synthetic\n", result[1])
            self.assertIn("中文\n\ufffd\n", result[1])
            self.assertEqual(result[2], "warning\n")
            self.assertIn("[pytest] warning\n", forwarded.getvalue())
            self.assertIn("中文\n\ufffd\n", forwarded.getvalue())

    def test_nonzero_exit_keeps_output_and_timeout_flag(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_subprocess_streaming(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err',file=sys.stderr); sys.exit(7)",
                ],
                timeout=5,
            )
        self.assertEqual(result, (7, "out\n", "err\n", False))

    def test_missing_executable_preserves_os_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                runner.run_subprocess_streaming(
                    [str(Path(directory) / "missing")], timeout=1
                )

    def test_timeout_without_descendants_keeps_partial_output(self):
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_subprocess_streaming(
                [
                    sys.executable,
                    "-c",
                    "import time; print('ready',flush=True); time.sleep(30)",
                ],
                timeout=1,
            )
        self.assertEqual(result, (-signal.SIGKILL, "ready\n", "", True))
        self.assertLess(time.monotonic() - started, 4)

    def test_group_signal_errors_still_stop_direct_child(self):
        for error in (ProcessLookupError, PermissionError):
            with self.subTest(error=error.__name__):
                started = time.monotonic()
                with (
                    patch.object(runner.os, "killpg", side_effect=error),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = runner.run_subprocess_streaming(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        timeout=0.2,
                    )
                self.assertEqual(result, (-signal.SIGKILL, "", "", True))
                self.assertLess(time.monotonic() - started, 4)

    def test_timeout_stops_descendant_with_redirected_pipes(self):
        self._check_descendant_timeout(inherit_pipes=False)

    def test_timeout_stops_descendant_holding_output_pipes(self):
        self._check_descendant_timeout(inherit_pipes=True)

    def test_leader_exit_still_collects_short_lived_descendant_output(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Path(directory) / "finished_child.py"
            child.write_text(
                "import sys,time\ntime.sleep(.2)\nprint('child-out')\nprint('child-err',file=sys.stderr)\n"
            )
            code = (
                "import subprocess,sys; subprocess.Popen([sys.executable,sys.argv[1]])"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.run_subprocess_streaming(
                    [sys.executable, "-c", code, str(child)], timeout=5
                )
            self.assertEqual(result, (0, "child-out\n", "child-err\n", False))

    def test_keyboard_interrupt_propagates_and_owned_child_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            pid_file = fixture / "interrupt-child.pid"
            caught_file = fixture / "caught"
            child = fixture / "interrupt_child.py"
            child.write_text(
                "from pathlib import Path\n"
                "import os,sys,time\n"
                "print('before-interrupt',flush=True)\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n"
            )
            harness = fixture / "interrupt_harness.py"
            harness.write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
                "from subprocess_streamer import run_subprocess_streaming\n"
                "try:\n    run_subprocess_streaming([sys.executable,sys.argv[2],sys.argv[3]],timeout=20)\n"
                "except KeyboardInterrupt:\n    Path(sys.argv[4]).write_text('propagated')\n    raise\n"
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(harness),
                    str(SCRIPT.parent),
                    str(child),
                    str(pid_file),
                    str(caught_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            pid = None
            try:
                pid = self._wait_for_pid(pid_file)
                # This fresh session contains only this synthetic harness and its child.
                self.assertEqual(os.getpgid(proc.pid), proc.pid)
                os.killpg(proc.pid, signal.SIGINT)
                stdout, _ = proc.communicate(timeout=5)
                self.assertIn(proc.returncode, (-signal.SIGINT, 130))
                self.assertEqual(caught_file.read_text(), "propagated")
                self.assertIn("before-interrupt\n", stdout)
                self._wait_until_stopped(pid)
                self.assertFalse(running(pid), "interrupted child survived")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate(timeout=5)
                self._cleanup_owned_child(pid, child)

    def _check_descendant_timeout(self, inherit_pipes):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            child = fixture / "timeout_child.py"
            child.write_text(
                "from pathlib import Path\nimport os,sys,time\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                "time.sleep(1.4)\nPath(sys.argv[2]).write_text('late')\n"
                "print('late-out',flush=True)\ntime.sleep(30)\n"
            )
            pid_file = fixture / "descendant.pid"
            late_file = fixture / "late"
            redirect = (
                ""
                if inherit_pipes
                else ",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL"
            )
            code = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3]]"
                + redirect
                + "); "
                "print('parent-ready',flush=True); time.sleep(30)"
            )
            pid = None
            try:
                started = time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = runner.run_subprocess_streaming(
                        [
                            sys.executable,
                            "-c",
                            code,
                            str(child),
                            str(pid_file),
                            str(late_file),
                        ],
                        timeout=1,
                    )
                elapsed = time.monotonic() - started
                pid = self._wait_for_pid(pid_file)
                time.sleep(0.5)
                self.assertEqual(result, (-signal.SIGKILL, "parent-ready\n", "", True))
                self._wait_until_stopped(pid)
                self.assertFalse(running(pid), "descendant survived timeout")
                self.assertFalse(late_file.exists(), "descendant wrote after timeout")
                self.assertLess(elapsed, 4, "inherited pipes delayed reader shutdown")
            finally:
                self._cleanup_owned_child(pid, child)

    def _wait_for_pid(self, pid_file):
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pid_file.exists(), "synthetic child did not start")
        return int(pid_file.read_text())

    @staticmethod
    def _wait_until_stopped(pid):
        deadline = time.monotonic() + 1
        while running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)

    @staticmethod
    def _cleanup_owned_child(pid, child_script):
        if pid is None or not running(pid):
            return
        # A unique fixture script path establishes ownership before any signal.
        cmdline = Path("/proc", str(pid), "cmdline").read_bytes()
        if str(child_script).encode() not in cmdline.split(b"\0"):
            raise RuntimeError("refusing to signal a PID without fixture ownership")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
