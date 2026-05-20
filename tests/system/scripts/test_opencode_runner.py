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
            (tmp_path / f"session_{i}.json").write_text('{}')
        runner = OpencodeRunner(keep_session=True, session_dir=str(tmp_path))
        sessions = runner.list_sessions()
        assert len(sessions) == 3
        runner.cleanup_all_sessions()
        assert len(runner.list_sessions()) == 0

    @staticmethod
    def test_list_sessions(tmp_path):
        for i in range(2):
            (tmp_path / f"session_{i}.json").write_text('{}')
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
        cmd = runner.build_command("test prompt", additional_args=["--verbose", "--debug"])
        assert "--verbose" in cmd
        assert "--debug" in cmd


class TestOpencodeRunnerResult:
    """测试运行结果"""

    @patch('subprocess.run')
    def test_successful_run(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"result": "success"}',
            stderr=''
        )

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is True
        assert result.output == '{"result": "success"'
        assert result.error == ""

    @patch('subprocess.run')
    def test_failed_run(self, mock_run):
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr='Error occurred'
        )

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is False
        assert "Error occurred" in result.error

    @patch('subprocess.run')
    def test_timeout_run(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="opencode", timeout=300)

        runner = OpencodeRunner(timeout=300)
        result = runner.run("test prompt")

        assert result.success is False
        assert "timed out" in result.error

    @patch('subprocess.run')
    def test_json_output_parsing(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type": "response", "content": "Hello World"}',
            stderr=''
        )

        runner = OpencodeRunner()
        result = runner.run("test prompt")

        assert result.success is True
        assert "parsed_output" in result.metadata


class TestOpencodeRunnerStream:
    """测试流式输出"""

    @patch('subprocess.Popen')
    def test_stream_output(self, mock_popen):
        mock_process = Mock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = [
            '{"type": "start"}',
            '{"type": "chunk", "data": "Hello"}',
            '{"type": "chunk", "data": "World"}',
            ''
        ]
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = ''
        mock_process.poll.side_effect = [None, None, None, 0]
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        runner = OpencodeRunner()
        chunks = list(runner.run_stream("test prompt"))

        assert len(chunks) >= 3
        assert chunks[-1]["type"] == "complete"

    @patch('subprocess.Popen')
    def test_stream_error_handling(self, mock_popen):
        mock_process = Mock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = ['']
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = 'Error message'
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        runner = OpencodeRunner()
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

    @patch('subprocess.run')
    def test_resume_existing_session(self, mock_run):
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        temp_file.write(b'{}')
        temp_file.close()

        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"result": "resumed"}',
            stderr=''
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
            'opencode_runner.py',
            'test prompt',
            '--model', 'gpt-4',
            '--keep-session',
            '--stream',
            '--verbose'
        ]

        try:
            with patch.object(OpencodeRunner, 'run_stream') as mock_stream:
                mock_stream.return_value = iter([{"type": "complete", "data": "test"}])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
