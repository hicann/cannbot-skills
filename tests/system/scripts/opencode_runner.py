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
OpencodeRunner - 封装 opencode run --format json 运行方式

功能：
1. 支持用户指定模型
2. 流式输出 opencode 运行结果
3. 支持用户控制是否保持 session 文件
4. 解析 JSON 格式的输出结果
5. 支持指定 opencode 可执行文件路径
6. 自动检测 opencode 是否可用

用法：
    from opencode_runner import OpencodeRunner

    runner = OpencodeRunner(model="gpt-4", keep_session=True)
    result = runner.run("请帮我写一个测试用例")

    # 或指定 opencode 路径
    runner = OpencodeRunner(opencode_path="D:/tools/opencode.exe")
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class OpencodeResult:
    success: bool
    output: str
    error: str
    session_file: Optional[str]
    metadata: Dict[str, Any]


class OpencodeNotFoundError(Exception):
    """opencode 命令未找到异常"""
    pass


class OpencodeRunner:
    OPENCODE_COMMANDS = ["opencode", "opencode.exe", "opencode-cli", "opencode-cli.exe"]

    def __init__(
            self,
            model: Optional[str] = None,
            variant: Optional[str] = None,
            keep_session: bool = False,
            session_dir: Optional[str] = None,
            timeout: int = 600,
            verbose: bool = False,
            workdir: Optional[str] = None,
            opencode_path: Optional[str] = None,
    ):
        self.model = model
        self.variant = variant
        self.keep_session = keep_session
        default_session_dir = Path(__file__).parent.parent / "logs"
        self.session_dir = Path(session_dir) if session_dir else default_session_dir
        self.timeout = timeout
        self.verbose = verbose
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.logger = logging.getLogger(f"{__name__}.OpencodeRunner")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.opencode_path = opencode_path
        self._current_session_file: Optional[str] = None
        self._detected_opencode_path: Optional[str] = None

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.opencode_session_id = None
        self._session_name: Optional[str] = None
        self._detect_opencode()

    # ---- Public instance methods ----

    def is_available(self) -> bool:
        return self._detected_opencode_path is not None

    def get_opencode_path(self) -> Optional[str]:
        return self._detected_opencode_path or self.opencode_path

    def run(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            session_name: Optional[str] = None,
    ) -> OpencodeResult:
        return self._run_non_streaming(prompt, skill, additional_args, session_name)

    def run_stream(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            session_name: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        return self.run_streaming(prompt, skill, additional_args, session_name)

    def run_streaming(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            session_name: Optional[str] = None,
            resume_session_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        self._session_name = session_name
        self._current_session_file = None

        try:
            cmd = self._build_command(
                prompt, skill, additional_args, resume_session_id
            )
        except OpencodeNotFoundError as e:
            yield {
                "type": "error",
                "data": str(e),
                "error_type": "opencode_not_found",
                "session_file": None
            }
            return

        yield from self._execute_streaming(cmd)

    def export_session_data(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        if not self.opencode_session_id:
            return {
                "success": False,
                "error": "No session ID available. Run a session first.",
                "data": None
            }

        opencode_cmd = self.get_opencode_path()
        if not opencode_cmd:
            return {
                "success": False,
                "error": "opencode command not found",
                "data": None
            }

        export_file = self._get_export_file_path(output_file)
        cmd = [opencode_cmd, "export", self.opencode_session_id]

        try:
            result = self._run_subprocess(cmd, description="Export")

            if result.returncode not in (0, -1):
                return {
                    "success": False,
                    "error": result.stderr or f"Export failed with code {result.returncode}",
                    "data": None,
                    "returncode": result.returncode
                }

            timed_out = (result.returncode == -1)
            if timed_out:
                self.logger.warning("Export timed out, attempting to use partial data (%d bytes)",
                                    len(result.stdout or ""))

            try:
                session_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                session_data = {"raw_output": result.stdout}

            self._write_session_data(session_data, export_file)

            return {
                "success": True,
                "timed_out": timed_out,
                "error": result.stderr if timed_out else None,
                "data": session_data,
                "export_file": export_file,
                "session_id": self.opencode_session_id
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Export timed out",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    def import_session_data(self, file_path: str) -> Dict[str, Any]:
        if not Path(file_path).exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "data": None
            }

        opencode_cmd = self.get_opencode_path()
        if not opencode_cmd:
            return {
                "success": False,
                "error": "opencode command not found",
                "data": None
            }

        cmd = [opencode_cmd, "import", file_path]

        try:
            result = self._run_subprocess(cmd, description="Import")

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Import failed with code {result.returncode}",
                    "data": None,
                    "returncode": result.returncode
                }

            import_result = self._parse_import_result(result.stdout)

            if import_result.get("sessionID"):
                self.opencode_session_id = import_result.get("sessionID")
                self._save_session_info(self.opencode_session_id)

            return {
                "success": True,
                "error": None,
                "data": import_result,
                "session_id": self.opencode_session_id,
                "source_file": file_path
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Import timed out",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    def get_session_messages(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        target_session_id = session_id or self.opencode_session_id
        if not target_session_id:
            return {
                "success": False,
                "error": "No session ID available",
                "messages": None
            }

        opencode_cmd = self.get_opencode_path()
        if not opencode_cmd:
            return {
                "success": False,
                "error": "opencode command not found",
                "messages": None
            }

        cmd = [opencode_cmd, "session", "list", "--format", "json"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workdir),
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Failed to list sessions",
                    "messages": None
                }

            sessions = json.loads(result.stdout) if result.stdout.strip() else []
            target_session = None
            for session in sessions:
                if session.get("id") == target_session_id:
                    target_session = session
                    break

            return {
                "success": True,
                "error": None,
                "session": target_session,
                "session_id": target_session_id,
                "all_sessions": sessions
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "messages": None
            }

    def cleanup_session(self, session_file: Optional[str] = None):
        target_file = session_file or self._current_session_file

        if target_file and Path(target_file).exists():
            Path(target_file).unlink()
            self._current_session_file = None

    def cleanup_all_sessions(self):
        for session_file in self.session_dir.glob("*.json"):
            if session_file.name.endswith("ses.json"):
                continue
            session_file.unlink()

    def list_sessions(self) -> List[str]:
        return [
            str(f) for f in self.session_dir.glob("*.json")
        ]

    def resume_session(
            self,
            session_file: str,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
    ) -> OpencodeResult:
        if not Path(session_file).exists():
            return OpencodeResult(
                success=False,
                output="",
                error=f"Session file not found: {session_file}",
                session_file=None,
                metadata={"timestamp": datetime.now(tz=timezone.utc).isoformat()}
            )

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            resume_session_id = session_info.get("session_id")
        except (json.JSONDecodeError, IOError):
            return OpencodeResult(
                success=False,
                output="",
                error=f"Invalid session file: {session_file}",
                session_file=None,
                metadata={"timestamp": datetime.now(tz=timezone.utc).isoformat()}
            )

        if not resume_session_id:
            return OpencodeResult(
                success=False,
                output="",
                error=f"No session_id in file: {session_file}",
                session_file=None,
                metadata={"timestamp": datetime.now(tz=timezone.utc).isoformat()}
            )

        self.keep_session = True
        self._current_session_file = session_file
        self._session_name = Path(session_file).stem

        return self._run_non_streaming(prompt, skill, additional_args, None, resume_session_id)

    def resume_session_stream(
            self,
            session_file: str,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        if not Path(session_file).exists():
            yield {
                "type": "error",
                "data": f"Session file not found: {session_file}",
                "session_file": None
            }
            return

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            resume_session_id = session_info.get("session_id")
        except (json.JSONDecodeError, IOError):
            yield {
                "type": "error",
                "data": f"Invalid session file: {session_file}",
                "session_file": None
            }
            return

        if not resume_session_id:
            yield {
                "type": "error",
                "data": f"No session_id in file: {session_file}",
                "session_file": None
            }
            return

        self.keep_session = True
        self._current_session_file = session_file
        self._session_name = Path(session_file).stem

        yield from self._run_streaming(prompt, skill, additional_args, None, resume_session_id)

    def build_command(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            resume_session_id: Optional[str] = None,
    ) -> List[str]:
        """构建 opencode 命令行参数列表"""
        return self._build_command(prompt, skill, additional_args, resume_session_id)

    # ---- Private static helpers ----

    @staticmethod
    def _make_error_result(error, cmd=None, output="", **metadata_extra):
        """Create an OpencodeResult for error cases."""
        metadata = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }
        if cmd:
            metadata["command"] = cmd
        metadata.update(metadata_extra)
        return OpencodeResult(
            success=False,
            output=output,
            error=error,
            session_file=None,
            metadata=metadata
        )

    # ---- Private instance helpers ----

    @staticmethod
    def _safe_env() -> dict:
        """Build a minimal environment for opencode subprocess execution.

        Strips sensitive variables (API keys, tokens) inherited from the parent process
        to prevent exposure to untrusted prompts that may execute arbitrary tools.
        """
        safe = {}
        for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "SHELL",
                     "TERM", "DISPLAY", "LOGNAME", "PWD"):
            val = os.environ.get(key)
            if val is not None:
                safe[key] = val
        # Pass through LLM API keys required by opencode for authentication.
        # opencode may also read keys from its own config file, but
        # environment variables are the primary mechanism in CI/CD pipelines.
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
                     "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
                     "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                     "ANTHROPIC_MODEL", "EVAL_MODEL", "EVAL_MODEL_VARIANT"):
            val = os.environ.get(key)
            if val is not None:
                safe[key] = val
        safe["PYTHONUNBUFFERED"] = "1"
        return safe

    def _save_session_info(self, session_id: str):
        if not self.keep_session:
            return

        if self._session_name:
            session_name = self._session_name
        else:
            timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
            session_name = f"opencode_session_{timestamp}"
        if session_name.endswith(".json"):
            session_name = session_name[:-5]

        session_path = self.session_dir / f"{session_name}.json"

        session_info = {
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "name": session_name
        }

        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, ensure_ascii=False, indent=2)

        self._current_session_file = str(session_path)

    def _build_command(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            resume_session_id: Optional[str] = None,
    ) -> List[str]:
        opencode_cmd = self.get_opencode_path()

        if not opencode_cmd:
            raise OpencodeNotFoundError(
                "opencode 命令未找到。请确保 opencode 已安装并添加到系统 PATH，"
                "或通过 opencode_path 参数指定路径。\n"
                "Windows 用户可能需要使用完整路径，如: "
                "OpencodeRunner(opencode_path='D:/path/to/opencode.exe')"
            )
        cmd = [opencode_cmd, "run", "--format", "json", "--dangerously-skip-permissions"]

        if self.model:
            cmd.extend(["--model", self.model])
        if self.variant:
            cmd.extend(["--variant", self.variant])
        if resume_session_id:
            cmd.extend(["--session", resume_session_id])

        if skill:
            skill_dir = self.workdir / skill
            if skill_dir.exists():
                cmd.extend(["--dir", str(skill_dir)])

        if additional_args:
            cmd.extend(additional_args)

        cmd.extend(["--", prompt])

        return cmd

    def _run_non_streaming(
            self,
            prompt: str,
            skill: Optional[str] = None,
            additional_args: Optional[List[str]] = None,
            session_name: Optional[str] = None,
            resume_session_id: Optional[str] = None,
    ) -> OpencodeResult:
        self._session_name = session_name

        try:
            cmd = self._build_command(prompt, skill, additional_args, resume_session_id)
        except OpencodeNotFoundError as e:
            return self._make_error_result(str(e), error_type="opencode_not_found")

        result = self._execute_command(cmd)
        if isinstance(result, OpencodeResult):
            return result
        return self._parse_json_output(result, cmd)

    def _execute_command(self, cmd):
        """Execute a command via subprocess. Returns CompletedProcess or OpencodeResult on error."""
        try:
            return subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workdir),
                env=self._safe_env(),
                encoding='utf-8',
                errors='replace'
            )
        except subprocess.TimeoutExpired:
            return self._make_error_result(
                f"Process timed out after {self.timeout} seconds",
                cmd=cmd, timeout=self.timeout
            )
        except FileNotFoundError as e:
            return self._make_error_result(
                f"opencode 命令未找到: {e}\n请安装 opencode 或指定 opencode_path 参数",
                cmd=cmd, error_type="file_not_found", exception=str(e)
            )
        except Exception as e:
            return self._make_error_result(str(e), cmd=cmd, exception=str(e))

    def _parse_json_output(self, result, cmd):
        """Parse subprocess output and return OpencodeResult."""
        if result.returncode != 0:
            return self._make_error_result(
                result.stderr or f"Process exited with code {result.returncode}",
                cmd=cmd, output=result.stdout, returncode=result.returncode
            )

        try:
            output_lines = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
            for item in output_lines:
                if item.get("type") == "step_start" and item.get("sessionID"):
                    self.opencode_session_id = item.get("sessionID")
                    self._save_session_info(self.opencode_session_id)
                    break
            return OpencodeResult(
                success=True,
                output=result.stdout,
                error="",
                session_file=self._current_session_file if self.keep_session else None,
                metadata={
                    "parsed_output": output_lines,
                    "session_id": self.opencode_session_id,
                    "returncode": result.returncode,
                    "command": cmd,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat()
                }
            )
        except json.JSONDecodeError:
            return OpencodeResult(
                success=True,
                output=result.stdout,
                error="",
                session_file=self._current_session_file if self.keep_session else None,
                metadata={
                    "json_parse_error": "Output is not valid JSON",
                    "returncode": result.returncode,
                    "command": cmd,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat()
                }
            )

    def _setup_streaming_process(self, cmd):
        """Set up a streaming subprocess with timeout and stderr monitoring."""
        env = self._safe_env()

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.workdir),
            env=env,
            bufsize=0
        )

        _timed_out = False
        stderr_output = ""

        def _kill_on_timeout():
            nonlocal _timed_out
            _timed_out = True
            try:
                process.kill()
            except Exception:
                self.logger.debug("终止超时进程时出现异常（进程可能已结束）")

        timer = threading.Timer(self.timeout, _kill_on_timeout)
        timer.daemon = True
        timer.start()

        def read_stderr():
            nonlocal stderr_output
            try:
                stderr_output = process.stderr.read().decode(
                    'utf-8', errors='replace'
                )
            except (UnicodeDecodeError, AttributeError):
                pass

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        return (
            process,
            timer,
            stderr_thread,
            lambda: _timed_out,
            lambda: stderr_output,
        )

    def _execute_streaming(self, cmd):
        """Execute a streaming command and yield results."""
        try:
            process, timer, stderr_thread, is_timed_out, get_stderr = \
                self._setup_streaming_process(cmd)
            output_buffer = []

            try:
                yield from self._stream_output_loop(
                    process, output_buffer, is_timed_out
                )
            finally:
                timer.cancel()

            stderr_thread.join(timeout=1)

            yield from self._yield_stream_result(
                process, output_buffer, get_stderr(), is_timed_out()
            )

        except FileNotFoundError as e:
            yield {
                "type": "error",
                "data": f"opencode 命令未找到: {e}\n请安装 opencode 或指定 opencode_path 参数",
                "error_type": "file_not_found",
                "session_file": self._current_session_file
            }
        except Exception as e:
            yield {
                "type": "exception",
                "data": str(e),
                "session_file": self._current_session_file
            }

    def _stream_output_loop(self, process, output_buffer, is_timed_out):
        """Read and yield output lines from the streaming process."""
        while True:
            line = process.stdout.readline()

            if not line:
                if process.poll() is not None:
                    break
                if is_timed_out():
                    break
                time.sleep(0.01)
                continue

            try:
                line = line.decode('utf-8', errors='replace').strip()
            except (UnicodeDecodeError, AttributeError):
                line = line.strip() if isinstance(line, str) else str(line)

            output_buffer.append(line)

            try:
                data = json.loads(line)
                if data.get("type") == "step_start" and data.get("sessionID"):
                    self.opencode_session_id = data.get("sessionID")
                    self._save_session_info(data.get("sessionID"))
                yield {
                    "type": "json_output",
                    "data": data,
                    "raw_line": line,
                    "session_file": self._current_session_file
                }
            except json.JSONDecodeError:
                yield {
                    "type": "raw_output",
                    "data": line,
                    "session_file": self._current_session_file
                }

    def _yield_stream_result(self, process, output_buffer, stderr_output, timed_out):
        """Yield the final result after stream processing completes."""
        if timed_out:
            yield {
                "type": "error",
                "data": f"Process timed out after {self.timeout} seconds",
                "returncode": process.returncode,
                "session_file": self._current_session_file,
                "error_type": "timeout"
            }
        elif process.returncode != 0:
            yield {
                "type": "error",
                "data": stderr_output or f"Process exited with code {process.returncode}",
                "returncode": process.returncode,
                "session_file": self._current_session_file
            }
        else:
            yield {
                "type": "complete",
                "data": "\n".join(output_buffer),
                "returncode": process.returncode,
                "session_file": self._current_session_file if self.keep_session else None
            }

    def _detect_opencode(self) -> bool:
        if self.opencode_path:
            return self._check_specified_path()

        for cmd in self.OPENCODE_COMMANDS:
            if self._try_command(cmd):
                return True

        return False

    def _check_specified_path(self) -> bool:
        """检查用户指定的 opencode 路径"""
        if Path(self.opencode_path).exists():
            self._detected_opencode_path = self.opencode_path
            return True
        return False

    def _try_command(self, cmd: str) -> bool:
        """尝试执行单个命令检测 opencode"""
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode in (0, 1):
                self._detected_opencode_path = cmd
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _run_subprocess(self, cmd: list, description: str = "") -> subprocess.CompletedProcess:
        """Run an opencode subcommand with common timeout and encoding settings."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.workdir),
                env=self._safe_env(),
                encoding='utf-8',
                errors='replace'
            )
        except subprocess.TimeoutExpired as e:
            self.logger.warning("%s timed out after 300s, retaining partial output (%d bytes)",
                                description or cmd, len(e.stdout or ""))
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\n(Process timed out after 300s)"
            )

    def _get_export_file_path(self, output_file: Optional[str]) -> str:
        """确定导出文件路径"""
        if output_file:
            return output_file
        if self._current_session_file:
            base_path = Path(self._current_session_file)
            return str(base_path.parent / f"{base_path.stem}_ses.json")
        return str(self.session_dir / f"export_{self.opencode_session_id}_ses.json")

    def _write_session_data(self, session_data: Dict, export_file: str) -> None:
        """保存 session 数据到文件"""
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def _parse_import_result(self, stdout: str) -> Dict[str, Any]:
        """解析 import 命令的输出"""
        if not stdout.strip():
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"raw_output": stdout}


def _parse_cli_args():
    """解析 CLI 参数"""
    parser = argparse.ArgumentParser(description="OpencodeRunner CLI")
    parser.add_argument("prompt", nargs='?', default="", help="Prompt to send to opencode")
    parser.add_argument("--model", help="Model to use (e.g., gpt-4, claude-3)")
    parser.add_argument("--skill", help="Skill to use")
    parser.add_argument("--keep-session", action="store_true", help="Keep session file after run")
    parser.add_argument("--session-dir", help="Directory to store session files")
    parser.add_argument("--resume-session", help="Resume from existing session file")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--stream", action="store_true", help="Use streaming output")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--workdir", help="Working directory")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup all session files after run")
    parser.add_argument("--opencode-path", help="Path to opencode executable")
    parser.add_argument("--check", action="store_true", help="Check if opencode is available")
    return parser.parse_args()


def _handle_stream_output(chunk: Dict[str, Any], keep_session: bool) -> None:
    """处理流式输出块"""
    chunk_type = chunk.get("type")
    if chunk_type == "json_output":
        logger.info(json.dumps(chunk["data"], indent=2))
    elif chunk_type == "raw_output":
        logger.info(chunk["data"])
    elif chunk_type == "error":
        logger.error("ERROR: %s", chunk['data'])
    elif chunk_type == "complete":
        logger.info("COMPLETE (returncode=%s)", chunk['returncode'])
        if keep_session and chunk.get("session_file"):
            logger.info("Session file: %s", chunk['session_file'])


def _run_stream_mode(runner: OpencodeRunner, prompt: str, skill: Optional[str],
                     resume_session: Optional[str], keep_session: bool) -> None:
    """运行流式模式"""
    if resume_session:
        for chunk in runner.resume_session_stream(resume_session, prompt, skill):
            _handle_stream_output(chunk, keep_session)
    else:
        for chunk in runner.run_stream(prompt, skill):
            _handle_stream_output(chunk, keep_session)


def _run_normal_mode(runner: OpencodeRunner, prompt: str, skill: Optional[str],
                     resume_session: Optional[str], keep_session: bool) -> bool:
    """运行普通模式，返回是否成功"""
    if resume_session:
        result = runner.resume_session(resume_session, prompt, skill)
    else:
        result = runner.run(prompt, skill)

    if result.success:
        logger.info(result.output)
        if keep_session and result.session_file:
            logger.info("Session file: %s", result.session_file)
        return True

    logger.error("ERROR: %s", result.error)
    return False


def main():
    """CLI 主入口函数，返回退出码"""
    args = _parse_cli_args()

    runner = OpencodeRunner(
        model=args.model,
        keep_session=args.keep_session,
        session_dir=args.session_dir,
        timeout=args.timeout,
        verbose=args.verbose,
        workdir=args.workdir,
        opencode_path=args.opencode_path
    )

    if args.check:
        if runner.is_available():
            logger.info("opencode is available: %s", runner.get_opencode_path())
        else:
            logger.error("opencode not found")
            logger.info("解决方案:")
            logger.info("  1. 安装 opencode 并添加到系统 PATH")
            logger.info("  2. 使用 --opencode-path 参数指定路径")
        return 0

    if not args.prompt:
        logger.error("prompt is required when not using --check")
        return 1

    if args.stream:
        _run_stream_mode(runner, args.prompt, args.skill,
                         args.resume_session, args.keep_session)
    else:
        success = _run_normal_mode(runner, args.prompt, args.skill,
                                    args.resume_session, args.keep_session)
        if not success:
            return 1

    if args.cleanup:
        runner.cleanup_all_sessions()
        logger.info("All session files cleaned up.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)