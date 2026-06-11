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
子进程实时流式输出运行器 + 心跳机制。

替代 subprocess.run(capture_output=True)，在子进程执行期间将 stdout/stderr
实时转发到父进程的 sys.stdout，并在此过程中添加定时心跳输出，防止 CI SSH
会话因长时间无输出而中断。
"""

import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple


def _reader_thread(stream, lines: List[str], done: threading.Event,
                   last_output: List[float], prefix: str = ""):
    """从 stream 逐行读取，追加到 lines 列表并实时输出到 stdout。"""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            lines.append(line)
            if prefix:
                sys.stdout.write(prefix)
            sys.stdout.write(line)
            sys.stdout.flush()
            last_output[0] = time.time()
    finally:
        stream.close()


def _heartbeat_thread(done: threading.Event, last_output: List[float],
                      interval: int, label: str):
    """每隔 interval 秒检查一次，若输出静默则打出一条心跳日志到 stdout。"""
    while not done.wait(5):
        # 在循环开始时检查一次 done，而不是依靠 wait 的返回值
        if done.is_set():
            break
        now = time.time()
        if now - last_output[0] >= interval:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            hb = f"[HEARTBEAT {ts}] {label} still running...\n"
            sys.stdout.write(hb)
            sys.stdout.flush()
            last_output[0] = now


def _start_streaming_threads(process: subprocess.Popen,
                              stdout_lines: List[str],
                              stderr_lines: List[str],
                              done: threading.Event,
                              last_output: List[float],
                              label: str) -> List[threading.Thread]:
    """启动 stdout/stderr 读取线程和心跳守护线程。"""
    threads: List[threading.Thread] = []

    t_out = threading.Thread(
        target=_reader_thread,
        args=(process.stdout, stdout_lines, done, last_output),
        daemon=True,
    )
    t_out.start()
    threads.append(t_out)

    t_err = threading.Thread(
        target=_reader_thread,
        args=(process.stderr, stderr_lines, done, last_output, "[pytest] "),
        daemon=True,
    )
    t_err.start()
    threads.append(t_err)

    t_hb = threading.Thread(
        target=_heartbeat_thread,
        args=(done, last_output, 60, label),
        daemon=True,
    )
    t_hb.start()
    threads.append(t_hb)

    return threads


def run_subprocess_streaming(
    cmd: List[str],
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    label: str = "",
) -> Tuple[int, str, str, bool]:
    """
    启动子进程并将 stdout/stderr 实时流式输出到 sys.stdout。

    参数：
        cmd:     子进程命令列表
        timeout: 超时秒数（None 表示不超时）
        cwd:     工作目录
        env:     环境变量
        label:   心跳消息中标识当前任务的标签

    返回：
        (returncode, captured_stdout, captured_stderr, timed_out)
    """
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1, cwd=cwd, env=env,
    )

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    last_output: List[float] = [time.time()]
    done = threading.Event()

    threads = _start_streaming_threads(
        process, stdout_lines, stderr_lines, done, last_output,
        label or " ".join(cmd[:3]),
    )

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        timed_out = True

    done.set()
    for t in threads:
        t.join(timeout=5)

    return (process.returncode, "".join(stdout_lines),
            "".join(stderr_lines), timed_out)
