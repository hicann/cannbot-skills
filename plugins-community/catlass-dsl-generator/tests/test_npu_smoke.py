# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "CATLASS_DSL_BENCH_SOLUTION",
    "CATLASS_DSL_BENCH_WORKLOAD",
    "CATLASS_DSL_BENCH_DEFINITION",
)


def configured_file(variable):
    raw = os.environ[variable]
    path = Path(raw).expanduser()
    assert path.is_absolute(), "{} 必须是绝对路径".format(variable)
    for component in (path,) + tuple(path.parents):
        assert not component.is_symlink(), "{} 路径不得经过符号链接：{}".format(
            variable, component
        )
    resolved = path.resolve(strict=True)
    assert resolved.is_file(), "{} 必须指向普通文件".format(variable)
    return resolved


def configured_device():
    value = os.environ.get("CATLASS_DSL_BENCH_DEVICE", "npu:0")
    assert value == "npu" or (
        value.startswith("npu:") and value[4:].isdigit()
    ), "CATLASS_DSL_BENCH_DEVICE 必须为 npu[:index]"
    return value


@pytest.mark.parametrize("value", ["npu", "npu:3"])
def test_configured_device_accepts_npu(monkeypatch, value):
    monkeypatch.setenv("CATLASS_DSL_BENCH_DEVICE", value)
    assert configured_device() == value


def test_configured_device_rejects_cpu(monkeypatch):
    monkeypatch.setenv("CATLASS_DSL_BENCH_DEVICE", "cpu")
    with pytest.raises(AssertionError, match="npu"):
        configured_device()


@pytest.mark.npu
def test_configured_npu_benchmark(tmp_path):
    configured = [name for name in REQUIRED if os.environ.get(name)]
    if not configured:
        pytest.skip("设置 CATLASS_DSL_BENCH_* 三个输入后运行真实 NPU smoke")
    if len(configured) != len(REQUIRED):
        pytest.fail("NPU smoke 自定义输入必须同时提供：" + ", ".join(REQUIRED))

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("当前 pytest 进程无法访问 NPU")
    solution, workload, definition = [configured_file(name) for name in REQUIRED]
    output = tmp_path / "npu-smoke"
    command = [
        sys.executable,
        str(ROOT / "skills/catlass-dsl-bench/scripts/bench.py"),
        "--solution",
        str(solution),
        "--workload",
        str(workload),
        "--definition",
        str(definition),
        "--device",
        configured_device(),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, "stdout:\n{}\nstderr:\n{}".format(
        completed.stdout, completed.stderr
    )
    assert (output / "result.json").is_file()
