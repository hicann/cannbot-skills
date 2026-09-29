#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""按指定的 pipe stage 编译最小冒烟内核，判定 CANNBotDSL 的生成 / 编译能力。

用法:
    python3 probe_compile_stage.py translate   # DSL 翻译：bisheng 不运行
    python3 probe_compile_stage.py compile     # DSL 编译：bisheng 运行，产出可加载 .so

通过 `cannbotdsl.compile(@host 入口, specs...)` 获取 ProviderCallable，判定依据:
    translate / transform -> program.so_path 为空，bisheng 不运行
    compile               -> bisheng 运行，写出 .so

输出:
    stdout 末行 `RESULT=PASS:<说明>` 或 `RESULT=FAIL:<原因>`；失败另输出 `PROBE_ERROR=<错误码>`
    stderr 保留完整异常堆栈；错误码记录失败阶段，不推断异常的根因
    退出码 0 = PASS / 1 = FAIL
"""

import os
import sys
import traceback

VALID_STAGES = ("translate", "compile")

if len(sys.argv) < 2 or sys.argv[1] not in VALID_STAGES:
    sys.stderr.write(f"用法: {sys.argv[0]} <{'|'.join(VALID_STAGES)}>\n")
    sys.exit(2)

stage = sys.argv[1]
# 必须在 import cannbotdsl 之前设置：管线阶段由环境变量控制
os.environ["CANNBOTDSL_PIPE_STAGE"] = stage

HERE = os.path.dirname(os.path.abspath(__file__))
# probe_kernel.py 与本脚本同目录（均属本 skill 的 scripts/）
sys.path.insert(0, HERE)


def fail(code: str, msg: str) -> None:
    print(f"PROBE_ERROR={code}")
    print(f"RESULT=FAIL:{' '.join(msg.split())}")


program = None
failure = None
phase = "CANNBOTDSL_IMPORT_FAILED"
try:
    import cannbotdsl

    phase = "PROBE_INTERFACE_MISMATCH"
    for name in ("compile", "host", "kernel", "Tensor", "TensorSpec"):
        if not callable(getattr(cannbotdsl, name, None)):
            raise RuntimeError(f"当前 cannbotdsl 缺少可调用接口 {name}")
    from cannbotdsl import TensorSpec, dtypes
    from probe_kernel import probe_entry

    specs = (TensorSpec((1,), dtypes.float32),)
    phase = "DSL_TRANSLATE_FAILED" if stage == "translate" else "DSL_COMPILE_FAILED"
    program = cannbotdsl.compile(probe_entry, *specs)

    phase = "PROBE_INTERFACE_MISMATCH"
    so_path = program.so_path
    if not callable(getattr(program, "close", None)):
        raise RuntimeError("编译返回对象缺少 close()")

    phase = "PROBE_ARTIFACT_INVALID"
    if stage == "compile":
        if not so_path or not os.path.isfile(so_path) or os.path.getsize(so_path) == 0:
            raise RuntimeError("compile 阶段未产出非空 .so")
        result = f"已生成 .so，size={os.path.getsize(so_path)}；未执行 DSL kernel"
    else:
        if so_path:
            raise RuntimeError(f"translate 阶段不应产出 .so，却得到 {so_path}")
        result = "so_path 为空（bisheng 未运行，符合 translate 语义）"
except Exception as exc:
    traceback.print_exc()
    failure = (phase, f"{type(exc).__name__}: {exc}")
finally:
    if program is not None:
        try:
            program.close()
        except Exception as exc:
            traceback.print_exc()
            if failure is None:
                failure = ("PROBE_CLEANUP_FAILED", f"{type(exc).__name__}: {exc}")

if failure is not None:
    fail(*failure)
    raise SystemExit(1)
print(f"RESULT=PASS:{result}")
