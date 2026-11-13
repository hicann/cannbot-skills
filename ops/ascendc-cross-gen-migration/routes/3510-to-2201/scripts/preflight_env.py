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
"""环境预检：CANN、Python 依赖、构建工具与 vendor 库。

用法：
    python3 scripts/preflight_env.py [--json]

输出：
    - 每项检查的状态与建议；
    - 末尾 BLOCKING 列表（为空表示没有硬性缺失）；
    - --json 时输出机器可读结果。

只做检查，不修改环境。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys


def _cmd(name: str) -> str | None:
    return shutil.which(name)


def _module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def check() -> dict:
    rows = []

    def add(item: str, ok: bool, detail: str, fix: str = ""):
        rows.append({"item": item, "ok": ok, "detail": detail, "fix": fix})

    # CANN
    cann_home = _env("ASCEND_HOME_PATH") or _env("ASCEND_TOOLKIT_HOME")
    add(
        "CANN toolkit",
        bool(cann_home),
        cann_home or "ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME 未设置",
        "source <cann>/set_env.sh",
    )
    add(
        "ASCEND_OPP_PATH",
        bool(_env("ASCEND_OPP_PATH")),
        _env("ASCEND_OPP_PATH") or "未设置",
        "source <cann>/set_env.sh",
    )

    # build tools
    for cmd in ("cmake", "cpack", "bash"):
        p = _cmd(cmd)
        add(f"命令 {cmd}", bool(p), p or "未找到", "安装或加入 PATH")

    # python modules
    mods = {
        "torch": "PyTorch（跑仓库 pytest 时必需）",
        "torch_npu": "torch_npu（NPU 侧 API/测试）",
        "pytest": "pytest（仓库用例）",
        "pandas": "pandas（部分用例的数据表格）",
        "numpy": "numpy（直调 harness 必需）",
        "tensorflow": "tensorflow（仅 dtype 映射时可用 stub 替代）",
        "ml_dtypes": "ml_dtypes（bf16 映射，可 stub）",
    }
    for m, why in mods.items():
        ok = _module(m)
        add(
            f"Python 模块 {m}",
            ok,
            why if ok else f"缺失：{why}",
            "离线时从同版本环境拷贝 site-packages；仅 dtype 映射可写 stub",
        )

    # vendor library
    vendor = _env("ASCEND_CUSTOM_OPP_PATH") or ""
    libs = []
    if vendor:
        for root, _dirs, files in os.walk(vendor):
            if "libcust_opapi.so" in files:
                libs.append(os.path.join(root, "libcust_opapi.so"))
                break
    add(
        "vendor libcust_opapi.so",
        bool(libs),
        libs[0] if libs else "未找到（检查 ASCEND_CUSTOM_OPP_PATH 或安装产物）",
        "source <vendor>/bin/set_env.bash 或设置 LD_LIBRARY_PATH",
    )

    # optional npu-smi
    smi = _cmd("npu-smi")
    add("npu-smi", bool(smi), smi or "未找到（无 NPU 时可忽略）", "确认设备驱动")

    skip_prefixes = ("tensorflow", "ml_dtypes", "npu-smi")
    blocking = [
        r["item"]
        for r in rows
        if not r["ok"] and not r["item"].startswith(skip_prefixes)
    ]
    return {"rows": rows, "blocking": blocking}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = check()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("| 检查项 | 状态 | 说明/建议 |")
        print("|---|---|---|")
        for r in res["rows"]:
            print(f"| {r['item']} | {'OK' if r['ok'] else 'MISSING'} | {r['detail']} |")
        print()
        if res["blocking"]:
            print("BLOCKING:", ", ".join(res["blocking"]))
        else:
            print("BLOCKING: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
