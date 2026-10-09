#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""可选单文件语法冒烟；复用已有编译配置，不生成构建树，不代替链接/device验证。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def compiler_arguments(entry, selected, target, *, env=None, object_output=None):
    """Rewrite a known compile entry, removing object/dependency outputs.

    Extra output and compiler forwarding modes are deliberately unsupported.
    Callers verify source/configuration identity and choose the output location.
    """
    directory = Path(entry["directory"]).resolve()
    argv = entry.get("arguments") or shlex.split(entry["command"])
    if any(x in argv for x in ("&&", ";", "|", "||", ">", "<")) or any(
        x.startswith("@") for x in argv
    ):
        raise ValueError("含 shell 操作或响应文件，不能安全重写为语法检查")
    if any(
        token.startswith(
            (
                "-Wp,",
                "-save-temps",
                "-fplugin",
                "-specs=",
                "-fprofile",
                "-fdump",
                "-dump",
                "-aux",
                "-MJ",
                "-gsplit-dwarf",
                "--serialize-diagnostics",
                "-fopt-info",
                "-foptimization-record",
                "-fsave-optimization-record",
                "-fdiagnostics",
                "-fmodule",
                "-ftime-trace",
                "--output",
                "--dependency-file",
            )
        )
        for token in argv
    ):
        raise ValueError("含额外输出或编译器扩展选项，需人工检查")
    if any(
        token in argv
        for token in (
            "-M",
            "-MM",
            "-E",
            "-S",
            "-Xclang",
            "-Xpreprocessor",
            "-wrapper",
            "--coverage",
            "-ftest-coverage",
        )
    ):
        raise ValueError("含额外输出或编译器转发选项，需人工检查")
    if not argv or "-c" not in argv:
        raise ValueError("不是可识别的单文件编译命令")
    if Path(argv[0]).name in ("ccache", "sccache"):
        argv = argv[1:]
    compiler_name = str((directory / argv[0]).resolve()) if "/" in argv[0] else argv[0]
    compiler = shutil.which(compiler_name, path=(env or os.environ).get("PATH"))
    if not compiler:
        raise ValueError("编译器不可用")
    args, index, replaced = [compiler], 1, False
    while index < len(argv):
        token = argv[index]
        if token in ("-o", "-MF", "-MT", "-MQ"):
            index += 2
            continue
        if token in ("-c", "-MD", "-MMD", "-MP"):
            index += 1
            continue
        if token.startswith(("-MF", "-MT", "-MQ", "-o")):
            index += 1
            continue
        if not token.startswith("-") and (directory / token).resolve() == selected:
            token, replaced = str(target), True
        args.append(token)
        index += 1
    if not replaced:
        raise ValueError("无法唯一定位编译输入")
    if object_output is None:
        args.append("-fsyntax-only")
    else:
        args.extend(["-c", "-o", str(object_output)])
    return args


def run(
    repo,
    target,
    database=None,
    *,
    env=None,
    check_only=False,
    cann=None,
    reference=None,
):
    repo = Path(repo).resolve()
    target = (repo / target).resolve()
    database = (
        Path(database).resolve() if database else repo / "build/compile_commands.json"
    )

    def skip(reason):
        return {
            "status": "SKIP",
            "reason": reason,
            "database": str(database),
            "target": str(target),
        }

    if not target.is_file() or repo not in target.parents:
        raise ValueError("冒烟目标必须为当前源码仓内文件")
    if not database.is_file():
        return skip("没有已有 compile_commands.json；不为冒烟创建完整构建树")
    cache = database.parent / "CMakeCache.txt"
    if not cache.is_file():
        return skip("缺少 CMakeCache.txt，无法核对配置来源")
    settings = {}
    for line in cache.read_text().splitlines():
        if "=" in line and ":" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            settings[key.split(":")[0]] = value
    source = Path(settings.get("CMAKE_HOME_DIRECTORY", "/")).resolve()
    if source not in (repo, repo / "cmake/device"):
        return skip("编译缓存来自其他源码目录")
    configured_cann = settings.get("ASCEND_INSTALL_PATH") or settings.get(
        "ASCEND_CANN_PACKAGE_PATH"
    )
    if (
        not cann
        or not configured_cann
        or Path(configured_cann).resolve() != Path(cann).resolve()
    ):
        return skip("无法证明编译缓存与指定 CANN 配置一致")
    selected = target
    if reference:
        selected = (repo / reference).resolve()
        if not selected.is_file() or selected.parent != target.parent:
            raise ValueError(
                "显式 reference 必须为目标同目录真实源码；需人工核对 target 专属编译定义"
            )
    entries = []
    for entry in json.loads(database.read_text()):
        directory = Path(entry["directory"]).resolve()
        if (directory / entry["file"]).resolve() == selected:
            entries.append(entry)
    if len(entries) != 1:
        return skip("目标没有唯一编译条目；可人工核对后显式使用 --reference")
    entry = entries[0]
    directory = Path(entry["directory"]).resolve()
    if not directory.is_dir() or not (
        directory == database.parent or database.parent in directory.parents
    ):
        return skip("编译工作目录不存在或不属于当前编译数据库")
    try:
        args = compiler_arguments(entry, selected, target, env=env)
    except ValueError as exc:
        return skip(str(exc))
    result = {
        "status": "READY",
        "target": str(target),
        "reference": str(selected) if reference else None,
        "database": str(database),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "configuration": "device"
        if source == repo / "cmake/device" or "aarch64" in args[0]
        else "host",
        "argv": args,
        "directory": str(directory),
        "limitation": "仅此配置语法检查；引用邻文件参数须人工核对差异，不代替全构建、链接或另一端编译。",
    }
    if check_only:
        return result
    started = time.monotonic()
    proc = subprocess.run(
        args, cwd=directory, env=env, capture_output=True, text=True, timeout=120
    )
    result.update(
        status="PASS" if proc.returncode == 0 else "FAIL",
        exit_code=proc.returncode,
        elapsed_seconds=time.monotonic() - started,
        output=(proc.stdout + proc.stderr)[-12000:],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "target", "cann"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--database")
    parser.add_argument(
        "--reference", help="仅限已人工核对定义/包含目录差异的同目录蓝本"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.repo,
            args.target,
            args.database,
            cann=args.cann,
            reference=args.reference,
            check_only=args.check_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result["status"] == "FAIL")
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
