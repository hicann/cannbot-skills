# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile a generated probe with an existing entry; never write the original build tree."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def prepare(repo, worktree, target, output, database):
    repo, worktree, target, output, database = map(
        lambda p: Path(p).resolve(), (repo, worktree, target, output, database)
    )
    if worktree not in target.parents or not target.is_file():
        raise ValueError("probe 必须在指定工作树内")
    if any(output == p or p in output.parents for p in (repo, worktree)):
        raise ValueError("probe 输出必须位于源码与工作树之外")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("probe 输出须为已有临时目录中的新文件")
    if repo != worktree:

        def git(path, *args):
            return subprocess.check_output(
                ["git", "-C", str(path), *args], stderr=subprocess.PIPE
            )

        if git(repo, "rev-parse", "HEAD") != git(worktree, "rev-parse", "HEAD"):
            raise ValueError("编译数据库源码仓与隔离工作树的 HEAD 不一致")
        if git(repo, "diff", "HEAD", "--"):
            raise ValueError("原仓有未提交修改，不能与 HEAD 隔离工作树混用编译配置")
    reference = (
        repo
        / target.relative_to(worktree).parent
        / "ins_temp_all_reduce_mesh_1D_one_shot.cc"
    )
    entries = [
        entry
        for entry in json.loads(database.read_text())
        if (Path(entry["directory"]) / entry["file"]).resolve() == reference
    ]
    if len(entries) != 1 or not reference.is_file():
        raise ValueError("没有唯一的 one-shot 参照编译条目")
    entry = entries[0]
    directory = Path(entry["directory"]).resolve()
    if not directory.is_dir() or repo not in directory.parents:
        raise ValueError("编译条目工作目录不属于原源码仓")
    module_path = (
        Path(__file__).resolve().parents[2]
        / "hccl-aicpu-best-practice/scripts/compile_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("hccl_compile_smoke", module_path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    argv = smoke.compiler_arguments(entry, reference, target, object_output=output)

    def relocate(value):
        path = (directory / value).resolve()
        # Generated build headers use the existing build configuration. All
        # source include paths must come from the same worktree as the probe.
        if path == repo or repo in path.parents:
            if path != repo / "build" and repo / "build" not in path.parents:
                path = worktree / path.relative_to(repo)
        return str(path)

    includes = ("-isystem", "-iquote", "-include", "-imacros", "-I")
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in includes:
            if index + 1 >= len(argv):
                raise ValueError("include 参数缺少路径")
            argv[index + 1] = relocate(argv[index + 1])
            index += 2
            continue
        for flag in includes:
            if token.startswith(flag) and len(token) > len(flag):
                argv[index] = flag + relocate(token[len(flag):])
                break
        index += 1
    return argv, str(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "worktree", "target", "output", "database"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        argv, directory = prepare(
            args.repo, args.worktree, args.target, args.output, args.database
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("SKIP " + str(exc))
        return 3
    try:
        result = subprocess.run(
            argv, cwd=directory, capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            print((result.stdout + result.stderr)[-4000:])
        return int(result.returncode != 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
