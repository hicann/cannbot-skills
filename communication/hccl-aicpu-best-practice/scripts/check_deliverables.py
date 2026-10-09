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

"""Hash explicitly listed deliverables and detect missing/ignored Git delivery.

Read-only for the target repository. Output is created exclusively, never staged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
    )


def inspect(repo, manifest):
    repo = Path(repo).resolve()
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    tracked = set(git(repo, "ls-files", "-z").decode().split("\0"))
    items = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("manifest 必须包含非空 files 数组")
    entries, errors, seen = [], [], set()
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or not isinstance(item.get("role"), str)
            or not item["role"]
            or item.get("delivery") not in ("git", "artifact")
        ):
            raise ValueError("每项必须指定 path、role 和 delivery（git/artifact）")
        raw = Path(item["path"])
        path = raw if raw.is_absolute() else repo / raw
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError("重复交付路径: " + str(path))
        seen.add(resolved)
        try:
            relative = resolved.relative_to(repo).as_posix()
        except ValueError:
            relative = None
        if item["delivery"] == "git" and relative is None:
            raise ValueError("git 交付文件必须位于 repo 内: " + str(path))
        # Do not hash a symlink target as if it were a delivered regular file.
        symlink = path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents
        )
        exists = path.is_file() and not symlink
        is_tracked = relative in tracked if relative is not None else False
        ignored = False
        if relative is not None:
            result = subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", "--", relative],
                capture_output=True,
            )
            if result.returncode not in (0, 1):
                raise ValueError(
                    "git check-ignore 失败: " + result.stderr.decode(errors="replace")
                )
            ignored = result.returncode == 0
        digest = None
        if exists:
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            digest = checksum.hexdigest()
        entry = dict(
            item,
            path=str(path.absolute()),
            exists=exists,
            tracked=is_tracked,
            ignored=ignored,
            sha256=digest,
        )
        entries.append(entry)
        if not exists:
            errors.append(str(path) + ": 缺失、非普通文件或符号链接")
        if item["delivery"] == "git" and not is_tracked:
            errors.append(
                str(path)
                + ": 计划随代码交付但未跟踪"
                + ("（被忽略）" if ignored else "")
            )
    return dict(
        repo=str(repo), head=head, files=entries, errors=errors, passed=not errors
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        manifest_path = Path(args.manifest)
        raw = manifest_path.read_bytes()
        result = inspect(args.repo, json.loads(raw))
        result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, str(exc) + "\n")
    print("PASS" if result["passed"] else "\n".join(result["errors"]))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
