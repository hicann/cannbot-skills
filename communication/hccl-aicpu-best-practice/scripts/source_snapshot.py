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

"""Capture recoverable tracked/untracked changes; restore only into a new clone.

The base commit is not bundled: restoration requires a repository containing it.
Ignored files are intentionally excluded. Changed/new symlinks are rejected.
New snapshots preserve the full Unix mode of changed tracked and untracked files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or any(x in ("..", ".git") for x in p.parts)
        or str(p) != name
    ):
        raise ValueError("unsafe snapshot path: " + repr(name))
    return p


def regular_file(root, name):
    rel = safe_relative(name)
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("changed/new symlinks are unsupported: " + str(path))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("not a regular file: " + str(path))
    return path


def capture(repo, output):
    repo = Path(repo).resolve()
    root = Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if repo != root:
        raise ValueError("repo must be the repository root")
    out = Path(output).absolute()
    resolved = out.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("snapshot must be outside the source repository")
    if out.exists() or out.is_symlink():
        raise ValueError("snapshot output already exists")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    patch = git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
    )
    changed = git(repo, "diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
    tracked_modes = []
    for raw in filter(None, changed):
        name = os.fsdecode(raw)
        safe_relative(name)
        p = root / name
        if p.exists() or p.is_symlink():
            path = regular_file(root, name)
            tracked_modes.append(
                dict(path=name, mode=stat.S_IMODE(path.stat().st_mode))
            )
    names = git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    entries, contents = [], []
    for raw in filter(None, names):
        name = os.fsdecode(raw)
        path = regular_file(root, name)
        data = path.read_bytes()
        blob = "files/" + str(len(entries))
        entries.append(
            dict(
                path=name,
                blob=blob,
                sha256=sha(data),
                mode=stat.S_IMODE(path.stat().st_mode),
            )
        )
        contents.append(data)
    out.mkdir(parents=True, exist_ok=False)
    (out / "files").mkdir()
    (out / "source.diff").write_bytes(patch)
    for entry, data in zip(entries, contents):
        (out / entry["blob"]).write_bytes(data)
    manifest = dict(
        schema_version=1,
        base_commit=head,
        source_repository=str(root),
        patch="source.diff",
        patch_sha256=sha(patch),
        tracked_modes=tracked_modes,
        untracked=entries,
        ignored_files="excluded",
        symlinks="changed/new rejected",
    )
    (out / "snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n"
    )
    verify(out)
    return out / "snapshot.json"


def verify(snapshot):
    root = Path(snapshot).absolute()
    if root.is_symlink():
        raise ValueError("snapshot directory cannot be a symlink")
    manifest = json.loads(regular_file(root, "snapshot.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    commit = manifest.get("base_commit", "")
    if len(commit) not in (40, 64) or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("invalid base commit")
    if (
        sha(regular_file(root, manifest["patch"]).read_bytes())
        != manifest["patch_sha256"]
    ):
        raise ValueError("source patch hash mismatch")
    seen = set()
    for entry in manifest.get("tracked_modes", []):
        safe_relative(entry["path"])
        if entry["path"] in seen:
            raise ValueError("duplicate source path")
        seen.add(entry["path"])
        if not isinstance(entry["mode"], int) or not 0 <= entry["mode"] <= 0o777:
            raise ValueError("invalid file mode")
    for entry in manifest["untracked"]:
        safe_relative(entry["path"])
        if entry["path"] in seen:
            raise ValueError("duplicate source path")
        seen.add(entry["path"])
        if not isinstance(entry["mode"], int) or not 0 <= entry["mode"] <= 0o777:
            raise ValueError("invalid file mode")
        if sha(regular_file(root, entry["blob"]).read_bytes()) != entry["sha256"]:
            raise ValueError("untracked file hash mismatch: " + entry["path"])
    return manifest


def restore(snapshot, repository, output):
    snapshot = Path(snapshot).absolute()
    manifest = verify(snapshot)
    repo = Path(repository).resolve()
    out = Path(output).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("restore output must not exist")
    if repo == out.resolve() or repo in out.resolve().parents:
        raise ValueError("restore output must be outside the source repository")
    git(repo, "cat-file", "-e", manifest["base_commit"] + "^{commit}")
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            "--",
            str(repo),
            str(out),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    git(
        out,
        "-c",
        "core.hooksPath=/dev/null",
        "checkout",
        "--detach",
        manifest["base_commit"],
    )
    patch = regular_file(snapshot, manifest["patch"])
    if patch.stat().st_size:
        git(out, "apply", "--binary", str(patch))
    for entry in manifest.get("tracked_modes", []):
        regular_file(out, entry["path"]).chmod(entry["mode"])
    for entry in manifest["untracked"]:
        p = out
        for part in safe_relative(entry["path"]).parts[:-1]:
            p = p / part
            if p.is_symlink():
                raise ValueError("restore parent is a symlink")
            p.mkdir(exist_ok=True)
        target = out / entry["path"]
        with target.open("xb") as f:
            f.write(regular_file(snapshot, entry["blob"]).read_bytes())
        target.chmod(entry["mode"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("capture")
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("verify")
    p.add_argument("snapshot")
    p = subs.add_parser("restore")
    p.add_argument("snapshot")
    p.add_argument(
        "--repository", required=True, help="repository containing the base commit"
    )
    p.add_argument("--output", required=True)
    a = parser.parse_args()
    try:
        if a.command == "capture":
            print(capture(a.repo, a.output))
        elif a.command == "verify":
            print(json.dumps(verify(a.snapshot), ensure_ascii=True))
        else:
            print(restore(a.snapshot, a.repository, a.output))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
