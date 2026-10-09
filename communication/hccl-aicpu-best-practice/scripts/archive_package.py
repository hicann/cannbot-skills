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

"""归档已有构建包和证据；不构建、不安装、不切换源码。输出目录必须不存在。"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def archive(a):
    package, log, rc = map(
        lambda s: Path(s).resolve(), (a.package, a.build_log, a.exit_code_file)
    )
    if int(rc.read_text().strip()) != 0:
        raise ValueError("构建退出码非零，不归档为可复用包")
    if not package.is_file() or not log.is_file() or package.stat().st_size == 0:
        raise ValueError("缺少非空安装包或构建日志")
    out = Path(a.output).resolve()
    repo = Path(a.repo).resolve()
    clean_roots = [package.parent, repo / "build", repo / "build_out"]
    clean_roots += [Path(p).resolve() for p in a.clean_root]
    if any(out == p or p in out.parents for p in clean_roots):
        raise ValueError("归档目录必须位于构建和包输出清理目录之外")
    libraries = {
        "host": digest(Path(a.host_library)),
        "device": digest(Path(a.device_library)),
    }
    source = getattr(a, "source_snapshot", None)
    snapshot_manifest = None
    if source:
        spec = importlib.util.spec_from_file_location(
            "hccl_source_snapshot", Path(__file__).with_name("source_snapshot.py")
        )
        snapshots = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snapshots)
        snapshot_manifest = snapshots.verify(source)
    out.mkdir(parents=True, exist_ok=False)
    # 固定文件名避免包名与日志文件名冲突；原名保存在 manifest 中。
    target = out / ("package" + "".join(package.suffixes))
    shutil.copy2(package, target)
    shutil.copy2(log, out / "build.log")
    shutil.copy2(rc, out / "build.exit_code")
    manifest = dict(
        schema_version=1,
        source_id=a.source_id,
        environment_id=a.environment_id,
        build_command=a.build_command,
        original_package=str(package),
        package=target.name,
        package_sha256=digest(target),
        build_log_sha256=digest(out / "build.log"),
        build_exit_code=0,
        expected_libraries=libraries,
    )
    if snapshot_manifest is not None:
        # Copy only verified files, never arbitrary extra paths or symlinks.
        snapshot_out = out / "source-snapshot"
        snapshot_out.mkdir()
        files = ["snapshot.json", snapshot_manifest["patch"]]
        files += [entry["blob"] for entry in snapshot_manifest["untracked"]]
        for name in set(files):
            destination = snapshot_out / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshots.regular_file(Path(source), name), destination)
        snapshots.verify(snapshot_out)
        manifest["source_snapshot"] = dict(
            path="source-snapshot",
            manifest_sha256=digest(snapshot_out / "snapshot.json"),
            base_commit=snapshot_manifest["base_commit"],
            recovery_requires="repository containing base_commit",
        )
    (out / "artifact.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return out / "artifact.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "repo",
        "package",
        "build-log",
        "exit-code-file",
        "source-id",
        "environment-id",
        "build-command",
        "output",
        "host-library",
        "device-library",
    ):
        p.add_argument("--" + flag, required=True)
    p.add_argument(
        "--clean-root", action="append", default=[], help="额外构建清理目录，可重复"
    )
    p.add_argument(
        "--source-snapshot",
        help="可恢复源码快照目录，由 source_snapshot.py capture 生成",
    )
    a = p.parse_args()
    try:
        print(archive(a))
    except (ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
