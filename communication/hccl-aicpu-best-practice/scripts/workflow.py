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

"""一次性源码定位和候选构建前基线门禁；不安装，不启动测试。"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
    )


def files(repo):
    return sorted(
        set(
            p.decode()
            for p in git(repo, "ls-files", "-co", "--exclude-standard", "-z").split(
                b"\0"
            )
            if p
        )
    )


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def task_inventory(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("task-dir 不存在: " + str(root))
    entries = []
    # 只发现交接元数据，不遍历源码/包内容，不跟随目录链接。
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in {".git", "build", "build_out", "node_modules"}
            and not (Path(current) / d).is_symlink()
        )
        for name in sorted(names):
            lower = name.lower()
            if name not in {
                "artifact.json",
                "snapshot.json",
                "source.json",
                "task-index.json",
            } and not (
                lower.endswith(".md")
                and any(k in lower for k in ("spec", "report", "handoff"))
            ):
                continue
            path = Path(current) / name
            if path.is_symlink():
                continue
            entries.append(
                {"path": str(path.relative_to(root)), "sha256": digest(path)}
            )
    return {
        "directory": str(root),
        "entries": entries,
        "fingerprint": hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode()
        ).hexdigest(),
    }


def build_environment(a):
    """解析 build.sh 的 -p / HOME / OPP / 默认路径；仅对子进程 source。"""
    env = dict(os.environ)
    explicit = getattr(a, "cann", None)
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.name == "set_env.sh":
            root = root.parent
        origin = "--cann"
    elif env.get("ASCEND_HOME_PATH"):
        root, origin = Path(env["ASCEND_HOME_PATH"]).resolve(), "ASCEND_HOME_PATH"
    elif env.get("ASCEND_OPP_PATH"):
        root, origin = Path(env["ASCEND_OPP_PATH"]).resolve().parent, "ASCEND_OPP_PATH"
    else:
        base = Path("/usr/local") if os.getuid() == 0 else Path.home()
        candidates = [base / "Ascend/ascend-toolkit/latest", base / "Ascend/latest"]
        root = next((p.resolve() for p in candidates if p.is_dir()), None)
        origin = "build.sh default"
    if root is None or not root.is_dir() or not (root / "set_env.sh").is_file():
        raise ValueError(
            "CANN 环境不可用；使用 --cann <安装目录或 set_env.sh>，或设置有效 ASCEND_HOME_PATH / ASCEND_OPP_PATH"
        )
    # shell stdout 留给诊断；通过独立文件读取环境，避免 source 的输出污染。
    with tempfile.TemporaryDirectory(prefix="hccl-env-") as temp:
        dump = Path(temp) / "environment"
        proc = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" >&2 && env -0 > "$2"',
                "hccl-env",
                str(root / "set_env.sh"),
                str(dump),
            ],
            env=env,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode:
            raise ValueError(
                "CANN set_env.sh 加载失败: "
                + proc.stderr.decode(errors="replace")[-2000:]
            )
        env = dict(
            item.split("=", 1)
            for item in dump.read_bytes().decode().split("\0")
            if item
        )
    required = ["bash", "cmake", "make", "gcc", "g++", "ar", "ld", "nm", "strip"]
    tools = {name: shutil.which(name, path=env.get("PATH")) for name in required}
    for name in ("gcc", "g++", "ar", "ranlib", "strip", "ld", "nm", "objcopy"):
        path = root / "toolkit/toolchain/hcc/bin" / ("aarch64-target-linux-gnu-" + name)
        tools["device-" + name] = (
            str(path) if path.is_file() and os.access(path, os.X_OK) else None
        )
    missing = [k for k, value in tools.items() if not value]
    if missing:
        raise ValueError("完整 host/device 构建缺少工具: " + ", ".join(missing))
    return env, {
        "cann": str(root),
        "origin": origin,
        "set_env_sha256": digest(root / "set_env.sh"),
        "tools": tools,
    }


def local_module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(a):
    started_at = datetime.now(timezone.utc).isoformat()
    repo = Path(a.repo).resolve()
    op = re.sub(r"[^a-z0-9]", "", a.op.lower())
    if not op:
        raise ValueError("op 不能为空")
    hits, fingerprints = [], {}
    symbol = re.compile(
        r"GetRes|GetThreadNum|HCCL_ALGO|HCCL_USE_NEW_SELECTOR"
        r"|REGISTER|Register|CalcCostCoeff|Select|selector"
        r"|Executor|executor"
    )
    for name in files(repo):
        path = repo / name
        if (
            path.suffix.lower() not in {".cc", ".cpp", ".c", ".h", ".hpp"}
            or not path.is_file()
        ):
            continue
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        shared = (
            name.startswith("src/ops/op_common/selector/")
            or name.startswith("src/ops/op_common/algorithm/topo_match/")
            or name.startswith("src/ops/op_common/algorithm/executor/")
            or name
            in (
                "src/common/alg_parse.h",
                "src/common/alg_parse.cc",
                "src/common/alg_env_config.h",
                "src/common/alg_env_config.cc",
                "src/ops/op_common/op_common.cc",
                "src/ops/op_common/op_common.h",
                "src/ops/op_common/algorithm/template/alg_v2_template_base.h",
            )
        )
        # 扫描目标算子源码，以及共享选路/配置/映射入口；仅输出定位，不输出全文件。
        if not shared and op not in normalized:
            continue
        body = path.read_text(errors="replace")
        fingerprints[name] = digest(path)
        lines = [
            {"line": i, "text": line.strip()[:240]}
            for i, line in enumerate(body.splitlines(), 1)
            if symbol.search(line)
        ]
        hits.append({"path": name, "locations": lines})
    environment = {str(Path(p).resolve()): digest(p) for p in a.environment_file}
    result = {
        "schema_version": 1,
        "started_at": started_at,
        "repo": str(repo),
        "head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "op": op,
        "files": fingerprints,
        "environment_files": environment,
        "locations": hits,
        "limitation": "文件指纹仅用于发现变化；不证明选路、资源契约或布局语义正确。只定位目标算子及共享入口，不能替代调用链审查。",
    }
    if getattr(a, "task_dir", None):
        result["task"] = task_inventory(a.task_dir)
    if a.previous:
        previous = json.loads(Path(a.previous).read_text())
        result["changed_files"] = sorted(
            k
            for k in set(fingerprints) | set(previous.get("files", {}))
            if fingerprints.get(k) != previous.get("files", {}).get(k)
        )
        result["changed_environment_files"] = sorted(
            k
            for k in set(environment) | set(previous.get("environment_files", {}))
            if environment.get(k) != previous.get("environment_files", {}).get(k)
        )
        result["context_changed"] = any(
            result.get(k) != previous.get(k) for k in ("repo", "head", "op")
        )
    write_new(a.output, result)
    return result


def evidence_module():
    path = Path(__file__).resolve().parents[2] / "hccl-test-tool-hvm/evidence.py"
    spec = importlib.util.spec_from_file_location("hccl_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outside(path, roots):
    path = Path(path).resolve()
    if any(path == r or r in path.parents for r in roots):
        raise ValueError("证据/产物路径位于构建清理目录: " + str(path))
    return path


def gate(a):
    repo = Path(a.repo).resolve()
    roots = [(repo / p).resolve() for p in ("build", "build_out")] + [
        Path(p).resolve() for p in a.clean_root
    ]
    out = outside(a.output, roots)
    if out == repo or repo in out.parents:
        raise ValueError(
            "候选输出目录必须位于源码仓之外（包含可恢复源码快照）: " + str(out)
        )
    if out.exists():
        raise ValueError("输出目录已存在，禁止覆盖或重复构建: " + str(out))
    artifact_path = outside(a.artifact, roots)
    artifact = json.loads(artifact_path.read_text())
    package = outside(artifact_path.parent / artifact["package"], roots)
    log = outside(artifact_path.parent / "build.log", roots)
    rc = outside(artifact_path.parent / "build.exit_code", roots)
    if (
        not package.is_file()
        or package.stat().st_size == 0
        or digest(package) != artifact["package_sha256"]
    ):
        raise ValueError("旧包不存在、为空或哈希不匹配")
    if (
        digest(log) != artifact["build_log_sha256"]
        or int(rc.read_text().strip()) != 0
        or artifact["build_exit_code"] != 0
    ):
        raise ValueError("旧包构建日志或退出码不可信")
    evidence = evidence_module()
    selectors, summaries = set(), []
    for run in a.baseline:
        root = outside(run, roots)
        outside(root / "manifest.json", roots)
        outside(root / "results.jsonl", roots)
        manifest, rows = evidence.validate(root)
        if manifest.get("role") != "baseline":
            raise ValueError("构建前证据必须显式 role=baseline: " + str(root))
        if manifest["declared_artifact"] != artifact:
            raise ValueError("基线声明的归档身份与提供的旧包不一致")
        if not rows:
            raise ValueError("基线不能为空")
        if manifest["environment"].get("HCCL_ALGO") or any(
            r["effective_algo_configs"] for r in rows.values()
        ):
            raise ValueError("默认基线不得设置 HCCL_ALGO 或通信域覆盖")
        for row in rows.values():
            outside(row["raw_log"], roots)
        selector = manifest["environment"].get("HCCL_USE_NEW_SELECTOR", "0")
        if selector not in ("0", "1"):
            raise ValueError("基线 selector 必须为 0 或 1")
        selectors.add(selector)
        summaries.append(
            {
                "run_dir": str(root),
                "selector": selector,
                "cases": len(rows),
                "failures": sum(r["verdict"] != "PASS" for r in rows.values()),
            }
        )
    needed = {"0", "1"} if a.require_new_selector else {"0"}
    if not needed <= selectors:
        raise ValueError(
            "缺少默认基线 selector=" + ",".join(sorted(needed - selectors))
        )
    if a.jobs < 1:
        raise ValueError("jobs 必须为正整数")
    return {
        "repo": str(repo),
        "artifact": str(artifact_path),
        "artifact_sha256": digest(artifact_path),
        "archive_files_sha256": {str(p): digest(p) for p in (package, log, rc)},
        "baseline": summaries,
        "baseline_all_passed": all(x["failures"] == 0 for x in summaries),
        "note": "基线完整允许构建；FAIL 必须保留，不能作为回归通过。",
        "argv": ["bash", "build.sh", "--pkg", "--full", "-j" + str(a.jobs)],
    }


def source_identity(repo):
    diff = git(repo, "diff", "--binary", "HEAD", "--")
    tracked = git(repo, "ls-files", "-z")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    added = {}
    for raw in untracked.split(b"\0"):
        if raw:
            name = raw.decode()
            path = repo / name
            if path.is_file():
                added[name] = digest(path)
    return {
        "head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_paths_sha256": hashlib.sha256(tracked).hexdigest(),
        "untracked_files": added,
    }, diff


def build(a):
    env, environment = build_environment(a)
    repo = Path(a.repo).resolve()
    # 以 git common dir 为身份，同一仓库的不同 worktree 共用锁；不写 .git。
    common = git(repo, "rev-parse", "--git-common-dir").decode().strip()
    identity = str((repo / common).resolve())
    lock = Path(tempfile.gettempdir()) / (
        "hccl-build-" + hashlib.sha256(identity.encode()).hexdigest() + ".lock"
    )
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("同仓库已有构建工作流运行")
        report = gate(a)
        report["environment"] = environment
        report["argv"] += ["-p", environment["cann"]]
        if getattr(a, "compile_smoke", None):
            report["compile_smoke"] = local_module("compile_smoke").run(
                repo,
                a.compile_smoke,
                getattr(a, "compile_database", None),
                env=env,
                check_only=a.check_only,
                cann=environment["cann"],
            )
            if report["compile_smoke"]["status"] == "FAIL":
                raise ValueError(
                    "单文件编译冒烟失败: "
                    + json.dumps(report["compile_smoke"], ensure_ascii=False)
                )
        if a.check_only:
            return report
        source, diff = source_identity(repo)
        out = Path(a.output).resolve()
        out.mkdir(parents=True, exist_ok=False)
        snapshot = local_module("source_snapshot").capture(
            repo, out / "source-snapshot"
        )
        report["source_snapshot"] = str(snapshot)
        (out / "source.diff").write_bytes(diff)
        write_new(out / "source.json", source)
        report.update(started_at=datetime.now(timezone.utc).isoformat(), source=source)
        write_new(out / "build.json", report)
        started = time.monotonic()
        with (out / "build.log").open("xb") as log:
            try:
                code = subprocess.call(
                    report["argv"],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                log.write((str(exc) + "\n").encode())
                code = 127
        (out / "build.exit_code").write_text(str(code) + "\n")
        completion = {
            "exit_code": code,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "build_log_sha256": digest(out / "build.log"),
        }
        write_new(out / "completion.json", completion)
        return dict(report, **completion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    for flag in ("repo", "op", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--previous")
    p.add_argument("--task-dir")
    p.add_argument("--environment-file", action="append", default=[])
    p = sub.add_parser("build")
    for flag in ("repo", "artifact", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--baseline", action="append", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--clean-root", action="append", default=[])
    p.add_argument("--require-new-selector", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--cann", help="CANN 安装目录或 set_env.sh，加载仅影响子进程")
    p.add_argument("--compile-smoke", help="可选单文件编译，不生成构建树")
    p.add_argument(
        "--compile-database", help="已有 compile_commands.json；默认 repo/build 下"
    )
    a = parser.parse_args()
    try:
        result = preflight(a) if a.command == "preflight" else build(a)
        if a.command == "preflight":
            result = {
                k: v
                for k, v in result.items()
                if k not in ("files", "locations", "environment_files")
            }
            result["output"] = str(Path(a.output).resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("exit_code", 0) == 0 else 1
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
