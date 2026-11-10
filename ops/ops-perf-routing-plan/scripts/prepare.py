#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the Step-1 work directory, baseline build and profiling.

Commands:
  init      create the work tree, settle/copy the base code, normalize cases,
            detect the repo, discover golden and generate test-script skeletons
  build     compile and install the baseline operator into <work>/cann_opp
  profile   archive or collect <work>/profiling/baseline/performance.json

Commands and profile commands are executed as argv lists without shell parsing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    LOGGER,
    die,
    emit,
    load_json,
    read_cases,
    write_cases,
    write_json,
)

GOLDEN_NAMES = ("golden.py", "golden_impl.py", "golden_impl", "ref.py", "reference.py")
ST_PATTERNS = ("st_", "golden", "reference")
WORK_DIRS = (
    "code/ori_code",
    "inputs/golden",
    "profiling/baseline",
    "theoretical",
    "routing/coverage",
    "routing/templates",
    "logs",
)
META_KEYS = (
    "repo_type",
    "build_root",
    "op_rel",
    "vendor_dir",
    "torch_ext",
    "build_cmd",
    "pkg_pattern",
    "install_cmd",
    "soc",
    "ops",
)


# ---------------------------------------------------------------------------
# work tree
# ---------------------------------------------------------------------------
def _make_dirs(work: str) -> None:
    for rel in WORK_DIRS:
        os.makedirs(os.path.join(work, rel), exist_ok=True)


def _base_meta(work: str, op_name: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "operator_name": op_name,
        "source_operator_path": "",
        "repo_type": "",
        "build_root": "",
        "operator_rel_path": "",
        "vendor": "",
        "base_commit": "",
        "base_snapshot": "",
        "uncommitted_at_start": False,
        "golden_source": "",
        "work_dir": work,
        "hardware_file": os.path.join(work, "inputs", "hardware.json"),
        "cases_file": os.path.join(work, "inputs", "cases.csv"),
        "golden_flow": os.path.join(work, "inputs", "golden", "golden_flow.json"),
        "base_code": os.path.join(work, "code", "ori_code", op_name),
        "routing_plan": os.path.join(work, "routing", "coverage", "plan.json"),
        "tiling_routes": os.path.join(work, "routing", "tiling_routes.json"),
        "classification_report": os.path.join(
            work, "routing", "classification_report.md"
        ),
    }


def _write_input_placeholders(work: str, op_name: str) -> None:
    write_json(
        os.path.join(work, "inputs", "pipeline_meta.json"), _base_meta(work, op_name)
    )
    write_json(os.path.join(work, "inputs", "hardware.json"), {})
    write_json(
        os.path.join(work, "inputs", "golden", "golden_flow.json"),
        {
            "operator": op_name,
            "memory": {
                "gm_read_bytes": "numel * elem_bytes",
                "gm_write_bytes": "numel * elem_bytes",
                "working_set_bytes": "gm_read_bytes + gm_write_bytes",
            },
            "compute": [
                {
                    "kind": "vector",
                    "op": "Add",
                    "elements": "numel",
                    "dtype": "float32",
                    "count": 1,
                }
            ],
            "bandwidth_model": "auto",
        },
    )
    review = os.path.join(work, "inputs", "golden", "golden_flow_review.md")
    Path(review).write_text(
        "# Golden flow 建模说明\n\n"
        "请记录：\n"
        "1. 输入/输出张量及其字节数来源（依据 st golden / 算子语义）\n"
        "2. vector 指令列表及依据\n"
        "3. 是否存在 Cube / 多 pass / 额外搬运\n"
        "4. 与实测时间的对比（实测不得低于理论上限）\n",
        encoding="utf-8",
    )


def _write_theoretical_placeholders(work: str) -> None:
    write_json(os.path.join(work, "theoretical", "shape_limits.json"), {})
    path = os.path.join(work, "theoretical", "theoretical_review.md")
    Path(path).write_text(
        "# 理论上限复核\n\n"
        "请记录：\n"
        "1. T_vector / T_copy / T_cube 的计算依据\n"
        "2. 每个 case 的 t_hw 与 t_candidate 对比\n"
        "3. 门控：实测不得低于理论下限；异常时修正 golden flow / profiling 口径\n",
        encoding="utf-8",
    )


def _write_routing_placeholders(work: str) -> None:
    write_json(os.path.join(work, "routing", "tiling_routes.json"), {})
    write_json(os.path.join(work, "routing", "coverage_report.json"), {})
    Path(work, "routing", "tiling_table.csv").write_text("", encoding="utf-8")


def _create_structure(work: str, op_name: str) -> None:
    _make_dirs(work)
    _write_input_placeholders(work, op_name)
    _write_theoretical_placeholders(work)
    _write_routing_placeholders(work)


# ---------------------------------------------------------------------------
# git base
# ---------------------------------------------------------------------------
def _git(args: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=False
        )
    except OSError:
        return subprocess.CompletedProcess(args, 127, "", "git not found")


def _git_root(op_dir: str) -> str:
    result = _git(["-C", op_dir, "rev-parse", "--show-toplevel"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_head(repo_root: str) -> str:
    result = _git(["-C", repo_root, "rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty(repo_root: str, rel: str) -> bool:
    result = _git(["-C", repo_root, "status", "--porcelain", "--", rel])
    return bool(result.stdout.strip())


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def _snapshot(op_dir: str) -> str:
    skip = re.compile(r"(^|/)(build|build_out|\.git|__pycache__)(/|$)|\.(o|so|whl)$")
    hashes = []
    for cur, dirs, files in os.walk(op_dir):
        kept = []
        for name in dirs:
            if not skip.search(name):
                kept.append(name)
        dirs[:] = kept
        for name in files:
            path = os.path.join(cur, name)
            digest = _file_hash(path)
            if digest and not skip.search(path):
                hashes.append(digest)
    hashes.sort()
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def _settle_base(op_dir: str, work: str, no_commit: bool) -> Dict[str, Any]:
    repo_root = _git_root(op_dir)
    rel = os.path.relpath(op_dir, repo_root) if repo_root else ""
    dirty = bool(repo_root and _git_dirty(repo_root, rel))
    if repo_root and dirty and not no_commit:
        LOGGER.info("[prepare] 检测到未提交改动，落账为 base_commit")
        _git(["-C", repo_root, "add", "-A", "--", rel])
        _git(
            [
                "-C",
                repo_root,
                "commit",
                "-m",
                "ops-perf-routing-plan: settle base code",
                "--",
                rel,
            ]
        )
    elif repo_root and dirty:
        LOGGER.info("[prepare] 检测到未提交改动，--no-commit 生效")
    base = {
        "op_dir": op_dir,
        "repo_root": repo_root,
        "base_commit": _git_head(repo_root) if repo_root else "",
        "base_snapshot": _snapshot(op_dir),
        "uncommitted_at_start": dirty,
        "settled": True,
    }
    write_json(os.path.join(work, "inputs", "base_code.json"), base)
    return base


def _base_problems(base: Dict[str, Any], op_dir: str, repo_root: str) -> List[str]:
    problems, rel = [], os.path.relpath(op_dir, repo_root) if repo_root else ""
    base_commit = base.get("base_commit") or ""
    base_snapshot = base.get("base_snapshot") or ""
    if repo_root and base_commit and _git_dirty(repo_root, rel):
        problems.append("工作树相对 base_commit 存在未提交改动")
    if base_snapshot and not base_commit and _snapshot(op_dir) != base_snapshot:
        problems.append("内容快照漂移")
    return problems


def _check_base(op_dir: str, base: Dict[str, Any], work: str) -> Dict[str, Any]:
    repo_root = _git_root(op_dir)
    warnings = []
    if repo_root and not base.get("base_commit"):
        warnings.append("落账文件缺少 base_commit，仅用内容快照校验")
    result = {
        "ok": not _base_problems(base, op_dir, repo_root),
        "op_dir": op_dir,
        "base_commit": base.get("base_commit") or "",
        "base_snapshot": base.get("base_snapshot") or "",
        "uncommitted_at_start": bool(base.get("uncommitted_at_start")),
        "problems": _base_problems(base, op_dir, repo_root),
        "warnings": warnings,
    }
    write_json(os.path.join(work, "logs", "base_check.json"), result)
    return result


# ---------------------------------------------------------------------------
# repo detection / build / install
# ---------------------------------------------------------------------------
def _snake_case(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"[^0-9a-z_]+", "_", s).strip("_")


def _build_root_for(repo_type: str, path: str) -> str:
    parts = path.split(os.sep)
    markers = {"nn": "ops-nn", "transformer": "ops-transformer"}
    if repo_type in markers and markers[repo_type] in parts:
        return os.sep.join(parts[: parts.index(markers[repo_type]) + 1])
    indexes = []
    for index, part in enumerate(parts):
        if part == "ascendc":
            indexes.append(index)
    return os.sep.join(parts[: indexes[-1] + 1]) if indexes else path


def _detect_repo_type(path: str) -> str:
    norm = path.replace(os.sep, "/")
    if "ops-nn" in norm:
        return "nn"
    if "ops-transformer" in norm:
        return "transformer"
    if "omni-ops/inference" in norm or "/inference/ascendc" in norm:
        return "omni-inference"
    if "omni-ops/training" in norm or "/training/ascendc" in norm:
        return "omni-training"
    return "unknown"


def _base_repo_info(op_dir, op_name, chip, vendor) -> Dict[str, Any]:
    build_root = _build_root_for(_detect_repo_type(op_dir), op_dir)
    return {
        "repo_type": _detect_repo_type(op_dir),
        "op_dir": op_dir,
        "build_root": build_root,
        "op_rel": os.path.relpath(op_dir, build_root) if build_root else ".",
        "soc": chip,
        "vendor": vendor,
        "ops": _snake_case(op_name or os.path.basename(op_dir)),
    }


def _ops_repo_info(repo_type, chip, vendor) -> Dict[str, Any]:
    is_nn = repo_type == "nn"
    label = "cann-ops-nn" if is_nn else "cann-ops-transformer"
    return {
        "vendor_dir": f"vendors/{vendor}" if is_nn else f"vendors/{vendor}_transformer",
        "torch_ext": "cann_ops_nn" if is_nn else "cann_ops_transformer",
        "build_cmd": f"bash build.sh --pkg --soc={chip} --ops={{ops}} --vendor_name={vendor} -j16",
        "pkg_pattern": f"build_out/{label}-{vendor}_linux-aarch64.run",
        "install_cmd": '*.run --install-path="$ASCEND_OPP_PATH"',
    }


def _omni_repo_info(repo_type, chip) -> Dict[str, Any]:
    is_infer = repo_type == "omni-inference"
    label = "omni_custom_ops" if is_infer else "omni_training_custom_ops"
    vendor = (
        "omni_custom_transformer" if is_infer else "omni_training_custom_transformer"
    )
    return {
        "vendor_dir": f"vendors/{vendor}",
        "torch_ext": label,
        "build_cmd": f'bash build.sh -n {{ops}} -c {chip} -p "$ASCEND_HOME_PATH"',
        "pkg_pattern": f"output/CANN-{label}-*.run",
        "install_cmd": '*.run --install-path="$ASCEND_OPP_PATH"',
    }


def _empty_repo_info() -> Dict[str, Any]:
    return {
        "vendor_dir": "",
        "torch_ext": "",
        "build_cmd": "",
        "pkg_pattern": "",
        "install_cmd": "",
    }


def _detect_repo(op_dir: str, op_name: str, chip: str, vendor: str) -> Dict[str, Any]:
    op_dir = os.path.abspath(op_dir)
    info = _base_repo_info(op_dir, op_name, chip, vendor)
    repo_type = info["repo_type"]
    if repo_type in ("nn", "transformer"):
        info.update(_ops_repo_info(repo_type, chip, vendor))
    elif repo_type in ("omni-inference", "omni-training"):
        info.update(_omni_repo_info(repo_type, chip))
    else:
        info.update(_empty_repo_info())
    return info


def _copy_code(op_dir: str, dest: str) -> None:
    ignore = shutil.ignore_patterns(
        "build",
        "build_out",
        "CMakeFiles",
        "*.o",
        "*.so",
        "*.whl",
        "__pycache__",
        ".git",
        ".cache",
        "dist",
        "*.egg-info",
    )
    shutil.copytree(op_dir, dest, dirs_exist_ok=True, ignore=ignore)


def _build_argv(info: Dict[str, Any]) -> List[str]:
    command = str(info.get("build_cmd") or "").replace(
        "{ops}", str(info.get("ops") or "")
    )
    if not command:
        return []
    try:
        return shlex.split(os.path.expandvars(command))
    except ValueError as exc:
        die(f"无法解析编译命令: {exc}")
    return []


def _execute_build(build_root, argv, opp, work):
    with open(os.path.join(work, "logs", "build.log"), "w", encoding="utf-8") as log:
        result = subprocess.run(
            argv,
            cwd=build_root,
            env=dict(os.environ, ASCEND_OPP_PATH=opp),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def _find_package(build_root: str, info: Dict[str, Any]) -> str:
    pattern = info.get("pkg_pattern") or ""
    if not pattern:
        return ""
    packages = glob.glob(os.path.join(build_root, pattern))
    return sorted(packages)[0] if packages else ""


def _install_package(package: str, opp: str, work: str) -> int:
    log_path = os.path.join(work, "logs", "install.log")
    with open(log_path, "a", encoding="utf-8") as log:
        result = subprocess.run(
            [package, "--quiet", f"--install-path={opp}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def _write_env(work: str, info: Dict[str, Any], opp: str, build_root: str) -> str:
    env_sh = os.path.join(work, "env.sh")
    vendor_dir = info.get("vendor_dir") or ""
    Path(env_sh).write_text(
        "#!/bin/bash\n"
        "# Generated by prepare.py; source this before profiling/tests.\n"
        f'export ASCEND_OPP_PATH="{opp}"\n'
        f'export OP_VENDOR="{info.get("vendor")}"\n'
        f'export REPO_TYPE="{info.get("repo_type")}"\n'
        f'export BUILD_ROOT="{build_root}"\n'
        f'export OP_REL="{info.get("op_rel")}"\n'
        f'export OPS="{info.get("ops")}"\n'
        f'export SOC="{info.get("soc")}"\n'
        f'[ -f "{opp}/{vendor_dir}/bin/set_env.bash" ] && '
        f'source "{opp}/{vendor_dir}/bin/set_env.bash"\n',
        encoding="utf-8",
    )
    os.chmod(env_sh, 0o755)
    return env_sh


def _kernel_file_count(opp: str) -> int:
    root = os.path.join(opp, "op_impl", "ai_core", "tbe", "kernel")
    count = 0
    for _, _, files in os.walk(root):
        for name in files:
            if name.endswith((".json", ".o")):
                count += 1
    return count


def _write_install_check(work, info, opp, package, rc):
    vendor_dir = info.get("vendor_dir") or ""
    Path(work, "logs", "install_check.log").write_text(
        "== install_check.log ==\n"
        f"1. ASCEND_OPP_PATH={opp} in WORK: yes\n"
        f"2. {vendor_dir} exists: {'yes' if os.path.isdir(os.path.join(opp, vendor_dir)) else 'NO'}\n"
        f"3. pkg={package}\n"
        f"4. op_impl kernel json/o: {_kernel_file_count(opp)}\n"
        "5. vendor 归因: 用 msprof task_time.csv 的 kernel_name 复核\n",
        encoding="utf-8",
    )
    if rc != 0:
        LOGGER.warning("[prepare] 装包返回 rc=%s，详见 logs/install.log", rc)


def _run_build(work: str, info: Dict[str, Any], dry_run: bool) -> str:
    build_root = info.get("build_root") or ""
    argv = _build_argv(info)
    if not build_root or not argv:
        die("无法判定仓类型，不能推导编译命令")
    opp = os.path.join(work, "cann_opp")
    os.makedirs(opp, exist_ok=True)
    LOGGER.info(
        "[prepare] repo=%s ops=%s build_root=%s",
        info.get("repo_type"),
        info.get("ops"),
        build_root,
    )
    if dry_run:
        LOGGER.info("[prepare] dry-run: %s", " ".join(argv))
        return opp
    rc = _execute_build(build_root, argv, opp, work)
    if rc != 0:
        die(f"编译失败 rc={rc}，详见 logs/build.log")
    package = _find_package(build_root, info)
    if not package:
        die("未找到编译产物 .run 包")
    install_rc = _install_package(package, opp, work)
    _write_env(work, info, opp, build_root)
    _write_install_check(work, info, opp, package, install_rc)
    LOGGER.info("[prepare] 编译安装完成: %s", opp)
    return opp


# ---------------------------------------------------------------------------
# golden / test skeleton
# ---------------------------------------------------------------------------
def _candidate_dirs(op_dir: str) -> List[str]:
    return [
        os.path.join(op_dir, "tests"),
        os.path.join(op_dir, "test"),
        os.path.join(op_dir, "st"),
        op_dir,
    ]


def _is_golden(name: str) -> bool:
    if name in GOLDEN_NAMES:
        return True
    low = name.lower()
    return low.startswith(ST_PATTERNS) and low.endswith(".py")


def _walk_golden_dir(directory: str, files: List[str]) -> None:
    for cur, dirs, names in os.walk(directory):
        kept = []
        for name in dirs:
            if name not in ("build", "build_out", ".git", "__pycache__"):
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = os.path.join(cur, name)
            if os.path.isfile(path) and _is_golden(name):
                files.append(os.path.abspath(path))


def _collect_golden_files(op_dir: str) -> List[str]:
    files: List[str] = []
    for directory in _candidate_dirs(op_dir):
        if os.path.isdir(directory):
            _walk_golden_dir(directory, files)
    return sorted(set(files))


def discover_golden(op_dir: str, explicit: str = "") -> Dict[str, Any]:
    op_dir = os.path.abspath(op_dir)
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.exists(path):
            die(f"显式 golden 不存在: {path}")
        return {
            "op_dir": op_dir,
            "found": True,
            "source": "explicit",
            "files": [path],
            "primary": path,
        }
    uniq = _collect_golden_files(op_dir)
    return {
        "op_dir": op_dir,
        "found": bool(uniq),
        "source": "st" if uniq else "none",
        "files": uniq,
        "primary": uniq[0] if uniq else "",
    }


_PRECISION_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精度测试（由 prepare.py 生成，依据 st golden 编写后执行）。

golden 来源：{golden}
cases     ：{cases}
"""

import csv
import json

import numpy as np

OP_NAME = "{op_name}"
CASES = "{cases}"


def golden_call(inputs, attrs):
    """TODO：依据 st golden 实现期望输出计算。"""
    raise NotImplementedError("请依据 st golden 补全 golden_call")


def op_call(inputs, attrs):
    """TODO：调用自定义算子。禁止用内置 at::/torch_npu 内核冒充自定义 kernel。"""
    raise NotImplementedError("请依据算子入口补全 op_call")


def main() -> int:
    ok = True
    with open(CASES, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cid = row["case_id"]
            shapes = json.loads(row["input_shape"])
            dtypes = json.loads(row["dtype"])
            attrs = json.loads(row["attrs"]) if row.get("attrs") else {{}}
            try:
                inputs = [
                    np.random.randn(*shape).astype(np.dtype(dtype))
                    for shape, dtype in zip(shapes, dtypes)
                ]
                exp = golden_call(inputs, attrs)
                got = op_call(inputs, attrs)
                if isinstance(exp, (list, tuple)):
                    assert len(exp) == len(got)
                    assert all(
                        np.allclose(e, g, atol=1e-3, rtol=1e-3)
                        for e, g in zip(exp, got)
                    ), f"{{cid}} mismatch"
                else:
                    assert np.allclose(
                        exp, got, atol=1e-3, rtol=1e-3
                    ), f"{{cid}} mismatch"
                print(f"[precision] {{cid}} PASS")
            except Exception as exc:
                ok = False
                print(f"[precision] {{cid}} FAIL: {{exc}}")
    print("[precision]", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _perf_template(op_name: str, golden: str, cases: str) -> str:
    return f'''#!/bin/bash
# 性能测试（由 prepare.py 生成，依据 st golden 编写后执行）
# 目标：用 msprof 采集 {op_name} 各 case 的设备时长，产出 performance.json。
set -euo pipefail

OP_NAME="{op_name}"
CASES="{cases}"
OUT="${{TILING_ROUTE_WORK:-${{PWD}}}}/profiling/baseline"
mkdir -p "$OUT"

echo "[perf-test] 请依据 st golden 补全自定义算子调用 + msprof 采集命令"
echo "[perf-test] golden 来源: {golden}"
echo "[perf-test] cases    : {cases}"

# TODO: 在此补全每 case 的调用与 msprof 采集，最终写：
# cat > "$OUT/performance.json" <<'EOF'
# {{"cases": [{{"case_id": "case_00", "elapsed_us": 123.4}}]}}
# EOF

echo "[perf-test][ERROR] perf_test.sh 尚未补全，不能生成 performance.json" >&2
exit 1
'''


def generate_test_scripts(
    golden_source: str, cases: str, op_name: str, out_dir: str
) -> List[str]:
    source = load_json(golden_source) or {}
    primary = source.get("primary") or ""
    if not primary or not os.path.exists(primary):
        die(f"golden 不存在或为空: {primary!r}")
    os.makedirs(out_dir, exist_ok=True)
    precision = os.path.join(out_dir, "precision_test.py")
    Path(precision).write_text(
        _PRECISION_TEMPLATE.format(
            op_name=op_name, golden=primary, cases=os.path.abspath(cases)
        ),
        encoding="utf-8",
    )
    perf = os.path.join(out_dir, "perf_test.sh")
    Path(perf).write_text(
        _perf_template(op_name, primary, os.path.abspath(cases)), encoding="utf-8"
    )
    os.chmod(perf, 0o755)
    return [precision, perf]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _choose_work(out_root: str, op_name: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(out_root, f"{op_name}_VIP_routing_plan", stamp)
    suffix = 1
    while os.path.exists(work):
        work = os.path.join(
            out_root, f"{op_name}_VIP_routing_plan", f"{stamp}_{suffix}"
        )
        suffix += 1
    return work


def _apply_repo_overrides(info: Dict[str, Any], build_root: str, op_rel: str) -> None:
    if build_root:
        info["build_root"] = build_root
    if op_rel:
        info["op_rel"] = op_rel


def _init_repo(work, args, op_dir):
    vendor = f"{args.op_name}_{os.getuid()}_{os.getpid()}"
    info = _detect_repo(op_dir, args.op_name, args.soc, vendor)
    _apply_repo_overrides(info, args.build_root, args.op_rel)
    write_json(os.path.join(work, "logs", "detect_repo.json"), info)
    if info["repo_type"] == "unknown":
        LOGGER.warning("[prepare] 无法判定仓类型，repo_type=unknown")
    return info, vendor


def _init_cases(work: str, cases_path: str, op_name: str) -> None:
    if not cases_path:
        return
    if not os.path.isfile(cases_path):
        die(f"cases 文件不存在: {cases_path}")
    cases = read_cases(cases_path, op_name)
    for case in cases:
        case["operator"] = op_name
    write_cases(os.path.join(work, "inputs", "cases.csv"), cases)
    LOGGER.info("[prepare] %d 个 shape → inputs/cases.csv", len(cases))


def _hardware_source(args) -> str:
    if args.hardware:
        return args.hardware
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "references", "hardware", f"ascend{args.chip}.json")
    return (
        path
        if os.path.isfile(path)
        else os.path.join(root, "references", "hardware", "ascend910b2.json")
    )


def _init_hardware(work: str, args) -> None:
    hardware = _hardware_source(args)
    if not os.path.isfile(hardware):
        die(f"硬件档案不存在: {hardware}")
    shutil.copy2(hardware, os.path.join(work, "inputs", "hardware.json"))


def _init_golden(work, op_dir, args):
    source_path = os.path.join(work, "inputs", "golden", "golden_source.json")
    golden = discover_golden(op_dir, args.golden)
    write_json(source_path, golden)
    if not golden["found"]:
        LOGGER.warning("[prepare] 未发现 golden；可用 --golden 显式指定")
        return golden
    try:
        generate_test_scripts(
            source_path,
            os.path.join(work, "inputs", "cases.csv"),
            args.op_name,
            os.path.join(work, "inputs"),
        )
    except Exception as exc:
        LOGGER.warning("[prepare] 生成测试脚本失败: %s", exc)
    return golden


def _write_pipeline_meta(work, args, op_dir, info, vendor, base, golden):
    path = os.path.join(work, "inputs", "pipeline_meta.json")
    meta = load_json(path) or {}
    meta.update(base)
    meta.update(
        {
            "source_operator_path": op_dir,
            "repo_type": info["repo_type"],
            "build_root": args.build_root or info["build_root"],
            "operator_rel_path": args.op_rel or info["op_rel"],
            "vendor": vendor,
            "operator_name": args.op_name,
            "out_root": os.path.dirname(os.path.dirname(work)),
            "golden_source": golden.get("primary", ""),
            "golden_files": golden.get("files", []),
        }
    )
    if args.tiling_key_base is not None:
        meta["tiling_key_base"] = args.tiling_key_base
    for key in META_KEYS:
        value = info.get(key, "")
        meta[f"repo_{key}"] = (
            str(value).replace("{ops}", str(info.get("ops", "")))
            if key == "build_cmd"
            else value
        )
    write_json(path, meta)


def cmd_init(args: argparse.Namespace) -> int:
    op_dir = os.path.abspath(args.op_dir)
    if not os.path.isdir(op_dir):
        die(f"算子目录不存在: {op_dir}")
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    work = _choose_work(out_root, args.op_name)
    _create_structure(work, args.op_name)
    LOGGER.info("[prepare] 骨架: %s", work)
    base = _settle_base(op_dir, work, args.no_commit)
    if not _check_base(op_dir, base, work)["ok"]:
        LOGGER.warning("[prepare] 基底漂移门控未通过（详见 logs/base_check.json）")
    _copy_code(op_dir, os.path.join(work, "code", "ori_code", args.op_name))
    info, vendor = _init_repo(work, args, op_dir)
    _init_cases(work, args.cases, args.op_name)
    _init_hardware(work, args)
    golden = _init_golden(work, op_dir, args)
    _write_pipeline_meta(work, args, op_dir, info, vendor, base, golden)
    emit(work)
    return 0


def _pipeline_meta(work: str) -> Dict[str, Any]:
    return load_json(os.path.join(work, "inputs", "pipeline_meta.json")) or {}


def cmd_build(args: argparse.Namespace) -> int:
    work = os.path.abspath(args.work)
    meta = _pipeline_meta(work)
    op_dir = os.path.abspath(args.op_dir or meta.get("source_operator_path") or "")
    if not os.path.isdir(op_dir):
        die(f"算子目录不存在: {op_dir}")
    op_name = args.op_name or meta.get("operator_name") or os.path.basename(op_dir)
    chip = args.chip or meta.get("repo_soc") or "ascend910b"
    vendor = (
        args.vendor or meta.get("vendor") or f"{op_name}_{os.getuid()}_{os.getpid()}"
    )
    info = _detect_repo(op_dir, op_name, chip, vendor)
    _apply_repo_overrides(
        info, meta.get("build_root", ""), meta.get("operator_rel_path", "")
    )
    write_json(os.path.join(work, "logs", "detect_repo.json"), info)
    emit(_run_build(work, info, args.dry_run))
    return 0


def _command_argv(command: str) -> List[str]:
    if not command:
        return []
    try:
        return shlex.split(os.path.expandvars(command))
    except ValueError as exc:
        die(f"无法解析 profiling command: {exc}")
    return []


def _default_profile_argv(work: str) -> List[str]:
    path = os.path.join(work, "inputs", "perf_test.sh")
    return [path] if os.access(path, os.X_OK) else []


def _env_from_script(path: str) -> Dict[str, str]:
    env = {}
    if not os.path.isfile(path):
        return env
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("export "):
            continue
        key, _, value = text[7:].partition("=")
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            continue
        if key and parts:
            env[key] = parts[0]
    return env


def _profile_env(work: str, device: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(_env_from_script(os.path.join(work, "env.sh")))
    env.update(TILING_ROUTE_WORK=work, DEVICE_ID=device)
    return env


def _run_profile(work: str, argv: List[str], device: str) -> None:
    result = subprocess.run(argv, cwd=work, env=_profile_env(work, device), check=False)
    if result.returncode != 0:
        die(f"profiling command 失败: rc={result.returncode}")


def cmd_profile(args: argparse.Namespace) -> int:
    work = os.path.abspath(args.work)
    os.makedirs(os.path.join(work, "profiling", "baseline"), exist_ok=True)
    target = os.path.join(work, "profiling", "baseline", "performance.json")
    perf = os.path.abspath(args.perf_json) if args.perf_json else ""
    argv = _command_argv(args.command) or ([] if perf else _default_profile_argv(work))
    if argv:
        _run_profile(work, argv, args.device)
    perf = perf or target
    if not os.path.isfile(perf):
        die("缺少 performance.json；请用 --perf-json 指定或 --command 生成")
    if os.path.abspath(perf) != os.path.abspath(target):
        shutil.copy2(perf, target)
    if args.pipe_json and os.path.isfile(args.pipe_json):
        shutil.copy2(
            args.pipe_json,
            os.path.join(work, "profiling", "baseline", "pipe_summary.json"),
        )
    emit(target)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Step-1 工作目录与基线准备")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="初始化工作目录")
    p.add_argument("--op-dir", required=True)
    p.add_argument("--op-name", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--out-root", default=".")
    p.add_argument("--chip", default="910b2")
    p.add_argument("--soc", default="ascend910b")
    p.add_argument("--golden", default="")
    p.add_argument("--hardware", default="")
    p.add_argument("--tiling-key-base", dest="tiling_key_base", default=None)
    p.add_argument("--build-root", default="")
    p.add_argument("--op-rel", default="")
    p.add_argument("--no-commit", action="store_true")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("build", help="编译并安装 baseline 算子")
    p.add_argument("--work", required=True)
    p.add_argument("--op-dir", default="")
    p.add_argument("--op-name", default="")
    p.add_argument("--chip", default="")
    p.add_argument("--vendor", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("profile", help="归档或采集 performance.json")
    p.add_argument("--work", required=True)
    p.add_argument("--perf-json", default="")
    p.add_argument("--command", default="", help="可执行文件及参数；不经过 shell 解析")
    p.add_argument("--device", default="0")
    p.add_argument("--pipe-json", default="")
    p.set_defaults(func=cmd_profile)
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
