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

"""只读比较已总结契约的来源文件；不扫描仓库、不自动刷新知识基线。"""

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = Path(__file__).resolve().parents[1] / "references/source-baseline.json"


def compare_group(root, entries):
    result = {"checked": 0, "changed": [], "missing": [], "unverified": []}
    if not entries:
        raise ValueError("来源分组不能为空")
    for name, expected in entries.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("来源必须为仓内相对路径: " + name)
        if root is None:
            result["unverified"].append(name)
            continue
        source = Path(root) / path
        if not source.is_file():
            result["missing"].append(name)
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        result["checked"] += 1
        if actual != expected:
            result["changed"].append(name)
    result["status"] = (
        "CHANGED"
        if result["changed"] or result["missing"]
        else "UNVERIFIED"
        if result["unverified"]
        else "MATCH"
    )
    return result


def check(baseline, repo, hcomm_repo, op, contracts=()):
    if baseline.get("schema_version") != 1:
        raise ValueError("不支持的来源基线版本")
    if op not in baseline["operators"]:
        raise ValueError("未收录该算子的来源基线，请定向核对: " + op)
    groups = {
        "hccl_shared": compare_group(repo, baseline["shared"]),
        "hccl_operator": compare_group(repo, baseline["operators"][op]),
        "hcomm_api": compare_group(hcomm_repo, baseline["hcomm"]),
    }
    selected = {}
    for name in contracts:
        entry = baseline.get("contracts", {}).get(name)
        if not entry or entry.get("op") != op:
            raise ValueError("该算子未收录此扩展契约，请定向核对: " + name)
        groups["hccl_contract:" + name] = compare_group(repo, entry["files"])
        selected[name] = {key: entry[key] for key in ("source_commit", "reference")}
    changed = any(g["changed"] or g["missing"] for g in groups.values())
    unverified = any(g["unverified"] for g in groups.values())
    return {
        "status": "CHANGED" if changed else ("UNVERIFIED" if unverified else "MATCH"),
        "op": op,
        "source_commits": baseline["source_commits"],
        "groups": groups,
        "contracts": selected,
        "limitation": "仅比较清单内实际文件内容；不验证传递依赖、所选 executor/分支适用性、部署库或算法正确性。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="调用方确认的 HCCL 仓根目录")
    parser.add_argument(
        "--hcomm-repo", help="调用方确认的 HCOMM 仓；缺失时报告 UNVERIFIED"
    )
    parser.add_argument("--op", required=True, help="算子目录名，如 all_reduce")
    parser.add_argument(
        "--contract",
        action="append",
        default=[],
        help="按需校验扩展契约，可重复；如 all-gather-parallel",
    )
    parser.add_argument(
        "--baseline", type=Path, default=BASELINE, help="已核验的来源基线 JSON"
    )
    args = parser.parse_args()
    try:
        for root in (args.repo, args.hcomm_repo):
            if root is not None and not Path(root).is_dir():
                raise ValueError("仓路径不存在: " + root)
        result = check(
            json.loads(args.baseline.read_text()),
            args.repo,
            args.hcomm_repo,
            args.op,
            args.contract,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
