#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""ascendc-sync-audit benchmark 最小集评测入口（双链回归）。

跑法（在 ops/ascendc-sync-audit 目录下）：
  python3 benchmark/eval_benchmark.py

覆盖两条检测链：
  A. sync_audit.py 链   : benchmark/defects + benchmark/correct，指标口径与
                          tests/eval_recall.py 全量评测一致
  B. ascendc_flow_analyzer.py 链 : references/examples/sync14_*.cpp 图证据回归
                          （预期 findings 2/1/1/0）

分级模式（--full，消费 tests/labels.json 全量 581 文件）：
  L0 门禁 : benchmark/labels.json（条例+行号双命中，随仓库）
  L1 全量 : 标注集（dataset/real_dataset，expected_sync + defect_lines）
            + 真实 PR 集（real_pr/real_pairs，文件级检出 + defect_lines 行号）
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sync_logging import init_logging  # noqa: E402

SYNC_SCRIPT = ROOT / "scripts" / "sync_audit.py"
FLOW_SCRIPT = ROOT / "scripts" / "ascendc_flow_analyzer.py"
TESTS_DIR = ROOT / "tests"
LABELS = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))["files"]

LOGGER = logging.getLogger('benchmark')
STDERR_LOGGER = logging.getLogger('benchmark.stderr')
init_logging(LOGGER, STDERR_LOGGER)

# analyzer 链期望：文件 → 预期 findings 数
FLOW_EXPECTED = {
    "references/examples/sync14_output_bundle_index.cpp": 2,
    "references/examples/sync14_same_name_index_rebind.cpp": 1,
    "references/examples/sync14_same_root_different_region.cpp": 1,
    "references/examples/sync14_equivalent_modulo_region.cpp": 0,
}


@dataclass
class SyncRegressionResult:
    """sync_audit 链统计结果（单对象传递，避免多返回值/多参数）。

    命名口径：defect/correct 两侧统一使用 passed/missed 成对命名——
    defect_passed=检出正确、defect_missed=漏检；correct_passed=零输出、correct_missed=误报。
    """

    defect_passed: int = 0          # defect：通过三层判定（有输出+条例命中+行号命中）数
    defect_missed: int = 0          # defect：未通过（无输出或条例/行号未命中）数
    correct_passed: int = 0         # correct：零输出（未误报）数
    correct_missed: int = 0         # correct：产生任意候选（误报）数
    correct_redline_missed: int = 0  # correct：产生红线候选（红线误报）数
    missed: list = field(default_factory=list)    # 未通过明细
    fp_files: list = field(default_factory=list)  # 误报明细


def audit_sync(relpath: str, base_dir: Path = None):
    base = base_dir if base_dir is not None else HERE
    try:
        out = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), str(base / relpath), "--format", "json"],
            capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout)
    except Exception:
        return None
    findings = data.get("findings", [])
    return {
        "n": len(findings),
        "codes": {f["code"] for f in findings},
        "lines": {f.get("line", 0) for f in findings},
        "redlines": sum(1 for f in findings if f.get("severity") == "红线"),
    }


def audit_flow(relpath: str):
    try:
        out = subprocess.run(
            [sys.executable, str(FLOW_SCRIPT), str(ROOT / relpath), "--format", "json"],
            capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout)
    except Exception:
        return None
    return len(data.get("findings", []))


def run_flow_regression() -> int:
    """analyzer 链图证据回归（预期 findings 2/1/1/0），返回失败数。"""
    LOGGER.info("[analyzer 链] SYNC-14 图证据回归")
    failed = 0
    for rel, expected in sorted(FLOW_EXPECTED.items()):
        got = audit_flow(rel)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        LOGGER.info("  [%s] %s: 预期 %d, 实际 %s", status, Path(rel).name, expected, got)
        failed += 0 if ok else 1
    return failed


def audit_suite(labels: dict, base_dir: Path = None) -> SyncRegressionResult:
    """对给定 labels 全集跑 sync_audit 链，返回统计（复用行号命中层）。"""
    result = SyncRegressionResult()
    for rel, meta in sorted(labels.items()):
        res = audit_sync(rel, base_dir)
        if res is None:
            STDERR_LOGGER.warning("[ERR] %s: 评测失败", rel)
            continue
        if meta["label"] == "defect":
            exp = meta.get("expected_sync", "")
            defect_lines = meta.get("defect_lines", [])
            hit = res["n"] > 0 and (not exp or exp in res["codes"])
            if hit and defect_lines:
                hit = bool(res["lines"] & set(defect_lines))
            if hit:
                result.defect_passed += 1
            else:
                result.defect_missed += 1
                why = "条例未命中" if (exp and exp not in res["codes"]) else "无输出或行号未命中"
                result.missed.append(f"{rel}({why})")
        else:
            if res["n"] > 0:
                result.correct_missed += 1
                result.fp_files.append(rel)
            else:
                result.correct_passed += 1
            result.correct_redline_missed += 1 if res["redlines"] > 0 else 0
    return result


def run_full_regression() -> int:
    """消费 tests/（submodule）labels.json 全量：L2 标注集 + L3 真实 PR 集。"""
    labels_file = TESTS_DIR / "labels.json"
    if not labels_file.exists():
        LOGGER.info("[L1 全量] tests/ 语料未拉取（git submodule update --init --recursive），跳过 L1")
        return 0
    full_labels = json.loads(labels_file.read_text(encoding="utf-8"))["files"]
    l2 = {k: v for k, v in full_labels.items()
          if k.split("/")[0] in ("dataset", "real_dataset")}
    l3 = {k: v for k, v in full_labels.items()
          if k.split("/")[0] in ("real_pr", "real_pairs")}
    ok = 0
    for name, subset in (("L1 全量·标注集", l2), ("L1 全量·真实 PR 集", l3)):
        r = audit_suite(subset, TESTS_DIR)
        LOGGER.info("[%s] 缺陷通过(条例/行号): %d/%d  误报: %d/%d  红线: %d",
                    name, r.defect_passed, r.defect_passed + r.defect_missed,
                    r.correct_missed,
                    r.correct_missed + sum(1 for m in subset.values() if m["label"] == "correct"),
                    r.correct_redline_missed)
        ok += r.defect_missed + r.correct_missed
    return 1 if ok else 0


def run_sync_regression() -> SyncRegressionResult:
    """sync_audit 链评测：defect 必须「有输出且期望条例命中」，correct 必须零输出。"""
    result = SyncRegressionResult()
    for rel, meta in sorted(LABELS.items()):
        res = audit_sync(rel)
        if res is None:
            STDERR_LOGGER.warning("[ERR] %s: 评测失败", rel)
            continue
        if meta["label"] == "defect":
            exp = meta.get("expected_sync", "")
            defect_lines = meta.get("defect_lines", [])
            hit = res["n"] > 0 and exp in res["codes"]
            if hit and defect_lines:
                hit = bool(res["lines"] & set(defect_lines))
            if hit:
                result.defect_passed += 1
            else:
                result.defect_missed += 1
                where = f", 行{defect_lines}未命中" if defect_lines and res["n"] > 0 else ""
                result.missed.append(f"{rel}(期望{exp}, 检出{res['codes']}{where})")
        else:
            if res["n"] > 0:
                result.correct_missed += 1
                result.fp_files.append(f"{rel}({res['codes']})")
            else:
                result.correct_passed += 1
            result.correct_redline_missed += 1 if res["redlines"] > 0 else 0
    return result


def print_report(result: SyncRegressionResult) -> None:
    """输出 sync_audit 链指标。"""
    n_def = result.defect_passed + result.defect_missed
    n_ok = result.correct_missed + (sum(1 for m in LABELS.values() if m["label"] == "correct")
                                        - result.correct_missed)
    if n_def:
        LOGGER.info("[sync_audit 链] 缺陷检出通过(条例+行号): %d/%d", result.defect_passed, n_def)
    else:
        LOGGER.info("[sync_audit 链] 无缺陷样例")
    if n_ok:
        LOGGER.info("[sync_audit 链] 文件级误报率: %d/%d = %.1f%%", result.correct_missed, n_ok,
                    result.correct_missed / n_ok * 100)
        LOGGER.info("[sync_audit 链] 红线误报率: %d/%d = %.1f%%", result.correct_redline_missed, n_ok,
                    result.correct_redline_missed / n_ok * 100)
    else:
        LOGGER.info("[sync_audit 链] 无 correct 样例")
    if result.missed:
        LOGGER.info("")
        LOGGER.info("未通过（无输出或条例未命中）: %s", result.missed)
    if result.fp_files:
        LOGGER.info("误报: %s", result.fp_files)


def main() -> int:
    if '--full' in sys.argv:
        n_def = sum(1 for m in LABELS.values() if m["label"] == "defect")
        LOGGER.info("ascendc-sync-audit benchmark（--full 分级：L0 门禁 + L1 全量）")
        LOGGER.info("  L0 门禁: %d 样例（%d 缺陷 + %d 正确）", len(LABELS),
                    n_def, len(LABELS) - n_def)
        result = run_sync_regression()
        print_report(result)
        LOGGER.info("")
        l1_ok = result.correct_missed or result.defect_missed
        full_fail = run_full_regression()
        LOGGER.info("")
        flow_failed = run_flow_regression()
        return 1 if (l1_ok or full_fail or flow_failed) else 0
    n_def = sum(1 for m in LABELS.values() if m["label"] == "defect")
    n_ok = sum(1 for m in LABELS.values() if m["label"] == "correct")
    LOGGER.info("ascendc-sync-audit benchmark 最小集评测（双链）")
    LOGGER.info("  sync_audit 链样例: %d（缺陷 %d + 正确 %d）", len(LABELS), n_def, n_ok)
    LOGGER.info("")
    result = run_sync_regression()
    print_report(result)
    LOGGER.info("")
    flow_failed = run_flow_regression()
    return 1 if (result.correct_missed or result.defect_missed or flow_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())