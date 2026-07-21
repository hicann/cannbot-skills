#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Session性能对比工具

用法:
    python compare_sessions.py <session_id_1> <session_id_2>
    python compare_sessions.py --op fastgelu --top 5
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_engine"))
from session_manager import SessionManager

logger = logging.getLogger(__name__)


def format_duration(us: float) -> str:
    """格式化时间"""
    if us < 1000:
        return f"{us:.2f} us"
    elif us < 1_000_000:
        return f"{us/1000:.2f} ms"
    else:
        return f"{us/1_000_000:.2f} s"


def compare_two_sessions(sm: SessionManager, session_id_1: str, session_id_2: str):
    """对比两个sessions"""
    s1 = sm.load_session(session_id_1)
    s2 = sm.load_session(session_id_2)

    if not s1 or not s2:
        logger.info(f"❌ One or both sessions not found")
        return

    logger.info("=" * 80)
    logger.info(f"Session Performance Comparison")
    logger.info("=" * 80)
    logger.info("")

    # 基本信息
    logger.info(f"Session 1: {s1.session_id}")
    logger.info(f"  Operator: {s1.operator}")
    logger.info(f"  Created: {s1.created_at}")
    logger.info(f"  Status: {s1.status}")
    logger.info("")

    logger.info(f"Session 2: {s2.session_id}")
    logger.info(f"  Operator: {s2.operator}")
    logger.info(f"  Created: {s2.created_at}")
    logger.info(f"  Status: {s2.status}")
    logger.info("")

    # 性能对比
    logger.info("-" * 80)
    logger.info("Performance Metrics")
    logger.info("-" * 80)

    perf1 = s1.performance
    perf2 = s2.performance

    # Initial duration
    dur1_init = perf1.get("initial_duration_us", 0)
    dur2_init = perf2.get("initial_duration_us", 0)

    # Current duration
    dur1_curr = perf1.get("current_duration_us", 0)
    dur2_curr = perf2.get("current_duration_us", 0)

    # Improvement
    imp1 = perf1.get("improvement_pct", 0)
    imp2 = perf2.get("improvement_pct", 0)

    logger.info(f"{'Metric':<30} {'Session 1':<20} {'Session 2':<20} {'Diff'}")
    logger.info(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")

    if dur1_init > 0 and dur2_init > 0:
        logger.info(f"{'Initial Duration':<30} {format_duration(dur1_init):<20} {format_duration(dur2_init):<20}")

    if dur1_curr > 0 and dur2_curr > 0:
        print(f"{'Current Duration':<30} {format_duration(dur1_curr):<20} {format_duration(dur2_curr):<20}", end="")
        if dur1_curr != dur2_curr:
            diff_pct = ((dur2_curr - dur1_curr) / dur1_curr) * 100
            symbol = "⬇️" if diff_pct < 0 else "⬆️"
            print(f" {symbol} {abs(diff_pct):.1f}%")
        else:
            print(" Same")

    print(f"{'Improvement':<30} {imp1:.1f}%{' (' + ('✅' if perf1.get('target_met') else '❌') + ')':<20} {imp2:.1f}%{' (' + ('✅' if perf2.get('target_met') else '❌') + ')':<20}", end="")
    if imp1 != imp2:
        diff = imp2 - imp1
        symbol = "⬆️" if diff > 0 else "⬇️"
        print(f" {symbol} {abs(diff):.1f}%")
    else:
        print(" Same")

    # Iterations
    iter1 = s1.iterations.get("total", 0)
    iter2 = s2.iterations.get("total", 0)
    logger.info(f"{'Iterations':<30} {iter1:<20} {iter2:<20}")

    # Winner
    logger.info("")
    logger.info("-" * 80)
    if dur1_curr > 0 and dur2_curr > 0:
        if dur1_curr < dur2_curr:
            logger.info(f"🏆 Winner: Session 1 (faster by {((dur2_curr - dur1_curr) / dur2_curr * 100):.1f}%)")
        elif dur2_curr < dur1_curr:
            logger.info(f"🏆 Winner: Session 2 (faster by {((dur1_curr - dur2_curr) / dur1_curr * 100):.1f}%)")
        else:
            logger.info(f"🤝 Tie: Both sessions have same performance")
    else:
        if imp1 > imp2:
            logger.info(f"🏆 Winner: Session 1 (better improvement: {imp1:.1f}% vs {imp2:.1f}%)")
        elif imp2 > imp1:
            logger.info(f"🏆 Winner: Session 2 (better improvement: {imp2:.1f}% vs {imp1:.1f}%)")
        else:
            logger.info(f"🤝 Tie: Both sessions have same improvement")


def compare_operator_sessions(sm: SessionManager, op_name: str, top: int = 10):
    """对比同一算子的多个sessions"""
    sessions = sm.list_sessions(op_name=op_name, limit=top)

    if not sessions:
        logger.info(f"❌ No sessions found for operator: {op_name}")
        return

    logger.info("=" * 100)
    logger.info(f"Performance Comparison for Operator: {op_name}")
    logger.info("=" * 100)
    logger.info("")

    # 表头
    logger.info(f"{'#':<4} {'Session ID':<40} {'Status':<12} {'Duration':<15} {'Improvement':<12} {'Target':<8}")
    logger.info(f"{'-'*4} {'-'*40} {'-'*12} {'-'*15} {'-'*12} {'-'*8}")

    # 排序: 按improvement降序
    sessions_sorted = sorted(sessions, key=lambda s: s.performance.get("improvement_pct", 0), reverse=True)

    for i, s in enumerate(sessions_sorted[:top], 1):
        perf = s.performance
        dur = perf.get("current_duration_us", 0)
        imp = perf.get("improvement_pct", 0)
        met = perf.get("target_met", False)

        status_icon = {
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "interrupted": "⏸️"
        }.get(s.status, "❓")

        target_icon = "✅" if met else "❌"

        logger.info(
            f"{i:<4} {s.session_id:<40} {status_icon} {s.status:<11} {format_duration(dur):<15} {imp:>6.1f}% {' '*4} {target_icon}")

    # 统计
    logger.info("")
    logger.info("-" * 100)
    logger.info("Statistics:")
    completed = [s for s in sessions if s.status == "completed"]
    if completed:
        improvements = [s.performance.get("improvement_pct", 0) for s in completed]
        best = max(improvements)
        worst = min(improvements)
        avg = sum(improvements) / len(improvements)

        logger.info(f"  Best improvement: {best:.1f}%")
        logger.info(f"  Worst improvement: {worst:.1f}%")
        logger.info(f"  Average improvement: {avg:.1f}%")
        logger.info(
            f"  Success rate: {len([s for s in completed if s.performance.get('target_met', False)]) / len(completed) * 100:.1f}%")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Session性能对比工具")

    # 模式1: 对比两个特定sessions
    parser.add_argument("session_ids", nargs="*", help="Session IDs to compare (2 IDs)")

    # 模式2: 对比算子的所有sessions
    parser.add_argument("--op", help="Operator name (compare all sessions)")
    parser.add_argument("--top", type=int, default=10, help="Top N sessions to show")

    args = parser.parse_args()

    workspace = Path("workspace")
    sm = SessionManager(workspace)

    if args.op:
        # 模式2: 对比算子sessions
        compare_operator_sessions(sm, args.op, args.top)
    elif len(args.session_ids) == 2:
        # 模式1: 对比两个sessions
        compare_two_sessions(sm, args.session_ids[0], args.session_ids[1])
    else:
        logger.info("Usage:")
        logger.info("  1. Compare two sessions:")
        logger.info("     python compare_sessions.py <session_id_1> <session_id_2>")
        logger.info("")
        logger.info("  2. Compare all sessions for an operator:")
        logger.info("     python compare_sessions.py --op <operator_name> [--top N]")
        sys.exit(1)


if __name__ == "__main__":
    main()
