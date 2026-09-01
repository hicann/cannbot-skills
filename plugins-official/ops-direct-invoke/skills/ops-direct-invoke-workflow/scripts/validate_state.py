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
state.json 校验脚本（恢复 #801 引入的工作流状态校验特性，适配 skill 驱动工作流）

功能：
1. 检查 JSON 是否可解析（禁止 // 注释）
2. 检查必填键是否齐全（operator / current_stage / completed_stages）
3. 检查编号合法性（current_stage 与 completed_stages 的编号 ∈ 基类流程表编号，或插件步骤编号 plugin-*-N）
4. 检查 completed_stages 与 current_stage 的顺序一致性（基类编号保持流程表相对顺序；current_stage
   不得已出现在 completed_stages）
5. 检查 blocked / rounds / pending_questionnaire / pending_user_review / deliverables 的结构合法性

用法：
python3 validate_state.py .cannbot/<算子名>/state.json

返回：
- 0: 校验通过（⚠ 警告不影响退出码）
- 1: 校验失败
"""

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("validate_state")


# 基类统一流程表编号（按流程表顺序；阶段 2/3 的方案线与测试线按表内顺序排列）
CANONICAL_STAGES = [
    "0", "CP0",  # 阶段0 开发准备
    "1.1", "CP1",  # 阶段1 需求分析
    "2.1", "CP2.1", "2.2", "CP2.2",  # 阶段2 方案设计（测试线/方案线）
    "3.1", "3.2", "3.3", "3.4", "CP3",  # 阶段3 代码开发（开发线/测试线/联调）
    "4.1", "CP4",  # 阶段4 性能验收
    "CP5",  # 阶段5 代码检视
    "6.1",  # 阶段6 上库准备
    "7.1", "7.2",  # 阶段7 开发总结
]
STAGE_SET = set(CANONICAL_STAGES)
# 可插拔流程插件的内部步骤编号，如 plugin-pr-submit-1 / plugin-perf-iteration-2
PLUGIN_STAGE_RE = re.compile(r"^plugin-[a-z0-9]+(-[a-z0-9]+)*-[0-9]+$")
# blocked.loop 取值：design / joint_debug / acceptance 对应 error-handling.md 轮次表；ci 属上库插件
LOOP_VALUES = {"design", "joint_debug", "acceptance", "ci"}
QUESTIONNAIRE_STATUS = {"sent", "answered"}

REQUIRED_TOP = {"operator", "current_stage", "completed_stages"}
REVIEW_KEYS = ("at", "decision", "scope", "evidence_path", "decided_by", "decided_at")


def is_valid_stage(stage_id):
    """编号合法：基类流程表编号或插件步骤编号"""
    if not isinstance(stage_id, str) or not stage_id:
        return False
    return stage_id in STAGE_SET or bool(PLUGIN_STAGE_RE.match(stage_id))


def is_positive_int(value):
    """正整数（bool 不算）"""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_order(state, errors, warnings):
    """completed_stages 与 current_stage 的顺序一致性"""
    completed = state.get("completed_stages", [])
    current = state.get("current_stage")

    # 基类编号保持流程表相对顺序（插件编号位置由插件自身定义，不参与排序校验）
    canonical_index = {s: i for i, s in enumerate(CANONICAL_STAGES)}
    positions = [canonical_index[s] for s in completed if s in STAGE_SET]
    if positions != sorted(positions):
        base_seq = [s for s in completed if s in STAGE_SET]
        errors.append("completed_stages 中基类编号未按流程表顺序排列: " + " -> ".join(base_seq))

    # 重复编号
    if len(set(completed)) != len(completed):
        duplicates = sorted({s for s in completed if completed.count(s) > 1})
        errors.append("completed_stages 存在重复编号: " + ", ".join(duplicates))

    # current_stage 不得已出现在 completed_stages（收尾态放宽为警告）
    if current in completed:
        warnings.append(
            f"current_stage={current} 已出现在 completed_stages（收尾态可接受，进行中状态属于异常）"
        )


def check_blocked(blocked, errors):
    """blocked 字段结构：at / reason / round / loop"""
    if not isinstance(blocked, dict):
        errors.append("blocked 必须是对象")
        return
    if not is_valid_stage(blocked.get("at")):
        errors.append(f"blocked.at 不是合法编号: {blocked.get('at')!r}")
    if not isinstance(blocked.get("reason"), str) or not blocked.get("reason"):
        errors.append("blocked.reason 必须是非空字符串")
    if not is_positive_int(blocked.get("round")):
        errors.append("blocked.round 必须是 >=1 的整数")
    if blocked.get("loop") not in LOOP_VALUES:
        allowed = sorted(LOOP_VALUES)
        errors.append(f"blocked.loop 取值必须是 {allowed} 之一，当前: {blocked.get('loop')!r}")


def check_rounds(rounds, errors):
    """rounds 字段结构：编号 -> 正整数"""
    valid = isinstance(rounds, dict) and all(
        is_valid_stage(k) and is_positive_int(v) for k, v in rounds.items()
    )
    if not valid:
        errors.append("rounds 必须是 编号->正整数 的映射")


def check_questionnaire(pq, errors):
    """pending_questionnaire 字段结构：cp / path / reply_path / status"""
    if not isinstance(pq, dict):
        errors.append("pending_questionnaire 必须是对象")
        return
    if not is_valid_stage(pq.get("cp")):
        errors.append(f"pending_questionnaire.cp 不是合法编号: {pq.get('cp')!r}")
    for key in ("path", "reply_path"):
        if not isinstance(pq.get(key), str) or not pq.get(key):
            errors.append(f"pending_questionnaire.{key} 必须是非空字符串")
    if pq.get("status") not in QUESTIONNAIRE_STATUS:
        allowed = sorted(QUESTIONNAIRE_STATUS)
        errors.append(f"pending_questionnaire.status 取值必须是 {allowed} 之一，当前: {pq.get('status')!r}")


def check_review_item(item, index, errors):
    """pending_user_review 单条结构"""
    if not isinstance(item, dict):
        errors.append(f"pending_user_review[{index}] 必须是对象")
        return
    for key in REVIEW_KEYS:
        if not isinstance(item.get(key), str) or not item.get(key):
            errors.append(f"pending_user_review[{index}].{key} 必须是非空字符串")


def check_deliverables(deliverables, errors):
    """deliverables 字段结构：非空字符串 -> 非空字符串"""
    valid = isinstance(deliverables, dict) and all(
        isinstance(k, str) and k and isinstance(v, str) and v
        for k, v in deliverables.items()
    )
    if not valid:
        errors.append("deliverables 必须是 非空字符串->非空字符串 的映射")


def check_structures(state, errors):
    """可选字段的结构合法性（存在时才校验）"""
    if "chip" in state and not isinstance(state["chip"], str):
        errors.append("chip 必须是字符串")
    if "updated_at" in state and (not isinstance(state["updated_at"], str) or not state["updated_at"]):
        errors.append("updated_at 必须是非空字符串（ISO8601）")
    if state.get("blocked") is not None:
        check_blocked(state["blocked"], errors)
    if state.get("rounds") is not None:
        check_rounds(state["rounds"], errors)
    if state.get("pending_questionnaire") is not None:
        check_questionnaire(state["pending_questionnaire"], errors)
    reviews = state.get("pending_user_review")
    if reviews is not None:
        if not isinstance(reviews, list):
            errors.append("pending_user_review 必须是数组")
        else:
            for i, item in enumerate(reviews):
                check_review_item(item, i, errors)
    if state.get("deliverables") is not None:
        check_deliverables(state["deliverables"], errors)


def check_required_keys(data, errors):
    """必填键：operator / current_stage / completed_stages"""
    for key in sorted(REQUIRED_TOP - set(data.keys())):
        errors.append(f"缺少必填键: {key}")
    if isinstance(data.get("operator"), str) and not data["operator"].strip():
        errors.append("operator 必须是非空字符串")


def check_stage_ids(data, errors):
    """current_stage 与 completed_stages 的编号合法性，返回 completed_stages 原值"""
    if "current_stage" in data and not is_valid_stage(data.get("current_stage")):
        errors.append(
            f"current_stage 不是合法编号: {data.get('current_stage')!r}（基类流程表编号或 plugin-*-N）"
        )
    completed = data.get("completed_stages")
    if "completed_stages" not in data:
        return completed
    if not isinstance(completed, list):
        errors.append("completed_stages 必须是数组")
        return completed
    for stage in completed:
        if not is_valid_stage(stage):
            errors.append(f"completed_stages 含非法编号: {stage!r}")
    return completed


def validate(state_path):
    """校验 state.json，返回 (是否通过, 消息列表)"""
    errors, warnings = [], []
    p = Path(state_path)
    if not p.is_file():
        return False, [f"❌ state.json 不存在: {state_path}"]

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"❌ JSON 解析失败: {e}"]

    if not isinstance(data, dict):
        return False, ["❌ state.json 顶层必须是 JSON 对象"]

    check_required_keys(data, errors)
    completed = check_stage_ids(data, errors)
    if isinstance(completed, list):
        check_order(data, errors, warnings)
    check_structures(data, errors)

    messages = [f"❌ {msg}" if not msg.startswith("❌") else msg for msg in errors]
    messages += [f"⚠ {msg}" for msg in warnings]
    if not errors:
        summary = f"✅ 校验通过: {state_path}"
        if warnings:
            summary += f"（{len(warnings)} 条警告）"
        messages.append(summary)
        return True, messages
    messages.insert(0, f"❌ 校验失败: {state_path}（{len(errors)} 项错误）")
    return False, messages


def main():
    if len(sys.argv) != 2:
        LOGGER.info(__doc__)
        return 1
    if sys.argv[1] in ("-h", "--help"):
        LOGGER.info(__doc__)
        return 0
    ok, messages = validate(sys.argv[1])
    for msg in messages:
        LOGGER.info(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
