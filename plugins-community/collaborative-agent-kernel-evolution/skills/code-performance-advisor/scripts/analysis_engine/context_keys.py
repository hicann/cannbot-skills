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
Context key constants for workflow.py state machine.

单一真相来源（Single Source of Truth）：
所有 workflow 阶段之间通过 WorkflowState.context 字典传递数据，
本模块集中定义全部 key 常量，消除分散的字符串字面量，防止拼写错误。

使用方式:
    from context_keys import CK
    self.state.context[CK.BASELINE_DURATION_US] = value
    value = self.state.context.get(CK.PROFILING_DATA)
"""


class CK:
    """WorkflowState.context 字典的所有 key 常量"""

    # ── INIT 阶段产出 ──────────────────────────────────────────────
    # ProfilingData.to_dict() 结果，包含 task_type / raw_metrics / relevant_metrics 等
    PROFILING_DATA = "profiling_data"
    # 基准 Task Duration（us），用于 COMPARE 阶段计算改善幅度
    BASELINE_DURATION_US = "baseline_duration_us"
    # 基准 CSV 文件的 mtime（秒），用于 COMPARE 阶段过滤"新于 baseline"的 profiling
    BASELINE_CSV_MTIME = "baseline_csv_mtime"
    # workspace/inputs/{op}/ 的绝对路径字符串
    INPUT_DIR = "input_dir"
    # 是否存在 profiling 数据目录（bool）
    HAS_PROFILING = "has_profiling"
    # 当前 session 目录绝对路径字符串
    SESSION_DIR = "session_dir"
    # 当前 session 工作副本代码目录绝对路径字符串
    WORKING_CODE_DIR = "working_code_dir"
    # 当前 session 不可变 baseline 快照目录绝对路径字符串
    BASELINE_SNAPSHOT_DIR = "baseline_snapshot_dir"

    # ── TAG 阶段产出 ───────────────────────────────────────────────
    # 当前使用的 tag JSON 文件的绝对路径字符串
    TAG_FILE = "tag_file"
    # TAG 阶段用于判断缓存有效性的代码目录绝对路径字符串
    TAG_SOURCE_CODE_DIR = "tag_source_code_dir"
    # 是否在下一次进入 TAG 时强制重新生成 tag（bool，消费后重置为 False）
    FORCE_RETAG = "force_retag"

    # ── SCORE 阶段产出 ─────────────────────────────────────────────
    # cli.py score 输出的完整 JSON 对象（含 results 列表）
    SCORED_RESULTS = "scored_results"

    # ── ROUTE 阶段产出 ─────────────────────────────────────────────
    # RouteDecision.asdict() 结果，含 path / max_score / top_rule_id / confidence
    ROUTE = "route"

    # ── SUGGEST 阶段产出 ───────────────────────────────────────────
    # 代码在 SUGGEST 暂停前的 MD5 哈希（8 位十六进制），供 APPLY 验证代码是否被修改
    PRE_APPLY_CODE_HASH = "pre_apply_code_hash"
    # suggestions 目录中第一个 .md 文件的绝对路径字符串
    SUGGESTION_FILE = "suggestion_file"

    # ── APPLY 阶段产出 ─────────────────────────────────────────────
    # 代码在 APPLY 成功后的 MD5 哈希（8 位十六进制），供 COMPARE 死循环检测
    POST_APPLY_CODE_HASH = "post_apply_code_hash"

    # ── COMPARE 阶段产出 ───────────────────────────────────────────
    # 当前优化轮次（int），用于防止超过 MAX_ROUNDS 的死循环
    OPTIMIZATION_ROUNDS = "optimization_rounds"
    # 用户设置的性能目标（dict，含 relative_improvement 等字段）
    PERFORMANCE_GOAL = "performance_goal"
    # 最新一次 EVALUATE 后的 Task Duration（us）
    LAST_EVAL_DURATION_US = "last_eval_duration_us"
    # 最新一次 EVALUATE 后的改善幅度（float，正值为改善，负值为退步）
    LAST_EVAL_IMPROVEMENT = "last_eval_improvement"
    # COMPARE 判断结果：是否满足目标改善幅度（bool）
    PERFORMANCE_IMPROVED = "performance_improved"

    # ── 错误处理 ───────────────────────────────────────────────────
    # _handle_error() 写入的错误信息 dict，含 type / message / traceback / phase
    ERROR = "error"
