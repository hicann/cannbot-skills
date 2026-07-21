# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Utility modules for Code Performance Advisor."""

from .goal_loader import GoalConfig, load_goal, create_default_goal_md
from .rule_parser import RuleParser, RuleInfo, parse_rule, load_code_snippet
from .code_matcher import CodeMatcher, CodeMatch, PatternMatchResult, analyze_code, find_optimization_opportunities
from .evidence_extractor import EvidenceExtractor, Evidence, ProfilingMetrics, extract_evidence

__all__ = [
    "GoalConfig",
    "load_goal",
    "create_default_goal_md",
    "RuleParser",
    "RuleInfo",
    "parse_rule",
    "load_code_snippet",
    "CodeMatcher",
    "CodeMatch",
    "PatternMatchResult",
    "analyze_code",
    "find_optimization_opportunities",
    "EvidenceExtractor",
    "Evidence",
    "ProfilingMetrics",
    "extract_evidence"
]
