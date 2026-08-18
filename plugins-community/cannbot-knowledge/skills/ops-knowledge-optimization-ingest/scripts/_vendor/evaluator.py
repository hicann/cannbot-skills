# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Evaluator wrapper around the local msprof evaluation client.

A thin object that holds operator identity and forwards an operator working-copy
directory to ``eval_client.evaluate_op_local`` (local build + msprof collection).
The skill only needs the single-shot ``evaluate`` entry point.
"""

from typing import Any, Dict, Optional, Tuple

from . import eval_client, msprof_collect


class KernelEvaluator:
    """本地 msprof 采集：对单个算子工作副本返回完整评估结果。"""

    def __init__(
        self,
        log_dir: str,
        op_name: str,
        op_category: str,
    ):
        self.log_dir = log_dir
        self.op_name = op_name
        self.op_category = op_category

    @staticmethod
    def evaluate(
        op_dir: str,
        op_name: str,
        *,
        build_cmd=None,
        run_cmd=None,
        full: bool = False,
        quick: bool = False,
        device: Optional[int] = None,
        metrics: Tuple[str, ...] = msprof_collect.METRICS_DEFAULT,
        workdir: Optional[str] = None,
        kernel_match: Optional[str] = None,
    ) -> Dict[str, Any]:
        """本地编译 + msprof 采集，返回完整结果（含 score 和 performance_metrics）。"""
        return eval_client.evaluate_op_local(
            op_dir,
            op_name,
            build_cmd=build_cmd,
            run_cmd=run_cmd,
            metrics=metrics,
            full=full,
            quick=quick,
            device=device,
            workdir=workdir,
            kernel_match=kernel_match,
        )

    def evaluate_score(self, op_dir: str, op_name: str, **kwargs) -> float:
        """只取分数（失败返回 -10000）。"""
        result = self.evaluate(op_dir, op_name, **kwargs)
        return float(result.get("score", eval_client.FAIL_SCORE))
