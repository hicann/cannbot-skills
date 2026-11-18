# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Document validator - main entry for validation"""

from typing import Any, Optional

from llm.client import LLMClient

from ..core.models import Evidence, ValidationResult
from .consistency import ConsistencyChecker
from .metric import ModificationMetric


class DocumentValidator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        self.consistency = ConsistencyChecker(llm_client)
        self.metric = ModificationMetric()

    def validate(
        self, doc_original: str, evidences, check_metric: bool = False, doc_part: str = "", **kwargs
    ) -> ValidationResult:
        if check_metric:
            doc_generated = kwargs.get("doc_generated", "None")
            evidences_extracted = kwargs.get("evidences_extracted", {})
            # 1. Consistency use doc_generated
            consistency_result = self.consistency.check(doc_generated, evidences, doc_part=doc_part)
            # 2. Modification Metric
            metric_result = self.metric.check(doc_original, doc_generated, evidences_extracted)
        else:
            # 1. Consistency use doc_original
            consistency_result = self.consistency.check(doc_original, evidences, doc_part=doc_part)
            # 2. PASS Modification Metric
            metric_result = {"result": "N/A", "suggestion": "None", "edit_distance_ratio": 0.0, "avg_confidence": 0.0}

        return ValidationResult(
            consistency_passed=consistency_result.get("passed", True),
            accuracy_rate=consistency_result.get("accuracy_rate", 1.0),
            consistency_scores=consistency_result.get("consistency_scores", {}),
            total_evidence=consistency_result.get("total_evidence", 0),
            total_batches=consistency_result.get("total_batches", 0),
            issues=consistency_result.get("issues", []),
            metric_passed=metric_result.get("result", "N/A"),
            edit_distance_ratio=metric_result.get("edit_distance_ratio", 0.0),
            avg_confidence=metric_result.get("avg_confidence", 0.0),
            suggestion=metric_result.get("suggestion", "None"),
        )
