# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Document generator - main entry for document generation"""

import json
from typing import List, Optional

from rich.console import Console

from llm.client import LLMClient
from src.logger import debug, info, log_result

from ..core.models import (
    ModificationResult,
    StyleExtraction,
    StyleSummary,
)
from ..dashboard import dashboard_log, get_dashboard
from ..style.loader import TemplateLoader
from .writer import TextWriter


class DocumentGenerator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        template_loader: Optional[TemplateLoader] = None,
    ):
        self.writer = TextWriter(llm_client, template_loader)

    @staticmethod
    def _clean_evidences(evidences: dict) -> dict:
        """Remove category field from evidences dict"""
        if not isinstance(evidences, dict):
            return evidences

        cleaned = {}
        for field_name, items in evidences.items():
            if isinstance(items, dict):
                cleaned[field_name] = {}
                for sub_field, sub_items in items.items():
                    if isinstance(sub_items, list):
                        cleaned[field_name][sub_field] = [
                            {k: v for k, v in item.items() if k != "category"}
                            for item in sub_items
                            if isinstance(item, dict)
                        ]
                    else:
                        cleaned[field_name][sub_field] = sub_items
            elif isinstance(items, list):
                cleaned[field_name] = [
                    {k: v for k, v in item.items() if k != "category"}
                    for item in items
                    if isinstance(item, dict)
                ]
            else:
                cleaned[field_name] = items
        return cleaned

    def generate(
        self,
        original_content: dict,
        evidences: dict = None,
        style_summary: StyleSummary = None,
        style_extraction: StyleExtraction = None,
        doc_part: str = "",
        sub_key: str = "",
        previous_issues: list = None,
        generated_doc: str = "",
        passed_indices: list = None,
        change_reason: Optional[dict] = None,
    ) -> List[ModificationResult]:

        doc_original_md_str = original_content.get("md", "")
        console = Console()

        if get_dashboard():
            dashboard_log("  > Generating document sections")
        else:
            console.print("  > Generating document sections")
        cleaned_evidences = self._clean_evidences(evidences) if evidences else None
        result = self.writer.generate(
            original=doc_original_md_str,
            evidences=cleaned_evidences,
            style_summary=style_summary,
            style_extraction=style_extraction,
            doc_part=doc_part,
            sub_key=sub_key,
            previous_issues=previous_issues,
            generated_doc=generated_doc,
            change_reason=change_reason,
        )

        changes = result.modification_summary.get("changes", [])
        analysis = result.modification_summary.get("analysis", {})
        result.modification_summary = {
            "sub_key": sub_key,
            "changes_identified": len(changes),
            "evidence_used": len(changes),
            "analysis": analysis,
            "changes": changes,
        }

        log_result(f"Result")
        debug(f"[modification_summary]:\n{json.dumps(result.modification_summary, ensure_ascii=False, indent=2)}")
        log_result(f"[changes]:\n{json.dumps(changes, ensure_ascii=False, indent=2)}")
        log_result(f"[generated_text]:\n```markdown\n{result.generated_text}\n```\n")

        return [result]
