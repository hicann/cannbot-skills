# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Evidence parser - rule-based parsing"""

from typing import Optional

from llm.client import LLMClient

from ..core.models import (
    ParsedResult,
    RawEvidenceChainInput,
)
from .extractor import EvidenceExtractor


class EvidenceParser:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.extractor = EvidenceExtractor(llm_client)

    @staticmethod
    def _get_target_groups(target):
        if target is None:
            return [None]
        if isinstance(target, str):
            return [target]
        if isinstance(target, list):
            groups = []
            for t in target:
                if isinstance(t, list) and len(t) > 0:
                    groups.append(t[0])
                elif isinstance(t, str):
                    groups.append(t)
            if groups:
                return groups
        return [str(target)]

    def parse_raw_input(self, raw_input: RawEvidenceChainInput, deduplicate: bool = False) -> ParsedResult:
        raw_evidences = self.extractor.extract_raw_evidences(raw_input, deduplicate=deduplicate)
        doc_original_parsed = raw_input.doc.sub_part
        doc_original_md = raw_input.doc.md_format
        doc_original_arch = raw_input.doc.arch
        doc_original_arch_sources = raw_input.doc.arch_sources
        doc_original_chip = raw_input.doc.chip
        doc_original_chip_sources = raw_input.doc.chip_sources

        evidences = []
        evidences_by_target = {}

        for ev in raw_evidences:
            target = ev.gene_rule.get("target") if ev.gene_rule else None
            target_groups = self._get_target_groups(target)

            ev_data = {
                "evidence_id": ev.evidence_id,
                "type": ev.source,
                "target": target,
                "content": ev.content,
                "support_chips": ev.support_chips or {},
            }
            evidences.append(ev_data)

            group_data = {k: v for k, v in ev_data.items() if k != "target"}
            for target_group in target_groups:
                evidences_by_target.setdefault(target_group, []).append(group_data.copy())

        return ParsedResult(
            doc_part=raw_input.doc.doc_part,
            evidences=evidences,
            doc_original_parsed=doc_original_parsed,
            doc_original_md=doc_original_md,
            doc_original_arch=doc_original_arch,
            doc_original_arch_sources=doc_original_arch_sources,
            doc_original_chip=doc_original_chip,
            doc_original_chip_sources=doc_original_chip_sources,
            total_evidences=len(evidences),
            evidences_by_target=evidences_by_target,
        )
