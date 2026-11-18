# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Evidence extractor - extract structured data from evidences by doc_part"""

import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import yaml
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn  # noqa: E402
from rich.table import Table  # noqa: E402

from src.logger import log_result

from .. import config
from ..core.models import (
    DocPart,
    RawEvidence,
    RawEvidenceChainInput,
)
from ..dashboard import dashboard_log, get_current_doc_part, get_dashboard, set_dashboard_context
from ..evidence.chip_mapper import ChipMapper
from ..evidence.ranker import TARGET_PREFERENCE
from ..evidence.support_verifier import SupportVerifier

console = Console(force_terminal=True)


class EvidenceExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._console_lock = threading.Lock()

    @staticmethod
    def _deduplicate_anchors_by_file(anchors: list) -> list:
        file_paths = set()
        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            if anchor_info.get("type") == "file":
                fp = anchor_info.get("level1", {}).get("file_path")
                if fp:
                    file_paths.add(fp)
        filtered = []
        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            anchor_type = anchor_info.get("type", "unknown")
            fp = anchor_info.get("level1", {}).get("file_path")
            if fp and fp in file_paths and anchor_type != "file":
                continue
            filtered.append(anchor)
        return filtered

    @staticmethod
    def extract_raw_evidences(raw_input: RawEvidenceChainInput, deduplicate: bool = False) -> list[RawEvidence]:
        """Extract raw evidences from anchors"""
        evidences = []

        anchors = raw_input.anchor_tuple
        if deduplicate:
            anchors = EvidenceExtractor._deduplicate_anchors_by_file(anchors)

        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            anchor_type = anchor_info.get("type", "unknown")
            for ev in anchor.evidence:
                content = (ev.evidence_information or {}).copy()
                evidences.append(
                    RawEvidence(
                        evidence_id=ev.get_evidence_id(),
                        source=anchor_type,
                        content=content,
                        gene_rule=ev.gene_rule,
                        support_chips=ev.support_chips,
                    )
                )

        return evidences

    @staticmethod
    def _merge_and_deduplicate(list1: list, list2: list) -> list:
        """Merge two lists and deduplicate by evidence_id"""
        seen = set()
        result = []
        for item in list1 + list2:
            refs = item.get("evidence_id", [])
            ref_key = tuple(sorted(refs)) if isinstance(refs, list) else tuple(refs) if refs else ()
            if ref_key not in seen:
                seen.add(ref_key)
                result.append(item)
        return result

    @staticmethod
    def _prepare_evidences_content(ev_list: list[dict]) -> list[dict]:
        """Prepare evidences content by filtering and adding evidence_id"""
        contents = []
        for ev in ev_list:
            evidence_id = ev.get("evidence_id", "")
            content = ev.get("content", {})
            ev_type = ev.get("type", "")

            if isinstance(content, dict):
                if ev_type == "function":
                    content = {k: v for k, v in content.items() if k not in ["call", "callee", "return"]}
                    if "content" in content and isinstance(content["content"], list):
                        content["content"] = [
                            {k: v for k, v in item.items() if k != "full_function"}
                            for item in content["content"]
                        ]
                elif ev_type in ("note", "section"):
                    content = {k: v for k, v in content.items() if k not in ["start_line", "end_line", "token_hash"]}
                elif ev_type == "file":
                    # file 类型证据的 file_content 是头文件全文（动辄数万字符），
                    # 原样塞进 LLM prompt 会触发 token 截断或噪声淹没，导致算子声明
                    # （常位于文件靠后位置）被切掉，校验误报"证据中无算子声明"。
                    # 改为提取关键代码段（函数声明/doxygen/static_assert/条件编译）。
                    fc = content.get("file_content", "")
                    if isinstance(fc, str) and len(fc) > 4000:
                        content = dict(content)
                        content["file_content"] = EvidenceExtractor._summarize_file_content(fc)
                content["evidence_id"] = evidence_id
                contents.append(content)
            else:
                contents.append({"evidence_id": evidence_id, "content": content})
        return contents

    @staticmethod
    def _summarize_file_content(content: str, max_chars: int = 30000) -> str:
        """从文件全文提取关键代码段，避免 file 证据全文（数万字符）淹没 LLM。

        提取内容：
          - doxygen 注释块（含 @brief/@param/@return/@ingroup/deprecated）
          - inline 函数声明（收集到 ';' 结束，含重载/参数，即算子声明本体）
          - static_assert 约束
          - #error / #if __NPU_ARCH__ 条件编译

        这样 reg_vector.h 的 8 万字符全文 → 几十条函数声明清单（含目标算子声明），
        LLM 能精准定位算子声明，不再因全文截断/噪声而漏看。
        """
        lines = content.splitlines()
        kept: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()

            # doxygen 注释块 /** ... */
            if stripped.startswith("/*"):
                block = [line]
                if "*/" in line:
                    j = i + 1
                else:
                    j = i + 1
                    while j < len(lines) and "*/" not in lines[j]:
                        block.append(lines[j].rstrip())
                        j += 1
                    if j < len(lines):
                        block.append(lines[j].rstrip())
                        j += 1
                block_text = "\n".join(block)
                # 仅保留有信息量的注释（brief/param/return/deprecated 等）
                if any(
                    t in block_text
                    for t in (
                        "@brief",
                        "@param",
                        "@return",
                        "@ingroup",
                        "deprecated",
                        "@deprecated",
                    )
                ):
                    kept.append(block_text)
                i = j
                continue

            # inline 函数声明：含 inline 且有 '('，收集到 ';' 结束
            if "inline" in line and "(" in line:
                sig = [line]
                j = i + 1
                while j < len(lines) and ";" not in sig[-1]:
                    sig.append(lines[j].rstrip())
                    j += 1
                    if j - i > 15:  # 防失控
                        break
                kept.append("\n".join(sig))
                i = j
                continue

            # static_assert / #error / __NPU_ARCH__ 条件编译
            if "static_assert" in stripped:
                kept.append(line)
            elif stripped.startswith("#error"):
                kept.append(line)
            elif "__NPU_ARCH__" in stripped and stripped.startswith("#"):
                kept.append(line)

            i += 1

        summary = "\n\n".join(kept)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + f"\n...[file_content 已摘要化，原文 {len(content)} 字符]"
        return summary

    @staticmethod
    def _get_target_category(target: str) -> str:
        """Get category for target from ranker config"""
        return TARGET_PREFERENCE.get(target, {}).get("category", "fact")

    @staticmethod
    def _group_evidences_by_confidence(evidences_by_target: dict, modified_file_basenames: set = None) -> dict:
        """Group evidences by confidence level.

        Modified files get priority layer (1.0) to ensure they're extracted first.
        """
        from pathlib import Path

        grouped = {}
        for target_key, ev_list in evidences_by_target.items():
            groups = {1.0: [], 0.9: [], 0.8: [], 0.7: [], 0.6: [], 0.5: []}
            for ev in ev_list:
                conf = ev.get("confidence", 0.5)

                # Check if this evidence is from a modified file
                is_modified = False
                if modified_file_basenames:
                    file_path = ""
                    content = ev.get("content", {})
                    if isinstance(content, dict):
                        file_path = content.get("file_path", "")
                    elif isinstance(content, list) and content:
                        if isinstance(content[0], dict):
                            file_path = content[0].get("file_path", "")

                    if file_path and Path(file_path).name in modified_file_basenames:
                        is_modified = True

                if is_modified:
                    groups[1.0].append(ev)
                elif conf >= 0.85:
                    groups[0.9].append(ev)
                elif conf >= 0.75:
                    groups[0.8].append(ev)
                elif conf >= 0.65:
                    groups[0.7].append(ev)
                elif conf >= 0.55:
                    groups[0.6].append(ev)
                else:
                    groups[0.5].append(ev)
            grouped[target_key] = [groups[1.0], groups[0.9], groups[0.8], groups[0.7], groups[0.6], groups[0.5]]
        return grouped

    @staticmethod
    def _add_metadata_to_extraction(extraction: dict, category: str, confidence: float) -> None:
        """Add category, confidence and source to each item in extraction"""
        for _, items in extraction.items():
            for item in items:
                if isinstance(item, dict):
                    item["category"] = category
                    item["confidence"] = confidence

    @staticmethod
    def _base_progressive_prompt(
        task_content: str,
        content: str,
        doc_original: str,
        category_instruction: str,
        return_fields: str,
    ) -> str:
        """Base prompt template for progressive extraction"""
        return f"""# 任务
从以下证据中提取新版资料书写所需内容，并对自己现有证据的解析结果进行自评打分。

# Category说明
{category_instruction}

# 资料书写所需内容
{task_content}

# 现有证据
{content}

# 旧版文档信息（仅做参考）
{doc_original}

# 打分依据
自评时请从以下维度打分：
- completeness: 字段覆盖完整性，是否提取了所有必要字段
- accuracy: 内容准确度，提取的内容是否准确反映证据信息
- confidence: 置信度，对当前结果的自信程度
- 严重遗漏情况最高分不超过 0.8

# 注意
1. extraction的内容必须严格来自`# 现有证据`，禁止修改证据。
2. extraction中的"其他"字段仅存放资料书写所需的必须内容。
3. 仅当`# 现有证据`中不同代码文件对同一事实给出**明确且不可调和**的不同值（如头文件与实现文件对同一函数给出不同签名、A 文件明确支持某类型而 B 文件明确排除），且差异无法用表述详略、注释滞后、
作用域/关注点不同等合理原因解释时，才在 source_conflicts 中报告（并设置 `"definite": true`）；即使你已依据优先级选择了其中一个值，此类明确冲突仍须报告。
4. 若差异**不能完全确定**为真实冲突（如仅因描述详略不同、注释未同步、不同文件关注点不同、或仅个别字段写法差异），**不要**报告到 source_conflicts，视为正常证据差异，按优先级选用值即可。
source_conflicts 仅关注`# 现有证据`中不同代码文件之间的不一致，`# 旧版文档信息`仅供参考，不作为冲突检测的来源。
5. source_conflicts 的 values 中每个值必须附带来源文件名，格式为"值 (文件名)"，如 "out_new (aclnn_abs.h)"。

请返回JSON格式：
{{
  "extraction": {{
    {return_fields}
  }},
  "self_score": {{
    "score": 0.0-1.0,
    "reason": "一句话自评理由"
  }},
  "source_conflicts": [
    {{
      "fact": "冲突的事实（如：函数签名参数名）",
      "values": ["来源1的值", "来源2的值"],
      "source_files": ["aclnn_abs.h", "aclnn_abs.cpp"],
      "definite": true
    }}
  ]
}}
"""

    def extract_by_doc_part(
        self,
        doc_part: str,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        doc_arch: list = None,
        doc_arch_sources: dict = None,
        doc_chip: list = None,
        doc_chip_sources: dict = None,
        modified_files: list = None,
        doc_original_md: str = "",
    ) -> tuple[dict, list]:
        """Extract structured data based on doc_part and target groups.

        Returns:
            tuple: (extracted_dict, source_conflicts_list)
        """
        if doc_part == DocPart.SUPPORT:
            return self._extract_support(
                doc_arch or [],
                doc_arch_sources or {},
                doc_chip or [],
                doc_chip_sources or {},
                evidences_by_target,
                doc_original_md=doc_original_md,
            )
        if doc_part in (
            DocPart.SUMMARY,
            DocPart.IMPL_PRINCIPLE,
            DocPart.FUNCTION_PROTO,
            DocPart.PARA_CLARIFY,
            DocPart.RETURN_CLARIFY,
            DocPart.CONSTRAINT,
            DocPart.EXAMPLE,
        ):
            return self._extract_with_llm(
                evidences_by_target,
                doc_original,
                progressive,
                threshold,
                doc_part=doc_part,
                modified_files=modified_files,
            )
        return {}, []

    def _extract_support(
        self,
        doc_arch: list,
        doc_arch_sources: dict,
        doc_chip: list,
        doc_chip_sources: dict,
        evidences_by_target: dict,
        doc_original_md: str = "",
    ) -> tuple[dict, list]:
        """Extract support matrix from evidences.

        Returns:
            tuple: (result_dict, source_conflicts_list) - support matrix doesn't detect conflicts
        """
        all_chips = set(doc_chip or [])
        all_archs = set(doc_arch or [])

        # Step 1: 提取 section 的 support_chips 结果（兼容 legacy Pipeline B）
        for _, ev_list in evidences_by_target.items():
            for ev in ev_list:
                if ev.get("type") == "section":
                    support_chips = ev.get("support_chips", {})
                    all_chips.update(support_chips.get("chips", []) + support_chips.get("socs", []))
                    all_archs.update(support_chips.get("archs", []))

        # Step 2: [已删除] LLM 验证 — 已被上游 ch1_chip_support._scan_chip_registration 静态提取替代

        # Step 3: 映射产品型号信息
        mapper = ChipMapper()
        supported, stats = mapper.collect_supported_products(list(all_chips), list(all_archs))
        mapper.print_mapping_summary(stats)
        mapper.print_mapping_warnings(stats)
        # 输出格式跟随原文：原文产品名无 <term> 标签时同样输出纯文本，避免格式漂移。
        # 原文缺失（空 md_format，如新增文档）时保持默认带标签格式（向后兼容）。
        use_term_tags = True
        if doc_original_md:
            use_term_tags = "<term>" in doc_original_md
        md_header = "## 产品支持情况\n\n"
        md_content = mapper.generate_markdown(supported, use_term_tags=use_term_tags)
        result = {
            "support_matrix": {
                "support_chips": list(all_chips),
                "support_archs": list(all_archs),
                "supported": list(supported),
                "md_content": md_header + md_content,
            }
        }
        return result, []

    def _extract_with_llm(
        self,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        doc_part: str = "",
        modified_files: list = None,
    ) -> tuple[dict, list]:
        """Generic LLM-based extraction for doc_parts with progressive self-scoring.

        Returns:
            tuple: (extracted_dict, source_conflicts_list)
        """
        if not self.llm_client:
            return {}, []

        from pathlib import Path

        modified_file_basenames = None
        if modified_files:
            modified_file_basenames = {Path(f).name for f in modified_files}

        grouped = self._group_evidences_by_confidence(evidences_by_target, modified_file_basenames)
        targets = list(grouped.items())

        if progressive:
            return self._parallel_extract_targets(targets, doc_original, doc_part, threshold)

        result = {}
        extraction_result, all_conflicts = self._parallel_extract_targets(targets, doc_original, doc_part, 0.0)
        for group_key, extraction in extraction_result.items():
            result[group_key] = extraction.get("extraction", extraction)
        return result, all_conflicts

    def _call_llm(self, prompt: str, system_prompt: str) -> dict:
        """Call LLM client with prompt"""
        return self.llm_client.chat_with_json(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        )

    def _print_extraction_result(
        self, extraction, target_key, status_icon, batch_idx, reason, score, threshold
    ) -> None:
        if get_dashboard():
            dashboard_log(
                f"  {status_icon} [bold cyan]{target_key}[/bold cyan] Level {batch_idx + 1} score={score:.2f}"
            )
            return
        with self._console_lock:
            header_table = Table(show_header=False, border_style="cyan", box=None)
            header_table.add_column(style="cyan")

            info_table = Table(show_header=True, border_style="cyan")
            info_table.add_column("Batch", style="cyan")
            info_table.add_column("Score", justify="center")
            info_table.add_column("Status", justify="center")
            info_table.add_row(str(batch_idx + 1), f"[bold]{score:.2f}[/bold]", status_icon)

            reason_text = f"[dim]{reason}[/dim]"

            extraction_table = Table(show_header=True, header_style="bold cyan", border_style="cyan", expand=True)
            extraction_table.add_column("Target", style="cyan", no_wrap=True, ratio=1)
            extraction_table.add_column("Confidence", justify="center", no_wrap=True, ratio=1)
            extraction_table.add_column("Content", style="white", overflow="fold", ratio=4)
            extraction_table.add_column("Evidence", style="dim", ratio=2)
            for target, items in extraction.items():
                if items:
                    for item in items:
                        conf = item.get("confidence", 0)
                        content = item.get("content", "")
                        if not isinstance(content, str):
                            content = json.dumps(content, ensure_ascii=False)
                        evidence_ids = ", ".join(item.get("evidence_id", []))
                        conf_icon = "[green]✓[/green]" if conf >= threshold else "[yellow]~[/yellow]"
                        extraction_table.add_row(
                            target, f"{conf:.2f} {conf_icon}", content, f"[dim]{evidence_ids}[/dim]"
                        )

            content_panel = Panel(
                extraction_table if extraction else "[dim]No extraction results[/dim]",
                title="Extraction",
                border_style="cyan",
            )

            result_layout = Table(show_header=False, border_style="cyan", box=None)
            result_layout.add_column(style="cyan", overflow="fold")
            result_layout.add_row(header_table)
            result_layout.add_row(info_table)
            result_layout.add_row(reason_text)
            result_layout.add_row(content_panel)

            console.print()
            console.print(
                Panel(
                    result_layout,
                    title=f"[bold cyan]{target_key}[/bold cyan] - [cyan]Level {batch_idx + 1}[/cyan]",
                    border_style="cyan",
                )
            )

    def _parallel_extract_targets(
        self,
        targets: list[tuple[str, list[list[dict]]]],
        doc_original: dict,
        doc_part: str,
        threshold: float,
    ) -> tuple[dict, list]:
        """Parallel extraction across independent targets.

        Returns:
            tuple: (result_dict, all_conflicts_list)
        """
        result = {}
        all_conflicts = []
        current_dashboard = get_dashboard()
        current_doc_part = get_current_doc_part()

        def _run_with_context(target_key, ev_groups):
            if current_dashboard:
                tokens = set_dashboard_context(current_dashboard, current_doc_part)
                try:
                    return self._progressive_extract_single_target(
                        target_key=target_key,
                        ev_groups=ev_groups,
                        doc_original=doc_original,
                        doc_part=doc_part,
                        threshold=threshold,
                    )
                finally:
                    for t in reversed(tokens):
                        t.var.reset(t)
            else:
                return self._progressive_extract_single_target(
                    target_key=target_key,
                    ev_groups=ev_groups,
                    doc_original=doc_original,
                    doc_part=doc_part,
                    threshold=threshold,
                )

        with ThreadPoolExecutor(max_workers=config.EXTRACT_MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _run_with_context,
                    target_key=group_key,
                    ev_groups=ev_groups,
                ): group_key
                for group_key, ev_groups in targets
            }
            _is_tty = sys.stdout.isatty()
            _has_dashboard = current_dashboard is not None
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=console,
                disable=not _is_tty or _has_dashboard,
            ) as progress:
                task = progress.add_task(f"提取 {config.format_doc_part(doc_part)}", total=len(futures))
                for future in as_completed(futures):
                    group_key = futures[future]
                    extraction, conflicts = future.result()
                    result[group_key] = extraction
                    all_conflicts.extend(conflicts)
                    progress.update(task, advance=1)
        return result, all_conflicts

    def _progressive_extract_single_target(
        self,
        target_key: str,
        ev_groups: list[list[dict]],
        doc_original: dict,
        doc_part: str,
        threshold: float = 0.8,
    ) -> tuple[dict, list]:
        """Progressive extraction for single target with self-scoring.

        Returns:
            tuple: (extraction_dict, source_conflicts_list)
        """
        if not self.llm_client:
            return {}, []

        final_result = {}
        all_source_conflicts = []
        batch_idx = 0
        round_num = 0

        while batch_idx < len(ev_groups):
            current_evs = list(ev_groups[batch_idx])

            # First round: merge 1.0 (modified files) + 0.9 (highest confidence)
            if batch_idx == 0 and len(ev_groups) > 1:
                layer_1_0_count = len(current_evs)
                layer_0_9_count = len(ev_groups[batch_idx + 1])
                current_evs.extend(ev_groups[batch_idx + 1])
                batch_idx = 2  # Skip 0.9 layer, next round starts from 0.8
                log_result(
                    f"[{target_key}] Merged 1.0 layer ({layer_1_0_count}) + 0.9 layer ({layer_0_9_count}) = "
                    f"{len(current_evs)} evidences"
                )
            else:
                batch_idx += 1

            if not current_evs:
                continue

            round_num += 1

            prompt = self._build_progressive_prompt(
                target=target_key,
                current_evidences=current_evs,
                doc_original=doc_original,
                doc_part=doc_part,
            )

            try:
                response = self._call_llm(prompt, "你是一个技术文档助手，擅长从代码和注释中提取信息。")
            except Exception as e:
                log_result(f"[{target_key}] LLM call failed: {e}")
                return {}, []

            if not isinstance(response, dict):
                log_result(f"[{target_key}] LLM returned {type(response).__name__} instead of dict: {response}")
                return {}, []
            extraction = response.get("extraction", {})
            self_score = response.get("self_score", {})
            if not extraction and isinstance(self_score, dict):
                log_result(f"[{target_key}] extraction is empty, attempting to recover from top-level keys")
                extraction = {
                    k: v
                    for k, v in response.items()
                    if k not in ("self_score", "extraction", "source_conflicts") and isinstance(v, list)
                }
            if not isinstance(extraction, dict):
                log_result(f"[{target_key}] extraction is {type(extraction).__name__}, expected dict")
                extraction = {}
            else:
                non_list_keys = [k for k, v in extraction.items() if not isinstance(v, list)]
                if non_list_keys:
                    log_result(f"[{target_key}] filtering non-list values for keys: {non_list_keys}")
                    extraction = {k: v for k, v in extraction.items() if isinstance(v, list)}
            if not isinstance(self_score, dict):
                log_result(f"[{target_key}] self_score is {type(self_score).__name__}, expected dict")
                self_score = {}
            score = self_score.get("score", 0.0)
            reason = self_score.get("reason", "N/A")
            category = self._get_target_category(target_key)
            self._add_metadata_to_extraction(extraction, category, score)

            # Collect source_conflicts from response
            source_conflicts = response.get("source_conflicts", [])
            if source_conflicts:
                log_result(f"[{target_key}] Detected {len(source_conflicts)} source conflict(s)")
                for conflict in source_conflicts:
                    log_result(
                        f"  - {conflict.get('fact', '?')}: {conflict.get('values', [])} (evidences: "
                        f"{conflict.get('source_files', [])})"
                    )
                all_source_conflicts.extend(source_conflicts)

            status_icon = "[green]✓[/green]" if score >= threshold else "[yellow]✗[/yellow]"

            log_result(
                f"[{target_key} - Level {round_num}] batch = {round_num}, "
                f"score = {score}, reason = {reason}, extraction = {json.dumps(extraction, ensure_ascii=False)}"
            )

            self._print_extraction_result(extraction, target_key, status_icon, round_num - 1, reason, score, threshold)

            if score >= threshold:
                final_result = extraction
                break
            else:
                final_result = self._merge_results(final_result, extraction)

        return final_result, all_source_conflicts

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_prompt_config(doc_part: str) -> dict:
        """Load prompt config from YAML file."""
        path = config.RESOURCES_DIR / "prompts" / f"{doc_part}.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_progressive_prompt(
        self,
        target: str,
        current_evidences: list[dict],
        doc_original: dict,
        doc_part: str,
    ) -> str:
        """Build prompt for progressive extraction with self-scoring"""
        category = self._get_target_category(target)
        category_instruction = (
            "请严格遵循证据中的事实型内容" if category == "fact" else "可基于证据中的描述型内容进行适当发挥和补充"
        )

        contents = self._prepare_evidences_content(current_evidences)
        content_json = json.dumps(contents, ensure_ascii=False, indent=2)

        prompt_config = self._load_prompt_config(doc_part)

        if "task_content" not in prompt_config:
            if target in prompt_config:
                prompt_config = prompt_config[target]
            else:
                raise ValueError(f"Unknown target '{target}' for doc_part '{doc_part}'")

        task_content = prompt_config.get("task_content", "").strip()
        return_fields = prompt_config.get("return_fields", "").strip()

        if prompt_config.get("use_target_key"):
            doc_original_input = doc_original.get(target, {})
        else:
            doc_original_input = doc_original

        doc_original_json = json.dumps(doc_original_input, ensure_ascii=False, indent=2)

        return self._base_progressive_prompt(
            task_content, content_json, doc_original_json, category_instruction, return_fields
        )

    def _merge_results(self, prev: dict, new: dict) -> dict:
        """Merge previous and new extraction results"""
        if not prev:
            return new
        if not new:
            return prev

        merged = {}
        all_keys = set(prev.keys()) | set(new.keys())
        for key in all_keys:
            prev_items = prev.get(key, [])
            new_items = new.get(key, [])
            merged[key] = self._merge_and_deduplicate(prev_items, new_items)
        return merged
