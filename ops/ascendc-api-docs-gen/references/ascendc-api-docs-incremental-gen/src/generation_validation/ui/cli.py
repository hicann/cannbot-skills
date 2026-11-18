# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CLI UI for CodeDocAutoEditor - Rich terminal interface."""

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import questionary
from rich.box import SIMPLE  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from llm import LLMClient
from src.logger import info as log_info
from src.logger import setup_logger
from src.logger import warning as log_warning

from .. import config
from ..core import ValidationResult
from ..dashboard import (
    dashboard_done,
    dashboard_error,
    dashboard_log,
    dashboard_progress,
    dashboard_result,
    dashboard_step,
    dashboard_warn,
    get_current_doc_part,
    get_dashboard,
)
from ..evidence.chip_mapper import ChipMapper
from ..pipeline import GenerationStage, StyleStage, ValidationStage
from ..pipeline.helpers import get_output_dir, setup_pipeline

console = Console()


@dataclass
class EvidenceFile:
    path: Path
    relative_path: str
    doc_part: str
    basename: str


@dataclass
class LogCapture:
    lines: list = field(default_factory=list)

    def append(self, msg: str):
        self.lines.append(msg)


@dataclass
class PipelineResult:
    parsed: dict = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)
    style_summary: dict = field(default_factory=dict)
    style_extraction: dict = field(default_factory=dict)
    generated: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    history: list = field(default_factory=list)
    runtime_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    evidence_conflict: bool = False
    no_change_needed: bool = False
    conflict_details: Optional[str] = None
    max_retries_hit: bool = False
    attempts_used: int = 1


def detect_doc_part(filename: str) -> Optional[str]:
    match = re.search(config.DOC_PART_PATTERN, filename)
    return match.group(1) if match else None


def scan_evidence_files(evidence_dir: Path) -> list[EvidenceFile]:
    files = []
    if not evidence_dir.exists():
        return files

    for f in evidence_dir.rglob("*.json"):
        if f.is_file():
            rel_path = str(f.relative_to(evidence_dir))
            doc_part = detect_doc_part(f.name)
            if doc_part:
                files.append(
                    EvidenceFile(
                        path=f,
                        relative_path=rel_path,
                        doc_part=doc_part,
                        basename=f.stem,
                    )
                )
    return sorted(files, key=lambda x: (x.doc_part, x.relative_path))


def group_files_by_doc_part(files: list[EvidenceFile]) -> dict[str, list[EvidenceFile]]:
    grouped = {}
    for f in files:
        if f.doc_part not in grouped:
            grouped[f.doc_part] = []
        grouped[f.doc_part].append(f)
    return grouped


console_style = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),
        ("pointer", "fg:green bold"),
        ("highlighted", "fg:green bold"),
        ("selected", "fg:green bold"),
    ]
)


def select_evidence_file(files: list[EvidenceFile]) -> EvidenceFile:
    grouped = group_files_by_doc_part(files)

    choices = []
    for doc_part in config.DOC_PARTS:
        if doc_part in grouped:
            for f in grouped[doc_part]:
                choices.append(f)

    return questionary.select(
        "选择证据文件:",
        choices=[questionary.Choice(f.relative_path, f) for f in choices],
        style=console_style,
    ).ask()


def confirm_doc_part(detected: str) -> str:

    console.print(f"自动检测 doc_part: [bold cyan]{config.format_doc_part(detected)}[/bold cyan]")
    selected = questionary.select(
        "保留还是更改？",
        choices=["保留检测值", "更改..."],
        style=console_style,
    ).ask()

    if selected == "保留检测值":
        return detected
    return questionary.select(
        "选择 doc_part:",
        choices=[questionary.Choice(config.format_doc_part(dp), dp) for dp in config.DOC_PARTS],
        style=console_style,
    ).ask()


def select_mode() -> str:
    return questionary.select(
        "选择模式:",
        choices=[
            questionary.Choice("生成并验证", "full"),
            questionary.Choice("仅验证", "validate_only"),
        ],
        style=console_style,
    ).ask()


def resolve_evidence_file(evidence_dir: Path, doc_part: str) -> Optional[EvidenceFile]:
    """根据 doc_part 从 evidence_dir 中解析对应的证据文件。"""
    files = scan_evidence_files(evidence_dir)
    for f in files:
        if f.doc_part == doc_part:
            return f
    return None


def select_doc_part(available_parts: list[str]) -> Optional[str]:
    """从 available_parts 列表中选择 doc_part。列表只有一个则 auto-select。"""
    if len(available_parts) == 1:
        return available_parts[0]
    return questionary.select(
        "选择要处理的 doc_part:",
        choices=available_parts,
        style=console_style,
    ).ask()


def display_header():
    header = Text()
    logo = r"""
     _                           _    ____
    / \   ___  ___ ___ _ __   __| |  / ___|
   / _ \ / __|/ __/ _ \ '_ \ / _` | | |
  / ___ \\__ \ (_|  __/ | | | (_| | | |___
 /_/   \_\___/\___\___|_| |_|\__,_|  \____|
 """
    header.append(
        f"{logo}\n" + "CANN 算子文档增量生成与一致性校验命令行工具",
        style="bold cyan",
    )
    header.append("\n")
    header.append("─" * 85, style="dim")
    header.append("\n")
    header.append("当前支持的算子类型:  Ascend C", style="italic dim")
    console.print(Panel(header, border_style="cyan", expand=False))


def _is_validation_passed(v) -> bool:
    if isinstance(v, dict):
        consistency_passed = v.get("consistency_passed", False)
        metric_passed = v.get("metric_passed")
    else:
        consistency_passed = getattr(v, "consistency_passed", False)
        metric_passed = getattr(v, "metric_passed", None)

    if consistency_passed:
        return True

    return metric_passed is None or metric_passed == "N/A" or metric_passed == "PASS"


def run_full_pipeline(
    evidence_file: EvidenceFile,
    doc_part: str,
    log_capture: LogCapture,
    style_dir: Optional[Path] = None,
    change_reason: Optional[dict] = None,
    **kwargs,
) -> PipelineResult:
    result = PipelineResult()
    start = time.time()
    modified_files = kwargs.get("modified_files", [])

    try:
        llm = LLMClient()

        if get_dashboard():
            dashboard_step("加载证据")
            dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 1:[/bold] 加载与解析证据")
        else:
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 1:[/bold] 加载与解析证据")
        evidence_stage, raw_input, parsed, ranked, _, _ = setup_pipeline(str(evidence_file.path), llm)
        if get_dashboard():
            dashboard_log(f"  [green]✓[/green] 已加载 {parsed.total_evidences} 条证据")
            dashboard_log(
                f"  [dim]  targets: {', '.join(f'{k}({len(v)})' for k, v in ranked.evidences_by_target.items())}[/dim]"
            )
        else:
            console.print(f"  [green]✓[/green] 已加载 {parsed.total_evidences} 条证据")
            console.print(
                f"  [dim]  targets: {', '.join(f'{k}({len(v)})' for k, v in ranked.evidences_by_target.items())}[/dim]"
            )

        if get_dashboard():
            dashboard_step("加载代码证据")
            dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 2:[/bold] 提取证据")
        else:
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 2:[/bold] 提取证据")
        get_output_dir()

        extracted, source_conflicts = evidence_stage.extract_evidence(
            ranked=ranked,
            doc_part=doc_part,
            doc_original=parsed.doc_original_parsed,
            progressive=True,
            evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
            doc_arch=parsed.doc_original_arch,
            doc_arch_sources=parsed.doc_original_arch_sources,
            doc_chip=parsed.doc_original_chip,
            doc_chip_sources=parsed.doc_original_chip_sources,
            modified_files=modified_files,
            doc_original_md=parsed.doc_original_md,
        )
        if get_dashboard():
            dashboard_log(f"  [green]✓[/green] 已提取字段: {', '.join(extracted.keys())}")
        else:
            console.print(f"  [green]✓[/green] 已提取字段: {', '.join(extracted.keys())}")

        # Check for source conflicts from extraction phase
        definite_conflicts = [c for c in source_conflicts if c.get("definite", False)]
        uncertain_conflicts = [c for c in source_conflicts if not c.get("definite", False)]
        if uncertain_conflicts:
            uncertain_msg = "; ".join(f"{c.get('fact', '?')}: {c.get('values', [])}" for c in uncertain_conflicts)
            log_info(f"[{doc_part}] 忽略不能完全确定的证据差异（不告警）: {uncertain_msg}")

        if definite_conflicts:
            conflict_details = "; ".join(f"{c.get('fact', '?')}: {c.get('values', [])}" for c in definite_conflicts)
            result.evidence_conflict = True
            result.conflict_details = conflict_details
            conflict_msg = f"提取阶段检测到明确的证据冲突: {conflict_details}"
            if get_dashboard():
                dashboard_log(f"  [red]⚠[/red] {conflict_msg}")
                db = get_dashboard()
                dp = get_current_doc_part()
                if db and dp:
                    db.update(dp, status="✗", result="FAIL")
            else:
                console.print(f"  [red]⚠[/red] {conflict_msg}")
            log_info(f"[{doc_part}] {conflict_msg}")
            result.generated = []
            result.validation = []
            result.runtime_seconds = time.time() - start
            return result

        if get_dashboard():
            dashboard_step("风格数据")
            dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 3:[/bold] 风格数据")
        else:
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 3:[/bold] 风格数据")
        style_stage = StyleStage(llm, style_dir)
        style_summary = style_stage.get_style_summary()
        style_extraction = style_stage.get_style_extraction()
        if get_dashboard():
            dashboard_log(f"  [green]✓[/green] 核心主题: {style_summary.core_topic}")
        else:
            console.print(f"  [green]✓[/green] 核心主题: {style_summary.core_topic}")

        if get_dashboard():
            dashboard_step("生成文档")
            dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 4:[/bold] 生成文档章节")
        else:
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 4:[/bold] 生成文档章节")
        validation_results_list = []
        final_results = []
        history: list[dict] = []

        if doc_part == "support":
            support_matrix = extracted.get("support_matrix", {})
            md_content = support_matrix.get("md_content")
            if get_dashboard():
                dashboard_log("  [green]✓[/green] 支持矩阵已生成")
                dashboard_step("验证")
                dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 5:[/bold] Validate")
            else:
                console.print("  [green]✓[/green] 支持矩阵已生成")
                console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 5:[/bold] Validate")
            validation_result = ValidationResult(
                consistency_passed=True,
                accuracy_rate=1.0,
                metric_passed="PASS",
                edit_distance_ratio=1.0,
                avg_confidence=1.0,
                issues=[],
                suggestion="Auto-generated from support matrix.",
                passed=True,
            )
            validation_results_list.append(validation_result)
            if get_dashboard():
                dashboard_log(
                    f"  [green]✓ PASS[/green] (acc={validation_result.accuracy_rate}, "
                    f"metric={validation_result.metric_passed})"
                )
            else:
                console.print(
                    f"  [green]✓ PASS[/green] (acc={validation_result.accuracy_rate}, "
                    f"metric={validation_result.metric_passed})"
                )
            final_results.append(
                {
                    "sub_key": "support_matrix",
                    "generated_text": md_content,
                    "modification_summary": {},
                }
            )
        else:
            generation_stage = GenerationStage(llm, style_stage.template_loader)
            validation_stage = ValidationStage(llm)

            previous_issues = []
            accumulated_issues = []
            previous_generated_doc: dict[str, str] = {}
            all_passed = False
            passed_indices = []

            for attempt in range(config.MAX_REGENERATION_RETRIES):
                if attempt > 0:
                    msg = f"重新生成尝试 {attempt + 1}/{config.MAX_REGENERATION_RETRIES}"
                    if get_dashboard():
                        dashboard_log(f"  [yellow]→[/yellow] {msg}")
                    else:
                        console.print(f"\n  [yellow]→[/yellow] {msg}")
                    log_info(f"[{doc_part}] {msg}")
                    if passed_indices:
                        skip_msg = f"跳过 {len(passed_indices)} 个已通过的章节"
                        if get_dashboard():
                            dashboard_log(f"  [dim]  {skip_msg}[/dim]")
                        else:
                            console.print(f"  [dim]  {skip_msg}[/dim]")
                        log_info(f"[{doc_part}] {skip_msg}")

                if get_dashboard():
                    dashboard_step("生成文档")
                generated_results = generation_stage.generate(
                    original_content={
                        "parsed": parsed.doc_original_parsed,
                        "md": parsed.doc_original_md,
                    },
                    evidences=extracted,
                    style_summary=style_summary,
                    style_extraction=style_extraction,
                    doc_part=doc_part,
                    previous_issues=previous_issues,
                    generated_doc=previous_generated_doc,
                    passed_indices=passed_indices,
                    change_reason=change_reason,
                )
                if get_dashboard():
                    dashboard_log(f"  [dim]  已生成 {len(generated_results)} 个章节[/dim]")
                else:
                    console.print(f"  [dim]  已生成 {len(generated_results)} 个章节[/dim]")
                    display_generation_results(generated_results)

                # Check for need_change/evidence_conflict signals from writer
                all_no_change = True
                any_conflict = False
                conflict_reasons = []
                for res in generated_results:
                    summary = res.modification_summary or {}
                    analysis = summary.get("analysis", {})
                    if isinstance(analysis, dict):
                        if analysis.get("need_change") is True:
                            all_no_change = False
                        if analysis.get("evidence_conflict") is True:
                            any_conflict = True
                            reason = analysis.get("reason", "")
                            if reason:
                                conflict_reasons.append(reason)
                    else:
                        # If no analysis at all, assume change is needed
                        all_no_change = False

                if all_no_change and generated_results:
                    no_change_reason = ""
                    if generated_results:
                        summary = generated_results[0].modification_summary or {}
                        analysis = summary.get("analysis", {})
                        if isinstance(analysis, dict):
                            no_change_reason = analysis.get("reason", "")
                    result.no_change_needed = True
                    result.conflict_details = no_change_reason
                    result.attempts_used = attempt + 1
                    result.success = True
                    result.generated = [
                        {
                            "sub_key": res.modification_summary.get("sub_key", f"section_{i + 1}"),
                            "generated_text": res.generated_text,
                            "modification_summary": res.modification_summary,
                        }
                        for i, res in enumerate(generated_results)
                    ]
                    if get_dashboard():
                        dashboard_log(f"  [yellow]→[/yellow] LLM 判断无需修改: {no_change_reason}")
                        db = get_dashboard()
                        dp = get_current_doc_part()
                        if db and dp:
                            db.update(dp, status="⚠", result="SKIP", progress_current=1, progress_total=1)
                    else:
                        console.print(f"  [yellow]→[/yellow] LLM 判断无需修改: {no_change_reason}")
                    log_info(f"[{doc_part}] LLM 判断无需修改: {no_change_reason}")
                    break

                if any_conflict:
                    result.evidence_conflict = True
                    result.conflict_details = "; ".join(conflict_reasons)
                    result.attempts_used = attempt + 1
                    result.success = False
                    if get_dashboard():
                        dashboard_log(f"  [red]⚠[/red] Writer 检测到证据冲突: {result.conflict_details}")
                        db = get_dashboard()
                        dp = get_current_doc_part()
                        if db and dp:
                            db.update(dp, status="✗", result="FAIL", progress_current=1, progress_total=1)
                    else:
                        console.print(f"  [red]⚠[/red] Writer 检测到证据冲突: {result.conflict_details}")
                    log_info(f"[{doc_part}] Writer 检测到证据冲突: {result.conflict_details}")
                    break

                if attempt == 0:
                    if get_dashboard():
                        dashboard_step("验证")
                        dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 5:[/bold] Validate")
                    else:
                        console.rule("[bold cyan]代码-文档校验[/bold cyan]")
                        console.print(
                            f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim][bold]Step 5:[/bold] Validate"
                        )

                validation_results_list.clear()
                final_results.clear()
                all_passed = True
                has_unresolvable = False
                current_issues = []
                current_generated_doc: dict[str, str] = {}
                current_passed_indices = []

                parsed_dedup = evidence_stage.parser.parse_raw_input(raw_input, deduplicate=True)
                evidences_for_validation = parsed_dedup.evidences
                _orig = parsed.total_evidences
                _dedup = _orig - parsed_dedup.total_evidences
                if _dedup > 0:
                    if get_dashboard():
                        dashboard_log(
                            f"  [dim]Dedup evidences for validation: {_orig} -> "
                            f"{parsed_dedup.total_evidences} (removed {_dedup} redundant)[/dim]"
                        )
                    else:
                        console.print(
                            f"  [dim]Dedup evidences for validation: {_orig} -> "
                            f"{parsed_dedup.total_evidences} (removed {_dedup} redundant)[/dim]"
                        )

                total_sections = len(generated_results)
                for i, res in enumerate(generated_results):
                    validation_result = validation_stage.validate(
                        doc_part=doc_part,
                        doc_original=parsed.doc_original_md,
                        evidences=evidences_for_validation,
                        doc_generated=res.generated_text,
                        evidences_extracted=extracted,
                        check_metric=True,
                        idx=i + 1,
                    )
                    if not get_dashboard():
                        ValidationStage.print_result(validation_result)
                    result_dict = ValidationStage.format_result(validation_result)

                    passed = _is_validation_passed(result_dict)
                    result_dict["passed"] = passed

                    status_icon = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
                    sub_key = res.modification_summary.get("sub_key", f"section_{i + 1}")
                    if get_dashboard():
                        dashboard_progress(i + 1, total_sections)
                        dashboard_log(
                            f"  {status_icon} {sub_key} (acc={result_dict.get('accuracy_rate', 0):.2f}, "
                            f"metric={result_dict.get('metric_passed', 'N/A')})"
                        )
                    else:
                        console.print(
                            f"  {status_icon} {sub_key} (acc={result_dict.get('accuracy_rate', 0):.2f}, "
                            f"metric={result_dict.get('metric_passed', 'N/A')})"
                        )

                    if not passed:
                        all_passed = False
                        issues = result_dict.get("issues", [])
                        has_unresolvable = False
                        if issues:
                            for issue in issues:
                                issue_id = issue.get("id", "?")
                                reason = issue.get("reason", "")
                                issue["sub_key"] = sub_key
                                if issue.get("unresolvable"):
                                    has_unresolvable = True
                                    if get_dashboard():
                                        dashboard_log(f"    [red]⚠[/red] [{issue_id}] 不可解决: {reason}")
                                    else:
                                        console.print(f"    [red]⚠[/red] [{issue_id}] 不可解决: {reason}")
                                else:
                                    if get_dashboard():
                                        dashboard_log(f"    [yellow]→[/yellow] [{issue_id}] {reason}")
                                    else:
                                        console.print(f"    [yellow]→[/yellow] [{issue_id}] {reason}")
                        current_issues.extend(issues)
                        current_generated_doc[sub_key] = res.generated_text
                        if has_unresolvable:
                            unresolvable_issues = [iss for iss in issues if iss.get("unresolvable")]
                            result.evidence_conflict = True
                            result.conflict_details = "; ".join(iss.get("reason", "") for iss in unresolvable_issues)
                            result.attempts_used = attempt + 1
                            if get_dashboard():
                                dashboard_log(f"  [red]⚠[/red] 检测到不可解决的证据冲突: {result.conflict_details}")
                                db = get_dashboard()
                                dp = get_current_doc_part()
                                if db and dp:
                                    db.update(dp, status="✗", result="FAIL")
                            else:
                                console.print(f"  [red]⚠[/red] 检测到不可解决的证据冲突: {result.conflict_details}")
                            log_info(f"[{doc_part}] 检测到不可解决的证据冲突: {result.conflict_details}")
                            break
                    else:
                        current_passed_indices.append(i)

                    validation_results_list.append(result_dict)
                    final_results.append(
                        {
                            "sub_key": res.modification_summary.get("sub_key"),
                            "generated_text": res.generated_text,
                            "modification_summary": res.modification_summary,
                        }
                    )

                # 记录本次 attempt 的完整快照（生成内容 + 注入的 issues + 验证结果）
                history.append(
                    {
                        "attempt": attempt + 1,
                        "injected_issues": [issue.copy() for issue in previous_issues],
                        "generated_sections": [
                            {
                                "sub_key": item.get("sub_key"),
                                "generated_text": item.get("generated_text"),
                                "modification_summary": item.get("modification_summary"),
                            }
                            for item in final_results
                        ],
                        "validation_results": [{k: v for k, v in vdict.items()} for vdict in validation_results_list],
                    }
                )

                # Break out of retry loop if unresolvable conflict detected
                if result.evidence_conflict and has_unresolvable:
                    result.success = False
                    break

                if all_passed:
                    result.attempts_used = attempt + 1
                    break
                elif attempt < config.MAX_REGENERATION_RETRIES - 1:
                    for issue in current_issues:
                        tagged = issue.copy()
                        tagged["from_attempt"] = attempt + 1
                        accumulated_issues.append(tagged)
                    previous_issues = accumulated_issues
                    previous_generated_doc = current_generated_doc
                    passed_indices = current_passed_indices
                    if get_dashboard():
                        dashboard_log(
                            f"  [yellow]→[/yellow] 收集 {len(current_issues)} 个问题 "
                            f"（累计 {len(accumulated_issues)} 个）用于重新生成..."
                        )
                    else:
                        console.print(
                            f"  [yellow]→[/yellow] 收集 {len(current_issues)} 个问题 "
                            f"（累计 {len(accumulated_issues)} 个）用于重新生成..."
                        )
                else:
                    result.max_retries_hit = True
                    result.attempts_used = attempt + 1
                    max_msg = f"已达到最大重试次数 ({config.MAX_REGENERATION_RETRIES})，使用最后一次结果继续。"
                    if get_dashboard():
                        dashboard_log(f"  [yellow]{max_msg}[/yellow]")
                    else:
                        console.print(f"  [yellow]{max_msg}[/yellow]")
                    log_warning(f"[{doc_part}] {max_msg}")

        result.parsed = {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
            "evidences_by_target": ranked.evidences_by_target,
        }
        result.extracted = extracted
        result.style_summary = {
            "core_topic": style_summary.core_topic,
            "key_points": style_summary.key_points,
            "context_dependencies": style_summary.context_dependencies,
        }
        result.style_extraction = {
            "terminology": style_extraction.terminology,
            "structure": style_extraction.structure,
            "format": style_extraction.format,
        }
        if not result.no_change_needed:
            result.generated = final_results
            result.validation = validation_results_list
            result.history = history

    except Exception as e:
        result.success = False
        result.error = str(e)
        if get_dashboard():
            dashboard_error(f"Error: {e}")
        else:
            console.print(f"\n[red]✗ 错误:[/red] {e}")

    if not result.success:
        dashboard_result(None)
    else:
        all_passed = all(getattr(v, "passed", True) for v in result.validation)
        dashboard_result(all_passed)
        dashboard_done()
    result.runtime_seconds = time.time() - start
    return result


def run_validate_only(evidence_file: EvidenceFile, doc_part: str, log_capture: LogCapture, **kwargs) -> PipelineResult:
    result = PipelineResult()
    start = time.time()

    try:
        llm = LLMClient()

        dashboard_step("加载证据")
        dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 1:[/bold] 加载与解析证据")
        if not get_dashboard():
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]Step 1:[/bold] 加载与解析证据")
        evidence_stage, raw_input, parsed, ranked, _, _ = setup_pipeline(str(evidence_file.path), llm, deduplicate=True)
        dashboard_log(f"  [green]✓[/green] 已加载 {parsed.total_evidences} 条证据")
        if not get_dashboard():
            console.print(f"  [green]✓[/green] 已加载 {parsed.total_evidences} 条证据")

        dashboard_step("验证")
        dashboard_log(f"[dim]\\[{doc_part}][/dim] [bold]Step 2:[/bold] Validate")
        if not get_dashboard():
            console.print(f"\n[dim]\\[{config.format_doc_part(doc_part)}][/dim] [bold]Step 2:[/bold] Validate")
        validation_stage = ValidationStage(llm)
        validation_results_list = []

        if doc_part == "support":
            extracted, _source_conflicts = evidence_stage.extract_evidence(
                ranked=ranked,
                doc_part=doc_part,
                doc_original=parsed.doc_original_md,
                progressive=True,
                evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
                doc_arch=parsed.doc_original_arch,
                doc_arch_sources=parsed.doc_original_arch_sources,
                doc_chip=parsed.doc_original_chip,
                doc_chip_sources=parsed.doc_original_chip_sources,
                doc_original_md=parsed.doc_original_md,
            )
            support_matrix = extracted.get("support_matrix", {})
            md_content = support_matrix.get("md_content", "")

            # DEBUG: 打印提取的数据来源
            log_info(f"静态提取 arch: {parsed.doc_original_arch}")
            log_info(f"静态提取 chip: {parsed.doc_original_chip}")

            # 提取侧无任何 chip/arch 证据时，生成矩阵只是"全集标不支持"的默认产物，
            # 与原文比对无意义（如聚合头识别失败导致 arch 全被剔除），判"无法判定"
            # 放行，不触发 SUPPORT_MISMATCH 强制生成
            skip_compare = not support_matrix.get("support_chips") and not support_matrix.get("support_archs")
            if skip_compare:
                log_info("支持矩阵提取侧无 chip/arch 证据，无法判定，跳过原文比对")

            # 用 ChipMapper 的同一套规则解析原文和生成内容
            mapper = ChipMapper()
            original_mapping = mapper.parse_support_md(parsed.doc_original_md)
            generated_mapping = mapper.parse_support_md(md_content)

            # 对比关联关系
            passed = skip_compare or original_mapping == generated_mapping

            # DEBUG: 打印对比详情
            log_info(f"原文与当前的产品支持是否一致: {passed}")

            if not passed:
                all_products = set(original_mapping.keys()) | set(generated_mapping.keys())
                for product in sorted(all_products):
                    orig = original_mapping.get(product)
                    gen = generated_mapping.get(product)
                    if orig != gen:
                        log_info(f"  差异: {product}: 原文={orig}, 生成={gen}")

            if passed:
                validation_result = ValidationResult(
                    consistency_passed=True,
                    accuracy_rate=1.0,
                    metric_passed="PASS",
                    edit_distance_ratio=1.0,
                    avg_confidence=1.0,
                    issues=[],
                    suggestion="产品支持情况与原文一致"
                    if not skip_compare
                    else "提取侧无 chip/arch 证据，无法判定支持矩阵，跳过比对",
                    passed=True,
                )
            else:
                validation_result = ValidationResult(
                    consistency_passed=False,
                    accuracy_rate=0.0,
                    metric_passed="FAIL",
                    edit_distance_ratio=0.0,
                    avg_confidence=1.0,
                    issues=[
                        {
                            "id": "SUPPORT_MISMATCH",
                            "reason": f"原文: {original_mapping}, 生成: {generated_mapping}",
                        }
                    ],
                    suggestion="产品支持情况与原文不一致，将在生成阶段更新",
                    passed=False,
                )

            validation_results_list.append(validation_result)
            result_dict = ValidationStage.format_result(validation_result)
            if not get_dashboard():
                ValidationStage.print_result(validation_result)
            if get_dashboard():
                dashboard_progress(1, 1)
            indices = []
        else:
            indices = [None]

        total_sections = len(indices)
        for i, idx in enumerate(indices):
            if isinstance(idx, int):
                doc_original = parsed.doc_original_md
            else:
                doc_original = parsed.doc_original_md
            validation_result = validation_stage.validate(
                doc_part=doc_part,
                doc_original=doc_original,
                evidences=parsed.evidences,
                check_metric=False,
                idx=idx,
            )
            result_dict = ValidationStage.format_result(validation_result)
            if not get_dashboard():
                ValidationStage.print_result(validation_result)
            validation_result.passed = _is_validation_passed(result_dict)
            validation_results_list.append(validation_result)
            if get_dashboard():
                dashboard_progress(i + 1, total_sections)

        result.parsed = {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
        }
        result.validation = validation_results_list

    except Exception as e:
        result.success = False
        result.error = str(e)
        dashboard_error(f"Error: {e}")
        if not get_dashboard():
            console.print(f"\n[red]✗ 错误:[/red] {e}")

    if not result.success:
        dashboard_result(None)
    else:
        all_passed = all(getattr(v, "passed", True) for v in result.validation)
        dashboard_result(all_passed)
        dashboard_done()
    result.runtime_seconds = time.time() - start
    return result


def display_generation_results(results: list, mode: str = "full"):
    if get_dashboard():
        return
    if not results:
        console.print("  [dim]未生成任何章节[/dim]")
        return

    table = Table(show_header=False, border_style="cyan", box=SIMPLE)
    table.add_column(style="cyan")
    table.add_column(highlight=False, overflow="fold")

    for i, item in enumerate(results):
        sub_key = ""
        text = ""
        if hasattr(item, "modification_summary"):
            sub_key = item.modification_summary.get("sub_key", f"section_{i + 1}")
            text = item.generated_text
        elif isinstance(item, dict):
            sub_key = item.get("sub_key", f"section_{i + 1}")
            text = item.get("generated_text", "")
        else:
            sub_key = f"section_{i + 1}"
            text = str(item)

        label = f"[bold]{sub_key}[/bold]" if sub_key else f"[bold]section_{i + 1}[/bold]"
        content_panel = Panel(text, border_style="cyan", expand=False)
        table.add_row(label, content_panel)

    console.print(Panel(table, title="已生成章节", border_style="cyan", expand=False))


def display_results(result: PipelineResult, mode: str):
    console.print()

    has_failures = any(not _is_validation_passed(v) for v in result.validation)
    if result.evidence_conflict:
        failed_icon = "[red]⚠ 证据冲突[/red]"
        status_text = failed_icon
        border_style = "red"
        title = "结果"
    elif result.no_change_needed:
        status_text = "[yellow]→ 无需修改[/yellow]"
        border_style = "yellow"
        title = "结果"
    elif has_failures:
        failed_icon = "[red]✗ 验证失败[/red]"
        status_text = failed_icon
        border_style = "red"
        title = "结果"
    else:
        status_text = "[bold green]✓ 所有验证通过[/bold green]"
        border_style = "cyan"
        title = "结果"

    runtime_text = f"{result.runtime_seconds:.2f}s"

    if mode == "validate_only":
        summary_table = Table(show_header=False, border_style=border_style, box=SIMPLE)
        summary_table.add_column(style="cyan")
        summary_table.add_column(highlight=False, overflow="fold")

        if result.validation:
            v_status = (
                "[green]PASS[/green]" if all(_is_validation_passed(v) for v in result.validation) else "[red]FAIL[/red]"
            )
        else:
            v_status = "[red]FAIL[/red]"
        summary_table.add_row(Text("状态", style="bold"), v_status)
        summary_table.add_row(Text("耗时", style="bold"), runtime_text)

        for vi, v in enumerate(result.validation):
            vdict = v if isinstance(v, dict) else v.model_dump()

            if len(result.validation) > 1:
                v_passed = _is_validation_passed(v)
                v_icon = "[green]PASS[/green]" if v_passed else "[red]FAIL[/red]"
                summary_table.add_row(Text(""), "")
                summary_table.add_row(Text(f"── 章节 {vi + 1} ──", style="bold"), v_icon)

            summary_table.add_row(Text("一致性通过:", style="bold cyan"), str(vdict.get("consistency_passed")))
            summary_table.add_row(Text("准确率:", style="bold cyan"), str(vdict.get("accuracy_rate")))
            summary_table.add_row(Text("指标通过:", style="bold cyan"), str(vdict.get("metric_passed")))
            summary_table.add_row(Text("编辑距离比:", style="bold cyan"), str(vdict.get("edit_distance_ratio")))
            summary_table.add_row(Text("平均置信度:", style="bold cyan"), str(vdict.get("avg_confidence")))
            summary_table.add_row(Text("通过:", style="bold cyan"), str(vdict.get("passed", "?")))

            issues = vdict.get("issues", [])
            if issues and isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        issue_lines = []
                        issue_lines.append(f"  [yellow]编号:[/yellow] {issue.get('id', '?')}")
                        issue_lines.append(f"  [yellow]位置:[/yellow] {issue.get('location', '?')}")
                        issue_lines.append(f"  [yellow]原因:[/yellow] {issue.get('reason', '')}")
                        suggestion = issue.get("suggestion", "")
                        if suggestion and str(suggestion) != "None":
                            issue_lines.append(f"  [yellow]建议:[/yellow] {suggestion}")
                        evidence = issue.get("evidence", "")
                        if evidence and str(evidence) != "None":
                            issue_lines.append(f"  [yellow]证据:[/yellow] {evidence}")
                        summary_table.add_row(Text("", style="dim"), "")
                        summary_table.add_row(Text("问题", style="bold red"), "\n".join(issue_lines))
            else:
                suggestion = vdict.get("suggestion", "")
                if suggestion and str(suggestion) != "None" and str(suggestion):
                    summary_table.add_row(Text("建议", style="bold cyan"), str(suggestion))

        console.print(Panel(summary_table, title=title, border_style=border_style, expand=False))
        return

    else:
        if result.generated:
            content_parts = []
            for i, item in enumerate(result.generated):
                sub_key = item.get("sub_key", "")
                text = item.get("generated_text", "")
                if sub_key:
                    content_parts.append(f"\n### {sub_key}\n\n{text}")
                else:
                    content_parts.append(text)

            if content_parts:
                content_text = "\n\n".join(content_parts).lstrip("\n")
                outer_table = Table(show_header=False, border_style=border_style, box=SIMPLE)
                outer_table.add_column(style="cyan")
                outer_table.add_column(overflow="fold")
                outer_table.add_row("[bold]状态[/bold]", status_text)
                outer_table.add_row("[bold]耗时[/bold]", runtime_text)
                outer_table.add_row()
                outer_table.add_row("[bold cyan]生成内容[/bold cyan]", "")
                content_panel = Panel(content_text, border_style=border_style, expand=False)
                outer_table.add_row("", content_panel)
                console.print(Panel(outer_table, title=title, border_style=border_style, expand=False))
            else:
                table = Table(show_header=True, header_style="bold cyan", border_style=border_style)
                table.add_column("状态", style="bold")
                table.add_column("耗时", justify="right")
                table.add_row(status_text, runtime_text)
                console.print(Panel(table, title=title, border_style=border_style, expand=False))
        elif result.validation:
            # No generated content, show validation details
            table = Table(show_header=True, header_style="bold cyan", border_style=border_style)
            table.add_column("描述", style="cyan")
            table.add_column("文档长度", justify="right")
            table.add_column("磁盘长度", justify="right")
            table.add_column("结果", style="bold")
            for vi, v in enumerate(result.validation):
                vdict = v if isinstance(v, dict) else v.model_dump()

                if len(result.validation) > 1:
                    v_passed = _is_validation_passed(v)
                    v_icon = "[green]PASS[/green]" if v_passed else "[red]FAIL[/red]"
                    table.add_row(Text(""), "")
                    table.add_row(Text(f"── 章节 {vi + 1} ──", style="bold"), v_icon)

                table.add_row(Text("一致性通过:", style="bold cyan"), str(vdict.get("consistency_passed")))

            console.print(Panel(table, title=title, border_style=border_style, expand=False))
        else:
            table = Table(show_header=True, header_style="bold cyan", border_style=border_style)
            table.add_column("状态", style="bold")
            table.add_column("耗时", justify="right")
            table.add_row(status_text, runtime_text)
            console.print(Panel(table, title=title, border_style=border_style, expand=False))


def run_cli():
    screen_console = Console(force_terminal=True)
    screen_console.clear()
    try:
        _run_cli_impl(screen_console)
    finally:
        screen_console.clear()


def _run_cli_impl(console: Console):
    setup_logger(console=False)

    files = scan_evidence_files(config.EVIDENCE_EXAMPLE_DIR)

    if not files:
        console.print("[red]✗ 在 data/evidences/ 中未找到证据文件[/red]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Doc IncrGen&Val - CLI")
    parser.add_argument("--file", "-f", type=str, help="Path to evidence file (relative to data/evidences/)")
    parser.add_argument("--doc-part", "-d", type=str, choices=config.DOC_PARTS, help="Document part")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["full", "validate_only"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available evidence files and exit")
    args = parser.parse_args()

    if args.list:
        console.print(Panel("可用证据文件", border_style="cyan"))
        grouped = group_files_by_doc_part(files)
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("文档模块", style="cyan")
        table.add_column("文件", style="white")
        for doc_part in config.DOC_PARTS:
            if doc_part in grouped:
                for f in grouped[doc_part]:
                    table.add_row(doc_part, f.relative_path)
        console.print(table)
        sys.exit(0)

    console.clear()
    display_header()
    console.print("[dim]扫描证据文件中...[/dim]")
    console.print(f"[green]✓[/green] 找到 [bold]{len(files)}[/bold] 个证据文件")
    console.print()

    selected_file = select_evidence_file(files)
    console.print(f"  [dim]已选择:[/dim] {selected_file.relative_path}")

    doc_part = confirm_doc_part(selected_file.doc_part)
    console.print(f"  [dim]文档模块:[/dim] [cyan]{config.format_doc_part(doc_part)}[/cyan]")

    console.print()
    mode = select_mode()
    console.print(f"  [dim]模式:[/dim] [yellow]{mode}[/yellow]")

    if not selected_file or not doc_part or not mode:
        return

    console.print()

    log_capture = LogCapture()

    if mode == "full":
        console.rule("[bold cyan]文档生成[/bold cyan]")
        result = run_full_pipeline(selected_file, doc_part, log_capture)
    else:
        console.rule("[bold cyan]代码-文档校验[/bold cyan]")
        result = run_validate_only(selected_file, doc_part, log_capture)

    display_results(result, mode)

    console.print()
    if questionary.confirm("再次运行？", default=False).ask():
        _run_cli_impl(console)


if __name__ == "__main__":
    # TEST CMD: python -m src.generation_validation.ui.cli
    run_cli()
