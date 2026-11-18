# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Main entry point - orchestrates the full document editing pipeline."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from llm import LLMClient
from src.logger import debug, highlight, info, log_result, section, setup_logger

from . import config
from .core import ModificationResult
from .pipeline import (
    GenerationStage,
    StyleStage,
    ValidationStage,
)
from .pipeline.helpers import get_output_dir, save_json, setup_pipeline
from .ui.cli import _is_validation_passed

setup_logger(level=logging.INFO)


def _lost_internal_links(full_original: str, generated_section: str, heading: str) -> list[str]:
    """确定性巡检：原章节有、生成章节丢的内部锚点链接。

    注：evidence json 的 md_format 只含本章节，锚点定义可能在其他章节而无法在此
    验证——跨章节锚点是否仍存在由回填后的全文档巡检（run_all_cli）负责。
    """
    m = re.search(rf"(?m)^{re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", full_original, re.S)
    orig_section = m.group(1) if m else ""
    if not orig_section:
        return []
    lost = set(re.findall(r"\]\(#([^)]+)\)", orig_section)) - set(re.findall(r"\]\(#([^)]+)\)", generated_section))
    return sorted(lost)


_CODE_BLOCK_RE = re.compile(r"```[\w]*\s*\n(.*?)```", re.S)
_PROTO_HINT_RE = re.compile(r"(__aicore__|__simd_callee__|\binline\b|\btemplate\b)")


def _unsourced_prototypes(generated_section: str, corpus: str) -> list[str]:
    """确定性巡检：函数原型章节中列出、但原文档与证据语料里都不存在的签名。

    针对 LLM 按已有重载模式"补齐"编造原型的场景（如仅 float 新增 repeat 重载，
    却给全部数据类型补出 repeat 原型）。只做校验告警，不直接改写生成结果。
    """

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    corpus_n = _norm(corpus)
    unsourced = []
    for block in _CODE_BLOCK_RE.findall(generated_section):
        for raw_line in block.splitlines():
            line = raw_line.strip().rstrip(";").strip()
            if not line or "(" not in line or "//" in line.split("(", 1)[0]:
                continue
            if not _PROTO_HINT_RE.search(line):
                continue
            if _norm(line) not in corpus_n:
                unsourced.append(line)
    return unsourced


_API_NAME_RE = re.compile(r"\basc_[a-z0-9_]+\b")


def _unsourced_api_names(generated_section: str, full_original: str) -> list[str]:
    """确定性巡检：生成章节中出现、但原文档从未提及的 asc_* 接口名。

    证据（文件级 anchor）的 code_map/上下文会带出整个头文件的兄弟接口，
    不构成提及依据；正常的新接口引用（如聚合文档函数族）原文档均已存在。
    """
    allowed = set(_API_NAME_RE.findall(full_original))
    return sorted({n for n in _API_NAME_RE.findall(generated_section) if n not in allowed})


def _norm_cn(s: str) -> str:
    """中文条目匹配归一化：去项目符号/强调符/全部空白/句尾标点。"""
    s = re.sub(r"^\s*[-*]\s+", "", s.strip())
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\s+", "", s)
    return s.rstrip("。.；;，,")


def _lost_constraint_items(orig_section: str, generated_section: str) -> list[str]:
    """确定性巡检：原约束说明中的条目在重写后丢失。"""
    gen_n = _norm_cn(generated_section)
    lost = []
    for line in orig_section.splitlines():
        t = line.strip()
        if not t or t.startswith(("<!--", "<a ", "|", "#")):
            continue
        n = _norm_cn(t)
        if n and n not in gen_n:
            lost.append(t)
    return lost


def run_full_pipeline(evidence_json: str, doc_part: str, style_dir: str = None, change_reason: dict = None) -> dict:
    """Run the complete pipeline: evidence -> style -> generation -> validation."""

    # Step 0
    section("Step 0: Configuration for File Path & LLM Client")

    style_dir_path = (
        (Path(style_dir) if Path(style_dir).is_absolute() else config.PROJECT_ROOT / style_dir)
        if style_dir
        else config.RESOURCES_DIR
    )
    llm = LLMClient()

    # Step 1: Load & parse evidence
    evidence_stage, raw_input, parsed, ranked, _, evidence_path = setup_pipeline(evidence_json, llm)
    info(f"[Original Markdown Doc]\n```markdown\n{parsed.doc_original_md}\n```")

    # Step 2
    section("Step 2: Extract Evidence")
    output_dir = get_output_dir()
    extracted_path = output_dir / f"extracted_{Path(evidence_json).stem}.json"

    read_cache_for_step_2 = False
    if read_cache_for_step_2:
        info(f"  Loading cached extracted data from {extracted_path}")
        with open(extracted_path, "r", encoding="utf-8") as f:
            extracted = json.load(f)
    else:
        extracted, _source_conflicts = evidence_stage.extract_evidence(
            ranked=ranked,
            doc_part=doc_part,
            doc_original=parsed.doc_original_parsed,
            progressive=True,
            evidence_confidence_threshold=config.EVIDENCE_CONFIDENCE_THRESHOLD,
            doc_arch=parsed.doc_original_arch,
            doc_arch_sources=parsed.doc_original_arch_sources,
            doc_chip=parsed.doc_original_chip,
            doc_chip_sources=parsed.doc_original_chip_sources,
            doc_original_md=parsed.doc_original_md,
        )
        save_json(extracted, extracted_path)
        highlight(f"  Saved extracted data to {extracted_path}")

    highlight(f"  extracted fields: {', '.join(extracted.keys())}")

    # Step 3
    section("Step 3: Style Data")
    style_stage = StyleStage(llm, style_dir_path)
    style_summary = style_stage.get_style_summary()  # 即时生成 doc_path=doc_full_path；否则读取预处理 summary.json 文件
    style_extraction = style_stage.get_style_extraction()  # 读取预处理 soul.json 文件

    highlight("[Doc Summary]\n" + json.dumps(style_summary.__dict__, ensure_ascii=False, indent=4))
    debug("[Style Extraction]\n" + json.dumps(style_extraction.__dict__, ensure_ascii=False, indent=4))

    # Step 4: Generate
    section("Step 4: Generate Document Sections")
    validation_results_list = []
    final_results = []

    if doc_part == "support":
        support_matrix = extracted.get("support_matrix", {})
        supported = support_matrix.get("supported")
        md_content = support_matrix.get("md_content")
        highlight(f"supported: {supported}\nmd_content:\n```markdown\n{md_content}\n```\n")
        validation_results_list.append(
            {
                "consistency_passed": True,  # 默认通过
                "accuracy_rate": 1.0,
                "metric_passed": True,
                "edit_distance_ratio": 1.0,
                "avg_confidence": 1.0,
                "issues": [],
                "suggestion": "Auto-generated from support matrix.",
            }
        )
        final_results.append(
            {
                "sub_key": "support_matrix",
                "generated_text": md_content,
                "modification_summary": {},
            }
        )
    else:
        gen_result_path = output_dir / f"gen_result_{Path(evidence_json).stem}.json"

        generation_stage = GenerationStage(llm, style_stage.template_loader)
        validation_stage = ValidationStage(llm)

        previous_issues = []
        accumulated_issues = []
        previous_generated_doc: dict[str, str] = {}
        all_passed = False
        passed_indices = []

        for attempt in range(config.MAX_REGENERATION_RETRIES):
            if attempt > 0:
                info(f"--- Regeneration attempt {attempt + 1}/{config.MAX_REGENERATION_RETRIES} ---")
                if passed_indices:
                    info(f"  Skipping {len(passed_indices)} already-passed section(s): {passed_indices}")

            read_cache_for_step_4 = False
            if read_cache_for_step_4:
                info(f"  Loading cached generated results from {gen_result_path}")
                with open(gen_result_path, "r", encoding="utf-8") as f:
                    results = [ModificationResult(**r) for r in json.load(f)]
            else:
                results = generation_stage.generate(
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
                save_json(
                    [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
                    gen_result_path,
                )
                highlight(f"  Saved results data to {gen_result_path}")

            section(f"Step 5: Validate (attempt {attempt + 1})")

            validation_results_list.clear()
            final_results.clear()
            all_passed = True
            current_issues = []
            current_generated_doc: dict[str, str] = {}
            current_passed_indices = []

            for i, result in enumerate(results):
                info(f"Validate [{i + 1}/{len(results)}]")
                generated_text = result.generated_text
                validation_result = validation_stage.validate(
                    doc_part=doc_part,
                    doc_original=parsed.doc_original_md,
                    evidences=parsed.evidences,
                    doc_generated=generated_text,
                    evidences_extracted=extracted,
                    check_metric=True,
                    idx=i + 1,
                )
                result_dict = ValidationStage.format_result(validation_result)
                ValidationStage.print_result(validation_result)

                # 确定性巡检：内部锚点链接丢失 → 计入 issues 触发重试再生成
                heading = f"## {style_stage.template_loader.get_label(doc_part)}"
                lost = _lost_internal_links(parsed.doc_original_md, generated_text, heading)
                if lost:
                    result_dict["consistency_passed"] = False
                    result_dict["issues"] = result_dict.get("issues", []) + [
                        {
                            "id": "internal_link_lost",
                            "reason": f"原章节中的内部锚点链接 #{l} 在生成结果中丢失，必须原样保留（如 [{l} 相关描述](#{l})），除非链接指向的内容已被删除",
                        }
                        for ln in lost
                    ]
                    info(f"  [link-check] lost internal links: {lost}")

                # 确定性巡检：编造原型（原文档与证据中均不存在的签名）→ 计入 issues 触发重试
                if doc_part == "function_proto":
                    corpus = (
                        parsed.doc_original_md
                        + json.dumps(raw_input, ensure_ascii=False, default=str)
                        + json.dumps(extracted, ensure_ascii=False, default=str)
                    )
                    unsourced = _unsourced_prototypes(generated_text, corpus)
                    if unsourced:
                        result_dict["consistency_passed"] = False
                        result_dict["issues"] = result_dict.get("issues", []) + [
                            {
                                "id": "prototype_unsourced",
                                "reason": f"函数原型 `{sig}` 在原文档与证据中均不存在，疑似按已有重载模式推测编造；"
                                f"只能列出证据中实际出现的签名，必须删除无证据支撑的原型",
                            }
                            for sig in unsourced
                        ]
                        info(f"  [proto-check] unsourced prototypes: {unsourced}")

                # 确定性巡检：跨接口串台（原文档从未提及的 asc_* 接口名）→ 计入 issues 触发重试
                stray_names = _unsourced_api_names(generated_text, parsed.doc_original_md)
                if stray_names:
                    result_dict["consistency_passed"] = False
                    result_dict["issues"] = result_dict.get("issues", []) + [
                        {
                            "id": "api_name_unsourced",
                            "reason": f"接口名 `{n}` 在原文档中从未出现，疑似从头文件上下文串台；"
                            f"仅当证据明确要求引入该新接口时才可保留，否则必须删除或改回本接口相关表述",
                        }
                        for n in stray_names
                    ]
                    info(f"  [name-check] unsourced api names: {stray_names}")

                # 确定性巡检：原约束条目在重写后丢失 → 计入 issues 触发重试
                if doc_part == "constraint":
                    m = re.search(rf"(?m)^{re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", parsed.doc_original_md, re.S)
                    lost_items = _lost_constraint_items(m.group(1) if m else "", generated_text)
                    if lost_items:
                        result_dict["consistency_passed"] = False
                        result_dict["issues"] = result_dict.get("issues", []) + [
                            {
                                "id": "constraint_item_lost",
                                "reason": f"原约束条目「{t}」在重写后丢失；除非证据表明该约束已不成立，否则必须保留",
                            }
                            for t in lost_items
                        ]
                        info(f"  [constraint-check] lost constraint items: {lost_items}")

                validation_failed = not _is_validation_passed(result_dict)

                sub_key = result.modification_summary.get("sub_key", f"section_{i + 1}")
                if validation_failed:
                    all_passed = False
                    issues = result_dict.get("issues", [])
                    for issue in issues:
                        issue["sub_key"] = sub_key
                    current_issues.extend(issues)
                    current_generated_doc[sub_key] = generated_text
                else:
                    current_passed_indices.append(i)

                validation_results_list.append(result_dict)
                final_results.append(
                    {
                        "sub_key": result.modification_summary.get("sub_key"),
                        "generated_text": result.generated_text,
                        "modification_summary": result.modification_summary,
                    }
                )

            if all_passed:
                info("All validations passed.")
                break
            elif attempt < config.MAX_REGENERATION_RETRIES - 1:
                for issue in current_issues:
                    tagged = issue.copy()
                    tagged["from_attempt"] = attempt + 1
                    accumulated_issues.append(tagged)
                previous_issues = accumulated_issues
                previous_generated_doc = current_generated_doc
                passed_indices = current_passed_indices
                info(
                    f"Validation failed. Collecting {len(current_issues)} issues "
                    f"({len(accumulated_issues)} total accumulated) for regeneration..."
                )
            else:
                highlight("Max regeneration attempts reached. Proceeding with last results.")

    return {
        "parsed": {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
            "evidences_by_target": ranked.evidences_by_target,
            "doc_original": parsed.doc_original_md,
        },
        "extracted": extracted,
        "style_summary": {
            "core_topic": style_summary.core_topic,
            "key_points": style_summary.key_points,
            "context_dependencies": style_summary.context_dependencies,
        },
        "style_extraction": {
            "terminology": style_extraction.terminology,
            "structure": style_extraction.structure,
            "format": style_extraction.format,
        },
        "generated": final_results,
        "validation": validation_results_list,
    }


def validate_doc_with_evidences(evidence_json: str, doc_part: str) -> dict:
    """Run the partial pipeline: evidence -> validation."""

    # Step 0 & 1: Setup pipeline and load evidence
    section("Step 0: Configuration for File Path & LLM Client")

    evidence_stage, raw_input, parsed, ranked, llm, evidence_path = setup_pipeline(evidence_json)

    # Step 2
    section("Step 2: Validate")

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
        supported = support_matrix.get("supported")
        md_content = support_matrix.get("md_content")
        highlight(f"supported: {supported}\nmd_content:\n```markdown\n{md_content}\n```\n")
        parsed.evidences = [
            {
                "evidence_id": "ev_001",
                "type": "section",
                "target": "support_matrix",
                "content": {
                    "supported": supported,
                    "md_content": md_content,
                },
                "confidence": 1.0,
                "category": "fact",
            }
        ]

        indices = [None]
    else:
        indices = [None]

    for idx in indices:
        if isinstance(idx, int):
            doc_orig = parsed.doc_original_md
        else:
            doc_orig = parsed.doc_original_md
        validation_result = validation_stage.validate(
            doc_part=doc_part,
            doc_original=doc_orig,
            evidences=parsed.evidences,
            check_metric=False,
            idx=idx,
        )
        result_dict = ValidationStage.format_result(validation_result)
        ValidationStage.print_result(validation_result)
        validation_results_list.append(result_dict)

    return {
        "parsed": {
            "doc_part": parsed.doc_part,
            "total_evidences": parsed.total_evidences,
            "evidences_by_target": ranked.evidences_by_target,
            "doc_original": parsed.doc_original_md,
        },
        "extracted": {},
        "style_summary": {},
        "style_extraction": {},
        "generated": [],
        "validation": validation_results_list,
    }


def run_pipeline(subparts, evidence_json, mode="full"):
    start = datetime.now(tz=timezone.utc)
    if mode == "full":
        result = run_full_pipeline(evidence_json=evidence_json, doc_part=subparts, style_dir="resources")
    elif mode == "validate_only":
        result = validate_doc_with_evidences(evidence_json=evidence_json, doc_part=subparts)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'full' or 'validate_only'.")
    end = datetime.now(tz=timezone.utc)

    if mode == "full":
        debug("FULL PIPELINE RESULT SUMMARY")
    else:
        debug("VALIDATION RESULT SUMMARY")
    debug("\n" + json.dumps(result, ensure_ascii=False, indent=2))

    if mode == "full":
        generated_list = result.get("generated", [])
        for item in generated_list:
            text = item.get("generated_text", "")
            if text:
                info(f"生成的文档内容：\n{text}")

    total_seconds = (end - start).total_seconds()
    total_minutes = total_seconds / 60
    highlight(f"runtime: {total_seconds:.2f} s ({total_minutes:.2f} min)")


def run_generate_and_validate(subparts, evidence_json):
    run_pipeline(subparts, evidence_json, mode="full")


def run_validate_only(subparts, evidence_json):
    run_pipeline(subparts, evidence_json, mode="validate_only")


if __name__ == "__main__":
    subparts = config.DOC_PARTS[2]
    evidence_json = f"data/evidences/example/aclnnAbs.md-{subparts}.json"
    run_generate_and_validate(subparts, evidence_json)
