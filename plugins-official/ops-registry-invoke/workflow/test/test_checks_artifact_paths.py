# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1].parent

OLD_REVIEW_PATHS = [
    "operators/{operator_name}/docs/SPEC_REVIEW.md",
    "operators/{operator_name}/docs/DESIGN_REVIEW.md",
    "operators/{operator_name}/docs/TEST_REVIEW.md",
    "operators/{operator_name}/docs/review-report.md",
    "{绝对路径}/{算子仓}/{子类别}/{算子名称}/docs/SPEC_REVIEW.md",
    "{绝对路径}/{算子仓}/{子类别}/{算子名称}/docs/DESIGN_REVIEW.md",
    "{绝对路径}/{算子仓}/{子类别}/{算子名称}/docs/TEST_REVIEW.md",
    "{绝对路径}/{算子仓}/{子类别}/{算子名称}/docs/review-report.md",
]

NEW_REVIEW_PATHS = [
    "operators/{operator_name}/tmp/checks/SPEC_REVIEW.md",
    "operators/{operator_name}/tmp/checks/DESIGN_REVIEW.md",
    "operators/{operator_name}/tmp/checks/TEST_REVIEW.md",
    # review-report.md 已删除 — 代码检视产物统一放 tmp/checks/
    # （{source_file}_review_summary.md / code_summary.md / api_prestudy.md）
]

OLD_REVIEW_INSTRUCTIONS = [
    "按 SPEC_REVIEW 修订 spec.yaml",
]

CODE_REVIEW_INPUTS = [
    "- 检视文件: operators/{operator_name}/op_kernel/ + op_host/ 下所有 .cpp/.h/.hpp",
    "- 报告路径: operators/{operator_name}/tmp/checks/{source_file}_review_summary.md",
    "- 概要分析输出：`operators/{operator_name}/tmp/checks/code_summary.md`",
    "- API 预研报告（如有）：`operators/{operator_name}/tmp/checks/api_prestudy.md`",
]


def production_markdown_files() -> list[Path]:
    roots = [PLUGIN_ROOT / "workflow", PLUGIN_ROOT / "agents"]
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*.md"):
            relative_parts = path.relative_to(PLUGIN_ROOT).parts
            if "test" in relative_parts:
                continue
            files.append(path)
    return sorted(files)


def read_all_checked_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in production_markdown_files())


def extract_section(text: str, heading: str) -> str:
    start = text.index(heading)
    next_heading = text.find("\n## ", start + len(heading))
    if next_heading == -1:
        return text[start:]
    return text[start:next_heading]


def stale_docs_review_rows(text: str) -> list[str]:
    artifact_names = ("SPEC_REVIEW.md", "DESIGN_REVIEW.md", "TEST_REVIEW.md")
    stale_rows: list[str] = []
    for line in text.splitlines():
        if "operators/{operator_name}/docs/" not in line:
            continue
        if any(name in line for name in artifact_names):
            stale_rows.append(line)
    return stale_rows


def test_review_artifacts_use_tmp_checks_directory() -> None:
    text = read_all_checked_text()
    stale_rows = [
        f"{path.relative_to(PLUGIN_ROOT)}: {line}"
        for path in production_markdown_files()
        for line in stale_docs_review_rows(path.read_text(encoding="utf-8"))
    ]

    assert stale_rows == [], "Review artifact rows still use docs directory:\n" + "\n".join(stale_rows)

    for old_path in OLD_REVIEW_PATHS:
        assert old_path not in text

    for old_instruction in OLD_REVIEW_INSTRUCTIONS:
        assert old_instruction not in text

    for new_path in NEW_REVIEW_PATHS:
        assert new_path in text


def test_code_review_prompt_uses_explicit_documents_not_docs_directory() -> None:
    task_prompts = (PLUGIN_ROOT / "workflow/resources/task-prompts.md").read_text(encoding="utf-8")
    code_review_prompt = extract_section(task_prompts, "## 4.2 代码检视")

    assert "- 设计文档：operators/{operator_name}/docs/" not in code_review_prompt
    assert "- 算子 README：operators/{operator_name}/README.md" not in code_review_prompt
    for code_review_input in CODE_REVIEW_INPUTS:
        assert code_review_input in code_review_prompt
