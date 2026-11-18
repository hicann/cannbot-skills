# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run All CLI - Entry point orchestrating generation and validation workflows.

Usage:
  Interactive:  python run_all_cli.py
  Non-interactive:
    python src/run_all_cli.py --original_op_dir <md_file> --modified_op_dir <md_file> \
        [--validate-only] [--skip-fix]
"""

import argparse
import json
import logging
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import questionary
from rich.console import Console
from rich.text import Text

from src.change_detection.doc_part_change_detection.docpart_file_extractor import DocpartFileExtractor
from src.generation_validation.config import (
    ENABLE_PRE_VALIDATION,
    GEN_VAL_MAX_WORKERS,
    POST_ANALYZE_NO_CHANGE_THRESHOLD,
    format_doc_part,
)
from src.generation_validation.dashboard import (
    ParallelDashboard,
    set_dashboard_context,
)
from src.generation_validation.ui.cli import (
    LogCapture,
    PipelineResult,
    display_header,
)
from src.generation_validation.ui.cli import display_results as _display_results
from src.generation_validation.ui.cli import (
    resolve_evidence_file,
    run_full_pipeline,
    run_validate_only,
    select_doc_part,
    select_mode,
)
from src.logger import (
    debug,
    error,
    highlight,
    info,
    redirect_stdout_to_log,
    section,
    setup_logger,
    subsection,
    warning,
)

from src.api_locator.config import DOC_TO_SUB_PART

CANCEL_SENTINEL = "CANCEL_SENTINEL"

_SUB_PART_NAMES = set(DOC_TO_SUB_PART.values())


def _normalize_doc_parts(parts: list[str], console: Optional[Console] = None) -> list[str]:
    if not parts:
        return []
    normalized = []
    failed = []
    for p in parts:
        if p in DOC_TO_SUB_PART:
            normalized.append(DOC_TO_SUB_PART[p])
        elif p in _SUB_PART_NAMES:
            normalized.append(p)
        else:
            failed.append(p)
    if failed:
        msg = f"无法归一化 doc_parts: {failed}. Available: {list(DOC_TO_SUB_PART.keys())}"
        error(msg)
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
    if not normalized and failed:
        raise ValueError(f"All doc parts failed to normalize: {parts}")
    return normalized


_DOC_PART_HEADING = {
    "support": "## 产品支持情况",
    "summary": "## 功能说明",
    "impl_principle": "## 实现原理",
    "function_proto": "## 函数原型",
    "para_clarify": "## 参数说明",
    "return_clarify": "## 返回值说明",
    "constraint": "## 约束说明",
    "example": "## 调用示例",
}


@dataclass
class PipelineContext:
    """Holds all paths, configuration, and mutable state for a single pipeline run."""

    workspace_dir: Path
    output_data_path: Path
    original_op_dir: Optional[Path] = None
    modified_op_dir: Optional[Path] = None
    original_repo_root: Optional[Path] = None
    modified_repo_root: Optional[Path] = None
    selected_op: str = ""
    validate_only: bool = False
    skip_fix: bool = False
    interactive: bool = False
    dashboard: bool = False
    # 聚合模式：外部预先构建好的快照目录（含 anchor/refactor/extra/doc_json），
    # 提供后跳过 api_locator 定位。默认 None 走原流程，单函数场景零影响。
    prebuilt_original_dir: Optional[Path] = None
    prebuilt_modified_dir: Optional[Path] = None

    snapshot_op_dir: Path = field(default_factory=lambda: Path("."))
    evidence_dir: Path = field(default_factory=lambda: Path("."))
    summary_dir: Path = field(default_factory=lambda: Path("."))
    selected_file: Optional[Path] = None
    doc_content: str = ""
    summary_file_dir: Path = field(default_factory=lambda: Path("."))
    re_detection_output_path: Path = field(default_factory=lambda: Path("."))

    doc_parts: list[str] = field(default_factory=list)
    modified_files: list = field(default_factory=list)
    arch_scope: list = field(default_factory=list)
    changed_files: list = field(default_factory=list)  # PR 全量改动（相对仓库根）
    extra_parts: list = field(default_factory=list)  # 快照补拷继承到的 doc_part
    extra_code_text: str = ""  # PR 全量代码新增行（必写要点来源）
    extra_code_removed: str = ""  # PR 全量代码删除行（重命名识别来源）
    evaluation_reasons: dict[str, dict] = field(default_factory=dict)
    backfill_map: dict[str, str] = field(default_factory=dict)
    passed_map: dict[str, bool] = field(default_factory=dict)
    detailed_results: dict[str, PipelineResult] = field(default_factory=dict)
    failed_doc_parts: list[str] = field(default_factory=list)
    pre_val_passed_map: dict[str, bool] = field(default_factory=dict)
    pre_val_detailed_results: dict[str, PipelineResult] = field(default_factory=dict)
    pre_check_inconsistent_list: list[str] = field(default_factory=list)

    elapsed: dict[str, float] = field(default_factory=dict)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    project_root = Path(__file__).resolve().parent.parent
    resolved = project_root / path
    if resolved.exists():
        return resolved.resolve()
    return path.resolve()


def _log_and_print(console: Console, msg: str, style: str = "", level: str = "info"):
    """Write a message to both the logger and the rich console."""
    {"info": info, "warn": lambda m: info(f"[WARN] {m}"), "error": error, "debug": debug, "highlight": highlight}[
        level
    ](msg)
    console.print(Text(msg, style=style) if style else msg)


def _run_step(console: Console, name: str, fn: Callable[[], Any]) -> Any:
    """Execute a pipeline step with unified timing, logging, and error handling."""
    subsection(name)
    start = time.time()
    info(f"{name} 开始")
    try:
        result = fn()
        elapsed = time.time() - start
        highlight(f"{name} 完成 ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start
        error(f"{name} 失败 ({elapsed:.2f}s): {e}")
        console.print(Text(f"{name} failed: {e}", style="bold red"))
        raise


def _auto_select_doc_file(console: Console, doc_files: list[Path]) -> Optional[Path]:
    default_files = [f for f in doc_files if f.name.startswith("aclnn")]
    if not default_files:
        console.print("    [yellow]未找到默认格式文档。[/yellow]")
        return None
    selected = default_files[0]
    console.print(f"    [dim]自动选择（首个默认）:[/dim] [cyan]{selected.name}[/cyan]")
    return selected


def _interactive_select_doc_file(console: Console, doc_files: list[Path]) -> Optional[Path]:
    categories: dict[str, list[Path]] = {"default": [], "inplace": [], "compound": [], "other": []}
    for f in doc_files:
        name = f.name
        if "&" in name:
            categories["compound"].append(f)
        elif "Inplace" in name:
            categories["inplace"].append(f)
        elif name.startswith("aclnn"):
            categories["default"].append(f)
        else:
            categories["other"].append(f)

    default_files = categories["default"]
    inplace_files = categories["inplace"]
    compound_files = categories["compound"]
    other_files = categories["other"]

    if len(default_files) == 1 and not any(v for k, v in categories.items() if k != "default"):
        console.print(f"    [dim]自动选择（默认）:[/dim] [cyan]{default_files[0].name}[/cyan]")
        return default_files[0]

    if not default_files:
        console.print("    [yellow]未找到默认格式文档。[/yellow]")
        return None

    has_unsupported = inplace_files or compound_files or other_files
    if has_unsupported:
        console.print(f"    [dim]找到 {len(doc_files)} 个 .md 文件:[/dim]")
        console.print(f"        [dim]默认:[/dim]       {len(default_files)} file(s)")
        if inplace_files:
            console.print(f"        [yellow]原位:[/yellow]     {len(inplace_files)} file(s)")
        if compound_files:
            console.print(f"        [yellow]复合:[/yellow]    {len(compound_files)} file(s)")
        if other_files:
            console.print(f"        [dim]其他:[/dim]        {len(other_files)} file(s)")
        console.print(f"    [dim]目前不支持 Inplace/Compound/Other 格式的处理。[/dim]")
        console.print()

    if len(default_files) == 1:
        selected = questionary.confirm(f"处理 {default_files[0].name}？", default=True).ask()
        if selected:
            return default_files[0]

    choices = [questionary.Choice([("bold green", "(*) "), ("", f.name)], value=f) for f in default_files]
    console.print()
    choices.append(questionary.Choice([("bold red", "取消")], value=CANCEL_SENTINEL))

    try:
        selected = questionary.select("选择要处理的文档（回车确认）:", choices=choices).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if selected in (CANCEL_SENTINEL, None):
        return None
    return selected


def _interactive_select_md_file(console: Console, root_dir: Path) -> Optional[Path]:
    """Navigate directory tree and select an .md file. Supports forward/backward navigation."""
    current_dir = root_dir

    while True:
        console.print()
        console.print(f"  [dim]Current:[/dim] [cyan]{current_dir}[/cyan]")

        subdirs = sorted(d for d in current_dir.iterdir() if d.is_dir())
        md_files = sorted(f for f in current_dir.iterdir() if f.is_file() and f.suffix == ".md")

        choices: list[questionary.Choice] = []

        if current_dir != root_dir:
            choices.append(questionary.Choice([("bold yellow", ".. (返回)")], value="__BACK__"))

        for d in subdirs:
            choices.append(questionary.Choice([("bold cyan", f"📁 {d.name}/")], value=("dir", d)))

        for f in md_files:
            choices.append(questionary.Choice([("bold green", f"📄 {f.name}")], value=("file", f)))

        if not choices:
            console.print("[red]此处未找到子目录或 .md 文件。[/red]")
            if current_dir != root_dir:
                current_dir = current_dir.parent
                continue
            return None

        choices.append(questionary.Choice([("bold red", "取消")], value="__CANCEL__"))

        try:
            selected = questionary.select(
                "导航目录或选择文档:",
                choices=choices,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None

        if selected is None or selected == "__CANCEL__":
            return None

        if selected == "__BACK__":
            current_dir = current_dir.parent
            continue

        kind, path = selected

        if kind == "dir":
            current_dir = path
            continue

        selected_file: Path = path
        console.print()
        console.print(f"  [dim]已选文档:[/dim] [bold green]{selected_file.name}[/bold green]")
        console.print(f"  [dim]完整路径:[/dim] {selected_file}")
        console.print()
        return selected_file


def _interactive_snapshot(console: Console, workspace_dir: Path) -> tuple[Optional[Path], str]:
    root_input = questionary.text(
        "输入算子仓库路径:",
        default="./data/asc-devkit-master/docs/api/SIMD-API/高阶API/数学计算/Acos接口/",
    ).ask()
    if not root_input:
        return None, ""

    root_dir = Path(root_input.rstrip("/").rstrip("\\"))
    if not root_dir.is_absolute():
        root_dir = (workspace_dir / root_dir).resolve()
    if not root_dir.is_dir():
        console.print(f"[red]目录未找到:[/red] [cyan]{root_dir}[/cyan]")
        return None, ""

    selected_file = _interactive_select_md_file(console, root_dir)
    if selected_file is None:
        return None, ""

    op_name = selected_file.stem
    return selected_file, op_name


def _find_repo_root(file_path: Path) -> Optional[Path]:
    """Derive repo root from a file path by walking up to find a directory containing 'docs/'."""
    current = file_path.parent if file_path.is_file() else file_path
    while current != current.parent:
        if (current / "docs").is_dir():
            return current
        current = current.parent
    return None


def _run_locator(console: Console, ctx: PipelineContext, doc_file: Path) -> bool:
    """Common locator logic shared by interactive and non-interactive flows."""
    if not doc_file.is_file():
        _log_and_print(console, f"文件未找到: {doc_file}", style="bold red")
        return False

    ctx.selected_file = doc_file
    ctx.selected_op = doc_file.stem
    ctx.doc_content = doc_file.read_text(encoding="utf-8")

    repo_root = _find_repo_root(doc_file)
    if repo_root is None:
        _log_and_print(console, f"无法从以下路径确定仓库根目录: {doc_file}", style="bold red")
        return False

    ctx.original_repo_root = repo_root

    info(f"算子: {ctx.selected_op}")
    console.print(f"  [dim]算子:[/dim] [cyan]{ctx.selected_op}[/cyan]")
    console.print(f"  [dim]文档文件:[/dim] {doc_file}")
    console.print()

    export_root = ctx.output_data_path / "Original" / ctx.selected_op
    if ctx.prebuilt_original_dir is not None:
        # 聚合模式：使用外部预构建快照，跳过定位
        export_root = Path(ctx.prebuilt_original_dir)
        info(f"使用预构建快照（聚合模式）: {export_root}")
        console.print(f"  [dim]使用预构建快照（聚合模式）:[/dim] {export_root}")
        console.print()
        ctx.snapshot_op_dir = export_root
        ctx.evidence_dir = export_root / "anchor"
        ctx.summary_dir = export_root / "style"
        return True

    source_doc_marker = export_root / ".source_doc"
    cached_doc = source_doc_marker.read_text(encoding="utf-8").strip() if source_doc_marker.exists() else None
    if export_root.is_dir() and any(export_root.iterdir()) and cached_doc == str(doc_file.resolve()):
        info(f"缓存命中: {export_root}")
        console.print(f"  [dim]缓存命中，跳过定位:[/dim] {export_root}")
        console.print()
    else:

        def _run():
            from src.api_locator.cli import run as locator_run

            with redirect_stdout_to_log():
                return locator_run(
                    doc_path=str(doc_file),
                    repo_root=str(repo_root),
                    output_dir=str(ctx.output_data_path / "Original"),
                )

        export_root = Path(_run_step(console, "API 定位", _run))
        export_root.mkdir(parents=True, exist_ok=True)
        source_doc_marker.write_text(str(doc_file.resolve()), encoding="utf-8")
        console.print(f"  [dim]导出根目录:[/dim] {export_root}")
        console.print()

    ctx.snapshot_op_dir = export_root
    ctx.evidence_dir = export_root / "anchor"
    ctx.summary_dir = export_root / "style"
    return True


def _step_api_change_detect(console: Console, ctx: PipelineContext) -> bool:
    """Step 0: Detect API changes between two repos, let user select affected doc."""
    console.rule("[bold magenta]API 变更检测[/bold magenta]")

    old_repo_input = questionary.text(
        "输入原始仓库路径:",
        default="./data/asc-devkit-master",
    ).ask()
    if not old_repo_input:
        return False
    old_repo_dir = _resolve_path(old_repo_input)
    if not old_repo_dir.is_dir():
        _log_and_print(console, f"原始仓库未找到: {old_repo_dir}", style="bold red")
        return False

    new_repo_input = questionary.text(
        "输入修改版仓库路径:",
        default="./data/asc-devkit-master_modified",
    ).ask()
    if not new_repo_input:
        return False
    new_repo_dir = _resolve_path(new_repo_input)
    if not new_repo_dir.is_dir():
        _log_and_print(console, f"修改版仓库未找到: {new_repo_dir}", style="bold red")
        return False

    console.print()
    info(f"原始仓库: {old_repo_dir}")
    info(f"修改版仓库: {new_repo_dir}")
    console.print(f"  [dim]原始仓库:[/dim] {old_repo_dir}")
    console.print(f"  [dim]修改版仓库:[/dim] {new_repo_dir}")
    console.print()

    def _run():
        import importlib
        import os as _os
        import sys as _sys

        _insert_path = _os.path.join(_os.path.dirname(__file__), "api_change_detect")
        _sys.path.insert(0, _insert_path)
        try:
            pipeline_mod = importlib.import_module("pipeline")
            return pipeline_mod.detect_api_changes(str(old_repo_dir), str(new_repo_dir), verbose=False)
        finally:
            _sys.path.remove(_insert_path)
            for mod_name in list(_sys.modules):
                mod = _sys.modules[mod_name]
                if getattr(mod, "__file__", None) and _insert_path in (mod.__file__ or ""):
                    del _sys.modules[mod_name]

    try:
        affected_apis = _run_step(console, "检测 API 变更", _run)
    except Exception as e:
        _log_and_print(console, f"API 变更检测失败: {e}", style="bold red", level="error")
        return False

    if not affected_apis:
        _log_and_print(
            console,
            f"无受变更影响的 API。\n  Original: {old_repo_dir}\n  Modified: {new_repo_dir}",
            style="yellow",
            level="warn",
        )
        return False

    console.print()
    console.print(f"  [bold]受影响的 API ({len(affected_apis)}):[/bold]")

    api_doc_map: dict[str, Path] = {}
    for api_name in sorted(affected_apis.keys()):
        doc_rel_path = affected_apis[api_name]
        if doc_rel_path:
            doc_abs_path = old_repo_dir / doc_rel_path
            if doc_abs_path.is_file():
                api_doc_map[api_name] = doc_abs_path
                console.print(f"    [cyan]{api_name}[/cyan]  ←  {doc_rel_path}")
            else:
                console.print(f"    [dim]{api_name}[/dim]  [yellow]（文档未找到: {doc_rel_path}）[/yellow]")
        else:
            console.print(f"    [dim]{api_name}[/dim]  [yellow]（无文档路径）[/yellow]")

    if not api_doc_map:
        api_list = ", ".join(sorted(affected_apis.keys()))
        _log_and_print(
            console,
            f"找到 {len(affected_apis)} 个受影响的 API 但无匹配的文档文件。\n"
            f"  APIs: {api_list}\n"
            f"  Search root: {old_repo_dir}",
            style="yellow",
            level="warn",
        )
        return False

    choices = [
        questionary.Choice(
            [("bold cyan", f"{name}"), ("", f"  ({doc.relative_to(old_repo_dir)})")],
            value=(name, doc),
        )
        for name, doc in api_doc_map.items()
    ]
    choices.append(questionary.Choice([("bold red", "Cancel")], value=CANCEL_SENTINEL))

    console.print()
    try:
        selected = questionary.select("选择要处理的 API:", choices=choices).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if selected is None or selected == CANCEL_SENTINEL:
        return False

    api_name, doc_file = selected
    ctx.selected_op = api_name
    ctx.selected_file = doc_file
    ctx.doc_content = doc_file.read_text(encoding="utf-8")
    ctx.modified_repo_root = new_repo_dir

    console.print()
    console.print(f"  [dim]已选 API:[/dim] [cyan]{api_name}[/cyan]")
    console.print(f"  [dim]文档文件:[/dim] {doc_file}")
    console.print()

    return True


def _step_snapshot(console: Console, ctx: PipelineContext) -> bool:
    """Step 1: Build operator code snapshot. Returns False to abort."""
    console.rule("[bold magenta]快照构建[/bold magenta]")

    if ctx.selected_file:
        doc_file = ctx.selected_file
    elif ctx.interactive:
        doc_file, ctx.selected_op = _interactive_snapshot(console, ctx.workspace_dir)
        if doc_file is None:
            return False
    else:
        doc_file = ctx.original_op_dir

    return _run_locator(console, ctx, doc_file)


def _step_config(console: Console, ctx: PipelineContext):
    """Step: Select mode."""
    subsection("配置：模式与路径")

    if ctx.interactive:
        console.print()
        mode = select_mode()
        if not mode:
            error("未选择模式")
            return
        ctx.validate_only = mode == "validate_only"

    info(f"Validate only: {ctx.validate_only}")
    info(f"Selected file: {ctx.selected_file.name if ctx.selected_file else None}")
    console.print()


def _step_summary(console: Console, ctx: PipelineContext):
    """Step 2: Generate original doc summary."""
    console.rule("\n[bold yellow]生成原始文档摘要[/bold yellow]")

    if ctx.selected_file is None:
        return

    filename = f"{ctx.selected_file.stem}-summary.json"
    ctx.summary_file_dir = ctx.summary_dir / filename

    def _run():
        from generation_validation.style.summarizer import gen_doc_summarize

        info(f"正在生成摘要 {ctx.selected_file.name}...")
        console.print(f"[dim]正在生成摘要 {ctx.selected_file.name}...[/dim]")
        gen_doc_summarize(ctx.doc_content, str(ctx.summary_file_dir), force=False)

    _run_step(console, "摘要生成", _run)
    _log_and_print(console, f"{ctx.selected_file.name} -> {filename}", style="green")


def _merge_precheck_inconsistent(
    doc_parts: list[str],
    pre_check_inconsistent_list: list[str],
    evaluation_reasons: Optional[dict],
) -> tuple[list[str], list[str]]:
    """合并评估通过的 doc_parts 与预校验不一致项，排除评估已判定无变化(keep=false)的项。

    预校验的 support 比对基于静态 chip 提取，证据不完整时会产生假阳性
    SUPPORT_MISMATCH（如算子代码无芯片注册信息，矩阵被全判"不支持"）。
    change_evaluator 的评估带有完整代码 diff 上下文，其 keep=false 结论
    （如"产品支持决策无变化"）优先于预校验失败，避免误重写章节。
    未被评估覆盖的 doc_part（不在 evaluation_reasons 中）保持原行为，仍参与合并。

    Returns:
        (merged_list, suppressed_list) — merged 为最终处理列表，suppressed 为被排除项
    """
    reasons = evaluation_reasons or {}
    suppressed = [p for p in pre_check_inconsistent_list if p in reasons and not reasons[p].get("keep", True)]
    kept = [p for p in pre_check_inconsistent_list if p not in suppressed]
    merged = list(dict.fromkeys(doc_parts + kept))
    return merged, suppressed


def _doc_changed_parts(original_md: str, modified_md: str) -> list[str]:
    """对比 PR 前后两份文档，找出内容有变化的章节（doc_part 代号）。

    PR 自己改了资料 = 这些章节本次必须重生成。单靠代码 diff 会漏掉
    “代码里看不出、但资料确实要跟着改”的语义变更（如取值范围 2^21→2^22）。
    """
    from src.generation_validation.config import DOC_PART_LABELS

    changed = []
    for part, label in DOC_PART_LABELS.items():
        before = _extract_md_section(original_md or "", label)
        after = _extract_md_section(modified_md or "", label)
        if before.strip() != after.strip():
            changed.append(part) if part not in changed else None
    return changed


_SNAPSHOT_ROOT = "refactor"


def _changed_code_added_text(ctx: PipelineContext, limit_lines: int = 4000, side: str = "+") -> str:
    """PR 全量代码改动的“新增行”文本（仅 include/ 与 impl/）。

    文档快照常只收录部分文件，仅靠快照内 diff 会漏掉类型/枚举的定义处
    （如 RegTraitNumTwo 定义在别的头文件），这里直接从两侧 worktree 做行级 diff。
    """
    if not ctx.changed_files or not ctx.original_repo_root or not ctx.modified_repo_root:
        return ""
    import difflib

    added: list[str] = []
    for raw in ctx.changed_files:
        rel = str(raw).replace("\\", "/").lstrip("./")
        if not (rel.startswith("include/") or rel.startswith("impl/")):
            continue
        a = ctx.original_repo_root / rel
        b = ctx.modified_repo_root / rel
        if not (a.is_file() and b.is_file()):
            continue
        try:
            ta = a.read_text(encoding="utf-8", errors="replace").splitlines()
            tb = b.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in difflib.unified_diff(ta, tb, n=0):
            if line.startswith(side) and not line.startswith(side * 3):
                added.append(line[1:])
        if len(added) >= limit_lines:
            break
    return "\n".join(added[:limit_lines])


def _stem_trunk(name: str) -> str:
    """去掉 _intf_impl / _impl / _intf 后缀的“主干名”，用于同族判定。"""
    stem = Path(name).stem
    for suf in ("_intf_impl", "_impl", "_intf"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _iter_snapshot_files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()]


def _snapshot_includes(root: Path, rel_files: list[str]) -> set[str]:
    """快照内源码文件的 include 目标（basename 集合），用于反向 include 判定。"""
    targets: set[str] = set()
    for rel in rel_files:
        if not rel.endswith((".h", ".hpp", ".cpp")):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for inc in re.findall(r'#\s*include\s*"([^"]+)"', text):
            targets.add(Path(inc).name)
    return targets


def _anchor_file_parts(evidence_dir: Path) -> dict[str, list[str]]:
    """{证据里出现的文件相对路径: [doc_part...]}（扫 anchor/*.json 的 file_path）。"""
    mapping: dict[str, list[str]] = {}
    if not evidence_dir.is_dir():
        return mapping
    for f in evidence_dir.glob("*.json"):
        part = f.stem
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in re.findall(r'"file_path"\s*:\s*"([^"]+)"', text):
            rel = raw.replace("\\", "/")
            bucket = mapping.setdefault(rel, [])
            if part not in bucket:
                bucket.append(part)
    return mapping


def _augment_snapshot(ctx: PipelineContext, console: Console) -> list[str]:
    """把 PR 改动里“快照缺失”的文件补拷进两侧快照（R1/R2/R3）。

    背景：证据快照按文件名匹配收录，会漏掉分架构目录（dav_*）下的同族实现，
    导致“文件被改、但两侧快照里都没有它”，变更检测 diff 恒为空。

    规则（命中任一）：
      R1 同族主干：改动文件与快照内某文件去掉 _impl/_intf 后缀的主干相同；
      R2 反向 include：改动文件被快照内某头文件 include；
      R3 符号出现：改动文件内容出现本文档 API 符号（含 _impl 形式）。

    补拷成对进行（before / now 各一份），并把被匹配快照文件的 doc_part 映射
    继承给新文件 —— 否则 affected_subparts 仍为空，变更评估会早退。

    Returns: 继承到的 doc_part 列表
    """
    if not ctx.changed_files or not ctx.original_repo_root or not ctx.modified_repo_root:
        return []
    before_snap = ctx.snapshot_op_dir / _SNAPSHOT_ROOT
    now_snap = ctx.modified_op_dir / _SNAPSHOT_ROOT
    if not before_snap.is_dir() or not now_snap.is_dir():
        return []

    snap_files = _iter_snapshot_files(before_snap)
    # 同族匹配优先 impl/、其次 include/、最后 tests/ 等，保证继承到的 doc_part 更贴近实现
    snap_files.sort(key=lambda p: (not p.startswith("impl/"), not p.startswith("include/"), len(p)))
    snap_set = set(snap_files)
    includes = _snapshot_includes(before_snap, snap_files)
    parts_map = _anchor_file_parts(ctx.evidence_dir)
    api = ctx.selected_op or ""

    copied: list[tuple[str, str, str]] = []  # (rel, matched, rule)
    for raw in ctx.changed_files:
        rel = str(raw).replace("\\", "/").lstrip("./")
        if not rel or rel in snap_set:
            continue
        src_before = ctx.original_repo_root / rel
        src_now = ctx.modified_repo_root / rel
        if not (src_before.is_file() and src_now.is_file()):
            continue  # 仅一侧存在（新增/删除）不在此规则内
        matched, rule = "", ""
        trunk = _stem_trunk(rel)
        for sf in snap_files:
            if _stem_trunk(sf) == trunk:
                matched, rule = sf, "R1"
                break
        if not matched and Path(rel).name in includes:
            matched = next((sf for sf in snap_files if Path(sf).parent == Path(rel).parent), snap_files[0])
            rule = "R2"
        if not matched and api:
            try:
                body = src_now.read_text(encoding="utf-8", errors="replace")
            except OSError:
                body = ""
            if re.search(rf"\b{re.escape(api)}(?:_impl)?\b", body):
                matched, rule = snap_files[0], "R3"
        if not matched:
            continue
        for root, snap_root in ((ctx.original_repo_root, before_snap), (ctx.modified_repo_root, now_snap)):
            dst = snap_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dst)
        copied.append((rel, matched, rule))

    if not copied:
        return []

    inherited: list[str] = []
    for rel, matched, rule in copied:
        parts = parts_map.get(matched, [])
        for p in parts:
            if p not in inherited:
                inherited.append(p)
        console.print(f"[dim]  快照补拷[{rule}]: {rel} ← 同族 {matched} （继承 doc_part: {parts or '无'}）[/dim]")
        info(f"快照补拷[{rule}] {rel} <- {matched} parts={parts}")
    return inherited


def _step_change_detection(console: Console, ctx: PipelineContext):
    """Step 3: Detect code changes between original and modified operator."""
    console.rule("[bold yellow]变更检测[/bold yellow]")

    if ctx.modified_repo_root:
        old_repo_root = _find_repo_root(ctx.selected_file)
        if old_repo_root:
            doc_rel_path = ctx.selected_file.relative_to(old_repo_root)
            modified_doc_file = ctx.modified_repo_root / doc_rel_path
            if not modified_doc_file.is_file():
                _log_and_print(console, f"修改版文档未找到: {modified_doc_file}", style="bold red")
                return
            console.print(f"  [dim]修改版文档（自动推导）:[/dim] {modified_doc_file}")
            console.print()
        else:
            _log_and_print(console, "无法确定原始仓库根目录", style="bold red")
            return
    elif ctx.interactive:
        console.print("现在，你可以修改算子代码。")
        console.print()
        if not questionary.confirm("完成？", default=True).ask():
            return
        console.print()
        modified_repo_input = questionary.text(
            "输入'修改版'仓库路径:",
            default=f"./data/asc-devkit-master_modified/docs/api/SIMD-API/高阶API/数学计算/Acos接口",
        ).ask()
        modified_repo_dir = _resolve_path(modified_repo_input) if modified_repo_input else None
        if not modified_repo_dir or not modified_repo_dir.is_dir():
            _log_and_print(console, f"修改版仓库未找到: {modified_repo_dir}", style="bold red")
            return
        console.print()

        console.print("[bold]选择修改版文档:[/bold]")
        modified_doc_file = _interactive_select_md_file(console, modified_repo_dir)
        if modified_doc_file is None:
            return
        ctx.modified_repo_root = _find_repo_root(modified_doc_file)
        if ctx.modified_repo_root is None:
            _log_and_print(console, f"无法从以下路径确定仓库根目录: {modified_doc_file}", style="bold red")
            return

        console.print()
    else:
        modified_doc_file = ctx.modified_op_dir
        if not modified_doc_file or not modified_doc_file.is_file():
            _log_and_print(console, f"修改版文档文件未找到: {modified_doc_file}", style="bold red")
            return
        ctx.modified_repo_root = _find_repo_root(modified_doc_file)
        if ctx.modified_repo_root is None:
            _log_and_print(console, f"无法从以下路径确定仓库根目录: {modified_doc_file}", style="bold red")
            return

    if not ctx.evidence_dir.is_dir():
        return

    info(f"修改版仓库根目录: {ctx.modified_repo_root}")
    info(f"修改版文档文件: {modified_doc_file}")
    console.print(f"  [dim]修改版仓库根目录:[/dim] {ctx.modified_repo_root}")
    console.print(f"  [dim]修改版文档文件:[/dim] {modified_doc_file}")

    def _run_locator_for_modified():
        from src.api_locator.cli import run as locator_run

        with redirect_stdout_to_log():
            export_root = locator_run(
                doc_path=str(modified_doc_file),
                repo_root=str(ctx.modified_repo_root),
                output_dir=str(ctx.output_data_path / "Modified"),
            )
        return Path(export_root)

    if ctx.prebuilt_modified_dir is not None:
        # 聚合模式：使用外部预构建的修改版快照，跳过定位
        ctx.modified_op_dir = Path(ctx.prebuilt_modified_dir)
        info(f"使用预构建修改版快照（聚合模式）: {ctx.modified_op_dir}")
    else:
        ctx.modified_op_dir = _run_step(console, "API 定位（修改版）", _run_locator_for_modified)
    ctx.re_detection_output_path = ctx.modified_op_dir / "re_detection"
    console.print(f"  [dim]修改版导出根目录:[/dim] {ctx.modified_op_dir}")
    console.print()

    info(f"证据目录: {ctx.evidence_dir}")
    info(f"原始算子目录: {ctx.snapshot_op_dir}")
    info(f"修改版算子目录: {ctx.modified_op_dir}")
    console.print(f"  [dim]证据目录:[/dim]    {ctx.evidence_dir}")
    console.print(f"  [dim]原始算子目录:[/dim] {ctx.snapshot_op_dir}")
    console.print(f"  [dim]修改版算子目录:[/dim]  {ctx.modified_op_dir}")

    run_cd = True
    if ctx.interactive:
        console.print()
        run_cd = questionary.confirm("运行变更检测？", default=True).ask()
    if not run_cd:
        return

    # 快照补拷（R1/R2/R3）：把 PR 改动里快照缺失的同类文件补进来，
    # 否则“文件被改但两侧快照都没有它”，diff 恒为空。
    ctx.extra_parts = _augment_snapshot(ctx, console)
    if ctx.extra_parts:
        info(f"快照补拷继承 doc_parts: {ctx.extra_parts}")
    ctx.extra_code_text = _changed_code_added_text(ctx)
    ctx.extra_code_removed = _changed_code_added_text(ctx, side="-")
    if ctx.extra_code_text:
        info(f"全量代码新增行: {len(ctx.extra_code_text.splitlines())} 行")

    def _run():
        from change_detection.doc_part_change_detection.change_evaluator import run_evaluation

        console.print()
        doc_parts, evaluation_reasons, modified_files = run_evaluation(
            evidence_dir=str(ctx.evidence_dir),
            before_dir=str(ctx.snapshot_op_dir / "refactor"),
            after_dir=str(ctx.modified_op_dir / "refactor"),
            doc_content=ctx.doc_content,
            fine_grained=True,
            console=console,
            api_name=ctx.selected_file.stem,
            extra_doc_parts=ctx.extra_parts,
            repo_root=str(ctx.modified_repo_root) if ctx.modified_repo_root else None,
            extra_code_text=ctx.extra_code_text,
            extra_code_removed=ctx.extra_code_removed,
        )
        ctx.evaluation_reasons = evaluation_reasons
        # PR 自身改过资料的章节：一并纳入本次必须处理的章节
        try:
            _now_md = modified_doc_file.read_text(encoding="utf-8") if modified_doc_file else ""
        except OSError:
            _now_md = ""
        for _p in _doc_changed_parts(ctx.doc_content or "", _now_md):
            if _p not in doc_parts:
                doc_parts.append(_p)
                evaluation_reasons.setdefault(
                    _p,
                    {
                        "keep": True,
                        "reason": "PR 同时修改了该章节资料，本次必须重生成",
                        "source": "doc_change",
                    },
                )
                info(f"资料自身变更 → 纳入处理: {_p}")
        if evaluation_reasons:
            skip_path = ctx.snapshot_op_dir / "results" / "evaluation_reasons.json"
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_path.write_text(json.dumps(evaluation_reasons, ensure_ascii=False, indent=2), encoding="utf-8")
            info(f"Evaluation reasons written to: {skip_path}")

        # 兜底：补充扫描 repo 级别的 examples/ 变更
        # 当 api_locator 未将 example 文件提取到 refactor/ 时，
        # run_evaluation 比较 refactor/ 会漏掉 examples/ 下的改动。
        if ctx.original_repo_root and ctx.modified_repo_root and "example" not in doc_parts:
            old_ex = ctx.original_repo_root / "examples"
            new_ex = ctx.modified_repo_root / "examples"
            if old_ex.is_dir() and new_ex.is_dir():
                from src.change_detection.doc_part_change_detection.file_change_detector import FileChangeDetector

                ex_detector = FileChangeDetector(str(old_ex), str(new_ex), fine_grained=True)
                ex_changes, _ = ex_detector.detect_changes()
                if ex_changes:
                    doc_parts.append("example")
                    info(f"兜底扫描发现 {len(ex_changes)} 个 example 文件变更，已注入 example doc_part")
                    console.print(f"  [dim]+ 兜底扫描发现 {len(ex_changes)} 个 example 文件变更[/dim]")

        return doc_parts, modified_files

    doc_parts_raw, ctx.modified_files = _run_step(console, "变更检测", _run)
    # 架构作用域：本次改动落在哪些 npu_arch_*/dav_* 目录下（回填时用于 npu 块校验）
    ctx.arch_scope = sorted(
        {
            m.group(1)
            for f in ctx.modified_files
            if (m := re.search(r"(?:npu_arch_|dav_)(\d+)", str(f).replace("\\", "/")))
        }
    )
    if ctx.arch_scope:
        info(f"架构作用域: {ctx.arch_scope}")
    ctx.doc_parts = _normalize_doc_parts(doc_parts_raw, console)

    if ctx.pre_check_inconsistent_list:
        merged, suppressed = _merge_precheck_inconsistent(
            ctx.doc_parts, ctx.pre_check_inconsistent_list, ctx.evaluation_reasons
        )
        if suppressed:
            info(f"评估判定无变化(keep=false)的 doc_part 不因预校验不一致强制重写: {suppressed}")
            for p in suppressed:
                console.print(f"  [yellow]{format_doc_part(p)}: 评估判定无需变更，忽略预校验不一致[/yellow]")
                console.print(f"    [dim]评估理由: {(ctx.evaluation_reasons or {}).get(p, {}).get('reason', '')}[/dim]")
        info(
            f"合并 doc_parts 与预校验不一致列表: {ctx.doc_parts} + "
            f"{[p for p in ctx.pre_check_inconsistent_list if p not in suppressed]} -> {merged}"
        )
        console.print(f"  [dim]+ 预验证不一致:[/dim] {[format_doc_part(p) for p in ctx.pre_check_inconsistent_list]}")
        console.print(f"  [bold green]最终处理:[/bold green] {[format_doc_part(p) for p in merged]}")
        ctx.doc_parts = merged

    # 评估判 keep=false 的章节一律不进生成队列。
    # 背景：support（产品支持情况）是按证据扫描“整章重建”的，PR 未涉及架构变化时
    # 重写会把原文正确的支持矩阵覆盖成扫描结果（实测 14 篇成片出错）。
    # support 额外收紧：整章重建风险高，目前**一律不自动生成、保留原文**
    # ——实测 27 篇里 PR 真正改动支持判定的为 0，而重建的可裁决样本 18/21 判错。
    # 将来若引入可靠的架构级判定，再放开此处。
    _dropped: list[str] = []
    for p in list(ctx.doc_parts):
        entry = (ctx.evaluation_reasons or {}).get(p) or {}
        if p == "support":
            _dropped.append(p)
        elif entry and not entry.get("keep", True):
            _dropped.append(p)
    if _dropped:
        ctx.doc_parts = [p for p in ctx.doc_parts if p not in _dropped]
        info(f"评估判定无需变更，剔除出生成队列: {_dropped}")
        for p in _dropped:
            console.print(f"  [yellow]{format_doc_part(p)}: 评估判定无需变更，跳过生成[/yellow]")

    highlight(f"{len(ctx.doc_parts)} 个子部分待处理")
    if ctx.doc_parts:
        info(f"待处理子部分: {[format_doc_part(p) for p in ctx.doc_parts]}")


def _step_redetect(console: Console, ctx: PipelineContext):
    """Step 4: Re-detect evidence for changed document parts."""
    console.rule("[bold yellow]证据重检测[/bold yellow]")

    if not ctx.doc_parts:
        info("文档无需变更，跳过证据重检测")
        console.print(f"[dim]文档无需任何变更！[/dim]")
        return

    run_re = True
    if ctx.interactive:
        run_re = questionary.confirm("开始证据重检测？", default=True).ask()
    console.print(f"  [dim]待处理列表:[/dim] {[format_doc_part(p) for p in ctx.doc_parts]}")
    if not run_re:
        return

    info(f"Re-detection output path: {ctx.re_detection_output_path}")
    info(f"Doc parts for re-detection: {ctx.doc_parts}")

    def _run():
        from src.change_detection.anchor_redirect import run_anchor_redirect_pipeline
        from src.change_detection.merge_evidence import merge_redirect_to_anchor

        console.print("  [dim][1/2] 运行 anchor 重定向流水线...[/dim]")
        with redirect_stdout_to_log():
            run_anchor_redirect_pipeline(
                anchor_path=str(ctx.snapshot_op_dir / "anchor"),
                origin_map_base=str(ctx.snapshot_op_dir / "extra" / "code_map"),
                change_map_base=str(ctx.modified_op_dir / "extra" / "code_map"),
                origin_repo_path=str(ctx.snapshot_op_dir / "refactor"),
                change_repo_path=str(ctx.modified_op_dir / "refactor"),
                output_dir=str(ctx.re_detection_output_path),
                verbose=False,
                anchor_type_filter=ctx.doc_parts,
                anchor_file_filter=ctx.modified_files,
            )

        console.print("  [dim][2/2] 合并重定向证据...[/dim]")
        merge_redirect_to_anchor(
            ctx.re_detection_output_path,
            ctx.doc_parts,
            anchor_dir=ctx.modified_op_dir / "anchor",
            verbose=False,
        )

    _run_step(console, "证据重检测", _run)


def _step_pre_validate(console: Console, ctx: PipelineContext) -> bool:
    """Pre-validation: validate existing doc before generation.

    Reuses _step_gen_val under validate_only mode. Returns True if all doc_parts passed.
    """
    console.rule("[bold yellow]预校验[/bold yellow]")
    info("在完整流水线前运行预校验")

    from src.generation_validation.ui.cli import scan_evidence_files

    evidence_files = scan_evidence_files(ctx.evidence_dir)
    ctx.doc_parts = [f.doc_part for f in evidence_files]

    if not ctx.doc_parts:
        info("未找到预校验的 doc_parts，跳过")
        console.print("[dim]未找到预校验的 doc_parts，跳过。[/dim]")
        return True

    info(f"预校验 doc_parts: {ctx.doc_parts}")
    console.print(f"[dim]预校验 doc_parts ({len(ctx.doc_parts)}):[/dim] {[format_doc_part(p) for p in ctx.doc_parts]}")

    original_interactive = ctx.interactive
    ctx.interactive = False
    pre_val_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    _step_gen_val(console, ctx, validate_only=True, dashboard=pre_val_dashboard)
    ctx.interactive = original_interactive

    ctx.pre_val_passed_map = dict(ctx.passed_map)
    ctx.pre_val_detailed_results = dict(ctx.detailed_results)
    ctx.passed_map.clear()
    ctx.detailed_results.clear()

    pre_val_results_path = ctx.snapshot_op_dir / "results" / "pre_val_detailed_results.json"
    try:
        pre_val_results_path.parent.mkdir(parents=True, exist_ok=True)
        pre_val_results_path.write_text(
            json.dumps(
                {k: _serialize_result(v) for k, v in ctx.pre_val_detailed_results.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log_and_print(console, f"Pre-validation results written to: {pre_val_results_path}", style="dim")
    except Exception as e:
        error(f"写入预校验结果失败: {e}")
        console.print(f"[yellow]写入预校验结果失败: {e}[/yellow]")

    info(f"pre_val_passed_map: {ctx.pre_val_passed_map}")

    if not ctx.dashboard:
        lines = "\n".join(
            f"  [dim]{format_doc_part(k)}[/dim]: {'[green]True[/green]' if v else '[red]False[/red]'}"
            for k, v in ctx.pre_val_passed_map.items()
        )
        console.print(f"\n预校验结果:\n{lines}")

        all_passed = all(ctx.pre_val_passed_map.values())
        ctx.pre_check_inconsistent_list = [k for k, v in ctx.pre_val_passed_map.items() if not v]
        info(f"pre_check_inconsistent_list: {ctx.pre_check_inconsistent_list}")

        if all_passed:
            info("预校验全部通过！")
            console.print("[green]预校验全部通过！[/green]")
        else:
            info("预校验存在失败项！")
            console.print("\n[yellow]预校验存在失败项！[/yellow]")
            console.print(
                f"\n[dim]不一致的 doc_parts:[/dim] {[format_doc_part(p) for p in ctx.pre_check_inconsistent_list]}\n"
            )
    else:
        all_passed = all(ctx.pre_val_passed_map.values())
        ctx.pre_check_inconsistent_list = [k for k, v in ctx.pre_val_passed_map.items() if not v]
        info(f"pre_check_inconsistent_list: {ctx.pre_check_inconsistent_list}")
        if ctx.pre_check_inconsistent_list:
            console.print(
                f"\n[yellow]预校验存在失败项:[/yellow] {[format_doc_part(p) for p in ctx.pre_check_inconsistent_list]}"
            )

    return all_passed


_ARCH_SEG_RE = re.compile(r"(?:npu_arch_\d+|dav_\d+)")


def _arch_segments(path: str) -> set[str]:
    """提取路径中的 arch 目录段（npu_arch_3510 / dav_2201 等）。

    用于证据过滤的同架构共址匹配：聚合文件（如 vector_compute_impl.h）被掏空/删除时，
    其 arch 目录下的子实现 anchor（asc_abs_impl.h）basename 虽不同，但属同一架构，
    应视为受影响——否则 arch 支持变化无法触达生成。
    """
    if not path:
        return set()
    return set(_ARCH_SEG_RE.findall(path))


def _anchor_matches_changed_files(
    anchor: dict, changed_basenames: set[str], changed_arch_segs: set[str] = None
) -> bool:
    # 收集该 anchor 涉及的所有文件路径（level1 + 各 evidence）
    file_paths: list[str] = []
    for ev in anchor.get("evidence", []):
        ev_info = ev.get("evidence_information", {})
        fp = ev_info.get("file_path", "")
        if fp:
            file_paths.append(fp)
    anchor_info = anchor.get("anchor_information", {})
    level1 = anchor_info.get("level1", {})
    fp = level1.get("file_path", "")
    if fp:
        file_paths.append(fp)

    # 1. basename 精确匹配
    for p in file_paths:
        if Path(p).name in changed_basenames:
            return True
    # 2. 同 arch 目录段共址匹配（聚合文件变更波及同架构子实现）
    if changed_arch_segs:
        for p in file_paths:
            if _arch_segments(p) & changed_arch_segs:
                return True
    return False


def _step_filter_evidence(console: Console, ctx: PipelineContext):
    """Filter evidence for passed doc_parts to only include changed files."""
    console.rule("[bold yellow]证据过滤[/bold yellow]")

    changed_basenames = {Path(f).name for f in ctx.modified_files}
    changed_arch_segs: set[str] = set()
    for f in ctx.modified_files:
        changed_arch_segs |= _arch_segments(f)
    info(f"变更文件列表: {changed_basenames}")
    if changed_arch_segs:
        info(f"变更涉及 arch 段: {changed_arch_segs}")
    console.print(f"[dim]变更文件列表:[/dim] {changed_basenames}")

    evidences_dir = ctx.re_detection_output_path / "evidences"
    filtered_dir = ctx.re_detection_output_path / "evidences_changed_only"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    skipped_parts = []

    for doc_part in list(ctx.doc_parts):
        if not ctx.pre_val_passed_map.get(doc_part, False):
            evidence_file = resolve_evidence_file(evidences_dir, doc_part)
            count = (
                len(json.loads(evidence_file.path.read_text(encoding="utf-8")).get("anchor_tuple", []))
                if evidence_file
                else 0
            )
            info(f"{doc_part}: 预校验失败，保留完整证据 ({count} anchors)")
            console.print(f"[dim]{format_doc_part(doc_part)}: 预校验失败，保留完整证据 ({count} anchors)[/dim]")
            continue

        evidence_file = resolve_evidence_file(evidences_dir, doc_part)
        if evidence_file is None:
            continue

        data = json.loads(evidence_file.path.read_text(encoding="utf-8"))
        anchor_tuples = data.get("anchor_tuple", [])

        filtered = [a for a in anchor_tuples if _anchor_matches_changed_files(a, changed_basenames, changed_arch_segs)]

        if not filtered:
            # 事实要点来自“关联文档”（如 RegTensor.md 说明的 2VL），本就不在变更文件里；
            # 这类章节不能按“无代码证据”跳过，否则要点永远落不了地。
            _must = ((ctx.evaluation_reasons or {}).get(doc_part) or {}).get("must_cover")
            if _must:
                filtered = list(anchor_tuples)
                info(f"{doc_part}: 无变更文件证据，但存在必写要点 {len(_must)} 条，保留全部 anchor")
                console.print(f"[cyan]{format_doc_part(doc_part)}: 无变更文件证据，但需覆盖事实要点，继续生成[/cyan]")
            else:
                warning(f"{doc_part}: pre_val passed but no changed-file evidence found, skipping generation")
                console.print(f"[yellow]{format_doc_part(doc_part)}: 预校验通过但无变更文件证据，跳过。[/yellow]")
                skipped_parts.append(doc_part)
                continue

        data["anchor_tuple"] = filtered
        out_path = filtered_dir / f"{doc_part}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        info(f"{doc_part}: 过滤 {len(anchor_tuples)} -> {len(filtered)} 个 anchor")
        console.print(f"[dim]{format_doc_part(doc_part)}: 过滤 {len(anchor_tuples)} -> {len(filtered)} 个 anchor[/dim]")

    for dp in skipped_parts:
        ctx.doc_parts.remove(dp)


def _analyze_gen_val_result(doc_part: str, result, ctx: PipelineContext) -> Optional[dict]:
    """Analyze the gen_val result for a doc_part and return a dict for evaluation_reasons.json."""

    def _vget(v, key, default=None):
        if isinstance(v, dict):
            return v.get(key, default)
        return getattr(v, key, default)

    if result.no_change_needed:
        return {
            "success": False,
            "reason": result.conflict_details or "LLM 判断证据不要求修改，无实际变更。",
        }
    if result.evidence_conflict:
        return {
            "success": False,
            "reason": f"证据间存在不可调和的冲突: {result.conflict_details or '详见验证日志'}",
        }

    validations = result.validation
    if validations and all(
        (_vget(v, "edit_distance_ratio") or 1.0) < POST_ANALYZE_NO_CHANGE_THRESHOLD for v in validations
    ):
        return {"success": False, "reason": "生成未产生实际变更，证据可能未要求修改文档内容。"}

    if result.max_retries_hit:
        failed = [v for v in validations if not _vget(v, "passed", True)]
        if failed:
            return {"success": False, "reason": f"验证反复失败，达到最大重试次数后仍有 {len(failed)} 个模块未通过。"}

    return {"success": True, "reason": "生成验证正常通过"}


def _step_post_analyze(console: Console, ctx: PipelineContext):
    """Post-generation analysis: detect no-change/conflict cases, update evaluation_reasons.json.

    Note for future syncs from nnac: nnac uses `ctx.version_dir` for the per-version output root;
    dnecsa's equivalent field is `ctx.snapshot_op_dir`. Do not replace it back when re-syncing.
    """
    console.rule("[bold yellow]生成验证后分析[/bold yellow]")

    reasons_path = ctx.snapshot_op_dir / "results" / "evaluation_reasons.json"
    if not reasons_path.exists():
        info("No evaluation_reasons.json found, skipping post-analyze")
        console.print("[dim]未找到 evaluation_reasons.json，跳过后置分析[/dim]")
        return

    all_reasons = json.loads(reasons_path.read_text(encoding="utf-8"))
    blocked_parts = []

    for doc_part in list(ctx.passed_map.keys()):
        if doc_part not in all_reasons:
            continue
        entry = all_reasons[doc_part]
        if not entry.get("keep", False):
            continue

        result = ctx.detailed_results.get(doc_part)
        if not result or not result.success:
            continue

        analysis = _analyze_gen_val_result(doc_part, result, ctx)
        if analysis:
            entry["gen_val_result"] = analysis
            if not analysis["success"]:
                blocked_parts.append(doc_part)
                ctx.passed_map[doc_part] = False

    reasons_path.write_text(json.dumps(all_reasons, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"Updated evaluation_reasons.json with gen_val_result for {len(all_reasons)} doc_parts")
    console.print(f"[dim]已更新 evaluation_reasons.json（{len(all_reasons)} 个 doc_parts）[/dim]")

    for dp in blocked_parts:
        ctx.backfill_map.pop(dp, None)
        reason = all_reasons[dp]["gen_val_result"]["reason"]
        console.print(f"[yellow]{format_doc_part(dp)}: 阻止回填 — {reason}[/yellow]")
        info(f"Blocked backfill for {dp}: {reason}")


def _step_gen_val(
    console: Console,
    ctx: PipelineContext,
    validate_only: bool = False,
    op_dir_override: Optional[Path] = None,
    dashboard: Optional[ParallelDashboard] = None,
):
    """Step 5: Generate and validate document parts."""
    subsection("5. 生成与验证")

    evidence_info: dict[str, str] = {}

    label = "代码-文档校验" if validate_only else "文档生成"
    console.rule(f"[bold yellow]{label}[/bold yellow]")

    def _process_one(doc_part: str):
        if not validate_only:
            changed_only_dir = ctx.re_detection_output_path / "evidences_changed_only"
            if ctx.pre_val_passed_map.get(doc_part, False) and (changed_only_dir / f"{doc_part}.json").exists():
                evidence_dir_path = changed_only_dir
            else:
                evidence_dir_path = ctx.re_detection_output_path / "evidences"
        else:
            evidence_dir_path = ctx.evidence_dir
        evidence_file = resolve_evidence_file(evidence_dir_path, doc_part)

        if evidence_file is None:
            info(f"未找到 doc_part '{doc_part}' 的证据文件 in {evidence_dir_path}")
            return

        evidence_path_str = evidence_file.relative_path
        evidence_info[doc_part] = evidence_path_str
        info(f"处理 doc_part: {doc_part}, evidence: {evidence_path_str}")

        log_capture = LogCapture()

        if validate_only:
            info(f"代码-文档校验 ({doc_part})")
            result = run_validate_only(
                evidence_file,
                doc_part,
                log_capture,
                original_op_dir=str(ctx.snapshot_op_dir),
                doc_json_dir=ctx.snapshot_op_dir / "doc_json" / f"{ctx.selected_file.stem}.json",
            )
        else:
            op_dir = op_dir_override or ctx.modified_op_dir
            info(f"文档生成 ({doc_part})")
            change_reason = ctx.evaluation_reasons.get(doc_part)
            result = run_full_pipeline(
                evidence_file,
                doc_part,
                log_capture,
                style_dir=ctx.summary_file_dir,
                modified_op_dir=str(op_dir),
                operator_name=ctx.selected_op,
                change_reason=change_reason,
                modified_files=ctx.modified_files,
            )
            generated_text = "\n\n".join(item["generated_text"] for item in result.generated)
            if generated_text:
                try:
                    evidence_text = Path(evidence_file.path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    evidence_text = ""
                from src.md_fill.postprocess import guard_arch_scope, guard_section

                # 汇总本次评估识别出的“重命名对”，供守卫区分合法更新与编造
                _renames: dict = {}
                for _v in (ctx.evaluation_reasons or {}).values():
                    if isinstance(_v, dict):
                        _renames.update(_v.get("renames") or {})
                # 本次改动的代码行（PR 全量增删行）：串味守卫用它判定
                # “符号是否真的与本次变更相关”。注意 PipelineContext 中用
                # extra_code_text/extra_code_removed 承载，不存在 changed_detail。
                _code_text = (ctx.extra_code_text or "") + "\n" + (ctx.extra_code_removed or "")
                generated_text, guard_issues = guard_section(
                    doc_part,
                    generated_text,
                    ctx.doc_content,
                    evidence_text,
                    api_stem=ctx.selected_op,
                    renames=_renames,
                    code_text=_code_text,
                )
                # 确定性收口：模型常忽略“改名”，这里把旧名直接替换为新名
                # （含“兼容/废弃”字样的说明句保持原样）
                if _renames and generated_text:
                    from src.md_fill.postprocess import apply_renames

                    generated_text, _rn_issues = apply_renames(generated_text, _renames)
                    if _rn_issues:
                        guard_issues = list(guard_issues) + _rn_issues
                arch_text, arch_issues = guard_arch_scope(generated_text, ctx.doc_content, ctx.arch_scope)
                if arch_issues:
                    generated_text = arch_text
                    guard_issues = list(guard_issues) + list(arch_issues)
                # 生成后收口（业务评审口径）：清提示词泄漏、术语、单条列表符、
                # 函数原型每行一参数——见 postprocess.apply_review_fixes
                if generated_text:
                    from src.md_fill.postprocess import apply_review_fixes

                    generated_text, _rv_issues = apply_review_fixes(doc_part, generated_text, ctx.doc_content)
                    for _ri in _rv_issues:
                        info(f"[{doc_part}] {_ri}")
                # 功能说明：删掉原文没有的 TPosition 二级条目（位置名属领域约定，
                # 代码证据里通常没有；补错不如不补）
                if doc_part == "summary" and generated_text:
                    from src.md_fill.postprocess import strip_unsupported_tpos_lines

                    generated_text, _tp_issues = strip_unsupported_tpos_lines(
                        generated_text, ctx.doc_content, _code_text
                    )
                    for _ti in _tp_issues:
                        info(f"[{doc_part}] {_ti}")
                # 确定性命中：来源开关（无地址参数）的读取起点
                if doc_part == "para_clarify" and generated_text:
                    from src.md_fill.postprocess import apply_buffer_source_offset_clause

                    generated_text, _off_issues = apply_buffer_source_offset_clause(generated_text)
                    for _oi in _off_issues:
                        info(f"[{doc_part}] {_oi}")
                for issue in guard_issues:
                    warning(f"[{doc_part}] 章节守卫: {issue}")
                if guard_issues:
                    highlight(f"{format_doc_part(doc_part)}: 命中章节守卫，已回退原文（{len(guard_issues)} 项）")
                    console.print(f"  [yellow]{format_doc_part(doc_part)}: 命中章节守卫，已回退原文[/yellow]")
                    for issue in guard_issues:
                        console.print(f"    [dim]- {issue}[/dim]")
            # 必写要点覆盖校验：代码枚举/常量与关联文档的数字型事实是否写进产物
            _must = (ctx.evaluation_reasons.get(doc_part) or {}).get("must_cover") or []
            if _must and generated_text:
                _missing = [p for p in _must if (_must_cover_key(p) or "\x00") not in generated_text]
                if _missing:
                    warning(f"[{doc_part}] 必写要点未覆盖 {len(_missing)}/{len(_must)}: {_missing}")
                    console.print(f"  [yellow]必写要点未覆盖 （{len(_missing)}/{len(_must)}）[/yellow]")
                    for _p in _missing:
                        console.print(f"    [dim]- {_p[:90]}[/dim]")
            # 兜底：生成结果为空（校验未过 / 模型空返回）但存在必写要点时，
            # 用“原文该章节 + 事实条目”强制落地，避免整篇被判“无改动”。
            if not (generated_text or "").strip() and _must:
                from src.md_fill.postprocess import _extract_section as _ex_sec

                _head = _DOC_PART_HEADING.get(doc_part, "")
                _orig = _ex_sec(ctx.doc_content, _head) if _head else ""
                _facts = []
                for _p in _must:
                    _t = _p.split("（来源")[0].strip().lstrip("-").strip()
                    if _t and _t not in _facts:
                        _facts.append(_t)
                _body = (_orig or "").strip()
                _extra = "\n".join(f"- {t}" for t in _facts)
                generated_text = f"## {_head.lstrip('# ').strip()}\n\n{_body}\n\n{_extra}\n"
                warning(f"[{doc_part}] 生成结果为空，按必写要点兜底：{len(_facts)} 条")
                console.print(f"  [yellow]{doc_part}: 生成失败，按必写要点兜底生成[/yellow]")
            ctx.backfill_map[doc_part] = generated_text

        if not result.success:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} 通过: False (error: {result.error})")
        elif result.evidence_conflict:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} 通过: False (evidence conflict)")
        elif result.no_change_needed:
            ctx.passed_map[doc_part] = False
            info(f"doc_part {doc_part} 通过: False (no change needed)")
        else:
            ctx.passed_map[doc_part] = all(getattr(item, "passed", True) for item in result.validation)
            info(f"doc_part {doc_part} 通过: {ctx.passed_map[doc_part]}")
        ctx.detailed_results[doc_part] = result

    def _print_one(doc_part: str):
        if doc_part not in ctx.detailed_results:
            return
        console.rule(f"[bold cyan]{label} ({format_doc_part(doc_part)})[/bold cyan]")
        console.print(f"  [dim]File:[/dim]       {evidence_info.get(doc_part, '?')}")
        console.print()
        _display_results(ctx.detailed_results[doc_part], "validate_only" if validate_only else "full")

    remaining = list(ctx.doc_parts)

    if ctx.interactive:
        while remaining:
            console.print()
            remaining_str = ", ".join(f"[cyan]{format_doc_part(p)}[/cyan]" for p in remaining)
            info(f"Remaining doc_parts ({len(remaining)}): {remaining}")
            console.print(f"[dim]Remaining doc_parts ({len(remaining)}):[/dim] {remaining_str}")

            doc_part = select_doc_part(remaining)
            if not doc_part:
                error("未选择 doc_part")
                return

            _process_one(doc_part)
            _print_one(doc_part)

            if questionary.confirm("结束生成验证？", default=False).ask():
                info("用户请求提前结束生成验证")
                break
            remaining.remove(doc_part)
    else:
        info(f"并行处理 {len(remaining)} 个 doc_parts (max_workers={GEN_VAL_MAX_WORKERS})")

        if dashboard:

            def _run_with_context(dp):
                tokens = set_dashboard_context(dashboard, dp)
                try:
                    _process_one(dp)
                finally:
                    for t in reversed(tokens):
                        t.var.reset(t)

            with dashboard.live_context():
                with ThreadPoolExecutor(max_workers=GEN_VAL_MAX_WORKERS) as executor:
                    futures = {executor.submit(_run_with_context, dp): dp for dp in remaining}
                    for future in as_completed(futures):
                        dp = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            import traceback as _tb

                            error(f"doc_part {dp} 失败: {e}\n{_tb.format_exc()}")
                            dashboard.update(dp, status="✗", result="ERROR")
                            dashboard.append_log(dp, f"[red]Error: {e}[/red]")
        else:
            console.print(f"[dim]并行处理 {len(remaining)} 个 doc_parts (max_workers={GEN_VAL_MAX_WORKERS})[/dim]")
            with ThreadPoolExecutor(max_workers=GEN_VAL_MAX_WORKERS) as executor:
                futures = {executor.submit(_process_one, dp): dp for dp in remaining}
                for future in as_completed(futures):
                    dp = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        import traceback as _tb2

                        error(f"doc_part {dp} 失败: {e}\n{_tb2.format_exc()}")
                        console.print(Text(f"doc_part {format_doc_part(dp)} 失败: {e}", style="bold red"))

            for dp in remaining:
                _print_one(dp)


def _step_fix_val(console: Console, ctx: PipelineContext):
    """Fix failed doc_parts by re-generating with full pipeline."""
    if ctx.skip_fix:
        info("跳过修复步骤 (--skip-fix)")
        console.print("[dim]跳过修复步骤 (--skip-fix)[/dim]")
        return

    failed_doc_parts = [dp for dp, passed in ctx.passed_map.items() if not passed]
    if not failed_doc_parts:
        info("所有 doc_parts 验证通过，无需修复")
        console.print("[dim]所有 doc_parts 验证通过，无需修复。[/dim]")
        return

    console.rule("[bold red]验证后修复[/bold red]")
    info(f"Failed doc_parts: {failed_doc_parts}")
    console.print(f"[dim]需修复的失败 doc_parts:[/dim] {[format_doc_part(p) for p in failed_doc_parts]}")

    ctx.failed_doc_parts = failed_doc_parts
    ctx.doc_parts = failed_doc_parts

    t = time.time()
    _step_summary(console, ctx)
    ctx.elapsed["fix_summary"] = time.time() - t

    t = time.time()
    fix_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    _step_gen_val(console, ctx, op_dir_override=ctx.original_op_dir, dashboard=fix_dashboard)
    ctx.elapsed["fix_gen_val"] = time.time() - t


_MD_LINK_RE = re.compile(r"\]\(#([^)]+)\)")
_MD_ANCHOR_RE = re.compile(r'<a id="([^"]+)"></a>')


def _extract_md_section(text: str, heading: str) -> str:
    """提取 markdown 文档中某 `## 标题` 章节的正文（到下一个 ## 章节为止）。"""
    m = re.search(rf"(?m)^{re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, re.S)
    return m.group(1) if m else ""


def _check_internal_links(original_path, new_path, headings, console: Console):
    """巡检重写章节是否丢失内部锚点链接（锚点仍存在于新文档时告警）。"""
    try:
        orig = Path(original_path).read_text(encoding="utf-8")
        new = Path(new_path).read_text(encoding="utf-8")
    except OSError:
        return
    anchors_in_new = set(_MD_ANCHOR_RE.findall(new))
    for heading in headings:
        lost = set(_MD_LINK_RE.findall(_extract_md_section(orig, heading))) - set(
            _MD_LINK_RE.findall(_extract_md_section(new, heading))
        )
        lost &= anchors_in_new  # 锚点已不存在时丢链接是合理的，不算问题
        if lost:
            warning(f"{heading}: 重写后丢失内部链接（锚点仍存在）: {sorted(lost)}")
            console.print(f"[yellow]{heading}: 检测到内部链接丢失（锚点 {sorted(lost)} 仍存在），请检查[/yellow]")


_MUST_KEY_RE = re.compile(r"[A-Z][A-Z0-9_]{2,}|\d+\s*VL|2\^\d+")
# 数据通路要点：可判定的关键词是「A -> B」写法，不是枚举名
_DP_KEY_RE = re.compile(r"数据通路[^\n]*（\s*([^）\n]+?)\s*）")


def _must_cover_key(point: str) -> str:
    """从必写要点里取一个可判定的关键词（通路写法 / 枚举名 / 2VL / 2^N）。

    数据通路要点里虽然有枚举名（OUT_TO_L1），但那一行只出现在参数说明；
    功能说明里可判定的是 `Global Memory -> L1 Buffer` 这种通路写法。
    """
    m = _DP_KEY_RE.search(point or "")
    if m:
        return m.group(1).strip()
    m = _MUST_KEY_RE.search(point or "")
    return m.group(0).strip() if m else ""


def _decl_header_from_evidence(evidence_dir: Path) -> str:
    """从 anchor/function_proto.json 取接口声明所在头文件（取证失败返回空串）。"""
    f = evidence_dir / "function_proto.json"
    if not f.is_file():
        return ""
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    paths = re.findall(r'"file_path"\s*:\s*"([^"]+)"', text)
    if not paths:
        return ""
    # 优先取公共声明头（include/ 下、非 impl/），避免命中实现文件
    cands = [p for p in paths if p.startswith("include/") and "impl/" not in p]
    return (cands or paths)[0]


def _step_fill_back(console: Console, ctx: PipelineContext):
    """Step 6: Fill generated content back to markdown."""
    subsection("6. 回填到原始 Markdown 文档")

    if not (ctx.selected_file and ctx.backfill_map):
        info("无需回填 (no selected_file or empty backfill_map)")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ctx.snapshot_op_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(results_dir / f"[new]{ctx.selected_file.stem}_{timestamp}.md")
    info(f"回填至: {output_path}")

    def _run():
        from md_fill.main_fill import update_md_with_dict

        update_md_with_dict(ctx.backfill_map, ctx.selected_file, output_path)
        # ① 头文件声明归一：功能说明首段声明一次，其它章节不重复。
        # 放在回填之后执行，可覆盖“守卫整节回退把该句一起丢掉”的情况。
        try:
            from src.md_fill.postprocess import ensure_header_statement

            decl = _decl_header_from_evidence(ctx.evidence_dir)
            if decl:
                out = Path(output_path)
                text = out.read_text(encoding="utf-8")
                fixed, hdr_issues = ensure_header_statement(text, decl)
                if fixed != text:
                    out.write_text(fixed, encoding="utf-8")
                for it in hdr_issues:
                    info(f"头文件声明: {it}")
                if hdr_issues:
                    console.print(f"  [dim]头文件声明: {'；'.join(hdr_issues)}[/dim]")
        except Exception as e:  # 声明归一失败不影响主流程
            warning(f"头文件声明归一失败: {e}")

    _run_step(console, "Markdown 回填", _run)
    # backfill_map 键是 doc_part 代号（如 constraint），章节标题取自 value 首行（如 ## 约束说明）
    headings = [v.splitlines()[0] for v in ctx.backfill_map.values() if v and v.splitlines()]
    _check_internal_links(ctx.selected_file, output_path, headings, console)
    console.print(f"[dim]新 Markdown 文件已生成: {output_path}[/dim]")


def _step_export_results(console: Console, ctx: PipelineContext):
    """Step 7: Export detailed results to JSON."""
    detailed_results_path = ctx.snapshot_op_dir / "results" / "detailed_results.json"
    try:
        detailed_results_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_results_path.write_text(
            json.dumps(
                {k: _serialize_result(v) for k, v in ctx.detailed_results.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log_and_print(console, f"Detailed results written to: {detailed_results_path}", style="dim")
    except Exception as e:
        error(f"写入详细结果失败: {e}")
        console.print(f"[yellow]写入详细结果失败: {e}[/yellow]")


def _serialize_result(r: PipelineResult) -> dict:
    d = asdict(r)
    d["validation"] = [v.model_dump() if hasattr(v, "model_dump") else v for v in r.validation]
    return d


def _run_pipeline(
    console: Console,
    interactive: bool = False,
    original_op_dir: Optional[Path] = None,
    modified_op_dir: Optional[Path] = None,
    validate_only: bool = False,
    skip_fix: bool = False,
    workspace_dir: Optional[Path] = None,
    output_data_path: Optional[Path] = None,
    dashboard: bool = False,
    prebuilt_original_dir: Optional[Path] = None,
    prebuilt_modified_dir: Optional[Path] = None,
    changed_files: Optional[list] = None,
):
    setup_logger(level=logging.INFO, console=False)
    section("流水线启动")
    info(f"模式: {'交互式' if interactive else '非交互式'}, validate_only={validate_only}, skip_fix={skip_fix}")
    console.clear()
    display_header()

    ws = (workspace_dir or Path(__file__).resolve().parent.parent).resolve()
    output_data = (output_data_path or ws / "data" / "outputs" / "snapshot").resolve()
    output_data.mkdir(parents=True, exist_ok=True)

    subsection("0. 工作空间")
    info(f"工作空间: {ws}")
    info(f"数据路径: {output_data}")
    console.print(f"  [dim]工作空间:[/dim]  {ws}")
    console.print(f"  [dim]数据路径:[/dim]  {output_data}")
    info("测试 LLM 模型连通性...")
    from src.llm.client import LLMClient

    try:
        client = LLMClient()
        console.print(f"  [dim]{client.model} 模型连通性测试...[/dim]")
        client.test_connection()
        info(f"LLM 连通性测试通过 [OK] model={client.model}")
        console.print(f"  [dim][LLM OK][/dim]")
    except Exception as e:
        error(f"LLM 连通性测试失败，流水线终止: {e}")
        console.print(f"  [red][LLM ERROR] LLM 模型不可用，全流程终止: {e}[/red]")
        return

    ctx = PipelineContext(
        workspace_dir=ws,
        output_data_path=output_data,
        original_op_dir=original_op_dir,
        modified_op_dir=modified_op_dir,
        validate_only=validate_only,
        skip_fix=skip_fix,
        interactive=interactive,
        dashboard=dashboard,
        prebuilt_original_dir=prebuilt_original_dir,
        prebuilt_modified_dir=prebuilt_modified_dir,
        changed_files=list(changed_files or []),
    )

    start_total = time.time()

    if ctx.interactive:
        if not _step_api_change_detect(console, ctx):
            return
        ctx.elapsed["api_change_detect"] = time.time() - start_total

    if not _step_snapshot(console, ctx):
        return
    ctx.elapsed["snapshot"] = time.time() - start_total
    info(f"快照耗时: {ctx.elapsed['snapshot']:.2f}s")

    _step_config(console, ctx)
    if not ctx.selected_file:
        return

    # 预校验 — 用原始证据验证现有文档与代码一致性
    t = time.time()
    if ENABLE_PRE_VALIDATION:
        _step_pre_validate(console, ctx)
    else:
        console.rule("[bold yellow]预验证（已关闭）[/bold yellow]")
        console.print("[dim]ENABLE_PRE_VALIDATION=False, 跳过预验证[/dim]")
    ctx.elapsed["pre_validate"] = time.time() - t

    # --validate-only: 预校验 → 修复+回填(除非 --skip-fix) → 导出 → 结束
    if ctx.validate_only:
        # 从预校验结果恢复 passed_map，供 fix 步骤判断
        ctx.passed_map = dict(ctx.pre_val_passed_map)

        if ctx.interactive and not all(ctx.pre_val_passed_map.values()):
            if not questionary.confirm("预校验存在失败项，是否执行修复？", default=True).ask():
                ctx.skip_fix = True

        _step_fix_val(console, ctx)
        _step_fill_back(console, ctx)
        _step_export_results(console, ctx)
        _print_stats(ctx, console)
        section("流水线完成")
        console.rule("[bold cyan]Validate-only 流程结束[/bold cyan]")
        return

    # 变更检测 — 对比 before/after 代码，识别受影响的 doc_parts 和变更文件列表
    t = time.time()
    _step_change_detection(console, ctx)
    ctx.elapsed["change_detection"] = time.time() - t

    # 取并集：预校验失败 ∪ 代码变更影响的 doc_parts 为空
    if not ctx.doc_parts:
        info("无需处理任何 doc_parts（预校验全部通过且无代码变更）")
        console.print("[green]无需处理任何 doc_parts（预校验全部通过且无代码变更）[/green]")
        _step_export_results(console, ctx)
        _print_stats(ctx, console)
        section("流水线完成")
        console.rule("[bold cyan]流程结束（无需变更）[/bold cyan]")
        return

    # 生成文档风格摘要
    t = time.time()
    _step_summary(console, ctx)
    ctx.elapsed["summary"] = time.time() - t

    # 证据重检
    t = time.time()
    _step_redetect(console, ctx)
    ctx.elapsed["redetect"] = time.time() - t

    # 证据过滤
    t = time.time()
    _step_filter_evidence(console, ctx)
    ctx.elapsed["filter_evidence"] = time.time() - t

    # 生成与验证
    t = time.time()
    gen_val_dashboard = ParallelDashboard(ctx.doc_parts) if ctx.dashboard else None
    _step_gen_val(console, ctx, dashboard=gen_val_dashboard)
    ctx.elapsed["gen_val"] = time.time() - t

    # Step 8.5: 生成验证后分析 — 检测无变更/证据冲突，更新 evaluation_reasons.json
    _step_post_analyze(console, ctx)

    # 全部待改章节都被判定「无需变更」时，明确打出标志：否则调用方只能看到
    # “没有产物”，会把“模型认为不用改”误判成“生成失败”（6512 实测）。
    if not ctx.backfill_map:
        _hard_fail = [
            p
            for p, r in (ctx.detailed_results or {}).items()
            if not getattr(r, "success", True)
            and not getattr(r, "no_change_needed", False)
            and not getattr(r, "evidence_conflict", False)
        ]
        if not _hard_fail:
            info("（无需变更）全部待改章节经评估判定无需修改")
            console.print("[green]（无需变更）全部待改章节经评估判定无需修改[/green]")

    _step_fill_back(console, ctx)
    _step_export_results(console, ctx)

    _print_stats(ctx, console)
    section("流水线完成")
    console.rule("[bold cyan]流程结束[/bold cyan]")


def _print_stats(ctx: PipelineContext, console: Console):
    """Print pipeline statistics."""
    subsection("统计")
    for name, secs in ctx.elapsed.items():
        info(f"耗时 {name}: {secs:.2f}s")
    info(f"backfill_map: {ctx.backfill_map}")
    info(f"passed_map: {ctx.passed_map}")
    console.rule("[bold cyan]统计[/bold cyan]")
    for name, secs in ctx.elapsed.items():
        console.print(f"[dim]{name}:\t{secs:.2f}s[/dim]")
    console.print(
        f"[dim]backfill_map[/dim] {{{', '.join(f'{k!r}: {v[:50]!r}...' for k, v in ctx.backfill_map.items())}}}"
    )
    console.print(f"[dim]passed_map[/dim] {ctx.passed_map}")


def run_all(
    original_op_dir: str,
    modified_op_dir: str,
    validate_only: bool = False,
    skip_fix: bool = False,
    workspace_dir: Optional[str] = None,
    output_data_path: Optional[str] = None,
    dashboard: bool = False,
    prebuilt_original_dir: Optional[str] = None,
    prebuilt_modified_dir: Optional[str] = None,
    changed_files_path: Optional[str] = None,
):
    console = Console(force_terminal=True)
    console.clear()
    changed_files: list[str] = []
    if changed_files_path and Path(changed_files_path).is_file():
        changed_files = [
            ln.strip()
            for ln in Path(changed_files_path).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    try:
        _run_pipeline(
            console,
            original_op_dir=_resolve_path(original_op_dir),
            modified_op_dir=_resolve_path(modified_op_dir),
            validate_only=validate_only,
            skip_fix=skip_fix,
            workspace_dir=Path(workspace_dir).resolve() if workspace_dir else Path(__file__).resolve().parent.parent,
            output_data_path=_resolve_path(output_data_path) if output_data_path else None,
            interactive=False,
            dashboard=dashboard,
            prebuilt_original_dir=Path(prebuilt_original_dir).resolve() if prebuilt_original_dir else None,
            prebuilt_modified_dir=Path(prebuilt_modified_dir).resolve() if prebuilt_modified_dir else None,
            changed_files=changed_files,
        )
    finally:
        console.clear()


def run_cli():
    console = Console(force_terminal=True)
    console.clear()
    try:
        _run_pipeline(console, interactive=True)
    finally:
        console.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANN doc generation pipeline")
    parser.add_argument("--original_op_dir", help="Operator document path (.md file)")
    parser.add_argument("--modified_op_dir", help="Modified operator document path (.md file)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        dest="validate_only",
        help="Only run pre-validation, skip change detection and generation",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        dest="skip_fix",
        help="Skip the fix step for failed validations (only affects --validate-only)",
    )
    parser.add_argument("--workspace_dir", default=None, help="Workspace root directory (default: project root)")
    parser.add_argument(
        "--output_data_path",
        default=None,
        help="Output data directory (default: <workspace_dir>/data/outputs/snapshot)",
    )
    parser.add_argument(
        "--dashboard", action="store_true", default=False, help="Enable live dashboard for parallel doc_part processing"
    )
    parser.add_argument(
        "--prebuilt_original_dir", default=None, help="Prebuilt original snapshot dir (aggregate mode, skip locator)"
    )
    parser.add_argument(
        "--prebuilt_modified_dir", default=None, help="Prebuilt modified snapshot dir (aggregate mode, skip locator)"
    )
    parser.add_argument(
        "--changed_files", default=None, help="PR 全量改动文件清单（每行一个相对路径）；用于快照补拷 R1/R2/R3"
    )
    args = parser.parse_args()

    if args.original_op_dir and args.modified_op_dir:
        run_all(
            original_op_dir=args.original_op_dir,
            modified_op_dir=args.modified_op_dir,
            validate_only=args.validate_only,
            skip_fix=args.skip_fix,
            workspace_dir=args.workspace_dir,
            output_data_path=args.output_data_path,
            dashboard=args.dashboard,
            prebuilt_original_dir=args.prebuilt_original_dir,
            prebuilt_modified_dir=args.prebuilt_modified_dir,
            changed_files_path=args.changed_files,
        )
    else:
        run_cli()
