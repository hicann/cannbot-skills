# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Change Detection Pipeline - 融合变更检测与文档生成的非交互式流水线 (Stage 2-6)。

将 run_all_cli.py 的阶段2-6逻辑与 change_deteciton_cli() 合并为单一脚本,
供 VSCode 扩展以子进程方式调用。workspace_dir 为绝对路径, 其余路径均为相对路径。

输出标记:
  - ===change-detection-result=== / ===/change-detection-result=== (由 change_deteciton_cli 透传)
  - ==result== / ==/result== (最终 markdown 列表 JSON)
  - DONE 文件 (成功) / FAILED 文件 (异常)
"""

import argparse
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

# 设置路径: 同时添加 src/ 和 cann-document-generation/ 到搜索路径
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root.parent))

from snapshot_construction.pub_ref import PubRefConfig

# 从环境变量设置 LLM 配置 (与 run_change_detection_cli.py 一致)
PubRefConfig.ALIYUN_KEY = os.environ.get("ALIYUN_KEY")
PubRefConfig.ALIYUN_URL = os.environ.get("ALIYUN_URL")
PubRefConfig.MODEL = os.environ.get("ALIYUN_MODEL", "glm-5")

if not PubRefConfig.ALIYUN_KEY or not PubRefConfig.ALIYUN_URL or not PubRefConfig.MODEL:
    raise Exception("模型相关环境变量未提供！")

# SSL 证书配置: 如果证书不存在则禁用 SSL 验证
# 使用 try/except 避免在无 httpx 依赖的环境下导入失败
try:
    from llm import config as llm_config

    if llm_config.SSL_CERT is not None and not Path(str(llm_config.SSL_CERT)).exists():
        print(f"[run_change_detection_pipeline] SSL cert not found: {llm_config.SSL_CERT}, disabling")
        llm_config.SSL_CERT = None
except ImportError:
    pass


def _auto_select_doc_file(doc_files: list) -> Optional[Path]:
    """自动选择第一个以 aclnn 开头的文档文件 (与 run_all_cli.py 逻辑一致)"""
    default_files = [f for f in doc_files if f.name.startswith("aclnn")]
    if not default_files:
        return None
    selected = default_files[0]
    print(f"  Auto-selected doc file: {selected.name}")
    return selected


def _output_result(results: dict, data_path_resolved: Path):
    """输出结果标记并写入 DONE 文件"""
    print("\n=== Pipeline Complete ===")
    print("==result==")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("==/result==")

    done_file = data_path_resolved / "DONE"
    done_file.write_text(
        json.dumps({"completedAt": datetime.now().isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"DONE file written: {done_file}")


def run_pipeline(
    snapshot_id: str,
    workspace_dir: str,
    original_op_dir: str,
    modified_op_dir: str,
    data_path: str,
    operator_version: str = "base",
    verbose: bool = False,
):
    """执行阶段2-6的变更检测+文档生成流水线"""
    from change_detection.doc_part_change_detection.change_evaluator import run_evaluation
    from generation_validation.style.summarizer import gen_doc_summarize
    from generation_validation.ui.cli import (
        resolve_evidence_file,
        run_full_pipeline,
    )
    from logger import setup_logger
    from md_fill.main_fill import update_md_with_dict
    from src.change_detection.merge_evidence import merge_redirect_to_anchor
    from src.change_detection.run_change_detection_cli import change_deteciton_cli

    setup_logger(level=logging.INFO, console=False)

    # ── 路径解析 ─────────────────────────────────────────────────
    workspace = Path(workspace_dir).resolve()
    data_path_resolved = (workspace / data_path).resolve()
    data_path_resolved.mkdir(parents=True, exist_ok=True)
    original_op_dir_resolved = (workspace / original_op_dir).resolve()
    modified_op_dir_resolved = (workspace / modified_op_dir).resolve()

    selected_op = original_op_dir_resolved.name
    print(f"  Workspace:  {workspace}")
    print(f"  Operator:   {selected_op}")
    print(f"  Data path:  {data_path_resolved}")

    results = {
        "snapshotId": snapshot_id,
        "version": operator_version,
        "markdownFiles": [],
        "generatedAt": None,
    }

    versions_dir = data_path_resolved / selected_op
    operator_storage_dir = data_path_resolved / "operator_storage" / selected_op

    if not operator_storage_dir.exists():
        raise Exception(f"算子快照目录不存在: {operator_storage_dir}")

    # ── 选择算子版本 (与 run_all_cli.py 非交互模式一致) ─────────
    available_versions = sorted(d.name for d in operator_storage_dir.iterdir() if d.is_dir())
    if not available_versions:
        raise Exception(f"未找到可用的算子版本: {operator_storage_dir}")

    if len(available_versions) == 1:
        selected_op_version = available_versions[0]
    elif operator_version in available_versions:
        selected_op_version = operator_version
    else:
        print(f"  Version '{operator_version}' not in {available_versions}, fallback to first")
        selected_op_version = available_versions[0]
    print(f"  Version:    {selected_op_version}")

    # ── 配置各阶段路径 ───────────────────────────────────────────
    version_dir = versions_dir / selected_op_version
    snapshot_op_dir = operator_storage_dir / selected_op_version
    evidence_dir = version_dir / "anchor"
    source_docs = snapshot_op_dir / "docs"
    summary_dir = version_dir / "style"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 修改后算子需进行版本解析处理以匹配编辑前版本内容 (与 run_all_cli.py 非交互模式一致)
    from api_version_split.split import operator_version_split

    modified_output = data_path_resolved / "operator_storage" / f"{modified_op_dir_resolved.name}_modified"
    operator_version_split(str(modified_op_dir_resolved), str(modified_output))
    modified_op_dir_version = modified_output / modified_op_dir_resolved.name / selected_op_version

    # ── 选择文档文件 ─────────────────────────────────────────────
    if not source_docs.exists():
        raise Exception(f"文档目录不存在: {source_docs}")

    doc_files = sorted(source_docs.glob("*.md"))
    if not doc_files:
        raise Exception(f"未找到 .md 文件: {source_docs}")

    selected_file = _auto_select_doc_file(doc_files)
    if selected_file is None:
        raise Exception(f"未找到以 aclnn 开头的文档文件: {source_docs}")
    doc_content = Path(selected_file).read_text(encoding="utf-8")

    # ===== 阶段2: 生成文档摘要 =====
    print("\n=== Stage 2: Generate Doc Summary ===")
    summary_file_dir = summary_dir / f"{selected_file.stem}-summary.json"
    gen_doc_summarize(doc_content, str(summary_file_dir), force=False)
    print(f"Summary saved: {summary_file_dir}")

    # ===== 阶段3: 变更评估 =====
    print("\n=== Stage 3: Change Evaluation ===")
    print(f"  Evidence dir:   {evidence_dir}")
    print(f"  Before dir:     {snapshot_op_dir}")
    print(f"  After dir:      {modified_op_dir_version}")

    console = Console()
    verified_todo_list, _evaluation_reasons, modified_file_dir_list = run_evaluation(
        evidence_dir=str(evidence_dir),
        before_dir=str(snapshot_op_dir),
        after_dir=str(modified_op_dir_version),
        doc_content=doc_content,
        fine_grained=True,
        console=console,
    )
    doc_subpart_todo_list = verified_todo_list
    print(f"  Todo list: {doc_subpart_todo_list}")

    # 如果无需变更, 跳过阶段4-6, 输出空结果
    if not doc_subpart_todo_list:
        print("\nNo document changes detected, skipping Stage 4-6")
        results["generatedAt"] = datetime.now().isoformat()
        _output_result(results, data_path_resolved)
        return results

    # ===== 阶段4: 证据重检测 (change_deteciton_cli + merge) =====
    print("\n=== Stage 4: Evidence Re-detection ===")
    re_detection_output_path = version_dir / "re_detection"
    print(f"  Output path: {re_detection_output_path}")
    print(f"  Todo list:   {doc_subpart_todo_list}")

    # 过滤 “example”
    redirection_doc_parts = [p for p in doc_subpart_todo_list if p != "example"]

    # 传入绝对路径, 不传 workspace_dir (默认 None, 与 run_all_cli.py 一致)
    change_deteciton_cli(
        str(snapshot_op_dir),
        str(modified_op_dir_version),
        str(data_path_resolved),
        selected_op_version,
        str(re_detection_output_path),
        redirection_doc_parts,
        selected_op,
        file_dir_list=modified_file_dir_list,
        verbose=verbose,
    )
    merge_redirect_to_anchor(re_detection_output_path, redirection_doc_parts, verbose=verbose)
    print("Re-detection completed")

    # ===== 阶段5: 生成与验证 =====
    print("\n=== Stage 5: Generate & Validate ===")
    backfill_map = {}
    evidence_dir_path = re_detection_output_path / "evidences"

    def _process_doc_part(doc_part: str):
        print(f"\nProcessing doc_part: {doc_part}")
        evidence_file = resolve_evidence_file(evidence_dir_path, doc_part)

        if evidence_file is None and doc_part != "example":
            print(f"  Warning: No evidence file for {doc_part}, skipping")
            return doc_part, None

        result = run_full_pipeline(
            evidence_file,
            doc_part,
            None,  # log_capture
            style_dir=summary_file_dir,
            modified_op_dir=str(modified_op_dir_version),
            operator_name=selected_op,
        )
        generated_text = "".join(item["generated_text"] for item in result.generated)
        print(f"  Generated {len(generated_text)} chars")
        return doc_part, generated_text

    max_workers = PubRefConfig.DOC_PART_MAX_WORKERS
    print(f"  Processing {len(doc_subpart_todo_list)} doc_parts in parallel (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_doc_part, dp): dp for dp in doc_subpart_todo_list}
        for future in as_completed(futures):
            dp = futures[future]
            try:
                doc_part, generated_text = future.result()
                if generated_text is not None:
                    backfill_map[doc_part] = generated_text
            except Exception as e:
                print(f"  Error processing doc_part {dp}: {e}")

    # ===== 阶段6: 回填到 Markdown =====
    print("\n=== Stage 6: Fill Back to Markdown ===")

    if selected_file and backfill_map:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = data_path_resolved / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        new_op_doc_path = results_dir / f"[new]{selected_file.stem}_{timestamp}.md"
        update_md_with_dict(backfill_map, selected_file, str(new_op_doc_path))
        print(f"New markdown generated: {new_op_doc_path}")

        rel_md_path = selected_file.relative_to(snapshot_op_dir)
        original_path = modified_op_dir_resolved / rel_md_path
        results["markdownFiles"].append(
            {
                "docPart": "full",
                "fileName": f"[new]{selected_file.stem}_{timestamp}.md",
                "originalPath": str(original_path),
                "generatedPath": str(new_op_doc_path),
            }
        )

    results["generatedAt"] = datetime.now().isoformat()
    _output_result(results, data_path_resolved)
    return results


def main():
    parser = argparse.ArgumentParser(description="Change Detection Pipeline - 融合变更检测与文档生成 (Stage 2-6)")
    parser.add_argument("--snapshot-id", required=True, help="快照 ID")
    parser.add_argument("--workspace-dir", required=True, help="工作区根目录 (绝对路径)")
    parser.add_argument("--original-op-dir", required=True, help="修改前算子目录 (相对于 workspace-dir)")
    parser.add_argument("--modified-op-dir", required=True, help="修改后算子目录 (相对于 workspace-dir)")
    parser.add_argument("--data-path", required=True, help="数据输出路径 (相对于 workspace-dir)")
    parser.add_argument("--operator-version", default="base", help="算子版本 (默认: base)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    try:
        run_pipeline(
            snapshot_id=args.snapshot_id,
            workspace_dir=args.workspace_dir,
            original_op_dir=args.original_op_dir,
            modified_op_dir=args.modified_op_dir,
            data_path=args.data_path,
            operator_version=args.operator_version,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nPipeline FAILED: {e}")
        traceback.print_exc()

        # 写入 FAILED 文件
        workspace = Path(args.workspace_dir).resolve()
        data_path_resolved = (workspace / args.data_path).resolve()
        data_path_resolved.mkdir(parents=True, exist_ok=True)
        failed_file = data_path_resolved / "FAILED"
        failed_file.write_text(
            json.dumps(
                {"error": str(e), "failedAt": datetime.now().isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"FAILED file written: {failed_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
