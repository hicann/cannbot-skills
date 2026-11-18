# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import shutil
import sys
from pathlib import Path

from rich.console import Console

# 确保子模块可被导入（需要绝对路径，避免从不同目录运行时相对路径解析错误）
_script_dir = Path(__file__).resolve().parent  # change_detection/ 绝对路径
_src_dir = _script_dir.parent  # src/ 绝对路径
_project_root = _src_dir.parent  # 项目根目录 绝对路径

sys.path.insert(0, str(_script_dir))  # change_detection/（子模块直接导入如 anchor_build）
sys.path.insert(0, str(_src_dir))  # src/（src 内其他模块）
sys.path.insert(0, str(_project_root))  # 项目根（from src.xxx 绝对导入）

from anchor_build import anchor_rebuild
from anchor_redirect import run_anchor_redirect_pipeline


def change_deteciton_cli(
    origin_operator_path,
    changed_operator_path,
    data_path,
    output_path,
    filter_list,
    operator_name,
    anchor_type_filter=None,
    file_dir_list=None,
    workspace_dir=None,
    verbose=False,
):
    """
    origin_operator_path, 修改之前的算子目录
    changed_operator_path, 修改之后的算子目录
    data_path, 快照构建的结果。
    output_path, 输出结果的保存路径
    filter_list, [constraint, ...]
    anchor_type_filter, [support, summary, ...] 只处理文件名匹配 {item}.json 的 anchor，为空时处理全部
    operator_name, 算子名称
    file_dir_list, [(abs dir for changed file), ...]
    workspace_dir, 算子工程根目录(ops_base)
    """
    console = Console()
    workspace_base_dir = Path(workspace_dir) if workspace_dir else Path("./")

    changed_operator_path = workspace_base_dir / changed_operator_path
    snapshot_path = workspace_base_dir / data_path

    output_path = workspace_base_dir / output_path
    origin_snapshot_path = workspace_base_dir / data_path
    origin_operator_path = workspace_base_dir / origin_operator_path
    snapshot_path = origin_snapshot_path
    new_snapshot_path = output_path

    op_name_path = snapshot_path / operator_name
    anchor_path = op_name_path / "anchor"
    origin_code_map = op_name_path / "code_map"
    new_code_map = new_snapshot_path / "code_map" / operator_name

    # 执行算子重建
    console.print("  > 算子重建")
    # anchor_rebuild_cli(str(changed_operator_path), str(new_snapshot_path), filter_list)
    old_tree_id_path = str(op_name_path / "tree_id") if file_dir_list else None
    result = anchor_rebuild(
        operator_name,
        str(changed_operator_path),
        str(new_snapshot_path),
        filter_list=filter_list,
        file_dir_list=file_dir_list,
    )
    print(f"===change-detection-result===")
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"===/change-detection-result===")

    # 执行重定向
    console.print("  > 算子重定向")
    run_anchor_redirect_pipeline(
        anchor_path=str(anchor_path),
        origin_map_base=str(origin_code_map),
        change_map_base=str(new_code_map),
        origin_repo_path=str(origin_operator_path),
        change_repo_path=str(changed_operator_path),
        output_dir=str(new_snapshot_path),
        verbose=verbose,
        anchor_type_filter=anchor_type_filter,
    )


if __name__ == "__main__":
    # ── 使用 asc-devkit-master 作为测试数据 ──
    # 测试数据目录：默认取本仓 resources 旁的 asc-devkit-master，可用环境变量覆盖，
    # 不把本机绝对路径写死（合规：避免泄漏个人目录结构）
    REPO_ROOT = os.environ.get("ASCDEVKIT_REPO_ROOT") or str(
        Path(__file__).resolve().parents[2] / "data" / "asc-devkit-master"
    )

    # 创建临时输出目录
    import tempfile

    test_output = os.path.join(tempfile.gettempdir(), "change_detection_test")
    if os.path.exists(test_output):
        shutil.rmtree(test_output)
    os.makedirs(test_output, exist_ok=True)

    test_console = Console()
    test_console.print("=" * 60)
    test_console.print("[bold]变更检测 CLI 测试[/bold]")
    test_console.print("=" * 60)
    test_console.print(f"算子仓库: {REPO_ROOT}")
    test_console.print(f"输出目录: {test_output}")

    # Step 1: 先用 load_snapshot 创建原始快照（跳过 copytree）
    test_console.print("\n[cyan]Step 1: 创建原始快照[/cyan]")
    from anchor_adaptor import _adapt_output, _ensure_anchor_fields, _find_doc_path
    from anchor_build import load_snapshot

    operator_name = "Add"

    # 为 load_snapshot 构造正确的目录结构
    # load_snapshot 会把 anchor 输出到 out_path 下，但 redirect 需要
    # {out_path}/{operator_name}/anchor/ 结构
    origin_snapshot_dir = os.path.join(test_output, "origin_snapshot")

    # 直接调用 load_snapshot（不经过 anchor_rebuild 的 copytree）
    snapshot_out_path = os.path.join(origin_snapshot_dir, operator_name)
    os.makedirs(snapshot_out_path, exist_ok=True)

    load_snapshot(
        operator_path=REPO_ROOT,
        out_path=snapshot_out_path,
        operator_name=operator_name,
    )

    test_console.print(f"原始快照已生成: {snapshot_out_path}")

    # Step 2: 调用 change_deteciton_cli（原始 = 变更后，模拟无变更场景）
    test_console.print("\n[cyan]Step 2: 运行变更检测 CLI[/cyan]")
    change_deteciton_cli(
        origin_operator_path=REPO_ROOT,
        changed_operator_path=REPO_ROOT,
        data_path=os.path.join(origin_snapshot_dir, ""),
        output_path=os.path.join(test_output, "change_detect_output"),
        filter_list=["support", "summary", "function_proto", "constraint"],
        operator_name=operator_name,
        workspace_dir="",
        verbose=True,
    )

    test_console.print("\n[bold green]测试完成[/bold green]")
    test_console.print(f"输出目录: {test_output} (可手动检查后删除)")
