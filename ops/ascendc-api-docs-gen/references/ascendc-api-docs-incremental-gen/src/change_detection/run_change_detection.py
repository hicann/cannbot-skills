# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 确保子模块可被导入
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from anchor_build import anchor_rebuild
from anchor_redirect import run_anchor_redirect_pipeline


def change_deteciton_cli():
    """
    origin_operator_path, 修改之前的算子目录
    changed_operator_path, 修改之后的算子目录
    data_path, 快照构建的结果。
    output_path, 输出结果的保存路径
    filter_list, [constraint, ...]
    """
    parser = argparse.ArgumentParser(description="Change Detection Pipeline")

    # build 子命令
    # build_parser = parser.add_parser('build', help='构建 anchor')
    parser.add_argument("--workspace-dir", help="工作路径", default="./")
    parser.add_argument(
        "--origin-operator-path",
        help="原始算子目录路径",
        default="D:/work_files/代码仓/cann-document-generation/.cann-helper/backups/snapshot-1779181031137",
    )
    parser.add_argument("--changed-operator-path", help="发生修改后的算子目录路径", default="abs")
    parser.add_argument("--operator-name", help="算子名称", required=True)
    parser.add_argument("--operator-version", help="算子版本。", default="base")
    parser.add_argument(
        "--data-path",
        help="保存快照的位置",
        default="D:/work_files/代码仓/cann-document-generation/.cann-helper/output/snapshot-1779181031137",
    )
    parser.add_argument(
        "--output-path",
        help="更新算子的输出路径",
        default="D:/work_files/代码仓/cann-document-generation/.cann-helper/output/snapshot-1779181031137/change_detect",
    )  # Warning:要指定新的文件夹，不要和data-path相同

    args = parser.parse_args()

    try:
        workspace_base_dir = Path(args.workspace_dir)

        changed_operator_path = workspace_base_dir / args.changed_operator_path
        snapshot_path = workspace_base_dir / args.data_path
        op_version = args.operator_version

        output_path = workspace_base_dir / args.output_path
        origin_snapshot_path = workspace_base_dir / args.data_path
        origin_operator_path = workspace_base_dir / args.origin_operator_path
        snapshot_path = origin_snapshot_path
        new_snapshot_path = output_path
        operator_name = args.operator_name

        op_name_version_path = (
            snapshot_path / operator_name / op_version
        )  # 获取到"out/imageId/算子名称/版本/"这个目录下
        anchor_path = op_name_version_path / "anchor"
        origin_code_map = op_name_version_path / "code_map" / op_version
        new_code_map = new_snapshot_path / "code_map" / operator_name

        filter_list = ["support", "summary", "function_proto", "sub_function_hash", "constraint"]
        # 执行算子重建
        rebuild_result = anchor_rebuild(
            operator_name, str(changed_operator_path), str(new_snapshot_path), filter_list=filter_list
        )

        # 执行重定向
        run_anchor_redirect_pipeline(
            anchor_path=str(anchor_path),
            origin_map_base=str(origin_code_map),
            change_map_base=str(new_code_map),
            origin_repo_path=str(origin_operator_path),
            change_repo_path=str(changed_operator_path),
            output_dir=str(snapshot_path),
            verbose=False,
        )

        print(f"===change-detection-result===")
        print(json.dumps(rebuild_result, ensure_ascii=False, indent=4))

        done_file = snapshot_path / "DONE"
        with open(done_file, "w", encoding="utf-8") as f:
            json.dump({"completedAt": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        failed_file = snapshot_path / "FAILED"
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump({"error": str(e), "failedAt": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        raise


if __name__ == "__main__":
    change_deteciton_cli()
