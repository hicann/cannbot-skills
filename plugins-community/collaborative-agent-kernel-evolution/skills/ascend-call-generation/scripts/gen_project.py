#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""
Use msopgen to create Asceng C project.

Usage:
    python3 gen_project.py <op_name> <json_file_path> [--output-dir <dir>]

Examples:
    python3 gen_project.py relu /abs/path/to/relu_project.json
    python3 gen_project.py relu /abs/path/to/relu_project.json --output-dir /abs/path/to/shared/

Notes:
    - <json_file_path> should be an absolute path; relative paths may fail because
      msopgen runs from a different working directory internally.
    - --output-dir controls where the CMake project is created (default: output/).
"""

import shutil
from pathlib import Path
import subprocess
import logging
import argparse


# 定义自定义异常（细分错误类型，便于业务精准处理）
class AscendDeviceError(Exception):
    """昇腾设备信息获取基类异常"""

    pass


class NpuSmiNotFoundError(AscendDeviceError):
    """npu-smi 命令未找到异常"""

    pass


class NpuSmiExecuteError(AscendDeviceError):
    """npu-smi 命令执行失败异常（超时/执行错误/系统错误）"""

    pass


class ChipNameExtractionError(AscendDeviceError):
    """有效芯片名称提取失败异常"""

    pass


def get_ascend_device() -> str:
    """
    获取算子使用的昇腾计算资源标识，格式为 ai_core-{soc version}
    核心改造：通过字符串空格切分+列索引定位，精准提取Chip Name列内容
    步骤：1.校验npu-smi命令 2.执行命令获取输出 3.按空格切分提取Chip Name 4.过滤有效值并拼接格式
    异常：所有错误均抛出对应自定义异常，成功必返回指定格式字符串
    """
    # 步骤1：校验npu-smi命令是否存在，无则抛出异常
    _npu_smi_path = shutil.which("npu-smi")
    if not _npu_smi_path:
        raise NpuSmiNotFoundError(
            "未找到npu-smi命令，请检查昇腾CANN环境是否安装并配置环境变量"
        )

    # 步骤2：执行npu-smi info -m命令，捕获执行异常并抛出
    try:
        cmd_output = subprocess.run(
            [_npu_smi_path, "info", "-m"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        raise NpuSmiExecuteError(
            f"npu-smi info -m 命令执行失败，错误类型：{type(e).__name__}，详情：{str(e)}"
        ) from e

    # 步骤3：按空格切分提取Chip Name列内容（核心改造逻辑）
    valid_chip_name = None
    # 按行遍历命令输出，跳过空行
    for line in cmd_output.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # 按**任意多个空格**切分行为列表（适配命令输出的列对齐空格）
        line_parts = list(
            filter(None, line.split("   "))
        )  # filter(None) 去除切分后的空字符串
        # 命令输出列结构：NPU ID | Chip ID | Chip Logic ID | Chip Name → 索引3为Chip Name列
        if len(line_parts) >= 4:
            chip_name_candidate = line_parts[3]
            # 过滤无效值（Mcu/- 等非计算核心标识），仅保留Ascend开头的有效芯片名
            if chip_name_candidate.startswith("Ascend"):
                valid_chip_name = chip_name_candidate
                break  # 找到第一个有效芯片名后立即退出遍历

    # 步骤4：校验提取结果，无有效值则抛出异常
    if not valid_chip_name:
        raise ChipNameExtractionError(
            f"从npu-smi输出中未提取到有效Chip Name（Ascend开头），命令输出：\n{cmd_output}"
        )

    # 步骤5：按指定格式拼接并返回结果
    return f"ai_core-{valid_chip_name}"


def underscore_to_pascalcase(underscore_str):
    """
    Convert underscore-separated string to PascalCase.

    Preserves existing capitalization within each segment so that already-PascalCase
    inputs (e.g. "LocalResponseNorm_custom") are handled correctly.

    Args:
        underscore_str (str): Input string, e.g. "vector_add" or "LocalResponseNorm_custom"

    Returns:
        str: PascalCase version, e.g. "VectorAdd" or "LocalResponseNormCustom"
    """
    if not underscore_str:  # Handle empty string
        return ""

    parts = underscore_str.split("_")
    # Uppercase only the first character of each part; preserve the rest as-is.
    # str.capitalize() would lowercase everything after the first char, breaking
    # already-PascalCase segments like "LocalResponseNorm" → "Localresponsenorm".
    return "".join(word[0].upper() + word[1:] for word in parts if word)


def patch_build_sh_cleanup(project_dir: Path):
    """
    Append libcust_opapi.so stale-cache cleanup to the generated build.sh.

    Root cause: after modifying OpDef Attrs and rebuilding,
    CANN may still load the old libcust_opapi.so from the global installation path
    instead of the newly built one, causing EXEC_NPU_CMD segfaults.  The fix removes
    any globally-installed copy before the build so the fresh .run package is always
    picked up on next install.
    """
    build_sh = project_dir / "build.sh"
    if not build_sh.exists():
        logging.warning(f"build.sh not found at {build_sh}, skipping stale-cache cleanup patch")
        return

    cleanup_snippet = """
# --- stale libcust_opapi.so cleanup ---
# When OpDef Attrs change, the old globally-installed libcust_opapi.so may be
# loaded by the framework instead of the newly built one, causing segfaults.
# Remove any globally-installed copy so the fresh .run package takes effect.
for _stale_opapi in \\
    /usr/local/Ascend/*/opp/vendors/customize/op_api/lib/libcust_opapi.so \\
    /usr/local/Ascend/ascend-toolkit/*/opp/vendors/customize/op_api/lib/libcust_opapi.so; do
    if [ -f "$_stale_opapi" ]; then
        echo "[CAKE] Removing stale libcust_opapi.so: $_stale_opapi"
        rm -f "$_stale_opapi" 2>/dev/null || echo "[CAKE] Warning: could not remove $_stale_opapi (permission denied?), continuing"
    fi
done
# --- end stale cleanup ---
"""

    content = build_sh.read_text()
    if "stale libcust_opapi.so cleanup" in content:
        logging.info("build.sh already has stale-cache cleanup, skipping")
        return

    build_sh.write_text(content.rstrip() + "\n" + cleanup_snippet)
    logging.info(f"Patched build.sh with stale libcust_opapi.so cleanup: {build_sh}")


def prepare_ascend_project(op_name: str, project_json: Path, output_base_dir: Path = None) -> Path:
    """
    创建 AscendC 算子工程目录并生成项目骨架。

    Args:
        op_name (str): 原始算子名，如 'relu'
        project_json (Path): 创建工程所需的json文件路径，如'output/{op_name}_project.json'
        output_base_dir (Path, optional): Base directory for output. Defaults to Path("output").

    Returns:
        Path: 生成的 AscendC 工程目录路径（如 ./output/ReluCustom）

    Raises:
        Exception: msopgen 生成失败或文件操作异常
        FileNotFoundError: project_json 指定的文件不存在
    """
    # 处理 project_json：当作文件路径读取
    if not project_json.exists():
        raise FileNotFoundError(f"Project JSON file not found: {project_json}")
    # Resolve to absolute path so msopgen (which runs from op_engineer_dir) can find it
    project_json = project_json.resolve()
    ascendc_device = get_ascend_device()
    ascendc_device = ascendc_device.lower().replace(" ", "")

    # 标准化路径
    if output_base_dir is None:
        output_base_dir = Path("output")
    op_engineer_dir = output_base_dir.resolve()   # msopgen 在此目录运行，直接生成 {OpNameCustom}/
    op_custom = op_name + "_custom"
    op_capital = underscore_to_pascalcase(op_custom)
    target_dir = op_engineer_dir.joinpath(op_capital)

    # 确保工程目录存在
    op_engineer_dir.mkdir(parents=True, exist_ok=True)

    # 调用 msopgen 生成工程
    try:
        logging.info(f"Begin creating operator project for '{op_name}'")

        cmd = [
            "msopgen",
            "gen",
            "-i",
            project_json,
            "-c",
            ascendc_device,
            "-lan",
            "cpp",
            "-out",
            op_capital,
        ]

        subprocess.run(cmd, cwd=str(op_engineer_dir), check=True, capture_output=True, text=True)
        logging.info("Create operator project succeeded")
        patch_build_sh_cleanup(target_dir)
        return target_dir

    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Exit Code: {e.returncode}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}"
        )
        raise Exception(f"Failed to create AscendC project via msopgen!\n{error_msg}") from e

    except Exception as e:
        raise Exception(f"Unexpected error during project preparation: {e}") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="AscendC operator project generator")
    parser.add_argument("op_name", type=str, help="Operator name (e.g., 'relu', 'LocalResponseNorm')")
    parser.add_argument("project_json", type=Path, help="Absolute path to project json file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base directory for CMake project output (default: output/). "
                             "The project is created directly at <output-dir>/{OpNameCustom}/ "
                             "(no extra {op_name}/ wrapper directory).")

    args = parser.parse_args()
    try:
        project_path = prepare_ascend_project(args.op_name, args.project_json, args.output_dir)
        logging.info(f"Create Ascend C project at: {project_path}")
    except Exception as e:
        logging.error(e)
