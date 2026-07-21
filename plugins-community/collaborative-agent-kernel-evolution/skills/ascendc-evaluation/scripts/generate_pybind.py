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
生成 AscendC 算子的 PyBind 绑定并编译。

用法:
    python generate_pybind.py <op_name>
    python generate_pybind.py <op_name> --work-dir /path/to/work_dir

示例:
    # 为 Add 算子生成 PyBind 绑定 (默认从 output/Add/ 读取)
    python generate_pybind.py Add

    # 指定工作目录
    python generate_pybind.py Conv1d --work-dir output/Conv1d_evo/round_1/parallel_0

说明:
    该脚本会从工作目录下读取 <op_name>.cpp 文件，
    生成 PyBind11 绑定代码并编译成 Python 可导入的 wheel 包。
    编译成功后会自动安装生成的 wheel 包。
    默认工作目录为 output/<op_name>/，可通过 --work-dir 参数覆盖。

依赖:
    - <work_dir>/<op_name>.cpp 必须存在
    - skills/ascendc-evaluation/scripts/template/ 目录必须存在
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import zipfile
import logging
import argparse


def generate_pybind_bindings(work_dir: Path, op_cpp: Path) -> None:
    """
    生成 PyBind 绑定并编译 ascendc 算子。

    Args:
        work_dir (Path): 工作目录路径
        op_cpp (Path): 算子 op.cpp 文件路径

    Raises:
        FileNotFoundError: 模板目录或必要的文件不存在
        subprocess.CalledProcessError: 编译失败时抛出异常，包含错误信息
        Exception: 其他异常情况
    """
    work_dir = Path(work_dir).resolve()
    template_dir = Path(__file__).parent.joinpath("template")
    target_dir = work_dir.joinpath("ascend_op_pybind")

    # 检查模板目录是否存在
    if not template_dir.exists():
        raise FileNotFoundError(f"模板目录不存在: {template_dir}")

    # 如果目标目录不存在，则拷贝模板目录
    if not target_dir.exists():
        logging.info(f"拷贝模板目录到: {target_dir}")
        shutil.copytree(template_dir, target_dir)
    else:
        logging.info(f"目标目录已存在: {target_dir}")

    # 处理 op.cpp 文件
    op_cpp_path = Path(op_cpp).resolve()
    cpp_path = target_dir.joinpath("CppExtension/csrc/op.cpp")

    if not op_cpp_path.exists():
        raise FileNotFoundError(f"op.cpp 源文件不存在: {op_cpp_path}")

    # 如果目标文件存在，先删除
    if cpp_path.exists():
        logging.info(f"删除已存在的 op.cpp: {cpp_path}")
        cpp_path.unlink()

    # 拷贝源文件到目标位置
    logging.info(f"拷贝 op.cpp 文件: {op_cpp_path} -> {cpp_path}")
    cpp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(op_cpp_path, cpp_path)

    # 执行编译
    try:
        logging.info(f"开始编译 PyBind 绑定")
        extension_dir = target_dir.joinpath("CppExtension")

        result = subprocess.run(
            [sys.executable, 'setup.py', 'build', 'bdist_wheel'],
            cwd=str(extension_dir),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logging.info("编译 wheel 包成功")

            # 安装 wheel 包
            dist_dir = extension_dir.joinpath("dist")
            if dist_dir.exists():
                for wheel_file in dist_dir.glob("*.whl"):
                    # Copy .so to work_dir/pybind_lib/ for parallel operator isolation
                    pybind_lib_dir = work_dir / "pybind_lib"
                    pybind_lib_dir.mkdir(parents=True, exist_ok=True)
                    # Extract .so from wheel (it's a zip)
                    with zipfile.ZipFile(str(wheel_file), 'r') as whl:
                        for name in whl.namelist():
                            if name.endswith('.so') or name.endswith('.pyd'):
                                whl.extract(name, str(pybind_lib_dir))
                                # Move from nested dir to pybind_lib root
                                extracted = pybind_lib_dir / name
                                target = pybind_lib_dir / os.path.basename(name)
                                if extracted != target:
                                    shutil.move(str(extracted), str(target))
                                logging.info(f"Extracted {name} -> {target}")
                    logging.info(f"pybind .so isolated in {pybind_lib_dir}")
                    # evaluate.py._inject_pybind_lib() inserts pybind_lib/ at sys.path[0],
                    # so each variant loads its own .so without touching the global site-packages.
                    # Global pip install is intentionally removed to prevent parallel variants
                    # from overwriting each other's installed package.

            logging.info(f"PyBind 绑定编译成功")
        else:
            error_msg = (
                f"编译失败！\n"
                f"Exit Code: {result.returncode}\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            logging.error(error_msg)
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    except subprocess.TimeoutExpired as e:
        raise Exception(f"编译超时（超过 300 秒）") from e
    except Exception as e:
        raise Exception(f"编译过程中发生错误: {str(e)}") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Generate PyBind bindings for AscendC operators")
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Work directory containing op_name.cpp (default: output/<op_name>)")

    args = parser.parse_args()

    try:
        if args.work_dir:
            work_dir = Path(args.work_dir).resolve()
        else:
            work_dir = Path("output").joinpath(args.op_name).resolve()
        op_cpp = work_dir.joinpath(f"{args.op_name}.cpp")
        generate_pybind_bindings(work_dir, op_cpp)
        logging.info("PyBind bindings generated successfully")
    except Exception as e:
        logging.error(e)
