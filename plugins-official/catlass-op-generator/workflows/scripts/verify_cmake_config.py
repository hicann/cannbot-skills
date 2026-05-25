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
Catlass CMake 配置验证脚本

功能（在通用 Ascend C 校验基础上扩展 catlass 检视项）：
1. 通用：find_package(ASC REQUIRED)、LANGUAGES ASC CXX、--npu-arch、tiling_api 等
2. catlass：必须注入 -I<catlass>/include + -DCATLASS_ARCH=<arch>
3. 禁用 catlass DeviceGemm 适配器（仅 example 用）

用法：
python verify_cmake_config.py operators/{operator_name}/CMakeLists.txt

返回：
- 0: 验证通过
- 1: 验证失败
"""

import sys
import os
import re


def verify_cmake(cmake_file):
    if not os.path.exists(cmake_file):
        return [f"❌ CMakeLists.txt 不存在: {cmake_file}"], []

    with open(cmake_file, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    warnings = []

    # ---- 通用 Ascend C 检查 ----
    if "find_package(ASC REQUIRED)" not in content:
        errors.append("❌ 缺少 find_package(ASC REQUIRED)")

    if not re.search(r"project\s*\([^)]*LANGUAGES[^)]*ASC", content, re.IGNORECASE):
        errors.append("❌ project() 必须包含 LANGUAGES ASC CXX（当前缺少 ASC）")

    if "asc_add_ops_executable" in content:
        errors.append(
            "❌ 禁止使用 asc_add_ops_executable（不存在的函数，请使用 add_executable）"
        )

    if "tiling_api" not in content:
        errors.append("❌ 必须链接 tiling_api 库")

    if "register" not in content:
        warnings.append("⚠️ 建议链接 register 库")

    if "platform" not in content:
        warnings.append("⚠️ 建议链接 platform 库")

    if "--npu-arch" not in content:
        errors.append("❌ 必须设置 --npu-arch 参数（如 --npu-arch=dav-2201）")

    if "add_executable" not in content:
        errors.append("❌ 必须使用 add_executable 定义可执行文件")

    if "target_link_libraries" in content:
        if " m" not in content and "\nm" not in content:
            warnings.append("⚠️ 建议链接数学库 m")
        if " dl" not in content and "\ndl" not in content:
            warnings.append("⚠️ 建议链接动态链接库 dl")

    # ---- catlass 专属检查 ----
    # C3-a：必须注入 -I<catlass>/include
    if not re.search(r"-I[^\s\"']*catlass[/\\]include", content):
        errors.append(
            "❌ catlass C3：必须在 target_compile_options 中注入 "
            "`-I${CMAKE_SOURCE_DIR}/../../catlass/include` 或等价的 catlass include 路径"
        )

    # C3-b：必须设置 -DCATLASS_ARCH
    if "CATLASS_ARCH" not in content:
        errors.append(
            "❌ catlass C3：必须设置 -DCATLASS_ARCH=<架构号>（如 220 / 100），"
            "否则 catlass 模板无法选择对应 ArchTag 实现"
        )

    # C4：CMakeLists.txt 不应链接到 example 用的 DeviceGemm 库（如有该库名出现，提示警告）
    if re.search(r"\bDeviceGemm\b", content):
        warnings.append(
            "⚠️ catlass C4：发现 `DeviceGemm` 字样。op_kernel 严禁使用 catlass `DeviceGemm` "
            "适配器，请确认该字样仅出现在注释或不影响实际链接"
        )

    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_cmake_config.py <CMakeLists.txt路径>")
        print("示例: python verify_cmake_config.py operators/catlass_matmul_add/CMakeLists.txt")
        sys.exit(1)

    cmake_file = sys.argv[1]

    print("=" * 70)
    print("Catlass CMake 配置验证")
    print("=" * 70)
    print(f"📄 CMakeLists.txt: {cmake_file}")
    print()

    errors, warnings = verify_cmake(cmake_file)

    if warnings:
        print("⚠️  警告：")
        for warning in warnings:
            print(f"  {warning}")
        print()

    if errors:
        print("❌ 错误：")
        for error in errors:
            print(f"  {error}")
        print()
        print("=" * 70)
        print("❌ 验证失败")
        print("=" * 70)
        print()
        print("📖 catlass CMake 配置要点：")
        print("   1. find_package(ASC REQUIRED)")
        print("   2. project(... LANGUAGES ASC CXX)")
        print("   3. add_executable + tiling_api / register / platform / m / dl")
        print("   4. target_compile_options 注入：")
        print("      $<$<COMPILE_LANGUAGE:ASC>:-I${CMAKE_SOURCE_DIR}/../../catlass/include>")
        print("      $<$<COMPILE_LANGUAGE:ASC>:-DCATLASS_ARCH=<架构号>>")
        print("   5. target_compile_options 注入 --npu-arch=dav-2201")
        sys.exit(1)
    else:
        print("=" * 70)
        print("✅ Catlass CMake 配置验证通过")
        print("=" * 70)
        print()
        print("✓ 通用 Ascend C 配置完整")
        print("✓ catlass include 路径已注入（C3）")
        print("✓ CATLASS_ARCH 已设置（C3）")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
