# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#!/bin/bash
# init_operator_project.sh - 初始化 Catlass 算子项目目录结构
# 使用：bash init_operator_project.sh <operator_name>

set -e

OPERATOR_NAME=${1:-""}
if [ -z "$OPERATOR_NAME" ]; then
    echo "用法: $0 <operator_name>"
    exit 1
fi

PROJECT_ROOT="operators/${OPERATOR_NAME}"

echo "================================================================"
echo "Catlass 算子项目初始化"
echo "================================================================"
echo ""
echo "算子名称: ${OPERATOR_NAME}"
echo "项目路径: ${PROJECT_ROOT}"
echo ""

if [ -d "$PROJECT_ROOT" ]; then
    echo "  ⚠ 目录已存在，跳过创建"
else
    echo "[1/2] 创建目录结构..."
    mkdir -p "${PROJECT_ROOT}/docs"
    mkdir -p "${PROJECT_ROOT}/build"
    mkdir -p "${PROJECT_ROOT}/test"
    echo "  ✓ docs/ build/ test/"

    echo ""
    echo "[2/2] 创建 README.md..."
    cat > "${PROJECT_ROOT}/README.md" << README_EOF
# ${OPERATOR_NAME} 算子

## 基本信息

- **算子名称**：${OPERATOR_NAME}
- **创建时间**：$(date '+%Y-%m-%d %H:%M:%S')
- **开发状态**：开发中

## 目录结构

\`\`\`
${OPERATOR_NAME}/
├── docs/           # 设计、计划、环境、审查、串讲文档
├── build/          # 编译输出
├── test/           # 测试数据
├── CMakeLists.txt  # 构建脚本（待创建）
├── run.sh          # 运行脚本（待创建）
├── gen_data.py     # 测试数据生成（待创建）
└── golden.py       # Golden 数据计算（待创建）
\`\`\`

## Catlass 编译要求

编译时需注入 catlass 头文件与架构号：
- \`-I\$\{CMAKE_SOURCE_DIR\}/../../catlass/include\`
- \`-DCATLASS_ARCH=<架构号>\`（910b/910_93=2201；950=3510）
README_EOF
    echo "  ✓ README.md"
fi

echo ""
echo "✅ 项目初始化完成"
echo ""
echo "后续步骤："
echo "  1. 运行环境验证："
echo "     bash workflows/scripts/verify_environment.sh ${OPERATOR_NAME}"
echo "  2. 检查 catlass 源码就绪："
echo "     bash workflows/scripts/verify_catlass_ready.sh"
echo "  3. 开始设计阶段（Step 2）"
echo ""
echo "================================================================"
