#!/bin/bash
###############################################################################
# Markdown断链一键修复脚本
# 功能：自动修复常见断链问题
# 用法：bash fix_links.sh <仓库名>
# 示例：bash fix_links.sh ops-math
###############################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [ $# -lt 1 ]; then
    echo -e "${RED}错误: 缺少仓库名参数${NC}"
    echo "用法: bash fix_links.sh <仓库名>"
    echo "示例: bash fix_links.sh ops-math"
    echo ""
    echo "支持的仓库: ops-transformer, ops-nn, ops-cv, ops-math"
    exit 1
fi

REPO_NAME=$1
# 默认为当前工作目录下的子目录（支持任意工作目录）
REPO_ROOT="${PWD}/${REPO_NAME}"

# 检查仓库是否存在
if [ ! -d "$REPO_ROOT" ]; then
    echo -e "${RED}错误: 仓库 ${REPO_NAME} 不存在${NC}"
    echo "路径: ${REPO_ROOT}"
    echo ""
    echo "提示: 请确保当前工作目录下存在 ${REPO_NAME}/ 子目录"
    echo "或者在脚本中指定绝对路径: REPO_ROOT=/path/to/${REPO_NAME}"
    exit 1
fi

echo "=================================================="
echo "  Markdown断链一键修复工具"
echo "=================================================="
echo ""
echo "仓库名: ${REPO_NAME}"
echo "仓库路径: ${REPO_ROOT}"
echo ""

# 确认修复
read -p "是否继续修复？这将修改仓库中的Markdown文件 (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo -e "${YELLOW}已取消修复${NC}"
    exit 0
fi

cd "$REPO_ROOT"
echo ""
echo "开始修复..."
echo ""

# 统计修复数量
total_fixes=0

###############################################################################
# 修复1: URL编码问题（%26 -> &）
###############################################################################
echo -e "${GREEN}[1/5] 修复URL编码问题（%26 -> &）...${NC}"
# 只修复本地路径中的 %26，排除 http:// 和 https:// URL 中的合法编码
count=$(grep -r "%26" --include="*.md" . | grep -v "http://" | grep -v "https://" | wc -l || true)
if [ "$count" -gt 0 ]; then
    find . -name "*.md" -type f -exec sed -i -e '/http:\/\//b' -e '/https:\/\//b' -e 's/%26/\&/g' {} \;
    echo "   ✅ 已修复 ${count} 处本地路径中的URL编码问题（已跳过http链接）"
    total_fixes=$((total_fixes + count))
else
    echo "   ℹ️  未发现URL编码问题"
fi

###############################################################################
# 修复2: URL编码问题（%20 -> 空格，可选）
###############################################################################
echo -e "${GREEN}[2/5] 检查URL空格编码问题（%20）...${NC}"
count=$(grep -r "%20" --include="*.md" . | wc -l || true)
if [ "$count" -gt 0 ]; then
    echo -e "   ${YELLOW}发现 ${count} 处%20编码，建议手动确认是否需要修复${NC}"
    echo "   提示: %20表示空格，某些文件名确实包含空格"
else
    echo "   ℹ️  未发现%20编码问题"
fi

###############################################################################
# 修复3: 路径缺少zh层级
###############################################################################
echo -e "${GREEN}[3/5] 修复路径缺少zh层级问题...${NC}"
count=$(grep -r "docs/context/" --include="*.md" . | grep -v "docs/zh/context/" | wc -l || true)
if [ "$count" -gt 0 ]; then
    find . -name "*.md" -type f -exec sed -i 's|docs/context/|docs/zh/context/|g' {} \;
    echo "   ✅ 已修复 ${count} 处缺少zh层级问题"
    total_fixes=$((total_fixes + count))
else
    echo "   ℹ️  未发现zh层级问题"
fi

###############################################################################
# 修复4: context路径错误（应为install）
###############################################################################
echo -e "${GREEN}[4/5] 修复context路径错误（quick_install和build）...${NC}"
count1=$(grep -r "docs/zh/context/quick_install" --include="*.md" . | wc -l || true)
count2=$(grep -r "docs/zh/context/build\.md" --include="*.md" . | wc -l || true)
count=$((count1 + count2))

if [ "$count" -gt 0 ]; then
    find . -name "*.md" -type f -exec sed -i 's|docs/zh/context/quick_install\.md|docs/zh/install/quick_install.md|g' {} \;
    find . -name "*.md" -type f -exec sed -i 's|docs/zh/context/build\.md|docs/zh/install/build.md|g' {} \;
    echo "   ✅ 已修复 ${count} 处context路径错误"
    total_fixes=$((total_fixes + count))
else
    echo "   ℹ️  未发现context路径错误"
fi

###############################################################################
# 修复5: 双斜杠问题
###############################################################################
echo -e "${GREEN}[5/5] 修复双斜杠问题（//zh -> /zh）...${NC}"
# 只匹配本地路径中的双斜杠，避免误伤 https://zh 等外链 URL
count=$(grep -rE "(docs|\.\.)//zh" --include="*.md" . | wc -l || true)
if [ "$count" -gt 0 ]; then
    find . -name "*.md" -type f -exec sed -i -e 's|docs//zh|docs/zh|g' -e 's|\.\.//zh|../zh|g' {} \;
    echo "   ✅ 已修复 ${count} 处双斜杠问题（仅修复本地路径，不影响外链URL）"
    total_fixes=$((total_fixes + count))
else
    echo "   ℹ️  未发现双斜杠问题"
fi

###############################################################################
# 修复总结
###############################################################################
echo ""
echo "=================================================="
echo -e "${GREEN}修复完成！${NC}"
echo "=================================================="
echo ""
echo "修复统计:"
echo "  - 共修复问题: ${total_fixes} 处"
echo ""
echo "后续建议:"
echo "  1. 运行扫描脚本验证修复效果:"
echo "     python3 .opencode/skills/link-checker/scan_links.py ${REPO_NAME}"
echo ""
echo "  2. 手动检查无法自动修复的问题:"
echo "     - 源码文件缺失（需要补充文件）"
echo "     - 目录不存在（需要确认是否应该存在）"
echo ""
echo "  3. 提交修复后的更改:"
echo "     git add ."
echo "     git commit -m 'fix: 修复Markdown断链问题'"
echo ""