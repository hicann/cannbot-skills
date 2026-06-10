---
name: tool-link-checker
description: Ascend C 算子仓库 Markdown 断链扫描与修复技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 算子仓库的 Markdown 文件内部链接，检测断链并分类统计，提供一键修复脚本。当用户询问断链检查、链接修复、文档可达性问题时使用。
---

# Markdown断链扫描与修复技能

## 技能概述

本技能用于扫描和修复Markdown仓库中的断链问题，支持：
- ✅ 自动扫描所有Markdown文件中的内部链接
- ✅ 检测断链并分类统计
- ✅ 识别URL编码问题、路径错误、文件缺失等问题
- ✅ 提供一键修复脚本
- ✅ 生成详细的扫描报告

## 使用方法

### 1. 扫描指定仓库

```bash
# 扫描ops-transformer仓库
python3 .opencode/skills/link-checker/scan_links.py ops-transformer

# 扫描ops-nn仓库
python3 .opencode/skills/link-checker/scan_links.py ops-nn

# 扫描ops-cv仓库
python3 .opencode/skills/link-checker/scan_links.py ops-cv

# 扫描ops-math仓库
python3 .opencode/skills/link-checker/scan_links.py ops-math
```

### 2. 查看扫描结果

扫描结果会保存在：
- 控制台输出：实时显示断链信息
- 报告文件：`/tmp/{仓库名}_broken_links_report.txt`

### 3. 执行修复

```bash
# 修复ops-math仓库的断链
bash .opencode/skills/link-checker/fix_links.sh ops-math

# 修复ops-transformer仓库的断链
bash .opencode/skills/link-checker/fix_links.sh ops-transformer
```

## 支持的断链类型

| 断链类型 | 检测能力 | 修复能力 |
|---------|---------|---------|
| URL编码问题（%26） | ✅ | ✅ 一键修复 |
| 路径缺少zh层级 | ✅ | ✅ 一键修复 |
| context路径错误 | ✅ | ✅ 一键修复 |
| common目录引用错误 | ✅ | ⚠️ 需确认 |
| 文件不存在 | ✅ | ⚠️ 需手动补充 |
| 目录不存在 | ✅ | ⚠️ 需手动确认 |

## 典型问题示例

### 问题1：URL编码

```markdown
<!-- 错误 -->
[aclnnNanToNum](./docs/aclnnNanToNum%26aclnnInplaceNanToNum.md)

<!-- 正确 -->
[aclnnNanToNum](./docs/aclnnNanToNum&aclnnInplaceNanToNum.md)
```

### 问题2：路径缺少zh层级

```markdown
<!-- 错误 -->
[两段式接口](../../../docs/context/两段式接口.md)

<!-- 正确 -->
[两段式接口](../../../docs/zh/context/两段式接口.md)
```

### 问题3：context路径错误

```markdown
<!-- 错误 -->
[环境部署](../../docs/zh/context/quick_install.md)

<!-- 正确 -->
[环境部署](../../docs/zh/install/quick_install.md)
```

## 输出示例

```
================================================================================
扫描统计
================================================================================
扫描md文件数: 722个
检查链接数: 3393个
发现断链数: 98个
断链率: 2.89%

================================================================================
断链分类统计
================================================================================
文档文件不存在: 84个 (85.7%)
路径错误(缺少zh层级): 12个 (12.2%)
源码文件不存在: 2个 (2.0%)
```

## 注意事项

1. **备份重要**：执行修复前建议提交或备份代码
2. **验证结果**：修复后建议再次扫描验证
3. **手动检查**：部分断链需要手动补充缺失文件
4. **外部链接**：脚本会自动识别并排除外部链接

## 技能文件

- `scan_links.py` - 断链扫描脚本
- `fix_links.sh` - 一键修复脚本
- `SKILL.md` - 本说明文档

## 更新历史

- 2026-04-09: 初始版本，支持ops系列仓库断链扫描和修复