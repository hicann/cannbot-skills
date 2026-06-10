---
description: Scan repository documentation quality (correctness, understandability, normative)
---

执行仓库文档质量扫描，分析目标：$ARGUMENTS

如果没有指定仓库，默认扫描 ops-nn。

支持的仓库类型：
- ops-math
- ops-nn
- ops-transformer
- ops-cv

可选参数：
- `--skip-exec` - 跳过执行验证，仅静态分析（默认会执行验证）
- `--no-exec` - 同 `--skip-exec`，跳过执行验证

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-repo-docs_report_{HHMMSS}.md
        └── issues/
            ├── doc_error_issue_{HHMMSS}.md
            └── link_error_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现严重问题

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 资料正确性-严重 | Documentation | ✅ 自动生成 |
| 断链问题（TOP10） | Documentation | ✅ 自动生成 |

---

## 执行流程

### Step 0: 解析参数与环境检查

1. 解析用户参数：
   - 提取仓库类型（第一个参数）
   - 检查是否有 `--skip-exec` 或 `--no-exec` 参数
   
2. 检查执行验证环境：
   - 检查 CANN 环境变量 `ASCEND_HOME_PATH` 或 `/usr/local/Ascend/cann`
   - 检查 NPU 设备 `npu-smi info`
   - 检查编译工具 `cmake`, `gcc`, `python3`
   
3. 确定执行策略：
   - 用户指定 `--skip-exec` → 仅静态分析
   - 环境就绪 → 执行验证（编译、运行命令等）
   - 环境不就绪 → 标记"环境未就绪"，仅静态分析，继续完成所有扫描项

### Step 1: 断链批量检查

调用现有 tool-link-checker 技能的脚本：

```bash
python .opencode/skills/tool-link-checker/scan_links.py {repo}
```

输出断链清单，保存到 `/tmp/{repo}_broken_links_report.txt`

### Step 2: Skill 逐项分析其他扫描项

**严格按照 Skill 定义的所有扫描项逐一检查，不得遗漏。**

对于每个扫描项，Skill 执行：

```
1. 找到文档
   - 根据检查项确定目标文档
   - 读取文档内容

2. 分析检查项
   - 理解检查目标

3. 从文档提取相关信息
   - 命令、目录结构描述、内容等

4. 进行检查
   - 静态检查 + 语义分析

5. 执行验证（如需要且未指定 --skip-exec）
   - 需执行验证的项：编译命令、运行命令、环境部署
   - 检查环境是否满足
   - 环境满足 → 从文档提取命令 → 执行文档写的命令 → 记录结果
   - 环境不满足 → 标记"环境未就绪，跳过执行验证"，继续静态分析

6. 输出结果
   - 状态：passed / failed / partial / skipped
   - 问题描述（如有）
   - 执行验证结果（如有）
```

### Step 3: 生成报告与 Issue 文件

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

合并断链结果 + 各项分析结果，生成报告和 Issue 文件。

```bash
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# 创建报告目录
mkdir -p reports/${date_str}/${repo}/issues

# 报告路径
report_file="reports/${date_str}/${repo}/scan-repo-docs_report_${time_str}.md"

# Issue 文件路径
doc_error_issue="reports/${date_str}/${repo}/issues/doc_error_issue_${time_str}.md"
link_error_issue="reports/${date_str}/${repo}/issues/link_error_issue_${time_str}.md"

# 使用统一模板生成报告
# 模板路径: templates/unified_report_template.md
```

**报告格式要求**：
- 采用统一报告模板（详见 `templates/unified_report_template.md`）
- Issue 格式问题详情（便于直接转换为 GitCode Issue）

---

## 扫描维度（共31项）

### 资料正确性（21项）

1. 环境部署
2. QUICKSTART
3. 源码下载
4. 算子调用-源码编译（需执行验证）
5. 算子调用-执行算子样例（需执行验证）
6. 算子开发-AI Core算子开发全流程(aclnn调用)
7. 算子开发-AI CPU算子开发全流程(aclnn调用)
8. 算子开发-AI Core算子开发全流程(图模式调用)
9. 算子开发-AI CPU算子开发全流程(图模式调用)
10. 算子开发-附录(算子工程迁移)
11. 算子调试调优-调试定位(AI Core/AI CPU算子)
12. 算子调试调优-性能调优>上板
13. 算子调试调优-性能调优>仿真
14. 仿真工具
15. examples下-AI Core算子
16. examples下-AI CPU算子
17. examples下-fast_kernel算子
18. 项目文档-目录说明
19. 项目目录结构
20. 贡献指南
21. 算子列表验证

### 资料易理解性（7项）

1. README大纲和脉络逻辑
2. README Latest News
3. README版本配套
4. README QUICKSTART/环境部署/算子调用/算子开发/调试调优
5. README算子开发-附录
6. README仿真工具
7. README基本概念

### 资料规范性（3项）

1. README安全声明
2. README许可证
3. 所有md链接直达性

---

## 报告输出格式

### 报告结构

```markdown
# {repo} 仓库文档质量扫描报告

## 报告元信息
- 扫描时间、仓库、报告类型、执行环境状态

## 执行摘要
- 扫描统计（31项总数）
- 问题严重程度统计
- 整体评分

## 问题分类与统计
- 资料正确性扫描结果表
- 资料易理解性扫描结果表
- 资料规范性扫描结果表

## 执行验证详情
- 编译命令验证结果
- 运行命令验证结果

## 断链详细清单
- 断链 TOP10 表格

## 问题详情记录
- 每个问题的详细信息（Issue 格式）

## GitCode Issue 文件
- 已生成的 Issue 文件列表

## 修复建议
- 批量修复断链脚本
- 手动修复清单

## 附录
- 完整断链清单
- 扫描检查项清单
```

---

## 完成后确认

- [ ] 已生成 `reports/{date}/{repo}/scan-repo-docs_report_{time}.md`
- [ ] 已生成 Issue 文件（严重问题自动生成）
- [ ] 已输出统计摘要（31项总数，各类别明细）
- [ ] 已标注整体评分
- [ ] 已说明执行验证情况（是否执行、环境状态）
- [ ] 问题详情已按 Issue 格式组织
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

示例用法：
- `/scan-repo-docs ops-cv` - 扫描 ops-cv 文档质量（默认执行验证）
- `/scan-repo-docs ops-math` - 扫描 ops-math 文档质量（默认执行验证）
- `/scan-repo-docs ops-nn --skip-exec` - 仅静态分析，跳过执行验证