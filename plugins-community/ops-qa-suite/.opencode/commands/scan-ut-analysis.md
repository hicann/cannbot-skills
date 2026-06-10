---
description: Scan UT missing for specified repository
---

执行 UT 缺失扫描，分析目标：$ARGUMENTS

如果没有指定仓库，默认分析 ops-nn。

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-ut-analysis_report_{HHMMSS}.md
        └── issues/
            └── ut_missing_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现高优先级 UT 缺失

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 高优先级 UT 缺失 | Bug-Report | ✅ 自动生成 |
| 中优先级 UT 缺失 | Requirement | ⚠️ 可选生成 |

---

## 报告格式要求

采用统一报告模板（详见 `templates/unified_report_template.md`），便于直接转换为 GitCode Issue 提交：

### 报告结构

```
一、扫描概览
- 统计摘要表格
- 按分类统计表格

二、缺失详情汇总表
- 缺失算子列表（路径 + 缺失类型 + UT路径建议）

三、问题详情（Issue格式）
- 每个缺失项按 Issue 模板组织

四、批量补充建议
- 按优先级排序
- 不需要独立 UT 的算子（详细分析模式）
```

### Issue 格式字段

| 字段 | 内容 |
|------|------|
| **标题** | `[Requirement|需求建议]: 补充 {算子名} 的 {UT类型} UT` |
| **Background** | 该算子存在源文件但缺少 UT |
| **Origin** | 源文件路径、UT文件缺失状态 |
| **Benefit/Necessity** | 补充 UT 的必要性 |
| **Design** | 建议 UT 文件路径和内容要求 |

---

## 执行步骤

1. 调用扫描脚本生成 JSON 数据：
```bash
date_str=$(date +"%Y%m%d")
python .opencode/skills/scan-ut-analysis/scripts/ut_missing_scan.py --repo $ARGUMENTS
```

2. 调用报告脚本生成 Markdown 报告：
```bash
python .opencode/skills/scan-ut-analysis/scripts/gen_report.py --input reports/${date_str}/${repo}/scan-ut-analysis_data.json
```

---

## 完成后确认

- [ ] 已生成 `reports/{date}/{repo}/scan-ut-analysis_report_{time}.md`
- [ ] 已生成 Issue 文件（高优先级缺失自动生成）
- [ ] 已输出统计摘要表格
- [ ] 已输出分类统计表格
- [ ] 问题详情已按 Issue 格式组织
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

示例用法：
- `/scan-ut-analysis ops-nn` - 扫描 ops-nn UT 缺失
- `/scan-ut-analysis ops-transformer` - 扫描 ops-transformer UT 缺失