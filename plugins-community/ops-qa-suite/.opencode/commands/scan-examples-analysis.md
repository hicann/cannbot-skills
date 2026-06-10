---
description: Scan operator examples missing for specified repository
---

执行 examples 缺失扫描，分析目标：$ARGUMENTS

如果没有指定仓库，默认分析 ops-math。

支持的仓库类型：
- ops-math
- ops-nn
- ops-transformer
- ops-cv

可选参数：
- `--smart` - 智能分析模式（深入分析算子实现类型）
- 默认为基础分析模式（快速扫描目录结构）

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-examples-analysis_report_{HHMMSS}.md
        └── issues/
            └── examples_missing_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现高优先级 examples 缺失

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 高优先级 examples 缺失 | Requirement | ✅ 自动生成 |
| 中优先级 examples 缺失 | Requirement | ⚠️ 可选生成 |

---

## 判断规则

### 不需要 Examples 的场景

| 场景 | 判断条件 |
|-----|---------|
| 无调用接口 | 无 `op_api/aclnn_*.cpp` 且无 `op_graph/*_proto.*` |
| 仅 aclnn 接口 | README 说"仅 aclnn 接口" AND op_kernel 无 cpp |
| 无 kernel 实现 | op_kernel 目录无 cpp 文件 |

### 需要 Examples 的场景

| 场景 | 判断条件 |
|-----|---------|
| 有 kernel | op_kernel 有 cpp 文件 |
| 有调用接口 | 有 aclnn 或 op_graph proto |

---

## 报告格式要求

采用 Issue 友好格式，便于直接转换为 GitCode Issue 提交：

### 报告结构

```
一、扫描概览
- 统计摘要表格
- 按分类统计表格

二、缺失详情汇总表
- 需要补充 examples 的算子列表

三、问题详情（Issue格式）
- 每个缺失项按 Issue 模板组织

四、不需要 examples 的算子说明
- 无调用接口
- 仅 aclnn 接口

五、判断规则说明
```

### Issue 格式字段

| 字段 | 内容 |
|------|------|
| **标题** | `[Requirement|需求建议]: 补充 {算子名} 的 examples 调用示例` |
| **Background** | 该算子存在 kernel 和调用接口但缺少 examples |
| **Origin** | 算子路径、kernel文件数、调用接口类型 |
| **Benefit/Necessity** | 补充 examples 的必要性 |
| **Design** | 建议 examples 文件路径和内容要求 |

---

## 执行流程

Step 1: 扫描算子目录
Step 2: 检查调用接口
Step 3: 检查 examples 目录
Step 4: 检查 kernel 实现（智能分析）
Step 5: 综合判定并生成报告（Issue格式）

---

## 完成后确认

- [ ] 已输出统计摘要表格
- [ ] 已输出不需要 examples 的算子列表
- [ ] 已输出需要补充 examples 的算子列表
- [ ] 已生成报告 `reports/{date}/{repo}/scan-examples-analysis_report_{time}.md`
- [ ] 已生成 Issue 文件（高优先级缺失自动生成）
- [ ] 问题详情已按 Issue 格式组织
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

示例用法：
- `/scan-examples-analysis ops-math` - 基础扫描 ops-math examples 缺失
- `/scan-examples-analysis ops-nn --smart` - 智能分析 ops-nn examples 缺失
- `/scan-examples-analysis` - 默认扫描 ops-math