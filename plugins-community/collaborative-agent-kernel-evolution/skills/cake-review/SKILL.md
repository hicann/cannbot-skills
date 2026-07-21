---
name: cake-review
description: >
  Comprehensive review of the entire kernel generation process after all stages complete.
  Analyzes problems encountered during generation, compile/environment issues, 
  and skill/agent document quality.
  Produces a REVIEW.md in the kernel output directory.
  Use when a formal review of the kernel generation process is needed, or when manually triggered by the user.
---

# Cake Review Skill

## 概述

在算子生成流程全部完成后，对整个过程进行正式、全面的回顾与总结。产出结构化的 REVIEW.md 文件。

## 工作流程概览

```
┌──────────────────────────────────────────────────────────┐
│  Step 1   收集信息                                        │
│           ├── 读取 PROGRESS.md（含 Detailed Log）         │
│           ├── 扫描 output/{op_name}/ 产物目录             │
│           └── 收集编译日志、精度/性能报告                  │
│                                                          │
│  Step 2   分析问题（五大维度）                             │
│           ├── 算子生成问题（阶段 1-7）                    │
│           ├── 编译与环境问题                              │
│           ├── Skill/Agent 文档质量                       │
│           ├── 性能与精度评估                              │
│           └── 关键经验与模式                              │
│                                                          │
│  Step 3   生成 output/{op_name}/REVIEW.md               │
│                                                          │
│  Step 4   输出确认 + Stage 12 commit                     │
└──────────────────────────────────────────────────────────┘
```

## 输入

- `output/{op_name}/` 目录下的全部生成产物
- `output/{op_name}/PROGRESS.md` 中的详细日志
- 当前会话中的上下文（编译日志、错误信息、修复记录等）

## 详细步骤

### 步骤 1: 收集信息

1. 读取 `output/{op_name}/PROGRESS.md` 的完整内容（包括 Detailed Log 部分）
2. 扫描 `output/{op_name}/` 目录结构，确认各阶段产物是否齐全
3. 检查 git log 获取各阶段 commit 记录：
   ```bash
   cd output/{op_name} && git log --oneline --all
   ```
4. 如有编译日志、评估结果（profiling 数据、精度报告），一并收集

### 步骤 2: 分析问题（五大维度）

按以下五个维度逐一分析，每个维度需给出具体问题描述、影响程度和改进建议：

#### 维度 1: 算子生成问题

回顾阶段 1-7 中遇到的问题：
- op_desc 生成是否顺利？是否需要多次修正？
- 参考实现（reference）是否准确？
- Functional 转换是否丢失语义？
- DSL baseline 是否正确反映算子逻辑？
- DSL lowering 到 AscendC 是否有语法/语义错误？
- 各阶段重试次数和失败原因

#### 维度 2: 编译与环境问题

回顾编译、部署、评估过程中的环境相关问题：
- 编译错误类型和根因（头文件缺失、API 不兼容、语法错误等）
- CANN 版本兼容性问题
- 远程/本地模式切换是否顺畅
- NPU 运行时错误（超时、内存、精度异常等）
- profiling 工具是否正常工作

#### 维度 3: Skill 与 Agent 文档质量

评估当前 skill 和 agent 文档是否存在可改进之处：
- 哪些 skill 的指引不够清晰，导致执行偏差？
- 哪些 skill 缺少关键的错误处理指引？
- agent 主流程文档是否有歧义或遗漏？
- 是否有 skill 之间的衔接问题？
- 推荐具体的文档改进建议（精确到 skill 名称和改进内容）

#### 维度 4: 性能与精度评估

总结评估阶段的结果：
- 精度是否通过？若失败，根因是什么？
- 性能 speedup 达到多少？是否符合预期？
- Advisor 精炼是否执行？效果如何？
- 与同类算子的性能对比（如有参考）
- 精度通过的情况下，检测是否存在擅自修改api_description中的golden/test_cases等作弊行为，有的话明确记录

#### 维度 5: 关键经验与模式

提炼可复用的经验：
- 本次生成中发现的新 pattern 或 trick
- 可推广到其他算子的优化方法
- 值得标准化的代码模板或流程改进

### 步骤 3: 生成 REVIEW.md

在 `output/{op_name}/` 目录下生成 `REVIEW.md`，格式如下：

```markdown
# {op_name} 算子生成回顾

**生成日期**: {date}
**算子类型**: Vector / CV
**最终状态**: ✅ 成功 / ⚠️ 部分成功 / ❌ 失败

---

## 1. 流程概览

| 阶段 | 状态 | 耗时 | 重试次数 | 备注 |
|------|------|------|---------|------|
| 0. 环境检测 | ✅ | - | 0 | 本地/远程 |
| 1. op_desc 生成 | ✅/❌ | - | N | ... |
| ... | ... | ... | ... | ... |

## 2. 问题清单

### 2.1 算子生成问题
- **[P1]** 问题描述...
  - 影响: ...
  - 解决方案: ...
  - 改进建议: ...

### 2.2 编译与环境问题
- **[P2]** 问题描述...
  - 影响: ...
  - 解决方案: ...

### 2.3 Skill/Agent 文档问题
- **[建议]** 问题描述...
  - 涉及 Skill: ...
  - 改进建议: ...

## 3. 评估结果

| 指标 | 结果 |
|------|------|
| 精度 | ✅ PASS / ❌ FAIL (max_diff=...) |
| 性能 Speedup | X.Xx |
| Advisor 精炼 | 执行/跳过 (提升 X%) |

## 4. 关键经验

1. ...
2. ...

## 5. 改进建议摘要

### 对 Skill 文档的建议
1. ...

### 对 Agent 流程的建议
1. ...

### 对工具链的建议
1. ...
```

### 步骤 4: 输出确认

生成完成后输出：

```
📝 算子生成回顾已完成：output/{op_name}/REVIEW.md
   - 发现 {N} 个问题（{P1}个P1, {P2}个P2, {P3}个P3）
   - 提出 {K} 条改进建议
```

然后按 `git-version-management` skill **模块2** 执行 stage 12 commit。

## 注意事项

- 问题分析要基于事实（日志、错误信息），不要臆测
- 对 Skill/Agent 文档的改进建议要具体可执行，避免空泛建议
- REVIEW.md 应简洁但完整，避免冗余描述
- Continue to the next step in agent workflow
