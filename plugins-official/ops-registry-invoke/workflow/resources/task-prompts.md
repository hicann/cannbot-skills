# Task 调用参数详情

本文档包含主Agent调用的所有Subagent详细参数。

## 任务恢复映射表

| 中断阶段 | Subagent | 恢复说明 |
|---------|----------|---------|
| 1.1 开发准备 | `general` | 读取日志继续 |
| 1.2 需求分析 | `ascendc-ops-architect` | 读取日志继续 |
| 1.3 方案设计 | `ascendc-ops-architect` (scene: design) | 读取日志继续 |
| 1.3R 方案评审 | `ascendc-ops-architect` (scene: design-review) | 读取日志继续（失败时重跑前确认 1.3 已按 DESIGN_REVIEW 修复） |
| 1.4 测试设计 | `ascendc-ops-tester` | 读取日志继续 |
| 2-迭代一-A1-Main | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代一-A2 | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代一-B | `ascendc-ops-tester` | 读取日志继续 |
| 2-迭代二-A1-Main | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代二-A2 | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代二-B | `ascendc-ops-tester` | 读取日志继续 |
| 2-迭代三-A1-Main | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代三-A2 | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代三-B | `ascendc-ops-tester` | 读取日志继续 |
| 2-汇合验证 | `ascendc-ops-developer` | 读取日志继续 |
| 2-测试工程师验收 | `ascendc-ops-tester` | 读取日志继续 |
| 3.1 性能验收 | `ascendc-ops-developer` | 读取日志继续 |
| 4.1 文档与示例 | `general` | 读取日志继续 |
| 4.2 代码检视 | `ascendc-ops-reviewer` | 读取日志继续 |
| 4.3 开发总结 | `general` | 读取日志继续 |

## 开发日志记录原则

所有 Subagent 更新 LOG.md 时遵循：
- **只记结论**：状态变化、关键决策用 1-2 行摘要
- **问题另存**：满足以下任一条件时，**必须**创建 `./issues/issue_{YYYYMMDD}_{关键词}_序号.md`（如 `issue_20260403_opbuild-dt-float16_01.md`）：
  - 排查过程超过 2 轮尝试
  - 涉及底层 API 行为与文档不符
  - 可能复现或需要后续跟进

- **简单问题无需创建 issue**：
  - 命令拼写错误、文档查看遗漏
  - 1 次尝试即解决的环境配置问题
  - 明显的代码笔误
- **日志只放链接**：LOG.md 中只记录问题摘要 + issue 链接
- **不放代码**：代码片段放 commit 或设计文档
- **不放结果**：测试结果放 `./tests/reports/` 目录

## Subagent 日志摘要输出要求

每个 Subagent 任务完成后，必须在输出末尾追加【日志摘要】段落，格式如下：

```markdown
---
## 日志摘要（供主 Agent 写入 LOG.md）
- **状态**: ✅完成 / ❌失败
- **关键结论**: 1 行摘要
- **新增文件**: 相对路径列表
- **问题**:
  - 简单问题（1 行可描述）：直接写解决方案
  - 复杂问题：必须已创建 `./issues/issue_{YYYYMMDD}_{关键词}_序号.md`（如 `issue_20260403_opbuild-dt-float16_01.md`），此处只放链接
```

**注意**：Subagent 不直接修改 LOG.md，由主 Agent 汇总后按模板结构更新。
**强制**：如有复杂问题但 issue 文件不存在，主 Agent 必须拒绝该日志摘要并要求 Subagent 创建。

**主 Agent Git 操作**：基于日志摘要中的「新增文件」列表执行 `git add` + `git commit`，Checkpoint 点额外执行 `git tag`（详见 [SKILL.md](../SKILL.md) 各阶段「Git 操作」段）。

**拒绝恢复流程**：
1. 主 Agent 检查 Subagent 【日志摘要】中的问题链接
2. 如 issue 文件不存在，通知 Subagent 创建
3. Subagent 创建 issue 文件后重新输出日志摘要
4. 主 Agent 确认 issue 文件存在后写入 LOG.md
5. 最多重试 2 次，超过后主 Agent 使用 **Write 工具直接创建** issue 文件（基于日志摘要内容，调用 `general` subagent）

## 报告格式通用规范

所有验收报告必须包含以下字段，供主 Agent 解析判断：

```markdown
**状态**: ✅通过 / ❌失败

**验证摘要**:
| 验证项 | 结果 | 详情 |
|-------|------|------|
| ... | 通过/失败 | ... |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%

**失败用例**（如有）:
- 列出失败的测试用例及原因
```

**⚠️ 重要约束**：
- 如有失败用例，状态必须标记为 `❌失败`，禁止标记为 `✅通过`
- 仅编译通过不等于验证通过，必须实际运行测试

---

## 1.1 开发准备

```
Task 调用参数：
{
  "description": "开发准备",
  "subagent_type": "general",
  "prompt": "
执行开发准备任务。

【输入】
- 用户原始需求：{用户输入}
- 算子名称：{operator_name} (snake_case风格，如add_custom、matmul_v2)
- 环境检查指南：使用 ascendc-env-check skill

【输出】
- 开发日志：operators/{operator_name}/docs/LOG.md
- 问题目录：operators/{operator_name}/issues/

【验收标准】
- 开发日志文件已创建
- 问题目录已创建
- 用户原始需求已完整记录
- 环境检查已执行（使用 ascendc-env-check skill，芯片号、CANN包版本、路径、NPU设备信息等已记录）
  "
}
```

## 1.2 需求分析

```
Task 调用参数：
{
  "description": "需求分析",
  "subagent_type": "ascendc-ops-architect",
  "prompt": "
scene: requirement-analysis

执行需求分析任务。

【输入】
- 用户原始需求：从开发日志 operators/{operator_name}/docs/LOG.md 读取
- 算子名称：{operator_name}

【输出】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- aclnnAPI 接口文档：operators/{operator_name}/docs/aclnn{OperatorName}.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 需求文档包含：算子功能描述、数学公式、输入输出规格、支持数据类型、精度要求、芯片类型、可行性评估
- aclnnAPI 接口文档包含：产品支持情况、功能说明、函数原型、参数说明、返回值、约束说明（调用示例占位）
- 日志摘要已输出
  "
}
```

## 1.3 方案设计

```
Task 调用参数：
{
  "description": "方案设计",
  "subagent_type": "ascendc-ops-architect",
  "prompt": "
scene: design

执行方案设计任务。

【输入】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 算子目录：operators/{operator_name}/

【输出】
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 迭代执行计划：operators/{operator_name}/docs/PLAN.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
1. 详细设计文档包含：Tiling策略规划、Kernel模板选择、数据类型支持方案、API映射方案、数据流设计、内存管理策略
2. 迭代执行计划包含：迭代计划、迭代目标、迭代结果判定规则
3. 日志摘要已输出
  "
}
```

## 1.3R 方案评审

```
Task 调用参数：
{
  "description": "方案设计评审",
  "subagent_type": "ascendc-ops-architect",
  "prompt": "
scene: design-review

执行方案设计评审任务（CP2 前置、不触达用户）。

评审方法论、评审维度、报告格式、强制规则详见 `ascendc-ops-architect` Agent 定义中的场景三。

【输入】
- 需求文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 迭代执行计划：operators/{operator_name}/docs/PLAN.md

【输出】
- 方案评审报告：operators/{operator_name}/docs/DESIGN_REVIEW.md
- 日志摘要：输出到响应末尾（格式见本文档顶部『Subagent 日志摘要输出要求』）

【主 Agent 处理规则】（供调用方参考、非本任务执行项）
- 状态=✅ → 进入 CP2 用户确认
- 状态=❌ → 主 Agent 自动回调 architect (scene: design) 按 DESIGN_REVIEW.md 修订 DESIGN.md，修订后重跑 1.3R；最多重试 2 次
- 禁止把 ❌ 报告直接抛给用户
  "
}
```

## 1.4 测试设计

```
Task 调用参数：
{
  "description": "测试设计",
  "subagent_type": "ascendc-ops-tester",
  "prompt": "
scene: test-design

执行测试设计任务。

【输入】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 算子文档：{operator_name}.md

【输出】
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- 测试用例：operators/{operator_name}/tests/st/testcases/
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求")

【验收标准】
- 测试场景覆盖正常/边界
- 用例分级完成（L0门槛/L1功能）
- 精度标准已定义（从需求文档读取，默认社区标准）
- 日志摘要已输出
  "
}
```

## 新算子开发

```
Task 调用参数：
{
  "description": "迭代 {N} 新算子开发",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
执行 迭代 {N} 新算子开发任务。

【输入】
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 算子目录：operators/{operator_name}/

【输出】
- 算子代码：operators/{operator_name}/（Kernel、Tiling、aclnn等）
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 自定义算子包编译通过
- Kernel二进制成功生成
- 日志摘要已输出
  "
}
```

## A2-UT开发

```
Task 调用参数：
{
  "description": "迭代 {N} UT开发",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
UT开发

执行 迭代 {N} UT开发任务。

【输入】
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 算子目录：operators/{operator_name}/（已编译通过）

【输出】
- UT用例：operators/{operator_name}/tests/ut/
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 迭代一：核心路径UT通过
- 迭代二：Tiling分支UT覆盖达标
- 迭代三：UT全覆盖且无回归
- 日志摘要已输出
  "
}
```

## B-ST测试工程开发

```
Task 调用参数：
{
  "description": "迭代 {N} C++ ST测试工程开发",
  "subagent_type": "ascendc-ops-tester",
  "prompt": "
scene: test-development

执行 迭代 {N} C++ ST测试工程开发任务。

【输入】
- 需求文档（含ACLNN接口定义）：operators/{operator_name}/docs/REQUIREMENTS.md
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- 测试用例：operators/{operator_name}/tests/st/testcases/（L0_test_cases.csv、L1_test_cases.csv）
- 算子目录：operators/{operator_name}/

【输出】
- C++ ST测试工程：operators/{operator_name}/tests/st/
  - test_aclnn_${op_name}.cpp
  - CMakeLists.txt
  - run.sh
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 迭代一：L0标准用例（基础shape + 单dtype）已实现，Mock编译+CPU Golden自测通过
- 迭代二：多shape用例已添加，Mock编译+CPU Golden自测通过
- 迭代三：全dtype + 边界 + 广播用例已添加，Mock编译+CPU Golden自测通过（全量）
- 日志摘要已输出

⚠️ **注意**：本任务只开发 C++ 测试，覆盖 L0+L1 全量用例。
  "
}
```

## 联调验证

```
Task 调用参数：
{
  "description": "迭代 {N} 联调验证",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
执行 迭代 {N} 联调验证任务。

【输入】
- 算子目录：operators/{operator_name}/
- 迭代编号：{N}

【输出】
- 联调验证报告：operators/{operator_name}/tests/reports/iter{N}-integration-report.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【报告格式】（必须包含以下字段）
```
**状态**: ✅通过 / ❌失败

**验证摘要**:
| 验证项 | 结果 | 详情 |
|-------|------|------|
| UT验证 | 通过/失败 | 通过率: X% |
| ST验证 | 通过/失败 | 通过率: X% |
| 前序回归 | 通过/失败/不适用 | - |

**关键指标**:
- UT 总用例数: X, 通过: Y, 失败: Z
- ST 总用例数: X, 通过: Y, 失败: Z
- ST 通过率: X%
```

【验收标准】
1. UT验证和ST验证通过（NPU结果与golden数据比对）
2. 当前迭代用例通过（迭代1/2：增量用例；迭代3：全量）
3. 前序迭代用例无回归（仅迭代2/3需要）
4. 日志摘要已输出

⚠️ **仅编译通过不等于验证通过，必须实际运行测试并确认通过率 = 100%**
  "
}
```

## 测试工程师验收

```
Task 调用参数：
{
  "description": "迭代 {N} 测试工程师验收",
  "subagent_type": "ascendc-ops-tester",
  "prompt": "
scene: test-execution

执行 迭代 {N} 测试工程师验收任务。

【测试方式】使用 **C++ 原生测试**（快速验证）
- 执行命令：cd operators/{operator_name}/tests/st && bash run.sh

【输入】
- 算子目录：operators/{operator_name}/
- 迭代编号：{N}
- 汇合验证结果：operators/{operator_name}/tests/reports/iter{N}-integration-report.md
- 测试设计文档：operators/{operator_name}/docs/TEST.md

【输出】
- 迭代验收报告：operators/{operator_name}/tests/reports/iter{N}-acceptance-report.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【报告格式】（必须包含以下字段）
```
**状态**: ✅通过 / ❌失败

**验收摘要**:
| 验收项 | 结果 | 详情 |
|-------|------|------|
| 用例覆盖 | 通过/失败 | 覆盖率: X% |
| ST通过率 | 通过/失败 | 通过率: X% (Y/Z) |
| 回归测试 | 通过/失败/不适用 | 通过率: X% |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%
```

【验收标准】
- 迭代一：L0用例覆盖完整，ST通过率 = 100%
- 迭代二：多shape用例通过，TilingKey分支覆盖达标，累计通过率 = 100%
- 迭代三：全dtype + 边界 + 广播用例通过，累计通过率 = 100%（无回归）
- 日志摘要已输出
  "
}
```

---

## 3.1 性能达标验收

```
Task 调用参数：
{
  "description": "性能达标验收",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
执行性能达标验收任务。

【输入】
- 算子目录：operators/{operator_name}/
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md

【输出】
- 最终性能验收报告：operators/{operator_name}/docs/performance-report.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【报告格式】（必须包含以下字段）
```
**状态**: ✅通过 / ❌失败

**性能摘要**:
| 指标 | 目标值 | 实际值 | 达标 |
|------|-------|-------|------|
| 吞吐量 | X GFLOPS | Y GFLOPS | 是/否 |
| 延迟 | X ms | Y ms | 是/否 |

**性能分析**:
- 理论算力利用率: X%
- 内存带宽利用率: X%
```

【验收标准】
- 性能符合预期或达到对标水平
- 日志摘要已输出
  "
}
```

## 4.1 文档与示例

```
Task 调用参数：
{
  "description": "文档与示例生成",
  "subagent_type": "general",
  "prompt": "
执行文档与示例生成任务。

【输入】
- 算子目录：operators/{operator_name}/
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- aclnnAPI 接口文档：operators/{operator_name}/docs/aclnn{OperatorName}.md

【输出】
- 算子 README：operators/{operator_name}/README.md
- 调用示例代码：operators/{operator_name}/examples/test_aclnn_{operator_name}.cpp
- 调用示例代码：operators/{operator_name}/examples/test_geir_{operator_name}.cpp
- 构建脚本：operators/{operator_name}/examples/CMakeLists.txt
- 运行脚本：operators/{operator_name}/examples/run.sh
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- README.md 包含：产品支持情况、功能说明、参数说明、约束说明、调用说明（样例链接指向 examples/）
- examples/test_aclnn_{operator_name}.cpp 已生成，能正确编译并运行通过
- examples/test_geir_{operator_name}.cpp 已生成，能正确编译并运行通过
- 日志摘要已输出
  "
}
```

## 4.2 代码检视

```
Task 调用参数：
{
  "description": "代码检视",
  "subagent_type": "ascendc-ops-reviewer",
  "prompt": "
执行代码检视任务。

【输入】
- 算子目录：operators/{operator_name}/
- 设计文档：operators/{operator_name}/docs/

【输出】
- 代码检视报告：operators/{operator_name}/docs/review-report.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 检查代码规范、与设计文档一致性、潜在问题和风险点
- 日志摘要已输出
  "
}
```

## 4.3 开发总结

```
Task 调用参数：
{
  "description": "开发总结",
  "subagent_type": "general",
  "prompt": "
执行开发总结任务。

【输入】
- 算子目录：operators/{operator_name}/

【输出】
- 更新后的 LOG.md
- 更新后的 aclnnAPI 接口文档（补充调用示例代码）
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 开发日志刷新完整
- aclnnAPI 接口文档中的调用示例已补充完整
- 日志摘要已输出
  "
}
```
