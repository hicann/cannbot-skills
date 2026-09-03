# Task 调用参数详情

本文档包含主Agent调用的所有Subagent详细参数。

## 任务恢复映射表

| 中断阶段 | Subagent | 恢复说明 |
|---------|----------|---------|
| 1.1 开发准备 | `general` | 读取日志继续 |
| 1.2 需求分析 | `ascendc-ops-architect` | 读取日志继续 |
| 1.2.5 spec 生成 | `ascendc-ops-architect` (scene: spec-generation) | 读取日志继续 |
| 1.2.5R spec 评审 | `ascendc-ops-spec-reviewer` | 读取日志继续（失败时重跑前确认 1.2.5 已按 operators/{operator_name}/tmp/checks/SPEC_REVIEW.md 修复） |
| 1.3a 设计准备 | `ascendc-ops-designer` (scene: design-prepare) | 读取日志继续；DESIGN_PREP.md 已存在且完整则跳过 |
| 1.3b 分段切片 | 主 Agent（Bash） | 重跑 slice_design_inputs.py（--force 幂等） |
| 1.3c 分段生成 | `ascendc-ops-designer` (scene: generate-section-*) | **独立恢复**：检查 `operators/{op}/.spec-to-design/sections/`，缺失分段重启对应 scene；.spec-to-design 丢失先重跑 1.3b |
| 1.3d 组装校验 | 主 Agent（Bash） | 重跑 assemble + validate |
| 1.3R 方案评审 | `ascendc-ops-design-reviewer` (scene: design-review) | 读取日志继续（失败时重跑前确认已按 operators/{operator_name}/tmp/checks/DESIGN_REVIEW.md 完成分段重生成或 design-fix） |
| 1.4 测试设计 | `ascendc-ops-tester` | 读取日志继续 |
| 1.4R 测试设计评审 | `ascendc-ops-test-design-reviewer` | 读取日志继续（失败时重跑前确认 1.4 已按 operators/{operator_name}/tmp/checks/TEST_REVIEW.md 修复） |
| 2-迭代{N}-A1-Main | `ascendc-ops-developer` | 读取日志继续（N 从 PLAN.md frontmatter 的 iteration_count 确定） |
| 2-迭代{N}-A1-P | `ascendc-ops-developer` | **独立恢复**：读取 `probe/PROBE_SUMMARY.md`，未完成的重启（仅当 iterations[N].wave1.a1_p 非空时） |
| 2-迭代{N}-A1-P-Retry | `ascendc-ops-developer` | **独立恢复**：读取 `probe/PROBE_SUMMARY.md`，未完成的重启 |
| 2-迭代{N}-A2 | `ascendc-ops-developer` | 读取日志继续 |
| 2-迭代{N}-B | `ascendc-ops-tester` | 读取日志继续 |
| 2-汇合验证 | `ascendc-ops-developer` | 读取日志继续 |
| 2-测试工程师验收 | `ascendc-ops-tester` | 读取日志继续 |
| 3.1 精度验收 | `ascendc-ops-tester` | 读取日志继续 |
| 3.2 性能验收 | `ascendc-ops-developer` | 读取日志继续 |
| 4.1 文档与示例 | `general` | 读取日志继续 |
| 4.2 代码检视 | 主 Agent 加载 skill | 调用 /ascendc-code-review，由该 skill 接管检视流程 |
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

**⚠️ A1-P 穿刺任务额外字段**：穿刺任务日志摘要必须包含 `**运行环境**` 字段：

```markdown
- **运行环境**: NPU / Mock  （必须如实标注，主 Agent 将校验此字段）
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

## 固定门禁脚本通用要求

阶段门禁以固定脚本输出为准，不能用 Markdown 摘要或 LOG.md 当前阶段文字替代。

黑盒生成要求：
- 黑盒用例必须按 `ascendc-st-design` 的完整流程和默认目标产出；禁止手写少量用例或只生成 smoke 用例替代。

白盒生成要求：
- CP3 前必须按仓库内 `ascendc-whitebox-design` skill 启动白盒子任务；禁止经 `ascendc-ops-tester` 间接转派白盒分析。
- 白盒检查 high/full case set（覆盖 `ascendc-whitebox-design` 的 high 档全量用例集）。

固定门禁脚本：
- CP2：`python3 workflow/resources/validate-workflow-state.py --stage cp2 --operator-dir operators/{operator_name}`
- CP3：`python3 workflow/resources/validate-workflow-state.py --stage cp3 --operator-dir operators/{operator_name}`
- 主 Agent 只执行固定门禁脚本并根据 `STATUS` 推进或阻塞。
- 命令输出不是 `STATUS: PASSED` 时，主 Agent 必须按校验器列出的差距调度对应 Subagent/任务修复，然后重跑门禁脚本；最多 2 轮，仍失败则创建阻塞 issue 并停止推进。

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

## 1.2.5 spec 生成

```
Task 调用参数：
{
  "description": "L0 数学契约 spec.yaml 生成与 11-stage 校验",
  "subagent_type": "ascendc-ops-architect",
  "prompt": "
scene: spec-generation

执行 L0 数学契约 spec.yaml 生成任务。本任务是 1.3 设计与 1.4 测试的共同前置。

【输入】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 算子名称：{operator_name}

【输出】
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（11-stage 全 PASS）
- 报告（格式见 ascendc-ops-architect 场景二报告模板）
- 日志摘要：输出到响应末尾
  "
}
```

## 1.2.5R spec 评审

```
Task 调用参数：
{
  "description": "spec.yaml 评审（17 条 SPEC-* 条款）",
  "subagent_type": "ascendc-ops-spec-reviewer",
  "prompt": "
执行 spec.yaml 评审任务（CP1.5 前置、不触达用户）。

评审方法论、17 条 SPEC-* 条款定义、报告格式、强制规则详见 `ascendc-ops-spec-reviewer`
Agent 定义。

【输入】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（11-stage 已 PASS）

【输出】
- 评审报告：operators/{operator_name}/tmp/checks/SPEC_REVIEW.md
  - 17 条 SPEC-* 条款逐项 ✓/⚠/❌ + 证据
  - 必看清单（CP1.5 展示用）：公式数学意图 / tolerance 合理性 / boundary 业务覆盖 / composition 拆分 / oracle 选择
  - 状态字段 = ✅通过 / ❌失败
- 日志摘要：输出到响应末尾

  "
}
```

**CP1.5 用户响应处理**（主 Agent 流程，非 task 本身）：

| 用户响应 | 主 Agent 动作 |
|---------|-------------|
| `yes` | 进入 1.3a |
| `modify: <字段路径>=<新值>` | 调 (scene: spec-generation) 按用户指令改 spec → 重跑 11-stage → 重跑 1.2.5R → 重提 CP1.5 |
| `abort` | 退回 1.2 修需求 |

## 1.3 方案设计（主 Agent 编排）

> 1.3 分四步：1.3a 设计准备（designer）→ 1.3b 分段切片（主 Agent）→ 1.3c 并行分段生成（5 个分段 Agent 同响应发起）→ 1.3d 组装+校验（主 Agent）。1.4 在 1.3R 通过后执行。

## 1.3a 设计准备

```
Task 调用参数：
{
  "description": "设计准备",
  "subagent_type": "ascendc-ops-designer",
  "prompt": "
scene: design-prepare

执行设计准备任务（路线决策 + Kernel 模板选型 + API 验证）。

【输入】
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（dtype/shape/invariant/boundary/tolerance 真值源）
- 算子目录：operators/{operator_name}/

【输出】
- 设计准备结论：operators/{operator_name}/docs/DESIGN_PREP.md（必填章节见 Agent 定义场景一）
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）
  "
}
```

## 1.3b 分段切片（主 Agent）

主 Agent 直接用 Bash 执行（非 Task）：

```bash
SPEC_TO_DESIGN_SKILL_DIR="${SPEC_TO_DESIGN_SKILL_DIR:-}"
if [ -z "$SPEC_TO_DESIGN_SKILL_DIR" ]; then
    for p in .opencode/skills/spec-to-design .claude/skills/spec-to-design; do
        [ -d "$p" ] && SPEC_TO_DESIGN_SKILL_DIR="$p" && break
    done
fi
python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/slice_design_inputs.py" \
    operators/{operator_name}/docs/spec.yaml \
    "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
    operators/{operator_name}/.spec-to-design \
    --requirements operators/{operator_name}/docs/REQUIREMENTS.md \
    --plan-template "$SPEC_TO_DESIGN_SKILL_DIR/templates/PLAN.md.templ" \
    --force
```

执行后主 Agent 把输出中的 `Paradigm trace:`（命中的 paradigm 及其 patterns.md / 未命中 / 是否走通用方法论 fallback）摘录到 LOG.md 的 1.3 记录。

## 1.3c 并行分段生成

**⚠️ 强制要求**：以下 5 个 Task 必须在同一次响应中同时发起，禁止逐个串行。

| 分段 | subagent_type | bundle |
|------|---------------|--------|
| 概述+契约 | `ascendc-ops-designer` | `01-overview-contract.md` |
| 架构设计 | `ascendc-ops-designer` | `02-architecture.md` |
| 实现方案 | `ascendc-ops-designer` | `03-implementation.md` |
| 质量+规划 | `ascendc-ops-designer` | `04-quality-plan.md` |
| 迭代计划 | `ascendc-ops-designer` | `05-plan.md` |

每个 Task 调用参数模板（5 个 Task 仅 description / {bundle} 不同，subagent_type 统一为 `ascendc-ops-designer`）：

```
Task 调用参数：
{
  "description": "设计分段 {bundle}",
  "subagent_type": "ascendc-ops-designer",
  "prompt": "
生成 {bundle} 对应分段。

【输入】
- 分段输入包：operators/{operator_name}/.spec-to-design/bundles/{bundle}
- 设计准备结论：operators/{operator_name}/docs/DESIGN_PREP.md

【输出】
- operators/{operator_name}/.spec-to-design/sections/{bundle}
- 完成报告（格式见 Agent 定义）

【约束】
- 内容约束、章节边界、标题契约全部按 Agent 定义执行；只使用 bundle 内切片 + DESIGN_PREP.md，禁止补外部事实
  "
}
```

**失败分段重炉**：1.3d 校验 FAIL 时，仅重发失败分段的 Task，prompt 末尾追加：

```
【修复】上轮校验失败信息如下，仅修复本分段问题后重写输出文件：
{validate 报错摘要}
```

每个分段最多重炉 2 次。

## 1.3d 组装+校验（主 Agent）

先范式设计方案审查修正 03-implementation.md，再组装+校验。

### 范式设计方案审查

| 分段 | subagent_type | 输入 | 输出 |
|------|---------------|------|------|
| 审查 | `ascendc-ops-design-reviewer` (scene: impl-inspect) | patterns.md + 03-implementation.md | MDE_REVIEW.md |
| 修正 | `ascendc-ops-designer` | bundle + DESIGN_PREP.md + MDE_REVIEW.md | operators/{operator_name}/.spec-to-design/sections/03-implementation.md |

**步骤**（串行，前一步不完成禁止进入下一步）：

1. 派审查 Agent，等待返回：`MDE_REVIEW.md` 落盘（Agent 内部已按自然章节切 3 区间串行深审并合并）；成功进入步骤 2，失败则重发。终止条件：审查失败次数 > 2，终止流程，报告用户
2. 派修正 Agent

**审查**:

```
{
  "description": "范式设计方案审查 — impl-inspect",
  "subagent_type": "ascendc-ops-design-reviewer",
  "prompt": "
scene: impl-inspect

审查 03-implementation.md：内部按自然章节切为 3 个互不重叠区间，逐区间串行深审，找出偏离 pattern 的结构性差距，合并输出单份差距清单。

【输入】
- 审查文件：operators/{operator_name}/.spec-to-design/sections/03-implementation.md
- 参考资料：{paradigm_skill}/references/paradigms/{paradigm}/patterns.md

【输出】
- operators/{operator_name}/tmp/checks/MDE_REVIEW.md
  "
}
```

**修正**:

```
{
  "description": "范式设计方案审查 — 修正 implementation",
  "subagent_type": "ascendc-ops-designer",
  "prompt": "
根据审查报告修正 03-implementation.md。

【输入】
- 分段输入包：operators/{operator_name}/.spec-to-design/bundles/03-implementation.md
- 设计准备结论：operators/{operator_name}/docs/DESIGN_PREP.md
- 审查报告：operators/{operator_name}/tmp/checks/MDE_REVIEW.md

【输出】
- operators/{operator_name}/.spec-to-design/sections/03-implementation.md
  "
}
```

### 组装+校验

```bash
python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/assemble_design.py" \
    --spec operators/{operator_name}/docs/spec.yaml \
    --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
    --sections operators/{operator_name}/.spec-to-design/sections \
    --output operators/{operator_name}/docs/DESIGN.md \
    --plan-output operators/{operator_name}/docs/PLAN.md \
    --plan-template "$SPEC_TO_DESIGN_SKILL_DIR/templates/PLAN.md.templ"

python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/validate_design.py" \
    --spec operators/{operator_name}/docs/spec.yaml \
    --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
    --design operators/{operator_name}/docs/DESIGN.md \
    --plan operators/{operator_name}/docs/PLAN.md

python3 "$SPEC_TO_DESIGN_SKILL_DIR/scripts/validate_completeness.py" \
    --spec operators/{operator_name}/docs/spec.yaml \
    --template "$SPEC_TO_DESIGN_SKILL_DIR/templates/DESIGN.md.templ" \
    --design operators/{operator_name}/docs/DESIGN.md \
    --plan operators/{operator_name}/docs/PLAN.md
```

组装前主 Agent 检查跨章节一致性：dtype、shape、TilingKey、API 验证状态、UB 容量、迭代计划必须一致（合并契约见 spec-to-design `references/section-map.md`）。



## 1.3R 方案评审

```
Task 调用参数：
{
  "description": "方案设计评审",
  "subagent_type": "ascendc-ops-design-reviewer",
  "prompt": "
scene: design-review

执行方案设计评审任务（CP2 前置、不触达用户）。

评审方法论、评审维度、报告格式、强制规则详见 `ascendc-ops-design-reviewer` Agent 定义中的场景二（design-review）。

【输入】
- 需求文档：operators/{operator_name}/docs/REQUIREMENTS.md
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（用于 DESIGN-SPEC-1 一致性条款）
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 迭代执行计划：operators/{operator_name}/docs/PLAN.md

【字段优先级】
- 按 `ascendc-ops-design-reviewer` Agent 定义中场景二的「输入与优先级」执行

【输出】
- 方案评审报告：operators/{operator_name}/tmp/checks/DESIGN_REVIEW.md
- 日志摘要：输出到响应末尾（格式见本文档顶部『Subagent 日志摘要输出要求』）

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
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（dtype/shape/boundary/extreme/tolerance 真值源；测试设计的 dtype 矩阵 / 边界场景 / 精度标准必须与之一致）
- 算子文档：{operator_name}.md
- 测试设计评审报告（重新设计时）：operators/{operator_name}/tmp/checks/TEST_REVIEW.md（如果存在）

【字段优先级】
- spec.yaml 是测试设计的真值源
- TEST.md 必须包含「spec.yaml 测试映射」章节，说明测试来源与边界

【输出】
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- 测试用例：operators/{operator_name}/tests/st/testcases/
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求")

【验收标准】
- 测试场景覆盖正常/边界（boundary_conditions / extreme_inputs 覆盖 spec.yaml 中各项）
- 用例分级完成（L0门槛/L1功能/L2异常），并按 `ascendc-st-design` 默认目标产出黑盒用例
- 精度标准已定义（从 spec.yaml 的 numerical_tolerance.per_dtype 读取），并包含 spec.yaml 测试映射
- 日志摘要已输出

  "
}
```

## 1.4R 测试设计评审

```
Task 调用参数：
{
  "description": "测试设计评审",
  "subagent_type": "ascendc-ops-test-design-reviewer",
  "prompt": "
执行测试设计评审任务（CP2 前置、不触达用户）。

评审方法论、评审维度、报告格式、强制规则详见 `ascendc-ops-test-design-reviewer` Agent 定义。

【输入】
- 需求文档：operators/{operator_name}/docs/REQUIREMENTS.md
- L0 数学契约：operators/{operator_name}/docs/spec.yaml
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- 测试用例：operators/{operator_name}/tests/st/testcases/

【输出】
- 测试设计评审报告：operators/{operator_name}/tmp/checks/TEST_REVIEW.md
- 日志摘要：输出到响应末尾

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
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（dtype/shape/invariant/boundary 真值源）
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md（其中 "Design Contract" 章节为强制约束）
- 算子目录：operators/{operator_name}/

【输出】
- 算子代码：operators/{operator_name}/（Kernel、Tiling、aclnn等）
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 自定义算子包编译通过
- Kernel二进制成功生成
- 实现符合 Design Contract 标记的全部章节约束
- 日志摘要已输出
  "
}
```

## 模板穿刺

> ⚠️ **强制并行** - 所有穿刺Task + 主线Task 必须在同一次响应中同时发起

**📌 参数来源**：从 `PLAN.md` frontmatter 的 `iterations[{i}].wave1.a1_p` 列表中提取

```
Task 调用参数（⚠️ 单次响应必须同时发起所有Task）：

{
  "description": "模板穿刺 {task_name}",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
执行模板穿刺任务（迭代 {N}）。

【任务信息】
- Task Name: {task_name}
- TilingKey: {tiling_key}
- Dtype: {dtype}
- Memory Strategy: {memory_strategy}
- 算子名称: {operator_name}
- 迭代编号: {N}

【输入】
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（dtype/shape/invariant/boundary 真值源）
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md（其中 "Design Contract" 章节为强制约束）
- 需求分析文档：operators/{operator_name}/docs/REQUIREMENTS.md
- 迭代执行计划：operators/{operator_name}/docs/PLAN.md
- ⚪ 前序迭代主线代码：operators/{operator_name}/op_kernel/（仅当 N > 1 时存在，已编译通过）

【输出】
- 输出目录：operators/{operator_name}/probe/{task_name}/
- 验证结果：operators/{operator_name}/probe/{task_name}/RESULT.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
1. 编译通过
2. NPU 运行结果比对通过（精度要求从需求分析文档获取）
3. RESULT.md 已生成，包含状态和验证摘要
4. 实现符合 Design Contract 标记的全部章节约束
5. 日志摘要已输出

⚠️ **仅完成编译不算通过，必须在 NPU 上实际运行并验证**
  "
}
```

## 模板穿刺-失败重试

> 与 A2 UT 开发**强制并行**启动

**📌 基础模板**：复用「模板穿刺」，仅替换/新增以下内容

**替换项**：
- `description` → `"失败穿刺重试 {task_name}"`

**【任务信息】新增字段**：
- 原失败原因：从 probe/{task_name}/RESULT.md 获取
- 重试轮次: 第 {N} 迭代第二波

**【输入】新增**：
- 原穿刺结果：operators/{operator_name}/probe/{task_name}/RESULT.md
- 当前主线代码：operators/{operator_name}/op_kernel/（当前迭代 A1-Main 产物）
- 穿刺汇总：operators/{operator_name}/probe/PROBE_SUMMARY.md（筛选条件：状态=失败 AND 重试次数<2）

**【重试策略】新增段落**：
1. 分析原 RESULT.md 中的失败原因
2. 基于更新后的主线代码重新实现
3. 针对性修复已知的失败问题

**【验收标准】修改第 2 条**（其余不变）：
2. NPU 运行已完成（成功则比对通过，失败则 RESULT.md 记录失败原因和已尝试的修复措施）

**【输出】新增**：
- 更新 RESULT.md：operators/{operator_name}/probe/{task_name}/RESULT.md（追加重试记录）
- 更新 PROBE_SUMMARY.md：
  - 成功：状态改为 ✅ 通过，重试次数+1
  - 失败：状态保持 ❌ 失败，重试次数+1
- PROBE_SUMMARY.md 格式：
  ```markdown
  | 任务 | 状态 | 重试次数 |
  |------|------|----------|
  | {task_name} | ✅/❌ | {0|1|2} |
  ```

**⚠️ 强制要求**：必须与 A2 在同一次响应中同时发起，禁止串行执行

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
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（dtype/shape/invariant/boundary 真值源）
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md
- 算子目录：operators/{operator_name}/（已编译通过）

【输出】
- UT用例：operators/{operator_name}/tests/ut/
- UT机器证据
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
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（精度 tolerance / boundary case 真值）
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- 测试用例：operators/{operator_name}/tests/st/testcases/（L0/L1/L2 CSV；禁止使用 tests/cases/）
- 算子目录：operators/{operator_name}/

【输出】
- C++ ST测试工程：operators/{operator_name}/tests/st/
  - test_aclnn_${op_name}.cpp
  - CMakeLists.txt
  - run.sh
  - 黑盒执行清单
  - 开发期执行结果
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 迭代一：L0 标准用例已实现，Mock 编译+CPU Golden 自测通过，机器证据覆盖当前必需用例
- 迭代二：L0+L1 用例已实现，Mock 编译+CPU Golden 自测通过，机器证据覆盖累计必需用例
- 迭代三：L0/L1/L2 全量用例已执行或路由验证，ST/UT 源码、runner 脚本或执行日志保留可执行 case_id 证据
- 非 `smoke_only` 的 ST 路由必须在测试源码、runner 或执行日志中保留实现调用证据；仅由脚本按 CSV 机械生成 result JSON、但没有对应 `case_id` 执行证据的结果必须判为失败
- 单 case debug、真实 NPU debug 或失败复现结果不得覆盖开发期主证据
- 日志摘要已输出

⚠️ **注意**：本任务只开发 C++ 测试。PyTorch 测试由独立 C 任务一次性完成（L0+L1全量），在最终验收前执行。
  "
}
```

## 算子迭代

```
Task 调用参数：
{
  "description": "迭代二 算子迭代",
  "subagent_type": "ascendc-ops-developer",
  "prompt": "
执行算子迭代任务（基于穿刺结果）。

【输入】
- 骨架代码：operators/{operator_name}/op_kernel/（迭代一主线）
- 穿刺目录：operators/{operator_name}/probe/
- 穿刺汇总：operators/{operator_name}/probe/PROBE_SUMMARY.md
- 详细设计文档：operators/{operator_name}/docs/DESIGN.md

【整合策略】
1. ✅ 成功的穿刺任务（状态=成功）：
   - 复用 .asc 文件中的 Kernel 实现逻辑
   - 适配主线工程结构（Tiling参数、Kernel类命名）
   
2. ⚠️ 部分成功的穿刺任务：
   - 参考实现逻辑，修正边界处理
   - 补充缺失的测试case

3. ❌ 失败的穿刺任务：
   - 基于设计文档重新实现
   - 记录失败原因作为参考

【输出】
- 整合后的算子代码：operators/{operator_name}/op_kernel/
- 整合报告：operators/{operator_name}/docs/INTEGRATION_REPORT.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【验收标准】
- 自定义算子包编译通过
- 多TilingKey代码整合完成
- 无命名冲突
- 日志摘要已输出
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
| ST验证 | 通过/失败 | 通过率: X%，required case_id 缺口: N |
| 前序回归 | 通过/失败/不适用 | - |

**关键指标**:
- UT 总用例数: X, 通过: Y, 失败: Z
- ST 总用例数: X, 通过: Y, 失败: Z
- ST 通过率: X%
```

【验收标准】
1. UT验证和ST验证通过（NPU结果与golden数据比对）
2. 当前迭代用例通过（验收标准从 PLAN.md frontmatter 的 iterations[{N}].acceptance 读取）
3. 前序迭代用例无回归（仅当 N > 1 时需要）
4. 最后一个迭代（N = iteration_count）必须保留可执行 case_id 证据，并覆盖全部必需用例
5. 联调过程中的单 case debug 或真实 NPU debug 不得覆盖开发期主证据
6. 日志摘要已输出

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
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（验收对照精度 tolerance）

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
| 黑盒CSV覆盖 | 通过/失败 | 期望: X, 执行: Y, 路由验证: Z, 缺口: N |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%
```

【验收标准】
- 验收标准从 PLAN.md frontmatter 的 iterations[{N}].acceptance 读取
- 最后一个迭代（N = iteration_count）：全部必需用例均已执行或路由验证，ST/UT 源码、runner 脚本或执行日志保留可执行 case_id 证据，累计通过率 = 100%（无回归）
- 非最后迭代：当前迭代用例覆盖完整，ST通过率 = 100%
- 真实 NPU 单 case 调试输出不得覆盖最终主证据
- 日志摘要已输出
  "
}
```

---

## 白盒测试生成与用例汇合（主 Agent 编排）

本任务由主 Agent 按 `ascendc-whitebox-design` skill 启动白盒子 Agent/子任务完成白盒用例生成，不能通过 `ascendc-ops-tester` 间接转派白盒分析。主 Agent 只负责提供上下文和接收日志摘要，白盒生成要求以该 skill 定义的工作流为准。

【输入】
- 算子目录：operators/{operator_name}/
- 需求文档、spec.yaml、DESIGN.md、TEST.md
- 当前实现源码与 UT/ST 证据

【输出】
- 白盒 skill 定义的用例与执行结果

【验收标准】
- 白盒生成由 `ascendc-whitebox-design` skill 定义的工作流完成
- 白盒用例、执行结果和证据汇总已产出

---

## PyTorch ST 测试开发（独立任务）

```
Task 调用参数：
{
  "description": "PyTorch ST测试开发",
  "subagent_type": "ascendc-ops-tester",
  "prompt": "
scene: test-development

执行 PyTorch ST 测试工程开发任务。

【任务说明】
本任务独立于 C++ ST 测试开发（B任务），一次性完成 PyTorch 适配层开发和 L0+L1 全量测试用例实现。
建议在迭代三验收后、最终精度验收前执行（此时算子功能已完整，可直接开发全量用例）。

【输入】
- 需求文档（含ACLNN接口定义）：operators/{operator_name}/docs/REQUIREMENTS.md
- L0 数学契约：operators/{operator_name}/docs/spec.yaml（精度 tolerance / boundary case 真值）
- 测试设计文档：operators/{operator_name}/docs/TEST.md
- C++ ST测试工程：operators/{operator_name}/tests/st/（参考 CPU golden 实现）
- 算子目录：operators/{operator_name}/

【输出】
- PyTorch ST测试工程：operators/{operator_name}/tests/st/torch/
  - CMakeLists.txt          # PyTorch 适配层构建配置
  - test.py                 # 测试入口（用例定义 + 调度）
  - golden.py               # CPU golden 计算
  - compare.py              # 精度比对逻辑
  - torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装
  - build/libtorch_adapter.so  # 编译产物
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【技术参考】
- 完整样例：skills/ascendc-registry-invoke-template/references/add_example/tests/st/torch/
- 开发指南：skills/ascendc-registry-invoke-template/references/st-test-guide.md → 第4章

【验收标准】
- [ ] torch/ 目录结构完整
- [ ] golden.py 实现正确（CPU golden 自测通过）
- [ ] compare.py 精度比对逻辑正确（使用 MERE/MARE 社区标准）
- [ ] test.py L0+L1 全量用例已实现（对应 C++ 全量用例覆盖）
- [ ] torch_adapter.cpp 开发完成（含 ACLNN 两段式封装）
- [ ] CMakeLists.txt 配置正确
- [ ] 编译通过（生成 libtorch_adapter.so）
- [ ] CPU Golden 自测通过（python3 test.py --lib ./build/libtorch_adapter.so）
- 日志摘要已输出

⚠️ **重要**：本任务在最终精度验收前一次性完成，不分迭代。完成后方可执行最终精度验收。
  "
}
```

---

## 3.1 最终精度验收

```
Task 调用参数：
{
  "description": "最终精度验收",
  "subagent_type": "ascendc-ops-tester",
  "prompt": "
scene: test-execution

执行最终精度验收任务。

【输入】
- 算子目录：operators/{operator_name}/

【输出】
- 最终精度验收报告：operators/{operator_name}/docs/precision-report.md
- 日志摘要：输出到响应末尾（格式见"Subagent 日志摘要输出要求"）

【报告格式】（必须包含以下字段）
```
**状态**: ✅通过 / ❌失败

**验收摘要**:
| 验收项 | 结果 | 详情 |
|-------|------|------|
| NPU精度验证 | 通过/失败 | 通过率: X% |
| dtype覆盖 | 通过/失败 | fp16/fp32/bf16/int/uint |

**关键指标**:
- 总用例数: X
- 通过数: Y
- 失败数: Z
- 通过率: X%
```

【验收标准】
- 使用PyTorch测试执行完整L0+L1用例批量验收
- PytTorch NPU结果比对通过（真实NPU）
- 通过率 = 100%
- 日志摘要已输出
  "
}
```

## 3.2 性能达标验收

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
- 运行日志：operators/{operator_name}/examples/run.log（运行 examples/run.sh 将输出重定向到此文件）
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

> ⚠️ 本步骤由 `/ascendc-code-review` skill 接管，主 Agent 加载 skill 后按其工作流执行。子 Agent 的派发由 skill 内部编排自行管理。
> 检视流程的内部编排由 skill 自行管理，详见 SKILL.md 4.2 节。

### 4.2 输入参数

```
4.2 代码检视：
  - 检视文件: operators/{operator_name}/op_kernel/ + op_host/ 下所有 .cpp/.h/.hpp
  - 概要分析输出：`operators/{operator_name}/tmp/checks/code_summary.md`
  - API 预研报告（如有）：`operators/{operator_name}/tmp/checks/api_prestudy.md`
  - 报告路径: operators/{operator_name}/tmp/checks/{source_file}_review_summary.md
```

### 4.2 验收标准

```
4.2：检视报告已生成，无 HIGH 级别"发现问题"，设计一致性 S1-S7 判定无 ❌ 项（若报告含设计一致性章节）
```

### 4.2 检视结果处理规则

```
├─ 无 HIGH + 无 ❌ → 进入 CP5
├─ 有 HIGH（仅规范）+ 无 ❌ → 修复 → 重跑 4.2
├─ 有 HIGH（仅规范）+ 有 ❌ → 修复 → 重跑 4.2 → 重跑阶段三
├─ 有 HIGH（逻辑）→ 修复 → 重跑 4.2 → 重跑阶段三
└─ 有 ❌ → 修复 → 重跑 4.2 → 重跑阶段三
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
