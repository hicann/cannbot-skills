# Contributing to CANNBot Skills

欢迎参与 CANNBot Skills 生态建设。本文档说明如何贡献 Skill、Agent、Plugin(Team)以及如何参与社区治理。

> **相关文档**:
> - [GOVERNANCE.md](./GOVERNANCE.md) — 治理模型、合入规则、角色职责、修订流程
> - [STANDARDS.md](./STANDARDS.md) — 命名规范、结构规范、分类体系、代码规范

---

## 快速开始

### 前置条件

1. 阅读[社区行为准则](https://gitcode.com/cann/community)
2. 签署 [CLA 协议](https://gitcode.com/cann/community)
3. 了解本项目的[治理模型](./GOVERNANCE.md)和[开发规范](./STANDARDS.md)

### 我能贡献什么

| 贡献类型 | 说明 | 对应文档 |
|---------|------|---------|
| 新增 Skill | 添加新的技能模块 | [STANDARDS.md §Skills 开发规范](./STANDARDS.md#skills-开发规范) |
| 新增 Agent | 添加新的子代理角色 | [STANDARDS.md §Agents 开发规范](./STANDARDS.md#agents-开发规范) |
| 新增 Plugin (Team) | 添加完整的应用插件 | [STANDARDS.md §Teams 配置](./STANDARDS.md#teams-配置) |
| 优化已有内容 | 改进已有 Skill/Agent/Plugin 的效果、准确性或性能 | 直接提交 PR，建议附优化效果说明 |
| Bug 修复 | 修复已有 Skill/Agent 的问题 | 直接提交 PR |
| 文档改进 | 修正文档错误、补充说明 | 直接提交 PR |
| 测试补充 | 增加测试覆盖 | 参考 `tests/` 目录 |

---

## 贡献流程

### 总体流程

```
提出 Issue → 讨论方案 → 提交 PR → CI 自动检查 → Committer 审批 → 合入
```

### 何时需要先提交 Issue

以下情况**必须**先提交 Issue 进行方案讨论:

- 新增 Skill、Agent 或 Plugin(Team)
- 对现有 Skill 的功能范围做重大调整
- 修改 Skill 分类归属
- 不确定改动是否需要讨论时

以下情况可直接提交 PR:

- 文档纠错(错别字、格式修正)
- 不改变逻辑的代码风格修正
- 简单的 Bug 修复(经 Issue 讨论确认后)

### 提交 Issue

在[代码仓讨论区](https://gitcode.com/cann/cannbot-skills/discussions)提交 Issue，需包含:

**新增 Skill**:
- 解决的问题和使用场景(附事实来源)
- **案例佐证**:真实案例说明该 Skill 解决的痛点(如典型报错场景、用户实际困难、同类问题的反复出现)
- 与已有 Skill 的差异分析
- 初步的设计思路

**新增 Plugin(Team)**:
- Plugin 的目标用户和使用场景
- 申请目标:`plugins-official` 或 `plugins-community`
- 对于 official 申请，附生产价值佐证和维护计划

### 提交 PR

PR 需包含:
- 新增/变更的能力文件
- 功能说明
- 更新 `CHANGELOG.md`

> PR 模板请参考仓库中的 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`

### 成果展示(推荐)

为方便 Committer 了解贡献效果，鼓励在 [Discussions](https://gitcode.com/cann/cannbot-skills/discussions) 中简述 PR 的实际使用效果，例如:解决了什么用户的什么问题、与现有方案的实际对比等。

---

## 质量与准入门槛

### 禁止合入的情况

| 禁止项 | 裁决方式 | 说明 |
|--------|---------|------|
| 单元测试不通过 | 成文法(CI 自动) | 任何 error 级别规则失败 |
| 破坏已有功能 | 成文法(CI 自动) | 导致已有 Skill 的 L2 行为测试或 Plugin 的 L3 集成测试回归 |
| 事实溯源缺失 | 判例法(Committer) | 技术内容无对应的可信来源(见 [GOVERNANCE §1.3](./GOVERNANCE.md#13-事实来源与设计论证原则)) |
| 设计无论证 | 判例法(Committer) | 设计决策缺少案例对比、代价权衡等论证过程 |
| 编造 API/参数 | 判例法(Committer) | 包含未验证的 API 名称、参数、行为描述 |
| 与已有内容功能重叠但无差异化价值 | 判例法(Committer) | 新 Skill 与现有 Skill 解决同一问题且无显著改进 |
| 质量低劣 | 判例法(Committer) | 描述模糊、步骤缺失、无法独立完成其声称的任务 |

### 可合入但需标注的情况

| 情况 | 要求 |
|------|------|
| 实验性 Skill | 放入 `ops-lab/` 或对应实验目录 |
| 部分功能未完成 | 在 SKILL.md 顶部明确标注限制和 TODO |
| 依赖尚在开发的组件 | 在 description 中声明前置条件 |

### plugins-official 准入标准

| 标准 | 裁决方式 | 要求 |
|------|---------|------|
| 生产价值 | 判例法 | 能产生可量化的较大生产价值，成果能在 CANN 社区中体现 |
| 长期承诺 | 判例法 | 有持续演进和维护的明确计划，有指定看护人 |
| 看护机制 | 判例法 | 有 Issue 响应 SLA、PR Review 责任人 |
| 定位不重叠 | 判例法 | 与已有 official plugin 在应用定位上不重叠 |
| 测试覆盖 | 成文法 | 全部 L1 + L2 测试通过 |
| 稳定性 | 成文法 + 判例法 | CI 全绿 + 已在 `plugins-community` 或本地环境经历充分验证 |

### plugins-community 准入标准

| 标准 | 裁决方式 | 要求 |
|------|---------|------|
| 基本质量 | 成文法 | L1 单元测试全部通过 |
| 事实来源 | 判例法 | 内容来自可信来源，无编造 |
| 有使用价值 | 判例法 | 对至少一部分用户有实际帮助 |
| 非破坏性 | 成文法 + 判例法 | CI 通过 + 不影响已有 official plugin 的正常运行 |

### Community → Official 升级路径

1. 已在 community 阶段积累**至少 1 个月**的实际使用数据
2. 提供生产价值的**量化佐证**(如用户反馈、使用量统计、效率提升数据)
3. 有明确的**长期维护计划**和**指定看护人**
4. 通过 Committer 评审

---

## 开发规范速查

- 命名:`{domain}-{name}`，kebab-case
- 设计:单一职责、渐进式披露、信息来源可信、知识依赖单向性

完整规范见 [STANDARDS.md](./STANDARDS.md)。

---

## 评审流程

### CI 自动检查

PR 提交后 CI 自动执行:

| 检查层级 | 内容 | 规则集 |
|---------|------|--------|
| L1 单元测试 | SKILL.md / AGENT.md / AGENTS.md 结构与内容格式 | S-STR-\* / S-CON-\* / A-STR-\* / A-CON-\* / T-STR-\* / T-CON-\* |
| L2 行为测试 | Skill 功能行为验证 | B-\* |
| 增量 CI | 仅测试变更组件，加速反馈 | — |

### Committer 审批

- 合入需 2 位 Committer 审批:1 位 LGTM + 1 位 Approve

---

## 角色概述

| 角色 | 职责 |
|------|------|
| **Contributor** | 提交 Skill/Agent/Plugin 的提案和实现，参与设计讨论 |
| **Committer** | PR 合入审批，判例法决策，Official plugin 准入评估 |
| **SIG** | 受理 Committer 层面无法达成共识的争议，修订治理规范 |

> Committer 清单和 SIG 信息见 [SIG CANNBot 社区页面](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)。

---

## 更多资源

- [治理规范 (GOVERNANCE.md)](./GOVERNANCE.md) — 成文法+判例法模型、判例记录、规范修订
- [开发规范 (STANDARDS.md)](./STANDARDS.md) — 完整的技术标准和代码规范
- [SIG 会议纪要](https://etherpad-cann.meeting.osinfra.cn/p/sig-cannbot) — 判例决策和设计讨论记录
- [CANN 社区代码仓](https://gitcode.com/cann) — 官方文档和社区代码
