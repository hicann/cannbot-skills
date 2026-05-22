# CANNBot Skills 治理规范

本文档定义 CANNBot Skills 生态的治理模型、角色职责与决策流程。

---

## 一、治理原则:成文法 + 判例法

采用 **成文法(Statutory Law)与判例法(Case Law)互补** 的治理模型:

### 1.1 成文法(格式规则，UT 看护)

**适用范围**:Skill / Agent / Team / PR 的基础格式和结构规范。

**执法机制**:以自动化测试(UT + 行为测试)作为唯一合规判断依据，无人为主观裁量空间。

| 成文法规 | 执法方式 | 规则集 |
|---------|---------|--------|
| SKILL.md 前置元数据格式 | `tests/unit/skills/test-structure.sh` | S-STR-01 ~ S-STR-16 |
| SKILL.md 内容质量 | `tests/unit/skills/test-content.sh` | S-CON-01 ~ S-CON-09 |
| AGENT.md 前置元数据格式 | `tests/unit/agents/test-structure.sh` | A-STR-01 ~ A-STR-09 |
| AGENT.md 内容质量 | `tests/unit/agents/test-content.sh` | A-CON-01 ~ A-CON-09 |
| AGENTS.md / Team 结构 | `tests/unit/teams/test-structure.sh` | T-STR-01 ~ T-STR-08 |
| AGENTS.md / Team 内容 | `tests/unit/teams/test-content.sh` | T-CON-01 ~ T-CON-03 |
| Team 版本号 | `tests/unit/teams/test-version.sh` | SemVer + marketplace 一致性 |
| 命名规范 (kebab-case / 域前缀) | 上述结构测试中 | S-STR-02, S-STR-10, S-CON-01 |
| 知识依赖单向性 | `tests/unit/skills/test-content.sh` | S-CON-07(渐进式披露约束) |

**原则**:
- 测试通过则格式合规，测试不通过则禁止合入。
- 成文法修改需同步更新测试规则文件(`tests/lib/rules.yaml`、`tests/lib/skill_validator.py`)及对应的 `test-*.sh`，确保规则与执法一致。
- 新增规则需在 PR 中提供动机说明和覆盖范围分析。

### 1.2 判例法(设计决策，讨论裁决)

**适用范围**:成文法无法覆盖的设计层面问题，需人工判断的场景。

**裁决机制**:
```
提出(Contributor PR/Issue) → Committer 评审决策
                                    ↓
                              ┌─ 复杂/争议性 → SIG 会议讨论决议
                              │
                              └─ 常规 → Committer 独立裁决
                                    ↓
                              归纳为判例(来源可为 SIG 纪要 或 PR 评审意见)
                                    ↓
                              存在争议 → 发起 SIG 重新讨论
```

**判例来源**:

| 来源 | 说明 |
|------|------|
| SIG 会议纪要 | 集体讨论形成的决议 |
| PR 评审意见 | Committer 在 PR 中做出且**实际被采纳**的设计决策(被驳回的意见不构成判例) |

**判例冲突处理**:SIG 会议纪要 > PR 评审意见。当两者存在分歧时，以 SIG 集体决议为准，PR 评审意见被覆盖(不视为无效，但被 SIG 决议取代)。

**典型适用场景**(非穷尽):
- Skill 功能边界划分(一个新 Skill 应独立存在还是并入已有 Skill)
- Agent 职责范围设计(某工作应由哪个 Agent 负责)
- Skill 分类归属(知识库类 vs 工具辅助类的边界判断)
- 知识依赖方向设计(哪个 Skill 是真源 Skill)
- 跨域 Skill 的领域归属
- 新域前缀的引入

**判例生命周期**:解决一类没出现过的问题 -> 记录判例 -> 形成基础规则 -> 发生争议时在 SIG 重新讨论。

### 1.3 事实来源与设计论证原则

**规则和判例的可靠性来自两个维度的约束**:

**事实溯源**(适用于技术内容声明——API、参数、行为、错误码等):

| 可信来源 | 示例 |
|---------|------|
| 仓库内已确立的标准规范 | `docs/STANDARDS.md`、`tests/lib/rules.yaml` |
| CANN 官方文档 | 产品文档、API 参考、发布说明 |
| [CANN 社区代码仓](https://gitcode.com/cann)中的实际实现 | 源码、示例工程 |
| 用户在生产环境中的反馈 | Issue、需求单、实际报错日志 |
| 自动化测试中积累的统计数据 | 行为测试用例、覆盖率报告 |

> 技术声明必须有可信来源引证，**禁止编造 API/参数/行为**。

**设计论证**(适用于架构与设计决策——边界划分、职责分配、依赖方向等):

设计知识本身未必基于事实，而是基于判断。此类决策要求**有合理的推理论证**，包括但不限于:
- 真实案例对比(方案 A vs 方案 B 在同类场景下的表现差异)
- 代价权衡分析(选择某方案的可接受代价是什么)
- 已有判例参考(历史同类决策如何处理的)
- 长期演进影响评估

> 设计决策禁止"我认为"式的武断判断——可以基于判断，但判断必须有论证过程。

---

## 二、准入裁决

合入规则的详细 checklist 见 [CONTRIBUTING.md](./CONTRIBUTING.md#质量与准入门槛)。裁决机制与治理原则一致:

- **成文法事项**(格式、命名、结构合规，已有功能回归)→ CI 自动裁决，不通过则阻断
- **判例法事项**(设计评估、价值判断、定位分析)→ Committer 人工裁决

---

## 三、角色与职责

### 3.1 Committer

Committer 清单见 [SIG CANNBot 社区页面](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)。

- 负责判例法决策:就 Skill/Agent 设计问题进行讨论和裁决
- 负责 PR 合入审批
- 负责 Official plugin 准入评估
- 负责争议升级到 SIG 之前的一级裁决

### 3.2 Contributor

- 提交 Skill / Agent / Team / Plugin 的提案和实现
- 参与设计讨论，提供使用场景和需求的真实数据
- 对判例决策有异议时，可发起 SIG 重议请求

### 3.3 SIG(Special Interest Group)

> **SIG 信息**:[社区页面](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)(含 Committer 清单等正式信息) | [会议纪要 & 签到](https://etherpad-cann.meeting.osinfra.cn/p/sig-cannbot)

- 受理 Committer 层面无法达成共识的争议
- 讨论并修订治理规范本身
- 判例法规则升级为成文法的决策

---

## 四、判例记录

判例不单独上库，援引日期和来源即可。细节在来源中体现:

| 判例来源 | 记载位置 |
|---------|---------|
| SIG 会议决议 | [SIG 会议纪要](https://etherpad-cann.meeting.osinfra.cn/p/sig-cannbot) |
| PR 评审意见 | 对应 PR 中被采纳的评论记录 |

**引用格式**:在相关 Issue / PR / SKILL.md 中以流水线方式引用，例如:

> *此设计决策依据 2026-04-15 SIG CANNBot 会议纪要第 3 项决议。*
> *此设计决策依据 PR #123 中 Committer @xxx 的评审意见。*

引用时若发现不同来源的判例存在冲突，以 SIG 会议纪要为准。

---

## 五、规范修订

| 修订范围 | 流程 |
|---------|------|
| 成文法(测试规则) | PR → CI 通过 → Committer 审批 |
| 判例法变更(推翻已有判例) | 发起 SIG 讨论 → 达成共识 → 记录于会议纪要 |
| 本治理规范 | PR → 至少 2 位 Maintainer 审批 |
