# CANNBot Skills

## 📢 项目概述

### 项目定位

**CANNBot** 是面向 CANN 开发的用于提升开发效率的系列智能体，本仓库为其提供可复用的 Skills 模块，目前已覆盖 Ascend C / PyPTO / TileLang 算子开发流程和 NPU 模型推理端到端优化。

### 目标用户

- CANN 社区开发者
- 昇腾 NPU 平台 AI 应用开发者
- Ascend C / PyPTO / TileLang 算子开发者
- 使用昇腾 NPU 进行模型推理优化的开发者
- 希望贡献 Skills / Agents 的社区贡献者

&nbsp;

## 🔥 最新动态
- **2026-04-30** — 【代码仓库更名】https://gitcode.com/cann/skills 更名为 https://gitcode.com/cann/cannbot-skills ，原名称和路径可继续访问，建议使用新名称和路径。
- **2026-04-29** — 新增自定义算子注册调用的脚手架工程，支持通过ACLNN和GEIR接入，ascendc-registry-invoke-template的Skill。
- **2026-04-28** — 新增支持TRAE安装。
- **2026-04-25** — 增强 ascendc-precision-debug和ascendc-runtime-debug的调试能力。
- **2026-04-24** — 新增 Ascend C 性能调优知识货架。
- **2026-04-23** — 在 Readme.md 新增 Skills 的使用样例。
- **2026-04-21** — 修复测试框架并解决识别到的多项校验问题
- **2026-04-20** — 新增 regbase 配置最佳实践，修复环境检查设备计数 bug，统一算子目录命名（ops → operators）。
- **2026-04-18** — 修复 ops-profiling 技能名称不一致的问题。
- **2026-04-17** — 新增精度模式自动生成、完善 broadcast tiling 设计，新增初始化脚本和快速入门指南
- **2026-04-16** — 新增 Plugin 化安装体系（Claude Code / OpenCode），新增 NPU 模型推理端到端优化 Skill 体系（含 3 个 SubAgent 和 infer-model-optimize-team）。
- **2026-04-14** — 新增 Ascend 950 仿真 Skill，增强 UT / ST 测试能力，修复 verify_environment.sh 设备计数的问题。
- **2026-04-13** — 支持 Team 级代码检视全量覆盖，子 Agent 并行检视提升效果。
- **2026-04-10** — 新增 PyPTO 算子开发全套体系（8 个 Skills + 3 个 Agents + 1 个 Team）。

&nbsp;
> 详细变更记录，详见 [CHANGELOG.md](CHANGELOG.md) 文件。

&nbsp;

## ⚡️快速开始

### 方式一：Plugin 安装（推荐）

#### Claude Code

```bash
# 注册 marketplace（首次）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
# 安装 Ascend C 算子直调开发
/plugin install ops-direct-invoke@cannbot
# 或安装 PyPTO 算子开发
/plugin install pypto-op-orchestrator@cannbot
# 激活插件（加载 Skills/Agents/Hooks）
/reload-plugins
# 触发初始化：以下任一方式均可（注入 CANNBot 上下文）
# 方式 a：新开会话（推荐，自然触发 SessionStart）
# 方式 b：在当前会话中执行 /clear（会清空当前对话历史）
```

#### OpenCode

```bash
# 项目级安装
opencode plugin cannbot@git+https://gitcode.com/cann/cannbot-skills.git
# 全局安装（所有项目可用）
opencode plugin cannbot@git+https://gitcode.com/cann/cannbot-skills.git -g
```

安装后重启 OpenCode。按 Team 精简安装需编辑 `.opencode/opencode.json`：
```json
{
  "plugin": [["cannbot@git+https://gitcode.com/cann/cannbot-skills.git", {"team": "ops-direct-invoke"}]]
}
```

| Team | Agents | Skills |
|------|--------|--------|
| `all` | 8 | 25 |
| `ops-direct-invoke`（默认） | 3 | 11 |
| `pypto-op-orchestrator` | 3 | 8 |

也可以让 OpenCode 自动安装：
```bash
Fetch and follow instructions from https://gitcode.com/cann/cannbot-skills/blob/.opencode/INSTALL.md
```


### 方式二：脚本安装

**Ascend C 算子开发**

适用于 Ascend C 算子直调开发场景，自动安装 Skills、配置文件及 asc-devkit 工具包。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/plugins-official/ops-direct-invoke
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
```

详细说明见 [plugins-official/ops-direct-invoke/quickstart.md](plugins-official/ops-direct-invoke/quickstart.md)。

**PyPTO 算子开发**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/plugins-official/pypto-op-orchestrator
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
```

详细说明见 [plugins-official/pypto-op-orchestrator/quickstart.md](plugins-official/pypto-op-orchestrator/quickstart.md)。

**NPU 模型推理优化**

适用于 PyTorch 模型的昇腾 NPU 推理性能优化场景。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/model/teams/infer-model-optimize-team
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
```

详细说明见 [model/teams/infer-model-optimize-team/quickstart.md](model/teams/infer-model-optimize-team/quickstart.md)。

### 方式三：手动安装

仅安装 Skills 和 Agents，适用于自定义配置场景。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills
# OpenCode 用户
mkdir -p .opencode && ln -s ../ops .opencode/skills
# Claude 用户：将 .opencode 替换为 .claude
```

> 如需全局安装，OpenCode 用户将 `.opencode` 替换为 `~/.config/opencode`，Claude 用户替换为 `~/.claude`。

### 启动 CLI

```bash
opencode
```

&nbsp;

## 🔍 项目架构设计

### 整体架构

```
skills/
├── ops/                    # 算子 Skills（正式版：Ascend C + PyPTO）
├── ops-lab/               # 算子 Skills（实验 / 非正式版）
├── model/                 # 模型推理优化
│   ├── skills/            # 推理优化技能模块
│   ├── agents/            # 子 Agent（analyzer / implementer / reviewer）
│   └── teams/             # 多 Agent 协同
│       └── infer-model-optimize-team/  # NPU 推理端到端优化流程
│           ├── model-infer-optimize/   # 工作流入口 Skill
│           └── hooks/                  # Hook 约束脚本
└── tests/                 # 自动化测试框架
```

### 逻辑架构视图

项目遵循三层架构：Teams 编排 Agents，Agents 绑定 Skills。以下视图展示各层组件及其关联关系。

#### Ascend C 算子开发

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────┐        ┌─────────────────────────────┐      ║
║  │  ops-direct-invoke          │        │  ascendc-ops-dev-team ⬆    │      ║
║  │  Kernel 直调开发流程         │        │  自定义算子开发流程           │      ║
║  └──────┬──────┬──────┬────────┘        └──────┬──────┬──────┬────────┘      ║
║         │      │      │                        │      │      │               ║
╚═════════╪══════╪══════╪════════════════════════╪══════╪══════╪═══════════════╝
          │      │      │                        │      │      │
          ▼      ▼      ▼                        ▼      ▼      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ║
║  │  architect   │  │  developer ⬆│  │  reviewer    │  │  tester ⬆   │      ║
║  │   方案设计    │  │   代码开发   │  │   代码检视    │  │   代码测试    │      ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
             │ │ │ │       │ │ │ │       │ │ │ │       │ │ │ │
             ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─ 知识库类 ──────────────────────────────────────────────────────────────┐  ║
║  │  npu-arch             NPU 架构知识与芯片映射                             │  ║
║  │  tiling-design        Tiling 设计方法论                                 │  ║
║  │  api-best-practices   API 使用最佳实践                                   │  ║
║  │  ops-precision-standard 算子精度标准                                     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─ 工程模板类 ────────────────────────────────────────────────────────────┐  ║
║  │  registry-invoke-to-direct-invoke  注册算子直调改造模板                   │   ║
║  │  direct-invoke-template            Kernel直调工程模板                     │   ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 调试与测试类 ──────────────────────────────────────────────────────────┐  ║
║  │  precision-debug      精度调试与症状速查                                 │   ║
║  │  runtime-debug        运行时错误码解析                                   │  ║
║  │  ops-profiling        算子性能采集分析                                   │   ║
║  └────────────────────────────────────────────────────────────────────────┘  ║ 
║                                                                               ║
║  ┌─ 测试开发类 ────────────────────────────────────────────────────────────┐  ║
║  │  st-design            ST 测试用例设计                                   │   ║
║  │  ut-develop           UT 开发与覆盖率增强                               │   ║
║  │  code-review          代码检视规则                                      │  ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─ 工具辅助类 ────────────────────────────────────────────────────────────┐   ║
║  │  env-check            NPU 设备查询与环境验证                             │  ║
║  │  task-focus           长任务聚焦防迷失                                   │   ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

> 标注 ⬆ 的组件在规划中，后续会在社区上线

#### PyPTO 算子开发

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                    ┌─────────────────────────────────┐                       ║
║                    │      pypto-op-orchestrator      │                       ║
║                    │      PyPTO 算子开发流程         │                       ║
║                    └──────┬──────────┬──────────┬────┘                       ║
║                           │          │          │                            ║
╚═══════════════════════════╪══════════╪══════════╪════════════════════════════╝
                            │          │          │
                            ▼          ▼          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║             ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             ║
║             │  analyst     │  │  developer   │  │  perf-tuner  │             ║
║             │  需求与设计  │  │  实现与精度  │  │  性能调优    │             ║
║             └──────────────┘  └──────────────┘  └──────────────┘             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                     │ │ │           │ │ │           │ │ │
                     ▼ ▼ ▼           ▼ ▼ ▼           ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ 需求与设计 ────────────────────────────────────────────────────────────┐  ║
║  │  intent-understand    需求意图理解与规格生成                            │  ║
║  │  api-explore          API 可行性探索与分析                              │  ║
║  │  op-design            算子方案设计生成                                  │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 实现与验证 ────────────────────────────────────────────────────────────┐  ║
║  │  golden-generate      Golden 参考实现生成                               │  ║
║  │  op-develop           算子代码实现与调试                                │  ║
║  │  precision-debug      精度问题诊断                                      │  ║
║  │  precision-compare    精度对比分析                                      │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 性能调优 ──────────────────────────────────────────────────────────────┐  ║
║  │  op-perf-tune         算子性能分析与调优                                │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

&nbsp;

## 🚀 Skills 技能库

### Ascend C 算子开发

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **ascendc-api-best-practices** | API 使用最佳实践、参数限制 | — |
| **ascendc-npu-arch** | NPU 架构知识、芯片型号映射 | — |
| **ascendc-docs-search** | API 文档索引 + 在线搜索 | — |
| **ascendc-env-check** | NPU 设备查询、CANN 环境验证 | — |
| **ascendc-tiling-design** | Tiling 和 Kernel 设计方法论，按算子类别分类 | — |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 | — |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析、Kernel 挂起排查 | — |
| **ascendc-ut-develop** | UT 单元测试用例开发与覆盖率增强 | — |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0 / L1 测试用例生成 | — |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 | — |
| **ascendc-task-focus** | 任务聚焦，解决长任务“迷失在中间”的问题 | — |
| **ascendc-whitebox-design** | 白盒测试用例设计与生成 | — |
| **ascendc-registry-invoke-template** | 完整自定义算子工程模板，提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考 | — |
| **ascendc-registry-invoke-to-direct-invoke** | 注册调用算子转 `<<<>>>` kernel 直调 | [查看](docs/skills-usage.md#ascendc-registry-invoke-to-direct-invoke) |
| **ascendc-direct-invoke-template** | Kernel 直调工程模板，提供验证过的样例工程和修改指南 | — |
| **ops-profiling** | NPU 性能采集与分析，CSV 指标解读、瓶颈定位、优化建议 | — |
| **ops-precision-standard** | 算子精度标准，按 dtype 分类提供 atol/rtol 精度比对标准 | — |
| **ascendc-docs-gen** | 算子文档写作参考，支持需求分析、详细设计等多个标准模版 | — |
| **cann-simulator** | NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。 | — |

### PyPTO 算子开发

| Skill | 功能 |
|-------|------|
| **pypto-op-design** | 算子方案设计生成 |
| **pypto-op-develop** | 算子代码实现与测试 |
| **pypto-golden-generate** | Golden 参考实现生成 |
| **pypto-intent-understand** | 需求意图理解与规格生成 |
| **pypto-api-explore** | API 可行性探索与分析 |
| **pypto-precision-debug** | 精度问题代码层排查 |
| **pypto-precision-compare** | 精度中间结果对比分析 |
| **pypto-op-perf-tune** | 算子性能分析与自动调优 |

&nbsp;
### NPU 模型推理优化

| Skill | 功能 |
|-------|------|
| **model-infer-optimize** | 端到端优化编排入口，阶段 0-5 全流程 |
| **model-infer-migrator** | 框架适配与部署基线建立 |
| **model-infer-parallel-analysis** | 并行策略分析（TP/EP/DP） |
| **model-infer-parallel-impl** | 并行切分实施 |
| **model-infer-kvcache** | KVCache 优化 + FA 替换 |
| **model-infer-fusion** | torch_npu 融合算子分析与替换 |
| **model-infer-graph-mode** | torch.compile 图模式适配 |
| **model-infer-precision-debug** | NPU 推理精度诊断 |
| **model-infer-runtime-debug** | NPU 运行时错误诊断 |
| **model-infer-multi-stream** | 多流并行优化 |
| **model-infer-prefetch** | 权重预取适配 |
| **model-infer-superkernel** | SuperKernel 适配 |

&nbsp;

## 🚀 Agents 智能代理

### Ascend C 算子开发

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-reviewer** | 代码检视专家，支持快速检视和全功能检视两种模式 |
| **ascendc-kernel-architect** | Kernel直调架构师，支持需求分析、API验证、方案设计 |
| **ascendc-kernel-developer** | Kernel直调开发者，支持代码实现、编译测试、性能采集、文档编写 |
| **ascendc-kernel-reviewer** | Kernel直调审查者，支持独立构建验证、7维度评分、精度验证 |

### PyPTO 算子开发

| Agent | 功能 |
|-------|------|
| **pypto-op-analyst** | 需求分析与方案设计 |
| **pypto-op-developer** | 算子代码实现与精度调试 |
| **pypto-op-perf-tuner** | 性能分析与调优 |

&nbsp;

### NPU 模型推理优化

| Agent | 功能 |
|-------|------|
| **model-infer-analyzer** | 模型分析、方案设计、并行策略推荐 |
| **model-infer-implementer** | 代码改造、调试修复 |
| **model-infer-reviewer** | 精度验证、性能对比 |

&nbsp;

## 🛠️ 测试框架

自动化测试验证 Skills 和 Agents 的正确性，确保技能模块和智能代理的行为符合预期。
详见 [tests/README.md](tests/README.md)。

&nbsp;

## 💬相关信息
- [贡献指南、开发规范](docs/STANDARDS.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)

&nbsp;

## 💖 免责声明

感谢您关注 CANNBot Skills 项目，我们希望这些技能和知识能帮助您更好地进行 CANN 开发^_^

在使用之前，请您了解：

1. **关于功能满足度**：由于技术快速更新迭代，部分内容可能无法完全适用于所有场景。 本开源社区的功能和文档正在持续更新和完善、丰富场景中，如果想提出需求、发现问题、贡献想法，非常欢迎提 Issue、讨论来告诉我们，共创共建。

2. **关于自动生成**：自动代码生成工具所产出的内容，其完整性、准确性、合规性，受模型类型、模型版本、Skills 能力、语料质量、输入指令、运行环境等多种因素影响，无法保证完全精准、尽善尽美。所有生成代码作为辅助研发使用，请开发者务必进行测试验证、安全审查后再投入使用。

&nbsp;

## 🤝 联系我们
### 需求、问题、咨询、任务、文档
通过GitCode[【Issues】](https://gitcode.com/cann/cannbot-skills/issues)提交。
   
### 社区互动
通过GitCode[【讨论】](https://gitcode.com/cann/cannbot-skills/discussions)参与交流。
   
### 联系我们
[【微信交流群】](https://gitcode.com/cann/cannbot-skills/discussions/2)

