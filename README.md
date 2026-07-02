# CANNBot Skills

## 📢 项目概述

**CANNBot** 是面向 CANN 开发的用于提升开发效率的系列智能体，本仓库为其提供可复用的 Skills 模块，目前已覆盖 Ascend C / Catlass / PyPTO / TileLang / Triton 算子开发流程、torch.compile 图模式优化和 NPU 模型推理端到端优化。

**面向用户**：CANN 社区开发者、昇腾 NPU 平台 AI 应用开发者、Ascend C / Catlass / PyPTO / TileLang / Triton 算子开发者、torch.compile 图模式开发者、模型推理优化开发者、Skills/Agents 贡献者。

## 🚀 快速开始

### 前置条件

- [Node.js 18+](https://nodejs.org)（终端运行 `node --version` 检查，未安装请前往官网下载）
- 以下任一 AI 编程工具：[OpenCode](https://opencode.ai/docs/zh-cn) / [Claude Code](https://code.claude.com/docs/zh-CN/overview) / [Trae](https://www.trae.cn) / [Cursor](https://cursor.com) / [GitHub Copilot](https://github.com/features/copilot)

> CANN 开发环境为可选项，仅算子编译/运行/开发类 Skills 需要；知识检索、文档查阅类 Skills 无需 CANN 环境即可使用。

### 安装

本项目提供两种安装方式，按需选择：

| | 完整安装（推荐） | 独立 Skill 安装 |
|---|---|---|
| **包含内容** | Skills + Agents + Workflows + 外部依赖 | 仅 Skills |
| **安装工具** | install-helper / init.sh | install-helper / npx skills |
| **适用场景** | 需要完整开发流程编排 | 只需单个能力 |
| **支持工具** | OpenCode/Claude/Trae/Cursor/Copilot | 70+ 种 AI 编程工具 |

> **外部依赖**指安装时由安装脚本动态克隆的外部仓库（如 `asc-devkit`、`pypto`、`tilelang-ascend`、`cann-samples` 等）。

**完整安装（推荐）**：

方式一：通过 install-helper 一键安装

```bash
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash
```

安装后在你的工作目录运行 `install-helper` 启动交互式向导。

方式二：手动执行插件 init.sh 脚本

```bash
cd plugins-official/ops-direct-invoke
bash init.sh project opencode
```

`<tool>` 为目标 AI 编程工具名称（支持 `opencode` / `claude` / `trae` / `cursor` / `copilot`），各插件详细步骤参见对应目录下的 `quickstart.md`。

**独立 Skill 安装**：

本仓库 Skills 遵循 [Agent Skills](https://agentskills.io) 开放标准，可安装到 70+ 种 AI 编程工具。

```bash
# 浏览可用 Skills
npx skills add https://gitcode.com/cann/cannbot-skills.git --list

# 安装单个 Skill（交互式选择目标工具）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-env-check

# 查看已安装的 Skills
npx skills list
```

> 此方式仅安装独立 Skills，不含 Agents/Workflows。如需完整开发流程，请使用上方完整安装。

> 安装成功后，完整安装可运行 `install-helper status` 查看已安装插件，独立 Skill 安装可运行 `npx skills list` 查看已安装 Skills。

### 安装后使用

安装完成后，在你的工作目录中启动 AI 编程工具，直接用自然语言描述需求即可：

| 你想做什么 | 可以这样说 | 用到的能力 |
|-----------|-----------|-----------|
| 开发一个算子 | "帮我开发一个 Abs 算子，输入 float16，输出 float16" | Plugin: `ops-direct-invoke` |
| 调试精度问题 | "我的 Add 算子精度不达标，帮我排查一下" | Skill: `ascendc-precision-debug` |
| 查阅 API 文档 | "aclnnAdd 接口的参数和返回值是什么" | Skill: `ascendc-docs-search` |
| 检查开发环境 | "帮我检查一下当前的 CANN 开发环境" | Skill: `ascendc-env-check` |
| 代码检视 | "帮我检视这段 Kernel 代码是否符合规范" | Plugin: `ops-code-reviewer` |

> **Plugin** = Skills + Agents + Workflows 的组合包，提供端到端开发流程编排；**Skill** 是独立能力模块。表中标注 Plugin 的能力需通过完整安装获取，标注 Skill 的能力两种安装方式均可使用。

> 完整命令参考与故障排查详见 [安装指南](docs/installation-guide.md)。

## 📐 开发路径与核心能力

全量 Skills/Agents 清单见 [功能清单](docs/feature-list.md)，使用样例见 [Skills 使用样例](docs/skills-usage.md)。

| 领域 / 路径 | 适用场景 | 入口插件 | 代表性 Skills |
|------------|---------|---------|-------------|
| **Ascend C 直调** | `<<<>>>` 直接调用核函数 | ops-direct-invoke / ops-direct-invoke-flash | ascendc-tiling-design · ascendc-precision-debug · ascendc-crash-debug · ascendc-docs-search |
| **Ascend C 注册调用** | ACLNN/GEIR 接入框架的标准算子 | ops-registry-invoke | ascendc-registry-invoke-template · ascendc-code-review |
| **Catlass** | Cube/Matmul 高阶模板拼装 pipeline | catlass-op-generator | catlass-op-design · catlass-op-develop · catlass-op-perf-tune |
| **PyPTO** | Python API 快速验证与原型开发 | pypto-op-orchestrator | pypto-op-design · pypto-op-develop · pypto-op-perf-tune |
| **TileLang** | DSL + `@tilelang.jit`，Developer/Expert 双模式 | tilelang-op-orchestrator | tilelang-op-develop · tilelang-perf-optimization |
| **Triton** | `triton_ascend` DSL 生成并优化算子 | triton-op-generator | triton-op-coding · triton-latency-optimizer |
| **torch.compile 图模式** | npugraph_ex 图捕获与重放 | torch-compile | torch-npugraph-ex-knowledge · torch-npugraph-ex-dfx-triage · torch-custom-ops-guide |
| **模型推理优化** | NPU 推理端到端优化 | model-infer-optimize | model-infer-migrator · model-infer-fusion · model-infer-graph-mode · model-infer-quantization |
| **治理与协作** | Skill 审查、GitCode PR/Issue 自动化 | — | cannbot-skill-reviewer · gitcode-pr-handler · gitcode-issue-handler |

## 🔍 项目架构

### 整体架构

```
cannbot-skills/
├── ops/                  # 算子 Skills（正式版）
├── ops-lab/              # 算子 Skills（实验版）
├── model/                # 模型推理优化 Skills
├── graph/                # torch.compile 图模式 Skills
├── infra/                # 基础设施 Skills（治理/GitCode 协作）
├── plugins-official/     # 官方 Plugin（开发路径入口，含 Agents/Teams）
├── plugins-community/    # 社区 Plugin
├── docs/                 # 项目文档
└── tests/                # 自动化测试框架
```

### 逻辑架构视图

三层架构：Teams 编排 Agents，Agents 绑定 Skills。Teams 是应用编排层，定义完整开发流程中各 Agent 的协作顺序；Agents 是角色执行层，每个 Agent 承担方案设计、代码开发、代码检视等特定职责；Skills 是知识能力层，为 Agent 提供领域知识和工程模板。以 Ascend C 算子开发为例：

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────┐        ┌─────────────────────────────┐       ║
║  │  ops-direct-invoke          │        │  ops-registry-invoke        │       ║
║  │  Kernel 直调开发流程         │        │  算子注册调用开发流程         │       ║
║  └──────┬──────┬──────┬────────┘        └──────┬──────┬──────┬────────┘       ║
║         │      │      │                        │      │      │                ║
╚═════════╪══════╪══════╪════════════════════════╪══════╪══════╪════════════════╝
          │      │      │                        │      │      │
          ▼      ▼      ▼                        ▼      ▼      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       ║
║  │  architect   │  │  developer   │  │  reviewer    │  │  tester      │       ║
║  │   方案设计    │  │   代码开发   │  │   代码检视    │  │   代码测试    │       ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
             │ │ │ │       │ │ │ │       │ │ │ │       │ │ │ │
             ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ 知识库类 ──────────────────────────────────────────────────────────────┐  ║
║  │  npu-arch                   NPU 架构知识与芯片映射                       │  ║
║  │  ascendc-tiling-design      Tiling 设计方法论                           │  ║
║  │  ascendc-api-best-practices API 使用最佳实践                            │  ║
║  │  ops-precision-standard     算子精度标准                                │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌─ 工程模板类 ────────────────────────────────────────────────────────────┐  ║
║  │  ascendc-registry-invoke-to-direct-invoke 注册算子直调改造模板           │  ║
║  │  ascendc-direct-invoke-template           Kernel直调工程模板             │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─ 调试与测试类 ──────────────────────────────────────────────────────────┐  ║
║  │  ascendc-precision-debug 精度调试与症状速查                              │  ║
║  │  ascendc-runtime-debug   运行时错误码解析                                │  ║
║  │  ascendc-crash-debug     卡死/崩溃调试、Coredump 分析                    │  ║
║  │  ascendc-env-check       NPU 设备查询与环境验证                          │  ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  ┌─ 测试开发类 ────────────────────────────────────────────────────────────┐  ║
║  │  ascendc-st-design   ST 测试用例设计                                    │  ║
║  │  ascendc-ut-develop  UT 开发与覆盖率增强                                │  ║
║  │  ascendc-code-review 代码检视规则                                       │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌─ 工具辅助类 ────────────────────────────────────────────────────────────┐  ║
║  │  ops-profiling      算子性能采集分析                                     │  ║
║  │  ascendc-task-focus 长任务聚焦防迷失                                     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

其他路径（Catlass / PyPTO / TileLang / Triton / torch.compile / 模型推理）的逻辑架构图见 [项目架构设计](docs/architecture-design.md)。

## 🔥 最新动态

最新更新与历史记录详见 [CHANGELOG.md](CHANGELOG.md)。

## 💬 相关信息
- [贡献指南](docs/CONTRIBUTING.md) — 贡献流程、准入门槛、评审规则
- [治理模型](docs/GOVERNANCE.md) — 成文法+判例法、角色职责
- [开发规范](docs/STANDARDS.md) — 命名规范、结构规范、分类体系
- [测试框架](tests/README.md) — 自动化测试验证 Skills 和 Agents 的正确性
- [许可证](LICENSE)
- [所属 SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)

## 💖 免责声明

感谢您关注 CANNBot Skills 项目，我们希望这些技能和知识能帮助您更好地进行 CANN 开发 ^_^

在使用之前，请您了解：

1. **关于功能满足度**：由于技术快速更新迭代，部分内容可能无法完全适用于所有场景。本开源社区的功能和文档正在持续更新和完善中，如果想提出需求、发现问题、贡献想法，欢迎提 Issue 或参与讨论，共创共建。

2. **关于自动生成**：自动代码生成工具所产出的内容，其完整性、准确性、合规性受模型、Skills 能力、语料质量、输入指令等多种因素影响，无法保证完全精准。所有生成代码作为辅助研发使用，请开发者务必进行测试验证、安全审查后再投入使用。

## 🤝 社区交流

- **Issue 反馈**：[提交 Issue](https://gitcode.com/cann/cannbot-skills/issues)
- **社区讨论**：[参与讨论](https://gitcode.com/cann/cannbot-skills/discussions)
- **微信交流**：[加入群聊](https://gitcode.com/cann/cannbot-skills/discussions/2)
