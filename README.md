# CANNBot Skills

## 📢 项目概述

### 项目定位

**CANNBot** 是面向 CANN 开发的用于提升开发效率的系列智能体，本仓库为其提供可复用的 Skills 模块，目前已覆盖 Ascend C / PyPTO / TileLang / Triton 算子开发流程和 NPU 模型推理端到端优化。

### 目标用户

- CANN 社区开发者
- 昇腾 NPU 平台 AI 应用开发者
- Ascend C / PyPTO / TileLang / Triton 算子开发者
- 使用昇腾 NPU 进行模型推理优化的开发者
- 希望贡献 Skills / Agents 的社区贡献者

## 🚀 快速开始

### 前置条件

- [Node.js 18+](https://nodejs.org)（终端运行 `node --version` 检查，未安装请前往官网下载）
- 以下任一 AI 编程工具：[OpenCode](https://opencode.ai/docs/zh-cn) / [Claude Code](https://code.claude.com/docs/zh-CN/overview) / [Trae](https://www.trae.cn) / [Cursor](https://cursor.com)
- CANN 开发环境（仅算子编译/运行类 Skills 需要，知识检索类不受影响）

### 安装

#### 完整安装（推荐）

通过 `install-helper` 安装完整插件内容（Skills + Agents + Workflows + 工具链）：

| 方式 | 命令 | 说明 |
|------|------|------|
| **curl** | `curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh \| bash` | 一键安装（推荐） |
| **npx** | `npx @cannbot-ai/install-helper` | 免安装运行 |
| **npm** | `npm install -g @cannbot-ai/install-helper` | 全局安装 |

安装后在项目目录运行 `install-helper` 启动交互式向导：

1. 自动检测已安装的 AI 编程工具
2. 选择安装类型（完整插件 或 Skill）
3. 选择要安装的插件或Skill（空格选择，回车确认）
4. 确认并自动安装

#### 独立 Skill 安装

通过 [Agent Skills](https://agentskills.io) 标准安装单个 Skill：

```bash
# 浏览可用 Skills
npx skills add https://gitcode.com/cann/cannbot-skills.git --list

# 安装单个 Skill
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-tiling-design
```

安装后直接在 AI 工具中即可使用，无需额外配置。

> 独立 Skill 不含 Agents/Workflows/工具链，如需完整插件内容请使用上方完整安装。

### 安装后使用

安装完成后，在项目目录中启动你的 AI 编程工具，直接用自然语言描述需求即可：

| 你想做什么 | 可以这样说 | 用到的能力 |
|-----------|-----------|-----------|
| 开发一个算子 | "帮我开发一个 Abs 算子，输入 float16，输出 float16" | Plugin: `ops-direct-invoke` |
| 调试精度问题 | "我的 Add 算子精度不达标，帮我排查一下" | Skill: `ascendc-precision-debug` |
| 查阅 API 文档 | "aclnnAdd 接口的参数和返回值是什么" | Skill: `ascendc-docs-search` |
| 检查开发环境 | "帮我检查一下当前的 CANN 开发环境" | Skill: `ascendc-env-check` |
| 代码检视 | "帮我检视这段 Kernel 代码是否符合规范" | Plugin: `ops-code-reviewer` |

更多示例详见 [Skills 使用样例](docs/skills-usage.md)。

> 遇到问题？运行 `install-helper doctor --fix` 自动检测并修复常见问题。

## 📦 安装指南

### 完整安装

#### install-helper 命令参考

| 命令 | 说明 |
|------|------|
| `install-helper` | 交互式安装向导 |
| `install-helper install <name>` | 安装指定插件或 Skill（自动识别） |
| `install-helper install --list` | 按类别列出所有可用 Skills |
| `install-helper update [plugin]` | 更新已安装的插件 |
| `install-helper uninstall <plugin>` | 卸载指定插件 |
| `install-helper list` | 查看可用场景及安装状态 |
| `install-helper doctor --fix` | 健康检查 + 自动修复 |
| `install-helper lang set en_US` | 切换语言 |

> 完整命令参考和详细文档：[install-helper README](plugins-community/install-helper/README.md)

#### 手动执行安装脚本

如果不使用 install-helper，也可以进入对应插件目录手动执行 `init.sh` 安装脚本。以安装 AscendC Kernel 直调插件到 OpenCode 为例：

```bash
cd plugins-official/ops-direct-invoke
bash init.sh project opencode
```

`<tool>` 支持 `opencode` / `claude` / `trae` / `cursor` / `copilot`，各插件的详细安装步骤参见对应插件目录下的 `quickstart.md` 文档。

### 独立 Skill 安装

本仓库的 Skills 遵循 [Agent Skills](https://agentskills.io) 开放标准，可通过开源 [skills CLI](https://github.com/vercel-labs/skills) 安装到 70+ 种 AI 编程工具（OpenCode、Claude Code、Cursor、Codex、Trae 等）。

```bash
# 浏览可用 Skills
npx skills add https://gitcode.com/cann/cannbot-skills.git --list

# 安装单个 Skill（交互式选择目标工具）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-tiling-design

# 安装 Skill 到指定工具（支持 opencode / claude-code / trae / cursor 等）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-env-check --skill npu-arch --agent opencode

# 安装全部 Skill 到所有已检测到的工具（非交互式）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill '*' --agent '*' -y

# 查看已安装的 Skills
npx skills list

# 卸载
npx skills remove ascendc-tiling-design
```

> **此方式仅安装独立 Skills**。如需完整插件内容（Skills + Agents + Workflows + 工具链），请使用 `install-helper` 或 `init.sh` 脚本。

### 安装遇到问题？

运行 `install-helper doctor --fix` 自动检测并修复常见问题。

| 问题 | 解决方法 |
|------|---------|
| `install-helper` 报错 | 确认 Node.js >= 18：`node --version` |
| AI 工具无法识别 Skills | 重启工具或新开会话 |
| 软链接失效 | `install-helper doctor --fix` |
| 网络问题 | 配置 GitCode SSH Key 或设置代理 |
| CANN 环境未配置 | 仅影响代码编译/运行类 Skills，知识检索类不受影响 |

更多故障排查详见各场景对应的 quickstart 文档。

## 🔍 项目架构设计

### 整体架构

```
cannbot-skills/
├── ops/                             # 算子 Skills（正式版）
├── ops-lab/                         # 算子 Skills（实验 / 非正式版）
├── model/                           # 模型推理优化 Skills
├── plugins-official/                # 官方应用 Plugin
│   ├── ops-direct-invoke/           # AscendC Kernel 直调开发
│   ├── ops-registry-invoke/         # AscendC 算子注册调用开发
│   ├── pypto-op-orchestrator/       # PyPTO 算子开发
│   ├── catlass-op-generator/        # Catlass 算子直调开发
│   ├── ops-code-reviewer/           # 代码检视
│   ├── torch-compile/               # torch.compile 图模式
│   ├── model-infer-optimize/        # NPU 推理端到端优化流程
│   ├── triton-op-generator/         # Triton 算子代码生成与优化
│   └── tilelang-op-orchestrator/    # TileLang 算子开发
├── plugins-community/               # 社区 Plugin
│   ├── install-helper/              # CANNBot Install Helper 工具
│   ├── ops-easyasc-dsl/             # EasyASC DSL 算子开发
│   └── ops-qa-suite/                # 算子测试套件
├── infra/                           # 基础设施维护 Skills
└── tests/                           # 自动化测试框架
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
║  │  ops-direct-invoke          │        │  ops-registry-invoke        │      ║
║  │  Kernel 直调开发流程         │        │  算子注册调用开发流程         │      ║
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
║  │  architect   │  │  developer   │  │  reviewer    │  │  tester      │      ║
║  │   方案设计    │  │   代码开发   │   │   代码检视   │  │   代码测试    │      ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
             │ │ │ │       │ │ │ │       │ │ │ │       │ │ │ │
             ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ 知识库类 ──────────────────────────────────────────────────────────────┐   ║
║  │  npu-arch             NPU 架构知识与芯片映射                             │   ║
║  │  tiling-design        Tiling 设计方法论                                  │  ║
║  │  api-best-practices   API 使用最佳实践                                   │  ║
║  │  ops-precision-standard 算子精度标准                                     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 工程模板类 ────────────────────────────────────────────────────────────┐   ║
║  │  registry-invoke-to-direct-invoke  注册算子直调改造模板                  │   ║
║  │  direct-invoke-template            Kernel直调工程模板                   │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 调试与测试类 ──────────────────────────────────────────────────────────┐   ║
║  │  precision-debug      精度调试与症状速查                                 │  ║
║  │  runtime-debug        运行时错误码解析                                   │  ║
║  │  crash-debug          卡死/崩溃调试、Coredump 分析                       │  ║
║  │  env-check            NPU 设备查询与环境验证                             │  ║
║  └────────────────────────────────────────────────────────────────────────┘   ║ 
║                                                                               ║
║  ┌─ 测试开发类 ────────────────────────────────────────────────────────────┐   ║
║  │  st-design            ST 测试用例设计                                   │   ║
║  │  ut-develop           UT 开发与覆盖率增强                               │   ║
║  │  code-review          代码检视规则                                      │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 工具辅助类 ────────────────────────────────────────────────────────────┐   ║
║  │  ops-profiling        算子性能采集分析                                   │  ║
║  │  task-focus           长任务聚焦防迷失                                   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

#### PyPTO 算子开发

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                    ┌─────────────────────────────────┐                       ║
║                    │      pypto-op-orchestrator      │                       ║
║                    │      PyPTO 算子开发流程          │                       ║
║                    └──────┬──────────┬──────────┬────┘                       ║
║                           │          │          │                            ║
╚═══════════════════════════╪══════════╪══════════╪════════════════════════════╝
                            │          │          │
                            ▼          ▼          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║             ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             ║
║             │  analyst     │  │  developer   │  │  perf-tuner  │             ║
║             │  需求与设计   │  │  实现与精度   │  │  性能调优     │             ║
║             └──────────────┘  └──────────────┘  └──────────────┘             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                     │ │ │           │ │ │           │ │ │
                     ▼ ▼ ▼           ▼ ▼ ▼           ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ 需求与设计 ────────────────────────────────────────────────────────────┐   ║
║  │  intent-understand    需求意图理解与规格生成                             │   ║
║  │  api-explore          API 可行性探索与分析                               │   ║
║  │  op-design            算子方案设计生成                                   │   ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 实现与验证 ────────────────────────────────────────────────────────────┐   ║
║  │  golden-generate      Golden 参考实现生成                               │   ║
║  │  op-develop           算子代码实现与调试                                 |   ║
║  │  precision-debug      精度问题诊断                                      │   ║
║  │  precision-compare    精度对比分析                                      │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 性能调优 ──────────────────────────────────────────────────────────────┐   ║
║  │  op-perf-tune         算子性能分析与调优                                 │   ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

#### TileLang 算子开发

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                    ┌─────────────────────────────────┐                       ║
║                    │    tilelang-op-orchestrator     │                       ║
║                    │      TileLang 算子开发流程       │                       ║
║                    └──────┬──────────┬──────────┬────┘                       ║
║                           │          │          │                            ║
╚═══════════════════════════╪══════════╪══════════╪════════════════════════════╝
                            │          │          │
                            ▼          ▼          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║             ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             ║
║             │  analyst     │  │  developer   │  │  perf-tuner  │             ║
║             │  需求与设计   │  │  实现与精度   │  │  性能调优     │             ║
║             └──────────────┘  └──────────────┘  └──────────────┘             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                     │ │ │           │ │ │           │ │ │
                     ▼ ▼ ▼           ▼ ▼ ▼           ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─ 环境与准备 ───────────────────────────────────────────────────────────┐   ║
║  │  env-check               环境检查与配置验证                            │   ║
║  │  submodule-pull          三方库与子模块拉取                            │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 需求与设计 ───────────────────────────────────────────────────────────┐   ║
║  │  op-design               算子设计文档生成                              │   ║
║  │  programming-model-guide 模式选型与配置                                │   ║
║  │  api-best-practices      API 使用最佳实践                              │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 实现与验证 ───────────────────────────────────────────────────────────┐   ║
║  │  op-develop              算子代码实现与测试                            │   ║
║  │  op-test-design          测试设计与覆盖率分析                          │   ║
║  │  review                  代码格式检查与修复                            │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║  ┌─ 性能调优 ─────────────────────────────────────────────────────────────┐   ║
║  │  perf-optimization       性能调优与劣化模式检查                        │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## 🚀 Skills 技能库

### Ascend C 算子开发

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **ascendc-api-best-practices** | API 使用最佳实践、参数限制 | — |
| **npu-arch** | NPU 架构知识、芯片型号映射 | — |
| **ascendc-docs-search** | API 文档索引 + 在线搜索 | — |
| **ascendc-env-check** | NPU 设备查询、CANN 环境验证 | — |
| **ascendc-tiling-design** | Tiling 和 Kernel 设计方法论，按算子类别分类 | — |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 | — |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析（161xxx/361xxx/561xxx） | — |
| **ascendc-crash-debug** | 卡死/崩溃调试，Kernel 挂起、Coredump 分析 | — |
| **ascendc-ut-develop** | UT 单元测试用例开发与覆盖率增强 | — |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0 / L1 测试用例生成 | — |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 | — |
| **ascendc-task-focus** | 任务聚焦，解决长任务“迷失在中间”的问题 | — |
| **ascendc-whitebox-design** | 白盒测试用例设计与生成 | — |
| **ascendc-registry-invoke-template** | 完整自定义算子工程模板，提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考 | — |
| **ascendc-registry-invoke-to-direct-invoke** | 注册调用算子转 `<<<>>>` kernel 直调 | [查看](docs/skills-usage.md#ascendc-registry-invoke-to-direct-invoke) |
| **ascendc-direct-invoke-to-registry-invoke** | `<<<>>>` kernel 直调转注册调用算子 | [查看](docs/skills-usage.md#ascendc-direct-invoke-to-registry-invoke) |
| **ascendc-direct-invoke-template** | Kernel 直调工程模板，提供验证过的样例工程和修改指南 | — |
| **ops-profiling** | NPU 性能采集与分析，CSV 指标解读、瓶颈定位、优化建议 | — |
| **ops-precision-standard** | 算子精度标准，按 dtype 分类提供 atol/rtol 精度比对标准 | — |
| **ascendc-docs-gen** | 算子文档写作参考，支持需求分析、详细设计等多个标准模版 | — |
| **ops-simulator** | NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。 | — |
| **cuda2ascend-simt** | CUDA 算子迁移到 Ascend C SIMT，支持 `standalone sample` / `torch_npu` / `pybind` 三类交付形态，根据原始工程形态自动选择。**仅支持 Ascend 950 PR平台**。当前不支持：native JIT（`nvrtc`、运行时编译、扩展 JIT 加载）、torch 复数 dtype、device 侧 `double`（FP64）、CUDA 生态库（cuBLAS / cuDNN / cuFFT / cuSPARSE / Thrust / CUB / NCCL 等）、协作组、Ascend C SIMD API、矢量编程 API | [查看](docs/skills-usage.md#cuda2ascend-simt) |
| **ascendc-blaze-best-practice** | Matmul/Cube/GEMM/BMM 单算子直调生成（Blaze/tensor_api 路径），覆盖模板选型、改造、Tiling 及排错 | — |
| **ascendc-performance-best-practices** | 按算子族组织的性能优化经验与参考代码总结 | — |
| **ascendc-regbase-best-practice** | DAV_3510 RegBase 算子 API 约束、实现结构、常见陷阱及真实参考算子 | — |
| **cann-env-setup** | 昇腾 NPU CANN 安装与环境配置指导 | — |
| **aiss-tiling-solver** | AISS-TilingSolver 工具自动求解最优 Tiling 参数，覆盖安装、输入构造、运行求解、结果解读 | — |

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

### TileLang 算子开发

| Skill | 功能 |
|-------|------|
| **tilelang-env-check** | TileLang-Ascend 环境检查与配置验证 |
| **tilelang-submodule-pull** | 自动拉取 tilelang 仓库及其三方子模块代码 |
| **tilelang-op-design** | 算子设计文档生成 |
| **tilelang-op-develop** | 基于设计文档生成算子实现代码与测试 |
| **tilelang-op-test-design** | 算子测试设计与测试覆盖率分析 |
| **tilelang-api-best-practices** | TileLang Ascend API 使用最佳实践 |
| **tilelang-programming-model-guide** | Developer/Expert 模式选择与 pass_configs 配置指南 |
| **tilelang-perf-optimization** | 性能调优与性能劣化模式检查 |
| **tilelang-review** | 代码格式检查与自动修复 |

### Triton 算子开发

| Skill | 功能 |
|-------|------|
| **triton-task-extractor** | 从用户输入中提取算子，构建任务文件 |
| **triton-op-designer** | 设计高质量算法，指导代码生成 |
| **triton-op-coding** | 根据设计生成 Triton 内核代码 |
| **triton-op-verifier** | 验证算子精度和性能测试 |
| **triton-latency-optimizer** | 逐步优化 Triton 代码性能 |

### NPU 模型推理优化

| Skill | 功能 |
|-------|------|
| **model-infer-migrator** | 框架适配与部署基线建立 |
| **model-infer-parallel-analysis** | 并行策略分析（TP/EP/DP） |
| **model-infer-parallel-impl** | 并行切分实施 |
| **model-infer-kvcache** | KVCache 优化 + FA 替换 |
| **model-infer-fusion** | torch_npu 融合算子分析与替换 |
| **model-infer-quantization** | compressed-tensors 量化适配改造 |
| **model-infer-graph-mode** | torch.compile 图模式适配 |
| **model-infer-precision-debug** | NPU 推理精度诊断 |
| **model-infer-runtime-debug** | NPU 运行时错误诊断 |
| **model-infer-multi-stream** | 多流并行优化 |
| **model-infer-prefetch** | 权重预取适配 |
| **model-infer-superkernel** | SuperKernel 适配 |

### Skill 治理工具

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **cannbot-skill-reviewer** | 审查新增或修改的 `SKILL.md` 是否符合 CANNBot 入库要求，输出自动门禁、九维评分、阻塞项和整改建议 | [查看](docs/skills-usage.md#cannbot-skill-reviewer) |

### GitCode 协作工具

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **gitcode-pr-handler** | 根据 GitCode PR 代码变更重新生成标题（约定式提交）与描述（沿用仓库 PR 模板）并写回 PR | [查看](docs/skills-usage.md#gitcode-pr-handler) |
| **gitcode-issue-gen** | 自动判断两条路径：(PR路径) 从 PR diff 生成关联 Issue 并完成双向关联；(手动路径) 交互式收集信息生成 Issue 草稿，经确认后提交 | [查看](docs/skills-usage.md#gitcode-issue-gen) |
| **gitcode-issue-handler** | GitCode Issue 端到端处置，按内容自动选择 PR 代码变更路径或 Comment 答复路径 | [查看](docs/skills-usage.md#gitcode-issue-handler) |
| **gitcode-toolkit** | GitCode API/Token/URL/日志/变更展示 + Git 克隆/diff/log/remote + PR 创建工作流共享参考（内部参考，不直接触发） | — |

## 🚀 Agents 智能代理

### Ascend C 算子开发

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-developer** | 算子开发者，支持代码实现、编译测试和精度验证 |
| **ascendc-ops-tester** | 算子测试者，支持ST/UT用例生成与执行 |
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

### TileLang 算子开发

| Agent | 功能 |
|-------|------|
| **tilelang-op-analyst** | 需求理解与算子设计 |
| **tilelang-op-developer** | 代码生成、测试与精度调试 |
| **tilelang-op-perf-tuner** | 性能分析、瓶颈定位与调优 |

### Triton 算子开发

| Agent | 功能 |
|-------|------|
| **triton-op-generator** | Triton 算子端到端生成与优化 |

### NPU 模型推理优化

| Agent | 功能 |
|-------|------|
| **model-infer-analyzer** | 模型分析、方案设计、并行策略推荐 |
| **model-infer-implementer** | 代码改造、调试修复 |
| **model-infer-reviewer** | 精度验证、性能对比 |

## 🛠️ 测试框架

自动化测试验证 Skills 和 Agents 的正确性，确保技能模块和智能代理的行为符合预期。
详见 [tests/README.md](tests/README.md)。

## 💬 相关信息
- [贡献指南、开发规范](docs/STANDARDS.md)
- [许可证](LICENSE)
- [所属 SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/cannbot)

## 💖 免责声明

感谢您关注 CANNBot Skills 项目，我们希望这些技能和知识能帮助您更好地进行 CANN 开发 ^_^

在使用之前，请您了解：

1. **关于功能满足度**：由于技术快速更新迭代，部分内容可能无法完全适用于所有场景。本开源社区的功能和文档正在持续更新和完善中，如果想提出需求、发现问题、贡献想法，欢迎提 Issue 或参与讨论，共创共建。

2. **关于自动生成**：自动代码生成工具所产出的内容，其完整性、准确性、合规性受模型、Skills 能力、语料质量、输入指令等多种因素影响，无法保证完全精准。所有生成代码作为辅助研发使用，请开发者务必进行测试验证、安全审查后再投入使用。

## 🔥 最新动态

- **2026-06-25** — 新增 `install-helper` 交互式安装助手 CLI 工具；README 安装文档全面优化，支持双轨制安装。
- **2026-06-24** — `ops-simulator` 新增基于 summary.json 的性能瓶颈分析能力；`ops-direct-invoke` 拆分 Design Reviewer Agent。
- **2026-06-23** — `ops-registry-invoke` 集成 infra Skills；补充 CV 融合算子流水间同步知识。
- **2026-06-22** — 新增 `model-infer-quantization` 量化 Skill，主链 Skill 按 cann-recipes-infer 框架重写。
- **2026-06-18** — 为 graph/ 和 infra/ 共 10 个 Skill 添加 ST 测试用例；改进 triton GPU kernel 迁移策略。
- **2026-06-17** — 集成 infra Skills 到 `ops-direct-invoke` 和 `ops-direct-invoke-flash`；新增 MM/GMM Skill。
- **2026-06-16** — 新增 TileLang 插件，为 24 个 ops Skill 添加 ST 测试用例。
- **2026-06-15** — SWAT 算法更新策略并适配量化，新增参数分析能力。
- **2026-06-13** — `ops-direct-invoke-flash` 补充 init.sh 安装脚本与 README 文档。
- **2026-06-12** — `blaze` Skill 支持后融合能力。
- **2026-06-11** — 新增算子评测框架到 `tests/benchmark`。

更多历史记录详见 [CHANGELOG.md](CHANGELOG.md)。

## 🤝 社区交流

- **Issue 反馈**：[提交 Issue](https://gitcode.com/cann/cannbot-skills/issues)
- **社区讨论**：[参与讨论](https://gitcode.com/cann/cannbot-skills/discussions)
- **微信交流**：[加入群聊](https://gitcode.com/cann/cannbot-skills/discussions/2)
