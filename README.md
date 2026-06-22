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

## 🔥 最新动态
- **2026-05-25** — 新增 `cuda2ascend-simt` 实验技能。
- **2026-05-23** — 性能调试/代码检视/PyPTO 多模块能力增强；测试框架修复跨平台稳定性并统一 License。
- **2026-05-22** — 新增 tiling-solver Skill 与社区治理模型；UT 与 CI/ST 测试框架能力增强。
- **2026-05-20** — 新增 4 个 GitCode 协作 Skills 与 skill 能力看护 CI 入口。
- **2026-05-19** — 新增 Triton 算子生成功能。
- **2026-05-16** — model-infer-optimize 迁移至 plugins-official；全场景插件支持 Trae 全局安装。
- **2026-05-15** — RegBase 最佳实践集成至算子直调工作流；init.sh 支持任意目录安装。
- **2026-05-14** — Skill `ascendc-npu-arch` 重命名为 `npu-arch`。
- **2026-05-12** — torch-compile 加入 plugin-official；TileLang 迁移至 plugins-community。
- **2026-05-11** — 新增 ascendc-crash-debug 技能；官方插件新增支持 Cursor IDE。

> 仅展示最近两周动态，更多历史记录详见 [CHANGELOG.md](CHANGELOG.md)。

## ⚡️快速开始

### 前置条件

安装以下任意一个 AI 编程工具：

| 工具 | 安装命令 | 适用安装方式 |
|------|---------|-------------|
| **Claude Code** | `npm install -g @anthropic-ai/claude-code` → [官方文档](https://code.claude.com/docs/zh-CN/overview) | 脚本安装 / Plugin 安装 |
| **OpenCode** | `npm install -g opencode-ai` → [官方文档](https://opencode.ai/docs/zh-cn) | 脚本安装 |
| **Trae** | 下载安装：https://www.trae.cn → [官方文档](https://www.trae.cn) | 脚本安装 |
| **Cursor** | 下载安装：https://cursor.com → [官方文档](https://cursor.com/cn/docs/get-started/quickstart) | 脚本安装 |


### 步骤一：克隆仓库

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills
```

### 步骤二：选择场景并安装

选择对应场景，将 `<tool>` 替换为你的 AI 工具（支持 `opencode` / `claude` / `trae` / `cursor` / `copilot`）后执行：

| 场景 | 安装命令 | 详细文档 |
|------|---------|---------|
| [**AscendC Kernel<<<>>>直调**](plugins-official/ops-direct-invoke/quickstart.md) | `cd plugins-official/ops-direct-invoke && bash init.sh project <tool>` | [quickstart](plugins-official/ops-direct-invoke/quickstart.md) |
| [**AscendC 算子注册调用**](plugins-official/ops-registry-invoke/quickstart.md) | `cd plugins-official/ops-registry-invoke && bash init.sh project <tool>` | [quickstart](plugins-official/ops-registry-invoke/quickstart.md) |
| [**PyPTO 算子**](plugins-official/pypto-op-orchestrator/quickstart.md) | `cd plugins-official/pypto-op-orchestrator && bash init.sh project <tool>` | [quickstart](plugins-official/pypto-op-orchestrator/quickstart.md) |
| [**Triton 算子生成**](plugins-official/triton-op-generator/quickstart.md) | `cd plugins-official/triton-op-generator && bash install.sh project <tool>` | [quickstart](plugins-official/triton-op-generator/quickstart.md) |
| [**TileLang 算子**](plugins-official/tilelang-op-orchestrator/quickstart.md) | `cd plugins-official/tilelang-op-orchestrator && bash init.sh project <tool>` | [quickstart](plugins-official/tilelang-op-orchestrator/quickstart.md) |
| [**NPU 推理优化**](plugins-official/model-infer-optimize/quickstart.md) | `cd plugins-official/model-infer-optimize && bash init.sh project <tool>` | [quickstart](plugins-official/model-infer-optimize/quickstart.md) |
| [**Catlass 算子直调开发**](plugins-official/catlass-op-generator/quickstart.md) | `cd plugins-official/catlass-op-generator && bash init.sh project <tool>` | [quickstart](plugins-official/catlass-op-generator/quickstart.md) |
| [**代码检视**](plugins-official/ops-code-reviewer/quickstart.md) | `cd plugins-official/ops-code-reviewer && bash init.sh project <tool>` | [quickstart](plugins-official/ops-code-reviewer/quickstart.md) |
| [**torch.compile 图模式**](plugins-official/torch-compile/quickstart.md) | Plugin 市场安装：`/plugin install torch-compile@cannbot` | [quickstart](plugins-official/torch-compile/quickstart.md) |

**示例**：如果你使用 Claude Code，想安装 AscendC Kernel 直调场景：
```bash
cd plugins-official/ops-direct-invoke && bash init.sh project claude
```

安装脚本会自动完成：创建软链接 → 生成配置文件 → 克隆依赖仓库 → 健康检查。看到 `Installation complete!` 即表示安装成功。

> **Claude Code 用户的备选方案**：如果你使用 Claude Code，也可以用 Plugin 方式安装（`/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git`，然后 `/plugin install <插件名>@cannbot`）。

### 步骤三：验证安装

安装完成后，检查以下内容确认安装成功：

```bash
# 检查 skills 和 agents 目录是否正确链接
ls .claude/skills/        # Claude Code 用户
ls .opencode/skills/      # OpenCode 用户
ls .trae/skills/          # Trae 用户
ls .cursor/skills/        # Cursor 用户
ls .github/skills/        # VS Code Copilot 用户

# 如果上述目录存在且包含多个子目录（如 npu-arch、ascendc-env-check 等），说明安装成功。
```

你也可以启动 AI 编程工具后输入以下内容来快速验证 Skills 是否被正确加载：

> "请列出当前可用的 CANNBot Skills"

如果 AI 能列出 `npu-arch`、`ascendc-env-check` 等技能名称，说明安装完全成功。

### 步骤四：开始使用

启动 AI 工具，直接描述开发需求即可。

以下是一些入门提示词示例：

| 你想做什么 | 可以这样说 |
|-----------|-----------|
| 开发一个算子 | "帮我开发一个 Abs 算子，输入 float16，输出 float16" |
| 调试精度问题 | "我的 Add 算子精度不达标，帮我排查一下" |
| 查阅 API 文档 | "aclnnAdd 接口的参数和返回值是什么" |
| 检查开发环境 | "帮我检查一下当前的 CANN 开发环境" |
| 代码检视 | "帮我检视这段 Kernel 代码是否符合规范" |

更多示例详见 [Skills 使用样例](docs/skills-usage.md)。各场景的完整使用步骤参见对应 quickstart 文档。

### 安装遇到问题？

| 常见问题 | 解决方法 |
|---------|---------|
| `git clone` 失败（网络问题） | 尝试配置 GitCode SSH Key，或使用镜像地址 |
| `init.sh` 提示权限不足 | 执行 `chmod +x plugins-official/*/init.sh` |
| skills 目录为空 | 确认安装脚本输出中无报错，重新执行安装命令 |
| AI 工具无法识别 Skills | 重启工具或新开会话（某些工具需重新加载配置） |
| CANN 环境未配置 | 仅影响代码编译/运行类 Skills，知识检索类不受影响 |

更多安装选项（全局安装、Plugin 市场安装、手动安装）和故障排查详见各场景对应的 quickstart 文档。

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
│   └── ops-easyasc-dsl/             # EasyASC DSL 算子开发
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

## 🤝 社区交流

- **Issue 反馈**：[提交 Issue](https://gitcode.com/cann/cannbot-skills/issues)
- **社区讨论**：[参与讨论](https://gitcode.com/cann/cannbot-skills/discussions)
- **微信交流**：[加入群聊](https://gitcode.com/cann/cannbot-skills/discussions/2)
