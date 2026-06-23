# CANNBot TileLang 算子开发快速入门指南

## 概述

CANNBot TileLang 算子开发模式适用于通过 **TileLang-Ascend** 框架开发自定义算子。基于 TVM 编译器基础设施，使用 Python DSL + `@tilelang.jit` 编写 AI 计算 kernel，支持 Developer 模式（自动化）和 Expert 模式（手动控制）两种编程范式。

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（≥ 8.3），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装 PyTorch（≥ 2.6.0）和 torch_npu（≥ 2.6.0）
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### 操作步骤

#### 方式一：项目级安装（推荐）

在项目目录下安装，配置仅对当前项目生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入 TileLang 算子开发目录
cd cannbot-skills/plugins-official/tilelang-op-orchestrator

# 3. 执行初始化脚本（项目级）
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
bash init.sh project trae       # TRAE 用户
bash init.sh project cursor     # Cursor 用户
bash init.sh project copilot    # Copilot 用户

# 4. 进入 TileLang-Ascend 源码仓库，安装环境
cd tilelang-ascend
bash install_ascend.sh
cd ..
```

#### 方式二：全局安装

在用户目录下安装，配置全局生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入 TileLang 算子开发目录
cd cannbot-skills/plugins-official/tilelang-op-orchestrator

# 3. 执行初始化脚本（全局）
bash init.sh global opencode    # OpenCode 用户（默认）
bash init.sh global claude      # Claude Code 用户
bash init.sh global trae        # TRAE 用户
bash init.sh global cursor      # Cursor 用户
bash init.sh global copilot     # Copilot 用户

# 4. 进入 TileLang-Ascend 源码仓库，安装环境
cd tilelang-ascend
bash install_ascend.sh
cd ..
```

### 安装内容

init.sh 脚本会完成以下操作：

| 内容 | OpenCode 项目级 | OpenCode 全局 | Claude 项目级 | Claude 全局 | TRAE 项目级 |
|------|----------------|---------------|---------------|-------------|------------|
| Skills 技能模块 | `.opencode/skills/` | `~/.config/opencode/skills/` | `.claude/skills/` | `~/.claude/skills/` | `.trae/skills/` |
| Agents 子代理 | `.opencode/agents/` | `~/.config/opencode/agents/` | `.claude/agents/` | `~/.claude/agents/` | `.trae/agents/` |
| AGENTS.md | `.opencode/AGENTS.md` | `~/.config/opencode/AGENTS.md` | `.claude/CLAUDE.md` | `~/.claude/CLAUDE.md` | `.trae/AGENTS.md` |

#### Cursor 安装路径

| 内容 | Cursor 项目级 | Cursor 全局级 |
|------|--------------|--------------|
| Skills 技能模块 | `.cursor/skills/` | `~/.cursor/skills/` |
| Agents 子代理 | `.cursor/agents/` | `~/.cursor/agents/` |
| AGENTS.md | `.cursor/AGENTS.md` | `~/.cursor/AGENTS.md` |

#### Copilot 安装路径

| 内容 | Copilot 项目级 | Copilot 全局级 |
|------|---------------|---------------|
| Skills 技能模块 | `.github/skills/` | `~/.copilot/skills/` |
| Agents 子代理 | `.github/agents/` | `~/.copilot/agents/` |
| AGENTS.md | `.github/AGENTS.md` | `~/.copilot/AGENTS.md` |

### 环境校验

执行完上述步骤后，检查目录结构是否符合以下规范：

**项目级安装**：
```
cannbot-skills/plugins-official/tilelang-op-orchestrator/
├── .opencode/
│   ├── skills/                         # 技能模块（9 个）
│   │   ├── tilelang-env-check/
│   │   ├── tilelang-submodule-pull/
│   │   ├── tilelang-op-design/
│   │   ├── tilelang-op-develop/
│   │   ├── tilelang-op-test-design/
│   │   ├── tilelang-perf-optimization/
│   │   ├── tilelang-api-best-practices/
│   │   ├── tilelang-programming-model-guide/
│   │   └── tilelang-review/
│   ├── agents/                         # 3 个子代理（analyst / developer / perf-tuner）
│   ├── AGENTS.md                       # 编排器（Primary）配置
│   └── cannbot-manifest.json           # 安装清单
├── tilelang-ascend                     # tilelang代码仓
├── init.sh                             # 初始化脚本
└── quickstart.md                       # 本文档
```

## 二、快速上手

### 启动

在初始化完成的目录下执行：

```bash
opencode    # OpenCode 用户
```

### 开发算子示例

在交互界面中输入算子开发需求，CANNBot 会按照“算子方案设计-->算子代码实现-->算子精度验证”分阶段开发流程引导你完成：

```
帮我开发一个 softmax 算子方案设计
```

### 核心工作流

采用 3 阶段状态机编排，由 orchestrator 统一调度，确保算子开发质量：

```
Stage 1 算子设计（含需求理解） → Stage 2 代码实现 + 测试 + 精度调试（一站式） → Stage 3 性能调优（可选）
```

每阶段通过工件门禁校验后才进入下一阶段；Stage 2 内部完成"生成代码 → 跑测试 → 精度调试"全部循环，精度通过后才询问是否进入 Stage 3。支持断点续跑、失败恢复与设计回退（Subagent 返回 `[DESIGN_ERROR]` 时回退到 Stage 1 重做设计），详见 AGENTS.md。

### 产出物示例

TileLang 算子开发模式下，CANNBot 会在 `examples/{operator}/` 目录下生成文件：

```
examples/softmax/
├── DESIGN.md                   # Stage 1 设计文档
├── example_softmax.py          # Stage 2 kernel + 内嵌 golden + test 用例 + main 块
├── README.md                   # 实现说明（可选）
├── perf_tuning/                # Stage 3 性能调优产物（可选）
├── history_version/            # 设计回退 / 精度调试备份
└── .orchestrator_state.json    # 流程状态（自动维护，支持断点续跑）
```

## 三、可用技能与代理

| Skill | 用途 | 触发时机 |
|-------|------|---------|
| `tilelang-env-check` | 环境检查与自动修复（子模块 / 编译 / 环境变量） | Stage 1 启动前环境预检 |
| `tilelang-submodule-pull` | 拉取代码与子模块 | env-check 发现子模块缺失时 |
| `tilelang-op-design` | 算子方案设计，生成 DESIGN.md | Stage 1 |
| `tilelang-op-develop` | 基于 DESIGN.md 生成算子代码与测试 | Stage 2 |
| `tilelang-op-test-design` | 测试用例与精度标准设计 | Stage 2 辅助 |
| `tilelang-perf-optimization` | 性能瓶颈分析与优化 | Stage 3 |
| `tilelang-api-best-practices` | API 速查表与最佳实践 | 编写 kernel 查阅 API 时 |
| `tilelang-programming-model-guide` | Developer/Expert 模式对照与转换 | 选择编程模式时 |
| `tilelang-review` | 代码审查（Python + C++） | 代码 review 时 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `tilelang-op-orchestrator` | 流程编排、状态机、工件门禁、设计回退（Primary） | 全流程 |
| `tilelang-op-analyst` | 算子设计（含需求理解、设计回退） | Stage 1 |
| `tilelang-op-developer` | 代码实现 + 测试 + 精度调试（一站式） | Stage 2 |
| `tilelang-op-perf-tuner` | 性能分析与调优 | Stage 3 |

## 四、Developer 模式 vs Expert 模式

TileLang-Ascend 支持两种编程范式，开发前需先确认使用哪种模式：

| 维度 | Developer（自动化） | Expert（手动控制） |
|------|-------------------|-------------------|
| 内存分配 | `T.alloc_shared/fragment` 编译器自动映射 | `T.alloc_L1/ub/L0A/L0B/L0C` 显式指定 |
| 计算 | `T.Parallel` + 符号运算 | `T.tile.add/exp/max` 等 |
| 作用域 | 编译器自动分离 Cube/Vector | 显式 `with T.Scope("C"/"V")` |
| 同步 | 自动 | 手动 `T.barrier_all/set_flag/wait_flag` |
| 适合场景 | 快速开发、原型验证 | 需要精细控制性能 |

详细对照参见 `tilelang-programming-model-guide` skill。

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新技能模块？

重新执行 init.sh 即可，脚本会自动覆盖旧版本。

### Q: 如何选择 Developer 模式还是 Expert 模式？

| 场景 | 推荐模式 |
|------|---------|
| 快速验证算子可行性 | Developer 模式 |
| 原型开发和概念验证 | Developer 模式 |
| 需要精细控制硬件资源和内存层级 | Expert 模式 |
| 生产级高性能算子调优 | Expert 模式 |
| 混合使用（如 Cube 用 Developer，Vector 用 Expert） | 混合模式 |

---

## 总结

1. TileLang 算子开发模式通过 Python DSL 实现昇腾 NPU 算子的快速开发
2. 环境搭建核心两步：克隆仓库 → 执行 init.sh
3. `opencode` / `claude` 是核心交互指令
4. 开发前必须确认使用 Developer / Expert / 混合模式
