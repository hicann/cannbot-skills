# NPU 模型推理优化快速入门

## 概述

`model-infer-optimize` 是 NPU 模型推理端到端优化 plugin。它通过 `workflows/optimize-workflow.md` 编排 `model-infer-analyzer`、`model-infer-implementer`、`model-infer-reviewer` 三类 Subagent，覆盖并行策略、KVCache/FA、融合算子、量化适配、图模式、多流并行、预取和 SuperKernel 等优化路径。

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```text
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install model-infer-optimize@cannbot
/reload-plugins
```

安装后新开会话，或在当前会话执行 `/clear` 触发插件上下文加载。`model-infer-optimize` 是主对话入口，会把 AGENTS.md 注入上下文，并按强制规则读取 `workflows/optimize-workflow.md`。

验证：

```bash
claude plugin list
# 应看到 model-infer-optimize@cannbot ✔ enabled
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-optimize
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-optimize
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

验证：

```bash
opencode agent list
# 应看到 model-infer-analyzer / model-infer-implementer / model-infer-reviewer
```

### TRAE

仅支持项目级安装。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-optimize
bash init.sh project trae
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

### Cursor

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-optimize
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

### Copilot

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-optimize
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 model-infer-optimize@cannbot ✔ enabled

# OpenCode
opencode agent list
# 应看到 model-infer-analyzer / model-infer-implementer / model-infer-reviewer

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/ agents/ workflows/ AGENTS.md cannbot-manifest.json

# Cursor
ls .cursor/
# 应看到 skills/ agents/ workflows/ AGENTS.md cannbot-manifest.json
```

## 二、快速上手

### 启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

> **TRAE 用户**：TRAE 通过 IDE、VS Code 插件或 CLI 启动。init.sh 会自动检测 TRAE IDE（`~/.trae-cn`）、Plugin（`~/.marscode`）或 CLI（`~/.traecli`）并安装到对应目录。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。

### 模型优化示例

在目标 `cann-recipes-infer` 或模型仓中提出需求：

```text
帮我优化 deepseek-r1 模型的 NPU 推理性能
```

primary agent 会按 AGENTS.md 中的强制规则自动读取 `workflows/optimize-workflow.md` 并按阶段推进。

## 三、安装内容

| 内容 | 说明 |
| --- | --- |
| 原子 skills（12 个） | 来自 `model/model-infer-*`，覆盖推理优化各专项能力 |
| workflow 文档 | `plugins-official/model-infer-optimize/workflows/optimize-workflow.md` |
| Subagents | `plugins-official/model-infer-optimize/agents/model-infer-*.md` |
| hooks | 角色越界保护、progress.md 读取约束、自验证检查和长任务提醒 |
| 配置入口 | `AGENTS.md` / `CLAUDE.md`，强制读取 `workflows/optimize-workflow.md` |

## 四、核心工作流

```text
阶段 0：模型分析 + 性能基线
    ↓
阶段 1：并行化改造
    ↓
阶段 2：KVCache 静态化 + FA 算子替换
    ↓
阶段 3：融合算子优化
    ↓
阶段 4：量化适配（可选，用户提供 compressed-tensors 量化产物或明确要求量化时）
    ↓
阶段 5：图模式适配
    ↓
阶段 6：优化总结
```

每个阶段遵循：分析 → 方案确认 → 实施 → 验证 → 阶段总结。

## 五、可用技能（原子 skills）

| Skill | 用途 |
| --- | --- |
| `model-infer-migrator` | 框架适配与基线建立 |
| `model-infer-parallel-analysis` | 并行策略分析 |
| `model-infer-parallel-impl` | 并行切分实施 |
| `model-infer-kvcache` | KVCache + FA 优化 |
| `model-infer-fusion` | 融合算子分析与替换 |
| `model-infer-quantization` | compressed-tensors 量化适配、验证和收益评估 |
| `model-infer-graph-mode` | 图模式适配 |
| `model-infer-precision-debug` | NPU 推理精度诊断 |
| `model-infer-runtime-debug` | NPU 运行时错误诊断 |
| `model-infer-multi-stream` | 多流并行优化 |
| `model-infer-prefetch` | 权重预取 |
| `model-infer-superkernel` | SuperKernel 适配 |

端到端优化流程由 `workflows/optimize-workflow.md` 承载，由 primary agent 自动加载，不作为可独立调用的 skill 暴露。

## 六、可用 Agents

| Agent | 职责 |
| --- | --- |
| `model-infer-analyzer` | 模型分析、方案设计、并行策略推荐 |
| `model-infer-implementer` | 代码改造、调试修复、自验证 |
| `model-infer-reviewer` | 精度验证、性能对比、结构化诊断 |

## 七、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
# Claude Code
/plugin update model-infer-optimize@cannbot

# OpenCode (init.sh 方式)
cd cannbot-skills/plugins-official/model-infer-optimize && bash init.sh

# TRAE
cd cannbot-skills/plugins-official/model-infer-optimize && bash init.sh project trae

# Cursor
cd cannbot-skills/plugins-official/model-infer-optimize && bash init.sh project cursor
```

### Q: 端到端优化和单点优化如何选择？

| 场景 | 推荐方式 |
|------|---------|
| 模型从适配到性能达标的完整链路 | 端到端 plugin（`帮我优化 XX 模型的 NPU 推理性能`）|
| 已部署模型，仅需做 KVCache / FA 替换 | 直接调用 `model-infer-kvcache` skill |
| 已部署模型，仅需做并行策略分析或实施 | 调用 `model-infer-parallel-analysis` / `model-infer-parallel-impl` skill |
| 已部署模型，仅需做融合算子替换 | 直接调用 `model-infer-fusion` skill |
| 已部署模型，仅需接入 compressed-tensors 量化产物 | 直接调用 `model-infer-quantization` skill |
| 已部署模型，仅需做图模式适配 | 直接调用 `model-infer-graph-mode` skill |
| 已部署模型，仅需诊断精度或运行时错误 | 直接调用 `model-infer-precision-debug` / `model-infer-runtime-debug` skill |

> 单点 skill 由 Claude 通过描述匹配自动激活，不会触发 6 阶段端到端工作流。

---

## 总结

1. 端到端优化通过 `workflows/optimize-workflow.md` 编排 6 阶段流程，并在需要时插入可选量化阶段
2. Claude Code 用户用 `/plugin install` 一键安装，OpenCode/TRAE/Cursor 用户用 `init.sh` 脚本安装
3. `claude` / `opencode` 是核心交互指令；IDE 类工具（TRAE / Cursor）打开项目即自动加载
4. 单点优化（KVCache、并行、融合算子、量化等）由 12 个原子 skill 自动激活，不进入端到端流程
5. 所有阶段通过门禁驱动，支持断点续跑与失败恢复
