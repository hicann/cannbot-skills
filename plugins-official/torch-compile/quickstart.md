# CANNBot torch-compile 快速入门

## 概述

`torch-compile` 是 PyTorch `torch.compile` 图模式编排入口，用 `agents/` 下的 Subagent 组织不同图模式能力。当前已提供 `torch-npugraph-ex` Subagent，面向昇腾 NPU `npugraph_ex` / `aclgraph` 模式，覆盖 torch.compile + TorchAir 的配置指导、脚本迁移、自定义算子入图、编译错误和运行时错误诊断。

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

> `init.sh` 通过脚本自身路径定位插件根，**从任意目录调用均可**，下文统一用 `INIT=...` 变量指代脚本路径，无需 cd 进 `torch-compile/`。
>
> ```bash
> # 仅需 clone 一次；后续所有工具复用同一份脚本
> git clone https://gitcode.com/cann/cannbot-skills.git
> INIT=$(pwd)/cannbot-skills/plugins-official/torch-compile/init.sh
> ```

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install torch-compile@cannbot
/reload-plugins
```

安装后新开会话，或在当前会话执行 `/clear` 触发 `SessionStart`。`torch-compile` 是主对话入口，会把 `AGENTS.md` 注入当前 Claude 上下文，并通过 `agents/torch-npugraph-ex.md` 处理 npugraph_ex 专项工作。

**备选：init.sh 脚本**

```bash
bash "$INIT" project claude     # 项目级
bash "$INIT" global claude      # 全局级
```

### OpenCode

```bash
bash "$INIT" project opencode   # 项目级（默认）
bash "$INIT" global opencode    # 全局级
```

项目级会在**当前工作目录**生成 `.opencode/`，全局级落在 `~/.config/opencode/`，其中以软链方式注入 `skills/` 与 `agents/`，重启 OpenCode 后即可看到 `torch-npugraph-ex` Subagent。

### TRAE

仅支持项目级安装。

```bash
bash "$INIT" project trae
```

生成 `.trae/` 目录，结构与 Claude/OpenCode 基本一致。

### Cursor

```bash
bash "$INIT" project cursor     # 项目级
bash "$INIT" global cursor      # 全局级
```

生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

### Codex

```bash
bash "$INIT" project codex      # 项目级
bash "$INIT" global codex       # 全局级
```

生成 `.codex/`（或全局 `~/.codex/`），结构：`skills/ agents/ AGENTS.md cannbot-manifest.json`。

### Copilot

```bash
bash "$INIT" project copilot    # 项目级
bash "$INIT" global copilot     # 全局级
```

生成 `.github/`（项目级）或 `~/.copilot/`（全局级），结构与 Codex 一致。

> **关于"项目级"的位置说明**：项目级安装的 `.opencode/` `.claude/` 等目录落在执行 `init.sh` 时的**当前工作目录**下，请在希望生效的项目根目录调用脚本；如需全局生效，请使用 `global` 模式。

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 torch-compile@cannbot ✔ enabled

# OpenCode
opencode agent list
# 应看到 torch-npugraph-ex

# TRAE / Cursor / Codex / Copilot
ls .trae/ .cursor/ .codex/ .copilot/ 2>/dev/null
# 应看到 skills/ agents/ AGENTS.md cannbot-manifest.json
```

## 二、使用示例

```text
我想用 torch.compile 在昇腾 NPU 上加速推理，应该怎么配置 npugraph_ex？
```

```text
我的模型使用 npugraph_ex 编译失败了，帮我分析这段报错日志。
```

```text
怎么把自定义算子加入 npugraph_ex 图编译？
```

## 三、可用 Agents

| Agent | 用途 |
|------|------|
| `torch-compile` | torch.compile 图模式 primary 编排入口 |
| `torch-npugraph-ex` | npugraph_ex / aclgraph 模式专项 Subagent |

## 四、可用 Skills

| Skill | 用途 |
|------|------|
| `torch-npugraph-ex-knowledge` | npugraph_ex 基础知识与配置说明 |
| `torch-npugraph-ex-template` | npugraph_ex 代码模板 |
| `torch-npugraph-ex-dfx-triage` | 问题定位分诊 |
| `torch-npugraph-ex-compile-error-diagnosis` | 编译错误诊断 |
| `torch-npugraph-ex-runtime-error-diagnosis` | 运行时错误诊断 |
| `torch-custom-ops-guide` | 自定义算子入图指导 |

## 五、常见问题

### Q: 如何查看 init.sh 帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一环境，全局生效

### Q: 如何更新？

```bash
# Claude Code（marketplace 方式）
/plugin update torch-compile@cannbot

# 其它工具（init.sh 方式，从 cannbot-skills clone 拉最新代码后复用 $INIT 重跑即可）
cd cannbot-skills && git pull
bash cannbot-skills/plugins-official/torch-compile/init.sh project <tool>
```
