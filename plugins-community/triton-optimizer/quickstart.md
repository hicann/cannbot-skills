# Triton Optimizer 快速入门

## 概述

`triton-optimizer` 是 Triton Ascend NPU 算子 workflow 插件。它提供优化与转换两个 Subagent、2 个下沉到 `ops/` 的 Skills，以及用于 Claude/Cursor 的 workflow 状态 hooks。

它适合两类任务：

- 优化已有 Triton Ascend NPU 算子，按 baseline/round 迭代记录。
- 将 PyTorch 算子转换为 PyTorch-facing、Triton Ascend NPU-backed 的实现，并完成验证。

## 一、安装

### 方式一：install-helper

```bash
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash
cd /path/to/your/triton/project
install-helper install triton-optimizer --tool claude --level project
```

可用别名：

```bash
install-helper install triton-optimizer --tool claude
install-helper install triton-agent --tool claude
install-helper install triton-npu-optimizer --tool claude
install-helper install triton-optimize --tool claude
```

### 方式二：init.sh

`init.sh` 支持从任意目录调用，项目级安装会把配置写入你传入的项目目录。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
INIT=$(pwd)/cannbot-skills/plugins-official/triton-optimizer/init.sh

bash "$INIT" project claude /path/to/your/triton/project
bash "$INIT" global claude
```

也可以安装到其它 CANNBot 支持的 AI 编程工具：

```bash
bash "$INIT" project opencode /path/to/project
bash "$INIT" project trae /path/to/project
bash "$INIT" project cursor /path/to/project
bash "$INIT" project copilot /path/to/project
```

### 方式三：Claude Plugin Marketplace

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install triton-optimizer@cannbot
/reload-plugins
```

Marketplace 方式由 Claude Code 原生加载插件 manifest、agent、skills 依赖和 hooks。

## 二、安装内容

| 内容 | 来源 | 安装位置 |
| --- | --- | --- |
| Subagent | `plugins-official/triton-optimizer/agents/` | `.claude/agents/` 等工具配置目录 |
| Skills | `ops/triton-npu-*` | `.claude/skills/` 等工具配置目录 |
| Hooks | `plugins-official/triton-optimizer/hooks/` | `.claude/hooks/` 等工具配置目录 |
| 配置入口 | `plugins-official/triton-optimizer/AGENTS.md` | 项目根 `CLAUDE.md` / `AGENTS.md` |

## 三、使用

1. 在目标 Triton Ascend NPU 算子项目中启动 AI 编程工具
2. 优化任务：

```text
请使用 triton-npu-optimize 优化当前目录的 Triton 算子，并从 baseline 开始记录每轮结果。
```

3. 转换任务：

```text
请使用 triton-npu-convert 将 /path/to/op.py 转成 Triton Ascend NPU 实现，输出到 /path/to/triton_op.py 并验证。
```

## 四、验证安装

```bash
# 查看 manifest
cat .claude/cannbot-manifest.json

# 查看 skills/subagent/hooks
ls .claude/skills | grep -E 'triton-npu'
ls .claude/agents
ls .claude/hooks
```

Claude Marketplace 方式可使用：

```text
/plugin list
```

## 五、更新

```bash
cd cannbot-skills
git pull
bash plugins-official/triton-optimizer/init.sh project claude /path/to/your/triton/project
```

或在 Claude Marketplace 中：

```text
/plugin update triton-optimizer@cannbot
```
