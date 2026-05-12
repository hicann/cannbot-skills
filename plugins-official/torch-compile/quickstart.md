# CANNBot torch-compile 快速入门

## 概述

`torch-compile` 是 PyTorch `torch.compile` 图模式编排入口，用 `agents/` 下的 Subagent 组织不同图模式能力。当前已提供 `torch-npugraph-ex` Subagent，面向昇腾 NPU `npugraph_ex` / `aclgraph` 模式，覆盖 torch.compile + TorchAir 的配置指导、脚本迁移、自定义算子入图、编译错误和运行时错误诊断。

## 安装

### Claude Code

```bash
/plugin marketplace add https://gitcode.com/cann/skills.git
/plugin install torch-compile@cannbot
/reload-plugins
```

安装后新开会话，或在当前会话执行 `/clear` 触发 `SessionStart`。`torch-compile` 是主对话入口，会把 `AGENTS.md` 注入当前 Claude 上下文，并通过 `agents/torch-npugraph-ex.md` 处理 npugraph_ex 专项工作。

验证：

```bash
claude plugin list
# 应看到 torch-compile@cannbot ✔ enabled
```

### OpenCode

```bash
opencode plugin cannbot@git+https://gitcode.com/cann/skills.git
```

如需只启用该插件，在 `.opencode/opencode.json` 中配置：

```json
{
  "plugin": [["cannbot@git+https://gitcode.com/cann/skills.git", {"team": "torch-compile"}]]
}
```

配置后重启 OpenCode。OpenCode 会注册 `torch-npugraph-ex` Subagent，并通过插件消息 transform 把 `AGENTS.md` 注入主对话上下文。

如果更新插件后仍只看到旧的 skills、agents 或上下文没有变化，可能是 OpenCode 复用了旧插件缓存。可清理缓存后重新安装并重启：

```bash
rm -rf ~/.cache/opencode/
opencode plugin cannbot@git+https://gitcode.com/cann/skills.git
```

## 使用示例

```text
我想用 torch.compile 在昇腾 NPU 上加速推理，应该怎么配置 npugraph_ex？
```

```text
我的模型使用 npugraph_ex 编译失败了，帮我分析这段报错日志。
```

```text
怎么把自定义算子加入 npugraph_ex 图编译？
```

## 可用 Agents

| Agent | 用途 |
|------|------|
| `torch-compile` | torch.compile 图模式 primary 编排入口 |
| `torch-npugraph-ex` | npugraph_ex / aclgraph 模式专项 Subagent |

## 可用 Skills

| Skill | 用途 |
|------|------|
| `torch-npugraph-ex-knowledge` | npugraph_ex 基础知识与配置说明 |
| `torch-npugraph-ex-template` | npugraph_ex 代码模板 |
| `torch-npugraph-ex-dfx-triage` | 问题定位分诊 |
| `torch-npugraph-ex-compile-error-diagnosis` | 编译错误诊断 |
| `torch-npugraph-ex-runtime-error-diagnosis` | 运行时错误诊断 |
| `torch-custom-ops-guide` | 自定义算子入图指导 |