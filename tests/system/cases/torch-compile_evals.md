---
team_name: torch-compile
eval_mode: text
---

# Case 1: torch.compile 图模式介绍

## Config
- Disabled: true
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: torch-npugraph-ex-knowledge;torch-npugraph-ex-template;torch-npugraph-ex-compile-error-diagnosis;torch-npugraph-ex-runtime-error-diagnosis

## Prompt

我想了解在昇腾 NPU 上使用 torch.compile 图模式来加速模型推理，torch-compile 团队能提供什么帮助？

## Expected Output

回复应覆盖以下要点：
1. torch-compile 是 PyTorch torch.compile 图模式编排入口，负责识别模式和调度对应 Subagent
2. 当前主要支持 npugraph_ex / aclgraph 图模式
3. 包含 torch.compile + TorchAir 的配置、脚本迁移、调试诊断、性能优化、自定义算子入图等能力
4. 由 subagent 执行具体专科工作，主 agent 做编排和模式选择

## Expectations

---

# Case 2: 图模式不适用的场景

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我的模型在 NPU 上运行，什么时候不适合用 torch.compile 图模式？图模式有什么限制？

## Expected Output

回复应说明图模式适用的场景和不适用的情况：图模式主要适用于推理加速场景，对于需要动态图特性或频繁修改模型结构的场景可能不适用，应根据实际的模型结构和部署需求来判断是否使用图模式

## Expectations

---

# Case 3: 使用图模式前需要的信息

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想用 torch.compile 加速模型推理，先帮我看看需要准备哪些信息，先不急着开始。

## Expected Output

回复应在动手前主动确认必要信息：模型框架和来源、当前运行环境和 NPU 配置、是否已有可运行的推理脚本、性能基线数据等，而不是在缺少这些信息的情况下直接开始图模式适配

## Expectations

