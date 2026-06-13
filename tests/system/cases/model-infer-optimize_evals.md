---
team_name: model-infer-optimize
eval_mode: text
---

# Case 1: 端到端推理优化流程编排

## Config
- Max Tokens: 200000
- Timeout: 900
- Distractor skills: model-infer-kvcache;model-infer-fusion;model-infer-parallel-analysis;model-infer-graph-mode

## Prompt

我想对一个 PyTorch LLM 在昇腾 NPU 上的离线推理做端到端性能优化。请描述完整的优化流程：分哪些阶段、每阶段做什么、各阶段如何衔接和验证。不需要立即动手，先给出整体方案。

## Expected Output

回复应正确激活 model-infer-optimize 团队，基于 optimize-workflow 给出分阶段编排：
1. 阶段 0：模型架构分析与基线采集，单卡可跑通后再优化
2. 阶段 1：并行化改造（多卡部署时），确定 TP/EP/DP 后再做后续阶段
3. 阶段 2：KVCache 静态化 + FlashAttention 算子替换
4. 阶段 3：融合算子替换（Attention 子链路、MoE/FFN/Norm）
5. 阶段 4：图模式适配（仅 Decode）
6. 阶段 5：优化总结
应说明每阶段遵循 分析→确认→实施→验证 流程，验证达标才进下一阶段，由 analyzer/implementer/reviewer 三类 subagent 分工执行

## Expectations

- [contains] KVCache

# Case 2: 适用边界识别

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我想用 model-infer-optimize 优化一个模型在 GPU 上的训练吞吐，框架是 JAX。这个团队适用吗？只需回答适不适用并说明原因，不用展开方案。

## Expected Output

回复应说明 model-infer-optimize 仅覆盖昇腾 NPU + PyTorch 框架的推理优化，不适用于训练优化、非 PyTorch 框架、非昇腾平台，应建议用户改用对应平台的方案

## Expectations


# Case 3: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我对模型做端到端 NPU 推理优化。先告诉我你需要哪些信息，不要现在就开始改造。

## Expected Output

回复应在动手前先确认必要信息：模型工作目录、模型来源（HF 链接/本地权重/仓库内已有）、权重路径、部署卡数等，而不是在缺模型路径和部署配置的情况下直接开始改造

## Expectations

