---
skill_name: model-infer-fusion
eval_mode: text
---

# Case 1: 融合算子替换工作流

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我想把模型里的算子替换成 torch_npu 融合算子提速。整体应该按什么流程做？只讲方法，不用写代码。

## Expected Output

回复应给出分析→匹配→替换→验证的逐模块流程，强调用脚本查询 torch_npu 算子 docstring 确认接口，而不是凭记忆写 API

## Expectations

- [skill_activated] model-infer-fusion

# Case 2: 确认算子接口的方式

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

替换前怎么确认某个 torch_npu 融合算子的入参和功能，避免写错接口？

## Expected Output

回复应说明用本地 docstring 查询脚本检索算子接口、入参、功能，而不是凭记忆或猜测 API，必要时反向搜索算子名

## Expectations

- [skill_activated] model-infer-fusion

# Case 3: 覆盖的模块类型

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

融合算子替换一般覆盖模型的哪些模块？请列举可分析替换的模块类型。

## Expected Output

回复应覆盖 Attention 子链路（RoPE/Norm 等）、MoE、FFN、Norm 等模块，说明逐模块分析匹配

## Expectations

- [skill_activated] model-infer-fusion

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我把模型的算子换成融合算子。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接给替换代码

## Expectations

- [skill_activated] model-infer-fusion

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-kvcache;model-infer-graph-mode;model-infer-parallel-analysis

## Prompt

模型里 RMSNorm 和残差是手写的多个小算子，想合成一个 torch_npu 融合算子减少开销，往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-fusion，给出融合算子匹配替换方向，即使存在 KVCache、图模式、并行等相似 skill 也应选融合算子专项

## Expectations

- [skill_activated] model-infer-fusion

# Case 6: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我要重新设计一套量化算法并实现自定义量化内核，这个融合算子 skill 能做吗？

## Expected Output

回复应说明本 skill 只做既有 torch_npu 融合算子的识别与替换，不设计量化算法、不实现自定义算子，应建议用户改用量化或算子开发相关方案

## Expectations


