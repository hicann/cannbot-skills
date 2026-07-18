---
skill_name: model-infer-profiling
eval_mode: text
---

# Case 1: NPU profiling 正确采集的关键约束

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我要在昇腾 NPU 上对一个 PyTorch 推理模型采集 profiling，怎么配置才能拿到完整的 kernel 级数据？只讲方法和关键约束，不用写完整脚本。

## Expected Output

回复应说明用 torch_npu.profiler 采集，且必须把 ExperimentalConfig 配成 Level1 + PipeUtilization，否则 kernel_details.csv 列不全、op_statistic/api_statistic 不生成；产出与 Ascend TensorBoard 兼容的 ASCEND_PROFILER_OUTPUT 目录，并在采集后验收产物完整性。

## Expectations

- [contains] PipeUtilization
- [skill_activated] model-infer-profiling

# Case 2: 框架内置 vs 注入采集的路由判断

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

仓库的推理脚本里已经有 enable_profiler 开关了，我还需要自己写一套 profiler 采集代码吗？怎么判断走哪条路？只讲思路。

## Expected Output

回复应说明先做 Step 0 路由：仓库已内置 profiler（有 enable_profiler 开关或 profiler 封装）时走"框架内置"路径、只调参数不再重复注入；都没有时才走"注入采集"路径自己写 profiler。应优先用框架内置。

## Expectations

- [skill_activated] model-infer-profiling
