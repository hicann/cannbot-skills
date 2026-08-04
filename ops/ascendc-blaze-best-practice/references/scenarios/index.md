# 定制扩展场景索引

- Step 3 仅在 Blaze 官方库方案已记录明确 `native_gaps` 时查询本表。
- 只有唯一命中一个场景时，才读取该行的设计指导；具体前提由场景设计指导复核。

| 场景 ID | 支持范围 | 准入条件（均满足） | 设计指导 | 开发指导 |
|---|---|---|---|---|
| `elementwise-broadcast-epilogue-fusion` | MatMul 类主计算的最终输出后接一个或多个 Vector 运算的 Epilogue。Vector DAG 仅由 elementwise 运算、具有静态明确映射的 broadcast 运算及其组合构成；同形输入按逐元素映射处理。不支持 reduce/reduction 或其他非 elementwise/broadcast 的 Vector 运算，包括 scan、softmax、跨轴 normalization、gather/scatter、改变元素归属的 transpose/reorder、窗口/邻域运算、额外 MatMul/Convolution、数据相关索引或有状态更新。满足前述条件的 quant/dequant 不因类别被排除，但具体 API、dtype 和精度可行性仍由设计阶段确认。 | 主数据流为 MatMul 类主计算产生最终输出后执行一个或多个 Vector 节点；Vector DAG 的每个节点均可归类为 elementwise 或具有明确映射规则的 broadcast；每个最终输出元素只依赖对应 MatMul 输出元素、映射到同一输出位置的输入及其 Vector 中间结果；所有参与输入和输出的 shape、dtype、layout、broadcast 轴及索引规则均可冻结。 | [设计指导](elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-design.md) | [开发指导](elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-development.md) |
