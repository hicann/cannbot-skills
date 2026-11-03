# 性能验收与优化

在 V2 性能验收、V3 流水分析和每轮优化候选复验时读取本文件。

## 同口径采集

使用通过 V1 的构建，在相同 SoC、物理设备、输入和构建下完成预热；baseline 和 candidate 分别执行：

```bash
for sample in 1 2 3 4 5; do
  msprof --output="./perf/<baseline_or_candidate>/sample_${sample}" \
    --ai-core=on \
    --aic-mode=task-based \
    --aic-metrics=PipeUtilization \
    --task-time=on \
    --ascendcl=on \
    <single_measured_call_command>
done
```

每次命令只执行一次正式调用。从 `op_summary_*.csv` 按 `docs/design.md` 声明的完整 kernel
序列汇总 `Task Duration(us)`，取 5 次中位数，记为 `BasicInfo Task Duration(us)`。

在 `docs/validation.md` 记录设备与输入、5 次采样值、中位数、原始证据位置及波动；
`baseline`、`candidate` 和目标列只填写带单位的耗时值。证据必须能区分本轮构建和正式样本。

## 流水分析

性能未达标时采集与目标 SoC 工具能力匹配的 AIC/AIV、Cube、Vector、MTE、GM 带宽、等待和冲突指标，并映射到 `docs/design.md` 中实际的生产消费关系。

- Cube/Vector 占用用于判断计算侧是否有可执行工作；MTE 与 GM 带宽共同区分搬运带宽不足和依赖等待。
- AIC/AIV wait、flag 和 barrier 结合对应 ready/free 边判断同步阻塞，不能只按事件编号或单个比例下结论。
- 各 pipe 可以重叠，忙碌率、等待率和 Stage 时间不能直接相加为总耗时。
- 候选记录指标证据、影响的 Stage/Pipe、资源代价和风险；局部流水收益是保留门槛，当前整体耗时收益是本轮执行排序依据。

## 候选池

只要有指标依据表明候选能够减少任一 Stage/Pipe 的计算、搬运或等待开销，就使用稳定 ID 加入候选池；当前未缩短整体耗时不能作为删除理由。候选按当前 baseline 分类：

- `active`：预计能够缩短当前整体耗时，参与本轮排序。
- `deferred`：有局部流水收益，但当前收益被其它关键路径掩盖，跨轮保留且本轮不执行。
- `selected`：本轮从 `active` 中选出的唯一候选。
- `accepted`：已验证并保留的候选。
- `rejected`：无局部收益、不可执行，或引入精度、资源、性能回退的候选；必须记录原因。

在 `docs/validation.md` 的 V3 中维护候选池，至少记录候选 ID、方案、影响的 Stage/Pipe、局部收益依据、当前预计整体收益、状态和状态依据/失效条件。`active` 和 `selected` 的预计整体收益必须为正数；当前被掩盖的 `deferred` 记为 `0 us`。

V4 只从 `active` 中选择预计整体收益最大的一个并改为 `selected`。收益相同时，依次按证据可信度、实现成本和风险排序。没有 `active` 时记录“本轮无可执行候选”和原因，保留全部 `deferred`，不触发 `performance_candidate`。

## 候选迭代

每次选择一个主要变量，返回 03 更新设计，再经 04 实现和快速精度验证，最后重跑 V1 和 V2。候选保留或回退后，以当前代码为新 baseline 重新采集流水，并重新评估所有未终结的 `active` 和 `deferred` 候选；不得沿用旧 baseline 的状态或排序连续修改。

候选实测有局部收益但没有整体收益时，可以回退实现，但候选及实测证据必须保留并转为 `deferred`。新 baseline 使其影响的流水进入关键路径时，再转为 `active` 参与排序。

性能达标，或没有 `active` 候选时结束本轮优化。最终 `docs/validation.md` 保留每个候选的依据、精度、性能、流水变化、状态变化和结论，不得删除仅因当前被掩盖而延期的候选。
