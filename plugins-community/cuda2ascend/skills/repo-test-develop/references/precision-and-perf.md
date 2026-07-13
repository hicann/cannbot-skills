# 精度测试标准与性能采集方法

> 精度测试的执行标准与性能采集方法。权威精度容差以 `ops-precision-standard` 为准；性能采集用 `ops-profiling`；精度不达标诊断用 `ascendc-precision-debug`。测试工程搭建见 [test-framework.md](test-framework.md)。

## 精度测试

**执行流程**：

1. 加载 `ops-precision-standard`，确定各 dtype 的 atol/rtol 标准。
2. 构造精度测试用例：覆盖声明的所有 dtype，每个 dtype 至少含常规 shape 与边界 shape。
3. 在 NPU 上运行精度测试，与 CPU golden 比对。
4. 记录每个 (dtype, shape) 组合的实际误差（rtol、atol、max_error）。
5. 对照标准判定是否达标。

**默认精度标准**（权威值以 `ops-precision-standard` 为准）：

| 数据类型 | rtol | atol | 说明 |
|---------|------|------|------|
| FP32 | 1e-5 | 1e-5 | 默认标准 |
| FP16 | 1e-3 | 1e-3 | 半精度宽松标准 |
| BF16 | 1e-2 | 1e-2 | BF16 更宽松 |

**精度报告格式**：

```markdown
**精度状态**: ✅通过 / ❌失败

| dtype | shape | rtol | atol | max_error | 达标状态 |
|-------|-------|------|------|-----------|---------|
| ... | ... | ... | ... | ... | ✅/❌ |
```

**精度不达标处理**：先判断问题类型，再交回对应角色：

| 特征 | 问题类型 |
|------|---------|
| 某些元素输出全 0 / NaN，或仅特定核数据错误，或 Padding 区域参与计算 | 代码 bug |
| FP32 好但 FP16/BF16 差很多，或误差随规模线性增长，或所有 dtype 均匀不足 | 精度问题（混合精度/归约顺序/数值稳定性） |

调用 `ascendc-precision-debug` 诊断根因，在报告中记录问题类型与诊断结论。

## 性能采集

- 用测试工程搭建的性能采集框架跑出性能数据，覆盖需求关注的 shape/dtype。
- 通过 `ops-profiling` 执行 msprof op 采集、解读 CSV 指标、做达标判定。
- 采集数据归档（按轮次组织），记录达标判定；未达标时记录瓶颈分析。

**关注指标**：

- Task Duration（实际耗时与理论耗时对比，差距 <20% 为达标参考）
- 主导流水 / PipeUtilization 分布是否与算子类型匹配
- 核间负载均衡（各核耗时差异 <10%）
- Bank conflict 占比（<5%）、头开销占比（<10%）

> 性能瓶颈定位后，若根因落在上游 Tiling/切分/架构，回退给设计角色，不自行改设计决策。
