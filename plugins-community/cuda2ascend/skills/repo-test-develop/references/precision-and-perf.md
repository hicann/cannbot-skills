# 精度测试标准与性能采集方法

> 按需求文档的**评测来源模式**分两条路径：
> - **模式 B（无评测集，默认）**：精度按 `ops-precision-standard` 容差自检，性能用 `ops-profiling`（msprof op）自采。
> - **模式 A（有评测集）**：评测集评测以框架内置容差与 HAP 评分为最终裁定；开发期自检可参照 `ops-precision-standard`，性能由评测集评测脚本内置 profiler 完成。
>
> 测试工程搭建见 [test-framework.md](test-framework.md)。

## 精度测试

### 模式 B：自建精度测试

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

### 模式 A：评测集评测口径（最终裁定）

评测集评测器（如 cann-bench 的 `run_evaluation.sh`）执行 cases 全量用例，候选输出与 golden 比对，按框架内置容差判定每用例通过/失败。

**执行方式**：

```bash
# 仅精度验证（不采集性能）
<评测集>/scripts/run_evaluation.sh --source-dir <提交目录> --operator <Op> --no-perf
```

开发期自检标准同模式 B（`ops-precision-standard`），但**模式 A 下评测集框架内置容差为最终裁定**。

### 精度不达标处理（两模式通用）

先判断问题类型，再交回对应角色：

| 特征 | 问题类型 |
|------|---------|
| 某些元素输出全 0 / NaN，或仅特定核数据错误，或 Padding 区域参与计算 | 代码 bug |
| FP32 好但 FP16/BF16 差很多，或误差随规模线性增长，或所有 dtype 均匀不足 | 精度问题（混合精度/归约顺序/数值稳定性） |

调用 `ascendc-precision-debug` 诊断根因，在报告中记录问题类型与诊断结论。

### 同源与特殊用例纪律（两模式通用）

- **模式 B**：golden 为 CPU 侧独立实现，不复用被测 Kernel；本仓仅一份 golden，保持唯一。
- **模式 A**：golden 由评测集提供，提交方不重写，自测时引用评测集 golden，不另立第二份。
- **同源截断**：设备侧对输入所做的量化/截断，golden 必须对同一份数据做同样处理——整型造数据须先 round 再 clamp；fp16/bf16 须逐操作数先舍入到该 dtype 再计算。
- **特殊输入语义**：0 维标量张量（shape `[]`）与空 tensor（shape 含 0）是不同用例；特殊值 `±0/±inf/nan` 原样注入，不有限化为普通数。

## 性能采集

### 模式 B：自建性能采集

用测试工程搭建的性能采集框架跑出性能数据，覆盖需求关注的 shape/dtype。通过 `ops-profiling` 执行 msprof op 采集、解读 CSV 指标、做达标判定。

**关注指标**：

- Task Duration（实际耗时与理论耗时对比，差距 <20% 为达标参考）
- 主导流水 / PipeUtilization 分布是否与算子类型匹配
- 核间负载均衡（各核耗时差异 <10%）
- Bank conflict 占比（<5%）、头开销占比（<10%）

**性能目标**：以需求文档的可量化指标（耗时/带宽/利用率）表述；无明确目标时以瓶颈分析结论判定。

### 模式 A：HAP 评测（最终裁定）

评测集评测器对通过精度的用例采集性能，按 HAP 公式打分（详见 `repo-knowledge` 的 evaluation-contract.md）：

$$
\text{HAP}_i = \frac{T_{\text{baseline},i} - T_{\text{HW},i}}{(T_{\text{cand},i} - T_{\text{HW},i}) + (T_{\text{baseline},i} - T_{\text{HW},i})}
$$

- $T_{HW}$ = 评测集性能基线的 `t_hw_us`
- $T_{baseline}$ = `baseline_perf_us`
- $T_{cand}$ = 候选 kernel 实测耗时

**执行方式**：

```bash
# 精度 + 性能（含 HAP）
<评测集>/scripts/run_evaluation.sh --source-dir <提交目录> --operator <Op>
```

性能报告输出到 `reports/`（如 `cann_performance_eval_*.json`，各用例 HAP 得分、综合性能分）。

**HAP 是饱和型指标**：

| HAP | 含义 | 开发目标 |
|-----|------|---------|
| < 0.5 | 低于 baseline | 不达标，须优化 |
| = 0.5 | 等于 baseline | 最低可接受 |
| > 0.5 | 优于 baseline | 达标 |
| ≥ 1 | 达到/超过硬件理论上界 | 优秀（允许超 100 分） |

> 模式 A 下需求文档的性能预期以 HAP 阈值表述（如「HAP ≥ 0.5」即性能不低于 baseline）。

### 性能瓶颈定位（两模式通用）

性能瓶颈定位后，若根因落在上游 Tiling/切分/架构，回退给设计角色，不自行改设计决策。开发期可用 `ops-profiling` 辅助瓶颈分析。
