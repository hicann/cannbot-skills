---
type: CATLASS DSL Profiling Guide
title: torch_npu profiler 采集 DSL kernel 的可用口径
description: 说明 CATLASS DSL kernel 在 torch_npu profiler 产物中的可见性，给出 Level1 + aic_metrics 的采集口径、kernel_details/op_summary/trace_view 的解析顺序与回退判定。
tags: [catlass-dsl, profiler, torch-npu, aic-metrics, trace, kernel-details]
status: stable
generated: {by: process:catlass-dsl-profile-capture, at: '2026-11-06T18:45:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-11-06T18:45:00+08:00'}
  - {by: process:catlass-dsl-profile-probe, at: '2026-11-06T18:40:00+08:00'}
sources:
  - id: profiler-level
    resource: https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-ProfilerLevel.md
    title: torch_npu.profiler.ProfilerLevel
  - id: profiler-config
    resource: https://www.hiascend.com/document/detail/zh/mindstudio/2620/msTT_msIT/msprof_analyze/docs/zh/advanced_features/operator_mfu_instruct.md
    title: 算子 MFU 分析（含 _ExperimentalConfig 采集示例）
operator_families: [matmul]
arch: [c310]
---

# 接口与概念

CATLASS DSL kernel 经 `tla.compile` → `hivmc` → catlass runtime 自定义下发，不经过
torch dispatcher / aclnn。`torch_npu.profiler` 默认 `ProfilerLevel.Level0` 只提供
上层应用与算子级数据；项目案例中出现过 `async_npu` 只剩 aclnn op 和 `torch_to_npu`
拷贝、看不到 DSL kernel 的情况，但这不是必然结果，见下方实测。[^profiler-level]

要拿到 kernel 级耗时与 AIC pipe 指标，需要 `ProfilerLevel.Level1` 及以上；Level1
会额外采集 CANN 层 AscendCL 数据、NPU 上执行的 AI Core 性能指标，并启用
`aic_metrics=AiCMetrics.PipeUtilization`。[^profiler-level]

DSL kernel 会作为独立 kernel 行进入产物，`Accelerator Core` 为 `AI_VECTOR_CORE` 或
`AI_CORE`，带 `Block Num`、`Task Duration` 和 `aiv_vec/mte2/mte3/scalar` 的时间与占比。
采集产物里没有 `trace.json`，trace 文件是 `ASCEND_PROFILER_OUTPUT/trace_view.json`，
其 `name` 就是 kernel 名、`tid` 是 stream id；按 `trace.json` 查找会误判成"没有 kernel"。

# 用法

## 采集

```python
import torch
import torch_npu

cfg = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    export_type=[
        torch_npu.profiler.ExportType.Text,
        torch_npu.profiler.ExportType.Db,
    ],
    data_simplification=True,
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=1, active=1, repeat=1, skip_first=0
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof"),
    experimental_config=cfg,
) as prof:
    run_once()          # 与 benchmark 相同的 shape / dtype / launch
    prof.step()
```

采集前先单独跑一次正确性与稳定性基线；profiler 有开销，采集数据只用于归因，
不用于最终性能结论。[^profiler-config]

## 解析

按以下顺序找导出产物（`<out>/<host>_<pid>_<ts>_ascend_pt/` 下）：

1. `ASCEND_PROFILER_OUTPUT/kernel_details.csv`：两种 `data_simplification` 都会生成，
   含 `Name/Type/Accelerator Core/Start Time/Duration/Block Num` 与 `aic_*`、`aiv_*`
   pipe 字段
2. `mindstudio_profiler_output/op_summary_*.csv`：仅 `data_simplification=False` 生成，
   字段与 kernel_details 相同
3. `ASCEND_PROFILER_OUTPUT/trace_view.json`：核对 kernel 是否落在采集窗口

用 kernel 名前缀匹配目标 DSL kernel，读取 `Task Duration` 与 AIV/AIC pipe 占比。
`torch_npu.profiler.profile` 没有 `key_averages` 属性（类上就不存在），任何位置调用都报
`AttributeError: 'profile' object has no attribute 'key_averages'`；聚合只走导出 CSV 或
`ASCEND_PROFILER_OUTPUT/analysis.db`。

## trace 不可用时的回退

- 用 `msprof op --kernel-name=<前缀> --aic-metrics=Default` 拉起同一脚本采集
- 仍无 kernel 事件时，用 makespan 算术 + 单变量消融判定：块数 / 核数 / 波次、
  单块耗时、与实测 wall 对齐；profiling 记 `blocked` 并写精确重跑命令

# 代码模式

```text
正确性基线（不做 profiler）
  → Level1 采集（warmup 1 + active 1）
  → 读 kernel_details / op_summary
  → 用 MAC 占比选优化轴
  → 单变量消融复测
```

# 约束

- Level0 不提供 AI Core pipe 指标；只做 Level0 时不得据此判断 MAC/MTE 瓶颈
- 采集必须覆盖实际 launch；`schedule` 的 warmup 不足会把编译或首次 launch 计入
- 导出目录必须本次新建；读旧 CSV 会把上一轮数据当成当前 kernel
- DSL kernel 名可能带编译修饰，匹配用前缀，不要求全名相等
- `data_simplification=True` 时不生成 `mindstudio_profiler_output/` 整个目录，只有
  `ASCEND_PROFILER_OUTPUT/`；不要因为找不到 `op_summary` 判定采集失败
- 同一 kernel 多次采集的 `Task Duration` 有约 7% 抖动（温度/频率），归因只比占比，
  跨 run 比较绝对值要用同一次采集内的数据

# 失败表现

- `trace_view.json` 的 `async_npu` 只有 aclnn op 和 `torch_to_npu` 拷贝，没有目标 kernel
- 按 `trace.json` 找产物：该文件不存在，误判成没有 kernel
- `AttributeError: 'profile' object has no attribute 'key_averages'`：该属性在
  `torch_npu.profiler.profile` 上不存在，不是调用时机问题
- 只在 `mindstudio_profiler_output/` 找 `op_summary` 而本次用了
  `data_simplification=True`，误判为采集失败
- `kernel_details.csv` 中 task duration 为 0，或只有 aclnn 行
- 优化结论与实测 wall 对不上，且 profiler 的总时长明显小于 benchmark

# 验证方法

- 同一 shape 采集后，`kernel_details.csv` 中出现目标 kernel，且 task duration 与
  benchmark 的 candidate mean 同量级
- 用一条已知结构的消融（例如只改块数或单块 tile）验证归因方向：profiler 指标变化
  与 wall 变化一致，才把该指标用于决策
- profiling 失败时保留 `blocked` 记录与重跑命令，不用算术结论冒充 profiler 结果

## 探针基线（2026-11-06，Ascend950PR / CANN 9.3.0 / torch_npu 2.10.0.post7 / torch 2.10.0+cpu）

对象是 `examples/python/tla_dsl/tla/kernel/vec/basic_vadd` 的 `basic_vadd`（`--n 400
--dtype f32`，56 block）。先不带 profiler 跑一次完成编译，再
`Level1 + AiCMetrics.PipeUtilization` 采集一次：

| data_simplification | kernel_details.csv | op_summary_*.csv | trace_view.json |
| --- | --- | --- | --- |
| False | 有：9.188 us | 有 | 有：`name=basic_vadd`、`dur=9.188` |
| True | 有：9.875 us | 不生成 | 有：`name=basic_vadd`、`dur=9.875` |

两次都带 `aiv_time / aiv_vec_time / aiv_vec_ratio / aiv_mte2 / aiv_mte3 / aiv_scalar`，
例如 True 那次为 `aiv_time=9.12us`、`aiv_vec_ratio=0.027`、`aiv_mte3_ratio=0.503`，
足以支撑"该 kernel 受 MTE3 搬出主导"这类 pipe 归因。`key_averages` 在
`torch_npu.profiler.profile` 类上直接不存在，与 `with` 块位置无关。
