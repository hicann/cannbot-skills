# op_summary 表头字段解读（LLM 友好版）

本文件解释 op_summary CSV 的表头含义与使用建议。字段名保持原样，方便对照原始 CSV。

## 1) 基本信息

- `Device_id`: 设备编号。
- `Model ID`: 模型编号。
- `Task ID`: 任务编号（单个算子/任务的唯一标识）。
- `Stream ID`: 流编号（并行执行的流）。
- `Op Name`: 算子实例名（更具体，可能包含变体/后缀）。
- `OP Type`: 算子类型（用于聚类/索引）。
- `OP State`: 动态/静态状态（dynamic/static）。
- `Task Type`: 执行单元类型（如 AI_CORE / AI_VECTOR_CORE / AI_CPU / MIX_AIC）。
- `Context ID`: 上下文编号（可能用于多上下文/多图场景）。

## 2) 时间与调度

- `Task Start Time(us)`: 任务开始时间（微秒）。
- `Task Duration(us)`: 任务端到端耗时（微秒）。
- `Task Wait Time(us)`: 任务等待时间（微秒）。
- `Block Dim`: Block 切分数量（近似并行度线索）。
- `Mix Block Dim`: MIX 算子的 block 切分数量（若无则为 0 或 N/A）。
- `HF32 Eligible`: 是否支持/启用 HF32（硬件/算子能力相关）。

## 3) 输入输出信息（字符串字段）

- `Input Shapes`: 输入形状（字符串，可能包含多个输入）。
- `Input Data Types`: 输入数据类型（字符串，可能包含多个输入）。
- `Input Formats`: 输入布局格式（ND/NCHW/NHWC 等）。
- `Output Shapes`: 输出形状。
- `Output Data Types`: 输出数据类型。
- `Output Formats`: 输出布局格式。

## 4) Cube 侧指标（AI Core / Cube）

- `aicore_time(us)`: Cube 理论执行时间（理想并行口径，可能偏乐观）。
- `aic_total_cycles`: Cube 侧总 cycles。
- `aic_mac_time(us)`: MAC 指令累计时间。
- `aic_mac_ratio`: MAC 指令周期占比（0~1）。高表示计算主导。
- `aic_scalar_time(us)`: 标量/控制指令累计时间。
- `aic_scalar_ratio`: 标量/控制指令周期占比（0~1）。高表示控制开销重。
- `aic_mte1_time(us)`: MTE1 累计时间。
- `aic_mte1_ratio`: MTE1 周期占比（0~1）。高表示片上搬运压力大。
- `aic_mte2_time(us)`: MTE2 累计时间。
- `aic_mte2_ratio`: MTE2 周期占比（0~1）。高表示外部访存/搬运占用高。
- `aic_fixpipe_time(us)`: FixPipe 累计时间。
- `aic_fixpipe_ratio`: FixPipe 周期占比（0~1）。
- `aic_icache_miss_rate`: Cube 指令 cache miss 率。
- `cube_utilization(%)`: Cube 利用率（可能为百分比字符串；通常会归一化为 0~1）。

## 5) Vector 侧指标（AI Vector）

- `aiv_time(us)`: Vector 理论执行时间（理想并行口径）。
- `aiv_total_cycles`: Vector 侧总 cycles。
- `aiv_vec_time(us)`: Vector 指令累计时间。
- `aiv_vec_ratio`: Vector 指令周期占比（0~1）。高表示向量计算主导。
- `aiv_scalar_time(us)`: Vector 标量/控制指令累计时间。
- `aiv_scalar_ratio`: Vector 标量/控制周期占比（0~1）。高表示控制开销大。
- `aiv_mte2_time(us)`: Vector 侧 MTE2 累计时间。
- `aiv_mte2_ratio`: Vector 侧 MTE2 周期占比（0~1）。高表示搬运/访存占用高。
- `aiv_mte3_time(us)`: Vector 侧 MTE3 累计时间。
- `aiv_mte3_ratio`: Vector 侧 MTE3 周期占比（0~1）。高表示写回/搬出压力大。
- `aiv_icache_miss_rate`: Vector 指令 cache miss 率。

## 6) 读法提示（给模型的简单规则）

- `*_ratio` 一般在 0~1 之间，值越高表示该流水/模块占用越高。
- `*_time(us)` 是累计时间，通常与 ratio 一起看。
- `MTE2/MTE3` 高 + 计算利用率低 → 可能是搬运/访存瓶颈。
- `scalar_ratio` 高 → 标量/控制开销高。
- `icache_miss_rate` 高 → 指令获取效率低。

## 7) 缺失/空值说明

- 某些导出可能缺少 AIC/AIV 字段，空值或 0 并不代表真实为 0。
- `Task Type` 不同（AI_CORE/AI_VECTOR_CORE/MIX）会导致有效字段不同。
