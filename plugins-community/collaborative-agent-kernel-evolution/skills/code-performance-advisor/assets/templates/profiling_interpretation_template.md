# Profiling 流水图解读模板（Profiling Interpretation Template）

> 说明：用于标准化对 msprof / op_summary 等 profiling 输出的流水/甘特图的解读。模板以占位符与填表格方式引导分析，避免使用具体数值阈值，仅强调观察维度与趋势判断。

## 基本信息（Metadata）

- 算子名称（占位）：
- 硬件平台（占位）：
- 数据规模/示例形状（占位）：
- 工具来源：例如 `msprof`、`op_summary.csv`。

## 1. 图像结构识别（What you see）

- 关键阶段（填写是否存在与定性占比）：`CopyIn / Load`、`Compute`、`CopyOut / Store`、`Sync/Barrier`、`Prefetch`、`Reduce` 等。请以“存在/不存在/显著/次要”等定性词描述。
- 阶段时序关系：说明是否存在重叠、严格串行或 Ping-Pong Buffer 等特征。

## 2. 时间线结构（Macro形态）

- 选择最符合的形态（填写）：纯三段式 / 交错式 / Ping-Pong / 多阶段流水 / 带气泡 / 近理想重叠。
- 直观印象（简短结论）：例如“CopyIn 相对较长，Compute 等待数据”。

## 3. 瓶颈定位（Where is the bottleneck）

- 最长或主导阶段（填写）：例如 CopyIn / Compute / CopyOut / Sync / 不明显。
- 依据与证据：引用 profiling 字段或时间线特征作为判断依据（例如 Copy 与 Compute 的重叠度、等待气泡位置等）。

## 4. 重叠效率与调度评估（Overlap & Scheduling）

- Copy vs Compute 重叠度（定性）：高 / 中 / 低。
- 预取覆盖率与气泡比例：使用定性描述（良好 / 可优化 / 严重浪费）。

## 5. 资源视角（Resource Utilization）

- 计算资源（算力利用、MMA/Vector 使用、并行度）——定性说明。
- 存储/带宽资源（HBM 带宽、UB 利用、重用率）——定性说明。
- 调度与 Buffer（Ping-Pong / Triple Buffer、气泡与同步）——定性说明。

## 6. 模型化归因（Root Cause / Causal Chains）

- 列出主要因果链（主因、次因、结构性因素），每条用一句话概括原因与路径。例如：
  - CopyIn 长 → 中间结果重用低 → 带宽瓶颈
  - Buffer 数量有限 → 预取覆盖不足 → Compute 出现等待

## 7. 优化方向与建议（What to do next）

- 针对检测到的瓶颈列出可执行方向（示例）：增加 Tile、加强 Prefetch / 增加 Buffer、减少冗余 Copy、合并规约、减少同步点等。
- 对每项建议说明预期方向性效果（改善/上升/下降/可能造成的副作用）。

## 8. 验证建议（How to validate）

- 使用 `op_summary` 原始字段（例如 `Task Duration(us)`、`cube_utilization(%)`、`aic_mte2_ratio` 等）对比“优化前/后”的趋势。避免固定阈值，以趋势为主。
- 覆盖不同代表性形状，确保优化在多场景下均有一致方向性影响。

## 结论（Summary）

- 用一段话总结最关键的瓶颈与首要优化方向，以及可能的风险点或约束（例如 memory/UB 限制）。

---
