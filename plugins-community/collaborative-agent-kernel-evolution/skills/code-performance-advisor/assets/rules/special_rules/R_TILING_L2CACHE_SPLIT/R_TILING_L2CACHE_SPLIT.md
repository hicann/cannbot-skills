# 规则名称：L2Cache 切分，改善缓存命中与带宽抖动

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要对大块数据进行多次重复访问，且单次访问的数据总量超过了单核 L1 缓冲但未超过核心组 L2 Cache 容量。
- **形状/数据类型上下文**：大规模数据（`C.K.Large` 等）配合特定分块（`C.Tile.Small`）。

## 2. 模式描述 (Pattern)
- **优化原理**：在 Host 侧 Tiling 时引入 L2 Cache 友好的切分方案。通过限制单次迭代的数据覆盖范围，使其“在 L2 范围内闭环”。
- **目标**：提高数据的 L2 命中率（Reuse Ratio），减少数据请求穿透到 Global Memory (HBM) 的次数，平抑总线带宽抖动。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：穿透 L2 的访问具有极高的延时。如果 Tiling 导致 Cache 线被频繁替换（Thrashing），带宽将急剧下降。
- **事实桥接**：
  - 增加复用 -> 降低 `l2_cache_miss_rate`。
  - 稳定带宽 -> 实现 `S.TransferDominated` 场景下的吞吐最大化。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `l2_cache_miss_rate` (若 profiling 支持：L2 缺失率指标)
  - `aic_mte2_ratio`（外存搬运占比）
  - `hbm_bw_util`（带宽利用率）
- **如何解读（定性）**：
  - 观察到数据的访问次数异常偏高，且每次访问都伴随着大量的 GM 数据搬运。
  - 判定算子是否处于 Roofline 的访存受限区，且存在优化的 Tiling 空间。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_TILING_L2CACHE_SPLIT/code_snippets/`
- **实施步骤**：
  - 重新设计多核划分与 Data Path；
  - 确保同一核或相邻核在连续迭代中处理同一数据局部块；
  - 并在 tiling 参数中通过 `C.L2.Shared` 进行容量限定。

## 6. 约束与副作用 (Constraints)
- **并行核数**：切分过细可能导致单核工作量不足，引起负载均衡问题（`S.LowComputeUtil`）。
- **同步压力**：复杂的 Tiling 可能增加标量计算的指令数。
- **适用场景**：`S.CacheMiss`, `S.TransferDominated`。

## 7. 验证逻辑 (Verification)
- **验证原则**：关注外部存储（HBM）吞吐量的物理性压降。
- **推荐验证项**：
  - `l2_cache_miss_rate`：期望大幅下降；
  - `Task Duration(us)`：性能期望整体提升。
- **验证方法**：通过性能监测工具确认相同任务下 GM 的 Read 总量减少。

## 标签
- Domain: `U.DMA`, `O.General`
- Symptom: `S.CacheMiss`, `S.TransferDominated`
- Context: `C.L2.Shared`
