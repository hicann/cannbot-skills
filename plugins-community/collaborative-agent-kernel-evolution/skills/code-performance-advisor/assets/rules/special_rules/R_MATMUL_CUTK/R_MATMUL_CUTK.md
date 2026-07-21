# 规则名称：MatMul 切K模板优化，提升多核并行度与带宽利用率

## 1. 需求场景 (Requirement)
* **业务逻辑**：矩阵乘法算子，特别是 M、N 维度较小但 K 维度较大的 shape（如 (32, 4096) × (4096, 32)）。
* **物理瓶颈**：MTE2 bound，搬运成为瓶颈；传统按 M、N 轴分核时并行度不足，左右矩阵重复搬运严重。

## 2. 模式描述 (Pattern)
- **优化原理**：在按 M、N 切分的多核模板基础上，加入对 K 轴的切分（切K），使更多核参与搬运与计算，降低单核重复搬运量并提升总体带宽利用效率。
- **物理目标**：提高单核参与的输出维度覆盖和核总体参与度，减弱搬运单元（MTE2）在总耗时中的主导性，改善带宽利用趋势。

## 3. 性能损耗因果链 (Inference / Physics)
* **因果推导（需专家复核）**：
  $$T_{data\_copy} = \frac{M \times K + K \times N}{BW_{eff}}, \quad \text{切K后}: BW_{eff} \uparrow \text{（并行度增加）}$$
* **物理事实桥接**：
  - MTE2 bound 情况下，增加 singleCoreN 和 singleCoreM 可减少重复搬运
  - K 轴切分后，tileNum 增加，更多核参与计算
  - 单核内 K 轴累加保证确定性（避免 atomic 带来的不确定性）

## 4. 触发信号 (Triggers — 基于 op_summary 原始字段与标签体系)
- **需要查看的指标（以原始 profiling 表头为准）**：
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（用于判断 MTE2 在算子耗时中的占比／是否主导）
  - `Task Duration(us)`（算子总体耗时趋势）
  - `cube_utilization(%)`（核/算力利用情况）
  - `Block Dim`（切分粒度与并行度指示）
  - `Input Shapes` / `Output Shapes`（用于判定 M、N、K 的相对规模）
- **如何解读（定性化）**：
  - 检查是否为搬运主导的样例（MTE2 相关比重是主导项）；
  - 关注带宽利用与核利用是否偏低或存在并行度瓶颈；
  - 从 shape 字段判断是否属于“`C.MN.Small` 且 `C.K.Large`”的上下文；
  - 观察 `Block Dim` 是否较小以致并行核数不足。

## 5. 动作实现 (Action)
正反例代码参考同目录下的 `code_snippets/case_{x}/base_code` 或 `good_code`，实施时：
- 按规则在模板中加入 K 切分策略；
- 为每个并行分片分配合适的 workspace 缓冲以支持跨核累加；
- 使用单核内累加策略以避免原子操作开销。

## 6. 约束与副作用 (Constraints)
- **内存开销**：需要额外的 workspace 空间，开销与核数及输出尺寸相关；
- **适用场景**：
  - 以搬运相关（MTE2 / DMA）为主的 MatMul；
  - `C.MN.Small` 与 `C.K.Large` 的形状上下文；
  - 系统有足够的 UB/workspace 空间（受 `C.UB.Capacity` 约束）。
- **不适用场景**：
  - 纯计算受限（Compute bound）场景（切K 会增加同步开销且可能反而变差）；
  - 当 M、N 已足够大且传统按 M/N 分核已提供充分并行度时。

## 7. 验证逻辑 (Verification)
- **验证要点（用趋势/方向描述）**：
  - 观察 `Task Duration(us)` 的趋势，应呈下降（改善）；
  - 观察 `aic_mte2_ratio` / `aiv_mte2_ratio` 的趋势，应呈下降（MTE2 占比减弱）；
  - 观察 `Block Dim` 或实际参与核数的变化，应呈增加（并行度上升）；
  - 观察 `cube_utilization(%)` 或带宽相关指标的趋势，应呈上升（带宽/核利用改善）。
- **验证方法**：
  - 使用 `op_summary` 原始字段进行对比，侧重趋势而非绝对阈值；
  - 在不同 representative shapes 下对比“优化前 / 优化后”的指标走向；
  - 确认 workspace 使用情况符合 `C.UB.Capacity` 的约束。

参考资料：
 无
