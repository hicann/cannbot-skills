# 规则名称：L2 Cache 命中率提升：Z 型遍历减少 Cache 置换

## 1. 需求场景 (Requirement)
- **业务背景**：矩阵乘法（Matmul）或大算子，输入矩阵规模显著超过 L2 Cache 容量（Atlas A2 约 192MB）。
- **形状/数据类型上下文**：输入矩阵 $M \times K + K \times N > 192MB$。

## 2. 模式描述 (Pattern)
- **优化原理**：采用 "Z" 型循环展开遍历策略（蛇形遍历）。在切换行/列分片索引时，通过反转遍历方向，确保相邻计算轮次之间至少能够复用单边（左矩阵或右矩阵）驻留在 L2 中的数据。
- **目标**：将 L2 命中率从 30%~40% 提升至 70%~80%，减少对低带宽 HBM 的穿透访问。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：传统行优先遍历在“换行”瞬间，之前在 L2 预热的左矩阵数据会被全部强制置换，导致下一行首个 Tile 必须从 HBM 重新加载。
- **事实桥接**：
  - L2 带宽 ($4\sim5 TB/s$) 远高于 HBM 带宽 ($1.2 TB/s$)。
  - Z 型遍历使边界数据在 L2 中实现“原地复用”，降低了有效带宽需求压力。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio`（搬运占比）
  - `Task Duration(us)`（耗时走向）
  - `hbm_bw_util`（如果有相关带宽利用率指标）
- **如何解读（定性）**：
  - 当 $M, N$ 维度拆分较多且 $K$ 维度较大时，观察 MTE2 耗时是否随分片索引切换而出现周期性剧烈波动。
  - 定性判定：输入张量总和是否远超 192MB。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_TILING_L2CACHE_OPTIMIZE/code_snippets/`
- **实施步骤**：
  - 在外层循环引入 `reverse` 标志位；
  - 核心逻辑：`uint64_t nIdx = reverse ? (maxN - i - 1) : i;`；
  - 换行时执行 `reverse = !reverse;`。

## 6. 约束与副作用 (Constraints)
- **内存分片一致性**：要求 Tiling 策略必须能够支持这种灵活的索引寻址。
- **代码复杂度**：相比普通嵌套循环，Z 型遍历的代码可读性稍低，需注释清晰。
- **适用场景**：`U.DMA`, `S.MemoryBound`, `S.DmaOverhead`。

## 7. 验证逻辑 (Verification)
- **验证原则**：观察 HBM 物理流量的下降。
- **推荐验证项**：
  - `aic_mte2_ratio`：期望下降；
  - `Task Duration(us)`：期望下降 20%+。
- **验证方法**：检查甘特图中的 MTE2 流，确认其平稳度提升，无明显的长间隔重载空洞。

## 标签
- Domain: `U.DMA`, `O.MatMul`, `U.Cube`
- Symptom: `S.MemoryBound`, `S.DmaOverhead`, `S.LowHbmUtil`
- Context: `C.Arch.910B`, `C.K.Large`
