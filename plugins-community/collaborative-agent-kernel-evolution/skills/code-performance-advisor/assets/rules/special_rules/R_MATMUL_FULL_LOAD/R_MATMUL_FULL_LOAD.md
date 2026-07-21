# 规则名称：MatMul 全载模式：消除小矩阵重复搬运

## 1. 需求场景 (Requirement)
- **业务背景**：典型的 Weight-Static 或增量推理场景。其中一路输入（通常是左矩阵 A 或权重 B）相对较小。
- **形状/数据类型上下文**：Matmul 中有一侧数据可以完整装入单核的 L1 Buffer。

## 2. 模式描述 (Pattern)
- **优化原理**：
  - **不分核全载**：小矩阵在初始化阶段一次性搬入全核共享的 L1 或 L2。
  - **L1 驻留**：在计算循环内部，将该小矩阵标记为驻留，确保在处理不同的输出分片时，该矩阵仅被从 GM 搬运一次。
- **目标**：消除“小矩阵随着大矩阵遍历而反复被拉取（Repeated Fetch）”的无效流量。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：普通 Matmul 模板默认两路数据均不持久化。当 N 维度很大时，左矩阵 M*K 会被重复搬运 N/n_tile 次。
- **事实桥接**：
  - 减少 MTE2 命令发射 -> 降低指令调度开销。
  - 释放总线带宽 -> 缓解 `S.MteBusy` 症状。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio`（搬运占比）
  - `mte_busy`（搬运总忙碌度）
- **如何解读（定性）**：
  - 观察 `aic_mte2_ratio` 是否异常偏高（如 > 80%），且该算子属于矩阵乘法。
  - 检查 Tiling 参数，确认是否存在“一路极小、一路极大”的极端非对称 Shape。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MATMUL_FULL_LOAD/code_snippets/`
- **实施步骤**：
  - 在 Tiling 侧设置 `baseM/baseN` 为 FullLoad 模式；
  - 核心：`mm.SetTensorA(..., isFullLoad=true)` 或在 `REGIST_MATMUL_OBJ` 时指定全载模板。

## 6. 约束与副作用 (Constraints)
- **内存约束**：全载后的张量必须能塞进 L1。如果超出，会导致 L1 溢出到 L2 甚至报错。
- **适用场景**：`O.MatMul`, `S.MteBusy`, `S.TransferDominated`。

## 7. 验证逻辑 (Verification)
- **验证原则**：搬运总耗时的物理压缩。
- **推荐验证项**：
  - `aic_mte2_ratio`：期望显著下降；
  - `Task Duration(us)`：在 LLM 增量推理场景下性能提升通常在 15%~35% 之间。
- **验证方法**：检查 Profiling 甘特图，确认 MTE2 的流水段中原本重复的负载块被一个长期的驻留块合并或消失。

## 标签
- Domain: `U.Cube`, `O.MatMul`
- Symptom: `S.MteBusy`, `S.TransferDominated`, `S.DmaOverhead`
- Context: `C.Arch.910B`
