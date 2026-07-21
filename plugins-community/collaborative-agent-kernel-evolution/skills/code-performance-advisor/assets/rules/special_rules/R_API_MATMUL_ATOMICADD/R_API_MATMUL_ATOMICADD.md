# 规则名称：通过使能 Matmul AtomicAdd 消除冗余归并/搬运

## 1. 需求场景 (Requirement)
- **业务背景**：Matmul 的计算结果需要与 Global Memory 中已有的数据进行累加同步。
- **形状/数据类型上下文**：多核并行计算中涉及跨核归约或大矩阵累加的情形。

## 2. 模式描述 (Pattern)
- **优化原理**：在 `Matmul` 的 `IterateAll` 或 `GetTensorC` 接口中，将 `enAtomic` 参数设置为 1。利用硬件自身的原子写路径直接在 L2/HBM 侧完成加法。
- **目标**：消除“将 GM 数据搬入片上 -> 向量加法 -> 再搬回 GM”的冗余数据链路。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：软累加路径（Read-Modify-Write）由于涉及额外的同步屏蔽和双向带宽占用，极易引起总线拥塞。
- **事实桥接**：
  - 减少 MTE2/MTE3 负载 -> 缩短数据在总线上的暴露时间。
  - 硬件原语 -> 避免 Vector 单元处理这种简单的加法归并逻辑。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio` / `aic_mte3_ratio`（评估归并期间的搬运波峰）
  - `aiv_vec_ratio`（如果涉及显式加法）
  - `Task Duration(us)`（耗时走向）
- **如何解读（定性）**：
  - 在 Profiling 中观察到 Matmul 计算结束后紧跟着一段密集的“读-加-写”指令脉冲；
  - 判定算子是否存在跨分片的历史数据依赖。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_API_MATMUL_ATOMICADD/code_snippets/`
- **实施步骤**：
  - 在 `REGIST_MATMUL_OBJ` 后，配置 `mm.IterateAll(gm_dst, 1)`；
  - 确保输出缓冲区具备原子操作的对齐要求（如 `T.FP32`）。

## 6. 约束与副作用 (Constraints)
- **精度一致性**：原子操作的累加顺序由硬件调度决定，可能导致细微的浮点误差波动。
- **写冲突**：如果多核高频竞争同一 GM 地址，可能触发硬件写保护从而导致性能抖动。
- **适用场景**：`O.MatMul`, `S.MteBusy`。

## 7. 验证逻辑 (Verification)
- **验证原则**：搬运总量的显着下降与指令序列的净化。
- **推荐验证项**：
  - `aic_mte2_ratio`：期望下降（不再需要 Read-to-Add）；
  - `Task Duration(us)`：性能期望提升。
- **验证方法**：检查甘特图，确认 Matmul 之后原本独立的 Add 计算段消失。

## 标签
- Domain: `U.Cube`, `O.MatMul`
- Symptom: `S.MteBusy`, `S.DmaOverhead`
- Context: `C.Arch.910B`
