# 规则名称：Vector 算子 Counter 模式：降低控制开销

## 1. 需求场景 (Requirement)
- **业务背景**：Vector 算子需要处理尾块数据或执行大量重复的计数循环。
- **形状/数据类型上下文**：数据量无法被指令 Block 长度整除，存在复杂的边界分支。

## 2. 模式描述 (Pattern)
- **优化原理**：使用 `SetVectorMask` 的 `COUNTER` 模式。直接向硬件寄存器传递计算总量，由硬件自动处理循环与尾块掩码。
- **目标**：消除显式的 `for` 循环和冗余的分支判断（if-else），降低标量指令（Scalar）的流水压力。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：频繁的标量控制会导致指令流水线前段（Fetch/Decode）成为瓶颈。Counter 模式将这种控制下沉到向量指令执行器内部。
- **事实桥接**：
  - 简化指令流 -> 降低 `aic_scalar_ratio`。
  - 填满流水线 -> 提高 `aiv_vec_ratio` 的有效密度。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aiv_vec_ratio`（向量利用率）
  - `aic_scalar_ratio`（标量占比）
  - `Task Duration(us)`（耗时走向）
- **如何解读（定性）**：
  - 判定算子是否因微小的尾块计算（Tail logic）而导致耗时异常跳变。
  - Scalar 指令占比高于预期，阻碍了 AIV 的指令发射。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_API_VECTOR_COUNTER_MODE/code_snippets/`
- **实施步骤**：
  - 调用 `SetMaskCount()` 进入计数器模式；
  - 设置 `SetVectorMask<T, MaskMode::COUNTER>(ELE_SIZE)`；
  - 简化循环逻辑，直接调用向量接口（如 `Add`）。

## 6. 约束与副作用 (Constraints)
- **模式局限**：仅适用于特定的简单 Binary/Unary 算子。
- **配置一致性**：必须在计算结束后通过 `ResetMask()` 还原状态。
- **适用场景**：`S.LowVecUtil`, `S.ScalarBound`。

## 7. 验证逻辑 (Verification)
- **验证原则**：指令总数的物理缩减。
- **推荐验证项**：
  - `aiv_vec_ratio`：期望提升；
  - `aic_scalar_ratio`：期望显著下降。
- **验证方法**：统计指令发射计数，确认跳转指令（B/BL）和标量操作数量减少。

## 标签
- Domain: `U.Vector`, `O.Elementwise`
- Symptom: `S.LowVecUtil`, `S.ScalarBound`
- Context: `C.Arch.910B`
