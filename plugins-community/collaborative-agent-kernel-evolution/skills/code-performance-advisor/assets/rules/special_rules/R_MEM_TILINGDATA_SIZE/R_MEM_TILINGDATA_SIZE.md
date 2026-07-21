# 规则名称：限制 TilingData 结构大小，降低 GM→栈拷贝开销

## 1. 需求场景 (Requirement)
- **业务背景**：算子启动时需要从 Host 侧传递复杂的 TilingData 结构体至内核。
- **形状/数据类型上下文**：小 Shape 或高并行核数场景，此时 Tiling 数据的拷贝耗时占比不可忽视（`C.Tile.Small`）。

## 2. 模式描述 (Pattern)
- **优化原理**：优化 `TILING_DATA` 结构体，缩减其字段类型（如使用 `uint16_t` 代替 `uint64_t`），并根据内存对齐规则排列字段以减少补位填充（Padding）。
- **目标**：降低内核启动时 GM -> Stack 的内存拷贝延时。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：TilingData 的拷贝发生在内核启动的临界路径（Critical Path）上。结构体越大，Cache 污染和总线占用越严重。
- **事实桥接**：
  - 紧凑布局 -> 提高缓存命中率。
  - 减少搬运负载 -> 缩短内核初始化（GET_TILING_DATA）周期。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Task Duration(us)`（观察算子执行的初始段延时）
  - `aic_scalar_ratio`（标量拷贝流负载）
- **如何解读（定性）**：
  - 如果 TilingData 大小超过特定规模且对应极小的 Shape。
  - 观察到 `GET_TILING_DATA` 带来的同步开销在整个执行流程中占比较大。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_TILINGDATA_SIZE/code_snippets/`
- **实施步骤**：
  - 审查 `TILING_DATA_FIELD_DEF` 中的类型选择；
  - 按照 8 字节、4 字节、2 字节的顺序进行排布，消除补位空白；
  - 移除已不再使用的冗余字段。

## 6. 约束与副作用 (Constraints)
- **数据溢出**：需严格验证字段缩容后的数值范围是否覆盖业务场景。
- **对齐正确性**：不正确的对齐会导致硬件访问性能反而下降。
- **适用场景**：`S.ScalarBound`, `C.Tile.Small`。

## 7. 验证逻辑 (Verification)
- **验证原则**：关注算子启动“冷时间”的缩减趋势。
- **推荐验证项**：
  - `Task Duration(us)`：期望呈微弱下降趋势；
  - 内核启动后的第一个指令发射时间：期望前移。
- **验证方法**：检查二进制符号表或使用 Profiling 细化分析工具。

## 标签
- Domain: `U.CPU`, `O.General`
- Symptom: `S.LowComputeUtil`, `S.ScalarBound`
- Context: `C.Arch.910B`
