# 规则名称：避免在对象内创建/初始化 TPipe，减少重复初始化成本

## 1. 需求场景 (Requirement)
- **业务背景**：Ascend C 算子使用类结构进行逻辑封装（如 `KernelExample` 类）。
- **形状/数据类型上下文**：所有的算子执行场景（`O.General`）。

## 2. 模式描述 (Pattern)
- **优化原理**：将 `TPipe` 对象声明和初始化在类的外部（即 `__global__` 入口函数作用域内），而在类内部仅持有该对象的指针。
- **目标**：辅助编译器更好地进行标量（Scalar）指令的活跃分析和常量传播。减少类实例化伴随的高昂初始化与内存搬运开销。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：如果在类内定义大对象（如 `TPipe`），编译器在进行 Scalar 优化时会因“别名风险”或“栈排布复杂”而变得保守。
- **事实桥接**：
  - 编译器友好 -> 提升 Scalar 指令的流水性能。
  - 减少初始化 -> 降低算子启动的固定成本。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_scalar_ratio`（标量占比）
  - `Task Duration(us)`（耗时走向）
- **如何解读（定性）**：
  - 检查代码结构，判定 `TPipe pipe` 是否为类成员；
  - 判定 `L_LOW_COMPUTE_UTIL` 标签是否由异常的标量延迟引起。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_API_AVOID_TPIPE_INIT/code_snippets/`
- **实施步骤**：
  - 在入口函数声明 `TPipe pipe;`；
  - 修改类的 `Init` 接口，改为 `Init(..., TPipe* pipeIn)`。

## 6. 约束与副作用 (Constraints)
- **开发习惯**：需要改变现有的面向对象封装模式。
- **指针安全**：需确保指针引用的有效性。
- **适用场景**：`O.General`, `S.ScalarBound`。

## 7. 验证逻辑 (Verification)
- **验证原则**：标量指令周期的结构性改善。
- **推荐验证项**：
  - `aic_scalar_ratio`：期望下降；
  - `Task Duration(us)`：在逻辑复杂的算子中期望有小幅提升。
- **验证方法**：检查汇编代码，确认标量寄存器的溢出（Spill）情况减少。

## 标签
- Domain: `U.CPU`, `O.General`
- Symptom: `S.LowComputeUtil`, `S.ScalarBound`
- Context: `C.Arch.910B`
