# 规则名称：限制 ShapeInfo 维度，减少栈空间占用

## 1. 需求场景 (Requirement)
- **业务背景**：算子内部逻辑固定，不需要动态通过 `ShapeInfo` 系统获取或设置维度信息。
- **形状/数据类型上下文**：栈空间极其紧张、或属于高频调用的轻量级标量算子。

## 2. 模式描述 (Pattern)
- **优化原理**：通过宏定义 `#define K_MAX_SHAPE_DIM 0` 将 `ShapeInfo` 结构体的预留维度强制归零。
- **目标**：缩减每个 `LocalTensor` / `GlobalTensor` 对象在栈上的物理开销，降低 Scalar 执行流的 Cache 压力。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：默认 8 维的 `ShapeInfo` 在每个张量对象中都会产生冗余的栈空间分配。大量的张量实例会导致栈溢出风险并恶化内存本地性。
- **事实桥接**：
  - 减少栈开销 -> 提升标量指令（Scalar）的访问速度。
  - 降低内存指纹 -> 减少指令 Cache Miss 情况。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_scalar_ratio`（标量占比）
  - `Task Duration(us)`（定性评估）
- **如何解读（定性）**：
  - 判定算子是否为 Scalar Bound 类型；
  - 检查头文件中是否包含无用的维度维护逻辑。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_K_MAX_SHAPE_DIM_STACK/code_snippets/`
- **实施步骤**：
  - 在包含 `kernel_operator.h` 之前添加 `#define K_MAX_SHAPE_DIM 0`；
  - 重新编译并进行功能回归。

## 6. 约束与副作用 (Constraints)
- **功能不可逆**：设置后无法通过张量对象获取 `Shape`, `GetSize` 等动态信息。
- **适用场景**：`S.ScalarBound`, `U.CPU` 模拟执行或轻量级核。
- **宏位置**：必须位于所有头文件包含之首。

## 7. 验证逻辑 (Verification)
- **验证原则**：指令流的微弱抖动改善与端到端稳定性。
- **推荐验证项**：
  - `Task Duration(us)`：在极端小 Shape 下确认性能是否有微弱提升。
- **验证方法**：使用编译器的栈分析工具确认二进制对应的栈消耗。

## 标签
- Domain: `U.CPU`, `O.General`
- Symptom: `S.LowComputeUtil`, `S.ScalarBound`
- Context: `C.Arch.910B`
