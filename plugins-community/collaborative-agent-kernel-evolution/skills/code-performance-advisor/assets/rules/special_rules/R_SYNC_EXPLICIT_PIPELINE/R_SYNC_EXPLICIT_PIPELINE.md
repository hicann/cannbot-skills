# 规则名称：显式流水线同步控制

## 1. 需求场景 (Requirement)
- **业务背景**：复杂算子涉及多个执行单元（Scalar/Vector/MTE/Cube）协同工作，隐式同步机制可能无法保证数据依赖的正确性，导致脏读、数据竞争等问题。
- **形状/数据类型上下文**：算子涉及跨单元数据依赖，特别是 Scalar 单元计算索引后 Vector 单元使用，或 MTE 搬运后立即计算的场景。

## 2. 模式描述 (Pattern)
- **优化思路**：在关键数据依赖点插入显式同步指令（`PipeBarrier`, `SetFlag`, `WaitFlag`），确保前一阶段操作完成后再执行后续操作。
- **目标**：保证复杂流水线场景下的数据一致性，避免流水线冒险（Data Hazard）。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：显式同步会引入流水线气泡（Bubble），但能避免错误的数据读取和计算，保证功能正确性。
- **事实桥接**：
  - 数据依赖 -> 必须同步 -> 避免脏读/竞争
  - 合理同步点 -> 最小化气泡 -> 平衡正确性与性能

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Correctness`（功能正确性，观察是否有随机性错误）
  - `PipeStall`（流水线停顿信号）
  - `Op Complexity`（算子复杂度，多单元协同场景）
- **如何解读（定性）**：
  - 代码涉及 Scalar 计算索引后 Vector 使用的场景
  - MTE 搬运数据后立即进行计算
  - 算子运行结果不稳定或出现随机性错误
  - 使用标签 `S.PipeStall` 标注流水线问题

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d), `case2/` (batch_norm_v3)
- **实施步骤（示例性）**：
  1. 识别数据依赖点（如 Scalar 索引计算后 Vector 使用）
  2. 在依赖点插入对应的同步指令：
     - `PipeBarrier<PIPE_V>()`：Vector 单元内部同步
     - `PipeBarrier<PIPE_ALL>()`：全局同步
     - Scalar 到 Vector 同步：
       ```cpp
       event_t eventIDSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
       SetFlag<HardEvent::S_V>(eventIDSToV);
       WaitFlag<HardEvent::S_V>(eventIDSToV);
       ```
     - MTE 到 Vector 同步：
       ```cpp
       event_t eventIDMteToV = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
       SetFlag<HardEvent::MTE2_V>(eventIDMteToV);
       WaitFlag<HardEvent::MTE2_V>(eventIDMteToV);
       ```
  3. 在每个 Vector 指令之间插入 `PipeBarrier<PIPE_V>()` 确保数据就绪
  4. 使用 Profiling 工具验证同步是否生效

## 6. 约束与副作用 (Constraints)
- **性能开销**：同步指令会引入流水线气泡，可能增加少量延迟（通常 < 5%）
- **代码复杂度**：需要深入理解硬件流水线和数据依赖关系
- **适用场景**：`U.Mix`（多单元协同），`S.PipeStall`（流水线停顿）
- **不适用场景**：简单的单单元操作，隐式同步已足够的场景

## 7. 验证逻辑 (Verification)
- **验证原则**：功能正确性优先，性能损失可控
- **推荐验证项**：
  - `Correctness`：算子运行结果稳定且正确
  - `Task Duration(us)`：性能损失 < 5%（相比无同步版本）
  - `Random Errors`：消除随机性错误和数据竞争
- **验证方法**：
  - 对比有无显式同步的运行结果，确认功能正确性
  - 使用 PyTorch Golden Reference 验证数值正确性
  - 多次运行确认结果的稳定性和可重复性

## 标签
- Domain: `U.Mix`, `U.Vector`
- Symptom: `S.PipeStall`
- Context: `C.Arch.910B`, `C.Arch.910D`
