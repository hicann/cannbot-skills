Detected downgrade

- 当前检测到的降级类型：
- 触发原因：

Why this is a downgrade

- 原始实现中需要保留的一比一抽象层或多路径调度：
- 若继续当前简化方案，会丢失的能力：

Option A: preserve one-to-one abstraction/kernel split

- 实现方式：
- 迁移成本：
- 迁移难度：
- 迁移收益：

Option B: downgraded fallback

- 实现方式：
- 会简化掉的抽象层或调度路径：
- 短期收益：
- 长期代价：

Impact on reuse

- 对当前算子的影响：
- 对兄弟算子复用的影响：

Impact on validation

- 对验证范围的影响：
- 对结果可信度的影响：

Recovery path

- 若先接受降级，后续如何恢复到一比一方案：

Please choose

- 请选择 A 或 B；在你明确选择之前，我不会开始实现代码。
