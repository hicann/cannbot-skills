# 范式资料维护

每个范式提供独立的场景判定、算法模型和实现约束。现有结构参考 [Broadcast](broadcast/patterns.md) 与 [Reduction](reduction/patterns.md)。

## 组织方式

- `patterns.md`：场景判定、语言选择和主题索引。
- `generic/`：语言中立的算法、形状与轴处理、切分、资源和同步模型。
- `adapters/<lang>/`：具体语言的能力映射、API 约束、实现方法与可选参考源码。
- `common/`：跨范式的公共知识。

只创建有实际内容且能从索引到达的文档。文档按主题拆分；范式与场景名称使用真实名称。

## 注册与校验

1. 在 `routes.yaml` 的 `routes.paradigms` 中登记范式名称和入口路径；入口相对于本 Skill 根目录。
2. 在入口中列出已支持的语言。没有适配层时提供通用算法并标明待确认的语言能力。
3. 有参考实现时在 `example-routes.yaml` 按范式和语言登记其真实路径。
4. 检查索引目标存在、内部链接可达，且未知语言不会退回其他语言的实现。
5. 从本 Skill 根目录运行 `python3 -m unittest discover -s tests -v` 验证示例解析和索引行为。
