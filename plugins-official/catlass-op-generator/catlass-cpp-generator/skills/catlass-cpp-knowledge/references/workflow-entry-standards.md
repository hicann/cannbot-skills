# Workflow 入库标准

- `type` 固定为 `workflow`，文件直接位于 `knowledge/workflow/`。
- `consumers` 必须列出实际读取该知识的 `catlass-cpp-*` skill。
- 只保存可跨算子复用的五阶段流程、设计、开发和验证规则。
- family 特有的算法、公式或实现模式放入 `operator/<family>/`。
- 同一规则只保留一个权威文件，由多个 skill 通过 knowledge path 复用。
