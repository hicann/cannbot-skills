# ONNX Plugin Demo

本目录演示如何为自定义算子提供 ONNX 图模式插件。

- 这是一个**演示映射**：`AddExample` -> `ai.onnx::7::Add`。
- 实际使用时，请将 `add_example_onnx_plugin.cpp` 中的 `OriginOpType` 替换为目标算子对应的标准 `ai.onnx` OpType 与 opset（例如 `ai.onnx::11::Add`）。
- 插件随算子工程一起构建，并安装到 vendor 的 `framework/onnx` 目录（`packages/vendors/<vendor>/framework/onnx`）。
- 本模板不生成任何 workflow 相关代码或配置，仅提供插件注册与构建安装示例。
