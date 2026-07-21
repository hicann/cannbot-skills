# dsl-baseline-generation 参考资料

本目录收录 AscendDSL 生成的参考资料，供本 skill 生成 DSL baseline 时对照。

- `ascend_dsl.py` — AscendDSL 语法/API 参考
- `input_example/` — 输入侧示例（functional PyTorch 等），按算子命名
- `output_example/` — 期望生成的 DSL 产物示例（softmax、layer_norm、matmul、mse_loss、transpose、top_k 等），含对齐/非对齐、单核/多核、UB 复用、conversion/sort 等变体
