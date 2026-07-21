# ascend-call-generation 参考示例

本目录收录已生成的 AscendC 工程脚手架样例，供本 skill 生成 pybind / CMake / host 代码时对照。

每个算子子目录包含一组样例文件：

- `<op>_custom.py` — 算子的 Python 调用封装
- `<op>.cpp` — pybind 桥接代码
- `<op>_project.json` — `gen_project.py` 使用的工程配置

已收录算子：`average_pooling2d/`、`mse_loss/`、`layer_norm/`、`matmul/`、`sum_reduction_over_a_dimension/`。
