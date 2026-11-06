# 公式生成接口

已有 DAG 可直接使用，无需额外公式文件。需要生成 contraction 模型时，
在 `definition/` 提供 `formula.yaml` 和 `cube.yaml`，并保持 operator 的资源声明一致。

## 输入

`formula.yaml` 的 `formulas` 项声明 `kind/indices/inputs/output/reduce/expression`。
当前支持两个二维输入、两个输出自由轴、一个 sum 归约轴和乘法表达式。
张量名和轴名由模型选择，输出 dtype 与 accumulation_dtype 分别声明。

`cube.yaml` 指定 `formula_file`，各 stage 指定 `target_stage/formula/schedule/domain/tiles/storage`。
支持的 schedule 值为 `contraction_streaming` 与 `contraction_operand_resident`。
后者另需 `residency: {operand: lhs或rhs, level: L1, reduction_extent: full}`，
且不接受独立 loop_order。完整配置应满足所选模板的字段校验。

## 调用与输出

```text
python -m accel_model.kernel.dag.gen MODEL
python -m accel_model.kernel.dag.gen MODEL --check
```

普通调用会写入 `definition/dag.yaml`；先在模型副本中运行。
`--check` 比较生成内容与现有 DAG。该入口不提供 `--help` 参数。
生成后仍需完整 case 运行与数量、容量和适用数值检查。
候选附带的 `operator_buffers.patch.yaml` 等配置须与该候选其他输入一起核对采纳。

## 限制

物理转置、batch/broadcast、多归约轴及其他不支持组合会报错。
公式被接受不代表支持任意实现结构，也不表示已经完成搜索。
参数搜索和候选比较的输入分别见[DAG 搜索](dag-search.md)与[图搜索](graph-search.md)。
