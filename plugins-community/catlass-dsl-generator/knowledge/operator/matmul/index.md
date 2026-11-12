# Matmul 类算子

## 结构参考（catlass Python DSL 样例）

> 整体结构优先：先选阶段序列、数据流、任务映射、同步方式都接近的主参考；主参考是基线
> 不是约束，偏离写清理由。没有足够契合的主参考时，可以组合多个分步样例，但要显式设计
> 整体结构并照常跑 skeleton DoD。

- Cube-only：`tla/kernel/matmul/basic_mmad/basic_matmul.py`
- Cube-Vector：`experimental/kernel/matmul/matmul_softmax_row_stats/matmul_softmax_row_stats.py`
- Cube-Vector-Cube：`experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py`
- Vector-only：`tla/kernel/vec/basic_vadd/basic_vadd.py`、`dsl_tutorials/vector_simd_ops/`、
  `dsl_tutorials/vector_simt_ops/`

## 知识条目

- [Basic MMAD](basic-mmad.md)
- [Mixed、atomic add 与 mutex 变体](mixed-and-sync-variants.md)
- [Batched 与 Grouped Matmul](advanced-matmul.md)
- [MatMul](mat-mul.md)
- [QuantBatchMatmul](quant-batch-matmul.md)
- [FusedMatMul](fused-mat-mul.md)
