# Vector 类算子

## 结构参考（catlass Python DSL 样例）

> 整体结构优先：Cube-Vector / Cube-Vector-Cube 先选整体结构接近的主参考；vector 样例
> 用于补局部机制。主参考是基线不是约束，偏离写清理由；没有足够契合的主参考时，可以
> 组合多个样例，但要显式设计整体结构并照常跑 skeleton DoD。

- Vector-only：`tla/kernel/vec/basic_vadd/basic_vadd.py`、`dsl_tutorials/vector_simd_ops/`、
  `dsl_tutorials/vector_simt_ops/`
- Cube-Vector 的 vector 后处理段：
  `experimental/kernel/matmul/matmul_softmax_row_stats/matmul_softmax_row_stats.py`
- Cube-Vector-Cube 的中间 vector 段：
  `experimental/kernel/matmul/matmul_add_chain/matmul_add_chain.py`

## 知识条目

- [Basic VADD](basic-vadd.md)
- [Vector 操作索引](vector-ops.md)
