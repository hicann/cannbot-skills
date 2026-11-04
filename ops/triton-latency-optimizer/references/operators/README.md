# 算子特定优化经验（本地保留区）

本目录的优化经验文档逐步迁入 cannbot-knowledge 知识库
（`ops/triton/optimizations/operator_optimization/`，索引见 `references/Index.md` 的 `KB:` 路径）。

当前仅保留 **KB 暂未收录** 的 3 张优化期卡，待知识库侧补卡后删除：

| 文件 | 对应算子类型 |
|------|-------------|
| [grouped-matmul-tile-merge.md](grouped-matmul-tile-merge.md) | Grouped MatMul/MoE 专家分组矩阵乘 expert 对齐 tile 合并 |
| [lightning-indexer.md](lightning-indexer.md) | LightningIndexer（topk-select）：profile 归因修正 + 变体探查 + tile 放大 |
| [mla-paged-attention-optimization.md](mla-paged-attention-optimization.md) | MLA 类算子瓶颈判别、仿射 store 修复（`BLOCK_QO=1`）、`P_SPLIT` 精度补偿 |

**使用方式**：与 KB 卡一致——由 `references/Index.md` 的优化点索引按命中条件引用（表中以「KB 暂未收录，保留本地」标注）。

SparseFlashAttention 的优化经验通过 `references/Index.md` 按需读取知识卡 `KB:ops/triton/optimizations/operator_optimization/sparse_flash_attention_optimization.md`；设计与编码阶段命中块稀疏类别时，可参考块稀疏约束卡的 L1.10。知识库定位规则见 SKILL.md。
