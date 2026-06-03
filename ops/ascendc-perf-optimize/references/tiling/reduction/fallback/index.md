# Reduction 归约类 — 兜底算法

> 当前为占位文档，内容待补充。

## 适用算子

ReduceSum, Softmax, LayerNorm, ArgMax

## 待补充内容

- 多核切分策略（沿非归约轴切分）
- UB 切分策略（归约轴分 chunk 处理）
- Buffer 规划
- 分支覆盖（dtype、归约轴大小、尾块处理）

---

> 当前返回：「Reduction Tiling 建模暂未收录。」
