# 案例：GBSA 非因果 / 隐式残块下沉 A2

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/generic_block_sparse_attention`（+ `_metadata`） |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |

## 门禁（实测）

| 门禁 | 文件 | 报错原文 |
|---|---|---|
| mask | `op_host/generic_block_sparse_attention_tiling.cpp` | `maskType supports 1, and 0 on A5 non-quant, but got 0.` |
| residual | 同上 | `residual_block_mode=1 is unsupported; mode 1 only supports A5 non-quant currently.` |
| metadata | `..._metadata/op_host/op_api/aclnn_..._metadata.cpp` | 同样的 mask/residual 校验 |

## 改动点

- host tiling 放开非量化 A2 的 `mask_type=0` / `residual_block_mode=1`；
- metadata 同步放开；
- arch22 新增 18 个 tilingKey（FP16/BF16 × softmax 精度 × LSE × mask/residual）；
- arch22 kernel 按 `isCausal` / `hasResidualBlock` 改可见范围与尾块（topK+1、FD 块范围+1）。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| 因果基线 | PASS，`max_abs≈1.6e-5`，`matched_ratio=1.0` |
| 非因果 fp16/bf16 | PASS，`max_abs≈1.1e-5` |
| 隐式尾块 | PASS，`max_abs≈1.2e-5`（实现错误时为 `ratio≈0.49`） |
| 改动规模 | 13 文件 +181/−53 |
