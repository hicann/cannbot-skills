# 案例：SFAG（sparse_flash_attention_grad）sinks 下沉 A2

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/sparse_flash_attention_grad` |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 测试 | ACLNN 直调 `aclnnSparseFlashAttentionGradV2` |

## 基线现象

- 不带 sinks：`getws=0, run=0`（A2 基础路径正常）；
- 带 sinks：`The sinks input of aclnnSparseFlashAttentionGradV2 is only supported on Ascend950.`

## 改动点

- aclnn V2 放开 sinks 门禁；
- arch22 tiling 传递 sinks（含 sink 形状/头数校验）；
- arch22 kernel 消费 sinks，计算并输出 dSinks；
- 同步文档与（可选）UT。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| 带 sinks 直调 | PASS（`getws=0, run=0`） |
| 不带 sinks 回归 | PASS（`getws=0, run=0`） |
| 改动规模 | 9 文件 +313/−58 |
