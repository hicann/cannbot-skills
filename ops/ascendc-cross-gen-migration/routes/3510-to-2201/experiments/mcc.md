# 案例：MaskedCausalConv1d 下沉 A2（线性注意力）

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/masked_causal_conv1d` |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 测试 | ACLNN 直调 + CPU 因果卷积参考 |

## 基线现象

- 文档：Atlas A2 不支持；
- 构建包无 `ascend910b` kernel；
- 直调报：`SoC version ascend910b verification failed. This SoC is not configured through the AddConfig API of the OpDef class.`

## 改动点

- `op_host` 增加 A2 的 AddConfig 注册；
- arch22 kernel/tiling 路径（因果 depthwise 1D 卷积，`x[S,B,H]`、`weight[W,H]`、`mask[B,S]` BOOL）；
- 文档同步。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| A2 kernel 产物 | `op_impl/.../kernel/ascend910b/masked_causal_conv1d` |
| 无 mask 用例 | PASS，`corr≈1.0`，`max_abs≈2.7e-3` |
| 带 mask 用例 | PASS，`corr≈1.0`，`max_abs≈2.7e-3` |
| 改动规模 | 4–9 文件，+31/−4 ～ +83/−9 |
