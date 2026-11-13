# 案例：BSA（block_sparse_attention）BSND layout 下沉 A2

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/block_sparse_attention` |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 测试 | ACLNN 直调 + CPU 参考 |

## 基线现象

- `BSND` 布局被 host 门禁拒绝：`BSND inputLayout is supported only on chip 950`；
- A2 已有 arch22 kernel，但 `__CCE_AICORE__ == 220` 分支只注册 TND/BNSD 的 tilingKey。

## 改动点

- host tiling 放开 BSND（按 soc 分支）；
- tilingKey 增加 BSND 组合，arch22 入口注册新 key；
- arch22 kernel 增加 BSND 取址/步长分支；
- 同步文档与 UT。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| BNSD 回归 | PASS，`corr=1.0`，`max_abs≈5.2e-4` |
| BSND 隐藏用例 | PASS，`corr=1.0`，`max_abs≈5.2e-4` |
| 改动规模 | 7–9 文件，约 +96/−39 |
