# 案例：block_attn_res_prepare 整算子移植 A2

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/block_attn_res_prepare` |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 测试 | ACLNN 直调 `aclnnBlockAttnResPrepare` + 输出 finite 检查 |

## 基线现象

- 文档：Atlas A2 不支持；
- 构建包内没有 `ascend910b` kernel；
- 直调报：`BlockAttnResPrepare only supports Ascend 950 and Ascend 350.`

## 改动点

- `op_host/CMakeLists.txt` 增加 A2 构建注册；
- `block_attn_res_prepare_def.cpp` 增加目标 soc 配置；
- 新增 arch22 kernel/tiling 路径与 tilingKey；
- 同步 README/文档。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| A2 kernel 产物 | `op_impl/.../kernel/ascend910b/block_attn_res_prepare` |
| 直调 smoke | `getws=0, run=0`，`numerator_finite=true` |
| 改动规模 | 18 文件 +73/−30 |
