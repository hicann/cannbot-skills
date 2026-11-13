# 案例：SFA（sparse_flash_attention）LSE + PA_BSND 下沉 A2

> 实验记录，非通用规范。环境 910B3 + CANN 9.2.0 + torch_npu 2.7.1。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba5162411c919e15fbabb28298e8b420c11ad905` |
| 算子 | `attention/sparse_flash_attention` |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 测试 | 仓库 pytest（`PARAMSET_FILE` 切换参数集） |

## 门禁（实测）

`op_host/sparse_flash_attention_tiling.cpp`（约 2030 行）：

```cpp
OP_CHECK_IF(*opParamInfo_.returnSoftmaxLse && kvLayout_ == SFALayout::PA_BSND && !isA5_,
            OP_LOGE(... "only supported on Ascend 950PR/Ascend 950DT"), ...);
```

## 改动点

- 删除/放宽 A5 专属门禁（arch22 已有 LSE 实现，无需重写 kernel）；
- arch22 UT 改为合法 shape 下预期成功；
- README 与 aclnn/ascendir 文档同步 A2 支持。

## 收益数据

| 项 | 结果 |
|---|---|
| 构建 / 安装 | PASS / PASS |
| 可见用例（PA_BSND 非 LSE） | PASS（约 13 s） |
| 隐藏用例（PA_BSND + LSE，2 例） | PASS（约 21 s） |
| 改动规模 | 6–8 文件，约 +58/−20 |
| 预算优化收益 | 步数 305 → 110；总 tokens 84.5M → 9.85M；墙钟 59.8 → 33.7 min |
