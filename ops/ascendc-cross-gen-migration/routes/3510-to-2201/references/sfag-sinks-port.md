# SFAG（sparse_flash_attention_grad）sinks 反向移植指引

> 实测教训：SFAG 的 sinks 反向路径连续 3 轮失败（dq matched≈0.005、dk≈0.03、d_sinks≈0.25，bf16 还出现 AICore 507015），根因是**自行重写公式**而不是复用同族 arch22 实现。

## 1. 参考实现（MUST 先读）

仓库内已有可用的 arch22 反向 + sinks 实现，直接对照移植，不要从 arch35 重写：

| 文件 | 用途 |
|---|---|
| `attention/sparse_flash_mla_grad/op_kernel/arch22/basic_modules/sparse_flash_mla_grad_vec_op.h` | `CalDsinks`、`CopyOutDsinks`、dq/dk 后处理 |
| `attention/sparse_flash_mla_grad/op_kernel/arch22/post.h` | 反向 post 归约与输出 |
| `attention/sparse_flash_mla_grad/op_kernel/arch22/smlag_basic.h` | 主流程与 workspace 布局 |
| `attention/sparse_flash_mla_grad/op_host/arch22/` | tiling/注册样板 |

## 2. 必须照抄的三段逻辑

1. **sink 概率并入分母**
   `p_sink = exp(sink - row_max) / (exp_sum + exp(sink - row_max))`，dq/dk 的 ds 计算必须使用含 sink 的归一化分母，否则 dq/dk 整体偏。
2. **dSinks 公式**
   `d_sinks[g] = -Σ_i ds_i * p_sink_i`（参考 `CalDsinks`：`Mul` → `ReduceSum` → `Muls(-1)`）。
3. **跨核累加**
   `d_sinks`、`dq`、`dk` 多核写同一 GM 时必须 `SetAtomicAdd<float>()` → `DataCopyPad` → `SetAtomicNone()`（参考 `CopyOutDsinks`）；否则后写覆盖先写，表现为随机大面积错误。

## 3. 验证底线

- 逐项统计 `dq / dk / dv / dq_rope / dk_rope / d_sinks` 的 matched_ratio，全部 ≥0.99；
- bf16 用例不得出现 `507015`/`aicore exception`（通常是 workspace 越界或事件未配对）；
- 仅 `getws=0/run=0` 不能作为通过结论。

## 4. 已知反例

- 只加 host 门禁、白名单、aclnn 包装：数值不变，dq/dk 仍错；
- 只改 tiling：d_sinks 仍为 0.25；
- 用 arch35 公式直接改写：bf16 触发 AICore 异常。


## 5. 参考实现的代码级要点（已提取，按此实现）

### 5.1 sink 概率（前向统计参与反向）

```cpp
// sink 原值广播到 8 lane，再与 lse 相减后取 exp
Brcb(sinkTensor, sinkTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
Sub(sinkTensor, sinkTensor, lseTensor, processM * 8);
Exp(sinkTensor, sinkTensor, processM * 8);
// p_sink_i = exp(sink - lse_i)，lse 为前向 log-sum-exp（含 sink 项）
```

### 5.2 ds 的修正（含 row-sum 项）

```cpp
Mul(dSinkSumTensor, dPTensor, pTensor, dataSize);   // ds = dP * p
CalSub(dPTensor, rowSumOutTensor, processM, actualSelS2Align); // dP -= Σ dP*p
Mul(pTensor, pTensor, dPTensor, dataSize);          // ds 修正
```

### 5.3 dSinks 公式

```cpp
// 对每行：ds * p_sink 求和
Mul(dSinkSum[i*align + loop*64], dSinkSum[i*align + loop*64], sinkTensor[i*BLOCK_FP32], mask, 1, ...);
ReduceSum(sinkTensor[i*BLOCK_FP32], dSinkSum[i*align], sinkTmpBuf, curActualRight);
// 取负并累加到该 head 的 dSinkTensor
Muls(sinkTensor, sinkTensor, -1.0f, processM * BLOCK_FP32);
Add(dSinkTensor[mOffset * BLOCK_FP32], dSinkTensor[mOffset * BLOCK_FP32], sinkTensor, processM * BLOCK_FP32);
```

即：`d_sinks[g] = -Σ_i Σ_j ds_ij * p_sink_i`。

### 5.4 跨核写出（必须 atomic）

```cpp
SetAtomicAdd<float>();
DataCopyPad(dSinksGm, dSinkTensor, {static_cast<uint16_t>(dimG), sizeof(float), 0, 0});
SetAtomicNone();
```

`dq/dk` 若由多核写同一 GM，同样必须 `SetAtomicAdd<float>()`；否则后写覆盖先写，表现为 matched_ratio 接近 0 的随机错误。

### 5.5 workspace 与 post

- dSinks 在非确定性路径用 `AtomicAdd` 直接累加；确定性路径写 `dSinkWorkspaceGm`，由 post 阶段归约；
- `dq/dk` 的 NZ/ND 转换与 `NZVecClc` 后处理在参考 `arch22/post.h` 中已有，直接复用其 workspace 偏移与对齐规则；
- bf16 下出现 `507015/aicore exception` 时，优先检查 workspace 偏移（`dSinkWorkSpaceOffset`）、事件配对与 UB 越界。

### 5.6 最小验证脚本要求

- 逐输出 `dq/dk/dv/dq_rope/dk_rope/d_sinks` 计算 matched_ratio；
- 覆盖 `sinks=on/off`、`bf16/fp16`、`BSND/TND`；
- 输出 `RESULT PASS/FAIL` 与 max_abs，写入精度报告。
