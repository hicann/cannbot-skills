---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 DataCopyPad（DataCopyExtParams）GM 侧 srcStride/dstStride 单位是 1 字节，不是 32B 块——A3 习惯的 /32 写法在 A5 上静默错位读取"
description: "V351 DataCopyPad（DataCopyExtParams）GM 侧 srcStride/dstStride 单位是 1 字节，不是 32B 块——A3 习惯的 /32 写法在 A5 上静默错位读取. Applies: soc=Ascend950DT/950PR(V351/arch35,dav_3510); cann=9.2.0; mode=port_a3_to_a5; op_class=所有用 DataCopyPad + DataCopyExtParams 多块（blockCount>1）跨步读写的 kernel. Provenance: 42_CoTAttention fp32 全灭根因（2026-08-27，kimi 主线会话 dump 内部一致性分析 + FFT/逐行重建精确定位 + 卡 3 修复实测）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-datacopypad-extparams-stride-byte-unit-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, datacopypad, stride, byte-unit, misaligned-read]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# DataCopyPad ExtParams 的 GM 侧 stride 单位 = 1 字节（dav_3510 实现实证）；A3 的 "/32"（32B 块单位）写法在 A5 上静默读错行

## 事实（42_CoTAttention act2，Ascend950DT + CANN 9.2.0，卡 3 实测）
- CANN 9.2.0 dav_3510 实现（`asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h` `CopyGmToUbufAlignV2` / `CopyUbufToGmAlignV2`）：`unitOfBytes = isDataCopyPad ? 1 : 32`，`srcStride310 = srcStride * unitOfBytes + burstLength` —— **DataCopyPad 的 GM 侧 stride 单位是 1 字节**（普通 DataCopy 才是 32B）。且 3510 上 `DataCopyExtParams.srcStride/dstStride` 类型变成 `int64_t`（其它 arch 是 uint32_t），是语义差异的旁证。
- 事故现场：ReLUCastCore 用 `blockCount=32, blockLen=act*4, srcStride=(mS-act)*4/32` 做 32 行跨步 GM→UB（A3 移植惯性：/32 是 32B 块单位）。设备按 **744 字节**解释 srcStride → 行距变成 blockLen(512B)+744B=1256B=314 浮点（应为 6080）→ **每个 tile 的行 0 正确（不吃 stride）、行 1..31 全部错位读到错误数据**。
- 确定性、可精确重建：`t3T[img, rr, m0+j] == relu(t3AccT).flat[img][m0 + rr*314 + j]` 逐元素 EXACT 吻合（2914 行 × 全 tile 校验 0 失配）。**不是 race、不是时序敏感**——host 每级都有 `aclrtSynchronizeStream` + `ASCEND_LAUNCH_BLOCKING=1` 下照样复现。
- 伪装性极强（为什么活了这么久）：
  - 行 0 永远正确；本例 mid_s=64 的行 32..63 因源区与错位落点同为 padding 零而"恰好干净"，坏行集合呈现诡异的 1..31。
  - 下游 softmax 的行常数抵消特性把巨大的 attMean 偏差吸收到 attP 只剩 3e-4 噪声，最终 out 与"候选算法真值"差 <1e-3——**设备管线内部一致性检查（t3T==relu(t3AccT)? split 守恒? out==k1+attP*v?）是把"设备执行错"与"算法语义错"切开的关键手段**。
  - mS≤128 的形状 stride=0 无 bug → 小 shape 的 fp16/bf16 case 能过，大 shape 全灭。
- 伴随 bug（同 kernel 尾 tile）：GM→UB Pad 紧凑打包后 UB 行距 = `AlignUp(act*4,32)/4` 元素，写回时 `ubO[rr*COT_ACT_TILE_N]` 应为 `ubO[rr*act]`，否则 mS%128≠0 的尾 tile 写回错位（本例被 padding 零掩盖）。

## 动作规则
1. 迁移源码出现 `DataCopyPad` + `DataCopyExtParams` + `blockCount>1`：GM 侧 stride 直接写字节数（`(rowStride - blockElems) * sizeof(T)`），**禁止 /32**；UB→GM 同理（GM 侧 dstStride 也是 1 字节单位）。
2. UB 侧（dstStride for G2U / srcStride for U2G，Normal 模式）仍是 32B 单位——方向不同单位不同，逐个核对 impl 注释。
3. 多块读 + 紧凑打包的写回索引用实际 `act`（打包行距），不要用 TILE 常数。
4. 判别签名：每个 tile 行 0 正确、行 1..N-1 全坏、逐元素可用"行距错值"精确重建吻合 → 就是 stride 单位，不要再往时序/race 方向查。

## 证据
- 内部一致性分析：a5-data/cot-dsh-diag/internal_consistency.py（t3T≠relu(t3AccT) 而 split/softmax/fuse 链路全守恒）；trace_t3T_origin4.py（pitch=314 精确吻合，2914 行 0 失配）；CANN impl 单位实证（`CopyGmToUbufAlignV2`，`unitOfBytes = isDataCopyPad ? 1 : 32`）。
- 修复实测（卡 3 阻塞 case1，修复前后同流程）：t3T 真实列 bad 272625→0；attP max err 3.0e-4→7.7e-5；out 真实列 <1e-3 全清；MERE(vendor 口径) 0.0863→0.00446。修复点：work/kernel/cot_attention_kernels.h ReLUCastCore 两处（srcStride 去 /32 + 写回 rr*act；act1 的同类索引不要动——它逐行 blockCount=1 按 TILE_N 显式放置，本来就对）。
- OOB 变体：错位行距 < 真实行距，mS≲314 的形状末 image 末行块读出缓冲末尾之外 → 507035 device error type 3（42 O5 两次崩溃的统一嫌疑机制）。
- 与 `v351-datacopypad-32b-roundup-oob-001`（im2col 多读）是同算子独立缺陷，两者都修完后 fp32 链路才干净。

## 关联 KB
- `v351-datacopypad-32b-roundup-oob-001`（DataCopyPad 32B 上取整多读——同 API 的另一类边界坑）。
- `v351-pipe-all-tbuf-stale-001`（PB-21 事件配对——同算子 cot_softmax 楔死修复）。
