# quant_flash_attn 最佳实践（arch35 / Ascend950PR / DAV_3510）

> 量化 GQA FlashAttention，三套 dtype：FP8(E4M3) / HIF8 / MXFP8。源算子：`ops-transformer/attention/quant_flash_attn`。

## S1 选型

- 基础形态：GQA。trait：**量化**（Q/KV dtype ∉ {fp16,bf16}），按 dtype 分三套独立 kernel/tiling。
- quantMode 编码（[quant_flash_attn_template_tiling_key.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_template_tiling_key.h)）：0=HIF8_FP32、1=MXFP8_PREFILL、2=MXFP8_DECODE、6=GQA_FP8_FULLQUANT。注册优先级 mxfp8=1、fp8=2、hif8=3。

## 三套量化路线的本质差异（核心蒸馏）

| 维度 | FP8 GQA Fullquant | HIF8 | MXFP8 |
|---|---|---|---|
| 量化粒度 | Q/K per-token-per-head、V per-head、P per-tensor | 全 per-tensor 标量 | Q/K/V/P per-block E8M0（2 的幂） |
| **AntiQuant 位置** | **全在 AIV**：cube 纯 FP8 matmul，descale 融进 Vec1 softmax / Vec2 乘 V scale | **全在 AIV**：cube 传 dummyScale，Vec1 乘 descaleQK、Vec2 乘 vScale | **全在 AIC 硬件原生**：`MatmulFullMX` 把 E8M0 scale 作为 GEMM 输入，AIV 零 dequant |
| scale 驻留层 | **UB**（vec 侧消费）：Q scale 沿 S1 轴拷 UB，K scale 沿 S2 轴双缓冲拷 UB | 无逐 token scale buffer，3 个标量 | **L1**（cube 侧消费）：Q/K scale 沿 D 轴 per-32 组、V scale 沿 S2 轴 per-64 组驻留 L1 |
| KV 布局 | 强制 PA_BnNBsD | BSND/SBND/BNSD/TND/NTD | NO_PA/PA_BBND/PA_BNBD/PA_NZ |
| FlashDecode | 无 | 无 | **有**（quantMode=2） |

**蒸馏规则**：
1. **scale 跟随消费方驻留**：AIV 反量化 → scale 进 UB；cube 硬件反量化 → scale 进 L1 与数据同层。不用 FB/BT。
2. **I5 轴对齐实测**：Q/K scale 沿 reduction 轴（D 轴 per-32 组）；**V scale 沿 S_k 轴（per-64 组，不是 D）**——V 在 [B,Sk,Hkv,D] 下 innermost 是 D，但 P·V 的 reduction 轴是 S_k。
3. **MXFP8 的 P scale 反向回传**：AIV Vec1 在 UB 生成 per-32 E8M0 scale（`pScaleByteOffset=16640`），写回 **L1P 尾部**（`pScaleL1Offset=mBaseSize*s2BaseSize`）供 cube MM2 消费；DN 路径 fake scale 填 `0x7f7f`（E8M0 中 2^0=1.0 恒等编码）。

## S3 基本块（量化对 tile 的修正）

来源：[qfa_adjust_sinner_souter.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/quant_flash_attn/op_host/qfa_adjust_sinner_souter.h)

- **mBaseSize 恒 64**（kernel 生效值 = 64×r_cv = 128，与非量化一致）。
- **s2BaseSize：FP8/HIF8 → 256，MXFP8 → 512**，D=256 时回落 256。
- 对照 fp16 FIA 默认 s2BaseSize=128：**量化只放大 S2 方向基本块，不动 mBaseSize/D tile**。原因：FP8 字节宽减半 → AI_HBM 翻倍、K/V 驻留 L1 容量压力减半；MXFP8 dequant 在 cube 内免费 → 需要更大 S2 tile 喂饱 Cube 并摊薄 AIV softmax。
- MXFP8 kernel 内再按 `s2SplitSize=(dBase==256)?128:256` 把 512 切 subLoop，奇偶交替用 `l1PBuffers_.Get()/GetPre()` 控制 L0 占用。

## S4 Buffer 清单

| Buffer | FP8/HIF8 | MXFP8 |
|---|---|---|
| L1 | Q/K/V + P（`BuffersPolicy3buff<L1>`），L1 总预算恒 `Init(512KB)` | 同左 + Q/K/V scale 随数据驻留 |
| UB | bmm1 ×2（`BuffersPolicyDB`）、bmm2 ×1、stage1OutQue 各 16640B；FP8 另有 Q/K scale UB 缓冲（K scale ×2 各 `(s2BaseSize+8)*4*4`B，4 份解 bank 冲突） | stage1OutQue 各 16896=16640+256B（P scale 区） |
| L0 | L0A 恒 DB（32KB）；**L0B 在 s2BaseSize=256 且 D>128 时退单 buffer，否则 DB**；L0C 满足 `mBaseSize*s2BaseSize*4 ≤ L0C/4 && mBaseSize*dVBase*4 ≤ L0C/4` 用 4buff 否则 DB | 同左，`mm2LeftSize = mBaseSize*s2BaseSize*(1+1/32)`（P 数据+E8M0 scale）计入 L1P 首 policy |

## S5 流水编排

- 三套 kernel 调度同构（[quant_flash_attn_kernel_fp8.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_kernel_fp8.h) 等）：`PRELOAD_N=2`、`PRELOAD_TASK_CACHE_SIZE=3`，本轮 MM1/Vec1，**MM2/Vec2 延迟 2 个 task**（`loop≥2` 才跑 runInfoNegN）。
- MXFP8 的 MM1/Vec1 内部 subLoop（`c1v1Loop=CeilDiv(actS2, s2SplitSize)`）。
- HIF8/MXFP8 的 bmm2 结果另经 `BuffersPolicy3buff<GM>` 周转。

## S6 负载均衡

- 仅 MXFP8 支持 FlashDecode（[quant_flash_attn_block_vec_flashdecode.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_block_vec_flashdecode.h)，631 行）：
  - **AIV-only 归约**：只读各核 FA 写到 workspace 的 accumOut/lseSum/lseMax 合并，不碰 matmul；主循环 CopyLseIn→ComputeScaleValue(VF)→CopyAccumOutIn+ReduceFinalRes_VF→Cast→CopyOut。
  - 常量：`fdBalanceMBaseSize=8`、`FD_LSE_BUF_BYTES=6144`、总 UB `5*6144+4*FD_MM2_BUF_BYTES+4*256`。
  - **同步改用硬件信号量** `Mutex::Lock/Unlock<PIPE_V/MTE2/MTE3>(flag)`（flag 取 8/9/10/11/2/4，避开 TPipe AllocEventID 低段 id），不用 eventID。
  - **UB 分时复用**：FD 区落 FA vec block 瞬态区，`static_assert(FdTotalUbSize <= FaTransientUbSize)` 编译期校验。
  - 与 FA 侧 bit 级一致：`CalcMinCheckValueVF` 与 FA max 的 scaleValue×ln2+Truncate(CAST_CEIL) 量化对齐。
- FP8 GQA / HIF8 无 FD（tiling `flashDecodeFlag_=false`）。

## S7 自检结论

- I5 轴对齐：Q/K 沿 D、**V 沿 S_k**、P 由 AIV 生成回传 L1——全部满足，无量化轴错位。
- L0A 恒 pingpong；L0B 单 buffer 仅出现在 s2BaseSize=256 且 D>128 的容量死角（显式标注退化）；L0C 按容量 4buff/DB 静态分档。
- 基本块编译期常量：mBaseSize 恒 64，s2BaseSize 按 dtype/D 静态分档，无运行时动态块。
- 量化 AI 修正已入基本块：FP8 s2BaseSize 256、MXFP8 s2BaseSize 512。

## 平台依赖

`MatmulFullMX`（MXFP8 硬件 dequant，arch35 特有）、HIFLOAT8 dtype、KERNEL_TYPE_MIX_AIC_1_2、FixPipe、硬件信号量 Mutex——均 DAV_3510 口径。
