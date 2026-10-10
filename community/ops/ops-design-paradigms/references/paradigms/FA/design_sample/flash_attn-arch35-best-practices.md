# flash_attn 最佳实践（arch35 / Ascend950PR / DAV_3510）

> 非量化 GQA FlashAttention，FP16/BF16 only。源算子：`ops-transformer/attention/flash_attn`。
> 芯片口径：L1 512KB、L0A/L0B 64KB、L0C 256KB、UB 248KB/AIV、Cube:Vector=1:2。

## S1 选型

- 基础形态：**GQA**（M = Sq×G 合轴，I2），MHA 为 G=1 退化。
- trait：无（`flash_attn_template_tiling_key.h` 文件头明确"非量化，仅FP16/BF16"，`NoQuantMode=31`）。
- Feature：causal / attenMask / sinks（kernel 已实现、接口未开放）/ softmax_lse。

## S2 计算流

标准 C1→V1→C2→V2。arch35 特有：
- L0C→UB 走 **FixPipe 双目标写**：ND `dualDstCtl=1`、DN `dualDstCtl=2`，S/O_tile 从 L0C 直接按 M 维对半写入 2 个 AIV 的 UB，不落 GM。
- P 在 UB 侧 cast FP16 并**排成 NZ** 再经 UB→L1（仅 NZ 通路）供 C2。

**ND vs DN 双模板**（[flash_attn_tiling.cpp](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_host/arch35/flash_attn_tiling.cpp) `UpdateTilingKeyTemplateId`）：
- ND（templateId=0）：全功能通用路径，支持 attenMask、全部 config。
- DN（templateId=1）：性能特化，仅 `!hasAttenMask && config∈{0,2,6}`（mBaseSize=64 且 D≤128 或 192/128）启用；BMM1 转置计算（K 作左矩阵 Q 作右矩阵），省 mask copyIn。
- 规则：**功能全集走 ND，热点无 mask 场景叠加 DN，tilingKey 静态分流**。

## S3 基本块（编译期常量，8 档 config）

来源：[fa_adjust_sinner_souter.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_host/fa_adjust_sinner_souter.h)、[flash_attn_template_tiling_key.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_template_tiling_key.h)

| config | mBaseSize×s2BaseSize | D(qk,v) |
|---|---|---|
| 0/1 | 64×128 / 32×256 | (64,64) |
| 2/3 | 64×128 / 32×256 | (128,128)，**D=72 复用此模板**（L1 pad 72→80 分形对齐，Mmad K=72 精确） |
| 4/5 | 64×128 / 32×256 | (256,256) |
| 6/7 | 64×128 / 32×256 | (192,128) MLA 形态 |

- 默认 **64×128**；仅当 vHeadDim≤128 且 maxSeqQ≤64、maxSeqKv>128 等 decode 短行条件时切 **32×256**——按 shape 属性静态分流，无运行时动态块。
- kernel 侧实际生效 `mBaseSize_effective = mBaseSize × r_cv`（64→128，32→64），s2BaseSize 不变。

## S4 Buffer 清单（实测份数）

来源：[flash_attn_block_cube_nd.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_block_cube_nd.h)、[flash_attn_block_vec_nd.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_block_vec_nd.h)

| Buffer | 层 | dtype/格式 | 份数 |
|---|---|---|---|
| Q | L1 | INPUT_T/NZ | 2 |
| K/V 共享槽池 | L1 | INPUT_T/NZ | **4×64KB**（config=5 时 2×128KB `L1_KV_LARGE_BUF`） |
| P | L1 | INPUT_T/NZ | 3 |
| L0A / L0B / L0C | L0 | — | 2 / 2( config=5 退 1) / **4×64KB** |
| mmRes（S 与 O_tile **同槽复用**） | UB | FP32/ND | 4（D≤128）/ 2（D=256） |
| vec1Res（P） | UB | FP16/NZ | 2（每份 +1 行错 bank） |
| vec2Res（O_acc） | UB | FP32/ND | 1 |
| softmaxSum/Max/Exp | UB | FP32 | 各 3，按 `loop%3` 轮转 |
| maskBuffers | UB | uint8 | 2×8KB（仅 ND 有 mask） |

- UB 布局 `static_assert(sizeof(UbLayout) ≤ 248K)` 编译期兜底。
- S/O_tile 同槽复用由单调计数器 `mmResBufId_=(id+1)%4` 错开分配。

## S5 流水编排

来源：[flash_attn_kernel_nd.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_kernel_nd.h)

- **任务级 2 级 preload**：`PRELOAD_N=2`，本轮跑 MM1/Vec1，`loop≥2` 时跑 2 轮前的 MM2/Vec2（MM1(j) ∥ MM2(j−2)）。
- Cube 内部 MTE2→MTE1→M→FIX 硬件流水，靠 buffer 多份轮转。
- 同步：跨核 flag `CROSSCORE_MM_0~3`（S/O_tile）、`CROSSCORE_L1P_0~2`（P），双 AIV 用 `AIV0_AIV1_OFFSET=16` 双路 flag；FD 段前后 `SyncAll()`。

## S6 负载均衡

- 任务维度：**B × N2 × gS1 × s2** 四维，metadata 由 AICPU 算子 `flash_attn_metadata` 预生成（每 AIC 16 字段：bN2/gS1O/s2O Start/End + firstFdDataWsPos），kernel 不现场算均衡。
- **SectionStreamK**：按 L2 96MB 预算切 section，成本模型 `6*ceil(m/16)+10*ceil(s2/64)`；`maxCore=min(36, 块数)` 截断核数（核数运行时输入，禁止硬编码）。
- **FlashDecode**（[flash_attn_block_vec_flashdecode.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_block_vec_flashdecode.h)）：`fdBalanceMBaseSize=8` 行一组；`preLoadNum_=2` 双 buffer 预取 accumOut；FD 区与 FA 区 **UB 分时复用**（FD base=FA mmRes 区大小）；`ComputeScaleValue` 归约各 split 的 m/ℓ，`ReduceFinalRes_VF` 加权累加除 ℓ_global。FA 段按 AIC 分核、FD 段按 AIV 分核。
- causal：无 zigzag，`CalcCurS2StartEndWithSparse` token→block 窗口换算整块跳过。

## S7 自检结论

- K/V 槽池 4 ≥ r+1；状态 buffer task 内恒 1；L0A/L0C pingpong，L0B 仅 config=5（s2BaseSize=256 且 D>128）容量所迫退单 buffer（显式标注的退化，非设计目标）。
- UB 按每 AIV 校验：ND `(actM+1)/2` 行、DN `align32(actM)/2` 行；L1/L0 按全量 mBaseSize 校验。
- I4：split-KV 配 AIV 侧 log-sum-exp 归约；I5 不适用（无量化）。
- 数值：RowInvalid 行（max=−inf）输出清零、LSE 置 3e+99；mask 用有限负数。

## 平台依赖

FixPipe 双目标写（dualDstCtl）、KERNEL_TYPE_MIX_AIC_1_2、L2 96MB 预算切 section——均为 arch35/950PR 实测口径；核数 28/32 运行时查 `PlatformAscendC`。
