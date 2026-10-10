# sparse_flash_mla 最佳实践（arch35 / Ascend950PR / DAV_3510）

> MLA + 稀疏双 trait：latent 压缩 KV（kvHeadNum=1, D=512）+ 外部 topk 稀疏索引。源算子：`ops-transformer/attention/sparse_flash_mla`。

## S1 选型

- 基础形态：**MLA**（latent absorption，`n2Size≠1` 直接报错，D_latent 固定 512）。
- trait：**稀疏**（外部 `ori/cmp_sparse_indices` int32 topk 块索引 + `ori/cmp_topk_length` 有效长度裁剪，`TOPK_LIMIT=8192`）。
- 本算子**不做 nope/rope 拆分**（448+64 字段属兄弟算子 mixed_quant_sparse_flash_mla），本体不做量化。
- **5 种模板模式**（[sparse_flash_mla_common.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_common.h) `SMLATemplateMode`）：
  - **CSA（=2）**：cmp_kv + cmp_sparse_indices，压缩 KV 稀疏注意力，cmpRatio∈{1,2,4}；
  - HCA（=1）：cmp_kv 无索引，cmpRatio 必须 128；
  - ORI_SPARSE(=3)/ORI_CMP_SPARSE(=4)：仅 A5，稀疏索引直通；
  - **SWA（=0）**：无压缩分支的滑窗路径，无索引时强制 oriMaskMode=4 且默认 128 滑窗（winLeft=127, winRight=0）。
- mask mode：0=topk/全量、3=RightDownCausal、4=Band 滑窗；A5 上 ori∈{0,3,4}、cmp∈{0,3}。

## S3 基本块（编译期常量）

来源：[sparse_flash_mla_csa_block_cube_arch35.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_csa_block_cube_arch35.h)

- **mBaseSize=64、s2BaseSize=128、dBaseSize=512、dBaseMatmulSize=128、sparseBlockSize=1**。
- D=512 超宽不进基本块常量：cube 侧沿 D 以 128 迭代，Q 沿 D 切半两次载入（`qHalfNum=2`）。
- IS_SPLIT_G 时 `mRealSize = s1RealSize × gSplitSize`（G 维对折进 M 维）。
- paged KV：A5 blockSize∈[1,1024]（`BLOCK_SIZE_LIMIT=1024`）；向量化地址模式要求 blockSize 为 2 的幂。

## S4 Buffer 清单

来源：[smla_common_defs.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/common/smla_common_defs.h)、CSA/SWA kernel Init

| Buffer | 层 | 大小 | 份数 |
|---|---|---|---|
| l1P | L1 | 64×128 each | 2，**必须排 L1 最前**（保证与 vec 地址一致） |
| l1Q | L1 | 16384 elem(32KB) each | 3 |
| l1 K/V 共享槽池 | L1 | 65536 elem(128KB=128×512×2B) each | 3（`rightBufNum=3`） |
| L0A / L0B / L0C | L0 | 16KB / 32KB / 128KB | 2 / 2 / 2（恒 pingpong） |
| bmm2 | UB | 32×512 | 1 |
| bmm1 | UB | 32×128 each | 2 |
| vec UB 固定地址 | UB | `ATTEN_OUT_POP_BUF=184KB`、`LSE_POP_BUF=216KB` | — |
| 跨核 GM v0Res | GM | 128×512×2B/core each | 3（`TRIPLE_BUFFER_NUM`）+ S2 实长缓冲 `3*128*4B*aivNum` |

- L1 512KB 预算下，128KB 的 KV 单块使槽池上限=3（对应 r=2 的下界 r+1），**D 超宽是槽池份数的第一约束**。
- buffer 轮转用 `BuffersPolicy3buffSFA`（a/b/c 三块，Get/GetVec/GetCube/GetPre(Q 复用)/GetReused(KV 复用)）。

## S5 流水编排

- **CSA `PRELOAD_NUM=3`，SWA `PRELOAD_NUM=2`**（`RunInfo runInfo[3]` 轮转）——压缩稀疏双路负载更重，预取更深；滑窗短路降 2 级。
- 同步：`SYNC_MODE=4` 跨核 flag（CROSSCORE_L1P/BMM2/BMM1/V0RES）+ 核内 flag（INNERCORE_L0AB/L0C/L1Q/L1KV）。
- **GetKVPhyAddr 独立相位**：BLKTABLE_FREE=3/READY=8、SPARSEIDX_FREE=4/READY=6、KVADDR_READY=5/FREE=7——地址计算与数据搬运各自成对 flag，互不阻塞。
- vec 侧 KV 搬运粒度：`KV_COPYIN_UNIT=8` 行、`KV_PROCESS_UNIT=16` 行；FD 归约 `FD_VEC1_MAX_ROWS=32`、`FD_REDUCE_CHUNK_ROWS=16`。

## S6 负载均衡

- **metadata 预算分核**（arch35 标配，[sparse_flash_mla_kernel_metadata.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h)）：1024×int32 GM 张量（输入索引 17），`SmlaMetadata{faMetadata[36][9], fdMetadata[72][8]}`（AIC_CORE_MAX_NUM=36、AIV_CORE_MAX_NUM=72）。FA metadata 9 字段/AIC：CORE_ENABLE/BN2_START/M_START/S2_START/BN2_END/M_END/S2_END/FIRST_FD_WORKSPACE_IDX/S2_MAX_NUM；FD metadata 8 字段/AIV。metadata 为 nullptr kernel 早退——**arch35 必选数据面，kernel 不现场算均衡**。
- **FlashDecode（split-KV）**（[flash_decode.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/common/flash_decode.h)）：`FD_MAX_S2_SPLIT_NUM=2`、`FD_INCREMENTAL_MERGE_INPUT_NUM=2`；GM staging 三段 `[partial O][max][sum]`，log-sum-exp 合并 `M=max(m_i), G=Σexp(m_i−M)·s_i, O=Σw_i·p_i`；主循环后由空闲核做 `ProcessFlashDecode`。
- **split-G**：gSize>64（n1>64）时 AIC 一分为二处理 G 前后各半（tilingKey splitG 位）；batch consistency（确定性规约）时 intraCoreSlots=2×logicalSlots、crossCoreSlots=aicNum×33。
- **稀疏块跳过**（[sparse_flash_mla_kvcache.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_kvcache.h)）：`smlaSkipThreshold=min(−nextTokensOri, −nextTokensCmp)` 跳 S1 前部无效块；`ComputeS2LoopInfo` 按 topk_length 裁剪 S2，`loopEndIdx` 按 s2BaseSize=128 上取整。
- **paged KV 地址向量化**（[get_kv_phy_addr_vf.h](file:///Users/wangwei/Desktop/项目/ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/common/get_kv_phy_addr_vf.h)）：VF 寄存器级算物理地址，每 loop 处理 128 个 sparse idx（2×64 寄存器），`DataCopyGather` 收集 blockTable，int64 地址拆高低 32 位；`phyAddr = phyBlkId*kvStride + (sparseIdx*sparseBlockSize mod blockSize)*kvDim`；仅 A5 + CSA/稀疏路径且 UB 184KB 预算内启用（`vectorizeFlag`/`IS_VEC_S2PHYADDR`），稀疏块数 128 对齐（`SPARSE_BLOCK_ALIGN_NUM=128`）。

## S7 自检结论

- I2：kvHead=1 天然满足；split-G 把 G 折进 M 轴共享 KV 加载。
- I4：FD split-KV 配 GM staging + log-sum-exp 跨核归约。
- 稀疏不定长：buffer 按最坏块长分配（s2BaseSize=128 常量 + topk_length 运行时裁剪范围），基本块不随稀疏度变化。
- K/V 槽池 3 = r+1（r=2）满足下界；L0 三端口恒 pingpong。
- 数值：topk_length=0 的行按完全屏蔽处理，双侧跳过（GEMM 与状态更新都跳）。

## 平台依赖

A5 专属模板模式（ORI_SPARSE/ORI_CMP_SPARSE）、paged blockSize∈[1,1024] 放宽（A2/A3 需 16 对齐）、VF 向量化地址（184KB UB 预算）、metadata 36AIC/72AIV 上限、ssbuf/Regbase——均 DAV_3510 口径。
