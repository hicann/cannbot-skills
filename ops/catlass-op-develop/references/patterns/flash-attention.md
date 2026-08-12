# FlashAttention — BNSD QKV 类实现注意事项

> **导航**：设计路由见 design skill [kernels/flash-attention.md](../../../catlass-op-design/references/kernels/flash-attention.md)。本文聚焦实现期规则：BNSD 接口、PAGED/block_table 方案、host 布局转换、AIC/AIV 协作、精度/性能归档。
> **FA 内核资产（catlass 自带）**：`catlass/examples/23_flash_attention_infer/`（`FAInferKernel` + `FAInferFp16`/`FAInferBf16` 入口 + `FAInferTiling`，整经验证）。生成 FA 算子把该目录加入 include 路径、复用 catlass `examples/23_flash_attention_infer/` 的 FAInferKernel 即可，**复用 catlass examples/23 的 FAInferKernel**。

---

## 0. 实现策略：复用catlass examples/23 的 FAInferKernel，禁止从零拼装（★最高优先级）

**FA-2 是 AIC/AIV 跨核协作的混合 kernel，其 CrossCoreFlag 时序、`SetSyncBaseAddr`、in-pipe HardEvent flag 预置、online softmax 的 m/l UB 布局、mask GM 契约是一整套强耦合的时序契约。从零拼装 `BlockMmad + Epilogue` 极易触发跨核死锁（AIC 已 set flag、AIV 卡在 `CrossCoreWaitFlag`/MTE2 wait 不返回，507015 aicore 异常）。**

正确实现策略（按优先级）：
1. **首选：把本 skill `catlass/examples/23_flash_attention_infer/` 加入 include 路径，复用 catlass `examples/23_flash_attention_infer/` 的 `fai_kernel.cpp`/`fai.cpp`/`fai_tiling.cpp`/`kernel_common.hpp`（catlass 自带的 `FAInferKernel`），只改 host 侧做公开接口**。核函数含 `FAInferFp16`/`FAInferBf16` 入口、`FATilingData`/`FAIKernelParams`、`FAInferTiling` namespace，已验证可跑、flag 时序正确。
2. **复用 catlass examples/23 的 FAInferKernel**（23/81 等）：FA 内核由catlass `examples/23_flash_attention_infer/` 整体提供，组件已固化、host 不重选。
3. **禁止：用 `BlockMmad<MmadAtlasA2FAQK/FAPV>` + `BlockEpilogue<FASoftmax>` 从零拼装 FAKernel**——`FARescaleO` 特化体常缺失，且从零拼装的 flag 时序无法对齐，必死锁。

> 与 GMM/KDA 的差异：GMM/KDA 的组件（GroupedMatmul/LinearAttention）时序简单、可自由拼装；FA 必须复用整经验证的 kernel，这是 FA 的强制约束。

---

## 1. 公开接口必须是 BNSD，布局转换放 host

> **命名**：FA 算子统一命名 `FA-<layout>[-<mode>]`（如 `FA-BNSD` / `FA-BNSD-Causal` / `FA-TND`），见 design 路由「命名约定」。

FlashAttention 算子的**公开接口**取严格 BNSD：`Q=[B,H,Sq,D]`、`K/V=[B,H,Sk,D]`、`O=[B,H,Sq,D]`，全部连续 fp16。catlass kernel 内部消费一种布局（见下），**host 负责转换**，kernel 只认一种内部布局、不做运行时分支。

kernel 内部布局（对照 23/81 示例）：
- Q 取 **BSND** `[B,Sq,H,D]`（`strideQO = H*D`）
- K/V 取**块格式** `[numBlocks, blockSize, H, D]`（`blockSize=128`）
- O 出 BSND，host 转回 BNSD

host 转换要点：
```text
Q:   BNSD [B,H,Sq,D] → BSND [B,Sq,H,D]      （三重循环转置）
K/V: BNSD [B,H,Sk,D] → 块 [numBlocks,128,H,D]（按 128 块重排，尾块 0 填充）
O:   BSND [B,Sq,H,D] → BNSD [B,H,Sq,D]
```
**bin 读入尺寸与 device 尺寸分离**：BNSD K 的 bin 是 `B*H*Sk*D*2`，kernel 的块格式是 `numBlocks*128*H*D*2`；读 bin 按前者、拷 device 按后者（`kvBinSize` vs `kvSize` 两个变量）。

---

## 2. PAGED 方案：A2 上固定 `PAGED=true` + 恒等 block_table

**关键平台经验**：catlass 官方 23 示例的 `PAGED_CACHE_FLAG=false`（真非 Paged）路径在 **A2（dav-2201）触发 aicore exception（507015）**，多 shape 复现，官方未充分验证。因此：

- A2 上**固定 `PAGED=true`**，用**连续恒等 block_table**（`block_table[i]=i`，块 `b*MB+j` 存第 b 个 batch 的第 j 个 KV 块），语义与非 Paged BNSD 连续 KV 完全等价。
- 不要在生成流程里尝试修 `PAGED=false` 路径；把恒等 block_table 方案固化为默认。
- K 仍按 `LayoutK=ColumnMajor` 在 kernel 内消费，**host 不预转置 K**。

---

## 3. KV 非 128 对齐的尾块处理

`numBlocksPerBatch = ceil(Skv / 128)`。`Skv % 128 != 0` 时尾块 0 填充：
- 块 buffer 先 memset 0，再按有效长度填充；
- gen_data 与 host 都要用同一套 block 映射（`block[b*MB+j, t, h, d] = K_bnsd[b, h, j*128+t, d]`）；
- golden 用原始 BNSD K/V 计算，不受块格式影响。

---

## 4. AIC/AIV 协作与 CrossCoreFlag

catlass `FAInferKernel` 用 `operator()<AIC>` / `operator()<AIV>` 双特化，Cube 与 Vector 异核协作：
- AIC：`BlockMmadQK` 出 S（L0C→Fixpipe→GM/UB）、`BlockMmadPV` 出 O_block；
- AIV：`EpilogueOnlineSoftmax` 算 P/m/l、`EpilogueRescaleO` 做 O rescale；
- AIC↔AIV 用 `CrossCoreSetFlag/WaitFlag` 握手；**Set 与 Wait 两侧 modeId 必须一致**（A2 modeId 0/1/2；950PR 另有 4）。详见 `a2-a3-flash-attention-stage-design.md` §2。

workspace 用独立 GM buffer 段（`s/p/oTemp/oUpdate` 四段）指针透传，**禁 `GetUserWorkspace`/`SetSysWorkspaceForce`**。

---

## 5. 精度验证：dump O + aclnn 标杆对比

- kernel 算出 O 后，**必须 dump BNSD 输出**（fp32）到 `o.bin`，供外部对比；
- 精度双口径：
  1. 内部 golden（CPU numpy/torch fp32 累加 softmax attention）逐元素对比；
  2. **标杆接口对比**：`torch_npu.npu_fusion_attention(q, k, v, H, "BNSD", scale=scale)`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比；
- 判据（`ops-precision-standard` mixed tolerance）：`atol=0.02, rtol=0.1`，通过率 ≥ 99.9% 且 `max_abs < 0.05`。经验值：A2 上 custom vs aclnn `max_abs ~ 1e-4`。
- 报告每 case：`shape`、`dtype`、`atol/rtol`、`matched_ratio`、`max_abs`、`pass/fail`。

---

## 6. 性能归档

### 度量口径（device time，不是 wall time）
- **device kernel 时间**用 cannbot `ops-profiling`（msprof）的 `task_time(us)`，**取 min of N**（mean 会被 HBM/L2 cache 状态污染）；kernel 名 `FAInferFp16`（AIC/AIV 混合，Type=MIX_AIC）。
- 对比标杆：`npu_fusion_attention` 的 `FlashAttentionScore` kernel（同 msprof 口径）。
- **不要用 host 侧 wall-clock**比 kernel 性能（含 launch/sync/布局转换开销，FA 的 wall-clock 比 device time 大 ~15–20%）。

### 实测基线（910B3 / dav-2201，catlass examples/23 FAInferKernel，min of N）
| 算子 | shape | device time | vs npu FlashAttentionScore |
|---|---|---|---|
| FA-BNSD（全注意力）| MHA 1024 / 2048，D128 | 72 / 180 μs | 0.80 / 0.91× |
| FA-BNSD（GQA）| GQA 1024，D128 | 73 μs | 0.77× |
| FA-BNSD-Causal | MHA causal 1024 / 2048 | 53 / 117 μs | 1.01 / 0.94× |
| FA-TND（varlen）| MHA 1024 / 2048 | 72 / 180 μs | 0.85 / 0.97× |

> causal 比 full 快、甚至超 npu：内核结构性跳过对角块以上的整 KV 块（`noSkipKvS/noMaskKvS` 由 `qSBlockIdx` 推导），省约一半计算。整体 range **0.77–1.06×**。

### 性能字段（归档到 `docs/perf/round_NNN/`）
| 字段 | 说明 |
|------|------|
| baseline | `aclnnFlashAttentionScore` / `npu_fusion_attention` 同 shape 实测（msprof device time）|
| Task Duration | custom `FAInferFp16` vs baseline，同 shape、同 msprof 口径 |
| cube_util | AIC 利用率（A2 经验 ~96%，AIC/AIV 高度重叠）|
| dominant pipeline | Cube / MTE / Vector / 同步等待中的主瓶颈 |
| workspace peak / profiler path | 原始 msprof trace 路径 |

性能差距集中在 `Sq*H` 小于核数、尾块、GQA 场景时，先归因到调度/tiling 分支，不直接调单个 TileShape。

---

## 7. 已知问题与踩坑（实战总结，生成前必读）

这些问题在真实生成 BNSD-QKV-FA 时出现过，按发生频率与影响排序：

| # | 现象 | 根因 | 对策 | 来源 |
|---|------|------|------|------|
| T1 | AIC 已 set flag，AIV 卡在 softmax 内 `CrossCoreWaitFlag`/MTE2 wait 不返回（507015） | **从零拼装 FAKernel，flag 时序未对齐** | 复用 examples/23 的 FAInferKernel（§0）；绝不从零拼装 | examples/23 复用验证 |
| T2 | A2 上 `PAGED_CACHE_FLAG=false` 跑出 aicore exception | 官方该路径在 A2 未验证 | 固定 `PAGED=true` + 恒等 block_table（§2）| A2 平台复现 |
| T3 | 报某组件"未定义"/"只有声明无特化体" | 不同 catlass 版本组件名不同；某些 DispatchPolicy 只有 `struct X{}` 声明、无 `block_epilogue_*.hpp` 特化体 | 生成前 grep 核验真实组件名 + 是否有特化体（design §Step2）| catlass 版本差异实测 |
| T4 | 编译报 `MmadAtlasA2FAIQK 未定义` / `AscendC 未声明`（~20 errors） | 用了系统自带 catlass（canndev / CANN opp，**FAQK 新版 API**），与 FAInferKernel 要的 **FAIQK 旧版 API** 不兼容（6→9 参 BlockMmad、无 Tail） | 用 gitcode 的 catlass：`git clone https://gitcode.com/cann/catlass.git`（HEAD 兼容 FAIQK）；**不**用 canndev 自带 catlass | 部署实证（canndev catlass 不兼容）|
| T5 | gen_data 与算子/对比脚本的参数顺序不一致，数据 reshape 失败 | gen_data 顺序是 `batch qSeq kvSeq numHeads headSize` | 统一参数顺序，三方（gen_data/算子/compare）对齐 | 测试工程实证 |
| T6 | `o.bin` reshape 失败 / `block_table.bin size larger than buffer` | data 目录残留旧 shape 数据 | 每次 gen 前清 data 目录（`rm -rf data`）| 测试工程实证 |
| T7 | 首次 launch 偶发卡死，重试即正常 | NPU 首次初始化/`aclrtGetHardwareSyncAddr` 竞态 | 重试复现即排除；持续卡死才按 T1 处理 | NPU 首次初始化经验 |
| T8 | workspace 越界（MTE DDR） | 直调路径用了 `GetUserWorkspace`（丢入参返回 kfc 地址） | 独立 GM buffer 指针透传，禁 `GetUserWorkspace`/`SetSysWorkspaceForce`（§4）| catlass 直调路径实证 |

---

## 8. 强制检查表（FA1–FA11）

| # | 检查项 |
|---|--------|
| FA0 | **复用catlass `examples/23_flash_attention_infer/` 的 `FAInferKernel`（复用 catlass `examples/23_flash_attention_infer/` 的 FAInferKernel，来自 `catlass/examples/23_flash_attention_infer/`），未从零拼装 FAKernel**（从零拼装必跨核死锁）|
| FA1 | 公开接口为严格 BNSD（Q=[B,H,Sq,D]，K/V=[B,H,Sk,D]），布局转换在 host |
| FA2 | 冻结同语义 baseline（`aclnnFlashAttentionScore`），语义/dtype/layout/scale 一致 |
| FA3 | A2 上固定 `PAGED=true` + 恒等 block_table，不碰 `PAGED=false` 路径 |
| FA4 | KV 非 128 对齐时尾块 0 填充；bin 读入尺寸与 device 尺寸分离 |
| FA5 | workspace 用独立 GM buffer 指针透传，禁 `GetUserWorkspace` |
| FA6 | AIC↔AIV 的 CrossCoreFlag Set/Wait modeId 一致，无死锁 |
| FA7 | 组件选型为 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO（对照 23/81 示例）|
| FA8 | shape 覆盖矩阵含 B/Sq/Sk 组合、非 128 对齐尾块、MHA/GQA、D=64/128、数值边界 |
| FA9 | 精度双口径（内部 golden + aclnn 标杆），mixed tolerance，`max_abs`/`matched_ratio` 记录 |
| FA10 | 性能报告含 baseline、Task Duration、launch count、cube_util、profiler path |
| FA11 | 多 stage/AIC-AIV 场景已读取并执行 `a2-a3-flash-attention-stage-design.md` checklist |
| FA12 | **mask 模式正确**：掩码按内核 `MaskType` 选型（`NO_MASK`/`MASK_SPEC`/`MASK_CAUSUAL`）；下三角 causal（npu `sparse_mode` 2/3）用 `MASK_CAUSUAL`，**勿自实现 causal**。mask 张量语义/契约见 §9 与 design skill `kernels/flash-attention-npu-reference.md` |
| FA13 | **actualQseqlen/Kvseqlen 是 per-batch 原始长度（非累加）**；dense BNSD 传 `[Sq]×B`/`[Skv]×B`（见 §10）|

---

## 9. mask 模式（MaskType）

`FAInferTiling::MaskType`（见 `catlass/examples/23_flash_attention_infer/fai_tiling.cpp`）：

| 值 | 名称 | 含义 |
|---|------|------|
| 0 | `NO_MASK` | 全注意力，无 mask |
| 1 | `MASK_SPEC` | 用 mask 张量逐位屏蔽（mask 张量全程参与） |
| 2 | `MASK_CAUSUAL` | **下三角 causal**：结构性跳过 + 对角块读 mask |

- 下三角 causal（npu `sparse_mode` 2/3）一律用 `MASK_CAUSUAL`(=2)，**不要自实现 causal 逻辑**：内核按 query 块 `qSBlockIdx` 结构性跳过对角块以上的整 KV 块（`noSkipKvS/noMaskKvS` 由 `qSBlockIdx` 推导），仅对角块调用带 mask 的 softmax epilogue。
- mask 张量语义：epilogue `ApplyMask = score += mask·(−3e38)` → **mask≠0 屏蔽、0 保留**（与 npu `atten_mask` 的 `1=不参与` 一致）。
- mask 张量的具体形状 / stride / 取值、对角块索引方式、Sq≠Skv 时下三角对角线对齐（内核 `diffS=Skv−Sq`），**以 `catlass/examples/23_flash_attention_infer/fai_kernel.cpp` 与标杆映射 `kernels/flash-attention-npu-reference.md` 为准**（参考实现见 catlass example `83_fa_bnsd_causal`）。本规则文档不固化具体常量，避免随样例逆向值过时。
- 标杆对比：`npu_fusion_attention(input_layout="BNSD", sparse_mode=2, atten_mask=triu(ones,bool,1))`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比。

## 10. seqlen 契约：per-batch（非累加）

`FAIKernelParams.actualQseqlen/actualKvseqlen` 是**每 batch 原始长度**（int64 数组，长度=B），内核 `qBOffset += qSeqlen·strideQO` **自行累加偏移**。**不要**照搬 `npu_fusion_attention` 的累加 `actual_seq_*`。

- dense BNSD（每 batch 等长）：`actualQseqlen = [Sq]×B`，`actualKvseqlen = [Skv]×B`。
- varlen（每 batch 不等长）：传每 batch 原始 Sq/Skv（**非**累加和）。注意：本 skill 的 FA 内核 seqlen 约定与 npu TND 接口的累加 actual_seq 不同，必须按内核契约传。
