# FlashAttention 标杆接口参考规范

本文件用于 FA 类 Catlass 算子设计/实现前的**标杆对齐**。命中 FA 路由（见 [flash-attention.md](flash-attention.md) Step 0）时，先读本文锁定标杆接口与「标杆参数 → FA 内核参数」映射，**不要要求用户在 prompt 中重复提供参考路径**。

## 标杆接口定位

| 标杆 | 位置 | 用途 |
|---|---|---|
| `aclnnFlashAttentionScore` | CANN `op_api` 文档（ACL 单算子） | 底层标杆：语义 / dtype / layout / scale / mask 权威来源 |
| `torch_npu.npu_fusion_attention` | torch_npu op-plugin（封装 `aclnnFlashAttentionScore`） | Python 标杆：可直接 NPU 实测逐元素对比 |
| FA 内核资产 | catlass `examples/23_flash_attention_infer/`（`FAInferFp16`/`FAInferBf16` + `FAInferTiling`） | 生成 FA 算子复用的整经验证核函数 |

> 本 skill 前向只对齐 `attention_out = softmax(scale·Q·Kᵀ + mask)·V`。训练侧 dropout / `softmax_max`/`softmax_sum` 反向中间量 / `pse` / `sink` / `inner_precise` 等 `npu_fusion_attention` 字段**不在本 skill 前向生成范围**，命中时按「用户 contract 优先」处理（见下）。

## 标杆参数 → FA 内核参数映射

| 标杆参数 | FA 内核处理 |
|---|---|
| `input_layout`（`BNSD`/`TND`/`BSH`/`SBH`/`BSND`） | 内核内部固定：Q=`BSND [B,Sq,H,D]`，K/V=块格式 `[numBlocks,128,H,D]`。**host 做转换**，kernel 不做运行时分支（TND 下 Q/O 可直透传，仅 K/V 需转 paged） |
| `head_num=Nq`、`Nq%Nkv==0`（MHA/GQA） | 内核 `numHeads=Nq`、`kvHeads=Nkv`；query head `h` 用 kv head `h//group`（`group=Nq/Nkv`） |
| `scale`（默认 `1/√D`） | `FATilingData.scaleValue`。内核 `FAInferTiling` 默认按 `1/√D` 自算；若标杆传**自定义 scale**，host 须覆盖 `scaleValue` |
| `atten_mask` + `sparse_mode` | → 内核 `maskType`（见下表） |
| `actual_seq_qlen`/`actual_seq_kvlen`（npu 为**累加**和） | 内核 `actualQseqlen`/`actualKvseqlen` 为 **per-batch 原始长度**（host 做累加→原始的差分；dense 等长直接 `[S]×B`）。**两者约定不同，必须转换** |
| `keep_prob`/`dropout`（前向） | 本 skill 前向 `keep_prob=1`（无 dropout） |
| 输出 `attention_out` | 内核 `O`（BSND，host 转回公开 layout） |

`sparse_mode` → `maskType` 映射（重点）：

| npu `sparse_mode` | 含义 | 内核 `maskType` |
|---|---|---|
| 不传 `atten_mask` / `0`(defaultMask) 无掩码 | 全注意力 | `NO_MASK`(0) |
| `1`(allMask) / 任意 `atten_mask` 逐位 | 逐位掩码 | `MASK_SPEC`(1) |
| `2`(leftUpCausal) / `3`(rightDownCausal) — **下三角 causal**（Sq==Skv 时一致） | 因果/下三角 | **`MASK_CAUSUAL`(2)**（内核结构性跳过对角块以上整块 + 仅对角块读 mask，勿自实现 causal） |
| `4`(band) / `5-8`(prefix/varlen外切) | 由 host 按 `pre_tockens`/`next_tockens` 预生成 `atten_mask` 后走 `MASK_SPEC` | `MASK_SPEC`(1) |

mask 张量语义：内核 epilogue `ApplyMask = score += mask·(−3e38)` → **mask≠0 屏蔽、0 保留**（与 npu `atten_mask` 的 `1=不参与` 一致）。具体张量形状/stride/取值由内核 `LayoutMask` 与对角块索引决定，**以 `catlass/examples/23_flash_attention_infer/fai_kernel.cpp` 为准**（参考实现见 catlass example `83_fa_bnsd_causal`）。

> npu 标杆侧：`npu_fusion_attention` causal（sparse_mode 2/3）的 `atten_mask` 必须是压缩下三角 `[2048,2048]`（否则 tiling 报 `161001 "set atten_mask_shape to [2048,2048]"`）。

## 可继承 vs 不能直接继承

| 来源 | 可继承 | 不能直接继承 |
|---|---|---|
| `aclnnFA`/`npu_fusion_attention` | 数学语义、`scale=1/√D`、mask 语义（`atten_mask 1=不参与`）、layout、head 映射、baseline 对齐口径 | 把 npu 的**累加** `actual_seq` 当内核 seqlen；把 dropout/`pse`/`sink`/`inner_precise` 等**本 skill 未支持**字段静默继承 |
| catlass `examples/23_flash_attention_infer` | 核函数（`FAInferFp16`/`Bf16`）、`FAInferTiling` namespace、组件契约 | 自行改 kernel 内 CrossCoreFlag 时序 / 组件实例化（必跨核死锁） |

## 生成契约：用户 prompt 优先

FA 同族算子常存在多个相近但不等价的 contract（`scale` 作用位置、mask 模式与对齐、seqlen 累加/原始、输出 dtype、是否产出反向中间量）。生成时遵循：

- 用户 prompt 给的**数学公式与接口语义优先于标杆实现**。
- 标杆只用于对齐语义 / layout / mask / scale / baseline 口径；不能把标杆里本 skill 不支持的 contract 静默继承到目标算子。
- DESIGN.md 必须冻结本次 contract（`input_layout`、`maskType`、`scale`、seqlen 约定），并在 golden、verify、README、精度/性能报告中保持同一语义。
- 若标杆、本 skill 旧算子、用户公式三者不一致，**不自行择优**：在 DESIGN.md 记录差异并以用户公式为准，除非用户明确改口。
