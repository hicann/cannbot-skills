# Kernel 路由：FlashAttention 类算子

> 本指南覆盖 FlashAttention（FA-2）前向算子的 catlass 设计入口，包括 MHA / GQA 两种 head 形态。它不替代普通 Matmul、Grouped Matmul 路由；只有需求命中 attention 融合特征时才启用。

---

## 场景定义

FlashAttention 类算子的共同特点是：Q/K/V 三输入，计算图为 **Q·K^T → softmax → ·V** 的两段矩阵乘 + 中间 softmax 的融合，且为了长 KV 序列采用**分块 + online softmax（FA-2）**，避免物化完整 `[Sq, Sk]` 分数矩阵。典型 I/O 维度是 `(B, H, Sq/Sk, D)`（BNSD）。

触发信号包括：`flash attention`、`fused attention`、`attention`、`scaled_dot_product_attention`、`QK^T softmax V`、`MHA`、`GQA`、`multi-head attention`。

> **命名约定（强制）**：本 skill 生成的 FA 算子统一命名为 `FA-<layout>[-<mode>]`，如 `FA-BNSD`（BNSD 全注意力）、`FA-BNSD-Causal`（BNSD 下三角 causal）、`FA-TND`（TND varlen）。设计 / 实现 / 验证全程使用同一算子名。

> **FA 内核资产（catlass 自带，复用 catlass examples/23）**：
> - `catlass/examples/23_flash_attention_infer/` —— `FAInferKernel` + `FAInferFp16`/`FAInferBf16` 入口 + `FAInferTiling`，整经验证（Paged/varlen/mask 全功能）。
>
> 这是 `FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, PAGED>` 的固化核函数。设计阶段把该目录作为内核来源，**复用 catlass examples/23 的 FAInferKernel、不引用外部 attention 仓库**。

---

## Step 0: 自动锁定标杆参考

命中本路由时，先读取 [flash-attention-npu-reference.md](flash-attention-npu-reference.md)。该文件锁定 FA 标杆接口（`aclnnFlashAttentionScore` / `torch_npu.npu_fusion_attention`），维护「标杆参数 → FA 内核参数」映射（layout / scale / `sparse_mode`↔`maskType` / seqlen 累加↔per-batch）、可继承/不可继承边界与生成契约。**不要要求用户在 prompt 中重复提供参考路径**；标杆是 CANN/torch_npu 内置 API（有文档），无需 clone 外部仓库。

DESIGN.md 的 baseline/reference 章节必须写明：标杆来源（`aclnnFA` / `npu_fusion_attention`）、本次冻结的 contract（`input_layout`、`maskType`、`scale`、seqlen 约定），以及哪些语义来自标杆、哪些来自catlass `examples/23_flash_attention_infer`。

---

## Step 1: 冻结目标形态与同语义 baseline

```
目标是什么？
├── full-flow fused op（推荐）
│   ├── 必须冻结同语义 full-flow baseline
│   └── 首选 baseline: aclnnFlashAttentionScore（CANN 内置，见 op_api 文档）
└── stage operator（C1/QK^T 或 C2/PV 单独交付）
    ├── 必须说明上下游 stage 输入/输出契约
    └── golden 可用 stage-aware reference，但上游中间量必须先被 full-flow 独立验证
```

**禁止**把局部 helper、临时 NumPy 片段当作 full-flow baseline。baseline 的语义、dtype、layout、scale 规则必须与交付目标一致（`scale = 1/sqrt(D)`，`O = softmax(scale * QK^T) · V`）。

> 与 FA-2 数学模型的对应：`C1: S = Q·K^T`、`V1: P = exp(S − m)`（online softmax）、`C2: O_acc += P·V`、`V2: O = O_acc / l`。详见 `ascendc-tiling-design/references/flashattention/foundation/fundamentals.md`（skill 仓内方法论，不变量 I1–I5）。

---

## Step 2: FA 内核来源与组件契约（复用 catlass examples/23 FAInferKernel）

FA 是 AIC/AIV 跨核协作的混合 kernel，**整经验证的核函数由本 skill 提供**：`catlass/examples/23_flash_attention_infer/fai_kernel.cpp`（入口 `FAInferFp16`/`FAInferBf16`，配套 `flash_attention_infer_common.hpp`、`flash_attention_infer_tiling.hpp`）。生成 FA 算子时把 `catlass/examples/23_flash_attention_infer/` 加入 include 路径，复用 catlass `examples/23_flash_attention_infer/` 的 `fai_kernel.cpp`/`fai.cpp`/`fai_tiling.cpp`/`kernel_common.hpp`，**只改 host 侧做公开接口**。**复用 catlass examples/23 的 FAInferKernel**，不改 catlass 架构 `include/`——组件选型已固化在内核模板里，host 不重选。

> **★实现策略（最高优先级）**：必须复用本 skill 的 `FAInferKernel`。**禁止用 `BlockMmad + Epilogue` 从零拼装 FAKernel**（flag 时序无法对齐，必跨核死锁）。详见 develop skill `patterns/flash-attention.md` §0。

内核模板已固化的组件（见 `catlass/examples/23_flash_attention_infer/fai_kernel.cpp`，host 不重选，仅供理解/核验；组件名以核验结果为准）：

| 组件 | 取值 | 说明 |
|------|------|------|
| ArchTag | `Arch::AtlasA2`（A2/A3） | 由目标 SoC 定 |
| BlockMmadQK | `MmadAtlasA2FAIQK` | C1：Q·K^T，K 按 ColumnMajor 消费（host 不预转置）|
| BlockMmadPV | `MmadAtlasA2FAIPV` | C2：P·V |
| BlockMmadQKTail/PVTail | `MmadAtlasA2FAITailQK` / `MmadAtlasA2FAITailPV` | KV 非 128 对齐时的 tail 处理 |
| EpilogueOnlineSoftmax | `EpilogueAtlasA2OnlineSoftmax` | V1：online softmax + running max/sum |
| EpilogueRescaleO | `EpilogueAtlasA2RescaleO` | V2：按 `exp(m_old − m_new)` rescale O |
| Kernel | `FAInferKernel<...>`（AIC/AIV 双特化，catlass `examples/23_flash_attention_infer` 提供） | `operator()<AIC>` + `operator()<AIV>` |
| TileShape | `L1=L0=GemmShape<128,128,128>`（`L1TileShape::K==D`） | 内核固定，性能再迭代 |
| PAGED_CACHE_FLAG | `true`（A2） | 恒等 block_table 等价非 Paged |

> **组件名核验**：不同 catlass 版本组件名可能不同。设计/实现前 grep 核验真实组件名与是否有特化体：
> ```bash
> grep -rn "MmadAtlasA2.*QK\|MmadAtlasA2.*PV" catlass/include/catlass/gemm/dispatch_policy.hpp
> grep -rn "struct BlockEpilogue<EpilogueAtlasA2.*RescaleO" catlass/include/
> ```

**数据流**：`QK^T → L0C → Fixpipe → UB(S) → EpilogueOnlineSoftmax(P, m, l) → P·V → L0C → Fixpipe → UB(O_acc) → EpilogueRescaleO → O`。AIC 出 S/O，AIV 做 softmax/rescale，AIC↔AIV 用 CrossCoreFlag 握手。

---

## Step 3: workspace 与 AIC/AIV 协作

- 中间量 `S`/`P`/`O_tmp`/`O_update` 用**独立 GM workspace 段**（3 级轮转），用 GM buffer 指针透传，**禁 `GetUserWorkspace`（直调路径会丢弃入参返回 kfc 地址）**。
- AIC/AIV 跨核同步：A2 支持 `CrossCoreSetFlag/WaitFlag` modeId 0/1/2（950PR 另有 4）。Set 与 Wait 两侧 modeId 必须一致，否则死锁。
- 对 BNSD 接口：Q 经 host 转 BSND `[B,Sq,H,D]`，K/V 转块格式 `[numBlocks,128,H,D]` 后进 kernel；O 由 BSND 转回 BNSD。布局转换放 host，kernel 只消费一种内部布局。
- 详细 stage 设计见 `catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md`。

---

## Step 4: Shape 覆盖矩阵

FlashAttention 的测试 shape 按算法维度构造，不按随手枚举：

| 类别 | 构造要点 |
|------|----------|
| B/Sq/Sk 组合 | 单 batch、多 batch；`Sq == Sk`、`Sq < Sk`、`Sk` 非 128 倍数（尾块）|
| H / head 形态 | MHA（Hq==Hkv）、GQA（Hq > Hkv）；单头/多头 |
| D（headSize） | 64 / 128 等 cube 友好值，必须 `D % 16 == 0` |
| 对齐边界 | `Sk` 恰为 128 倍数 / 非倍数；尾块 0 填充路径 |
| 数值边界 | 全零 Q/K/V、近零 softmax 输出、极值 scale |

shape 来源写进 DESIGN/PLAN：用户实网规模、baseline 支持范围、cube 整数倍、边界类别。禁止硬编码无说明的 shape tuple。

---

## Step 5: 精度与性能基准在设计阶段冻结

### 精度
- 标准优先 `ops-precision-standard`，FA 浮点输出按 mixed tolerance：`abs(actual − golden) <= atol + rtol * abs(golden)`，再看 `matched_ratio` / `max_abs`。
- golden 首选 **`aclnnFlashAttentionScore`**（标杆接口）实测输出；次选独立 CPU 参考（numpy/torch fp32 累加）。
- 报告每 case 输出 `shape`、`dtype`、`atol`、`rtol`、`matched_ratio`、`max_abs`、`pass/fail`。

### 性能
| 优先级 | 基准 |
|:---:|------|
| 1 | 同语义 baseline（aclnnFA）实测 Task Duration |
| 2 | catlass 参考 example（23/81）同 shape 实测 |
| 3 | Cube/MTE/Vector 理论上限 |

性能报告至少含：custom/baseline 时长、speedup、launch count、主导流水（Cube/MTE/Vector/同步等待）、cube_util、workspace peak、profiler 路径。

---

## 常见陷阱

| 陷阱 | 后果 | 正确做法 |
|------|------|----------|
| ① **A2 上 `PAGED_CACHE_FLAG=false`（真非 Paged）路径触发 aicore exception** | 507015 运行崩溃 | A2 上固定 `PAGED=true` + **恒等 block_table**（语义等价非 Paged BNSD 连续 KV 块）；不要尝试修 Paged 路径 |
| ② 接口写成 BSND/块格式而不是 BNSD | 与用户 BNSD 期望不符 | host 做 BNSD→内部布局转换（Q→BSND、K/V→块格式、O→BNSD），公开接口保持 BNSD |
| ③ KV 非 128 对齐不做尾块填充 | 越界/错误结果 | 尾块 0 填充；bin 读入尺寸（BNSD）与 device 尺寸（块格式）分开 |
| ④ 直调路径用 `GetUserWorkspace` | 拿到 kfc 地址越界 | 用独立 GM buffer 指针透传 |
| ⑤ 未冻结 baseline 就开写 | 精度无从对标 | 设计期先冻结 `aclnnFlashAttentionScore` 语义 |
| ⑥ 把中间 S 物化到 HBM | 失去 FA 分块意义 | 走 L0C→Fixpipe→UB，保持分块 online softmax |

---

## 输出设计章节补充

命中本路由时，DESIGN.md 除 catlass 通用选型表外，必须额外包含：
1. full-flow 判定与 baseline 路径（aclnnFA 或 CPU 参考）
2. 组件选型表（BlockMmadQK/PV、EpilogueOnlineSoftmax/RescaleO、TileShape、PAGED 方案）
3. 数据流与 workspace（3 级轮转、AIC/AIV 同步）
4. BNSD 接口与 host 布局转换方案
5. shape 覆盖矩阵与每类 shape 依据
6. mixed tolerance 精度标准与 golden 来源
7. 性能对标口径
