# MLA Paged Attention — 实现契约

> **导航**：设计路由见 design skill [kernels/flash-attention.md](../../../catlass-op-design/references/kernels/flash-attention.md)（MLA 小节）。本文聚焦实现期规则：算子语义、内核架构、host 集成契约、shape 约束、精度与性能基线。
> **实现基点**：MLA 与 FA 同范式（分页 KV + MQA latent 压缩），本 skill 提供整经验证的 MLA 内核（通用 H=16/32/64 + H=128 特化 + AMLA 特化），host 只做封装。
> **来源**：MLA paged 工程 / MLA 手搓工程 工程实测（Ascend 910B3 / dav-2201，catlass gitcode HEAD），18+20 case 精度全 PASS，与 ATB 融合算子标杆三方互证一致。

---

## 0. 实现策略：手搓 MLA kernel（★最高优先级，配方见 [fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md)）

**MLA（Multi-head Latent Attention，分页 KV + MQA latent 压缩）与 FA 同范式：手搓路径已实证可一次跑通。**

**kernel 按 [fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md) 手搓**（MLAKernel
逐行配方：组件全在 catlass include 层 + CrossCoreFlag/HardEvent 真实用法 + prologue+延迟 PV 主循环
+ GM 槽位公式 + launch/build 契约）。§8 的组件链伪代码已被 recipe 替代——**以 recipe 为准**。

> **★2026-09 实测纠错（本会话手搓踩坑，必读）**：§8 伪代码缺两个事实，导致手搓必死锁，
> recipe 已补正：
> 1. **AIC/AIV 分流必须 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__` 预处理分支**，**不是**
>    `__mix__(1,2)`+runtime `g_coreType`（那是 kfc 单 TU 路径 §12.13，MLA CrossCoreFlag 路径不用）。
>    实测 `__mix__` 下 AIC 单独 pre-set+QK+drain 即死锁。
> 2. **主循环必须 prologue(QK0+softmax0 无PV) + loop[QK(n)+softmax(n) if非末 / 末块下task预取 /
>    PV(n-1)+rescale(n-1) 延迟一块]**，**不是** QK+PV 同块锁步。锁步的 CrossCoreFlag Set/Wait 顺序
>    与组件内部 HardEvent 配对不匹配→token 不平衡→死锁。
> 3. CrossCoreFlag 用法（§8.5 正确，实证 OK）：Set `Arch::CrossCoreSetFlag<0x2,PIPE_X>(flag)`，
>    Wait bare `Arch::CrossCoreWaitFlag(flag)`（dav-2201 上与 `<0x2,PIPE>` 配对）。
> 4. `glFlag[2]={1,1}` 真数组（勿 nullptr）；AIV `coreIdx=GetBlockIdx()/GetSubBlockNum()`（勿用裸 blockIdx）。

内核形态（覆盖矩阵）：

| 形态 | 适用 |
|---|---|
| 通用 MLA（H=16/32/64，fp16/bf16） | 模型侧 TP8/4/2 场景 |
| H=128 特化（TP1 规格，K 常驻 L1） | 模型侧 TP1 场景 |
| AMLA 特化（H=128） | AMLA 变体 |

组件 dispatch 固化为 `MmadAtlasA2MLAQK` / `MmadAtlasA2MLAPV`，host 不重选。

> **★路由修正（2026-08-28 定稿）**：MLA **手搓是唯一路径**，且必须按 §8 的**达标架构**（§8.9 五大性能支柱）写——单 kernel 锁步 + S/P 小槽轮转 + L1TileShape K=576 K 常驻 + MLAQK/MLAPV 专用 dispatch + flash-decoding split（组件全在 catlass include 层）。历史教训：朴素架构（双 launch + P 全量 GM 物化 + 通用 FAQK）实测仅 0.13~0.36，达标架构 1.16+——**慢的不是手搓，是走错架构**；两架构差异见 §8.9 对照表。

---

## 1. launch 契约：14 参，`aicCoreNum` 并度

```cpp
MLA<T><<<aicCoreNum, nullptr, stream>>>(
    hwSync,          // aclrtGetHardwareSyncAddr 取
    q, qRope,        // [numTokens, H, Dc+Dr 拆两段]
    k, kRope,        // paged KV cache: [numBlocks, 128, Dc] / [numBlocks, 128, Dr]
    blockTable,      // [batch, maxBlocksPerBatch] int32
    o,               // [numTokens, H, Dc] 输出
    s, p,            // workspace（见 §3）
    oTmp, globalo, oCoreTmp,  // workspace（见 §3）
    l,               // [numTokens, H] 归一化因子 workspace
    tiling);         // tiling buf 指针（GM）
```

**q_seqlen / kv_seqlen 只进 tiling，不是 kernel 入参**——与 FA 内核的 per-batch seqlen 契约不同，别照搬。

## 2. tiling：`GetMLATilingParam`

```cpp
uint32_t buf[(400 + numTokens * 17)];   // H==128 用 numTokens，否则 batch
MLAInfo info{...};                       // 含 batch/numTokens/qSeq/kvSeq/H/dtype
MLATiling::GetMLATilingParam(info, blockDim, buf);
// tilingKey 规则（launch 前设置）：
//   H != 128          → 0/1（MLA 通用，fp16/bf16）
//   H == 128 且 AMLA  → 4/5；当 numTokens%20<=10 且 batch<=40 时改走 7/8（MLATp1Spec）
```

**oCoreTmp / l 的尺寸依赖 tiling 结果**：先跑 `GetMLATilingParam`，从 `tiling[TILING_KVCORENUM=16]` 槽位读回 kvCoreNum 再分配这两个 buffer——不能在 tiling 之前按 aicCoreNum 猜。

## 3. workspace 分配表（实测验证的最小尺寸）

| buffer | dtype | 尺寸 |
|---|---|---|
| `s` | f32 | `aicCoreNum × 65536 × 2` |
| `p` | f16 | `aicCoreNum × 65536 × 2` |
| `oTmp` | f32 | `aicCoreNum × 65536 × 2` |
| `globalo` | f16 | `aicCoreNum × 65536 × 1` |
| `oCoreTmp` / `l` | - | 按 tiling 读回的 kvCoreNum 分配（§2） |

## 4. shape / 语义约束（host 侧必须前置校验）

- **算子语义**：`O = softmax(scale·(Q·c_kv + q_rope·k_rope))·V_latent`，`scale=1/√576`（576 = Dc+Dr，不是 √Dc）
- `q_seqlen ∈ [1,4]`（decode/MTP；官方 ATB 算子只支持 q=1，本内核支持 MTP q>1——这是对标杆标杆的反超点之一）
- `kv_seqlen ≥ 128`，`blockSize = 128`（尾块非对齐 OK）。★官方 gen_data 声明范围 [128,16384] **是验证范围非硬约束**——实测本内核与 ATB 标杆均可正确运行至 kv=204800（20万，精度同带 PASS，2026-08-24 长序列矩阵验证）；超 16384 后官方未验证，交付前须自测精度
- `Dc = 512`、`Dr = 64`
- **V 输入是 latent（c_kv 压缩域），不是解压后的 V**——语义写错 golden 直接错
- H ∈ {16, 32, 64, 128}，H=128 走 TP1 spec 路径（tilingKey 见 §2）

## 5. 精度验证

- verify 脚本按本 skill 模板编写（判定逻辑必须从模板复制），golden 用 numpy 纯 fp32 分解链
- 实测 18/18 语义矩阵 PASS（error_ratio=0，max_abs 1.3e-5 ~ 3.1e-4 vs 纯 fp32）；泛化 20 case 同带
- **三方互证**：本内核 vs 标杆 `npu_multi_head_latent_attention`（ATB）vs numpy golden——两两差异同数值量级（1.5e-5~1.2e-4）即互证通过

## 6. 性能基线与吞吐模型（910B3 实测，device 口径）

| 口径 | 结果 |
|---|---|
| vs torch_npu 分解链（gather+einsum+softmax+einsum） | 几何 33.5×（**墙钟口径，含 host 间隙，报告必须明示"融合 vs 分解"**） |
| vs 标杆 ATB `npu_multi_head_latent_attention` | **序列长扫描 gate 3.047**（同设备复测版；B2 固定、kv 128~204800：短 kv≤4096 持平 ~1.0，kv≥6144 ATB 线性退化 ~23μs/1k token、本侧 kv-split 平坦 ~3μs/1k → ratio 升至 **7.16**@kv20万，ATB 复测三次差 <1%）；batch 放大口径 gate 1.161（B2-10 反超 1.16~4.2×；B12-16 波谷 0.66-0.68；B22+ 持平） |
| 长 kv（单 batch 大 kv） | 本内核反快：kv8192 快 3.7×、kv16384 快 5×（ATB 长序列调度退化） |
| decode kernel 绝对时长 | 12.9 ~ 80.7 us（语义矩阵） |
| ★kvSplitCores 台阶 | kv≤4096 = 1（串行扫块）→ kv≥6144 = 8（并行切分）：kv=6144 比 kv=4096 快是 tiling 策略跳档的真实台阶，非测量误差（runner tiling 行实证） |

**吞吐定标模型**（构造/估时 case 用，见 [fa-perf-gate-evaluation.md](fa-perf-gate-evaluation.md)）：
`U = Σ T·H·kv_b`（score 单元数），饱和吞吐 H128≈64-72、H64≈33-42、H32≈20 units/ns。
- H32 慢 3×：通用内核无 K 常驻 L1，与 TP1 spec 差距的主因
- batch 增大吞吐升（B8→B40，H128 54→72 units/ns）——多 batch 是免费性能，评测矩阵应覆盖
- 时间与 B 非线性（kv-split 波次效应），定标需实测 2 轮迭代收敛

## 7. 已知问题

| 现象 | 根因 | 对策 |
|---|---|---|
| 官方 ATB 标杆 event 计时 4061us，device 实际 76us | ATB 每次 launch 重建 executor，host 开销巨大 | 标杆一律 profiler op_summary device 口径（见评测 doc 陷阱 2） |
| ATB 标杆跑 MTP（q>1）报错 | 官方算子仅支持 q=1 | 该 case 用 golden-only 验证；性能对比注明不可比 |
| NNAL/ATB 不可用（`libatb.so` load 失败） | 环境未装 NNAL | mindspore 镜像下 run 包 + `yes \|` 过 EULA 安装；装不了则报告明示"标杆对比不可做" |

---

## 8. MLA kernel 实现架构（手搓/复刻指南——从零写出 MLAKernel 的完整依据）

> **★权威源升级**：本节是设计依据（组件链/5 支柱/主循环伪代码/tiling 决策）。**实际生成代码以
> [fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md) 为准**（MLAKernel 逐行配方，
> 含 §8 本节缺的两个死锁关键：`__DAV_CUBE__/__DAV_VEC__` 预处理分支、prologue+延迟 PV 流水）。
> 本节 §8.10 组件链/§8.11 主循环伪代码与 recipe 一致时以 recipe 为准；不一致时**以 recipe 为准**
> （recipe 已实测纠错）。
>
> 目的：按本节设计依据 + recipe 的真实代码结构，从零写出性能达标的 `MLAKernel`（gate 1.16+）。
> host 契约见 §1-§4，FA 通用手搓细节见 [fa-kernel-handcraft.md](fa-kernel-handcraft.md) §1-9。

### 8.1 MLA 数学 → kernel 结构映射

```
O = softmax(scale · (Q·c_kv + q_rope·k_rope)) · V_latent
  c_kv:  latent 压缩 KV（key=value 共用），Dc=512
  q_rope/k_rope:  RoPE 拆分部分，Dr=64
  scale = 1/√576（Dc+Dr，不是 √Dc）
  → 关键：QK 是「两段点积相加」，不是单一 Q·K^T；V 就是 c_kv 本身
```

### 8.2 Params（14 参，与 launch 一一对应）

`q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling`（13 数据 + tiling 槽）。注意比 FA 多 **qRope/kRope**（RoPE 拆分是 MLA 特有）和 **oCoreTmp/l/oUpdate**（flash-decoding 归并用）。GM buffer 声明：`GlobalTensor<ElementS/P/OTmp/uint32_t>`（s/p/oTmp）+ `ElementQ/K/.../int32_t`（数据）+ `ElementO/OTmp`（AIV 输出）。

### 8.3 任务切分（AIC/AIV 共用同一遍历，flash-decoding 三重索引）

```
for (process = coreIdx; process < processNum; process += coreNum):
    curBatch     = process / (qHeadSplitNum × maxKvSplitCoreNum)
    qHeadSplitIdx = process % (qHeadSplitNum × maxKvSplitCoreNum) / maxKvSplitCoreNum
    curNIdx      = process % maxKvSplitCoreNum        // ← kv-split 段索引
    curKVSeqlen  = kvSplitPerCore（末段=剩余 kvSeqlen）
    rowNum = qHeadSplitSizeActual × qSeqlen
    nLoop  = (curKVSeqlen + blockSize-1) / blockSize  // 每核扫的 KV 块数
```
- 任务数 = batch × qHeadSplitNum × kvSplitCoreNum，`processNum` 由 tiling 给
- 每核独立扫自己那段 KV，算 partial O/l/m
- 行间 pingpong 交替调度：`isForward = !isForward`（把连续核的任务交错，均衡 L1 复用）

### 8.4 kv-split（flash-decoding）并行——MLA 长序列性能的关键

- tiling 算 `kvSplitPerCore`（每核 KV 段长）、`kvSplitCoreNum`（切几段，≤8）
- AIC 每核算自己段的 `S_j = QK_j`，AIV 做该段的 **online softmax 但保留 running max/sum 到 partial**（不跨段归一）
- **flash-decoding 归并**（`epilogueMLAFDRescaleO`）：所有核算完后，AIV 扫全部 `kvSplitCoreNum` 段的 partial，按 `w = exp(m_j − m_final)` 缩放各段 O 再累加，最后 `÷l`——这就是 kv-split 正确性的核心，**漏了它就错数**
- 归并读 `gOCoreTmp[oFdOffset × kvSplitCoreNum + ...]`（per-split 槽）+ `gl[lOffset × ...]`（per-split l）

### 8.5 AIC/AIV 跨核握手（3 个 CrossCoreFlag，modeId=0x2）

```
AIC 算 QK → SetFlag(qkReady, PIPE_FIX)   → AIV WaitFlag(qkReady) → 算 softmax
AIV 算 P   → SetFlag(softmaxReady, PIPE_MTE3) → AIC WaitFlag → 算 PV
AIC 算 PV  → SetFlag(pvReady, PIPE_FIX)   → AIV WaitFlag → 累加 O
```
- flag 对象：`qkReady/softmaxReady/pvReady`（id 常量 QK_READY_ID/SOFTMAX_READY_ID/PV_READY_ID）
- **modeId 必须 0x2**（双子核都要发/收，见 [fa-handcraft-pitfalls.md](fa-handcraft-pitfalls.md) H2）；`SetFlag` 指定 PIPE（FIX/MTE3）与 flag 事件要配对

### 8.6 HardEvent 预置（AIC/AIV 各自入口，逐事件 ID 写全）

```cpp
// AIC（25 个）：入口按此顺序 SetFlag，出口逆序 WaitFlag
M_MTE1:   EVENT_ID0..7            (8 个)
FIX_M:    EVENT_ID0, EVENT_ID1    (2 个)
MTE1_MTE2: EVENT_ID0..7           (8 个)
FIX_MTE1: EVENT_ID0..5            (6 个)
MTE2_FIX: EVENT_ID0               (1 个)
// AIV（15 个）：
MTE3_V:   EVENT_ID0, EVENT_ID2, EVENT_ID3      (3 个)
MTE3_MTE2: EVENT_ID0, EVENT_ID2..7             (7 个)
V_MTE2:   EVENT_ID0, EVENT_ID1, EVENT_ID4, EVENT_ID2, EVENT_ID3  (5 个)
```
- 漏一个或顺序错 → 流水死锁（与 FA kernel 同规则）；kernel 出口逐一 WaitFlag 配平。

### 8.7 双特化 + 模板链（写 kernel 时的骨架）

```cpp
template <class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax,
          class EpilogueMLARescaleO, class EpilogueMLAFDRescaleO>
class MLAKernel {
    using ElementS=BlockMmadQK::ElementC; using LayoutS=...;  // 派生所有类型
    struct Params {...};                    // 14 参
    CATLASS_DEVICE void operator()(Params&); // #ifdef __DAV_CUBE__/__DAV_VEC__ 双分支
    Arch::CrossCoreFlag qkReady/softmaxReady/pvReady;
};
// 外层 MLA<T> 模板实例化: dispatch MmadAtlasA2MLAQK/PV + MLASoftmax/RescaleO/FDRescaleO
```
- dispatch 固化 `MmadAtlasA2MLAQK`（QK 两段：Q·c_kv 与 q_rope·k_rope）/ `MmadAtlasA2MLAPV`（P·V）
- H=128 走 TP1 spec（`mla_kernel_tp1_spec.cpp`，4-block 基块、无 K 常驻）；AMLA 变体用 `amla_kernel_tp1_spec.cpp`

### 8.8 手搓自查（写完后逐条过）

- [ ] 14 参 launch 与 Params 字段一一对应
- [ ] 任务切分：process 遍历 + curBatch/qHeadSplitIdx/curNIdx 三重索引正确
- [ ] kv-split：kvSplitPerCore/kvSplitCoreNum 从 tiling 读、末段剩余长度处理
- [ ] flash-decoding 归并：per-split O/l 槽 + 跨段 rescale 累加（漏了错数）
- [ ] 3 个 CrossCoreFlag modeId=0x2、PIPE 配对、双子核都发
- [ ] HardEvent 入口 SetFlag 数量/顺序与出口 WaitFlag 配对
- [ ] q_rope/k_rope 正确参与 QK（两段点积相加）
- [ ] Dc=512/Dr=64/scale=1/√576/V=latent 语义正确

> 完成 §8 后，可直接按 [fa-perf-gate-evaluation.md](fa-perf-gate-evaluation.md) 评测。

### 8.13 依赖清单与闭卷实证（纯 md 生成的可行性证据）

- **依赖仅两样**：catlass 库头（`catlass/include`，§8.10 组件链 include 的全部 `catlass/*` 头）+ ASC 工具链（`kernel_operator.h` 自动提供）。无其它本地依赖，**不读任何外部样例**。
- **闭卷冒烟实证（2026-08-25，本节架构）**：全新工程只按契约+组件链从零组装：一次编译通过；B2/H16 kv=128/4096/6144/16384 四点精度 PASS（err_ratio=0，max_abs 1.5e-5~8.9e-5）；`kvSplitCores` 台阶正确（kv≤4096=1、kv≥6144=8）。
- **精度与性能全量证据**：语义矩阵 18/18 + 序列长分桶 20/20 PASS（含 MTP/尾块非对齐/bf16/varlen）；kv=204800（20万）精度同带 PASS（官方声明范围仅 16384，实测远超）；vs 标杆 ATB 序列长 gate 3.116（kv20万 时 7.44×）、batch 放大 1.161。

### 8.14 ★tiling 来源坑（M7 级元坑：验证的是链接的文件，不是目录里的文件）

实测案例：某验证工程 CMake `MLA_KERNEL_SOURCES` 实际链接的是**外部仓的 tiling**——工程根目录同名文件是历史手搓期修改版、**从未被链接使用**；据此打包的资产行为完全不同（kvSplit 恒 20、小 kv 更慢）。**教训：任何"以工程为事实源"的移植/打包前，必须核对 build 实际链接的文件**（CMake 源列表指向 + quoted-include 解析顺序），不能只看工程目录里有什么。

### 8.9 ★为什么"朴素手搓"慢一个数量级——快内核的 5 个性能支柱（对照表）

实测对照：朴素手搓（FAQK 组件 + 双 launch + GM 物化）gate **0.13~0.36**；本节架构的内核 **1.16+**。差距全在架构，不在微调：

| # | 性能支柱（快内核） | 朴素手搓的对应错误 | 量级影响 |
|---|---|---|---|
| 1 | **单 kernel AIC/AIV 锁步**（一次 launch，3 个 CrossCoreFlag 逐块握手） | 双 launch（A 算 QK+softmax、B 算 PV+归并），launch 间隙+两遍 GM 往返 | ~2× |
| 2 | **S/P 只在 per-core GM 小槽轮转**（TMP_SIZE=64K/半区 32K，双缓冲），P 不物化全量 | P_unnorm 全量 [rows×kv] 写 GM 再读回 | 内存带宽型，kv 越大越差 |
| 3 | **L1TileShape K=576=embed+rope**：K 的 L1 装载覆盖整个 embedding，Q 块复用 | 通用 FAQK 分块（K=128），K 反复重装 | cube 利用率 |
| 4 | **专用 dispatch**：`MmadAtlasA2MLAQK`（Q·c_kv 与 q_rope·k_rope 两段在组件内一次完成）+ `MmadAtlasA2MLAPV` | 用通用 QK 组件+手工拼两段点积（多一次 GM 中转） | ~1.3× |
| 5 | **flash-decoding kv-split**（长 KV 切多核 + FDRescaleO 归并） | 单核串行扫全 KV | 长序列 3-7× |

**结论：MLA 手搓必须照搬这 5 条**（组件全在 catlass include 层），否则必慢一个数量级。

### 8.10 ★组件选型与实例化链（逐行照写；全部 catlass include 层组件）

```cpp
// dtype: half / bfloat16_t；S/OTmp/Update = float；Q/K/V/P/O = Dtype；Layout 全 RowMajor(K=ColumnMajor)
using L1TileShape = GemmShape<128, 128, 576>;          // ★K 必须 = embed+embedRope(512+64)=576（K 常驻 L1 的实现方式）
using L0TileShape = L1TileShape;
using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;        // QK 两段点积一次完成
using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
using EpilogueMLASoftmax = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;
using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
using EpilogueMLARescaleO = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;
constexpr uint32_t ComputeEleNum = 6144;                // FD 归并的每次处理元素数
using EpilogueMLAFDRescaleO = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;
```
- H=128 特化：`mla_kernel_tp1_spec.cpp`（4-block 基块、无 K 常驻路径）；AMLA 变体同目录。手搓按本节通用路径，H=128 时参考特化的基块差异。

### 8.11 ★主循环逐步（性能关键：三处流水预取 + 全局 pingpongIdx）

常量：`TMP_SIZE=65536`（P/Update 槽）、`TMP_SIZE_DECODER=32768`（S 槽）、`BLOCK_SIZE=16`、flag id 1/2/3。

```
task 遍历（process = coreIdx; process < processNum; process += coreNum）：
  ├─ 正逆交替：realProcess = isForward ? process : (组内最大 process − process%coreNum)；isForward 翻转
  │   （把相邻任务的 KV 块顺序倒过来 → 同批 KV 的 L1 驻留被下一任务复用）
  ├─ kerFlag 翻转：curNIdx = maxKvSplitCoreNum−1−curNIdx（每两任务镜像 split 段，均衡尾段负载）
  ├─ 三重索引 → curBatch/qHeadSplitIdx/curNIdx；rowNum = qHeadSplitSizeActual × qSeqlen
  │
  ├─ [首任务 prologue] 发起块 0 的 QK → Set(qkReady)；isFirstTask=false
  │
  └─ for nIdx = 1 .. nLoop：                    ★1 块深流水
      ├─ (nIdx<nLoop) 发 QK(nIdx)：S → 槽[pp%2] → Set(qkReady)
      │    block_table 查表：blockId = table[batch*maxBlocks + startKV/128 + nIdx]
      │    kvOffset = blockId × 128 × strideKV（K/KRope 各一）
      ├─ (nIdx==nLoop 且有下个 process) ★任务级预取：向前扫描 coreNum 步找到下个有效 process，
      │    发它的块 0 QK（流水跨任务不断流）
      └─ 发 PV(nIdx−1)：P 槽[(pp−1)%2] → OTmp → Set(pvReady)
  AIV 侧对称：Wait(qkReady) → epilogueMLASoftmax(S→P) → Set(softmaxReady)
              Wait(pvReady) → epilogueMLARescaleO(O 累加/÷l；isLastNTile 时写 FD partial 槽)
（PV 的 blockMmadPV 调用把 softmaxReady 作为入参传入——组件内部 Wait 它保证 P 就绪后再 MMAD；
  locPingPongIdx 为 AIC 侧本地乒乓，与全局 pp 独立维护）
```

**GM 槽位公式（pp = 全局 pingpongIdx，跨任务连续递增）**：
```
S 槽    = coreIdx × TMP_SIZE_DECODER + (pp%2) × TMP_SIZE_DECODER/2
P 槽    = coreIdx × TMP_SIZE        + (pp%2) × TMP_SIZE/2
OTmp 槽 = coreIdx × TMP_SIZE×2      + ((pp−1)%2) × TMP_SIZE
```
- Q 指针不走 params：host 把**每 batch 的 Q/QRope/O/l/oFd 64 位地址拆 hi/lo 32 位写进 tiling 槽**（kernel `GetValue` 高低位拼回），params 只给各 tensor 基址——per-batch 地址差异全由 tiling 承载。
- blockMmadQK 调用形态：`blockMmadQK(gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope], gS[gSOffset], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, GemmCoord{rowNum, kSeqTile, embed+embedRope}, MatrixCoord{qHeadSplitSizeActual, embed}, qHeads, nIdx, pingpongIdx)`——**5 张量 + 5 layout + coord 三件**（QK 组件吃双 Q/K 对）。

**flash-decoding 归并（任务循环后、AIV 侧收尾）**：
```
headsProcess = min(ComputeEleNum/embed, HEADS_PROCESS_MAX)；loopsPerBatch = ⌈qHeads/headsProcess⌉
逐 loop 调 EpilogueMLAFDRescaleO(gO[...], gOCoreTmp[oFd 槽], gl[l 槽], actualHeads, headsProcess, embed, kvSplitCoreNum)
  —— w = exp(m_j − m_final) 跨 split 段缩放累加 + ÷l 一次完成
```

### 8.12 ★tiling 槽位表与 kv-split 决策（host 逐槽填）

头部槽（uint32 下标）：`0 batch, 1 numHeads, 2 headDim(512), 3 numBlocks, 4 blockSize(128), 5 maxBlocks, 6 tor(float 重解释), 7 kvHeads, 8 TILING_HEAD_SIZE(=头部长度), 9 TILING_PARASIZE(=per-batch 段长), 10 headSplitSize, 11 headSplitNum, 13 headDimRope(64), 14 maxKvSeqlen, 15 kvSplit, 16 kvCoreNum, 18 totalQTokens, 19 processNum`。
per-batch 段（基址 = 槽 8 的值 + 槽 9 的值 × b）：`+0 qSeqlen, +1 kvSeqlen, +2 realBatch(排序后原 batch 号), +4/5 qAddr hi/lo, +6/7 qRopeAddr hi/lo, +11/12 lAddr hi/lo, +13/14 oFdAddr hi/lo, +15 kvSplitPerCore, +16 kvSplitCoreNum`。

**kv-split 决策（T+1 均衡法，逐句可写）**：
```
isKVSplit = (maxKvSeqlen ≥ blockDim×2×128) && (batch ≤ blockDim×0.8) && (maxQSeqlen == 1)   // KV_SEQLEN_SLICE=128, SPLITKV_RATION=0.8
不切：每 batch kvSplitCoreNum=1、kvSplitPerCore=kvSeqlen，processNum=batch
切：MIN_TOKENS_PER_SPLIT = 4×blockSize；MAX_SPLIT_PER_TASK = 8
    batchSplitNum 全 1，totalAllocated=batch
    while totalAllocated < blockDim:
        选 currentLoad=alignedKv[i]/split[i] 最大且 (split[i]<8 且 nextLoad≥MIN_TOKENS) 的 i → split[i]++
    kvSplitPerCore = ⌈⌈kv/128⌉/split⌉ × 128；kvSplitCoreNum = ⌈kv/kvSplitPerCore⌉
    processNum = Σ kvSplitCoreNum；槽 16 填 MAX(split)
```
（batch 按 kvSeqLen 降序排序后填段——长 KV 先分；realBatch 槽携带原 batch 号供 block_table 寻址。）
