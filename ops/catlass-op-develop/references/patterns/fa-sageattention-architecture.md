# FA-SageAttention 完整架构规格（Ascend950 / dav-3510）

> 本文档含复刻该算子所需的全部精确细节：公式、内存布局（到字节）、
> 同步协议（到 flag id）、循环结构（到游标语义）。

## 0. 硬件前提

| 资源 | 容量 | 备注 |
|------|------|------|
| AIC L0A/L0B | 各 64KB | 两 block 槽位从 0 布局, 共享事件 id, **槽距必须相等** |
| AIC L0C | 256KB | 由构造函数 cursor 顺序分区 |
| L1 | 512KB | pL1(P staging) + QK L1A/L1B + PV L1B |
| AIV UB | 248KB/核 | 每 AIV 子核管 64 行, 各自独立布局 |
| 平台探测 | aicNum≈28, aivNum≈56 (PlatformAscendCManager) | 数量随时可能变, 需打印 |

```
__mix__(1, 2) 组成: 1 AIC + 2 AIV 子核; 每 AIV 子核管 64 行 (SH=64)
AIC 的 blockIdx = GetBlockIdx(); AIV 的 blockIdx = GetBlockIdx() >> 1
AIV 子核区分: GetSubBlockIdx() ∈ {0,1} (仅 mix kernel 有效!)
```

## 1. 数据布局契约（GM）

| 张量 | 布局 | 说明 |
|------|------|------|
| Q̂ (qhat) | BSND [B·sqPad, H·D] int8 | 量化后 Q, pad 行零 |
| K̂ (khat) | BSND [B·kvBlocks·128, Hkv·D] int8 | 平滑后量化 K |
| μ (muOut) | [B·Hkv·kvBlocks, D·8] fp32 | **fp32 原值 ×8 冗余** (写 4KB 可靠, 读 512B 即可) |
| SQ/SK scale | [blocks, 8] fp32 | ×8 冗余; **FA 读时下标必须 ×8** |
| V | BSND [B·skPad, Hkv·D] fp16 | host 从 HND 转 |
| O | HND [B,H,Sq,D] fp16 | kernel 直写 |
| po (splitK partial O) | [totalWork][128][128] fp32 | 未除 l 的 oacc, 行距 128 有效列 D |
| pml (partial m/l) | [totalWork][512] fp32 | {sub0: m[64]\|l[64]\|pad, sub1: 同} 每 sub 1KB |

## 2. Kernel A：SageQuantKernel（`__global__ __vector__` AIV-only）

工作单元 = K 块 (b,kvh,j) 共 B·Hkv·kvBlocks 个，或 Q 块 (b,h,i) 共 B·H·sqBlocks 个；
`for (u = GetBlockIdx(); u < total; u += GetBlockNum())` 分摊。

### K 单元
```
1) GM fp16 → UB half (DataCopyPad 2D, QKB×D) → Cast fp32 (两级: half→fp32 用 count-Cast)
2) μ = 列均值: 树归约 (Add 折半, ua/ub 乒乓, n 从 (QKB/2)·D 折到 D)
   → Muls(um, src, 1/nValid)   # nValid = min(128, sk−j·128)
   # 注意: 树归约会破坏 K 原值, 需从 uh 重新 Cast 恢复
3) smax = max(amax|K−μ| (寄存器行 max + 乒乓), amax|μ|)
4) sK = SageDivScr(127, smax)  # device 标量除法 helper, 见 §6
5) 量化: (K−μ)·sK → Cast RINT → fp16 → int8 (count-Cast 两级)
6) 写 GM: 128 行 2D strided; μ 复制 8 份 (Muls×8) 写 muOut 4KB
```

### Q 单元
同上去掉 μ/sK 路径；sQ = 127/amax(|Q|)。

## 3. Kernel B：SageFaKernel（`__mix__(1, 2)`）

### 3.1 catlass 组件选型（精确 Shape）

```cpp
using ArchTag = Arch::Ascend950;
// QK: FAIQK 读 Shape=(M, N, K) —— 注意与 FAIPV 相反!
using L1TSQK = tla::Shape<Int<128>, Int<128>, Int<256>>;  // K=256 是垫的!
//   A 槽 = M·K_static·1 = 32KB, B 槽 = K_static·N·1 = 32KB
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, int8_t, layout::RowMajor,
    int8_t, layout::ColumnMajor, int32_t, layout::RowMajor, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;   // SPLIT_M: fixpipe 双 AIV 各得半行
using BlockMmadQK = Gemm::Block::BlockMmadTla<Gemm::MmadFAIQK<ArchTag,false>,
    L1TSQK, L1TSQK, int8_t, int8_t, int32_t, void, TileCopyQK, TileMmadQK>;

// PV: FAIPV 读 Shape=(M, K, N)
using L1TSPV = tla::Shape<Int<128>, Int<128>, Int<128>>;
//   A 槽 = M·K·2 = 32KB ✓ 与 QK 对齐 (不变量 I1)
using TileCopyPV = ... <half, layout::zN, half, layout::RowMajor, float, RowMajor, SPLIT_M>;
using BlockMmadPV = ... MmadFAIPV<ArchTag,false>, L1TSPV, half, half, float ...>;

// L0C 分区 (cursor 顺序): QK [0,128KB) (64KB×2 stage), PV [128,256KB) — 恰好占满
```

**为什么 K_static=256**：L0A/L0B 槽距 = 静态 Shape 算出的字节数。QK 是 int8（128×128=16KB），
PV 是 half（128×128×2=32KB）。槽距不等 + 共享事件 id {0..3} → 事件计数错配 → 竞态。
垫 K=256 使 QK A 槽 = 128×256×1 = 32KB = PV 的 32KB。运行时 blockK=d=128 不受影响。

### 3.2 UB 布局（每 AIV 子核, 字节）

```
UB_S_OFF    = 0                    S int32 [64,128] ×2 slot (m2)     2×32KB
UB_O_OFF    = +64KB                O fp32  [64,128] ×2 slot (m2)     2×32KB
UB_OACC_OFF = +128KB               oacc fp32 [64,128]                 32KB
UB_O16_OFF  = +160KB               O fp16 staging                     16KB
UB_P_OFF    = +176KB               P fp16 [64,128] ×2 slot (m2)      2×16KB
UB_QF_OFF   = +208KB               Q̂ 切片: int8 8KB + half 8KB + fp32 16KB (chunk 式)
UB_MU_OFF   = +232KB               μ fp32 [128]
UB_ST_OFF   = +233KB               状态: 12 槽×[64] fp32 (lastMax/sumL/alpha 各 4 槽)
UB_SCR_OFF  = +236KB               nowMax + expSum 各 [64]
UB_DIV_OFF  = ...                  SageDivScr scratch 3×8 fp32
UB_COMP/THR/CIDX = ...             comp[64] / mask 阈值[64] / 列索引[128]
合计 ≈ 240KB ≤ 248KB
```

### 3.3 L1 布局

```
pL1[m3] ×4 slot: [128,128] half P staging (zN)    4×32KB = 128KB
QK L1A: 128×256 int8 ×2 stage                     64KB   (cursor: 128→192KB)
QK L1B: 128×256 int8 ×2 stage                     64KB   (192→256KB... 实际 N×K 布局)
PV L1B: [128,128] half ×2 stage                   64KB   (256→320KB... )
```

### 3.4 CrossCore flag 表（mode 4）

```
C1_V1    [2] = {0, 1}     AIC QK done → AIV SM 开始
V1_C2    [4] = {2,3,4,11} AIV P 已进 L1 → AIC PV 开始 (m3 = step%4)
C2_V2    [2] = {5, 6}     AIC PV done → AIV RS 开始 (m2 = step%2)
MM2      [2] = {7, 8}     AIV 消费完 oUb → AIC 下次 PV fixpipe 可写
MM1      [2] = {9, 10}    AIV 消费完 sUb → AIC 下次 QK fixpipe 可写
PV_CONSUMED[4] = {12..15} AIC PV 用完 pL1 → AIV SM 可覆盖 (pL1 槽复用保护)
```
调用规则（I6）：AIC set `id` 和 `16+id`；AIV set 单 `id`；AIC wait `id` 和 `16+id`。
kernel Init 时 AIV 预置 set MM1/MM2 各 id（bootstrap 首次使用）。

### 3.5 主循环（4 相位流水 + split-KV）

```
for (it = 0; it < P + 3; ++it) {
  AIC: QK(it)      it < P && cursor.Next(curQK)
       PV(it-2)    it≥2 && it-2 < P && cursor.Next(curPV)
  AIV: SM(it-1)    it≥1 && it-1 < P && cursor.Next(curSM)
       RS(it-3)    it≥3 && it-3 < P && cursor.Next(curRS)
}
P = 本核所有 work 的步数和; 每 work = (task, shard)
```

**游标（精确语义, 曾出 bug）**：
```cpp
struct SageCursor {
  int64_t k = 0, j = -1;   // 内部: j=-1 表示新 work 待重算起始
  int64_t jCur, jStart, gCur, jEnd, bh, iq, kvhG;
};
// work g → task = g/SK, shard = g%SK
// jEndF = SageJEnd(task)  (causal 时 = (iq·128+127+sk-sq)/128+1, 否则 kvBlocks)
// jStart = jEndF·shard/SK;  jEnd = jEndF·(shard+1)/SK   (floor 均分, 差≤1 无空片*)
// 每次 Next: jCur = j (本步); j++; j≥jEnd → k++, j=-1
// ★ 调用方一律用 c.jCur (返回值是前进后的内部 j!)
// ★ isFirst 判据 = (jCur == jStart)  不是 ==0  (split-K 时 shard 起步非 0!)
```

**SM 相位（AIV, 每 step）**：
```
wait C1_V1[m2]
qk8: Q̂ 本子核切片 int8 GM→UB (2D strided, srcStride=(H-1)·D)
μ_j: GM→UB 512B (下标 (kvhG·kvBlocks+jCur)·D·8)
cScale = sm/sQ (每 work 缓存, jCur==jStart 时重算)
chunk×2: Cast int8→half→fp32 (32行/块) → SageVfCompDot:
    comp[r] = ReduceSum(qff[r,:] ⊙ μ)·cScale    (fp32, 整数 dot ≤2M 精确)
needMask 时: thrUb.SetValue(r, lim-1) 逐行阈值 (causal: lim=col_row+1 上界)
SageVfDeqMaskMax: S = S_int·dScale + comp + mask; 行 max → nowMax
!isFirst: UpdMax(nowMax, lastMax[(m3+3)%4])     ← I5 链读槽
SageVfExpP: P = exp(S−nowMax), 行和 → expSum
WaitFlag<MTE3_V>(m3Prev)  ← 上次 CopyUb2L1 读 pUb 完成保护
Cast P fp32→fp16 (count-Cast)
!isFirst: UpdL(nm, lmPrev→lmW 本槽, alpha[m3], sumPrev→sum[m3])   ← I5 分槽
else:      CopyF(lastMax[m3], nowMax); CopyF(sumL[m3], expSum)
WaitFlag PV_CONSUMED[m3]  ← pL1 覆盖保护 (I8)
CopyUb2L1Tla: pUb[m2][64,128] fp16 RowMajor → pL1[m3] zN (本子核 64 行片)
set MM1[m2]; set V1_C2[m3]
```

**PV 相位（AIC）**：
```
wait V1_C2[m3] (id 和 16+id)
mmadPV(pL1[m3] zN [128,128], V tile [128,D], oUb[m2], shape=(128,D,128), taskId=m2)
     单次调用 n=D (FAIPV 内部 nLoop 处理 D=64 尾块)
set C2_V2[m2] (id 和 16+id); set PV_CONSUMED[m3]
```

**RS 相位（AIV, 每 step）**：
```
wait C2_V2[m2]
SageVfRescale: oacc = oacc·alpha[m3] + oUb[m2]   (行距 = D, 与 PV fixpipe 紧凑布局一致!)
isLast:
  splitK>1: 写 partial — po[gCur][row0..][0..D) fp32 2D strided (dstStride=(128-D)·4)
            + pml[gCur][sub·256] = {m=lastMax[m3], l=sumL[m3]} 各64 fp32 + pad → 1KB
            + PipeBarrier<PIPE_ALL>()  ← kernel 退出不保证 MTE3 排空!
  else:     DivL(oacc /= l) → Cast fp16 → GM 1D (validRows·D·2)
set MM2[m2]   ← 只 set 本步槽! (I3)
```

### 3.6 QK 相位（AIC）

```
tensorQ = GetTile(qhat, (b·sqPad + iq·128, h·D), (128, D))
tensorK = GetTile(khat 列主视图, (kvh·D, (kvhG/hkv)·kvBlocks·128 + jCur·128), (D, 128))
mmadQK(tensorQ, tensorK, sUb[m2], ..., taskId=m2,
      qkFirst=(jCur==jStart), qkLast=(jCur==jEnd-1))
  // qkFirst: GM→L1A 装 Q̂ (borrow: 任务内只装一次)
  // qkLast:  轮转 L1A stage
set C1_V1[m2] (id 和 16+id)
```

## 4. SageMergeKernel（`__global__ __vector__` AIV-only, split-KV 归并）

```
映射 (★ __vector__ 无 sub-block 概念, GetSubBlockIdx 恒 0):
  sub = GetBlockIdx() & 1;  task t = GetBlockIdx() >> 1;  步长 = GetBlockNum() >> 1

per task:
  1) 一次性整块载入本 task 全部分片 m/l (works 连续: pml[t·SK·512, +SK·512))
     → PipeBarrier (★勿用连续小拷贝+复用 event id, 实测不可靠)
  2) 紧凑化: 有效分片 (jStart<jEnd) 前移到前 nValid 槽
     (★ nValid==k 时只计数不拷贝, 连 continue 会跳过计数 → 首分片被覆盖)
  3) m* = max_k m_k; w_k = exp(m_k−m*) 原位; l* = Σ w_k·l_k
  4) oacc = Σ w_k·po_k[r,:]   (★ acc += w·po 用 Mul+Add; MulDstAdd 是 a·dst+b!)
  5) O = oacc/l* → fp16 → GM
```

## 5. Host（.asc 单 TU）

```
tiling: b/h/hkv/sq/sk/sqPad/skPad/d/sqBlocks/kvBlocks/totalTasks/
        splitK/totalWork/coreNum/groupSize/smScale/isCausal/quantScheme
splitK = ceil(aicNum / totalTasks), cap 到 kvBlocks, ≥1   (ceil 避免空闲核)
coreNum = min(aicNum, totalWork)
launch: SageQuantKernel<<<aivNum>>> → SageFaKernel<<<coreNum>>> → (splitK>1) SageMergeKernel<<<aivNum>>>
缓冲: qhat/khat/mu/sq/sk/o + po[totalWork·128·128·4] + pml[totalWork·512·4]
★ kernel 内 td 是逐字段从 GM 拷贝的 — 新增 tiling 字段必须加进拷贝列表!
```

## 6. 关键 helper

```cpp
// device 标量除法 (ccec 无法 lower 标量 fp div)
SageDivScr(a, b, scrA, scrB, scrDst):
  Duplicate(scrA, a, 8); Duplicate(scrB, b==0?1:b, 8);
  Div(scrDst, scrA, scrB, 8);          // count≥8, 禁原地
  PipeBarrier<PIPE_ALL>();             // Div(向量)→GetValue(标量) 需同步
  return scrDst.GetValue(0);
```

## 7. 性能现状与优化路线（已实测）

| 项 | 数据 |
|----|------|
| 每流水迭代 | ~10-11us/核 (满负载), 其中 ~6 次跨核 flag 往返 × 1.5-2us 未被流水隐藏 |
| 已做 | 去 V→V PipeBarrier(+9%), Q̂ L1A borrow, cScale 缓存 |
| split-KV | k1024: 81→43.5us（标杆 FA 25.2）— 短序列 ratio 0.67 |
| 路线① | KV tile 加宽到 256: QK 单 mmad 两块, 迭代数减半; L0C 需 QK 单缓冲 128K+PV 128K=256K 恰好放下 |
| 路线② | 加深 sUb/oUb 槽 (2→4) 让 MM1/MM2 等待预先满足 (UB +160KB 需先砍 QF 区) |
| 路线③ | S fixpipe 走 DEQF16 (int32→fp16 带 scale, catlass 支持) 省 32KB/步流量 |

## 8. 新版 catlass 适配注记（2026-08 重建上板，详见 fa-sageattention-troubleshooting.md §F）

本规格描述的是**算法/数据流设计**（数学公式、内存布局、flag 协议、不变量 I1-I8），与 catlass 版本无关。但**新版 catlass 的 `BlockMmadTla` 实现行为变了**，重建实现时须按以下适配（原旧版 catlass 的对应行为不同）：

- **MM1/MM2 wait 改由 catlass 内部完成**：新版 `BlockMmadTla`（FAIQK/FAIPV）内部已 `CrossCoreWaitFlag(MM1_RES_INTRA_EVENT[taskId])`/`MM2_RES_INTRA_EVENT[taskId]`（flag ID 与本算子 §3.4 的 MM1={9,10}/MM2={7,8} 全同）。故 §3.5 PV 相位、§3.6 QK 相位中**不再有 kernel 手动的 MM1/MM2 wait**——AIC 侧只 set C1_V1/C2_V2/PV_CONSUMED，AIV 侧 set MM1/MM2（catlass 等）。双重 wait 会死锁（pitfalls P31）。
- **PV_CONSUMED 用 step 守卫非 bootstrap**：循环前在 `PIPE_FIX` 上预置的 set 不传播（冷管道），改 §3.5 的 `if (it > SAGE_TASK_NUM3)` 守卫（首次填充跳过、复用才 wait），单 buffer P-scale 与 4-slot pL1 时序天然错开（pitfalls P32）。
- **merge 紧凑化 copy 512**：§4 SageMergeKernel 紧凑化 copy 整个 512-float slot（sub0+sub1 m|l），非 128（pitfalls P33）。
- **bf16 输入 host 转 fp16**：host 读入后 bf16→fp16(RNE) 再喂 kernel（pitfalls P34）。
- **构建**：CMakeLists 设 `CATLASS_ARCH=3510`；`aclrtMalloc`→`uint8_t*`+cast 喂 `GM_ADDR`（pitfalls P29/P30）。

> 验收：重建版 22/22 PASS（cos 0.9996~0.9999），见 fa-sageattention-recipe.md §6。FP8 PV（quantScheme=1）集成见 pitfalls P36（撞 catlass FP8 MX API，须先探针实测组件真实数据流，暂缓）。
