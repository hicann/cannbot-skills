## 公共工具：Cross Core Sync 与 WorkspaceQueue 详细参考

本文档包含 WorkspaceQueue 环形缓冲区、批量同步模式、CrossCore flag 的完整实现细节与代码示例。
概览与判断规则见 `.claude/references/dsl-to-ascendc/references/dsl2Ascendc.md`。

---

## 第三章：公共工具

### 0. 同步模式判断规则

| AscendC 信号位置 | 同步模式 | 实现 |
|:---|:---|:---|
| `CrossCoreSetFlag` 在 n_tile 循环**内部** | 逐 tile 同步（WorkspaceQueue） | 环形缓冲 + 每 tile Acquire/Release |
| `CrossCoreSetFlag` 在 n_tile 循环**外部** | 批量同步（Bulk Sync） | 单次 CrossCoreSetFlag/WaitFlag |

### 1. 通过 Workspace Queue 实现跨核同步

使用基于 workspace GM 的环形缓冲区进行 AIC → AIV 数据传输，配合 `CrossCoreSetFlag/WaitFlag` 同步：

> **⚠️ CANN 9.0.0 API 硬约束（A3 build-FAIL 教训 2026-05-27）—— pipe 不可模板化**：
> `CrossCoreSetFlag` 的 pipe 是**字面 `PIPE_*` 编译期常量**（`PIPE_FIX` / `PIPE_MTE3` /
> `PIPE_MTE2`），硬编码在每个 Release 方法体里。**绝对不要把 pipe 做成模板参数**——
> `template <..., AscendC::pipe_t PipeProd>` + `CrossCoreSetFlag<0x2, PipeProd>(...)`
> 在 CANN 9.0.0 **编不过**：(1) `AscendC::pipe_t` 不存在（`pipe_t` 是**全局**类型）；
> (2) pipe 当显式模板实参 → "no matching function / invalid explicitly-specified
> argument for template parameter 'pipe'"。多 producer pipe（FA：cube→vec 走 `PIPE_FIX`，
> vec→cube 走 `PIPE_MTE3`）用**分开的 Release 方法**（`ProducerReleaseFix` /
> `ProducerReleaseMte3`），各硬编码裸常量。下面是 cv-agent verified（A3 编过+跑过）的形式。

```cpp
template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    // AIV 初始化：将所有槽位标记为空闲（生产者可以写入）。裸常量 PIPE_MTE2。
    __aicore__ inline void InitFreeSlotsMte2() {
        for (uint32_t i = 0; i < DEPTH; ++i) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);
        }
    }

    // 生产者：等待空闲槽位，返回其 GM 视图填充
    __aicore__ inline AscendC::GlobalTensor<T> ProducerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(consumerNotifyProducerId_);  // 等待"槽位空闲"
        return workspace_[head_ % DEPTH * slotSize_];
    }
    // 生产者 release —— 数据经 Fixpipe 写出（FIX pipe）。裸常量 PIPE_FIX，不模板化。
    __aicore__ inline void ProducerReleaseFix() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(producerNotifyConsumerId_);  // "数据就绪"
        head_++;
    }
    // 生产者 release —— 数据经 DataCopy/MTE3 写出（MTE3 pipe）。裸常量 PIPE_MTE3，不模板化。
    __aicore__ inline void ProducerReleaseMte3() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(producerNotifyConsumerId_);  // "数据就绪"
        head_++;
    }

    // 消费者：等待数据就绪，返回其 GM 视图读取
    __aicore__ inline AscendC::GlobalTensor<T> ConsumerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(producerNotifyConsumerId_);  // 等待"数据就绪"
        return workspace_[tail_ % DEPTH * slotSize_];
    }
    // 消费者 release —— 读取耗尽（MTE2 pipe），归还空闲槽位。裸常量 PIPE_MTE2。
    __aicore__ inline void ConsumerReleaseMte2() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);  // "槽位空闲"
        tail_++;
    }
};
```

**关键同步流程：**
```
初始化阶段：
  AIV: InitFreeSlotsMte2() → 设置 DEPTH 个"槽位空闲"标志（PIPE_MTE2）

循环阶段（每个 tile）：
  AIC: ProducerAcquire() → 等待"槽位空闲"
  AIC: Fixpipe(slot, ...) → 写入 workspace GM
  AIC: ProducerReleaseFix() → 设置"数据就绪"（PIPE_FIX；若经 MTE3 写出则用 ProducerReleaseMte3）

  AIV: ConsumerAcquire() → 等待"数据就绪"
  AIV: DataCopy(local, slot) → 从 workspace GM 读取
  AIV: Process() → 计算
  AIV: ConsumerReleaseMte2() → 设置"槽位空闲"（PIPE_MTE2）
```

#### A. 批量同步模式（Bulk Sync，无环形缓冲）

当 AscendC Cube 分支需要**先完成所有 n_tile 写入 workspace 后**，Vector 才能开始处理时（如需要全局统计量的两遍处理），不使用 WorkspaceQueue 环形缓冲，而是使用**单次 CrossCoreSetFlag/WaitFlag 批量同步**：

**AscendC 实现**：
```cpp
// Cube 侧：循环所有 n_tile，全部写完后一次性信号
if ASCEND_IS_AIC {
    for (int by = 0; by < nTiles; by++) {
        auto wsBlock = wsGM_[bx * baseM * N + by * baseN];
        mm_.ComputeBlock(aBlock, bBlock, wsBlock, H_K, N);  // dstStride=N
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);  // 只发一次信号
}

// Vector 侧：等待一次信号，然后两遍处理
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);  // 只等一次

    // Pass 1: 全局扫描
    for (int by = 0; by < nTiles; by++) { /* ... accumulate stats ... */ }
    // Pass 2: 利用统计量处理
    for (int by = 0; by < nTiles; by++) { /* ... quantize ... */ }
}
```

### 2. WorkspaceQueue vs 批量同步对比

| 特性 | WorkspaceQueue（逐 tile 同步） | 批量同步（Bulk Sync） |
|:---|:---|:---|
| **信号次数** | 每个 tile 一次 Acquire/Release | Cube 全部完成后一次 |
| **Workspace 大小** | DEPTH × baseM × baseN × sizeof(T) | M × N × sizeof(T)（全输出） |
| **Vector 启动时机** | Cube 写完一个 tile 即可开始 | 必须等 Cube 全部写完 |
| **适用场景** | 逐 tile 独立处理（如 LeakyReLU、Scale） | 需要全局统计量（如 ReduceMax + 量化） |
| **AscendC 信号位置** | `CrossCoreSetFlag` 在循环内 | `CrossCoreSetFlag` 在循环外 |
| **核分配** | BlockScheduler 分配 mBlocks×nBlocks | 每核一个 m_block，Cube 内循环 n_tiles |

### 3. CrossCore flag 规则

#### 规则：批量同步只使用单个 CrossCore flag

批量同步使用**单个 flag ID**，由 AIC 设置并向所有 AIV 子块广播：

```cpp
// ✅ 正确：单 flag 广播给所有 AIV 子块（KERNEL_TYPE_MIX_AIC_1_2）
#define CUBE_NOTIFY_VECTOR_ID 0x8

if ASCEND_IS_AIC {
    // AIC 完成所有 tile 后发一次信号
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
}

if ASCEND_IS_AIV {
    // 所有 AIV 子块（vid=0, vid=1）都等待同一个 flag
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);
    // vid 由 GetSubBlockIdx() 区分各自的数据偏移
    int rowOffset = AscendC::GetSubBlockIdx() * subTileM;
}
```

**❌ 错误写法**（逐 AIV 子块发送不同 flag）：

```cpp
// 错误：AIC 发送 0x8 给 vid=0，发送 0x9 给 vid=1
for (int i = 0; i < VEC_NUM; i++) {
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + i);
}
// AIV: CrossCoreWaitFlag<0x2>(0x8 + vid_);
```

> **为什么会出错**：per-subblock flag 是逐 tile 同步（ring buffer）模式的写法，适用于 `matmul_leakyrelu` 中的 WorkspaceQueue。批量同步场景（two-pass 量化）只需一次信号，用单 flag 广播即可。

**识别口诀**：
- `CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)` 只调用一次
- 所有 AIV 子块调用 `CrossCoreWaitFlag<0x2>(flagId)`，共享同一 flag
- 批量同步场景用单 flag 广播，**不要**逐 AIV 子块发送不同 flag

#### 规则补充：group barrier 的两个 invariant（AIV→AIC 方向 + 写序，见 OL-190）

§3 above covers the **AIC→AIV broadcast** direction (one Set, all sub-blocks Wait the
same flag). The **AIV→AIC** direction adds two invariants — both confirmed on V220
`KERNEL_TYPE_MIX_AIC_1_2` (see `OPERATIONAL_KNOWLEDGE.md` OL-190):

1. **Symmetric participation (group-set count)** — `CrossCoreSetFlag<0x2>` is a *group*
   barrier. The AIC's single `CrossCoreWaitFlag<0x2>` clears only after **BOTH** AIV
   sub-blocks have Set. Gating the whole handshake on one sub-block
   (`if (subBlockIdx==0)`) or idling the second sub-block (e.g. zero-row work at small
   `BLOCK_M`) → the group-set never completes → **AIC deadlocks** (timeout, zero output).
2. **Set-after-own-write (ordering)** — each sub-block's `Set` must follow that
   sub-block's OWN real GM write on the matching pipe (`PIPE_MTE3` after a DataCopy store,
   `PIPE_FIX` after a Fixpipe). An *empty* Set (no preceding GM traffic) retires
   immediately → the group barrier clears before the data-bearing sub-block's write lands
   → AIC reads STALE workspace → **garbage output (not a hang)**.

**Single-producer resolution**: when the producer work isn't naturally row-splittable,
still make BOTH sub-blocks participate by SPLITTING the producer by row range (topK rows /
N1 tiles): each sub-block writes its own disjoint GM slice THEN Sets — union = full
workspace, counts balance, every Set certifies real retired data. (This is exactly the
cv-agent FA `ComputeVec1` pattern: both sub-blocks write `pSlot[rowStart_*BLOCK_N]` then
`ProducerReleaseMte3`.)

### 4. RUNNABLE deadlock-avoiding handshake for V351 MIX 1:1 (cube↔vec) — verdict: PUBLIC-API-runnable WITH the (C) asymmetric-pipe correction (2026-06-03)

`applies_to: soc=Ascend950PR (V351 / A5, Ascend950PR_9579); cann=9.0.0; op_class=any MIX 1:1 cube↔vec software-pipelined op (FlashAttention, MoE finalize, fused norm+matmul)`
`verified_on: V351 FA whole-port reference (runs 64/64) achieves the handshake via the BaseApi Buffer<CROSS_CORE_SYNC_FORWARD> abstraction (NOT a hand-roll); the hand-rolled recipe below is runnable ONLY with the (C) consumer-pipe = PIPE_V correction — the prior PIPE_FIX consumer-wait was empirically caught as the non-determinism root cause by kw-gb4 case7 (softmax_sum drift 55 across identical re-runs)`
`derived-from: cann-source (Mode 5, run fa_xcsync_20260603) + empirical correction (kw-gb4 graybox 2026-06-03) + whole-port reference cross-confirmation`

§1-§3 above give the CONTRACT (single-flag broadcast, symmetric participation,
set-after-own-write). They are an ILLUSTRATION — a hand-rolled handshake following
ONLY §1-§3 still **deadlocks at runtime on V351** (`torch.npu.synchronize()` hangs,
no aicore exception; kw-gb2 hermetic graybox 2026-06-03, see PB-35
`confirmed_on`). The four missing specifics below are what make the working
whole-port handshake NOT deadlock. **The working sync is built entirely on public
AscendC primitives** (`CrossCoreSetFlag`/`CrossCoreWaitFlag` + `SetFlag`/`WaitFlag`
+ a hand-implementable ring) — there is NO privileged vendor class required, so a
customer CAN reproduce it by hand. This CLOSES the FA whole-port reproducibility
question for the cross-core edge.

**(A) Use SYNC MODE 4, NOT mode 2.** The deadlocking hand-roll used
`CrossCoreSetFlag<0x2, ...>` (mode 2 = 1 AIC : 2 AIV broadcast). The working sync
uses **mode 4** (AIC↔AIV 1:1, where AIV0 and AIV1 are individually triggerable —
per `ascend950pr.md` 同步模式 table: 模式4 = "AICore 内 AIC↔AIV 1:1 比例,
AIV0/AIV1 可单独触发"). Mode 4's per-sub-block triggerability is the hardware basis
for (B).

**(B) Per-sub-block DISJOINT flag IDs (`id` and `id + 16`).** This is the
structural form of §3's symmetric-participation rule. One AIC pairs with 2 AIV
sub-blocks. Sub-block 0 uses flag id `k`; sub-block 1 uses `k + 16` (the offset is
a fixed 16). When the AIC is the consumer it issues **two** waits (`k` and `k+16`)
— one per sub-block; as producer it issues **two** sets. Each AIV sub-block
Sets/Waits ONLY its own id. Contrast the deadlock: hand-roll had both sub-blocks
Set the SAME id under mode 2, so the counter semantics mis-tracked which
sub-block had retired. (flag id range is 0–10 per sub-block per the public spec;
AIV1's `+16` lands in 16–26; reserved barrier ids live above.)

**(C) Direction-pinned LITERAL pipe — the pipe is ASYMMETRIC (Set-pipe ≠ Wait-pipe).**
The producer pins its `Set` to the pipe its write retires on; the consumer pins its
`Wait` to the pipe it will CONSUME on — these are DIFFERENT pipes. Pinning both ends
to the same pipe (the producer's) is the dominant FA cross-core bug (see correction
note below):
- cube→vec result (UB or GM): producer (cube) `Set` on `PIPE_FIX` (the result
  retires from Fixpipe L0C→UB/GM); consumer (vec) **`Wait` on `PIPE_V`** — the vec
  unit consumes the result on the vector pipe, so it waits there (use `PIPE_MTE3` only
  if the consumer's first touch is an MTE3 re-copy, not a vector op). The slot-release
  back to the producer is also `PIPE_V`/`PIPE_MTE3`. **`Wait` on `PIPE_FIX` here is
  WRONG** (that is the producer's pipe) — on arch35-AIV it both fails to establish the
  Fixpipe-write→vec-read happens-before AND trips the `<4, PIPE_FIX>` consumer compile
  reject; it is the root cause of the run-to-run softmax non-determinism (see note).
- vec→cube P-matrix (L1): producer (vec) `Set` on `PIPE_MTE3` (after the
  `DataCopy` store into L1); consumer (cube) `Wait` on `PIPE_MTE1`.
- vec→cube via GM (backward): vec `Set` on `PIPE_MTE3`; cube `Wait` on `PIPE_MTE2`.

  Note the cube→vec rule is the SAME asymmetric shape as the two vec→cube rules
  (which already pin Set≠Wait: MTE3→MTE1, MTE3→MTE2) — the original cube→vec entry was
  the lone symmetric (FIX→FIX) outlier, and that was the bug.

**(D) Set-after-own-write, ring depth ≥ pipeline skew.** Producer order is
`Wait(slot-free) → write GM/L1/UB → Set(ready)`; consumer is `Wait(ready) → read`.
The FA pipeline has 4 overlapped stages (C1 / V1 / C2 / V2 skewed by one tile each),
so each handshake buffer is a depth-2 or depth-3 ring (so a producer never
overwrites a slot the consumer hasn't drained). The 3 pipeline edges use DISJOINT
flag-id sets so they never alias: C1→V1 uses one id set, V1→C2 a second, C2→V2 a
third. Startup uses a single mailbox wait (`CrossCoreWaitFlag<4, PIPE_S>(15)`) on
the cube entry before the first tile.

Concrete anchor (public API only — hand-implementable, no vendor wrapper class):
```cpp
constexpr uint64_t SYNC_MODE = 4;           // AIC<->AIV 1:1; AIV0/AIV1 separate
constexpr uint32_t AIV1_FLAG_OFFSET = 16;   // sub-block-1 flag-id space = id + 16

// Edge cube->vec (cube produces a result buffer; both AIV sub-blocks consume):
if ASCEND_IS_AIC {                          // producer notifies each sub-block
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(resFlag);                     // -> AIV0
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(resFlag + AIV1_FLAG_OFFSET);  // -> AIV1
}
if ASCEND_IS_AIV {                          // each AIV waits ONLY its own id
    uint32_t myId = resFlag + (GetSubBlockIdx() == 1 ? AIV1_FLAG_OFFSET : 0);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(myId);   // consumer pipe = PIPE_V, NOT producer's PIPE_FIX
}

// Edge vec->cube (each AIV writes its own P slice to L1 THEN sets; cube waits both):
if ASCEND_IS_AIV {
    /* DataCopy own P slice into L1 here */
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(pFlag + (GetSubBlockIdx() == 1 ? AIV1_FLAG_OFFSET : 0));
}
if ASCEND_IS_AIC {
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(pFlag);                      // sub-0
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(pFlag + AIV1_FLAG_OFFSET);   // sub-1
}
```

**Why the hand-roll deadlocked (root cause, for diagnosis)**: it was missing (A)
mode-4, (B) the `+16` paired-sub-block disjoint-id scheme, and (C) the exact
direction→pipe pairing — NOT a missing vendor class. Switching `<0x2>`→`<4>`,
giving each sub-block its own id (`id` / `id+16`), pinning the pipe per direction,
and keeping Set after the matching-pipe write resolves the hang.

**CORRECTION (2026-06-03, empirically caught — supersedes the prior cube→vec pipe):**
the earlier revision of (C) pinned the cube→vec consumer `Wait` to `PIPE_FIX` (the
producer's pipe). That is wrong and was the dominant **run-to-run non-determinism**
root cause on arch35-AIV (Ascend950PR_9579, CANN 9.0.0) — distinct from the deadlock
above. Two empirically-confirmed facts:
- **Asymmetric pipe**: producer (cube) `Set` on `PIPE_FIX`; consumer (vec) `Wait` on
  **`PIPE_V`**. `Wait`-ing on `PIPE_FIX` does NOT establish the Fixpipe-write→vec-read
  happens-before AND trips the `<4, PIPE_FIX>` consumer compile reject (the observed
  `[4,6]` event-pool rejection was an id+pipe mismatch — **mode 4 itself is correct**;
  working code compiles `CrossCoreWaitFlag<4, PIPE_FIX>` on the producer side fine).
  The two other pipe choices that "compile" (`PIPE_S`, `PIPE_MTE2`) are the other two
  mis-matches: `PIPE_S` races, `PIPE_MTE2` deterministically stalls.
- **`PIPE_V` is correct ONLY for a UB-RESIDENT result; cube→vec-VIA-GM needs more (kw-gb5, 2026-06-03).**
  If the result is routed cube-Fixpipe→**GM** and the consumer's first touch is a `DataCopy`
  (MTE2 GM-read) — NOT a vector op — then `PIPE_V` gates the AIV's *vector* pipe but leaves the
  MTE2 GM-read un-ordered vs the cube's Fixpipe retire → still races (reads GM too early = init,
  or mid-write = `507015` aivec OOB). kw-gb5 case7 confirmed: `CrossCoreWaitFlag<4, PIPE_V>`
  compiled clean (no `[4,6]`) but `softmax_sum` still flipped 0↔64 across fresh-process runs.
  **Cleanest fix: route the result through UB** (`Fixpipe` L0C→**UB**, not →GM) so the consumer's
  first touch is genuinely a vector op and `PIPE_V` is exactly right. Alternative: keep GM-routing
  but add an explicit GM data-visibility fence — flag-pipe ordering ALONE is insufficient for a
  GM-routed result (and the `PIPE_MTE2` flag was separately found to stall, so it is not the GM
  fence). The whole-port reference sidesteps the question entirely by letting the
  `Buffer<SyncType=CROSS_CORE_SYNC_FORWARD>` abstraction choose routing + fence.
  **Determinism must be tested fresh-PROCESS ×≥3** (warm in-process re-runs mask the race — the
  first launch fixes a scheduling path that repeats within that process).
- **One AIC → N AIV requires N flags**: with 1 AIC driving 2 AIV, the producer MUST
  `Set` BOTH `id` and `id+16` (per (B)); sending only one leaves the second AIV with
  no happens-before → race. **Evidence**: kw-gb4 case7 (minimal 1 AIC + 2 AIV, single
  tile) showed `softmax_sum` drifting up to 55 across 4 identical re-runs — the exact
  single-flag-to-2nd-AIV race signature. Cross-confirmed by the whole-port reference,
  which does NOT hand-roll this: it routes cube↔vec result handshakes through the
  BaseApi `Buffer<SyncType=CROSS_CORE_SYNC_FORWARD>` abstraction (the abstraction picks
  the asymmetric pipe + managed event-id, which is why it is bit-exact + deadlock-free).
- **Event ids: never hand-pick literals.** Use a managed rotation pool (id in 0–10,
  second AIV at `id+16`, reserved/invalid above) so ids don't collide with the
  framework's pool — a hand-picked literal like `6` can alias a managed id and is part
  of why `[4,6]` was rejected.

Robust alternative to hand-rolling all three: use the BaseApi
`Buffer<SyncType=CROSS_CORE_SYNC_FORWARD>` cross-core abstraction (`.SetCrossCore()` /
`.WaitCrossCore()`), which selects the correct pipe + managed id for you.

**Evidence**: V351 FA whole-port reference (arch35-class, runs 64/64) implements
exactly this handshake; the four specifics were extracted by cann-learn Mode 5
(2026-06-03) and cross-checked against the public CrossCoreSetFlag spec in
`hardware/target/ascend950pr.md` (flagId 0–10, mode-4 = 1:1 AIV0/AIV1 individually
triggerable). The deadlocking counter-example is the kw-gb2 hand-roll (PB-35
`confirmed_on` 2026-06-03).

**Other instances predicted**: any V351 MIX 1:1 cube↔vec software-pipelined op
(MoE finalize routing, fused norm+matmul, two-stage producer/consumer across the
AIC↔AIV boundary). Same four specifics apply.

**Cross-ref**: OL-190 (the AIV→AIC symmetric-participation + set-after-own-write
invariants — this §4 is the V351 RUNNABLE refinement with the mode-4 + disjoint-id
mechanism); PB-35 (the deadlock datapoint + scope clarification: library-matmul
path clean, user-Mmad+hand-rolled-flags path deadlocks); §1 above (pipe-not-
templatable literal-PIPE rule, which (C) follows).

### 5. a3/220x multi-core FA — ACHIEVED via per-head-independent library route (20/20 precision, 0.186× vendor); the hand-rolled cross-core RING deadlocks but is UNNECESSARY (2026-07-20)

`applies_to: soc=Ascend910_9382 (220x / A3); cann=9.0.0; op_class=multi-core (blockDim>1) MIX 1:1 cube↔vec fused op (fused FlashAttention head/block parallel)`
`verified_on: device (Ascend910_9382, a3 CANN 9.0.0 container, dedicated NPU verified-clean). Multi-core FA built+run 20/20 (5 shapes × cores{1,2,4,20}); + 4-mode hand-rolled-handshake characterization probe (mode0/1/3/5).`
`derived-from: DS device build+measure 2026-07-20 (library route) + device probes 2026-07-19/20 (hand-rolled mode0/1/3/5), grounded in shipped c220 pto/npu/a2a3 TSyncCVID.hpp / TSync_Custom.hpp / TPush.hpp. Measured evidence (in-repo, disk-verifiable): fa_class/evidence/a3_multicore_fa_20260720/ (RESULT.md + mc_verify.json 20/20 + mc_perf.json 0.186x + GMSLOT_RUN_LOG.md).`

**★ MEASURED RESOLUTION (2026-07-20, device — supersedes any "multi-core is blocked" reading).** Multi-core (blockDim>1) fused-MIX FA on a3/220x **RUNS DEADLOCK-FREE and is CORRECT** — measured **20/20** (5 shapes × cores{1,2,4,20}: cos 0.999998+, deterministic, **bit-identical to the 1-core result at every shape**), perf **0.186× vendor** (`npu_fusion_attention` = a3 vendor baseline, which DID run: ~102µs @ B4N8S512D64 fp16; multi-core 550µs; **recovered 14.3× over single-core** 7866→550µs @ 20 AIC cores). Built first-try, library `MatmulImpl IterateAll<sync=true>` cube, **NO hand-rolled cross-core ring**.
**Why it works (key insight):** FA multi-core is **per-head-INDEPENDENT** — one head-slice per core, each core = a self-contained single-core FA, and `MODE2`/`CV_CORE_SYNC` scopes the per-pair `FLAG_S`/`FLAG_P` handshake to that core's OWN AIC↔AIV group (zero cross-core flag interaction). **So FA multi-core NEVER needs the cross-core LOOPED bidirectional ring that the hand-rolled probes below deadlock on** — that mechanism is both deadlock-prone AND unnecessary for head-parallel FA. The perf gap is **STRUCTURAL** (this kernel is non-flash: full [S,S] score+prob to GM per head + row-serial scalar softmax, vs vendor's fused online-softmax L1/UB-resident) — next perf lever = flash/online-softmax rework, NOT a sync fix.
The device-mode table below (mode0/1/3/5) characterizes the **hand-rolled cross-core** mechanism **for the record only** — it shows why NOT to hand-roll one; it is NOT the FA multi-core path.

**Contrast with §4 (V351/A5).** §4's mode-4 + disjoint-id (`id`/`id+16`) resolution is **arch35-ONLY** — on a3/220x `INTRA_MODE=4` collapses to mode-0 (`GetffstMsg` masks `mode & 0x3`), and the `matmul_client.h` per-sub-block+INTRA_MODE path is the A5 client. **§4 does NOT port to a3.** a3 CVSyncMode (c220): `C_ALL_CORE_SYNC=0` / `V_SUBCORES_SYNC=1` / `CV_CORE_SYNC=2`; user-free flag ids 0..10 (11–15 reserved: `SYNC_AIC_FLAG=11`, `SYNC_AIV_FLAG=12`).

**Device-measured verdict** (blockDim=2 = 2 concurrent MIX groups, 1 AIC + 2 AIV each):

| mode | handshake form | result |
|---|---|---|
| 0 | 1-shot, per-(block,sub) DISTINCT ids + asymmetric pipes (Set `PIPE_FIX` / Wait `PIPE_V`) | **PASS** deadlock-free (max_rel 8.8e-4) |
| 1 | 1-shot, SHARED id (mode-2 broadcast) | **PASS** deadlock-free |
| 3 | LOOPED (≥8 rounds) bidirectional, DISTINCT ids | **DEADLOCK** 507014 |
| 5 | LOOPED bidirectional, SHARED broadcast flags (`CV_COMM_CTRL=12`/13) + per-block GM comm-slot (`block_idx*512`) + `DataCacheCleanAndInvalid` — the shipped c220 `pto::TSYNC_CVID` recipe, per-round sync | **DEADLOCK** (hang, 90s timeout) |

**Conclusion — the deadlock is in a HAND-ROLLED mechanism the performant path does NOT use (NOT a hardware wall, NOT "multi-core is blocked").** Every mode above hand-rolls the cube↔vec cross-core handshake (`CrossCoreSetFlag`/GM-slot). Single-shot works; the LOOPED bidirectional hand-roll deadlocks under every per-round recipe tried (distinct-id mode3 AND shipped shared-flag+GM-slot mode5). A prior "GM-slot is the missing key" hypothesis was **device-refuted** by mode5. **But the vendor + compiler-generated existence-proof FA paths do NOT hand-roll this handshake:** vendor `matmul::Matmul<>` FA is **KFC-implicit with ZERO `CrossCoreSetFlag`** (multi-core coordination lives *inside* the matmul library — see `CAND-FA-A3-PERF-STRUCTURAL-1`, OL-210), and an independent compiler-generated AscendC FA ran MIX on **real a3 (Ascend910_9382)** at ~0.5× CANN. **So this entry's device data is a cautionary result — evidence to NOT hand-roll the multi-core cube↔vec handshake — not evidence that multi-core FA is unachievable on a3.**

**Correct multi-core route — extend the VERIFIED library primitive, do NOT hand-roll or bank.** (Refined 2026-07-20 from a Phase-0 recipe study — correcting an earlier imprecise "just port the KFC-implicit template" framing.)
1. **Library `MatmulImpl IterateAll<sync=true>` (VERIFIED runnable on a3-standalone)** — a genuine library cube (the library owns L1/L0/K-tiling/Fixpipe; NOT hand-`Mmad`, NOT a cross-core ring). The single-core base uses exactly this + ONE per-pair PB-55 handshake (17/17). Extend to multi-core by distributing head-slices across `blockDim`, keeping ONLY the per-pair MODE2 (`CV_CORE_SYNC`) handshake — NO added cross-core flags, NO cross-core ring. (§5's device data: 1-shot per-pair PASSES at blockDim=2; only hand-rolled LOOPED rings deadlock.) This route was **device-VALIDATED 2026-07-20 — see MEASURED RESOLUTION above (20/20 precision, 0.186× vendor, deadlock-free).**
   - **HONEST caveat — do NOT chase the async path:** the fully-implicit *async* KFC form (`matmul::Matmul` + `KfcServer`, 0-handshake) is **DOCUMENTED-BLOCKED on a3-standalone-pybind** (507014; `CAND-KFC-standalone-bootstrap-teardown` / PB-53 double-bootstrap / PB-54 MIX-KFC workspace coupling UNSOLVED 2026-07-17). `IterateAll<sync=true>` avoids the async KFC msg-ring — which is exactly why it builds + runs standalone where async deadlocks.
   - **Negative reference:** the on-disk `flash_attention_score_*.h` (earlier_fa_prototype) are a HAND-ROLLED CrossCore ring, NOT a KFC template — do not mistake them for the vendor mechanism. (Their **~0.25× vendor is a FULL-FA end-to-end** ratio for that SEPARATE earlier kernel, per `CAND-FA-A3-PERF-STRUCTURAL-1` — **NOT directly comparable** to this entry's 0.186× library multi-core: different kernel + shapes, no matched-shape re-measure. This entry makes NO claim the library route is faster/slower than that port; the only verified library numbers are 20/20 + 0.186× end-to-end.)
2. Structural references: P-P116 (a3 MIX template) + PB-53/OL-275 (KFC bootstrap discipline if the library path needs it) + the compiler-generated AscendC FA witness (ran MIX on real a3 ~0.5× CANN). (The shipped `TPush.hpp` `SyncPeriod` batched-credit is a hand-roll variant — subordinate to the library route above, not the primary recommendation.)

**Separate two questions (do not conflate):** *能不能做 (achievable?)* — only known by porting an existing recipe and running it, NOT inferable from a hand-rolled deadlock; *值不值得做 (worth it?)* — depends on whether `npu_fusion_attention` gives a usable a3 vendor baseline. That baseline's a3 availability is **TO VERIFY, not to assert either way**. "Achievable but currently unmeasurable" ≠ "not achievable".

**Host-tiling safety (for any a3 multi-core FA generation).** Prefer a **device-side head-loop** kernel (`MatmulImpl IterateAll<sync=true>` per head, internal K-tiling) over **hand-rolled HOST multi-head tiling** — the latter is the 507015 fault class (see PB-56). Per-core disjoint `[gridDim,S,S]` workspace slabs are the correct memory design for the (currently deadlocking) multi-core path; an unfused 3-kernel path is infeasible (`S[BN,S,S]` ≈ 32 GB for BN=32/S=512).

**507014 hangs are DESTRUCTIVE + CUMULATIVE but TRANSIENT** — the device error-latches (single-core then also hangs on that device) yet recovers on process-kill / long idle (>120s), NOT hardware damage. Do not attribute a shared-device cascade to the kernel-under-test.

**Cross-ref**: §4 (V351/A5 RUNNABLE handshake — this §5 is the a3/220x sibling with the OPPOSITE verdict for the looped form); PB-56 (deadlock datapoint + stomp-probe DETECTION methodology + large-D-refutes-507015); PB-55 / OL-190 (single-core PB-55 both-AIV-subblocks-must-Set rule; single-core fused-MIX FA works + verified 17/17 vs faithful dense golden).

### 6. Cross-core sync does NOT cover an intra-core UB WAR (GDR CAND-GDR-2, device-measured 2026-07-20)

A cross-core barrier — **either** whole-device `SyncAll<false>()` **or** a scoped `CrossCore`
MODE2 handshake — orders work BETWEEN the AIC and AIV groups. It does **nothing** for a
write-after-read hazard WITHIN one AIV core's own pipe. Reusing the same UB `LocalTensor`s across
consecutive AIV elementwise ops without a fence races the prior op's V-pipe read against the next
op's MTE2 `DataCopy` overwrite → intermittent, timing-dependent garbage. In the GDR forward
(a3/Ascend910_9382, 2026-07-20) this race **survived BOTH** whole-device `SyncAll` and scoped
`CrossCore` — the fix was an INTRA-core `PipeBarrier<PIPE_ALL>()` at the elementwise-helper entry
(→ bit-deterministic 20/20 + 8/8). Corollary corrections from the same op: "scoped CrossCore is
the race fix" was WRONG (the fence is), and the separate 507014 DEADLOCK was fixed by the
`KERNEL_TYPE_MIX_AIC_1_1` macro (PB-57), NOT by CrossCore. Keep CrossCore as the KB-correct
cross-core sync pattern — just do not expect it to protect an intra-core UB alias.

**Cross-ref**: CAND-GDR-2 (the WAR + fence), PB-17 (cross-row V→MTE2 alias hazard — sibling), PB-57 (the 1:1-macro fix for the co-occurring 507014 deadlock), P-P117 (the GDR pattern).
