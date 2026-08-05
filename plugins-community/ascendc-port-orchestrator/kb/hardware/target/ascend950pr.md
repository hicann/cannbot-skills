---
platform: Ascend950PR
type: target
verified: true
cann_version: 9.0.T501
soc_version: Ascend950PR_9589
npu_arch: 3510
arch_codename: DAV_3510  # canonical (= arch35 / NpuArch 3510 / cannbot ArchVersion::V3510). "David V100"/"DaVinci V351"/"V351" = DEPRECATED aliases (V351 = truncation of V3510). See "Naming (canonical)" section.
date_verified: 2026-03-28
---

# Ascend950PR (David V100) -- Target Platform

> The only Ascend chip supporting both SIMT and SIMD programming models.
> All data verified on A5 server (198.51.100.35) container can_torch_cann_device_1
>
> **完整版（含 PDF 手册页码）**: `docs/archive/A5_HARDWARE_DETAILS.md`
> **PDF 原始手册**: `/mnt/d/workspace/ai/a5/DavidV100用户手册*.pdf`
> unless otherwise noted.

## Naming / codename (CANONICAL — cannbot npu-arch §2.5 `DAV_3510` is single-source)

| Concept | Canonical | Deprecated aliases / notes |
|---|---|---|
| Product | **Ascend950PR** (prefill) | Ascend950DT = decode variant — same chip |
| SocVersion | **ASCEND950** | runtime string e.g. `Ascend950PR_957b` / `_9589`; the `_NNNN` suffix is the **silicon bin (server-specific)** — resolve via `GetSocName()` / `npu-smi`, do NOT hardcode |
| Architecture | **arch35** = `__NPU_ARCH__` 3510 = NpuArch 3510 | — |
| Chip codename | **DAV_3510** | cannbot/tensor_api code = `ArchVersion::V3510`. **"V351" (≈545× in our KB) = DEPRECATED truncation of V3510**; "David V100"/"DaVinci V351" = marketing/legacy. All ≡ DAV_3510 / arch35 / Ascend950PR. |

**Canonical static hardware constants** — cannbot npu-arch §2.5 (DAV_3510, INI-derived) is single-source; our doc keeps only **probe/measured increments** (UB bank-conflict, atomicAdd cycles, AXI concurrency):
- **UB** = 256KB physical / **248KB usable** (`ub_size` 253952 = 262144 − 8KB framework reserve). **Code MUST use `GetCoreMemSize(UB)`, never hardcode.** **Two-layer reservation**: SIMT (L3 / regbase) kernels reserve a *further* 40KB (`SIMT_UB_SIZE_BYTE` 40960) at the top of UB for SIMT DCache / thread state → **SIMT-effective UB ≈ 208KB** (253952 − 40960 = 212992). Probe increment (cannbot has the framework layer but not the SIMT layer); silent UB OOB if SIMT tiling assumes full capacity — see `PLATFORM_BUGS.md` PB-32.
- **BT (bias table)** = **4KB (4096)**.
- **sparsity = 0** — 950PR does **NOT** support 4:2 structured sparsity (contrast A2/A3 DAV_2201 = sparsity 1). Cube/matmul must not assume 4:2.
- L0C 256KB · L0A/L0B 64KB(+4KB MX) · L1 512KB · NpuArch INI = 3510.

## Compute

| Parameter | Value | Source |
|-----------|-------|--------|
| AICore total | 28 (PG **binned** variant, npu-smi confirmed); **full-die = 32** | npu-smi / whitepaper §3 表3-1 |
| AIV per AICore | 2 | CANN GetAicAivTaskRation() |
| **AIV total** | **56** (PG-28); full-die **64** | 28 x 2 / 32 x 2 |
| AIC total (Cube) | 28 (PG-28); full-die **32** | 28 x 1 / 32 x 1 |
| Warp size | 32 | simt_stub.h |
| Max threads / block (LAUNCH_BOUND) | 512 (typical), 1024, 2048 | DavidV100 manual |
| Max threads / AIV | 2048 | DavidV100 manual |
| Warp schedulers / AIV | 4 | DavidV100 manual |
| Register file / AIV | 128KB (64 reg/thread at 512 threads) | DavidV100 manual |
| Clock frequency | 1.65 GHz | DavidV100 manual |
| Block scheduling | **Software time-slicing**; benchmark block-count sensitivity on target | A5 benchmark |
| Max concurrent blocks | 56 (1 per AIV; beyond 56 = time-sliced) | A5 benchmark |

### SIMT vs SIMD compute width

| Mode | Compute width | Issue | Best for |
|------|--------------|-------|----------|
| SIMT | 4 x 128B = 512B | In-order, single-issue | Irregular access, scatter-add, cooperative |
| SIMD | 256B x 2 = 512B | Out-of-order, multi-issue | Contiguous data, vectorized compute |

### Warp-level API (SIMT mode, AscendC::Simt namespace)

- `WarpShflSync`, `WarpShflXorSync` -- warp shuffle
- `WarpReduceAddSync`, `WarpReduceMaxSync`, `WarpReduceMinSync` -- warp reduction
- `WarpBallotSync`, `WarpAllSync`, `WarpAnySync` -- warp vote
- `ThreadBarrier()` -- block-wide sync (__syncthreads equivalent)
- `ThreadFence()` -- memory fence
- `AtomicCas` -- compare-and-swap

## Memory

| Parameter | Value | Source |
|-----------|-------|--------|
| HBM type | BaiLu Memory (Huawei proprietary) | A5_HARDWARE_INFO.md |
| HBM capacity | 128GB (8 x 16GB) | npu-smi |
| HBM bandwidth (peak) | 1.6 TB/s | DavidV100 manual |
| HBM bandwidth (measured) | ~1.1 TB/s | A5 benchmark |
| L2 cache total | 128MB (2 die x 64MB, 16 bank/die x 4MB) | DavidV100 manual |
| L2 cacheline | 512B (4 x 128B sector) | DavidV100 manual |
| L2 read bandwidth | 5.28 TB/s | DavidV100 manual |
| UB (Unified Buffer) / AIV | **256KB physical / 248KB usable** (`GetCoreMemSize(UB)` = `ub_size` 253952; never hardcode — see Naming/canonical-constants above) | DavidV100 manual (phys) + cannbot npu-arch §2.5 (usable, single-source) |
| SIMT DCache | 32-128KB (carved from UB) | DavidV100 manual |
| SIMT shared memory | min 128KB (carved from UB, 128B aligned) | DavidV100 manual |
| Per-AICore AXI interface | 2 x 128B read + 1 x 128B write（硬件层）；AscendC 侧**并发 DataCopy 实测只有 ~10% 加速**（K2/K3=1.106×，见 probe_findings/2026-04-21_Q_mte2_parallel.md），即 per-AIV DataCopy 视角下是共享通道；双通道可能跨 AIC+AIV pair 分配 | DavidV100 manual + 2026-04-21 probe |
| MTE Read outstanding | 255 (≤256B burst) / 128 (512B burst) | DavidV100 manual |
| MTE Write outstanding | 128 (<256B burst) / 64 (512B burst) | DavidV100 manual |
| **L1 Buffer / AIC** | **512KB** (shared across 2 AIVs per AIC; 256KB effective per AIV if borrowed) | DavidV100 manual §25.1.1 |
| L0A Buffer / AIC | 64KB + 4KB L0A MX (MXFP8/MXFP4 mode) | DavidV100 manual |
| L0B Buffer / AIC | 64KB + 4KB L0B MX | DavidV100 manual |
| L0C Buffer / AIC | **256KB** (上升 from 128KB in prev gen) | DavidV100 manual |
| FixpParamBuffer | 1024 entries (量化因子 / BIAS) | DavidV100 manual |
| Shared Scalar Buffer (SSBUF) | 3KB (cross-SU communication for MIX 算子) | DavidV100 manual |
| Scalar I-cache | AIC 32KB, AIV 16KB | DavidV100 manual |
| Scalar D-cache | AIC 32KB, AIV 32KB | DavidV100 manual |
| `__ldg` cache hint | **NO measurable effect** (verified 2026-03-26) | tests/ldg_test/ |

**Key note**: L2 is primarily a read cache for DMA traffic. SIMT atomicAdd
reaches HBM and can serialize under high fan-in; verify it with msprof.

## Pipeline architecture: 6 pipelines + independent Scalar Unit

> Corrected 2026-04-21: previous KB said "4 管线并行 (MTE2+VEC+MTE3+scalar)". The
> actual architecture has **6 execution pipelines** plus Scalar Unit as an independent
> hardware unit (not on any pipeline). Scalar unit dispatches instructions to all
> 6 pipelines.

| Pipeline | Path | Primary Purpose |
|----------|------|-----------------|
| **CUBE** | L0A × L0B → L0C | Matrix multiply-accumulate (AIC-only) |
| **VEC** | UB → VEC ALU → UB | Vector compute, 256B × 2 (SIMD) or 4 × 128B (SIMT) |
| **MTE1** | L1 → L0A / L0B | Cube input feed: Load3Dv2, Load2D, Load2Dv2 (AIC-only path); also supports MX→L0A/B MX Buffer |
| **MTE2** | HBM → L1 / UB | Load from external memory: NDDMA, ND2NZ, DN2NZ, MovAlign |
| **MTE3** | UB → HBM / L1 | Store from UB; supports UB→L1 direct DMA |
| **FIXP** | L0C → UB / L1 / HBM | Cube output post-processing: quantization (REQ/DEQ), PRELU/RELU, layout reformat, 5D→4D, NCHWC0→NC1HW |
| *(Scalar Unit)* | (separate hardware unit) | Instruction fetch/decode/dispatch; handles scalar compute in AIV **outside VF**; dispatches to CUBE/VEC/MTE1/MTE2/MTE3/FIXP |

Scalar unit features (from DavidV100 table 25-1, row 06.02-06.05):
- Independent I-cache, 8KB
- Out-of-order execution, **5 instructions per cycle**
- Dual-issue EX instructions
- Hardware arbitration between MTE / L1 / VEC / SU / SC (row 04.08)

**Implication for optimization**:
- Scalar + VEC are **truly parallel hardware units**, not same-pipeline contenders.
- Apparent "scalar_ratio near-serial with VEC" (e.g. op#11 observed 0.44) is **data-dependency / sync-event driven**, not hardware bottleneck:
  - `GetValue(per_row_scalar) → Muls(work, scalar)` introduces real RAW dep
  - MTE2_S / S_V / V_S / S_V hardware sync events enforce ordering
- **Mitigation pattern**: prefer `Muls(dst, src, scalar_from_ub_buffer, n)` reading scalar from a UB-resident buffer (VEC-pipe load) over `GetValue → Muls(scalar literal)` (Scalar-unit detour). Or pre-broadcast via `Duplicate(scalar_buf, val, N)` once per launch.

### Aux Scalar (351x 新增, 公开文档)

"在 Regbase 架构中，**Aux Scalar 计算单元**单独处理 SIMD_VF 函数内的 Scalar 计算，Scalar 计算单元处理 SIMD_VF 函数外的 Scalar 计算。" — SIMD_VF 函数内外的标量运算由不同硬件单元处理。

## SIMD Register File — 新增存储层次 (351x, 公开文档)

> Source: hiascend.com CANN 9.0.0-beta.2 "NPU架构版本351x"

**新增层次**: UB → **Register** → VEC 计算，中间结果**无需回写 UB**。

| 寄存器类型 | 宽度 | 用途 |
|-----------|------|------|
| RegTensor | VL = 256B | 矢量计算数据 |
| MaskReg | VL/8 = 32B | 高维切分掩码 |
| UnalignRegForLoad | — | 优化 UB→RegTensor 不对齐访问（配合 `LoadUnAlignPre` + `LoadUnAlign`）|
| UnalignRegForStore | — | 优化 RegTensor→UB 不对齐写（配合 `StoreUnAlign` + `StoreUnAlignPost`）|
| AddrReg | — | 地址偏移寄存器，用于循环 stride 自增 |

**VEC 数据流（351x）**: GM → UB（via MTE2）→ **Register**（显式搬入）→ 计算 → Register → UB（可省略）→ GM（via MTE3）

**vs 220x**: 220x 的 VEC 源操作数直接来自 UB；351x 必须先把 UB 数据搬到 Register。中间结果可以留在 Register 里继续计算，**减少 UB 写压力**。

**MaskReg**: 高维切分 mask 在 351x 中进 MaskReg（而非 220x 的特殊掩码寄存器），写法有变化（见 AscendC API）。

## UB bank 结构与 bank-conflict 规则 (2026-04-21, 351x 公开文档)

> Source: hiascend.com CANN 9.0.0-beta.2 "NPU架构版本351x" 公开页面

**Bank conflict 规则 (351x)**: 每个 bank group 有 **2 组读口和 2 组写口**，最多同时允许：
- **2 读 0 写**（2 concurrent reads, no write）
- **1 读 1 写**（1 read + 1 write）

比较：220x 每个 bank group 只有 1 组读口 + 1 组写口（最多 1 拍完成 1 读或 1 写）。

**冲突类型**:
- **读写冲突**: 读操作和写操作同时访问同一个 bank
- **写写冲突**: 多个写操作同时访问同一个 bank group
- **读读冲突**: 两个读操作同时访问同一个 bank，或两个以上读操作同时访问同一个 bank group

> 注：bank 数量和每 bank 宽度在图3（图片），文字中未明确写出。

**2026-04-21 Q_ub_bank_count probe 实测**（见 `probe_findings/2026-04-21_Q_ub_bank_count.md`，verdict INCONCLUSIVE_PARTIAL）：
- 稳定测量：在 AIV 上 stride S ∈ {1..32} 的 `Mul(a, b, c, mask=64, repeatTimes=N)` 跑完整 sweep，每点 stdev < 0.5µs。
- **Actionable 发现**：dense-sequential（S=1）比任意 strided 访问快 ~22%。这是真正稳健的性能不变量。
- 无任何 stride 上 K× 的悬崖：排除了"单端口 K=8 banks × 32B 交织"假设（预测 S=8 应 8× 慢，实测 1.27×）。
- Mild mod-4 plateau：S∈{4,8,16} 比 S∈{3,5,7} 慢 ~6%，与"2R+2W per bank group"多端口网络吸收低阶冲突一致。
- **未解**：bank 数量 K 和每 bank 宽度 W 无法从 timing 反推（动态范围仅 +27%）；需要 msprof `bank_conflict` counter（在 CANN 9.0.0 中未暴露）或 sub-32B 访问粒度（AscendC `BinaryRepeatParams` 不支持）。
- **KB 可用结论**：dense UB > stride 22%，bank 子结构 ≤ 6% 差异不值得 micro-tune。

## AIC L1 共享空间 — AIV 可借用 (2026-04-21, 351x 公开文档确认)

**Hardware fact**: each AICORE has **512 KB L1 buffer** primarily sized for Cube FeatureMap + weights. Since AIC:AIV ratio is **1:2**, a single AICORE has 1 AIC + 2 AIVs sharing that 512 KB L1. If the AIC is idle (no Cube task running), the L1 is free capacity that AIVs can use via DMA.

**Direct DMA paths**:
- **351x public doc text** asserts bidirectional hard channel: "AIV 新增 Unified Buffer 和 L1 Buffer 之间的硬通道" + "增加 L0C Buffer → Unified Buffer、Unified Buffer ↔ L1 Buffer 的数据通路"
- **UB → L1: via MTE3 pipeline** (consistent with pipeline table §"6 pipelines" — MTE3 handles all UB-origin writes, incl. UB→HBM and UB→L1)
- **L1 → UB: via MTE1 pipeline** (consistent with pipeline table — MTE1 handles L1-origin loads; on 351x this path is newly extended beyond the traditional L1→L0A/L0B cube-feed use)
- HBM → L1: via MTE2 pipeline (AIV can drive)
- Note: the 351x page text does not spell out the pipe-to-direction binding in one sentence — the mapping above is taken from the pipeline-table semantics. Implication for optimization is concrete: UB→L1 spill competes with MTE3 writeback to HBM (same pipe); L1→UB refill competes with MTE1 Load2D/Load3D (same pipe). Consult AscendC API ref `DataCopy` for the UB↔L1 operand surface when wiring code.

**Potential use cases** (when AIV-only kernel is UB-budget-constrained):
1. **Spill read-only caches**: weight_scale / quant_scale / bias buffers that are loaded once per launch and read many times → put in L1, leaves UB for pipeline overlap
2. **Spill intermediate tensors**: large transient tensors (like tail/head swap buffers) → L1 scratch
3. **Enable `IN_QUE_DEPTH=2` despite tight UB**: if ≥20 KB of current UB buffers can move to L1, freed UB enables pipeline overlap (MTE2/VEC), often worth 2x perf gain

**Gating constraints**:
- **Hardware path confirmed** (351x public doc) — there IS a UB↔L1 channel.
- **Language-level: syntactically valid** per hiascend.com AscendC API ref (TPosition 07_0174, TBuf 07_0161, TBufPool): `TBuf<TPosition::A1>` compiles without warnings.
- **Runtime: NOT SUPPORTED on CANN 9.0.0 / bisheng 2026-03-21** (2026-04-21 probe finding — see `../probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md`):
  - Generic `DataCopy(UB_tensor, L1_tensor)` / `DataCopy(L1_tensor, UB_tensor)` in a pure AIV kernel (no Cube op) produces **aivec illegal-instruction error 259** (subErrType 0x4) at runtime
  - Adding explicit `SetFlag<HardEvent::MTE3_MTE1>` sync does NOT fix — the issue is the emitted opcode, not sync ordering
  - **Implication**: the 351x hardware UB↔L1 channel IS NOT exposed to pure-AIV kernels via the generic `DataCopy` overload on this toolchain. The correct AIV-scope intrinsic (if any) is either (a) a different API not yet documented, (b) gated on a mixed AIC+AIV task configuration, or (c) not yet landed in CANN 9.0.0.
  - **2026-04-21 CANN source confirmation** (via `~/workspace/cann/` scan):
    - Low-level intrinsic `DataCopyUB2L1Impl((__cbuf__ T*)dst, (__ubuf__ T*)src, DataCopyParams)` exists at `ops-nn/matmul/common/cmct/tile/copy_ub_to_l1.h` + catlass parallels (`ops-transformer/.../catlass/tile/copy_ub_to_l1.h`) — this IS the functional UB→L1 primitive.
    - Every `TPosition::A1` usage across ops-transformer / ops-nn / opbase / catlass is in **matmul/Cube kernels**. Zero pure-AIV usages.
    - Conclusion refined: option (b) is correct. `TPosition::A1/B1` is Cube-context only. The generic `DataCopy<LocalTensor<A1>, LocalTensor<UB>>` template resolves but doesn't lower to `DataCopyUB2L1Impl` without a Cube context in the same kernel. Pure AIV must either use a mixed AIC+AIV task layout or not borrow L1.
  - **Consequence for optimizer**: do NOT attempt L1-spill Kind-2 rewrites on UB-budget-overflow ops (op#11 / op#10 / op#9) on this toolchain. Use alternative paths: smaller tiles, fp16 intermediate storage with fp32 compute, split kernel passes, mixed AIC+AIV task layout. Re-probe after CANN version upgrade.
- L1 is shared between 2 AIVs per AIC → even if a future intrinsic enables AIV L1 access, effective per-AIV budget is ~256 KB.

**KB candidate OL-Lx**: "Fused-op UB budget overflow → consider L1 spill" as a Kind-2 architectural option when depth=2 queues can't fit.

## FIXPipe → UB 直接路径 (2026-04-21)

**Hardware fact**: FIXP (Cube output post-processor) writes can target **UB / L1 / HBM** directly. Read source is always L0C (Cube result matrix).

**Operations supported** (DavidV100 table 25-1 rows 04.01-04.07):
- Quantization: (V)REQ8/REQ4, (V)DEQ16/DEQS16, QF32_2_B8/S4, FP32 bypass, FP32→FP16/BF16/HiF8/E4M3
- Pre-stage activation: PRELU / LRELU / RELU / Clipped RELU
- Post-stage layout reformats: Channel merge / padding / Split, M-direction dummy remove
- Tensor conversion: 5D→4D, NCHWC0→NC1HW (all L0C→L1/UB/OUT)
- Parameter buffer: Quant-pre 4KB, Relu-pre 2KB, Bias 4KB

**Triggering**: FIXP only fires when AIC produces L0C data via CUBE. **Pure AIV kernel cannot use this path** (no L0C source). For a kernel to exploit FIXP→UB, the kernel must be configured as **AIC+AIV mixed task** (e.g. FlashAttention-style, explicitly cited in manual as native use case).

**Relevant optimization pattern**: post-Cube quant + activation that would otherwise do `Cube→L0C→MTE→UB→Cast→Muls→Abs` can be replaced by FIXP inlining, saving 1-2 UB round-trips.

## 跨核同步 (2026-04-21, 351x 公开文档更新)

Hardware features for multi-AIV / cross-AIC coordination (DavidV100 table 25-1 rows 20.02-20.04):
- **Inter-AISubsys Cross Core Synchronization 2.0** (upgraded from 1.0)
- **SSBuffer** — 3 KB SSBUF，AIC↔AIV 核间通信直接通过 SSBuf（vs 220x 用 GM）
- **Inter-Die Cross Core Synchronization** (NEW vs prev gen)

**CrossCoreSetFlag / CrossCoreWaitFlag 规格（351x 公开文档）**:

| 参数 | 规格 |
|------|------|
| flagId 范围 | 0–10 |
| 同一 flagId 最大计数 | 15 次 |
| CrossCoreWaitFlag 默认 pipe | PIPE_S（不需显式设置）|

> **Flag-space hardware grounding (whitepaper 2026-06-07, §4.4 STARS2.0)**: the `flagId 0–10` API range above is a **CANN-API / per-Group allocation convention, NOT a hardware limit**. The Ascend950 whitepaper states STARS2.0 provides **up to 128K single-bit OR up to 4096 32-bit multi-bit hardware sync flags** for inter-task-stream synchronization (see §STARS2.0 hardware scheduler below). The silicon flag pool is orders of magnitude larger than the per-Group API window; flag-id collision/isolation is therefore a software-allocation question, not hardware scarcity. Confirms the OL-210 "global cross-group flags" observation. Cross-ref OL-211 (STARS2.0).

**同步模式**:
- **模式 0**: 全部 AIC 之间，或全部 AIV 之间同步
- **模式 1**: 同一 AICore 内两个 AIV 之间同步
- **模式 2**: AICore 内 AIC ↔ AIV（1:2 比例）
- **模式 4**: AICore 内 AIC ↔ AIV（1:1 比例，AIV0/AIV1 可单独触发）

**注意**:
- Matmul 高阶 API 内部已使用 CrossCoreSetFlag，不能与其混用（flagId 冲突风险）
- SetFlag 和 WaitFlag 必须参数完全一致（模板参数 + flagId 均相同才算同一 EventID）

Far-end atomic support (row 11.04):
- CAS / EXCH: U32 / S32 / U64
- Write add / max / min: **FP32 / FP16 / BF16 / S32 / U32 / S64 / U64**
- Load MAX / MIN: S32 / U32 / S64 / U64
- **BF16 + FP16 atomic add are hardware-native** — relevant for bf16 / fp16 scatter-add ops

## Sort / Reduce VEC primitive specs (2026-04-21, hiascend.com API ref)

> Source: hiascend.com CANN 9.0.0-beta.2 AscendC API ref, SIMD API → 基础API → Memory矢量计算

### WholeReduceMax / WholeReduceMin / WholeReduceSum (API ref 07_0079/0080/0081)

Per-repeat horizontal reduction on 1 iteration-chunk. Returns value, (value,index), or index.

| Aspect | Spec |
|--------|------|
| Supported dtypes (A5) | `half`, `float` (Atlas 350 also: `uint16_t`, `int16_t`, `uint32_t`, `int32_t`) |
| Max elements / iteration | 128 (16-bit) / 64 (32-bit) / 32 (64-bit) |
| repeatTime range | [0, 255] |
| Output modes (`ReduceOrder`) | `ORDER_VALUE_INDEX` (default), `ORDER_INDEX_VALUE`, `ORDER_ONLY_VALUE`, `ORDER_ONLY_INDEX` — A5 supports all four |
| Index representation | Stored as 2-byte (half) or 4-byte (float) in the SAME dtype as value; read via `reinterpret_cast<uint16_t*>` / `<uint32_t*>`. **`ORDER_ONLY_INDEX` always uses uint32_t** |
| dst align | 4B (half) / 8B (float) |
| src align | 32B |
| TPosition | VECIN / VECCALC / VECOUT |

**Pattern use**: for softmax / topk / argmax per-row reductions, tile rows so each row fits in one iteration (≤128 half / ≤64 float); use `ORDER_VALUE_INDEX` or `ORDER_INDEX_VALUE` depending on downstream read pattern.

### MrgSort (API ref 07_0232, session-scan)

Merge up to 4 pre-sorted queues into one sorted output (same direction).

| Aspect | Spec |
|--------|------|
| Input queues | up to 4, each already sorted (descending for MrgSort) |
| Element format | 8-byte (score, index) pair: score is half/float, index is uint32 |
| Supported score dtypes | `half`, `float` only (no bf16, no int) |
| Max elements / queue | 4095 |
| repeatTimes | 1–255 |
| TPosition | VECIN / VECCALC / VECOUT (**not A1/A2**) |

**Pattern use**: tiled top-k / bitonic-family sort — generate sorted runs via `Sort32` or `RpSort16` per tile, merge via MrgSort4 hierarchically. Companion ops: `Concat`, `Extract`, `ProposalConcat`, `ProposalExtract`, `GetMrgSortResult`.

### BlockReduceMax / Min / Sum (API ref 07_0082…)

Reduces each 32-byte block (8 half or 8 float) to a single value. Useful for fan-out reductions where `WholeReduce`'s 128-element cap is the bottleneck.

### High-level Sort / TopK (API ref 07 排序操作)

`TopK`, `Sort`, `Concat`, `Extract` APIs exist as high-level wrappers over the primitives above. Consult tiling helpers `GetSortTmpSize`, `GetSortMaxMinTmpSize`, `GetSortOffset`, `GetSortLen` for workspace sizing.

### Sort / Reduce cycle data (2026-04-21 Q_instruction_cycles probe, bisheng 2026-03-21 / CANN 9.0.0)

Source: `probe_findings/2026-04-21_Q_instruction_cycles.md`. Loop-amplified 4096× to saturate launch overhead; warmup 15, measured 30, median reported. Cycle column assumes ~1.8 GHz AIV clock — the raw µs column in the probe's `timings.csv` is authoritative.

| primitive | elements/call | median µs | cyc/call | elts/cyc |
|-----------|---------------|-----------|----------|----------|
| `WholeReduceMax<float>` | 64 | 232.30 | ~102 | 0.63 |
| `BlockReduceMax<float>` | 64 (8×8-blk) | 225.18 | ~99 | 0.65 |
| `MrgSort` q=64 (4 queues × 64) | 256 output pairs | 413.94 | ~182 | 1.41 |
| `MrgSort` q=256 | 1024 output pairs | 1366.99 | ~601 | 1.70 |
| `MrgSort` q=1024 | 4096 output pairs | 5179.60 | ~2276 | 1.80 |

**Shape of cost**:
- `WholeReduceMax` ≈ `BlockReduceMax` at 64 fp32 — both dominated by a ~100-cycle setup+writeback floor. For ≤ 64 fp32 inputs, no benefit from chunking.
- `MrgSort` fits `~0.55 cyc × output_pairs + ~42 cyc` — **linear in output length**, not tree-log. Implication: prefer ONE long-queue `MrgSort` over multiple short-queue calls (per-call setup is the amortized loss).

**Op-gen heuristics derived**:
1. Reduce over ≤ 64 fp32 → no tiling benefit; run one call.
2. MrgSort plan → maximize per-call queue length, minimize call count.
3. BlockReduceMax wins only when per-output-block scaling matters (e.g. [N × 8]→[N] reductions where 8 is the block width).

Caveats: single-block, not cross-version portable, assumes uncontended NPU. Re-probe after any bisheng/CANN upgrade.

---

## Cube numerical behavior (2026-04-21)

**Two modes** (DavidV100 table 25-1 row 06.10):
- **IEEE 754 mode**: standard NAN/INF rules (propagate)
- **Saturation mode**: NAN/INF outputs clamped to ±FP_MAX or 0; hardware reports Overflow / Underflow / Input NAN / Input INF

**Supported MMAD dtype × shape [m, k, n]**:
- FP32 × FP32 + FP32 = FP32: [16, 4, 16]
- TF32 (HF32 × HF32 + FP32) = FP32: [16, 8, 16]
- BF16 × BF16 + FP32 = FP32: [16, 16, 16]
- FP16 × FP16 + FP32 = FP32: [16, 16, 16]
- HiFloat8 × HiF8 + FP32 = FP32: [16, 32, 16]
- FP8_E5M2 × E5M2 + FP32 = FP32: [16, 32, 16]
- MXFP4 (E1M2 or E2M1) + FP32 = FP32: [16, 64, 16]
- INT8 × INT8 + INT32 = INT32: [16, 32, 16]

**INT8 quantization** (row 03.04):
- Feature-map fused bias quantization: supported
- Weight fused bias quantization: supported
- Hardware +/- offset: **NOT supported**

**Bias** (row 03.02): FP32/S32/BF16/FP16 from 4KB parameter buffer

**⚠️ Regression from prev gen** (row 03.03): **NOT support 4:2 structural sparsity** (1971/1981 supported this for INT8/FP16/BF16/HF32/FP32 with "RAT changed not support" note). Any past KB advice assuming sparsity acceleration no longer applies on A5.

## Atomic operations

| Operation | Latency / Notes | Source |
|-----------|----------------|--------|
| **atomicAdd FP32 (HBM)** | **15.9 cycles/op** (high fan-in backward) | msprof verified |
| **atomicAdd FP32 (HBM)** | **3.3 cycles/op** (low fan-in forward) | msprof verified |
| atomicAdd FP16, BF16 | Supported (hardware native) | DavidV100 manual |
| atomicCAS U32/S32/U64 | Supported via Simt::AtomicCas | DavidV100 manual |
| atomicAdd on UB | **Also slow**; profile before choosing it | Expert E7-4 |
| HA Reduce AtomicStore | FP32/FP16/BF16/INT types | DavidV100 manual |

**更正 (Batch 14, 手册确认)**: L2 cache **支持** Reduce Atomic coalescing (HA.FS009.02, HA.FS010.03)。同 cacheline 的 atomicAdd 可在 L2 并行执行。但实测 (msprof) 仍显示 atomicAdd 是瓶颈——可能是高 fan-in 下 cacheline 冲突率太高，L2 coalescing 无法完全消除竞争。sorted-edge (P-P21) 仍是最有效的优化。

## Load/Store

| Access width | Relative speed | Source |
|-------------|---------------|--------|
| 128-bit | **2.1x** faster than 32-bit | tests/load_width_test/ (verified) |
| 64-bit | 1.4x faster than 32-bit | tests/load_width_test/ (verified) |
| 32-bit | baseline | tests/load_width_test/ |

- Stride-1 access: coalesced automatically
- Alignment: 128B for optimal AXI utilization
- vec4 loads (128-bit): enabled when `dim % 4 == 0` (Pattern P-P3)

## SIMT Architecture Details (from DavidV100 手册 分卷2)

| Parameter | Value | Source |
|-----------|-------|--------|
| Warp schedulers | 4 | 手册 Table 25-1 Row 06.08 |
| Instruction issue | In-order, single-issue, 128B/instr | 手册 p4 |
| dcache (from UB) | 32KB~128KB configurable, 128B cacheline | 手册 p4 |
| SIMT usable UB | 256KB - dcache_size | 手册 p4 |
| Shared memory | min 128KB (from UB), 128B aligned | 手册 p405, SQE format |
| Register file | 128KB total, 4B/reg, shared across threads | 手册 p4 |
| LSU | 1 set, 256 miss handler entries | 手册 Table 25-1 |
| Memory path | Thread → dcache → L2 (64MB/die) → HBM | 手册 Fig 25-3 |

**关键: SIMT 线程读 GM 经过 dcache + L2 cache 两级缓存**。dcache 从 UB SRAM 切出，128B cacheline。L2 是 64MB/die 共享 cache (512B tag, 128B sector, 8-way)。同一 expert 行被多个 token 读时，L2 会缓存。

### SIMT/SIMD 混合模式

**硬件支持 SIMT 和 SIMD 在 VF 内/间切换** (手册 p4):
> VEC 可以支持 SIMD/SIMT 编程模型, 可以在 VF 中切换, 也可以在 VF 间切换. 切换间, 数据在 UB 交换.

这意味着可以: SIMD DataCopy (MTE2) 把数据搬到 UB → 切 SIMT 用线程做不规则计算 → 切回 SIMD (MTE3) 写回。兼得 MTE2 块传输带宽 + SIMT 线程灵活性。

**AscendC 已暴露混合 API** (Batch 14-7 确认): 华为内置算子 `diag_part_simt_simd.h` 使用了完整的混合模式。方法: SIMD `TQue::AllocTensor()` 分配 UB → `GetPhyAddr()` 获取 `__ubuf__` 地址 → `Simt::VF_CALL` 传入 UB 地址 → SIMT 线程读 GM 写 UB → 回到 SIMD 用 `EnQue/DeQue/DataCopyPad` 写 GM。

### L2 Cache 控制

DavidV100 支持软件控制 L2 cache 分配 (手册 p32):
- **Alloc hint**: normal / not-alloc / inter-domain-share / exclusive
- **Victim hint**: first_victim / last_victim / persistent (控制驻留优先级)
- **ReadOnce coalescing**: 同 cacheline 的读请求并行执行 (HA.FS009.01)
- **Reduce Atomic coalescing**: 同 cacheline 的 atomicAdd 并行执行 (HA.FS009.02)
- **DavidV100 新增**: RO prefetch, RO multicast, WU MERGE (上一代无)

**AscendC 已暴露 L2 cache hint API**（HKV 专家代码确认）:
```cpp
// 读: L2 不分配 + L1/dcache 缓存（读完清理 L2 tag，防污染）
T val = __ldg<LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(ptr);

// 写: L2 正常写回 + 不走 L1（一次性写）
__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr, val);
```
来源: `HierarchicalKV-ascend/include/utils.h:139`, `score_functor.h:77`

**OL-18 "\_\_ldg 无效" 更正**: 之前测的是不带模板参数的默认 `__ldg`，默认行为可能是 `L2_CACHE_HINT_NORMAL`（正常缓存），在大范围顺序扫描下与不用 `__ldg` 无差异。带 hint 的版本未测试。

**SG 算子的推荐 hint 策略**:
- expert 行读取: `L2_CACHE_HINT_NORMAL` + `CACHEABLE`（同一 expert 被 ~512 token 重复读，应保留在 L2）
- output 写: `NON_CACHEABLE`（一次性写，不需要缓存）
- index/weight 读: `NOTALLOC_CLEAN`（顺序扫描用完即丢，防污染 L2）

**实验结果 (Batch 14-5)**: SIMT persistent SG forward 测试无正面效果——dim=64 慢 24%，其余无变化。dcache 对 persistent kernel 的顺序 token 遍历已足够有效。L2 hint 在跨 core 共享热点数据场景（如 HKV bucket 查找）才有价值。

## SIMT vs SIMD compatibility

| Feature | SIMT | SIMD |
|---------|------|------|
| Programming model | SIMT threads, 4 warp schedulers | TPipe/TQue/DataCopy pipeline |
| Memory path | dcache (UB) → L2 → HBM | MTE2 DMA → UB, VEC compute, MTE3 → HBM |
| Pipeline parallelism | **单管线** (VEC only, in-order) | **6 管线并行** (CUBE / VEC / MTE1 / MTE2 / MTE3 / FIXP) + 独立 Scalar Unit; 详见 §Pipeline architecture |
| Scatter/indirect access | Native (threadIdx) | Requires manual loop or SetAtomicAdd |
| **Recommendation (P-P9)** | **仅 scatter-write (atomicAdd)** | **所有其他场景** (含 indirect-read) |
| Hybrid mode | 可在 VF 内切换到 SIMD 使用 MTE2 | 可在 VF 内切换到 SIMT 做不规则计算 |

## 通用算子优化决策指南

基于以上硬件架构，对任意 AscendC 算子适用的决策规则：

### 编程模型选择

| 算子特征 | 选择 | 原因 |
|---------|------|------|
| 纯连续读写（elementwise, 矩阵乘） | SIMD | MTE2+VEC+MTE3 三管线并行 |
| 间接读 + 连续写（gather, embedding lookup） | SIMD | expert/embedding 行是连续内存，DataCopy 块传输高效 |
| 连续读 + 间接写（scatter-add, pooling backward） | SIMT | atomicAdd 需要线程级控制 |
| 间接读 + 间接写（稀疏矩阵运算） | SIMT | 两端都不规则 |
| 上述 + 需要极致性能 | **混合模式** | SIMD 搬数据到 UB，SIMT 做不规则计算 |

### 内存层次优化

| 层次 | 大小 | 延迟 | 优化方向 |
|------|------|------|---------|
| Register file | 128KB/AIV | ~1 cycle | 减少线程数提高每线程寄存器数（排序+累加器） |
| UB (SIMD) / dcache (SIMT) | 256KB/AIV | ~几 cycles | SIMD: TQue 管理；SIMT: dcache 配大获得更多缓存 |
| L2 cache | 64MB/die (128MB total) | ~几十 cycles | 数据复用：同一数据被多个 core 读时 L2 自动缓存 |
| HBM | 128GB, 1.5 TB/s | ~百 cycles | 减少总读量（循环重排、数据复用、预排序） |

### Roofline 快速判断

```
OI = FLOPs / Bytes
Ridge point (fp32) = 28 TFLOPS / 1.5 TB/s ≈ 19 FLOP/byte

OI < 1    → 深度内存受限：优化数据搬运（减少 GM 读、管线重叠）
OI 1~19   → 内存/计算混合：两边都需要优化
OI > 19   → 计算受限：优化 VEC 利用率（向量化、unroll）
```

### Reg-based vs Mem-based SIMD 实现 (已确认 — 官方文档)

> Source: hiascend.com CANN 9.0 beta2 — Reg矢量计算编程 (atlas_ascendc_10_10071.html, 2026-04-12 采集)

**核心区别**:

| 维度 | Mem-based (基础 API) | Reg-based (Reg 矢量计算 API) |
|------|---------------------|------------------------------|
| 操作数 | LocalTensor（UB 内存） | RegTensor（VF Reg 寄存器） |
| 数据流 | GM → UB → VEC 计算 → UB → GM | GM → UB → **Register** → VEC 计算 → **Register** → UB → GM |
| 中间结果 | 暂存在 UB（需要 DataCopy 搬进搬出） | **暂存在寄存器中，无需搬出到 UB** |
| 每次处理量 | 完整 LocalTensor（用户定 TILE_SIZE） | **VL 长度**（Vector Length，硬件决定） |
| 灵活性 | 框架管理搬运 | **用户自主控制搬运和计算** |
| 性能 | UB 访问 ~几 cycles | **寄存器访问 ~1 cycle，且减少 UB 搬运** |

**编程模型**:

```
调用层次: __global__ __aicore__ 核函数
           → __aicore__ 函数 (Compute)
             → asc_vf_call<VF函数>(args...)    // 调用 simd vf 函数
               → __simd_vf__ 函数 (AddVF)      // 寄存器级操作
                 → __simd_callee__ 子函数       // 可选嵌套

关键约束:
- __simd_vf__ 函数内只能调用 __simd_callee__ 和 constexpr aicore
- 不能在 __simd_vf__ 内调用 __aicore__ 或 simt 函数
- 不支持从 GM 直接加载到寄存器，必须经过 UB
```

**核心数据类型**:

| 类型 | 说明 | 位宽 |
|------|------|------|
| `RegTensor<T>` | 矢量数据寄存器，VEC 计算基本单元 | VL (Vector Length) |
| `MaskReg` | 掩码寄存器，选择参与计算的元素 | VL/8 |
| `AddrReg` | 地址寄存器，循环中自增偏移 | — |
| `UnalignRegForLoad/Store` | 非对齐缓冲寄存器 | — |

**典型代码模式**:

```cpp
// Reg-based Add: 在 __simd_vf__ 函数中
template<typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr,
                               __ubuf__ T* src1Addr, uint32_t count,
                               uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    Reg::MaskReg mask;
    Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = Reg::UpdateMask<T>(count);  // 每次消耗 VL 元素
        Reg::LoadAlign(srcReg0, src0Addr, aReg);   // UB → Register
        Reg::LoadAlign(srcReg1, src1Addr, aReg);   // UB → Register
        Reg::Add(dstReg, srcReg0, srcReg1, mask);  // Register 内计算
        Reg::StoreAlign(dstAddr, dstReg, aReg, mask); // Register → UB
    }
}

// 在 __aicore__ 函数中调用:
constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
uint16_t repeatTimes = CeilDivision(count, oneRepeatSize);
__ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
asc_vf_call<AddVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize, repeatTimes);
```

**流水线同步**: `Reg::LocalMemBar<src_pipe, dst_pipe>()` — 当写读使用不同寄存器时需要。同寄存器写读自动保序。

**性能优势场景**:
1. **多步融合计算** — 中间结果留在寄存器，省去 UB 搬运（如 `x*smooth → abs → max` 可以在寄存器内完成）
2. **非对齐数据访问** — UnalignReg 优化连续非对齐场景
3. **减少 UB bank 冲突** — 数据在寄存器中不占 UB bank

**对我们系统的影响**:
- 当前所有 kernel 使用 Mem-based（基础 API + LocalTensor）
- Reg-based 需要重写计算循环：`GetPhyAddr()` 获取 UB 地址 → `asc_vf_call` → `__simd_vf__` 函数
- 最大受益场景：多步融合（如 DynamicQuant 的 cast→mul→abs→max 可以在寄存器中流水，省 4 次 UB 搬运）
- **需要实验验证**: 在 A5 上编译 Reg-based kernel 确认编译器支持

状态: **VERIFIED** — 官方文档 + A5 编译验证通过 (2026-04-12)
- CANN 9.0.0 + bisheng 编译器支持全部 Reg API
- 测试文件: `tests/repro/regbase_minimal.cpp` (AddVF: LoadAlign + Add + StoreAlign)
- 编译命令: cmake + ascendc_library, SOC=Ascend950PR_9589, Release 模式
- 下一步: 运行时验证（精度 + 性能对比 mem-based）

### 线程数 vs 寄存器数 tradeoff

```
总寄存器 = 128KB = 32768 个 (4B/reg)
2048 线程: 16 reg/thread — 无法放累加器数组
1024 线程: 32 reg/thread — 可放 8 个 float 累加器
512 线程:  64 reg/thread — 可放 16 个 float 累加器 (MAX_ACCUM=16)
256 线程:  128 reg/thread — 充足，但并行度低

原则: scatter-add 用累加器需要多寄存器 → 减少线程到 512
      纯 elementwise 不需要累加器 → 增加线程到 1024-2048 隐藏延迟
```

### 原子操作优化路径

```
if 可预排序:
    sorted-edge + 寄存器累加 (P-P21) → atomicAdd 次数降 100x+
elif 可分区:
    per-core 本地累加 → 最后一次 atomicAdd 汇总
elif fan-in 低:
    直接 atomicAdd (3.3 cycles/op, L2 coalescing 帮助)
elif fan-in 高:
    WarpReduceAddSync → 每 warp 1 次 atomicAdd (减少 32x)
```

## Known bugs (CANN 9.0.T501 / bisheng 15.0.5)

| Bug | Severity | Workaround | Reference |
|-----|----------|------------|-----------|
| **TQue<VECIN,2> data corruption** | Critical | Use PipeBarrier<PIPE_ALL> instead of double-buffer | 99.5% elements corrupted; output/src/sparse_gather/sparse_gather_simd.h |
| **Typed kernel entry `_fp32` crash** | High | Use legacy entry points (extern "C" __global__) | Error code 507035 |
| **bf16 Cast not supported** | Medium | Use MicroAPI register-level Cast (Pattern F-P3) | bisheng 15.0.5 limitation |

## Compilation

```bash
# NPU mode
cmake .. -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9589 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.T501 \
  -DASC_DIR=/usr/local/Ascend/cann-9.0.T501/lib64/cmake
# Defines: __NPU_ARCH__=3510, uses arch35/ directory
```


---

## Reg-based intrinsics restrictions (W11, 2026-05-12, ROADMAP §1.5)

### `ToFloat<T>` is restricted on A5 — BF16 / FP8 only

A5 (V351, arch35) restricts `ToFloat<T>` template instantiation to:
- `T = bfloat16_t`
- `T = fp8_e5m2_t`
- `T = fp8_e4m3fn_t`

**FP16 (half) is NOT supported directly.** To convert FP16 → float on A5, first reinterpret-cast the tensor view to bfloat16:

```cpp
// V220 (works on A3 with FP16 source directly):
float v = ToFloat(logProbTensor.GetValue(0));     // T deduced as half — OK on A3

// V351 / arch35 (must reinterpret to bfloat16 first):
float v = ToFloat(logProbTensor.template ReinterpretCast<bfloat16_t>().GetValue(0));
```

The reinterpret is a **view-only** operation; no actual data conversion happens. The bit-pattern of the FP16 value is reinterpreted as the bit-pattern of a BF16 value. Because BF16 and FP16 have different bit layouts, this is NOT a value-preserving conversion — it's only valid in code paths where you're about to immediately `ToFloat` the result. Don't hold or use the reinterpreted view for anything else.

### `IsRegbase()` platform-gate API

`Ops::NN::AclnnUtil::IsRegbase()` — returns true on A5 (Ascend950 family / V351), false on A2/A3 (V220). Used in `op_api/<op>.cpp` router functions to gate A5-specific code paths.

Example usage:
```cpp
#include "aclnnop/aclnn_util.h"   // for IsRegbase()
// ...
if (Ops::NN::AclnnUtil::IsRegbase()) {
    // A5 path
} else {
    // A2/A3 path
}
```

When porting v2/v3-shared-aclnn op families to A5, the router-edit primitive #2 from **OL-131 (W9)** extends `IsV<N>AiCoreSupport` SoC checks to allow `IsRegbase()`:

```cpp
// Before:
if ((SocVersion != ASCEND910B) && (SocVersion != ASCEND910_93)) return false;
// After:
if ((SocVersion != ASCEND910B) && (SocVersion != ASCEND910_93) &&
    !Ops::NN::AclnnUtil::IsRegbase()) return false;
```

### `__CCE_AICORE__ == 220` is FALSE on A5

A5 defines `__CCE_AICORE__ == 300` (or equivalently `__NPU_ARCH__ == 3510`). The V220 conditional compile macro `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` is FALSE on A5; bodies inside are not compiled. When porting from A3, strip these blocks entirely (don't convert to `== 300` — see **P-P90 (W10)** for the surgical strip pattern).

### Cross-ref

- **OL-131 / W9**: cross-op router modification using `IsRegbase()` gate
- **P-P90 / W10**: V220→V351 op_kernel strip pattern (the `__CCE_AICORE__ == 220` removal lives there)
- **W8** `ops_nn_a5_artifact_layout.md`: where `<op>_apt.cpp` + arch35/ + op_def regbaseCfg fit

---

## Ascend950 whitepaper additions (2026-06-07)

> Source: 昇腾950 NPU 架构白皮书 (Huawei, 2026). Sections cited inline. These ADD to (do not
> overwrite) the manual-derived content above; where the whitepaper and the prior PDF manual
> overlap the numbers agree (audited 2026-06-07, no conflicts found). Note the whitepaper
> describes the **Ascend950** family (950PR + 950DT, third-gen DaVinci) — the PG-binned A5
> server (`Ascend950PR_9589`, 28 AICore) is one variant; full-die 950PR is 32 AICore.

### STARS2.0 hardware scheduler (§4.4)

`STARS` = **System Task and Resource Scheduler** — the full-chip hardware task-and-resource
dispatch processor (2nd gen on A5). Schedules AIC / AIV / CPU / DVPP / SDMA / UB / CCU engines,
chains data flow, and synchronizes between task streams. This is the silicon behind A5's
multi-core core-split + cross-core sync. See **OL-211** for the op-gen-facing summary.

| Capability | Spec | Note |
|-----------|------|------|
| **Group scheduling** | up to **8 Groups** | per-Group AI-Core count + security attrs software-configurable; intended for **by-Die grouping → L2-cache locality affinity** |
| **Compute partition (算力切分)** | AIC/AIV/SDMA into up to **16 resource pools**; other accelerators ≤8 pools | pools bind to VMs for isolation |
| **Hardware sync flags** | up to **128K single-bit** OR up to **4096 32-bit multi-bit** flags | inter-task-stream sync; the `CrossCoreSetFlag` flagId 0–10 API range is a per-Group allocation convention drawn from this pool, **not the HW ceiling** |
| **HSCB (High-Speed Control Bus)** | STARS↔AIC/AIV dedicated bus, **ns-level** dispatch | broadcast scheduling, interference-free vs NoC (data bus); explains A5's low launch overhead |
| **Task streams** | **2048** host→device | STARS prefetches/schedules per stream config, reports completion to host |
| Concurrent dispatch | ≤16 AI-CPU + ≤64 Host-CPU tasks; ≤64 UB jetty; ≤32 CCU; ≤32 SDMA channels | §4.4 |

Also supports compute-fused-communication tasks and conditional operators; provides real-time
TOP-DOWN Profiling (task time-trace, compute cost, bandwidth, power).

### BufferID sync — mutex-like AI-Core local-memory sync (§4.1.6, NEW on A5)

A5 adds a **BufferID synchronization** mechanism for AI-Core internal storage occupancy, used like
a programming-language mutex: **`get_buf()` = acquire (lock)**, **`rel_buf()` = release (unlock)**.
It directly expresses a pipeline's occupy/release of AI-Core local memory. Versus the older
`set_flag` / `wait_flag` mechanism it is **more cohesive and decoupled from other pipelines**,
lowering sync complexity. (Complements — does not replace — `CrossCoreSetFlag`/`WaitFlag` for the
cross-core case; BufferID is the intra-core local-memory-occupancy primitive.)

### CV-fusion channel — direct Cube-L1 ↔ Vector-UB data path + on-the-fly convert (§4.1.4)

A5 builds a **direct CV (Cube-Vector) data-transfer channel** between **Cube L1 Buffer** and
**Vector Unified Buffer**, raising intra-core data reuse and cutting L2-level data exchange to boost
CV-fusion-operator efficiency (cited native use case: **FlashAttention** bandwidth bottleneck).
Critically, the channel can do **on-the-fly numerical-precision + data-layout conversion during the
Cube→Vector transfer** — i.e. **NZ→ND** layout reformat and FP32→BF16/FP16/FP8 quantization happen
in-flight. This is the hardware confirmation of the **FixpipeC310 ROW_MAJOR (NZ→ND) write path** the
op-gen pipeline relies on (cross-ref §"FIXPipe → UB 直接路径" above + §4.1.1 "回写 Unified Buffer 阶段
直接完成数据量化与排布转换 NZ→ND/DN").

### Vector Core = double-issue Register-Based SIMD + SIMD/SIMT mixed (§4.1.2, §4.1.3)

Whitepaper confirms the project's "regbase" framing: the 3rd-gen Vector Core upgrades from
traditional SIMD to **双发射 Register-Based SIMD (double-issue Register-Based SIMD)** — a RegFile
sits between Unified Buffer and the Vector ALU as temp storage, giving higher bandwidth + data reuse
(matches §"SIMD Register File" / §"Reg-based vs Mem-based" above). Programming model is **SIMD/SIMT
mixed with SIMD primary, SIMT auxiliary** (以 SIMD 为主、SIMT 为辅): most vector compute rides SIMD
(double-issue ALU + out-of-order exec) as the main throughput path; SIMT is the differentiated
enhancement for irregular access (Gather/Scatter) and complex branch (Hash Insert). Per-core FP16/FP32
TFLOPS **+100% vs prev gen**, so the non-matrix ops of a CV-fusion op (e.g. FA softmax) no longer
bottleneck.

### FlashAttention hardware optimization (§3 主要特性 5, §4.1)

A5 has **Transformer-specific** hardware opt: for key operators like FlashAttention, by **fusing the
Cube-Vector path**, enhancing **Softmax compute efficiency** (§4.1.2 micro-arch tuned for
Softmax/GELU — reduces data-dependency "bubbles"), and supporting MXFP8/MXFP4 formats, **single-core
performance is 1.5–2× vs the previous-gen chip**. Larger L0C Buffer (256KB) gives more flexible Tiling
+ higher data reuse for GEMM/FlashAttention. This is the hardware basis for the FA-A5 perf work.
