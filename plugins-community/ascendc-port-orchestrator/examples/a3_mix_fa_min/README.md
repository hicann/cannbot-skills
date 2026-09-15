# a3 MIX_AIC_1_2 cross-core SYNC-WITNESS — handshake only (NOT an op)

`arch_scope: arch22 · 220x · soc=Ascend910_9382 · product=Atlas A3 · dav-2201 · cann=9.0.0`

> **This is a SYNC-WITNESS, not an operator.** It exists to be **BUILT + RUN**
> so a worker can **WITNESS** the a3 `MIX_AIC_1_2` cross-core AIC↔AIV handshake
> (**FLAG_S broadcast + FLAG_P both-AIV-set**) complete **deadlock-free**. The
> compute between the flags is a **PLACEHOLDER** (a trivial identity copy) —
> there is **NOTHING here to copy for a real op**. Learn that the handshake
> works, then **GENERATE YOUR OWN** kernel + host from the KB template
> **P-P116** (`patterns/domains/fa_class_a3_mix_template.md`) using CANN/catlass
> library primitives. See "Why there is nothing to copy" below.

## Why this exists

The op-gen pipeline had only a **prose** MIX skeleton (P-P116) and would decline
to hand-roll a fused single-kernel a3 attention MIX from prose — reasonably,
because the `MIX_AIC_1_2` sync is a known hang minefield (PB-34 / PB-35 / PB-55)
and "trust me it's safe" is not evidence. **Prose alone did not suffice.** This
witness is the missing **compilable + runnable** proof: a worker builds it and
watches `torch.npu.synchronize()` *return* (not hang), which is what let the
pipeline trust the cross-core handshake and go on to **generate** the fused MIX
kernel. The build+run witness is the load-bearing artifact; the prose is not.

## Why there is nothing to copy (the anti-copy backstop is STRUCTURAL)

The compute has been **deliberately hollowed out to a placeholder**:

- **The AIV step is a trivial identity copy** (`P = S`, a per-row `DataCopy`).
  The original numerically-stable softmax step-sequence and its hand-written
  `RowReduceMax` / `RowReduceSum` / `Align8` / `FloorPow2` helpers were
  **DELETED**. There is no real-op arithmetic left to lift.
- **One fixed shape only** — `seq=128, d=64`, fp16, **one head, one batch**. The
  pybind shim (`mix_fa_min_pybind.cpp`) *hard-rejects* every other shape.
- **No operator at all** — no softmax, no scale, no causal / attention mask, no
  dropout, no multi-dtype dispatch, no multi-head / batch loop, no KV-block
  tiling, no perf tuning.

So there is **no working operator to copy** — only a handshake to witness. What
you take away is the *knowledge that the sync closes deadlock-free*, not any
compute. The **DEBT-215** external-import / verbatim-copy scanner flags any
attempt to import a teaching reference as an op's compute. Learn the *handshake*
here; the *kernel* for your op is yours to **generate**.

## What is genuine (the load-bearing witness) vs placeholder

| part | status | why kept / why placeholder |
|---|---|---|
| `MIX_AIC_1_2` dispatch, AIC/AIV partition | **GENUINE** | the sync only exists in this launch mode |
| `S = Q @ Kᵀ` cube#1 (`MatmulImpl<>` + `IterateAll<sync=true>`) | **GENUINE** | real cube work must grab the FFTS slots for the handshake to be a real witness (not a mock) |
| `O = P @ V` cube#2 (`MatmulImpl<>` + `IterateAll<sync=true>`) | **GENUINE** | same — genuine cube on the far side of FLAG_P |
| PB-55 handshake: FLAG_S broadcast, FLAG_P **both-AIV-set** | **GENUINE** | this asymmetry is the whole point being witnessed |
| AIV op between the flags | **PLACEHOLDER** | `P = identity(S)` trivial `DataCopy` — nothing liftable |

## The handshake this witnesses (cross-referenced to the KB)

The device chain, one head, one batch, single tile:

```
S = Q @ K^T           (cube #1, AIC)   [seq, seq]   GENUINE MatmulImpl
P = identity(S)       (vector, AIV)    PLACEHOLDER — trivial DataCopy
O = P @ V             (cube #2, AIC)   [seq, d]     GENUINE MatmulImpl
```

Cross-sync chain (2 handshakes, DISTINCT non-zero flag ids per PB-35):

```
AIC : cube#1 -> CrossCoreSetFlag<MODE2,PIPE_FIX>(FLAG_S)   (S ready, BROADCAST)
      CrossCoreWaitFlag(FLAG_P)                            (block for P)
      cube#2                                               (O = P@V)
AIV : CrossCoreWaitFlag(FLAG_S)                            (block for S)
      identity(S)->P ; CrossCoreSetFlag<MODE2,PIPE_MTE3>(FLAG_P)  (BOTH subblocks)
```

1. **Dispatch — `MIX_AIC_1_2`, standard `aclrtlaunch`, NO task-type macro.**
   One `__global__ __aicore__` entry; the default MIX launch reaches BOTH cores;
   `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` partition. On **arch22** do **NOT**
   emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` — arch35-only,
   rejected on Ascend910_9382 → `107000` at `RegisterAscendBinary` (**PB-28**).

2. **Two GENUINE cube matmuls — `MatmulImpl<>` + `IterateAll<sync=true>`.**
   Never the async KFC `Iterate()`/`GetTensor()` path — that grabs the AIC↔AIV
   FFTS sync slots and deadlocks in this MIX mode (**PB-34**). The cubes are kept
   genuine because the handshake is only a *real* witness if real cube work sits
   on either side of the flags. (P-P116 §2 / §4.)

3. **FORWARD `FLAG_S` is BROADCAST; REVERSE `FLAG_P` is per-subblock COUNTED
   (PB-55 — the load-bearing rule).** One AIC `CrossCoreSetFlag(FLAG_S)` releases
   *both* AIV subblocks. But the single AIC `CrossCoreWaitFlag(FLAG_P)` requires
   a set from *every* AIV subblock of the 1:2 pair — so **BOTH** AIV subblocks
   run the (identical → benign) placeholder copy and **BOTH**
   `CrossCoreSetFlag(FLAG_P)`. A single-setter reverse hangs the AIC wait forever
   (measured DEADLOCK, no fault code). (P-P116 §3.)

4. **Distinct, non-zero flag ids.** `FLAG_S = 4`, `FLAG_P = 5` (user range 1..7).
   **Never flag id 0** — `event_t(0)` collides with the cube-internal pipe-sync
   chain → silent hang (**PB-35**). Mode `MIX_SYNC_MODE2` (mode 2) suffices on
   arch22. (P-P116 §4.)

## Build + run recipe (a3 CANN 9.0.0 container)

Prerequisites: an **a3 CANN 9.0.0 container** (arch22 / dav-2201, `Ascend910_9382`),
`torch` + `torch_npu` importable, `cmake ≥ 3.16`, the CANN `bisheng` toolchain.

```bash
# 1. Point at the CANN 9.0.0 install and source its env.
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann   # -> cann-9.0.0
source "$ASCEND_CANN_PACKAGE_PATH/set_env.sh"
export ASCEND_HOME_PATH="$ASCEND_CANN_PACKAGE_PATH"

# 2. Configure + build (from this directory).
cmake -B build \
      -DSOC_VERSION=Ascend910_9382 \
      -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build -j4
# -> build/_a3_mix_fa_min_example.cpython-*-aarch64-linux-gnu.so

# 3. Run on an idle NPU (device index maps via ASCEND_RT_VISIBLE_DEVICES).
export ASCEND_RT_VISIBLE_DEVICES=<idle-device>
export LD_LIBRARY_PATH="$ASCEND_CANN_PACKAGE_PATH/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH"
python3 run_example.py
```

`run_example.py` launches the fixed 128×64 chain and calls
`torch.npu.synchronize()`. If the reverse handshake were mis-wired
(single-setter), the process would **hang there forever** — witnessing it
*return* is the whole point. It also re-runs in-process to check bit-exact
determinism, and does a *secondary* cube-chain sanity check (`O ≈ (Q@Kᵀ)@V`,
the placeholder-consistent golden) only to confirm the two genuine cubes really
ran and the handshake delivered `S → P → cube#2`.

## Device-confirmed result

Built + ran on an **a3 CANN 9.0.0 container** (arch22 / dav-2201, `Ascend910_9382`),
one idle NPU:

```
[seq=128 d=64] handshake_closed=True (sync returned in 1.08ms) | cube-chain: \
               max_abs=2.5000e-01 rel=6.4433e-04 cosine=0.99999928 -> OK
WITNESS: MIX_AIC_1_2 AIC<->AIV handshake closed DEADLOCK-FREE (sync returned)
CUBE-CHAIN SANITY: OK
DETERMINISM (in-proc re-run bit-exact): True
```

The `MIX_AIC_1_2` chain **closes deadlock-free** (`torch.npu.synchronize()`
returned; no hang; no fault code); the two genuine cubes ran (cube-chain
cosine `0.99999928` vs the placeholder-consistent golden); bit-exact
deterministic across re-runs. There is no operator here — only the witnessed
handshake.

## Files

| file | role |
|---|---|
| `mix_fa_min_kernel.h`     | matmul config + the two GENUINE cube matmuls + the PLACEHOLDER identity copy (softmax + reduce helpers deleted) |
| `mix_fa_min_kernel.cpp`   | the `__global__` entry — AIC/AIV partition + the PB-55 2-handshake cross-sync chain |
| `mix_fa_min_pybind.cpp`   | pybind shim — metadata/alloc/contiguous only; hard-fixed to the 128×64 shape |
| `CMakeLists.txt`          | self-contained standalone build (CANN `ascendc_kernel_cmake`; no op-gen harness) |
| `run_example.py`          | build-and-run witness — headline deadlock-free check + determinism + cube-chain sanity |

## Cross-references

- **P-P116** (`patterns/domains/fa_class_a3_mix_template.md`) — the prose
  skeleton to **generate** your kernel + host from (three-stage pointer wiring).
  This witness proves the handshake it describes; the compute is yours to write.
- **PB-55** — the `MIX_AIC_1_2` AIC↔AIV handshake asymmetry (FLAG_S broadcast;
  reverse FLAG_P per-subblock counted, both-AIV-set; single-setter deadlocks).
  This is the load-bearing rule the witness exercises.
- **PB-34** — `MatmulImpl<>` + manual `CrossCoreSetFlag` + async KFC deadlock on
  a3 → why `IterateAll<sync=true>` is mandatory for the genuine cubes.
- **PB-35** — `event_t(0)` cube-internal pipe-sync collides with the cross-core
  flag chain → why flag ids are non-zero and distinct.
- **PB-28** — `KERNEL_TASK_TYPE_DEFAULT` is arch35-only (`107000` on arch22) →
  why no task-type macro here.
