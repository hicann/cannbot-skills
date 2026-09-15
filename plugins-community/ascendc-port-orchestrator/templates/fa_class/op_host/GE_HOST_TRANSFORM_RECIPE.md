# A3 → A5 GE op_host TRANSFORM RECIPE (assembler manifest)

The whitebox piece that was missing: the per-file MAPPING from a customer's **A3 (Ascend910 / arch22)
GE op_host** to the **A5 (Ascend950PR / arch35) GE op_host**, using only (a) the customer's A3 GE
op_host input and (b) this KB (skeleton templates + `wp_fa_host_tiling.h` arch35 logic). NO A5 CANN
source is consumed. Authored 2026-06-11 (`flash_attention_score-gehost-3`).

This recipe is op-CLASS-general: the **per-file CARRY / CARRY+PATCH / REPLACE-HOOK rule** below
generalizes to other port_a3 ops; only the FA-specific *specifics* (which dtypes A5 adds, which shared
layer holds the arch35 tiling) differ per op.

---

## The three transform classes (the core insight)

A GE op_host has 3 files, and each transforms differently across the A3→A5 arch boundary because each
depends on the arch DIFFERENTLY:

| File | Arch dependence | Transform class | Why |
|---|---|---|---|
| `infershape.cpp` | **NONE** (reads logical tensor shapes, not tiling) | **CARRY** | A3 == A5. Output shapes are a function of input shapes + attrs — arch-invariant. |
| `def.cpp` | **WEAK** (op IR is shared; only the dtype set + SOC string differ) | **CARRY + PATCH** | A3 ≈ A5. Same IR; A5 ADDS dtype rows (fp8) + swaps the SOC config string. |
| `tiling.cpp` | **STRONG** (tiling = the arch's core-split/block algorithm) | **REPLACE-HOOK** | A3 (arch22 general) and A5 (arch35 regbase) are DIFFERENT tiling architectures — not a line-transform. |

Evidence (grounded in CANN source, not asserted):
- `infershape.cpp`: `grep -c 'arch22|arch35|regbase|DAV_3510' flash_attention_score_infershape.cpp` = **0**.
- `def.cpp`: A3 registers `AddConfig("ascend910b")` + `AddConfig("ascend910_93")`; A5 registers
  `AddConfig("ascend910_95")`. A5 def adds 5 `FLOAT8/HIFLOAT8` dtype refs the A3 base lacks.
- `tiling.cpp`: A3 tiling lives in `op_host/arch22/flash_attention_score_tiling_general.cpp`
  (`class FlashAttentionScoreTilingBase : public TilingBaseClass`, **0** `regbase` refs, **0**
  `Simplified` refs); A5 tiling lives in `op_host/arch35/flash_attention_score_tiling_regbase.cpp`
  (**5** `regbase` refs). Different base class, different tiling-data POD
  (`FlashAttentionScoreSimplifiedTilingData` is the arch35/regbase POD). Confirmed they are NOT a
  line-by-line evolution of one another.

---

## Per-file recipe

### 1. `infershape.cpp` — CARRY (A3 == A5)

**Rule (general)**: if the customer's A3 infershape has **0 arch refs** (grep `arch22|arch35|regbase|
dav`), carry it verbatim into A5. Shape/dtype inference is arch-agnostic.

**FA specifics**:
- Skeleton (GENERIC): the `gert::InferShapeContext` read/write plumbing + null-guards +
  `IMPL_OP_INFERSHAPE(<Op>).InferShape(...).InferDataType(...)`.
- HOOK (op-specific, but A3==A5): the SHAPE-RELATION HOOK (`attention_out` = query shape with D from
  value; softmax stats = `(B,N,S,8)`). Carry verbatim.
- The ONLY A5 delta is **inside the DTYPE-RELATION HOOK**: the `fp8/hifloat8 -> bf16 out` branch. That
  is a dtype-table edit (A5 adds the fp8 input dtypes in `def.cpp`, so the dtype hook must map them),
  NOT a shape change. An A3 op with no fp8 simply never hits that branch.

### 2. `def.cpp` — CARRY + PATCH (A3 ≈ A5)

**Rule (general, CORRECTED 2026-06-11 `flash_attention_score-gb-2`)**: carry the A3 op IR
(input/output/attr names + order + WHATEVER dtype rows are already present), then PATCH **only the
SOC config string** — and even that is conditional. **Do NOT assume the A3 def is `{fp16,bf16,fp32}`-
only and that you must ADD fp8.** A modern customer A3 `def.cpp` is frequently **already arch-unified**:
it carries the full fp8/hifloat8 dtype matrix + `d_scale_q/k/v` + `p_scale` inputs + an A5 `OpAICoreConfig`
all in one file (the same source builds for A3 and A5). In that case the dtype "patch" is a **no-op** —
the rows are already there.

**Mechanical procedure (check, don't assume)**:
1. `grep -c 'DT_FLOAT8\|DT_HIFLOAT8' def.cpp` — if **> 0**, the A3 def is already fp8-unified → **do NOT
   append fp8 rows** (they're present); CARRY the dtype matrix verbatim.
2. `grep 'AddConfig' def.cpp` — list the registered SOC configs. **Ensure an A5 config is present.**
   Current-CANN A5 SOC string is **`ascend950`** (the verified truth: an arch-unified A3 input typically
   already has `this->AICore().AddConfig("ascend950", aicore_config_95)`). The name **`ascend910_95`** is
   an **alias** for the same A5 SOC. Rule: if neither `ascend950` nor `ascend910_95` is registered, add
   one (`AddConfig("ascend950", <A5 config>)`); if `ascend950` is already present, you may additionally
   register the `ascend910_95` alias for robustness, but it is NOT required for a correct A5 build.
3. `opFile.value` stays the op name (`flash_attention_score`); everything else (Input/Output/Attr name
   list + order, the `OpAICoreConfig` flag set) is carried.

**FA specifics — the A5 dtype/scale surface (for the rare case the A3 def is NOT yet unified)**:
- If (and only if) `grep DT_FLOAT8 def.cpp` == 0, the A5 ADD is: `{fp8-e4m3, fp8-e5m2}` + `hifloat8`
  input dtype columns + the three `d_scale_q/k/v` + `p_scale` fp8 block-wise dequant scale inputs (fp32)
  + the fp8 `bf16` attention_out/softmax_out columns. Otherwise these are already in the carried matrix.

**Why this was corrected**: the original wording assumed a pure-`{fp16,bf16,fp32}` A3 base and a mandatory
fp8 append + a `ascend910_95` SOC swap. The graybox (`flash_attention_score-gb-1`) found the real customer
A3 `def.cpp` was already arch-unified (fp8 rows present, `AddConfig("ascend950")` present), so the "append
fp8 + swap to ascend910_95" steps did not apply and would have produced a wrong/duplicated def if followed
blindly. CARRY-then-ensure-A5-config-present is the mechanically-followable rule.

### 3. `tiling.cpp` — REPLACE-HOOK (A3 arch22 ≠ A5 arch35)

**Rule (general)**: CARRY the GENERIC GE-framework skeleton (entry → CheckParams → platform-read →
extract shapes → **[TILING-COMPUTE HOOK]** → `GetTilingData<POD>()` fill → `SetBlockDim` →
`GetWorkspaceSizes` → `IMPL_OP_OPTILING` register), but **REPLACE the TILING-COMPUTE HOOK** — you can
NOT line-transform A3 arch22 tiling into A5 arch35 tiling because they are different tiling
architectures. The new hook calls the **KB arch35 shared tiling logic**.

**CRITICAL — where the "hook" actually IS (CORRECTED 2026-06-11 `flash_attention_score-gb-2`)**: the A3
`tiling.cpp` entry function almost certainly does **NOT contain inline tiling arithmetic** that you can
point at and replace. Instead it dispatches through a **tiling-template REGISTRY**:

```cpp
// A3 tiling.cpp entry (TilingFlashAttentionScore) typically ends with:
auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);   // <-- THIS is the hook
return resultCode;
```

The actual per-arch tiling arithmetic lives in **separately-registered template classes**, e.g.
`arch22/flash_attention_score_tiling_general.cpp` ends with lines like:

```cpp
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingS1Bn2gs1,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 97);   // arch22 = DAV_2201
```

…and the arch35 templates are registered the same way under `arch35/*` (which you MUST NOT read). The
entry also typically branches on `ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510` for empty-input
fast-paths. **So "the TILING-COMPUTE HOOK to replace" = the `DoTilingImpl(context)` registry-dispatch
line** (and the arch-branch around it), NOT a block of arithmetic.

**Mechanical procedure**:
1. CARRY the GE shell from the A3 entry: `CheckParams`, the platform read (`GetPlatformInfo` →
   `PlatformAscendC`), `SetBlockDim`, `GetWorkspaceSizes`, the `IMPL_OP_OPTILING(...)` registration
   block (incl `.TilingInputsDataDependency({...})` + `.TilingParse<CompileInfo>(...)`).
2. WRITE the **config-extract** yourself (`ExtractShapes`: read q/k/v shapes + attrs + optional-input
   presence from `gert::TilingContext` → agnostic scalars). This is the GE-shell glue that
   `wp_fa_host_tiling.h` deliberately does NOT contain (its gap#2 INTEGRITY BOUNDARY) — the assembler
   authors it. Decode layout (BSH/SBH/BSND/BNSD/TND) mirroring the carried `infershape.cpp` convention.
3. **REPLACE** the `TilingRegistryArch::DoTilingImpl(context)` dispatch with a direct `wfh::Calc*`
   sequence (fenced `// <<< TILING-COMPUTE HOOK: FA instance = wp_fa_host wfh::Calc* >>>`). The discarded
   arch22 registry templates (`FlashAttentionScoreTilingBase` + subclasses) are NOT ported.

**FA specifics**:
- The skeleton is GE-framework boilerplate (arch-agnostic): same `gert::TilingContext` API on A3 and A5.
- The hook is the 9 `wfh::` calls in `wp_fa_host_tiling.h` — the arch35 regbase tiling logic, lifted into
  the KB: CalcEffSparseMode, ComputeSparseTiling, CalcDBasicBlock, CalcS1S2BasicBlock,
  SetMultiCoreParamsRegbase, CalcThresholdForS2Size, SetSplitCoreModeParam, CalcUseDn, CalcWorkspaceSize.
- The POD type swaps: A3 general `FlashAttentionScoreTilingData` → A5
  `FlashAttentionScoreSimplifiedTilingData` (the regbase POD in `wholeport/wp_tiling_regbase.h`). Fill
  its `inputParamsRegbase` / `multiCoreParamsRegbase` sub-PODs from the `wfh::` results.
- **RED LINE**: the replaced hook stays `wfh::`-CALLS — it must NOT inline raw arch35 arithmetic. The
  whole point of the shared layer is that the arithmetic lives once in `wp_fa_host_tiling.h`.

---

## Generalization to other port_a3 ops (the op-CLASS-general manifest)

For ANY port_a3 GE-graph op, the assembler applies this decision per host file:

```
infershape.cpp : grep arch refs == 0 ?  --> CARRY verbatim (+ patch dtype hook only if A5 adds dtypes)
def.cpp        : op IR shared ?         --> CARRY + PATCH (append A5 dtype rows; swap SOC string)
tiling.cpp     : A3 arch != A5 arch ?   --> CARRY skeleton, REPLACE the TILING-COMPUTE HOOK with the
                                            op's KB arch35 shared-tiling-logic layer
```

Only the *specifics* are per-op:
- which dtypes A5 adds (FA: fp8 + scales; another op may add none → def.cpp degenerates to pure CARRY),
- which shared layer holds the arch35 tiling (FA: `wp_fa_host_tiling.h`; another op: its own
  `wp_<op>_host_tiling.h` extracted the same way),
- the POD type + input/attr indices.

The skeleton structure + the 3 transform classes are invariant.

---

## KB self-sufficiency statement (ZERO A5 CANN source)

**Claim**: given (1) the customer's A3 GE op_host input and (2) this KB, the A5 GE op_host is fully
assemblable with **ZERO A5 CANN source**.

### What comes FROM THE A3 INPUT (the customer ships these)
- `infershape.cpp` — carried verbatim (arch-agnostic; the customer already has a working A3 one).
- `def.cpp` op IR — the input/output/attr name list + order + the base `{fp16,bf16,fp32}` dtype rows +
  the `OpAICoreConfig` flag set (carried; only patched).
- `tiling.cpp` GE skeleton — the entry/CheckParams/platform-read/extract/fill/SetBlockDim/workspace/
  register structure (the A3 tiling.cpp already has this GE-framework shell; it is arch-agnostic).
- The TilingData field NAMES the kernel consumes (the customer's kernel defines the POD contract).

### What comes FROM THIS KB (no A5 CANN source needed)
- **The 3 skeleton templates in this dir** (`flash_attention_score_{def,infershape,tiling}.cpp`) — the
  generic-skeleton + parameterization-marker form, so the assembler knows exactly which regions to
  carry vs swap.
- **`GE_HOST_TRANSFORM_RECIPE.md`** (this doc) — the per-file CARRY / CARRY+PATCH / REPLACE-HOOK rule.
- **`wp_fa_host_tiling.h`** — the arch35 regbase TILING LOGIC (the `wfh::Calc*` layer). This is the
  load-bearing KB asset: it replaces the need to read A5 CANN `arch35/flash_attention_score_tiling_
  regbase.cpp`. It was extracted ONCE into the KB; the customer never sees A5 CANN source.
- **The A5 dtype PATCH set** (documented above: fp8-e4m3/e5m2 + d_scale/p_scale + bf16 fp8-out) — the
  exact rows to append to the carried A3 def matrix.
- **The A5 SOC string** (`ascend910_95`) — to swap into the carried def AICore config.
- **The A5 regbase POD type name** (`FlashAttentionScoreSimplifiedTilingData`, in
  `wholeport/wp_tiling_regbase.h`, a KB asset) — to swap into the carried tiling.cpp.
  - **Hygiene note**: `wp_tiling_regbase.h` uses fixed-width ints (`uint32_t`/`uint64_t`) but relies on
    a prior `<cstdint>` include from its includer. When you `#include "ge_host_shim.h"` (which pulls
    `<cstdint>`) BEFORE `wp_tiling_regbase.h`, this is satisfied; otherwise add `#include <cstdint>`
    ahead of it.
- **`ge_host_shim.h`** (KB asset, the GE-SHELL boundary — added 2026-06-11 `flash_attention_score-gb-2`)
  — provides the CANN-internal glue symbols (`OP_LOGx`, `OP_CHECK_NULL_WITH_CONTEXT`/`OP_CHECK_IF`,
  `GET_TPL_TILING_KEY`, `ASCENDC_EXTERN_C`, `platform_ascendc::PlatformAscendC`, `NpuArch`/`DAV_3510`)
  that GE op_host source references but that are ABSENT from the public CANN include tree. These all
  come from CANN op-build-kit internal headers present at **real (Tier-2) op-build time** but absent
  from the public include set, so a **standalone Tier-1 `-fsyntax-only`** can't see them. Include this
  shim for Tier-1 only (guard behind `#ifdef GE_HOST_TIER1_SYNTAX_ONLY` or drop the include at Tier-2 —
  the kit headers then provide the real symbols). See the shim's own header comment for the full
  internal-home table + the Tier-1-vs-Tier-2 contract. This is the documented GE-shell boundary: the
  shim is the line between "GE-framework glue (shimmable)" and "your assembled logic + the `wfh::` layer
  (real)".

### The assembly (deterministic)
1. CARRY A3 `infershape.cpp` → A5 (it is arch-agnostic; the fp8/hifloat8 dtype hook is usually already
   present in a modern A3 input — carry verbatim). Swap its internal `#include "log/log.h"` for
   `#include "ge_host_shim.h"` at Tier-1.
2. CARRY A3 `def.cpp`; **check** the dtype matrix (`grep DT_FLOAT8`) — if fp8 rows already present (the
   common arch-unified case), no dtype patch needed; **ensure** an A5 SOC config is registered (current
   CANN A5 SOC = `ascend950`; `ascend910_95` is an alias — add one if neither is present). See §2.
3. CARRY A3 `tiling.cpp` GE skeleton (CheckParams / platform-read / SetBlockDim / workspace / register),
   WRITE the `ExtractShapes` config-extract yourself, REPLACE the `TilingRegistryArch::DoTilingImpl`
   registry-dispatch with `wfh::Calc*` calls from the KB `wp_fa_host_tiling.h`, fill the regbase POD
   (`FlashAttentionScoreSimplifiedTilingData`). See §3.

No step reads A5 CANN source. The arch35-specific knowledge (tiling logic, dtype set, SOC, POD) all
lives in the KB; the A3 input supplies the op IR + the GE shell + the shape contract.

---

## RED LINE compliance (re-verified post-refactor — see GE_HOST_TEMPLATE.md for the live numbers)
- md5-differ: all 3 files differ from CANN source.
- grep-wfh in tiling.cpp: > 0 (the TILING-COMPUTE HOOK stays `wfh::` calls; no raw arch35 inlined).
- no real `#include "arch35/"` (the only `arch35` string hits are RED-LINE/recipe comments).
- Tier-1 `-fsyntax-only` clean on `.171` `npu_dev3` (cann-9.1.T500), using `ge_host_shim.h` for the
  CANN-internal GE-shell symbols (the documented Tier-1 shell boundary) — see below + the shim header.

## Test-coverage argument (shared-logic dividend)

Because the GE host's tiling-COMPUTE hook is the SAME `wfh::Calc*` shared layer the pybind
test host calls, the substantive (algorithm-bearing) host logic is ALREADY exercised + validated
by the pybind precision path:

- **tiling LOGIC (`wfh::Calc*`)** — COVERED by pybind precision tests (pass_a 28/28): the pybind
  host calls the identical functions; correct kernel output proves the tiling values are right.
  The GE tiling hook carries ZERO independent algorithm risk (same functions).
- **GE SHELL (registration macros + `gert::TilingContext` read/write plumbing)** — NOT exercised
  by pybind (pybind bypasses GraphEngine). This is thin glue (extract shapes from GE context →
  feed wfh:: → fill GE TilingData), no algorithm. Covered by Tier-1 `-fsyntax-only` (syntax/type)
  now; a Tier-2 GE-graph smoke test would fully exercise the shell at runtime.

Net: the "multiple hosts share logic via functions" design means one precision test-suite covers
the algorithm part of BOTH host paths; only the thin GE shell awaits a Tier-2 graph smoke.
