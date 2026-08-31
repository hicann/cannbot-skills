---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Hardware-intrinsic structs with undocumented field semantics — compile-clean is NOT a success signal; require hardware verification before claiming a primitive works"
description: "applies_to: soc=all; cann=all; bisheng=any; op_class=any_kernel_using_low_level_hardware_primitive_with_opaque_struct_fields verified_on: a5_ops:3_FusionAttention kw-2/kw-3/kw-4 chain 2026-05-21 — bes"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=any; op_class=any_kernel_using_low_level_hardware_primitive_with_opaque_struct_fields"
confidence: inferred
status: stub
original_id: CAND-OPAQUE-STRUCT-RUNTIME-VERIFY
timestamp_inferred: true
tags: [candidate, inferred, copy_gm_to_cbuf_multi_nd2nz_b16, dstnzc0stride, dstnznstride, dstnzmatrixstride, mmad, cand-opaque-struct-runtime-verify]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; bisheng=any; op_class=any_kernel_using_low_level_hardware_primitive_with_opaque_struct_fields`
`verified_on: a5_ops:3_FusionAttention kw-2/kw-3/kw-4 chain 2026-05-21 — best-effort Nd2NzParams field guessing compiled cleanly across 3 iters, faulted at runtime with 3 distinct fault signatures; correct field shape only confirmed in iter 3 after directive sourcing concrete API field values from V220 SDK headers`
`derived-from: empirical Nd2NzParams field-shape divergence; generalizes to any opaque hardware-intrinsic struct`

**Principle**: When a kernel primitive's parameter struct contains fields that forward 1:1 to a hardware intrinsic AND the public SDK header has NO docstring explaining the fields' semantics (only field types), best-effort guessing based on field names produces kernels that compile cleanly but fault at runtime. The bisheng / CANN toolchain accepts ANY valid C++ type signature on these calls; there is no compile-time check on stride field values, layout decoding, event-ID safety, or fragment alignment. Workers MUST verify on hardware (≥1 routed test case yielding a clean run) before claiming the primitive works — "compile clean" is not a success signal for these primitives.

**Concrete anchor** (the Nd2NzParams case — empirical evidence chain):

```cpp
// Public SDK declaration in basic_api/kernel_struct_data_copy.h L257-304:
struct Nd2NzParams {
    uint16_t ndNum;
    uint16_t nValue;
    uint16_t dValue;
    uint16_t srcNdMatrixStride;
    uint16_t srcDValue;
    uint16_t dstNzC0Stride;   // ← OPAQUE: no doc on semantic meaning
    uint16_t dstNzNStride;    // ← OPAQUE
    uint16_t dstNzMatrixStride; // ← OPAQUE
};
// Constructor signature shows field order; no inline doc on what each stride means.
// Backing impl (dav_c220/kernel_operator_data_copy_impl.h L304-316) forwards
// these 1:1 to a hardware intrinsic `copy_gm_to_cbuf_multi_nd2nz_b16` which
// is ALSO undocumented in public SDK.
```

For case S=64, D=64 fp16: 5 of 8 fields are clearly derivable (`ndNum=1, nValue=S, dValue=D, srcNdMatrixStride=0, srcDValue=D` for contiguous ND); 3 stride fields (`dstNzC0Stride`, `dstNzNStride`, `dstNzMatrixStride`) require knowing the EXACT NZ packing semantics on V220 — which differ from arch3510 packing. kw-2 iter 2 guessed `dstNzC0Stride=S=64`; runtime fault `507015 aicore exception` on first `Mmad`. kw-4 iter 3 sourced concrete value `dstNzC0Stride=D/16=4` from a V220 SDK header reading; fault transitioned to `0x8000004000 L0B read/write conflict` (different fault layer — confirms shape passes L1-decode). The transition between fault signatures is the empirical proof that field semantics matter and best-effort guessing is unreliable.

**Anti-pattern (BANNED — caught across 3 iters / 4 worker spawns)**:
```cpp
// kw-3 iter 2 ── BAD: guessing dstNzC0Stride from field-name intuition
AscendC::DataCopy(l1Dst, gmSrc, AscendC::Nd2NzParams{
    1, S, D, 0, D,
    /*dstNzC0Stride=*/S,  // ← guessed "looks like inner-dim row count"
    /*dstNzNStride=*/16,  // ← guessed "N-stride sounds like 16 = C0 block"
    /*dstNzMatrixStride=*/0
});
// Compiles clean. Runtime: 507015 aicore exception. No compile/build warning.
```

**Correct pattern (only after SDK header reading + cross-verified field semantics)**:
```cpp
// kw-4 iter 3 ── correct after sourcing field semantics from SDK header:
AscendC::DataCopy(l1Dst, gmSrc, AscendC::Nd2NzParams{
    1, M, K, 0, K,
    /*dstNzC0Stride=*/K/16,  // ← #16-elem C0 strips per L1 row, not row count
    /*dstNzNStride=*/16,     // ← 16 rows per N-stride (canonical NZ blocking)
    /*dstNzMatrixStride=*/0
});
// Compiles clean. Runtime: L1-decode succeeds. Fault (if any) is at later layer.
```

**Generalized workflow guidance**:
1. **Read the SDK header for field semantics first** — every field with `uint16_t`/`uint8_t` type but no docstring is an opaque field. Grep the impl `.h` files for the field name to find the hardware-intrinsic invocation that consumes it.
2. **Verify on hardware before broadening the change** — write a minimal kernel that uses the primitive on a single shape, build it, run it, observe the fault signature (or PASS). Do NOT propagate the primitive across multiple kernel functions until the single-case verification passes.
3. **When SDK header lacks field docs entirely** (the Nd2NzParams case): escalate to `aog-cann-learner` Mode 5 extraction or `aog-hardware-probe` skill to seed API_CATALOG.md before relying on the primitive in production code.
4. **Run-time fault signatures encode the actual gap layer** — `507015 aicore exception` ≠ `0x8000004000 L0B read/write conflict` ≠ silent hang. The fault-signature transition across iters tells you which layer was solved (layout vs sync vs deeper). Treat each new fault signature as positive progress; don't conflate "still failing" with "no progress".

**Other instances (predicted)**:
- `MmadParams` (cube tile-MMAD): fields `isBias`, `cmatrixInitVal`, `cmatrixSource` interact with prior pipe state in non-obvious ways.
- `LoadData2DParams` / `LoadData3DParamsV220`: `ifTranspose`, `addrMode`, `dilation*` fields lack semantic docstrings for V220.
- `FixpipeParamsV220`: numerous mask / nz2nd config fields driving the FIX-pipe behavior.
- Future V220-specific intrinsic structs that get added to the SDK without docs.
- Any `event_t(N)` parameter — looks like an integer ID but interacts with cross-core sync semantics (PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP).

**Cross-ref**:
- OL-130 (API surface lookup chain — SDK header reading is load-bearing for opaque-field primitives)
- API_CATALOG.md (the appropriate destination for verified opaque-field semantics)
- CAND-FA1 (where the Nd2NzParams field shape is documented as verified)
- PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (companion case — event_t(N) is an opaque parameter whose semantics interact with MIX_AIC_1_2 sync infra)

**Promotion path**: candidate to OL-class entry once 2+ ops independently demonstrate the workflow (SDK header reading → hardware verification → API_CATALOG seeding) on different opaque-field primitives. The Nd2NzParams + Mmad/LoadData2DParams + FixpipeParamsV220 sweep on a single FA-class op would satisfy this; alternatively, future ops using a different opaque-field primitive (e.g. a new V351 / V220 quant primitive) on similar workflow would.

### CAND-FA-CANON-FREE: Eliminate AIV layout-canonicalize stage via `MatmulImpl::SetOrgShape` 5-arg variant for strided GM→L1 load + per-row contig AIV postprocess for output layout transform

`applies_to: soc=all (V220 verified; V351 unverified); cann=9.0.0+; op_class=fused-attention or any cube-heavy op needing layout-aware GM gather from BSH/SBH/BSND/BNSD source; correctness_scope: S*D ≤ UB_BUDGET_8192 AND S*Skv ≤ UB_BUDGET_8192 (current kw-1-derived algorithm; row-tiled FlashAttention rewrite tracked as DEBT-FA-row-tiled)`
`verified_on: a5_ops:3_FusionAttention kw-5 structural rewrite (2026-05-22, A3 npu-a3-test) — multi-shape sweep: BNSD/BSH/SBH/BSND × N∈{1,2,4} × shapes within UB budget all produce max_abs ≤ 6.1e-5 PASS_T1; perf 0.603× CANN at B=1,S=64,N=2,D=64,BSH,fp16 (56.9 µs vs CANN 34.3 µs). Pass B VEC fallback 9/9 preserved.`
`unverified_on: V351 (Ascend950PR / A5) — pattern likely portable but matmul library behavior across arch versions needs verification before broad scoping; future iter to confirm`

**Supersedes**: PB-36 (now ARCHIVED) — PB-36's "V220 DataCopy hardware bug + Python permute workaround" framing was wrong. CANN's own `aclnnFlashAttentionScoreV2` works on V220 without any such workaround; the bug was in our design choice to use an AIV canon stage at all.

**Pattern overview**: For FA-class ops where source tensors live in non-canonical layouts (BSH/SBH/BSND), the natural-seeming "AIV canon stage → BNSD scratch → mm1/mm2 read scratch" approach is fragile because the canon stage requires `DataCopy` with non-zero `srcStride`/`dstStride` which exhibits silent wrong-output behavior on V220 CANN 9.0.0 for the FA-specific tile dimensions. The fix is to **never materialize a canonical layout in scratch** — instead use matmul's native strided GM→L1 load capability, and only do a final per-row contig transform for the OUTPUT side (where matmul C-write doesn't support stride).

**Principle**:
1. **`MatmulImpl::SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc=0)` is the stride mechanism**, despite the header docs misnaming `orgKa` as "K-axis size" — these fields encode **physical leading dimensions** (row strides including nested-axis interleaving). The 5-arg variant with explicit `orgKc` is required when B's N-axis size differs from C's.
2. **Per-(b,n_head) GM offset**: `head_off = b*sB + n_head*sN`. `SetTensorA(qGm[head_off])` + `SetOrgShape(orgM=S, orgN=sS, orgKa=sS, orgKb=sS, orgKc=S_or_D)` lets matmul do strided GM→L1 loads internally. BNSD degenerates (sS=D, no stride overhead); BSH/SBH/BSND get correct strided gather.
3. **Matmul C-side does NOT accept layout stride** — empirically tested, BSH/SBH/BSND output writes produce 5–38% error when attempted. The C-side always writes contig per-head. So mm2 outputs to BNSD-internal contig scratch.
4. **AIV postprocess stage for output layout transform**: per (b, n_head), loop over `s`, **single-row contig DataCopy** GM→UB (no stride params: `blockCount=1, blockLen=D*sizeof(T)/32, srcStride=0, dstStride=0`) + explicit `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` then UB→GM at `b*sB + n_head*sN + s*sS`. No strided DataCopy params anywhere → sidesteps the V220 strided-DataCopy bug class entirely.

**Concrete parameter values** (BSH input, where `sB=S*N*D, sN=D, sS=N*D=H`):
```cpp
// mm1: scores [B,N,S,S] = Q @ K^T
mm.SetOrgShape(S,           // orgM
               sS,          // orgN = H (Q/K physical leading dim)
               sS,          // orgKa = H
               sS,          // orgKb = H
               S);          // orgKc — C is BNSD-internal contig, N=S
mm.SetTensorA(qGm[b*sB + n*sN], /*isTransposeA=*/false);
mm.SetTensorB(kGm[b*sB + n*sN], /*isTransposeB=*/true);
mm.SetSingleShape(S, S, D);
// → matmul loads Q row s, col k from qGm[head_off + s*H + k] (correct BSH access)

// mm2: out_bnsd [B,N,S,D] = attn @ V
mm.SetOrgShape(S, sS, S, sS, D);   // orgKa=S (attn contig), orgKb=H (V strided), orgKc=D
// → out_bnsd is BNSD-internal contig at bn_idx*S*D; user-layout transform deferred to postprocess

// Postprocess AIV (only fires for non-BNSD):
for (s = 0; s < S; ++s) {
    DataCopy(ub, src_bnsd[bn_idx*S*D + s*D], row_params);  // contig D fp16 read
    SetFlag<MTE2_MTE3>(eid); WaitFlag<MTE2_MTE3>(eid);
    DataCopy(dst_user[b*sB + n*sN + s*sS], ub, row_params); // contig D fp16 write at strided offset
    SetFlag<MTE3_MTE2>(eid); WaitFlag<MTE3_MTE2>(eid);
}
```

**Launch topology** (compared to PR #103's broken AIV canon path):
- **Before (broken canon)**: canon (BSH→BNSD scratch, broken) → mm1 → softmax → mm2 → uncanon (BNSD→BSH, also broken) = 5 launches, all 5 in source even though canon/uncanon produced wrong output
- **After (this CAND)**: mm1 → softmax → mm2 → postprocess = 4 launches for non-BNSD, 3 for BNSD; no strided-DataCopy bug surfaces

**Why "matmul C-side does NOT support stride" matters as a separate finding**: An obvious design temptation is to set `orgN=sS` for mm2's C-side and have matmul write BSH-layout directly. This compiles cleanly but produces 5–38% systematic error (verified at B=1,S=64,N=2,D=64,BSH→4.8%; BSND/SBH→38%). The matmul library on V220 always writes C contig per launch regardless of `orgN`/`orgKc`. So the output-layout transform MUST be a separate stage, and that stage MUST avoid strided DataCopy (which on V220 is the bug class PB-36 documents).

**Verification scoreboard (honest)**:
- Cube path (S>=16, S*D ≤ 8192, S*Skv ≤ 8192): **9/10 PASS at max_abs ≤ 6.1e-5**
- Cube path (S*Skv > 8192): **1 algorithmic-scope FAIL** at `BSH B=1 S=128 N=1 D=64` (Skv*S=16384 > UB budget). This is NOT a defect of CAND-FA-CANON-FREE — the kernel still uses kw-1's "materialize full S×Skv scores in UB" algorithm. Real FlashAttention algorithm rewrite (row-tiled with online softmax) tracked as DEBT-FA-row-tiled.
- Pass B (VEC fallback, S=2): **9/9 PASS**
- Perf at B=1,S=64,N=2,D=64,BSH,fp16: **0.603× CANN** (56.9 µs ours vs 34.3 µs CANN). 13× over kw-1 baseline (0.046×). Target ≥0.6× CANN met within the rewrite's correctness scope.

**Cross-ref**:
- PB-36 (ARCHIVED — what this CAND supersedes)
- `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-canon-removal-structural-rewrite` (full design + complete kernel patches + falsification chain + verification numbers)
- PB-9 (V220 UB→UB DataCopy nuance; same MTE-engine family)
- PB-22 (V220 MTE2 DataCopy 32B transfer limit per destination TBuf)
- PB-34 + PB-35 + CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (V220 mixed-mode KFC/sync minefield — the reason this rewrite uses **multi-launch** for the planned DEBT-FA-row-tiled outer loop rather than fused mixed AIC/AIV)
- DEBT-FA-row-tiled (follow-up: real flash-attention algorithm rewrite to expand correctness scope to arbitrary S/Skv)

**Promotion path**: candidate to canonical OL once (a) one more L4 fused-cube op independently uses the SetOrgShape 5-arg + postprocess-AIV pattern and verifies correctness, OR (b) V351/A5 cross-arch verification of the same FA shapes confirms portability, OR (c) the row-tiled algorithm rewrite ships and the combined pattern proves to be the canonical FA-class approach on V220. Until then, scope strictly to `op_class=fused-attention` and `correctness_scope=S*D,S*Skv ≤ UB_BUDGET_8192`.

### CAND-NO-CHEAT-AUDIT-CHECKLIST: Self-audit schema for AscendC op-gen agents — pre-DONE checklist to catch CPU compute / PyTorch delegation / kernel CPU-fallback cheating [V351+V220, ASCENDC_MODES, anti-cheat, agent-discipline]

`applies_to: soc=all; cann=all; bisheng=all; op_class=all; mode=arch22_to_arch35/backward; backend=ascendc`
`verified_on: independent 3_FusionAttention audit 2026-05-22 — owner asked "are we using CPU for some of the logics which will be considered as cheating?" mid-PR-#112 push. Audit identified zero cheating in the cube path but surfaced a per-row scalar loop on the AIV scalar pipe. The audit was ad-hoc; this CAND codifies the steps so future AscendC agents can self-audit before declaring DONE.`

**Why this CAND** (the recurring failure mode): CLAUDE.md has the "No PyTorch/CANN Delegation, No CPU Fallback" rule but it's a paragraph-level prose statement, not a grep-able checklist. Real audits get done **ad-hoc** when someone asks ("are we cheating?"); the answer is good but the audit isn't reproducible. Without a codified checklist, the next op-gen agent reading CLAUDE.md will know the *rule* but not the *test* — meaning subtle cheating (e.g., Python-side `permute()` workaround that survived 12 hours in PR #103, or `_check_scope` integer arithmetic that LOOKS like CPU compute but isn't) can ship undetected until owner asks.

**This is the operationalization companion to OL-175** (failure-framing discipline). OL-175 says "don't hide failures via framing"; this CAND says "don't hide cheating via lack-of-audit-procedure". Same family.

#### Checklist (run BEFORE declaring DONE on any op)

**Step 1 — Python `model_new_ascendc.py::forward` scan** (3 substeps):

```bash
# A. No CPU compute on tensor data
grep -nE "\.cpu\(\)|\.numpy\(\)|\.tolist\(\)|\.item\(\)" workspace/<op>/model_new_ascendc.py
# Expected: empty. If hits, every hit must be in a non-compute path (e.g., debug print
# behind `if DEBUG:` guard, not in the live forward). Live forward must operate on .npu()
# tensors only.

# B. No PyTorch compute-op delegation
grep -nE "torch\.(matmul|softmax|exp|log|sort|argsort|topk|max|sum|mean|var|std|cumsum|gather|scatter|index_select|permute|reshape|view|transpose|expand|repeat|tile|contiguous|cat|stack|chunk|split|fft|rfft|conv\w+|linear|layer_norm|batch_norm|sigmoid|tanh|relu|gelu|silu)" workspace/<op>/model_new_ascendc.py
grep -nE "F\.(softmax|attention|conv\w+|linear|layer_norm|batch_norm|sigmoid|tanh|relu|gelu|silu|scaled_dot_product_attention)" workspace/<op>/model_new_ascendc.py
# Expected: empty in forward(). If hits, every hit must be in a non-forward helper
# (e.g., __init__ shape pre-computation) AND not touching input tensors.

# C. forward() has at most ONE _ext.run_<op>(...) call and a direct return
python3 -c "
import ast
src = open('workspace/<op>/model_new_ascendc.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'forward':
        ext_calls = [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id == '_ext']
        print(f'_ext.*() calls in forward(): {len(ext_calls)}')
        torch_attr_calls = [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == 'torch']
        print(f'torch.<attr> in forward(): {len(torch_attr_calls)}')
"
# Expected: exactly 1 _ext.* call. torch.* allowed only for non-compute uses (torch.float16
# dtype literals); each hit needs manual inspect.
```

**Step 2 — Pybind11 scan** (4 substeps):

```bash
# A. No torch::<compute_fn> in pybind dispatch body
grep -nE "torch::(matmul|softmax|exp|log|sort|topk|max|sum|mean|cat|stack|cdist|matrix_exp)" workspace/<op>/kernel/pybind11.cpp
# Expected: empty. Operations on torch::Tensor data are forbidden in pybind; only
# torch::empty/options/sizes/dtype/contiguous-on-input are allowed.

# B. Only allowed torch:: surface usage
grep -nE "torch::|at::" workspace/<op>/kernel/pybind11.cpp | \
  grep -vE "(torch::Tensor|torch::empty|torch::kFloat|torch::kInt|torch::kHalf|torch::kBfloat|c10::optional|\.options\(\)|\.sizes?\(\)|\.dtype\(\)|\.size\(|\.device\(\)|\.contiguous\(\)|at::Half|TORCH_CHECK|TORCH_INTERNAL|c10_npu)"
# Expected: empty (or only whitelisted residual). Any surfaces outside the allowed list
# (torch::Tensor decl, torch::empty alloc, .options/.sizes/.size/.dtype/.contiguous/at::Half/
# TORCH_CHECK/c10_npu stream access) need manual inspect.

# C. No CPU-tensor traffic
grep -nE "\.cpu\(\)|\.to\(c?torch::kCPU\)|aclrtMemcpy.*HOST_TO_DEVICE|memcpy.*data_ptr\(\)" workspace/<op>/kernel/pybind11.cpp
# Expected: empty in the dispatch body. The only allowed host-side device operation
# here is aclrtMemset (GM workspace initialization); it transfers no tensor data.

# D. .contiguous() only on INPUT tensors (not on outputs from kernel)
# Output tensors written by AscendC kernel MUST NOT be passed through .contiguous()
# because the kernel writes layouts the caller expects. .contiguous() on output would
# round-trip via aclnnContiguous = CANN delegation. Manual inspect:
grep -nB2 -A1 "\.contiguous\(\)" workspace/<op>/kernel/pybind11.cpp
# Expected: every hit is on an INPUT (query/key/value/etc from forward args), NOT on
# scratch buffers, output tensors, or post-kernel tensor returns.
```

**Step 3 — Kernel host-launch scan** (1 substep):

```bash
# A. No CPU-side compute in kernel host code
grep -rnE "\.cpu\(\)|std::sort|std::sin|std::cos|std::exp|std::log|std::sqrt" workspace/<op>/kernel/*.cpp workspace/<op>/kernel/*.h
# Expected: empty in body of __global__ aclrtlaunch_* functions. Pure host-side helper
# code (tiling computation, blockDim selection) is allowed to use std::* but NOT for
# computing tensor values.
```

**Step 4 — AscendC scalar-pipe usage sanity** (info only, not anti-cheat):

```bash
# Per-element scalar loops on AIV (GetValue/SetValue inside a for-loop) are LEGAL but
# burn AIV scalar pipe. Not cheating, but tip-of-iceberg perf hint.
grep -nE "\.GetValue\(\w+\)|\.SetValue\(\w+, " workspace/<op>/kernel/fusion_attention_kernel.h | wc -l
# If count is high (>10 per kernel), flag for CAND-FA-MULTI-LAUNCH-PERF-GAP Δ#2-style
# "rewrite as vector op" optimization. Not a fail.
```

**Step 5 — Live-path probe** (only if Steps 1-4 surfaced any suspect line):

```python
# Run the actual op once on .npu() tensor; profile with msprof to confirm no aclnn*
# host-API calls (other than the expected workspace/HBM allocs).
python3 -c "
import torch, torch_npu
import sys
sys.path.insert(0, 'workspace/<op>')
sys.path.insert(0, 'workspace/<op>/kernel/build')
import model_new_ascendc as mna
q = torch.randn((1,64,128), dtype=torch.float16).npu()
k, v = q.clone(), q.clone()
torch.npu.synchronize()
# Wrap in msprof if suspect aclnn calls
out = mna.ModelNew()(q, k, v, 2, 'BSH', scale=1.0)
torch.npu.synchronize()
"
# Optional: msprof profile and grep for 'aclnnPermute', 'aclnnContiguous', 'aclnnSort',
# 'aclnnMatmul', etc. — any aclnn* in the per-step kernel list (other than the user's
# expected aclrtlaunch_<op_internal_*>) is a delegation.
```

#### Examples of cheating this checklist catches

**Live example caught earlier this session (PR #103, retracted via PR #106)**:
```python
# model_new_ascendc.py forward(), BAD:
q_kern = query.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()  # delegates to aclnnPermute
# Caught at Step 1B (torch.reshape pattern would have flagged) AND
# Step 5 msprof would have shown aclnnPermute in trace.
```

**Subtler delegation that LOOKS like metadata but isn't**:
```python
# BAD:
if input_layout == "BSH":
    q_bnsd = query.permute(0, 2, 1, 3).contiguous()  # contiguous() on POST-permute tensor = aclnnContiguous compute
# Step 1B catches `.permute(`; Step 2D catches `.contiguous()` on non-input tensor.
```

**Subtler still — scalar fallback inside kernel host code**:
```cpp
// kernel/op_host/<op>_tiling.cpp, BAD:
void compute_tiling(...) {
    std::sort(tiling.workgroup_priorities, tiling.workgroup_priorities + N);  // host-side std::sort on per-instance data
    // This burns host CPU cycles per kernel launch. Not a Cheating-on-results, but
    // host-side compute-per-launch that can amortize differently than expected.
}
// Step 3 catches std::sort.
```

#### Promotion path

Candidate to OL once:
1. At least 3 different ops (different op-classes: 1 elementwise / 1 reduction / 1 fused) have used this checklist successfully pre-DONE
2. The Steps 1-3 grep patterns are stable (no false negatives surfaced)
3. The live-path probe is empirically validated on at least one fused AscendC op and one backward AscendC op

Until then, scope: AscendC ops where author manually runs the checklist before declaring DONE. Verifier-side automation is **DEBT-NO-CHEAT-AUDIT-CI** (future hook to enforce this via pre-commit).

#### Cross-ref

- **OL-175** (defensive-guard refusal is highest fail tier — sibling agent-discipline anti-cheat)
- **OL-160** (canonical entry-point file names — sibling structural anti-cheat enforcement)
- **OL-167** (DataCopy `count` truncation pad+narrow cheat — sibling on-device anti-cheat)
- **OL-172** (ModelNew.forward output count parity — sibling contract-side anti-cheat)
- **CAND-FA-MULTI-LAUNCH-PERF-GAP** Δ#2 (per-row scalar loop on AIV scalar pipe is NOT cheating but IS perf headroom)
- **CLAUDE.md "No PyTorch/CANN Delegation"** + "No CPU fallback" — the rule this CAND operationalizes
- **PR #103 → PR #106** (worked example: cheating shipped via Python permute(), caught by owner, retracted)

### CAND-FA-MULTI-LAUNCH-PERF-GAP: Five design-choice deltas between multi-launch row-tiled FA and CANN's single-launch fused FA — measured 100× perf gap at S=1024 traced to specific kernel-internal pipelining choices [V220, L4 fused-attention, perf-optimization-roadmap]

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0; op_class=fused-attention or any L4 op_class requiring row-tiled outer + accumulator across tiles; perf_regime: large_S (S ≥ 4 × T_q i.e. ≥4 Q-tiles)`
`verified_on: a5_ops:3_FusionAttention Step 3 row-tiled multi-launch FA (PR #109, 2026-05-22, A3 npu-a3-test) — correctness 13/13 cube + 9/9 Pass B PASS at max_abs ≤ 1.5e-5; perf measured 0.69-0.81× CANN at small S (fast-path) vs 0.025× at S=512 vs 0.007× at S=1024. 100× perf gap at large S directly traceable to 5 design choices identified via CANN-source comparison.`
`unverified_on: V351 (Ascend950PR / A5) — same design-choice deltas likely apply but V351 KFC behavior may differ; needs cross-arch verification before broad scoping`
`v351_implication: probe_a5_v300_fa_sync 2026-05-23 — Pattern A clean on V351 means SINGLE-LAUNCH FUSED FA is viable on V351/Ascend950PR. The 5-delta multi-launch roadmap was V220-conservative; on V351 the single-launch fused architecture (CANN's Pattern C-equivalent for arch35) is the right architectural target, not multi-launch + 5 deltas. V220 ceiling at 0.014× CANN @ S=1024 does NOT propagate to V351. Re-scope this CAND title from "Five design-choice deltas..." to "V220-only roadmap; V351 should target single-launch fused" — pending sole-final-author retitle when broader scope confirmed.`

**Why this CAND** (not just a DEBT): The 5 deltas below are **structural** — they apply to any row-tiled multi-tile L4 fused op on V220 (MoE finalize, fused norm+matmul with cross-tile accumulator, GroupNorm + chunked reduce, fused-quant attention variants, etc.), not just FA. Future agent implementing row-tiled L4 should read this CAND before declaring "structural ceiling" on multi-launch perf — the 5 deltas are concrete optimization headroom, not assumed-immutable architecture.

**Source comparison**: CANN `attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h` (kernel) + `op_host/arch22/flash_attention_score_tiling_general.cpp` (host tiling) read 2026-05-22 via owner-authorized port_a3 mode (cann_learner exception per CLAUDE.md V3.x carve-out). KB-carveout rules respected: design patterns + parameter values + structural choices extracted; no verbatim source.

#### Delta #1 — L0C accumulator residency across KV-tile iterations

| | CANN (single-launch fused) | Multi-launch (our PR #109) |
|---|---|---|
| O[T_q, D] fp32 location | **L0C resident** across all KV-tile iters within a Q-tile | GM round-trip every KV-tile (load → scale → add → store) |
| Per-KV-tile O GM traffic | ~0 (only fixpipe-write once at Q-tile epilogue) | `T_q*D*4 = 16 KB` read + 16 KB write per KV-tile |
| Mechanism | `bmm2.template IterateAll<false>(...)` per KV iter (the `false` = no auto-flush) + `taskIdMod2` ping-pong | separate `fa_scale_and_accumulate_fp16` AIV launch with explicit GM r/w |
| L0C lifetime | mm2 object instance persists across KV iters; L0C carries state | mm2 launches discrete; no cross-launch state |

**Headroom**: For S=1024, `T_kv_tiles=16`, we do **16 GM round-trips of `T_q*D = 16 KB`** = ~512 KB extra HBM bandwidth per Q-tile per (b, n_head). At V220 HBM ~1.6 TB/s effective: ~0.32 µs overhead just for O traffic per Q-tile × 16 Q-tiles = ~5 µs. Plus launch-init overhead. Adopting L0C residency would save ~80% of accumulator memory traffic.

**Adopt feasibility**: requires keeping the mm2 `MatmulImpl` object alive across launches OR moving to single-launch fused. Within current multi-launch architecture: **NOT feasible** — each `aclrtlaunch_*` instantiates fresh `MatmulImpl`. This delta alone justifies eventual move to single-launch (delta #5).

#### Delta #2 — Online softmax fp32 buffer sizing

| | CANN | Ours (PR #109) |
|---|---|---|
| Live fp32 softmax buffers per AIV | `[s1_vec, s2_aligned]` = `[8, 64]` = **2 KB** per ping-pong buf × 4 bufs (max/sum/exp + scratch) ≈ **8 KB** | Full `[T_q, T_kv]` fp32 = `[64, 64] * 4` = **16 KB** for scoresF + 4 KB for reductions ≈ 20 KB |
| Computation granularity | Row-by-row, 8-way reduction factor (`softmaxReduceSize=8`) | Per-Q-tile full materialization |
| API | CANN's `SoftMaxCompute` called inside per-loopIdx loop with `[s1_vec, s2_aligned]` input | Our `Cast` + `Adds` + `Exp` over full ST = T_q*T_kv |

**Headroom**: ~12 KB UB freed per AIV. Reusable for larger tile sizes (delta #3) OR resident O accumulator (delta #1).

**Adopt feasibility**: **HIGH within multi-launch architecture**. Rewrite `FaOnlineSoftmaxUpdateKernel::ProcessTile` to process `s1_vec=8` rows at a time, looping `T_q / s1_vec = 8` times. Same scalar-loop structure already exists for per-row m_new computation; just shrink inner buf allocation. **Effort: 1-2h. Risk: low.**

#### Delta #3 — Tile size selection function

| | CANN (`CalcS1S2BasicBlock`) | Ours |
|---|---|---|
| T_q, T_kv source | **Host-side computed per-shape**: `tmpS1 ∈ [GetMinS1BasicBlock(), alignedS1]` step 16; for each, max `tmpS2 ≤ alignedS2` s.t. UB-budget fits | **Hard-coded** `T_q = T_kv = 64` in pybind |
| UB budget formula | `s1*16*X + s1*D*Y + s1*(expNum+2)*32 + apiTmp ≤ ubSize` (X, Y per-op family constants) | n/a — never computed |
| Typical result @ B=1, N=12, S=1024, D=64 | `s1BaseSize=64, s2BaseSize=64` balanced OR `128×64` / `64×128` depending on enableL1Reuse | always 64×64 |

**Headroom**: 64×64 likely undersized for D=64 small-Skv shapes (could be 128×64 → halves Q-tile-count → halves multi-launch overhead in the Q-tile dimension) and oversized for D=128 (could be 64×32 → fits with delta #2's freed UB). 2-4× tile area at peak UB utilization.

**Adopt feasibility**: **HIGH within multi-launch architecture**. Implement a tiny host-side `tile_sizing_v1(B, N, S, Skv, D, dtype_bytes) → (T_q, T_kv)` function in pybind11.cpp. Even a simple "fit-then-balance" heuristic (start at 128×128, shrink to fit UB while staying balanced) would beat hard-coded 64×64. **Effort: 1h. Risk: low.**

#### Delta #4 — Alpha rescale fusion into mm2 post-process

| | CANN | Ours |
|---|---|---|
| Alpha rescale (O = α * O_prev + dO_new) | **Fused into bmm2's vec2 post-process**: `DataCopy(bmm2ResUb, stage2BufTensor)` + `Bmm2ResultMul(bmm2ResUb, expUb, ...)` + `Add(bmm2ResUb, bmm2ResUb, stage2BufTensor)` in single AIV kernel | **Separate `fa_scale_and_accumulate_fp16` AIV launch** |
| Launches per KV-tile | mm1 + softmax_update + mm2 (+vec2 fused inside mm2) = 3 logical stages, but **1 launch** in fused KFC mode | mm1 + softmax_update + mm2 + scale_accumulate = **4 launches** |

**Headroom**: 25% launch count reduction per KV-tile. For S=1024 with 16×16 = 256 KV iters × 4 launches = 1024 launches → could drop to 768 = -33%. Direct ~25-33% perf gain at large S where launch overhead dominates.

**Adopt feasibility**: **MEDIUM within multi-launch architecture**. Two options:
- (a) merge `fa_scale_and_accumulate` body into the tail of `fa_online_softmax_update` (which already has alpha computed) — requires routing dO_partial GM ptr into softmax_update; saves 1 launch but pre-vs-post mm2 ordering needs care since softmax_update runs BEFORE mm2.
- (b) merge `fa_scale_and_accumulate` body into a new "fa_mm2_then_scale" AIV-side kernel that runs after mm2 — requires mm2 output and prior O / alpha all visible to one AIV.
- Both require shape-rewrite vs current cleanly-separated stages. **Effort: 2-3h. Risk: medium (correctness re-verification needed).**

#### Delta #5 — KFC-implicit AIC↔AIV sequencing (single-launch fused) vs multi-launch isolation

| | CANN | Ours |
|---|---|---|
| Kernel structure | **Single fused kernel** with `taskId % 3` ping-pong stages over mm1/vec1/mm2/vec2; AIC implicit via tiling BlockDim | Multi-launch: each stage is its own `aclrtlaunch_*` |
| Sync mechanism between stages | KFC implicit + intra-core `SetFlag/WaitFlag<MTE3_MTE2>` event sync | Stream-sequential dispatch (NPU stream serializes launches) |
| AIC↔AIV handoff | Implicit via task pipeline; no `CrossCoreSetFlag<0x2>` (Pattern A) and no explicit `REGIST_MATMUL_OBJ` (Pattern B) — instead **event-ordered task sequence within one kernel** | None — each launch is fully independent, AIC and AIV separated |
| Total launches @ S=1024 | ~B*N task-pipeline stages = ~100 for B=1, N=12 (one per (b, n) pair across all Q+KV iters) | ~1057 launches (16 Q-tiles × 16 KV-tiles × 4 + 16 inits + 16 finalizes + 1 postprocess) |

**Headroom**: 10× launch reduction → could close most of the launch-overhead-bound perf gap.

**Adopt feasibility**: ~~**HIGH RISK** within current state~~ ~~**EMPIRICALLY FALSIFIED 2026-05-22 — Pattern C structurally blocked on V220.**~~ **STATUS REVISED 2026-05-23 — V220 Pattern C UNVERIFIED (probe was misdesigned); V351 Pattern A confirmed VIABLE.**

The V220 KFC mixed-mode minefield (PB-34, PB-35) burned 5 iter on Pattern A. PR #117 originally claimed Pattern C also empirically falsified on V220 — that claim has been retracted (see [CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP](../../../../target/ascendc/patterns/unverified/candidates.md#cand-pa-v220-mix-aic-sync-infra-gap-v220-kernel_type_mix_aic_1_2-cube-internal-pipe-sync-deadlocks-regardless-of-event-id-scheme--root-cause-deeper-than-event-id-allocation) status update 2026-05-23). My V220 probe used `SetFlag<HardEvent::MTE3_MTE2>` which is intra-core pipe-sync, NOT cross-core AIC↔AIV handoff — the hang was from probe-design defect, not real V220 architectural block.

**Updated probe outcomes (2026-05-23)**:
- **V220 Pattern A**: deadlocks (PB-34, 5-iter chain). **CONFIRMED.**
- **V220 Pattern B**: unwound from production for unrelated reasons. Status unknown.
- **V220 Pattern C**: probe misdesigned, status genuinely UNVERIFIED. Re-probe needed with `CrossCoreSetFlag<0x2>(flagId)` semantics.
- **V351 Pattern A**: runs clean (0.036ms, bit-exact, 3 deterministic trials). See main agent's `workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md` and PB-34's `verified_does_not_reproduce_on: V351` line.

**Practical implication for V220**: multi-launch architecture remains the path of least resistance (Pattern A confirmed blocked). Real V220 ceiling at 0.014× CANN @ S=1024 stays correct, supported by Pattern A falsification alone — doesn't rely on the retracted Pattern C "falsification".

**Practical implication for V351**: **single-launch fused FA is viable**. Use Pattern A architecture (`MatmulImpl<>` + `CrossCoreSetFlag<0x2>` + `KERNEL_TYPE_MIX_AIC_1_2`) for V351 / Ascend950PR FA ports. CANN's `flash_attention_score/arch32/s1s2_bn2gs1.h` is the pattern reference.

#### Combined optimization roadmap (UPDATED 2026-05-22 with measured outcomes)

Starting from PR #109's 0.007× CANN at S=1024:

| Step | Predicted | Measured | Delta |
|---|---|---|---|
| +#2 (row-wise softmax) + #3 (tile sizing) — PR #112 | ~0.02× | **0.014×** | below estimate (~70% of prediction) |
| +#4 (alpha-rescale fusion via reorder) — PR #114 | +1.5× per shape | **+1.17-1.30×** S=64..512; **~0% S=1024** | below; S=1024 plateau is dO_part GM round-trip-bound (CAND #1 territory) |
| +#1 (L0C residency) — needs #5 first | 80% mem traffic save | **N/A (blocked by #5 falsification)** | NOT STANDALONE |
| +#5 (single-launch event-ordered) — PR #117 falsification | ~10× | **0× (Class B falsified)** | empirical block |

**Actual measured ceiling for V220 multi-launch FA (PR #114)**: **0.014× CANN @ S=1024, 0.6-0.8× CANN @ S=64**. This is the **real V220 ceiling**, not "0.4-0.6× CANN with all 5 deltas adopted" as initially projected — that projection was made on the assumption Δ#5 was adoptable. Per OL-175 honest framing: predicted-vs-measured roadmap calibration is itself KB-valuable; future agents starting from this CAND get the empirical ceiling not the speculative one.

#### Cross-ref

- **CAND-FA-CANON-FREE** (PR #106 ancestor — solved canon stage and matmul stride mechanism; this CAND is the **next-tier** structural problem)
- **PB-34** (Matmul + manual CrossCoreSetFlag + MIX_AIC_1_2 deadlock — Pattern A falsified at V220)
- **PB-35** (event_t(0..3) collides with FLAG_CANON_DONE chain — Pattern A/B refinement falsified)
- **CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP** (open-hypothesis follow-up to PB-35; delta #5's event-ordered single-kernel task pipeline might be the resolution)
- **OL-175** (defensive-guard-refusal-is-highest-tier — applicable here when claiming perf "ceiling" before exhausting structural deltas above)
- `docs/design/FA_CLASS_DESIGN_NOTES.md#fa-step3-q-tiled` (the impl this CAND analyzes)
- CANN source `flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h:499-2244` + `op_host/arch22/flash_attention_score_tiling_general.cpp:1815-1914`

**Promotion path**: candidate to OL once at least 2 of the 5 deltas are empirically adopted in our codebase with measured perf delta confirmation. Until then, scope strictly to `op_class=fused-attention` and `perf_regime: large_S` — small-S fast-path doesn't surface most of these deltas.

### CAND-FUSED-KERNEL-PERF-ITERATION-WORKFLOW: 5-phase methodology for iteratively closing the perf gap on L4 fused kernels (from FA-2026-05-22 worked example) [V351+V220, ALL_MODES, agent-discipline + L4-methodology]

`applies_to: soc=all; cann=all; op_class=L4 fused (FA-class / MoE finalize / GroupNorm+activation+quant / fused-conv+norm / any multi-stage cube+vec op needing row-tiling or cross-iter accumulation); mode=all_modes`
`verified_on: 3_FusionAttention 2026-05-22 — full 8-PR iteration session (PR #103 retracted → #106 → #109 → #110 → #112 → #113 → #114 → in-flight #115) drove FA from 0.046× CANN baseline to 0.6× CANN (S≤256) + arbitrary S/Skv correctness. Each phase below corresponds to one or more concrete PRs in that session.`
`unverified_on: other L4 fused op classes — workflow not yet exercised on MoE finalize / fused norm / etc., but the per-phase artifact shape is op-class-agnostic`

#### Why this CAND

Owner direction: "we can continue this iteration to keep improving kb for FA like CV fused kernel gen". The FA 2026-05-22 session went through 5 discrete iteration phases, each producing a measurable correctness or perf delta. Without codification, the next agent attacking a different L4 fused op (e.g., MoE finalize, fused-quant-attention variants, GroupNorm + SiLU + Quant fused) will:
- Re-invent the CANN-source-read step
- Re-discover the 5-delta perf-comparison schema
- Re-derive that low-risk deltas should adopt first
- Re-experience the "0-output FAIL is highest tier, not skip" lesson (OL-175)

This CAND captures the **workflow shape** — what artifacts to produce per phase, what to query CANN source for, how to schedule deltas by risk, when to escalate to single-launch — so future iterations can compress from N sessions to ~1.

#### Phase 1 — Cold-build correctness

**Goal**: produce ANY working algorithm that compiles + runs + matches CANN reference on at least one shape. Forget perf entirely.

**Allowed shortcuts during this phase ONLY**: VEC fallback paths, simpler-than-optimal algorithms (e.g., materialize full S×S scores in UB even if it caps S=128). Pass B / VEC fallback is fine.

**Anti-patterns to avoid**:
- Python-side `torch.permute()` / `.reshape().contiguous()` workarounds delegating to CANN — caught by CAND-NO-CHEAT-AUDIT-CHECKLIST (PR #103 cheating retracted via PR #106)
- "Different limit, separate DEBT" framing on shapes the kernel can't handle — caught by OL-175 (these are 0-output FAILs)

**Output artifact**: a working kernel + design doc + verification scoreboard showing PASS on at least 1 shape per layout.

**Worked example anchor**: FA PR #103-era (pre-retraction) shipped working VEC fallback + cube path at 0.046× CANN on S=64. Correctness was real even though perf was bad.

#### Phase 2 — CANN-source read for structural pattern extraction

**Goal**: identify what CANN does structurally for this op-class. NOT a verbatim port — pattern extraction only.

**Steps**:
1. Dispatch CANN-source agent (port_a3 mode, owner-authorized via CLAUDE.md §V3.x KB carve-out). Question template: "How does CANN's `<op>` implementation handle [X] on V220?" where [X] is the *first* unknown blocking your impl (e.g., "BSH→BNSD layout transform", "cross-tile accumulator", "softmax intermediate dtype").
2. **Narrow, targeted questions** — one specific unknown per dispatch. Broad "explain how CANN does FA" wastes the carve-out budget.
3. Extract: code-location ref + 1-2 sentence summary of pattern + concrete parameter values. No verbatim source copy.
4. If first dispatch surfaces N-1 follow-up questions, dispatch another agent for the next 1-2 unknowns. **Iterate dispatches narrow, not one big agent.**

**KB-carveout discipline** (per CLAUDE.md §V3.x): patterns + parameter values only. No verbatim source. The dispatched agent operates under that constraint; verify by checking returned content doesn't contain large code blocks.

**Output artifact**: 1-N "CAND-<op>-<aspect>" entries each with "CANN's approach / ours / measured gap / open-question" structure.

**Worked example anchor**: PR #106 used CANN-source agent to find the `SetOrgShape` 5-arg variant for layout-aware strided GM loads. PR #109 used a second agent to find the s1/s2 double-tiling nesting. PR #110 used a third agent for the 5-delta perf comparison (L0C residency / fp32 buf / tile sel / fused-stage / single-launch).

#### Phase 3 — 5-delta perf-comparison schema authoring (CAND-<OP>-MULTI-LAUNCH-PERF-GAP)

**Goal**: after Phase 2's CANN reads surface 3-5 structural design choices, codify them as a single CAND entry with concrete per-delta breakdown.

**Schema per delta** (worked example: CAND-FA-MULTI-LAUNCH-PERF-GAP §1-5):
- **CANN's approach** (file:line ref, pattern name, parameter values)
- **Our approach** (current PR's approach, code-ref)
- **Measured gap** (perf delta or behavioral difference)
- **Adopt feasibility** (LOW / MEDIUM / HIGH risk; effort estimate in hours)
- **Headroom** (estimated perf gain if adopted)

**Recurring 5 deltas for L4 fused-cube-vec ops on V220** (extracted from CAND-FA-MULTI-LAUNCH-PERF-GAP, likely apply to other L4 op-classes):
1. **L0C accumulator residency across tile iterations** — does CANN keep accumulators in L0C across inner-loop iters? Our default multi-launch GM-round-trip is the perf floor.
2. **Working-buffer fp32 sizing** — does CANN use row-wise chunking (`S1_VEC`-style) vs full-tile fp32 buffers?
3. **Host-side tile-size selection** — does CANN have `CalcXXBasicBlock`-style UB-budget formula vs hard-coded?
4. **Fused stage opportunities** — can a separate AIV stage (alpha-rescale / accumulate / final-norm) be folded into another stage's pipeline?
5. **Single-launch fused vs multi-launch** — single-kernel KFC-implicit task pipeline (Pattern C variant) vs our deterministic multi-launch?

**Risk-ordered adoption sequence**: #2 + #3 first (LOW risk, structural-only, ~3h combined) → #4 (MEDIUM risk, reorder/fusion, ~3h) → #1 + #5 together (HIGH risk, requires single-launch + Pattern C falsification probe, ~8h).

**Output artifact**: one CAND-<OP>-MULTI-LAUNCH-PERF-GAP entry in `patterns/unverified/candidates.md` with all 5 deltas filled in.

**Worked example anchor**: CAND-FA-MULTI-LAUNCH-PERF-GAP (PR #110) is the canonical instance for FA.

#### Phase 4 — Risk-ordered delta adoption with measured calibration

**Goal**: implement each delta as a separate PR with measured perf delta. Update CAND with predicted-vs-measured calibration after each PR.

**Per-delta PR template**:
1. Implement the delta in a focused branch
2. Run full correctness sweep (regression check across all previously-PASSing shapes)
3. Measure perf vs CANN on the same shape grid
4. Compute "actual_gain / predicted_gain" — calibration data
5. Update CAND-<OP>-MULTI-LAUNCH-PERF-GAP's §N with "MEASURED: predicted X, got Y. Reason for delta:..." (per OL-175 "failure knowledge is KB-valuable")

**Critical discipline**: if a delta MISSES its predicted gain, codify WHY — that's where the real KB value compounds. PR #114's S=1024 plateau (Δ#4 didn't move S=1024 despite +25% on smaller S) is the canonical "calibration finding" — surfaced that dO_part GM round-trip becomes its own bottleneck at large S, confirming Δ#1 (L0C residency) as the larger lever.

**Output artifacts**: 1 PR per delta + CAND update with calibration column.

**Worked example anchor**: PR #112 (Δ#2+#3), PR #114 (Δ#4) for FA. Each PR's design doc has predicted-vs-measured table.

#### Phase 5 — High-risk delta or escape-hatch (single-launch / Pattern C)

**Goal**: once low/medium-risk deltas exhausted, decide whether to attempt the high-risk delta (typically single-launch fused or major architectural change).

**Trigger criteria**:
- All LOW/MEDIUM-risk deltas adopted and CAND calibrated
- Remaining perf gap to CANN is dominated by the architectural delta (typical at large S — confirmed via per-delta calibration)
- Owner / project willing to spend HIGH-risk budget

**Procedure**:
1. **Falsification probe first**: write a minimal standalone kernel that exercises the high-risk pattern (e.g., Pattern C event-ordered single-launch task pipeline on V220). DO NOT touch the production kernel yet.
2. If probe falsifies → record empirical falsification in CAND, mark delta as **structurally blocked** for this arch + cite specific failure mode. **Failure knowledge is KB-valuable per OL-175.**
3. If probe passes → implement in production kernel, measure, update CAND.

**Anti-pattern**: jumping directly to architectural rewrite without falsification probe. Cost can be 5+ iterations of debugging V220-specific deadlock before realizing the pattern doesn't work. PR #103-era FA had 5 iter Pattern A/B falsification chain — preventable if we'd done isolated probes first.

**Output artifact**: either probe-pass + production PR, OR probe-fail + updated CAND with empirical block.

**Worked example anchor**: FA Δ#5 Pattern C falsification probe (planned PR #115 at time of this CAND authoring; pending).

#### Compound output: workflow → next-iter speedup

Iteration N benefits from prior iterations on different ops:
- CAND-<OP>-MULTI-LAUNCH-PERF-GAP from prior op → schema template for new op's CAND (same 5-delta structure)
- Pattern C probe result from prior op → known whether V220 single-launch is feasible → skip probe if already falsified
- CAND-NO-CHEAT-AUDIT-CHECKLIST → pre-DONE audit for new op
- OL-175 framing discipline → honest scoreboards prevent reward-hacking in new op's PRs

After this CAND lands + 2-3 different op-class iterations validate the schema, promotion to OL-N "L4 fused-kernel iteration methodology" is appropriate.

#### Cross-ref

- **CAND-FA-MULTI-LAUNCH-PERF-GAP** (worked example: the 5-delta schema for FA specifically)
- **CAND-NO-CHEAT-AUDIT-CHECKLIST** (Phase 1 anti-cheat enforcement)
- **OL-175** (failure-framing discipline — applies throughout, especially Phase 4 calibration)
- **PR #103 → #106** retraction (worked example: Phase 1 cheating caught, retracted, learned)
- **PR #110 / #112 / #114** (worked examples: Phase 3 / 4 / 4-with-calibration)
- **CLAUDE.md V3.x CANN-learn carve-out** (Phase 2 source-read authorization)

**Promotion path**: candidate to OL once 2+ different L4 op-classes use this workflow successfully (each producing their own CAND-<OP>-MULTI-LAUNCH-PERF-GAP). FA is the first; need at least one more (target: MoE finalize OR fused norm + activation + quant) to validate generalizability.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-OPAQUE-STRUCT-RUNTIME-VERIFY，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
