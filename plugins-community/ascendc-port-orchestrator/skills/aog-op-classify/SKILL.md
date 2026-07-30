---
name: aog-op-classify
description: >
  Classify an op by reading its source (Python / PyTorch / AscendC) and
  emit `op_classification.json` with KB recommendations as the load-bearing
  output. Invoked by the Python orchestrator at Phase O1.7, or by a human for
  one-off inspection: `Skill(name="aog-op-classify", args="workspace/{op}")`.
  Use when Phase O1.7 needs operator classification and KB recommendations.
---

# /aog-op-classify (v2)

Classify an op by reading its source (Python / PyTorch / AscendC), and emit `op_classification.json` with **KB recommendations** as the load-bearing output. Tags are descriptive labels for human readability; downstream brief construction reads `kb_recommendations` to extend the manifest.

This skill replaces the hardcoded `op_taxonomy.OP_TAGS` map (P0aaj retrospective: name-keyed dict + regex source-scan are both Python-side hacks that do not generalize to new operators). LLM-driven classification reads source, applies op-class knowledge, and recommends KB by understanding — works for any in-scope operator.

## When invoked

- By Python orchestrator at Phase O1.7 (after O1.5 DET_POLICY classification, before O2.5 truth provisioning) via subprocess `claude --print --skill aog-op-classify <workspace_path>`.
- By human directly: `Skill(name="aog-op-classify", args="workspace/<op>")` for one-off inspection.

**Cache**: if `workspace/<op>/op_classification.json` exists AND its `source_sha256` field matches the SHA256 of concatenated source content (recomputed each run), skip and exit. mtime-based caching is unreliable across git restores; content hash is canonical.

## Inputs

- **workspace_path** (required): path to `workspace/<op>/` directory containing source files.

## Output schema (v3)

`workspace/<op>/op_classification.json`:

```json
{
  "op": "<op_name>",
  "schema_version": 3,
  "source_sha256": "<sha256 hex of concatenated source>",
  "op_class_tags": ["..."],
  "kb_recommendations": [
    {"path": "...", "reason": "..."}
  ],
  "algorithm_classification": "single_op | fused | other",
  "layered_implementation_plan": {
    "applicable": false,
    "rationale_when_inapplicable": "...",
    "layers": []
  },
  "rationale": "...",
  "source_signatures_observed": ["..."]
}
```

`op_class_tags` is descriptive metadata. `kb_recommendations` is the load-bearing
output for KB manifest construction. **`algorithm_classification` + `layered_implementation_plan`** are the load-bearing outputs for **Tier 3 (P0aau-c35.e, 2026-05-09)**
state machine routing — when `applicable: true`, the orchestrator routes worker
spawns through `await_layer_worker` instead of `await_worker`, building the kernel
one layer at a time per the plan.

### Schema v3 — algorithm_classification + layered_implementation_plan

**`algorithm_classification`** — one of:
- `single_op`: the op is a single algorithmic primitive or trivially-fused with ≤2 sub-steps where layered build adds no diagnostic value (e.g. 1_GELU, 3_Add, 18_FusedAddRmsnorm — 2-step fused but each step is single-line).
- `fused`: the op combines ≥3 distinct algorithmic sub-ops (matmul-chain / attention / MoE / fused-norm-with-quant / multi-stage-backward) where bug-localization at Phase D failure has high cost without per-layer evidence.
- `other`: ambiguous — emit `applicable: false` with a `rationale_when_inapplicable` explaining why layered build won't work for this op (e.g. RNG-driven, non-decomposable, scope-blocked).

**`layered_implementation_plan.applicable`**:
- `true` ONLY when `algorithm_classification == "fused"` AND each sub-op has a clear CPU-truth reference decomposition AND multi-output API contract (if any) is establishable from Layer 1.
- `false` otherwise. When false, `layers: []` and `rationale_when_inapplicable` MUST explain the gap (Tier 3 reviews this field for cousin-op coverage gaps).

**`layered_implementation_plan.layers`** — ordered list, each entry:
```json
{
  "layer": <1-based int>,
  "name": "<short descriptive name e.g. qk_matmul>",
  "sub_op": "<primitive class — matmul / softmax / reduce / scatter / quant / dequant / ...>",
  "inputs": ["<input tensor names from model.py forward signature>"],
  "outputs_added": ["<tensors this layer first produces>"],
  "outputs_placeholder": ["<final-output tensors initialized at Layer 1 with zero-fill, will be filled in later layers>"],
  "outputs_filled": ["<previously-placeholder tensors this layer fills with real values>"],
  "reference_decomposition": "<one-line Python expression callable as ref_layer_N(*inputs) → outputs_added; uses torch ops only>",
  "verify_against": "isolated_layer_ref | full_fixture",
  "optional": false
}
```

**Rules**:
- Layer 1's `outputs_placeholder` MUST list ALL final-output tensors that won't be fully computed until later layers — multi-output API contract established from Layer 1.
- Each subsequent layer's `outputs_filled` ⊆ Layer 1's `outputs_placeholder`. By the final layer, all placeholders must be filled.
- `reference_decomposition` is callable Python — the helper script `precision_eval_layer_ref.py` (Stage 2) builds the layer-N reference from this expression.
- `optional: true` indicates a feature-wiring layer (scale / mask / dropout / GQA / pse) that runs only when the fixture has cases exercising the feature; verify against `full_fixture` instead of an isolated synthetic ref.

**For `single_op`**: emit `algorithm_classification: "single_op"` + `layered_implementation_plan: {"applicable": false, "rationale_when_inapplicable": "single primitive, no fused decomposition", "layers": []}`. Brief construction routes through standard `await_worker`.

**For `other`**: same shape; rationale describes why (non-decomposable / RNG / scope-blocked / etc).

## Procedure

### Step 1 — Read source files

Glob and Read these (in priority order):
1. `<workspace>/kernel/*.h` — AscendC kernel header (highest signal: explicit `AscendC::*` / `Tanh(` / `Mul(` primitive calls + comments)
2. `<workspace>/kernel/*.cpp` — AscendC kernel main (entry + dispatch)
3. `<workspace>/model.py` — PyTorch reference (signature + call graph)
4. `<workspace>/model_new_ascendc.py` — AscendC ref if Python-driven path
5. `<workspace>/analysis.md` — prior worker's analysis (secondary, may bias — read last)

#### Migration / backward workspace layout (P0aak v3)

The two supported workflows may stage source outside a generated `kernel/` directory:

- **No `kernel/` subdir** — kernel files (`*_kernel.h`, `*_kernels.cpp`, `*_simd.h` etc.) sit flat in the workspace dir. Treat the workspace dir itself as the kernel dir; glob `<workspace>/*.h` and `<workspace>/*.cpp`.
- **No `model.py`** — read the staged source-architecture AscendC code and its operator-host metadata. If the caller provides `<workspace>/REFERENCE_PATH`, restrict it to the source operator tree.
- **No `analysis.md`** is fine; skip silently.
- **Project root inference**: walk up `<workspace>/../` until a dir with `docs/`, `src/`, or `tests/` siblings is found; that's the project root for sibling-tree references.

If the workspace fits neither migration nor backward-generation shape (for example, a truly bare directory), classify from whatever source files exist and emit a `layout_note` in the rationale field describing what was found and what was missing.

**File size sampling for fused / large kernels**: if any single file exceeds 50 KB, read `head 30 KB + grep results for primitive-call regex (AscendC::|Tanh\(|Sigmoid\(|Erf\(|Reduce|Cast<|DataCopy|cos\b|sin\b|softmax|atomic|cooperative_groups|__shfl) + tail 10 KB`. Cap total concatenated text at 200 KB.

#### Multi-implementation workspaces (P0aak v3)

Workspaces sometimes contain multiple kernel implementation strategies for the SAME op (e.g. sparse_gather has SIMT-scalar + SIMT-vec4 + SIMT-persistent + SIMD-pingpong + SIMD-expert-major + SIMD/SIMT-hybrid in one workspace). Treat them as ONE op for classification — the op-class character is union-of-all-implementations, not per-file. The `source_signatures_observed` field can cite per-file evidence; tags are aggregated.

Compute SHA256 of concatenated text → store as `source_sha256` in output (cache key).

### Step 2 — Identify op-class tags

Apply op-class knowledge to source content. Tags are descriptive — multiple may apply to the same op (e.g. `softmax` op IS `transcendental + reduction + softmax`; layer them, don't pick one).

#### Compute-structure tags

| Tag | Apply when source contains | KB sections to recommend |
|---|---|---|
| `transcendental` | `AscendC::(Tanh\|Sigmoid\|Erf\|Exp\|Log\|Sin\|Cos)` calls, OR `Tanh(`/`Sigmoid(`/`Erf(`/`Exp(`/`Log(` calls in .h/.cpp, OR `torch.(tanh\|sigmoid\|erf\|exp\|log\|sin\|cos)` in .py, OR `torch_npu.npu_(gelu\|swiglu\|silu\|softmax)` | OL-103, PB-24, PB-27, P-P88 |
| `elementwise` | Pure pointwise op, single-pass, no reduction/scatter/gather | (default-only) |
| `broadcast` | `expand`/`reshape`/`permute`/`transpose` on shape mismatch, `cat`/`split`/`pad` | patterns/domains/memory_access.md |
| `reduction` | `ReduceSum/ReduceMax/ReduceMin/ReduceMean`, `torch.(sum\|mean\|max\|min\|prod\|cumsum\|cumprod)` | OL-110, OL-114, patterns/domains/reduction_quant.md |
| `softmax` | `AscendC::Softmax`, `torch.softmax`, `nn.functional.softmax`, stable-max + exp + normalize chain | OL-110, patterns/domains/reduction_quant.md |
| `sort-select` | `Sort/TopK/GatherMask`, `torch.(sort\|topk\|argsort\|argmax\|argmin)` | EC-33, OL-83, patterns/unverified/candidates.md#CAND-PP79 (bf16 mantissa-collision tie cluster) and #CAND-PP80, patterns/domains/sort.md (P-P107 tiled softmax over V>UB; **P-P108 iterative nucleus selection = ANTI-PATTERN O(V²)**; **P-P109 hardware Sort+cumsum nucleus = the correct O(V log²V) replacement**) |
| `scatter-gather` | `Gather/Scatter/AtomicAdd`, `torch.(gather\|scatter\|index_select\|index_put)` AS DOMINANT op character (NOT plumbing — see anti-pattern §below) | OL-67, OL-90, patterns/PATTERN_INDEX.md (P-P67 atomic-add) |
| `fft` | `AscendC::FFT`, `torch.fft.*` | OL-83, OL-103 |
| `normalization` | `AscendC::(Normalize\|LayerNorm\|RmsNorm\|GroupNorm)`, `torch_npu.npu_(rms_norm\|layer_norm\|group_norm)`, `nn.functional.layer_norm`, mean-of-squares + rsqrt + scale chain | OL-110, OL-111, OL-120, candidates#CAND-PP82 |
| `quantization` | int8/int4/fp8 path: `round + clamp(-128, 127)`, `npu_quantize`, `dynamic_quant`, `antiquant_scale` | patterns/domains/reduction_quant.md, OL-118 |
| `matmul` | matrix-multiply compute: `AscendC::Mmad` / `MatmulImpl` / `REGIST_MATMUL_OBJ` / BMM, OR `torch.(matmul\|bmm\|einsum)` where the einsum subscripts express a contraction (e.g. `"bij,bjk->bik"`), OR the `@` operator on 2D+ tensors. Apply whenever a Cube/MAC contraction is part of the op's compute character — make this RELIABLE (it is the discriminator for `attention`), not sporadic. | patterns/PATTERN_INDEX.md (P-P102 cube_vector_fusion), matmul KB |

#### Op-family tags

| Tag | Apply when | KB |
|---|---|---|
| `fused` | Source combines **≥3 distinct compute-structure tags** above, OR explicit fused-op API (`npu_fused_*`, multi-primitive kernel chain) | patterns/PATTERN_INDEX.md (P-P51 / P-P66 / P-P86 / P-P87). **If the fused op is cube+vector (has `matmul`/`MatmulImpl` AND a vector epilogue → MIX_AIC_1_2 candidate)**, recommend **by TARGET SoC — each card is scoped by its own `applies_to: soc=`, do NOT hand a card to a SoC it excludes** (DEBT-208): **V220 (a3/a2)** → PB-34 (`soc=Ascend910_9382` — MatmulImpl+manual-CrossCore MIX_AIC_1_2 FFTS-slot deadlock: 507014/507015 silent hang; root cause + Pattern-A **XOR** Pattern-B fix), OL-275 (`soc=Ascend910_V220` — managed-cube KFC lifecycle; its own `unverified_on` says do NOT assume transfer to A5). **V351/A5** → **NOT PB-34** — it declares `soc=Ascend910_9382` and carries two `verified_does_not_reproduce_on: Ascend950PR` witnesses (GDN full-op light-port, 122/122 T1 PASS), and its Consequence line makes that light-port the **DEFAULT A5 route**; injecting it on A5 inverts the card's own advice. Recommend instead OL-220 (`soc=Ascend950PR` — cube+vec MIX `ascendc_library` build recipe = the light-port), EC-68 (`soc=Ascend950PR` — 507015 SetSysWorkspaceForce on ACLRT_LAUNCH MIX), PB-45 + OL-223 (arch35 TPipe::Reset frees the global event pool → use a library cube). **BOTH SoCs** → PB-35 (`applies_to: soc=Ascend910_9382,Ascend950PR_9579`, `op_class=mixed_aic_aiv_pattern_a_tile_mmad` — attacks Pattern A itself; `confirmed_on` V351/A5, so it is the mode that bites on A5) |
| `attention` | the op exhibits the **attention compute structure**: a score matmul (Q·Kᵀ, often scaled by `1/√d`) → **row-wise softmax** → a second matmul (·V). i.e. `matmul` AND `softmax` **co-occur in the QK^T·V dataflow** (the softmax sits BETWEEN two matmuls, normalizing the score matrix that feeds the value matmul). Covers flash-attention, fused-attention, GQA/MQA, MLA, paged/KV-cache attention, fused-attention-grad. **NOT** "merely has a softmax". See structural-detection guidance below. | FA-class KB (`target/ascendc/fa_class/`), patterns/PATTERN_INDEX.md (P-P102). **MIX cube+vec deadlock cards (MANDATORY for FA-class — the cube↔vec sync is where FA-class ops hang on V220)**: PB-34 (MatmulImpl + manual CrossCore + MIX_AIC_1_2 = FFTS sync-slot deadlock, 507014/507015 silent hang at torch.npu.synchronize(); root cause + Pattern A/B fix), OL-275 (managed-cube KFC lifecycle), OL-220 (cube+vec MIX ascendc_library build recipe), EC-68 (507015 SetSysWorkspaceForce on ACLRT_LAUNCH MIX). V220-only; BENIGN on V351/arch35 |
| `stateful-cache` | KV cache / paged attention / RoPE position state, indexed cache write-by-slot | OL-88 |
| `loss-bwd` | Backward / gradient computation (op name has `Backward` / `_bwd`, computes `grad_*`) | OL-89, OL-110 |
| `optimizer-update` | Adam / AdamW / SGD update step | OL-68, OL-118 |
| `sampling` | nucleus / top-P / top-K-top-P / categorical sampling: source contains `top_p`/`topp`/`top_k_top_p`/`nucleus`/`sampled_softmax`/`torch.multinomial`, OR a softmax→sort/cumsum→select nucleus dataflow, OR op name matches `*(top_p\|topk_topp\|nucleus\|multinomial\|sample)*`. **Defining risk**: the nucleus count is unbounded (≈ p·V), so naive iterative max-selection is O(V²) — the exact P-P108 anti-pattern trap. | patterns/domains/sort.md (**P-P109** hardware Sort+cumsum nucleus, O(V log²V) — USE THIS; **P-P108** iterative O(V²) selection — ANTI-PATTERN, do NOT replicate its peaked-distribution dodge) |

#### `matmul` + `attention` structural detection (task#31 — the FA-vs-Sinkhorn discriminator)

The `attention` tag is **load-bearing for routing** (`plugins/base.py::is_fa_class` narrows to `ATTENTION`; it selects the FA-class standard AscendC template-assembly brief, threshold, and finalize gates). It MUST be applied with structural precision so a vector-softmax op is not mis-routed as attention.

**Apply `matmul` first, reliably**, whenever a Cube/MAC contraction is part of the op's compute character (see the compute-structure table). It is the precondition for `attention`.

**Apply `attention` ONLY when reading the dataflow shows the QK^T·V structure** — `matmul` and `softmax` co-occurring such that:
1. a **score matmul** produces an attention-score matrix (Q·Kᵀ, typically scaled), AND
2. a **row-wise softmax** normalizes that score matrix, AND
3. a **second matmul** consumes the softmax output against V.

The discriminator is the **co-occurrence of matmul + softmax in the QK^T·V structure**, not the presence of either alone. A source-reading LLM classifier can see this in `model.py.forward` (e.g. `attn = (q @ k.transpose(-2,-1)) * scale; attn = attn.softmax(-1); out = attn @ v`) or in the AscendC kernel's matmul→softmax→matmul call sequence.

**MUST EXCLUDE (negative cases — the whole point):**
- `hc_split_sinkhorn` — Sinkhorn row/col normalization is softmax-like but has **no matmul / no Cube** → tag `fused` + `softmax` + `normalization`, **NOT** `attention`. (The §6 / task#28 false-positive.)
- `layer_norm` / `rmsnorm` / `group_norm` — reduction + normalize, no matmul, no softmax → `normalization`, NOT `attention`.
- a bare `matmul` / `mat_mul_v3` / GEMM — matmul but **no softmax straddling two matmuls** → `matmul`, NOT `attention`.
- standalone `softmax` — softmax but no matmul → `softmax`, NOT `attention`.

When the QK^T·V structure is present, emit BOTH `matmul` AND `attention` (and usually `fused` + `softmax` + `transcendental` + `reduction` as the underlying compute structure). When in doubt — i.e. a softmax is present but you cannot confirm it sits between two matmuls in a score→normalize→value dataflow — **do NOT emit `attention`** (under-tagging fails safe to the normal kw path; over-tagging mis-routes a non-FA op into the heavy IL chain).

#### Reference-modeling tags

| Tag | Apply when | KB |
|---|---|---|
| `reference-ub` | Reference is upper-bounded — kernel CANNOT bit-match the reference because reference output is one of many valid implementations. **Positive examples**: any libm-vs-NPU comparison (gelu/silu/softmax/tanh — all transcendental ops), sort tiebreak, argmax tiebreak, nonzero ordering, FMA grouping order. **Negative examples**: integer ops, exact-arithmetic add/mul, deterministic gather, stable-sort with unique keys. | OL-85, OL-104 |
| `path-a-cpu-truth` | Reference is computable on CPU+fp64 only (Path A pattern; torch_npu.* is deprecated for this op) | OL-68, OL-89, OL-118 |
| `t3-required` | Reference structurally undefined on CPU (CANN-on-NPU only) | OL-118 |

**Reference-ub default for transcendentals**: any op tagged `transcendental` should ALSO be tagged `reference-ub` unless the kernel is bit-canonical to its reference (rare).

#### Source-language tags (DROPPED in v2)

Source-language tags were informational and unlocked no distinct KB sections, so they were removed in v2. Source language is captured in `source_signatures_observed` and `rationale`.

### Step 3 — Layer multiple tags

Tags are NOT mutually exclusive. Layer when multiple apply:
- Softmax op = `softmax + reduction + transcendental`
- Fused rmsnorm+rope+kvcache = `fused + normalization + stateful-cache` (don't add `scatter-gather` if cache-write is plumbing — see anti-pattern below)
- LayerNorm backward = `normalization + reduction + loss-bwd`

If a tag is plumbing (e.g. shape-1 broadcast inside a fused op, indexed write that's a cache-update side-effect), **omit it**. Only tag the op's defining compute character.

#### Plumbing rules (P0aak v3 — concrete cases)

- **KV cache index-write**: tag as `stateful-cache`, NOT additionally `scatter-gather`. The index-write is a side-effect of cache update, not the op's compute character.
- **cos/sin shape-1 expand inside a fused norm/rope op**: NOT `broadcast`. Plumbing inside the fused chain.
- **Pre-sorted indices consumed as input** (e.g. `sorted_edges`, `expert_offsets` from upstream counting-sort): NOT `sort-select`. The kernel reads pre-sorted data; sorting happened elsewhere.
- **Atomic-accumulation non-determinism in backward kernels**: route to `loss-bwd` + OL-89 (Path-A CPU-truth methodology), NOT `reference-ub`. atomic-add order non-determinism is handled by Path-A reference, not by upper-bound matching tolerance.
- **Reshape / transpose / permute as part of layout setup before main compute**: NOT `broadcast`. Only tag `broadcast` when the broadcast/permute IS the op (e.g. `permute` op, `cat` op).

### Step 4 — Recommend KB sections

For each tag identified, look up the table in §Step 2 — DO NOT grep the filesystem to verify each path. The table is curated and current as of 2026-05-07. If you cite a section not in the table, you're guessing — STOP and either find it via Glob or omit it. Anti-pattern: fabricating `OL-XXX` numbers.

Don't repeat default-loaded files: `KB_INDEX`, `OPERATIONAL_KNOWLEDGE`, `PLATFORM_BUGS`, `ASCENDC_API_CATALOG`, `PATTERN_INDEX`, `ALWAYS_LOADED_RULES`, `hardware/target/ascend950pr.md`. The Python brief construction adds these regardless.

Each recommendation must include:
- `path` (relative to `${CLAUDE_PLUGIN_ROOT}/kb/`, e.g. `OPERATIONAL_KNOWLEDGE.md#OL-103`)
- `reason` (one sentence — why this section applies to THIS op, citing the source signature that triggered it)

### Step 4.5 — Determine algorithm_classification + layered_implementation_plan (schema v3, P0aau-c35.e)

After §Step 2-3 (tags) and §Step 4 (KB recs), determine:

**Step 4.5.1 — Pick algorithm_classification**:

Read `model.py` `forward()` carefully. Count distinct algorithmic sub-ops in the
reference decomposition (NOT plumbing — count actual compute primitives like
matmul, softmax, reduce, scatter, quant, dequant). Decision rule:

- **`single_op`**: 1-2 distinct sub-ops where each is a single line of torch
  code (e.g. `out = x * sigmoid(x * 1.702)` — 1 sub-op fused with one elementwise
  multiply; classify single_op). Examples from current backlog: 1_GELU, 3_Add,
  4_Abs, 18_FusedAddRmsnorm (2 sub-ops but each trivially decomposed).
- **`fused`**: ≥3 distinct sub-ops, OR multi-output API contract (returns
  Tuple[...]), OR sub-ops that share intermediate tensors with materialized state
  (e.g. attention's softmax output feeds matmul's input). Examples:
  3_FusionAttention, 6_MoeFinalizeRouting, 9_TopKTopP, 10_SwigluQuant,
  11_DequantSwigluQuant, 12_KvRmsnormRopeCache, 20_FusedRopeWithQkNormAndKvCacheUpdate,
  27_MultiMaskAttentionAggregation, 29_TanhGatedResidualAddBackward.
- **`other`**: stochastic (15_AttentionSoftmaxWithSoftcappingAndDropout), FFT
  (23_HyenaFftSizePaddingRfft — complex tensor decomposition unclear),
  non-decomposable optimizer-update (17_AdamW — algorithmic but state-coupled).

**Step 4.5.2 — Build the layer plan (only if algorithm_classification == "fused")**:

Walk `model.py.forward` line by line. For each line that produces a new
intermediate tensor, ask: "could a CPU-truth `ref_layer_N(*inputs) → outputs_added`
be expressed in one torch expression?" If yes, that's a layer.

Special cases:
- **Multi-output return**: if `forward()` returns Tuple, Layer 1 lists ALL
  Tuple element names in `outputs_placeholder`. Subsequent layers fill them
  via `outputs_filled`.
- **Backward ops**: layer order is gradient-propagation order (Layer 1 = output-side
  gradient, Layer N = input-side gradient). The `reference_decomposition` per
  layer is the corresponding torch.autograd partial.
- **Optional features**: if a feature appears in <30% of fixture cases (e.g. pse
  pre-bias in 1/61 FA cases), wire it as a final `optional: true` layer with
  `verify_against: full_fixture`. The non-feature path doesn't need feature wiring.

**Step 4.5.3 — Self-check before write**:

- Layer 1's `outputs_placeholder` covers ALL final-return-tuple elements? If
  multi-output and missed, regenerate.
- Each layer's `reference_decomposition` is valid Python expression with only
  imported names from torch / math / model.py forward signature? If unclear,
  regenerate.
- Total layer count is reasonable (typical 2-5 mandatory + 1 optional)? More than
  6 layers is suspicious — re-examine if some can merge.
- For non-fused ops: `applicable: false` with rationale string (don't leave
  empty).

If the layer plan can't be cleanly expressed (e.g. iterative algorithm with
data-dependent loop count, fused op where Layer N+1's input shape depends on
Layer N's output values not just dtype), set `applicable: false` with
`rationale_when_inapplicable` explaining the structural blocker. The Tier 3
methodology gracefully degrades — these ops still go through standard
`await_worker` path.

### Step 5 — Write rationale

Auditable prose (~5-10 sentences):
- Which primitives / patterns identified in source (cite file:line where possible)
- Which tags follow from those signatures
- Which KB sections are most load-bearing and why

Don't speculate about kernel-author decisions. Stick to "what's in the source" + "what KB applies".

### Step 6 — Write JSON

Output ONLY valid JSON to `workspace/<op>/op_classification.json`. Validate JSON before exit.

Print summary line:
```
OP_CLASSIFY_RESULT op=<op> tags=[<comma-sep>] n_kb_recs=<N> rationale_chars=<N>
```

## Anti-patterns (must avoid)

- **Don't recurse into `op_taxonomy.py`'s `OP_TAGS`**. That's the very thing this skill replaces.
- **Don't tag plumbing as compute character**. KV cache index-write is `stateful-cache`, NOT additionally `scatter-gather`. cos/sin shape-1 expand inside a fused op is plumbing, NOT `broadcast`.
- **Don't fabricate KB paths**. The tag→KB tables in §Step 2 are authoritative for v2. If you cite a section not in those tables, verify with Glob or omit.
- **Don't silently use undeclared target-side code as truth.** Any target, sibling,
  or archive source consulted for classification must be provenance-recorded and
  remains non-authoritative; migration truth is the fresh source-architecture NPU capture.
- **Don't tag `reference-ub` unless meaningful**. For non-transcendental deterministic ops, omit. For transcendentals, default-include (the libm-vs-NPU gap is real).

For AscendC-source-other-soc inputs (Atlas A2 / older arch), apply same compute-structure tag table; the primitives are similar names.

## Output handoff

Write `workspace/<op>/op_classification.json`. Print summary line. Exit.

If JSON write fails or content invalid, write `workspace/<op>/op_classification.error` with the error reason — Python orchestrator caller falls back to default-only KB load + warns.

## Provenance

- v1 (P0aaj 2026-05-06): initial draft
- v2 (P0aak 2026-05-07): post-validation iteration. Two test agents (1_gelu, 12_kvrms) found:
  - Free-form-vs-canonical tag confusion → resolved (descriptive metadata, KB recs are load-bearing)
  - `reference-ub` ambiguity → fixed with positive/negative examples
  - Source-language tags ungated → dropped (no distinct KB)
  - Path verification expensive → curated tag→KB tables in §Step 2
  - mtime cache fragile → content-hash (`source_sha256`)
  - Large-file sampling → 50KB head+tail+grep strategy
  - Source coverage gaps → source-signature guidance expanded
  - Multi-tag layering → §Step 3 explicit + plumbing anti-pattern
- v3 (P0aau-c35.e 2026-05-09): Tier 3 layered fused-op methodology. Origin:
  3_FusionAttention C35 audit showed fused-op single-shot worker spawn has zero
  diagnostic granularity at Phase D failure. Schema gains
  `algorithm_classification` + `layered_implementation_plan`; classifier walks
  `model.py.forward` to enumerate sub-ops and emit per-layer reference
  decomposition + multi-output API contract from Layer 1. State machine
  routing (Stage 2) reads `applicable: true` to dispatch through
  `await_layer_worker`. Backward-compat: schema_version=2 readers ignore the
  new fields; brief construction tolerates absent fields by falling back to
  standard `await_worker` routing.

- v3.1 (task#31 2026-06-04): added the `matmul` compute-structure tag and the `attention` op-family tag (the FA-vs-Sinkhorn structural discriminator) + structural-detection guidance. The `attention` tag lets `is_fa_class` narrow to `ATTENTION`; the deterministic op-name backstop (`is_attention_named`) remains a safety net because tag emission is probabilistic. Design: `docs/design/FA_CLASS_DESIGN_NOTES.md#task31-classifier-fa-structural-tag-design`.

C34 of /aog-self-critic flags the bench-name-keyed regression class this skill prevents.
