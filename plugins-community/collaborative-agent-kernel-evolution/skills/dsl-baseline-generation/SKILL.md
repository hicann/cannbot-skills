---
name: dsl-baseline-generation
disable-model-invocation: true
description: Generate initial AscendDSL code (class structure, compute, tiling) from functional PyTorch for pure Vector operators. Use after ascend-call-generation creates the project scaffold.
---

## What I do

Generate initial AscendDSL code with proper class structure, compute methods, and tiling logic for NPU architecture.

## Prerequisite: env.json

Read `output/{op_name}/env.json` before generating initial tiling. Use `ub_size` and `vector_core_cnt` for buffer budgets and core splits. Do not hardcode UB/core defaults when env fields are available.

## Sub-Agent Strategy

This skill reads **large reference files** that are only needed during generation and should not pollute the main orchestrator's context window.

**When called from cake/cake-evo**: The orchestrator SHOULD delegate this skill to a `general-purpose` sub-agent. The sub-agent prompt must include:
- `op_name` and path to `{op_name}_functional.py`
- Operator category (from `op_desc.json`)
- Tiling pre-classification results (category, TOP3 pitfalls, conservative starting point)
- The full text of this SKILL.md (so the sub-agent knows the workflow)
- Instruction to save the output to `output/{op_name}/{op_name}_dsl.py`

**Example orchestrator delegation**:
```
Agent({
  description: "Generate DSL baseline for {op_name}",
  prompt: "Generate AscendDSL code for operator {op_name}. [include: functional.py content, category, tiling classification, SKILL.md instructions]. Save to output/{op_name}/{op_name}_dsl.py. Report: file path and verification checklist results.",
})
```

The sub-agent reads all reference files, generates the DSL, verifies the checklist, and returns only the result summary according to the workflow below.

**When called directly** (not from cake): Execute the workflow below in the current context.

## Workflow

1. Read input functional PyTorch code `{op_name}_functional.py`
2. Select appropriate example (input in `references/input_example/`, matching
   output in `references/output_example/`). Pick by operator category /
   compute pattern:

   | category / pattern | example to read | notes |
   |---|---|---|
   | element-wise (unary/binary) | `leaky_relu` (+ `leaky_relu_unalign` for non-aligned tail) | |
   | reduction (sum/max over a dim) | `sum_reduction_over_a_dimension`, `reduce_sum` | accumulate across ALL tiles |
   | normalization | `rms_norm`, `layer_norm`, `softmax` | |
   | scan / prefix | `cumsum` | |
   | matmul-like | `matmul` | Vector path only; Cube+Vector uses the -cv skills |
   | gather / index | `gather_elements` | per-element indexed load |
   | pooling | `average_pooling2d` | |
   | loss | `mse_loss` | |
   | sort / top-k | `top_k` (output_example) | |
   | **conversion (transpose / permute / reshape / concat / split)** | **`transpose`** | see mandatory notes below |

3. read ascend dsl knowledge @references/ascend_dsl.py
4. generate Ascend DSL code
5. Save to `output/{op_name}/{op_name}_dsl.py`
6. Continue to the next step in agent workflow

#### Conversion / transpose operators (MANDATORY when category == conversion)

Do **not** invent a strided-copy scheme from scratch. Start from the proven
`references/{input,output}_example/transpose.py` baseline (generic N-D
"contiguous-row strided-gather"; correct for rank 2..8 × any perm × any dtype).

Two DataCopyPad hardware constraints the generated DSL MUST respect (each cost
a full runtime-debug round when missed):
- **(H1) 32B pad slots**: a strided gather lowers to multi-block DataCopyPad;
  each 1-element block lands in a 32B-aligned UB slot (NOT tightly packed). The
  contiguous store must mirror the same multi-block layout, and the UB buffer
  must be sized `chunk * 32B`. A tight read-back reads padding garbage → nan/wrong.
- **(H2) blockCount ≤ 4095**: DataCopyPad `blockCount` is a 12-bit field;
  4096 wraps to 0 and copies nothing. Chunk rows so `CHUNK ≤ 2048`.

Transpose anti-regression checklist (MUST pass before saving the DSL):
- **Perm semantics**: for `torch.permute(x, perm).contiguous()`, output dim `j`
  comes from input dim `perm[j]`. Therefore `outShape[j] = inShape[perm[j]]`,
  `lastDim = outShape[ndim-1]`, `S = inStride[perm[ndim-1]]`, and the head
  stride table is `inStridePerm[j] = inStride[perm[j]]`. Do **not** use
  `permInv[j]` as the input stride for output dim `j`. `permInv` is only valid
  if explicitly reconstructing `input_coords[perm[j]] = output_coords[j]`.
- **Row-major row decomposition**: `rowIdx` enumerates the first `ndim-1` output
  dims in row-major order, so decompose from high head dim to low head dim:
  `for j = ndim-2 downto 0: coord[j] = rem % outShape[j]; rem //= outShape[j]`.
  Decomposing from `j=0` upward is column-major and causes all-case precision
  mismatch even though the kernel compiles and runs.
- **DMA direction**: keep the baseline as output-row processing: strided gather
  from input, contiguous write to output. Do **not** rewrite it as contiguous
  input read plus strided output scatter; that path is fragile with packed UB
  queues and caused runtime crashes in prior runs.
- **No manual UB slot math**: after a strided gather, do not invent
  `srcStride = 32 - sizeof(T)` or treat 32B slots as byte gaps. The store must
  mirror the template's multi-block layout. Both a tight contiguous read-back
  and a hand-written `slotGap` read-back are known bad variants.
- **S==1 anti-timeout path**: do not process large contiguous fast-path tensors
  one output row at a time when adjacent rows are contiguous in input. Detect
  contiguous row runs and copy `runRows * lastDim` elements in chunked contiguous
  DMA. This is a correctness-preserving baseline optimization, not cake-evo.
- **Required self-check cases**: reason through at least one self-inverse perm
  (`[1, 0]`) and three non-self-inverse perms (`[0, 2, 3, 1]`, `[2, 0, 1]`,
  `[0, 3, 1, 2]`) before leaving stage 7. CPU offset simulation is helpful but
  not sufficient for DataCopyPad semantics; stage 8/9 must still run NPU
  micro-cases that cover `S == 1` and `S != 1`.

Performance: element-level strided gather is bandwidth-bound — fine for
small/medium tensors, slow on large 2D transposes. For the `S == 1` fast path,
MUST merge adjacent output rows when their input bases are also contiguous
(`input_base(next_row) == input_base(cur_row) + lastDim`), then copy the whole
run as larger contiguous chunks. This prevents million-row contiguous cases from
timeout due to one tiny DMA per row. Leave the vnchwconv / on-chip 2D-block
rewrite to a later optimization pass (cake-evo).

### input explain
- **[module_fn]**: a pure PyTorch functional implementation  
- **[Model Definition]**: a `Model(nn.Module)` calling [module_fn]  
- **[Configurations]**: hyper-parameters and input helper functions  

### generate requirment
Your task is to generate an **Ascend DSL** that replicates the computation in [module_fn], optimized for the input shape specified in Configurations.
You only can launch a kernel once.
Follow the implementation patterns demonstrated in the example, **always use `tl.num_vec_cores()` for dynamic core count** (never hard-code `n_cores = 16` or any constant), and adopt a similar core partitioning and tiling strategy where applicable.
Use **pivot distribution** to handle workloads not evenly divisible by core count:
```
n_cores   = tl.num_vec_cores()
n_used    = min(n_cores, total_work)
base      = total_work // n_used
pivot     = total_work % n_used       # first 'pivot' cores get base+1 units
# in kernel:  my_count = base + (1 if pid < pivot else 0)
#             my_start = pid * base + min(pid, pivot)
```
Note that the Ascend DSL host side must use the same argument list as module_fn.
Therefore, the inputs passed from the host to module_fn must already match the exact shapes expected by module_fn, including any shapes that result from transposed dimensions or similar transformations.
Natural-language comments are mandatory and are part of the DSL.

**⚠️ CRITICAL: NO SIMPLIFICATION OR PLACEHOLDERS ALLOWED**

You MUST implement the COMPLETE algorithm from module_fn. This is NOT a baseline or prototype - it is the FULL implementation.

**FORBIDDEN**:
- ❌ Placeholder values (e.g., `gate_val = 0.5`, `result = 0.0`)
- ❌ Simplified logic (e.g., skipping RMSNorm, using fixed constants)
- ❌ Comments like "simplified", "placeholder", "TODO", "will be refined later"
- ❌ Partial implementations (e.g., only processing first tile, skipping loops)
- ❌ Approximations that don't match the reference (e.g., `x * 0.5` instead of full sigmoid)

**REQUIRED**:
- ✅ Implement EVERY operation from module_fn exactly as specified
- ✅ Use ALL input tensors (verify each input is loaded and used)
- ✅ Implement ALL loops (tile loops, reduction loops, etc.)
- ✅ Use proper AscendC APIs (reduce_sum, vexp, vsqrt, etc.) - do NOT use Python math functions
- ✅ Accumulate across ALL tiles for reductions (not just last tile)
- ✅ Apply ALL activations and transformations correctly
- ✅ Handle ALL edge cases (boundary conditions, special values)

**Verification Checklist** (must pass before considering DSL complete):
- [ ] Does the DSL match module_fn's computation exactly?
- [ ] Are all input tensors loaded and used?
- [ ] Are all loops from module_fn present?
- [ ] Are reductions accumulated correctly across tiles?
- [ ] Are activations applied with proper DSL APIs?
- [ ] Will different inputs produce different outputs?
- [ ] Is there any placeholder or simplified logic? (must be NO)
- [ ] For transpose/permute: is `outShape[j] = inShape[perm[j]]` and
      `inStridePerm[j] = inStride[perm[j]]` used, with no accidental
      `permInv[j]` stride table?
- [ ] For transpose/permute: is row index decomposition performed from
      `ndim-2` down to `0`, not from `0` upward?
- [ ] For transpose/permute: does the data movement keep contiguous output
      stores and avoid strided output scatter / packed-UB multi-block stores?
- [ ] For transpose/permute: are DataCopyPad 32B slot layout and
      `blockCount ≤ 4095` respected by using the provided template?
- [ ] For transpose/permute: does the `S == 1` fast path merge adjacent
      contiguous rows into larger contiguous copies, avoiding one tiny DMA per
      row on million-row tensors?

If the algorithm is complex (e.g., multi-step fusion, convolution, attention), break it down into clear steps with comments, but implement EVERY step completely.
