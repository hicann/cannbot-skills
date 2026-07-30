---
name: aog-input-gen-builder
description: >
  Phase O2.5 input_gen.py + edge dataset generator for the bundled orchestrator.
  Reads a source-architecture AscendC package or a differentiable forward spec, infers
  the case_gen SCHEMA (tensor_inputs,
  scalar_inputs, shape_derive, invariants), and emits a ready-to-run
  input_gen.py under `workspace/{op}/`. Running that script produces the
  edge_inputs.pt + manifest.json artifacts the workflow_critic enforces at
  `O2_5.B.*` — removing the "hand-write input_gen.py per op" burden that
  used to make orchestrators reach for `.workflow_exception_O2_5` waivers.
  Use when Phase O2.5 needs deterministic edge inputs for a new operator.

  Status: V1 (2026-04-24). Three templates (simple / fused / scalar-shaped)
  cover the patterns seen in ops #3, #5, #6, #11, #20, #24, #25. Complex
  cases (FFT op#23, variable-length lists) still need manual authoring
  but should be rare.
---

# /aog-input-gen-builder


> **Anti-pressure protocols (MANDATORY, load first)**: read `${CLAUDE_PLUGIN_ROOT}/kb/shared/ANTI_PRESSURE_PROTOCOLS.md` before any decision. P1–P8 are LLM-pressure drift modes (user-watching, context-filling, batch-throughput, simple-op assumption, failure discomfort, infrastructure friction, closure desire, tool-path-of-least-resistance) that override technical rules under load. Cite the relevant Px at every high-leverage decision point (spawn / done / PARTIAL / skip-verify).

Invoked from the bundled orchestrator's Phase O2.5 Step 1 for migration or
backward generation, or by a human directly. Produces
`workspace/{op}/input_gen.py` + (after running it)
`edge_inputs.pt` + `manifest.json` + `edge_dataset.pt` — the four artifacts
the workflow_critic `O2_5.B.art{1..4}` rules require before aog-kernel-worker
is allowed to spawn.

## When to invoke

- **From the bundled orchestrator** (primary use): Phase O2.5 Step 1, immediately
  after the source contract or differentiable forward specification is staged.
  Any op that doesn't already have a
  valid `workspace/{op}/input_gen.py` + artifacts triggers this skill.
- **Standalone**: `Skill(name="aog-input-gen-builder", args="<op_source_path> [workspace_dir]")`
  produces a SCHEMA-ready `input_gen.py` at workspace_dir (default `workspace/{op_slug}/`).

Do NOT invoke when:
- `workspace/{op}/input_gen.py` already exists AND passes critic check
  (has `from case_gen import` + `COVERAGE_TIER` string constant).

## Contract

### Input
- `source_path`: absolute path to one of:
  - Source-architecture AscendC operator directory for `port_a3_to_a5`
  - PyTorch forward spec with `BACKWARD_SPEC` for `backward`
- `workspace_dir`: (optional) where to write `input_gen.py`. Default:
  `workspace/{lowercase_op_name}/`

### Output (files written to workspace_dir)
- `input_gen.py` — always (ready-to-run, passes critic inv1)
- After running `python3 input_gen.py`:
  - `edge_inputs.pt` — initial inputs
  - `manifest.json` — metadata, case list, sha256
- After running on-A5 edge_runner.py (for float-sensitive ops) OR in-python
  reference (for int-only ops):
  - `edge_dataset.pt` — inputs + reference outputs

## Workflow (5 steps)

### Step 1 — Read source

Identify the op class (`simple` / `fused` / `scalar-shaped` / `other`):

| Class | Markers | Template |
|-------|---------|----------|
| `simple` | Single return tensor, 1-3 input tensors, same shape, single dtype; no shape_derive needed | `templates/input_gen.simple.py` |
| `fused` | Multiple shape-related tensors (e.g. `[N,2H]` + `[1,2H]` + `[N,1]`); may have scalar probe values | `templates/input_gen.fused.py` |
| `scalar-shaped` | Tensor shapes depend on scalar input values (e.g. op#3: `sampled_token_ids=[num_queries,1]`) | `templates/input_gen.scalar_shaped.py` |
| `other` | In-place mutation on multiple tensors, variable-length lists, FFT rank variants, dropout RNG | Emit `scalar-shaped` stub + `# TODO:` markers + surface the ambiguity to user |

Read:
1. The Model class `forward()` signature — argument names + type annotations
2. The Model class docstring — dtype (look for `dtype: int64`, `torch.float16`), shape hints (`[num_seqs,]`, `[N, H]`), constraints (`Must be > 0`, `non-negative`, `index into ...`)
3. Committed source tests/fixtures or forward-spec examples, when present —
   actual shape patterns, scalar default values, and variation range

### Step 2 — Classify each input

For each argument in `forward(self, <args>)`:

**Tensor inputs** (infer):
- `dtype`: from docstring (`dtype: int64` → `torch.int64`) or a committed source case (`"dtype": "bf16"`)
- `shape_derive`: examine source cases — how does this tensor's shape relate to base_shape?
  - Always `base_shape` → no `shape_derive` needed (simple case)
  - Different rank / broadcast → `shape_derive: lambda s: [...]`
  - Shape depends on a scalar → `shape_derive: lambda s, sc: [sc["<scalar>"], ...]`
- `invariant` (from docstring keywords):
  - "index into" / "token id" / "mapping" / "permutation" → `"permutation"` (if shape matches output numel) or `"index_range:<scalar>"` (if indices into another tensor)
  - "non-negative" / ">= 0" → `"non_negative"`
  - "positive" / "must be > 0" / "seq_len" / "count" → `"positive"`
- `int_range` (for int tensors without invariant): examine source-case value ranges; default `(0, 1000)` is safe for general int64.

**Scalar inputs** (infer):
- `dtype`: `"int"` for named counts / indices; `"float"` for scale factors; `"bool"` for flags; `"str"` for enum modes
- `default`: from the first committed source case, or the most common value across cases
- `probe_values`: union of all distinct values seen in source cases (deduplicated, sorted)
- `derive(base_shape)`: if the scalar value equals a dimension of any tensor's shape (e.g. `num_seqs == input_tokens.shape[0]`), emit `lambda s: s[0]` (or equivalent). Skip `probe_values` when `derive` is set — the two are mutually exclusive by design (Band C skips derived scalars).
- `invariant`: "positive" / "non_negative" for counts; "le:<other>" for tight dependencies (informational, not enforced in case_gen yet)

### Step 3 — Pick template + fill SCHEMA

Copy the appropriate template from `templates/input_gen.<class>.py` to `workspace/{op}/input_gen.py`, replace:
- `<<<OP_NAME>>>`
- `<<<FORMULA>>>` — one-line pseudo-code from the docstring
- `<<<SCHEMA>>>` — the filled SCHEMA dict
- `<<<DTYPE>>>` — `torch.int64` / `torch.float32` / etc (global default; per-tensor override via `dtype` in tensor_input)
- `<<<COVERAGE_TIER>>>` — `pilot` for fast iteration, `sign_off` for archival op, `production` rarely.

Validate BEFORE writing the file:
- SCHEMA dict is syntactically valid Python (`ast.parse` the emitted SCHEMA lines)
- Every tensor_input has a `name`
- Every scalar with `derive` has no `probe_values` (mutually exclusive)
- If rank=1 and all tensor shape_derive produce `[base[0]]`, consider using no `shape_derive` (simpler SCHEMA)

### Step 4 — Run input_gen.py + validate

```bash
cd workspace/{op}
python3 input_gen.py
```

Expected stdout: `wrote edge_inputs.pt ({N} cases) + manifest.json` where N is
- `pilot`: 12-18
- `sign_off`: 28-45
- `production`: 50-80

Fail modes + fixes:
- `RuntimeError: input_gen.py was NOT customized per op` → SCHEMA has placeholder `<<<>>>` markers left in; re-check Step 3
- `ValueError: base_shape_filter excludes default base_shape AND every shape_plan entry` → filter is too strict; loosen `base_shape_filter` OR adjust `rank`
- `ValueError: shape_derive returned ... must be list[int]` → shape_derive callable returned a negative or non-int dim; check arithmetic
- `invariant=...<scalar>... must be positive integer` → scalar referenced by `index_range:<scalar>` evaluated to 0/negative; check its `derive` or `default`

### Step 5 — Produce edge_dataset.pt

Two paths:

**Path A — int-only op (no float rounding)**: reference is computable in pure Python/CPU bit-exact. Write a small `compute_reference.py` next to `input_gen.py` that loads `edge_inputs.pt`, applies the formula in Python, saves `edge_dataset.pt`. No A5 round-trip needed.

**Path B — float op (CANN/torch_npu reference needed)**: write an `edge_runner.py` adapted from `workspace/moefinalizerouting/edge_runner.py` (stage_dir pattern, `.npu().contiguous()` standard). Transfer to A5 via stage dir, run, scp back. This is the same pattern as op#5/op#6.

The skill emits a stub of the appropriate path based on input dtypes (all-int → Path A, any float → Path B). User/orchestrator adjusts the formula.

## Templates

See `templates/` for the three starting points:

- `input_gen.simple.py` — 1-3 tensors, same shape, no scalars or simple probe_values; rank 1-4. Covers op#5 (was overshooting for this class, this template is the MINIMUM SCHEMA).
- `input_gen.fused.py` — multi-tensor with shape_derive for broadcast relationships; rank 2-4. Covers ops #6 (MoeFinalizeRouting: 7 tensors, multiple shape_derive) and #11 (DequantSwigluQuant: shape-interdep).
- `input_gen.scalar_shaped.py` — tensor shapes depend on scalar values, scalars derived from base_shape; rank 1-2. Covers op#3 (AdvanceStepFlashattn) and similar vLLM-style ops.

## Integration with the bundled orchestrator

Phase O2.5 Step 1 should invoke this skill BEFORE any attempt to write input_gen.py manually. Orchestrator workflow:

```
# Phase O2.5 Step 1 (migration/backward)
if not (workspace/{op}/input_gen.py exists AND passes critic inv1):
    Skill(name="aog-input-gen-builder",
          args=f"{source_contract_path} workspace/{op}")
    # Now workspace/{op}/input_gen.py exists with valid SCHEMA

# Step 2: run input_gen.py to produce edge_inputs.pt + manifest.json
bash: cd workspace/{op} && python3 input_gen.py

# Step 3: produce edge_dataset.pt (Path A or B per dtype)
# ...
```

When the skill cannot confidently infer a field, it emits it with a
`# TODO: orchestrator review` comment and stops short of running
`python3 input_gen.py`. Orchestrator finalizes the SCHEMA (guided by the
TODOs), then runs it. If the TODO cannot be resolved even with domain
knowledge, orchestrator may file a case_gen feature request (DEBT) but
should NOT fall back to `.workflow_exception_O2_5` — that path is for
genuine engine-level gaps, not authoring-ergonomics gaps (aog-self-critic C17).

## CRITICAL RULE (P0aaa, 2026-05-06): Input requirements are immutable

**Never restrict the operator's input space to fit case_gen's expressive
power.** If the source contract or differentiable forward specification admits inputs that the current
case_gen SCHEMA cannot express (e.g. non-square chained matmul `[M,K1] →
[K1,N1] → [N1,N2]` with K1≠N1≠N2 needs rank-4 base_shape; rank-dependent
tuple lengths; dim-constrained shapes like NMS `[N, 4]`), the skill MUST:

1. **Emit BLOCK** with explicit engine-extension request, citing what
   primitive is missing. Do NOT write `input_gen.py`.
2. **Open a DEBT entry** for the missing case_gen primitive (the same
   DEBT may already exist; check `docs/DEBT.md`).
3. **Refuse to silently restrict** — e.g. simplifying chained-matmul to
   K=N square case is silent coverage fraud. The kernel passing on
   K=N doesn't validate the spec's K≠N admission.
4. **Refuse to emit `coverage_completeness: degraded`** — that's the
   same fraud with a label.

The acceptable outcomes are: (a) fully express the spec in SCHEMA and
proceed; (b) BLOCK + file engine extension request. There is no third
option.

User direction 2026-05-06: "we cannot modify or simplify the input
requirement." Caught when V1 of this skill silently restricted L4 2_FFN
to K=N square cases.

## Known limitations (V1) — current BLOCK-class

1. ~~**Variable-length input lists** (op Cat)~~ — **UNBLOCKED V1.7
   (2026-05-21)**: case_gen now supports `kind="list_of_tensors"` with
   `list_length_plan` sweep + per-item `per_item_shape_derive`. See
   §"V1.7 unblocked: variable-length tensor lists" below for the SCHEMA
   shape. 13_Cat / 14_Split / similar variadic-list ops can now be
   expressed without BLOCK.
2. **FFT / complex tensors** (op#23 HyenaFftSizePaddingRfft) — case_gen
   dtype plan doesn't emit torch.complex64/128. **BLOCK + extension request.**
3. **Dropout RNG reference** (op#15) — reference output depends on RNG
   state; not reproducible bit-exact. Phase O1.5 classifies as
   DET_POLICY=n/a; edge_dataset verification should compare *mask
   distribution* not bit-exact. Skill emits stub with warning.
4. **Docstring parsing is regex-based**, not proper NLP. If the op's
   docstring omits "must be > 0" but the constraint is implicit in the
   formula (e.g. indices into tensor of length N), skill may miss the
   invariant. Orchestrator must review.
5. **Dim-constraint shapes** like NMS `[N, 4]` (last dim must equal 4)
   — case_gen has `base_shape_filter` to reject invalid bases but no
   constructive primitive to PRODUCE valid 4-dim ones. **BLOCK + extension.**

## V1.5 unblocked classes (2026-05-18) — DO express, DO NOT BLOCK

Four previously-BLOCK-class limitations were resolved by case_gen V1.5
(commit `c2262f86`). DO NOT emit BLOCK on these patterns; DO express
them using the new SCHEMA fields.

### 1. Multi-rank ops — use `schema["ranks"]`

When the source contract admits inputs of multiple ranks
(e.g. 5_Cumsum / 8_Sort accept `x` of rank ∈ {1, 2, 3, 4}; 17_AdamW
accepts multi-rank parameter tensors), declare a **list** instead of
a single `rank`:

```python
SCHEMA = {
    "tensor_inputs": [{"name": "x", "role": "operand"}],
    "scalar_inputs": [...],
    "tensor_output": "out",
    "ranks": [1, 2, 3, 4],   # ← was: "rank": 4 (silent coverage reduction)
}
```

`generate_cases` loops per rank internally + merges. Each case carries
`meta.rank` for downstream filtering. `_shape_plan` already supports
rank 1..4 (1D / 2D / 3D-d3_* / 4D-d4_*) so multi-rank gets the
typical+edge shape coverage for each rank.

### 2. Multi-dtype ops — use `schema["dtypes"]`

When the reference is dtype-branched (5_Cumsum: CPU `np.cumsum` for
fp32/fp16, NPU `torch.cumsum` for bf16) OR when validating multiple
dtypes is required (sort / topk on fp32/fp16/bf16), declare a list:

```python
SCHEMA = {
    ...
    "dtypes": [torch.float32, torch.float16, torch.bfloat16],
}
```

`generate_cases` loops per dtype + merges. Each case carries
`meta.dtype` (str tag like `"float32"`). Tensor data is generated in
the per-case dtype. The single-dtype `dtype=` kwarg on
`generate_cases()` remains the V1 path; if `schema["dtypes"]` is
declared, it takes precedence.

### 3. Cross-scalar dependencies — use 2-arg `derive`

When one scalar's value depends on another scalar (e.g. TopK's `k`
depends on `dim` AND base_shape — `k = min(shape[dim], 8)`), declare
the derive with two parameters:

```python
def derive_dim(base_shape):  # 1-arg, V1 unchanged
    return -1

def derive_k(base_shape, scalars):  # 2-arg, NEW V1.5
    dim = scalars["dim"]
    return min(base_shape[dim], 8)

SCHEMA = {
    "scalar_inputs": [
        {"name": "dim", "dtype": "int", "default": 0, "derive": derive_dim},
        {"name": "k",   "dtype": "int", "default": 1, "derive": derive_k},
    ],
    ...
}
```

The skill inspects the callable signature at runtime: 1-param keeps
V1 semantics (base_shape only), 2-param gets `(base_shape, scalars)`
where `scalars` is the dict of already-resolved scalars from the same
case. Scalars are resolved in **declaration order** — a later derive
can reference any earlier scalar by name.

### 5. Rank-dependent variable-length tuple — use `tuple_of_int` + `length_derive`

(Added 2026-05-06 as case_gen "P0aaa task #105", documented in SKILL.md
2026-05-19. Previously this class was incorrectly emitting BLOCK because
the skill didn't surface the primitive.)

When a scalar parameter is a tuple of ints whose length depends on the
input tensor's rank (e.g. 15_Pad's `pad: tuple` — must have 2 ints per
padded dim, so `len(pad) = 2 * rank`), declare a `tuple_of_int` scalar
with `length_derive`:

```python
SCHEMA = {
    "op_name": "15_Pad",
    "tensor_inputs": [{"name": "x", "role": "operand"}],
    "scalar_inputs": [
        {"name": "pad", "dtype": "tuple_of_int",
         "length_derive": lambda base_shape: 2 * len(base_shape),
         "value_range": (0, 5)},     # each int drawn uniformly from [0, 5]
        {"name": "mode", "dtype": "str", "default": "constant",
         "probe_values": ["constant", "reflect", "replicate"]},
        # Note: `value` only meaningful when mode=="constant" — case_gen
        # passes it through as-is; the harness's CPU-truth bundler calls
        # Model.forward with all scalars; the model implementation should
        # ignore `value` when mode != "constant" (PyTorch's pad behavior).
        {"name": "value", "dtype": "float", "default": 0.0,
         "probe_values": [0.0, 1.0, -1.0]},
    ],
    "tensor_output": "out",
    "ranks": [2, 3, 4],
    "dtypes": [torch.float32, torch.float16],
}
```

`length_derive(base_shape)` returns the tuple's length; values are drawn
uniformly from `value_range` (default `(0, 5)`). If you also declare
`derive`, it overrides `length_derive` for that case.

For ops where a scalar's validity depends on another scalar's value
(15_Pad's `value` only used when `mode == "constant"`), the current
recipe is to **declare both scalars unconditionally** and let
Model.forward / the kernel ignore the irrelevant one. The case_gen layer
does not yet support conditional scalar sampling (`applies_when`); if
you need that semantic, BLOCK with note "applies_when conditional
scalar sampling not yet in case_gen — needs primitive extension".

### 6. Optional-tensor inputs (`weight=None, bias=None`)

**V1.6.B (2026-05-19, DEBT-069 Gap A): use `optional=True`**

When `forward()` declares optional tensor parameters that default to
`None` (e.g. 10_LayerNorm `forward(x, normalized_shape, weight=None,
bias=None)`, 25_NLLLoss `forward(input, target, weight=None, ...)`),
declare those tensors with `optional=True`. case_gen forks each case
into 2 variants: tensor materialized + tensor absent (`inputs[name] = None`).
With k optional tensors, case count multiplies by 2^k (all-pairs of
presence states).

```python
SCHEMA = {
    "tensor_inputs": [
        {"name": "x", "role": "operand"},
        {"name": "weight", "optional": True,
         "shape_derive": lambda base: [base[-1]]},
        {"name": "bias",   "optional": True,
         "shape_derive": lambda base: [base[-1]]},
    ],
    "scalar_inputs": [
        {"name": "normalized_shape", "dtype": "tuple_of_int",
         "length_derive": lambda base: 1,
         "value_range": (None, None)},
    ],
    ...
}
```

Behavior:
- Each base case forks into `2^k` cases where k = count of optional tensors
- Absent-case `inputs[name]` is Python `None` — Model.forward(weight=None)
  short-circuits PyTorch's un-weighted reduction path
- Per-case meta records `optional_<name>_present: True | False` for downstream
  filtering
- Case name suffix `_<name>T` or `_<name>F` indicates presence per tensor

Empirical anchor (2026-05-19): 25_NLLLoss kw-5 1-char fix (`wsum > 0.0f`
→ `wsum != 0.0f`) only catchable because the failing case had weight
present. With `optional=True`, both the weight-present and weight=None
paths get tested per case → kernel that crashes on `weight is None`
or that always reads `weight[target[i]]` without presence check gets
caught.

### 7. Variable-length tensor lists (`tensors: list[Tensor]`)

**V1.7 (2026-05-21, 13_Cat BLOCK class closure): use `kind="list_of_tensors"`**

When `forward()` accepts a Python list of tensors of variable length
(13_Cat `forward(tensors: list, dim: int)`, 14_Split-merge, block_diag,
etc.), declare with `kind="list_of_tensors"` + `list_length_plan`.
case_gen forks each case once per list length, producing a Python list
of N tensors per fork in `inputs[name]`.

```python
SCHEMA = {
    "tensor_inputs": [
        {
            "name": "tensors",
            "kind": "list_of_tensors",
            "list_length_plan": [2, 3, 4],  # sweep N ∈ {2, 3, 4}
            # OPTIONAL per-item shape: omit → all items use base_shape (uniform)
            # 3-arg: lambda(base_shape, item_idx, list_length) → shape_list
            # 4-arg: lambda(base_shape, item_idx, list_length, scalars) → shape_list
            "per_item_shape_derive": lambda s, i, N: [s[0] + i * 16] + list(s[1:]),
        },
    ],
    "scalar_inputs": [
        {"name": "dim", "dtype": "int", "default": 0, "probe_values": [0, 1, -1]},
    ],
    ...
}
```

Behavior:
- Each base case forks into `len(list_length_plan)` cases (multiplies case count).
- Per-case `inputs[name]` is a Python `list[torch.Tensor]` of length N.
- `per_item_shape_derive` can produce non-uniform shapes — e.g. concat along
  dim 0 where each item has a different dim-0 size.
- Per-case meta records `list_length_<name>: N` for filtering.
- Case name suffix `_<name>N<value>` indicates the list length per case.

For non-uniform concat-along-dim shapes (the 13_Cat use case), use the
4-arg form to read `scalars["dim"]` and vary only the concat axis:

```python
def per_item_shape(base_shape, i, N, scalars):
    dim = int(scalars.get("dim", 0)) % len(base_shape)
    s = list(base_shape)
    s[dim] = s[dim] + i * 16   # only the concat-axis varies
    return s
```

Empirical anchor (2026-05-21): 13_Cat documented as the canonical
BLOCK-class case in `SKILL.md` V1 §"Known limitations" #1. Without this
primitive, the skill correctly emitted `O2_5_BLOCK.md` (per P0aaa: no
silent restriction). With V1.7, 13_Cat / 14_Split / similar variadic
ops can be expressed; BLOCK is no longer required.

If `value_range` doesn't work because elements must EQUAL specific shape
indices (not just uniform-random), use `derive` to return the exact
tuple:

```python
{"name": "normalized_shape", "dtype": "tuple_of_int",
 "derive": lambda base: tuple([base[-1]])}   # last-dim only
```

### 4. Rank-dependent probe values — use callable `probe_values`

When a scalar's probe set depends on the rank (e.g. `dim` for rank=4
admits {-4..3}, for rank=1 admits {-1, 0}), declare `probe_values` as
a callable:

```python
SCHEMA = {
    "scalar_inputs": [
        {"name": "dim", "dtype": "int", "default": 0,
         "probe_values": lambda rank: list(range(-rank, rank))},
    ],
    "ranks": [1, 2, 3, 4],
    ...
}
```

Or shape-dependent (use parameter name `base_shape` to pick this form):

```python
"probe_values": lambda base_shape: list(range(-len(base_shape), len(base_shape)))
```

Currently applied in scalar-only mode + sign_off cross-probe pairs.
Tensor-mode Band C probe loop is not yet implemented; if you need
explicit scalar sweep at base_shape for a tensor-mode op, emit a BLOCK
with note "tensor-mode Band C probe not yet implemented".

### Worked schema — 5_Cumsum

```python
SCHEMA = {
    "op_name": "5_Cumsum",
    "formula": "out = torch.cumsum(x, dim=dim)  # prefix-sum along axis",
    "tensor_inputs": [{"name": "x", "role": "operand"}],
    "scalar_inputs": [
        {"name": "dim", "dtype": "int", "default": 0,
         "derive": lambda base_shape: -1},  # last axis works for any rank
    ],
    "tensor_output": "out",
    "ranks": [1, 2, 3, 4],
    "dtypes": [torch.float32, torch.float16, torch.bfloat16],
}
```

Produces `4 ranks × 3 dtypes × N_shape_plan` cases. No silent coverage
reduction. No BLOCK.

### Worked schema — 9_TopK (cross-scalar dependency)

```python
def derive_k(base_shape, scalars):
    return min(base_shape[scalars["dim"]], 8)

SCHEMA = {
    "op_name": "9_TopK",
    "tensor_inputs": [{"name": "x", "role": "operand"}],
    "scalar_inputs": [
        {"name": "k",        "dtype": "int", "default": 1, "derive": derive_k},
        {"name": "dim",      "dtype": "int", "default": 0,
         "derive": lambda base_shape: -1},
        {"name": "largest",  "dtype": "bool", "default": True,
         "probe_values": [True, False]},
        {"name": "sorted",   "dtype": "bool", "default": True,
         "probe_values": [True, False]},
    ],
    "tensor_output": "out",
    "ranks": [1, 2, 3, 4],
    "dtypes": [torch.float32, torch.float16, torch.bfloat16],
}
```

Note `k` is declared AFTER `dim` so `derive_k` can read
`scalars["dim"]` from the already-resolved dict.

## Provenance

Built 2026-04-24 after a correction that op generation must enter the complete
Skill-driven workflow instead of being hand-driven. Covers the Phase O2.5 automation gap previously
documented in handover §"未 codify 的隐含知识" #4 (case_gen invariants)
and the absence of an input_gen.py authoring skill. See
`${CLAUDE_PLUGIN_ROOT}/skills/aog-self-critic/SKILL.md` §C17 for the full incident analysis.
