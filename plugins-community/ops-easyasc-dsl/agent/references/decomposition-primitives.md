# Legal Decomposition Primitives

Reference clipboard used by `agent/playbooks/decompose-pytorch.md`. Each plan's
`refs/plan.md` lists the primitives it invokes, cross-referenced by ID (`P1`,
`L1`, `A1`, `F1`).

> For the **reverse** direction — how a finished composite / vendor API decomposes
> into small ISA primitives (a reusable recipe dictionary) — see
> `agent/references/composite-api-recipes.md`.

Four tiers:

- **✅ Tier P** — always legal, zero numerical error
- **🟡 Tier L** — lossy but legal under `loose` policy when measured and within budget
- **⚠️ Tier A** — legal but requires algorithmic rewrite (not a mechanical split)
- **❌ Tier F** — default forbidden; require explicit user opt-in with justification

## ✅ Tier P — Always legal (zero numerical error)

| ID | Name | Definition | Hardware mapping |
|---|---|---|---|
| **P1** | Outer-axis split | Split along a non-reduced axis; each slice runs independently. | Multi-core grid via `GetCubeIdx()` / `GetVecIdx()`; no reduction needed. |
| **P2** | Elementwise fusion | Chain two or more pointwise ops (`add`, `mul`, `relu`, cast, etc.) into a single kernel's vec path. | Single vec-pipe traversal; cuts GM round-trips. |
| **P5** | Matmul reduction split via L0C accumulation | Split a matmul along K; run slices with `mmad(..., is_init=False)` so L0C accumulates in fp32. | The `splitk` pattern in `easyasc/shortcuts/matmul.py`. Minimum split tile = 32. |
| **P6** | Kernel fission at a materialized boundary | Cut the DAG at any edge that already lives in GM; the two sides become separate kernels. | No algebraic change; just a launch boundary. |
| **P7** | Same-shape broadcast hoist | Move a broadcast-friendly op (e.g., per-row scale) to the tile boundary closest to its consumer. | Reduces UB traffic; semantics unchanged. |

IDs `P3` and `P4` are intentionally unused. Earlier drafts treated floating-point distributivity and GM atomic reduction as zero-error primitives; use `A4` and `L5` instead so the numerical policy is explicit.

## 🟡 Tier L — Lossy, budget-constrained

Legal only under `loose` policy or an explicit tolerance budget, and only with
measurement. Every L-class primitive invoked must appear in the plan-local
lossy/cast table in `refs/plan.md`.

| ID | Name | Definition | Constraints |
|---|---|---|---|
| **L1** | Matmul precision downcast | `A @ B` (fp32 × fp32) → `A.to(h).to(fp32) @ B.to(h).to(fp32)` with `h ∈ {fp16, bf16}`. | L0C still accumulates in fp32. Must appear in ledger and be measured against the unchanged oracle. Heuristic: wide range → bf16, bounded → fp16. |
| **L2** | Vec intermediate downcast | A vec-side intermediate that never feeds a reduction accumulator may be carried in fp16 or bf16 between fused vec ops, upcast back to fp32 at the final store. | Cannot cross a cube boundary without dtype alignment per L3. Must appear in ledger. |
| **L3** | Mixed-dtype alignment to matmul | When a value feeds a matmul, cast to match the matmul's declared input dtype; do not rely on implicit promotion. | Not a new source of loss if the matmul's input dtype is already in the ledger under L1. If not, this is L1 disguised — record it. |
| **L4** | Tile-local reduction reorder | Within a tile, the reduction order may differ from the mathematical reference (matches hardware-native mmad accumulation order). | Size of reorder ≤ tile size; never across tiles unless explicitly declared and measured. |
| **L5** | Floating reduction split with atomic merge | Split along a reduced axis; each shard writes its partial result to GM via `atomic_add`, or via `atomic_max` / `atomic_min` with declared finite-value semantics. | GM atomics. **Float dtype only.** `atomic_add` changes merge order and must appear in the ledger. `atomic_max` / `atomic_min` must state NaN and tie behavior; if it cannot match the PyTorch contract, the candidate is infeasible. |

## ⚠️ Tier A — Needs algorithmic rewrite

Legal but requires correctness work beyond a mechanical split. Flag to the user before proposing.

| ID | Name | Definition | Reference |
|---|---|---|---|
| **A1** | Streaming softmax (flash-style) | Replace a global softmax with running-max + running-denom correction so that a matmul → softmax → matmul chain can tile the reduced axis. | `agent/example/kernels/a2/attention/flash_attn_full.py`, `agent/example/kernels/a5/mha_ifa*.py` |
| **A2** | Gradient / activation recompute | Drop a large intermediate and recompute it from inputs to save memory. | Costs cube/vec time, saves on-chip/GM. Only if memory is the binding constraint. |
| **A3** | Chunked recurrence (state bridge) | Convert a recurrent reduction into a sequence of chunks that pass running state through GM. | `agent/example/kernels/a5/gdn_legacy/delta_h_state_bridge_v1_c8.py`. Size and synchronize every reused slot from the real delayed lifetime; see `agent/references/constraints/sync.md` and `agent/references/patterns/buffer-slot-lifetime.md`. |
| **A4** | Matmul distributivity | Rewrite `A @ (B + C)` as `A @ B + A @ C`, or the symmetric left-side form. | Algebraically valid but not bit-exact for floating point because rounding changes. If used with floating operands, record the rewrite in the ledger and verify end-to-end; forbidden under `bit-exact` unless the dtype/domain makes it exact. |

## ❌ Tier F — Default forbidden

| ID | Name | Why forbidden |
|---|---|---|
| **F1** | Floating-point reduction reorder beyond declared tile-local (`L4`) or atomic-merge (`L5`) scope without measurement | Error accumulates non-locally; ledger cannot bound it. |
| **F2** | Silent intermediate dtype change not declared in the ledger | Invisible precision loss. |
| **F3** | Dropping `is_init=True` on a matmul that should initialize L0C | Produces stale accumulation from prior tile. |
| **F4** | Atomic on non-float dtype | Hardware restriction. |
| **F5** | Any rewrite that depends on a specific input distribution without that being stated in `assumptions` | Unsafe under distribution shift. |

## Usage

In every artifact, list primitives inline next to the sub-kernel or step that invokes them:

```
sub1.mmad: [P1, P5, L1]   # outer split on M, K-split with L0C accum, bf16 matmul
sub2.norm: [P2, L2]       # fused elementwise, fp16 intermediate
```

Forbidden primitives (tier F) are never silently invoked. If one is needed — for example, a reduction reorder beyond a tile — lift it to user approval first, record the justification in `plan.md` §7 Assumptions, and add an explicit ledger row.

## Plan Count Triggers

Use the table below only to decide whether more than one plan is worth writing.
Default to one plan. Add another plan only when the signal leads to a materially
different topology, precision boundary, workspace layout, or merge strategy.

| Signal in user's function | Required candidate axis | Primitives to consider | Rationale |
|---|---|---|---|
| Single matmul on cube-friendly dtype | cube-only baseline | P1, P5 | Establishes a ceiling; later candidates must justify deviation. |
| Matmul with K large vs M·N (e.g., K ≥ 4 × max(M,N)) | reduction-axis split | P5 + (L1 if `loose` or explicit tolerance) | L0C accumulation avoids an explicit vec-side reduce. |
| fp32 weights/activations into matmul under `loose` or explicit tolerance | precision-downcast | L1 (bf16 wide-range, fp16 bounded) | Cube throughput win; ledger required. |
| softmax / online-max / running-denom inside a matmul chain | algorithmic | A1 + downstream P5/L1 | Tile reduction across S without materializing full attention. |
| Multiple producers writing to one GM region | atomic merge | L5 (atomic_add or atomic_max/min with declared NaN/tie) | Skips a vec-side reduction kernel entirely. |
| Recurrent / autoregressive reduction across chunks | state bridge | A3 (+ atomic care) | See `agent/example/kernels/a5/gdn_legacy/delta_h_state_bridge_v1_c8.py`. |
| `A @ (B + C)` or `(A + B) @ C` shape | matmul distributivity | A4 (recorded in ledger) | Avoid materializing the sum on UB; only under non-`bit-exact`. |
| Long elementwise chain after a matmul | fusion | P2 + (L2 under `loose`) | Cuts GM round-trips. |
| Existing similar kernel found by selector / catalog lookup | reuse-with-adapter | (primitive set inherited from study kernel) | Reuse beats re-decomposition; if the match covers ≥ 60% of the function, propose a candidate that adapts it. |
| Memory-binding workspace under `recurrent` / `autoregressive` | recompute | A2 | Trade cube/vec for on-chip footprint; only when memory is the binding constraint. |

If two rows fire on the same function, merge them into one plan when they lead
to the same topology, precision policy, and workspace/merge strategy.

If `precision_policy.mode` is `strict` or `bit-exact`, drop every L-tier and A4 row; keep the rest.
