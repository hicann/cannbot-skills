# Dataflow Pattern Index

Use this index after choosing a workflow and deriving the kernel contract. A
pattern explains how several DSL primitives compose into one reusable dataflow;
it does not replace primitive constraints or a runnable kernel example.

Default read order:

1. select the smallest matching pattern below;
2. open one runnable reference named by that pattern;
3. open only the constraint sections linked by the pattern;
4. enter `evidence-escalation.md` when no pattern fits or repository evidence
   conflicts with the pattern.

## Vec-only dataflow patterns

| Pattern id | Read | Use when | Primary runnable reference |
| --- | --- | --- | --- |
| `vec-row-reduce-broadcast` | `patterns/vec-row-reduce-broadcast.md` | one full row produces one runtime tensor scalar that is transformed and reused across the row | `agent/example/kernels/a2/vec_only/rowwise_reduce_broadcast.py` |
| `vec-group-reduce-broadcast` | `patterns/vec-group-reduce-broadcast.md` | each fixed-size group produces its own scalar and reuses it within that group | `agent/example/kernels/a2/vec_only/group32_stride4_wide128_probe.py` |
| `a5-fp4-cast-pack` | `patterns/a5-fp4-cast-pack.md` | native BF16-to-FP4 register cast must become a dense packed uint8 public carrier | `agent/example/kernels/a5/vec_only/bf16_to_fp4_e1m2.py` |
| `a5-uint2-pack-unpack` | `patterns/a5-uint2-pack-unpack.md` | exact `{0,1,2,3}` BF16 values cross a four-per-byte uint2 ABI | `agent/example/kernels/a5/vec_only/bf16_to_uint2.py` |
| `online-softmax-tail` | `patterns/online-softmax-tail.md` | tiled online softmax must place S1/S2/causal masks before the affected max, exp, or sum stage | `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` |

## Pipeline-topology patterns

| Pattern id | Read | Use when |
| --- | --- | --- |
| `cube-only` | `patterns/cube-only.md` | the formula stays on cube without a vec stage |
| `a2-mixed-pipeline` | `patterns/a2-mixed-pipeline.md` | an A2 kernel crosses cube and vec through supported staged bridges |
| `a5-mixed-pipeline` | `patterns/a5-mixed-pipeline.md` | an A5 kernel crosses cube and vec or uses a lookahead pipeline |
| `lookahead-drain` | `patterns/lookahead-drain.md` | producer and delayed consumers overlap across warmup, steady-state, and drain beats |
| `buffer-slot-lifetime` | `patterns/buffer-slot-lifetime.md` | delayed beats or overlapping local roles require explicit physical slot separation |

`agent/references/optimization/levers.md` belongs to the optimization workflow. Do not
load it while selecting a build-kernel dataflow pattern.

The A2/A5 topology pages contain several numbered dataflow families by design.
Read only the matching section; their shared entry, invariant, exit, and source
escape sections follow the page contract below.

## Pattern page contract

New dataflow patterns should stay small and use these sections:

- `Applies when`
- `Logical dataflow`
- `Physical invariants`
- `Minimal skeleton`
- `Failure signatures`
- `Runnable references`
- `Do not use when`
- `Source escape`

Keep primitive semantics in `constraints/`, full implementations in `agent/example/kernels/`,
and cross-route symptom routing in `pitfall-records.md`. Link the actual owner
instead of copying its full content into a pattern.

## Selecting examples by pattern

Catalog entries may carry a `patterns` list. Filter by it when the dataflow is
known:

```bash
python agent/scripts/select_kernel_example.py \
  --pattern vec-row-reduce-broadcast \
  --topology vec-only \
  --limit 3 --catalog
```

If no entry matches, do not force a nearby pattern. Follow
`agent/references/evidence-escalation.md` and return with a source-backed
invariant or minimal probe.
