# reference_provider — Phase O2.5 reference generation

These scripts support deterministic reference inputs and outputs for
cross-generation operator migration and forward-to-backward generation. For
new migration tasks the authoritative reference is normally the frozen
KernelBench-style task/sidecar bundle (task `.py` + same-stem `.json`/`.jsonl` sidecar pair); the
legacy source-NPU provider is used only when durable state explicitly selects
`reference.source=a3_live`. The resulting artifacts are consumed by the kernel
worker's external-reference verification step.

## Main components

| File | Purpose |
|---|---|
| `input_gen.template.py` | Deterministic representative and edge-case inputs |
| `autograd_backward_reference.py` | CPU/fp64 truth for backward operators |
| `backward_input_gen.py` | Backward input and gradient-output generation |
| `case_gen.py` / `case_schema.py` | Shared case construction and validation |
| `reference_regen.py` | Rebuild a CPU PyTorch reference dataset |
| `verify.py` | Compare an AscendC candidate with staged reference outputs |

## Workflow

1. For `npubench`, stage the supplied task/sidecar bundle by digest and preflight
   it; for `a3_live`, generate deterministic
   source inputs and capture the current arch22 source on its NPU.
2. For backward generation, evaluate the supplied forward specification with
   CPU/fp64 autograd.
3. Record the selected reference source, integrity metadata, and staged
   artifacts with the task.
4. Validate the scoped reference through the selected migration or backward
   provider.
5. Use `verify.py` to compare the generated AscendC implementation with the
   selected staged truth under the declared precision contract.

Keep per-case tensors small enough for precision validation. Large production
shapes belong in the later performance stage, not in the truth dataset.
