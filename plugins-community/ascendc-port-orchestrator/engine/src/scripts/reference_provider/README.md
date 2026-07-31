# reference_provider — Phase O2.5 reference generation

These scripts build deterministic PyTorch/CPU reference inputs and outputs for
cross-generation operator migration and forward-to-backward generation. The
resulting artifacts are consumed by the kernel worker's external-reference
verification step.

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

1. Generate deterministic inputs from the operator specification.
2. Evaluate the PyTorch forward or autograd backward reference on CPU.
3. Record integrity metadata and stage the reference dataset with the task.
4. Validate the scoped reference through the migration or backward provider.
5. Use `verify.py` to compare the generated AscendC implementation with the
   staged truth under the declared precision contract.

Keep per-case tensors small enough for precision validation. Large production
shapes belong in the later performance stage, not in the truth dataset.
