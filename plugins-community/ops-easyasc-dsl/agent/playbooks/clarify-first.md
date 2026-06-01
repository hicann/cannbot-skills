# Clarify-First Playbook

Use this file before you touch any new kernel or major kernel redesign.
Its job is to settle the exact contract before any DSL code is written.

## When this applies

- writing a new kernel
- replacing a kernel body in a major way
- redesigning a stage or pipeline during debugging

Skip this file only for trivial fixes that do not change the contract.

## Hard rule

You must fill in the Contract Template below before writing kernel code.
If any field has more than one plausible interpretation that the formula alone does not fix, STOP and ask the user.
Do not substitute "reasonable assumptions" for fields marked ASK.

If the request points to a PyTorch file, an existing kernel, or another indirect reference, read:
- `agent/references/contract-intake.md`

That page owns the intake rules, device-default rules, and anti-patterns.

## Contract Template

```text
Target formula:      <pytorch expression>
Input tensors:       name, dtype, shape, logical layout (row/col-major, KM/MK, etc.)
Output tensor:       name, dtype, shape
Scalar/aux tensors:  name, dtype, shape (include broadcast intent when ambiguous)
Reduction axes:      which axes are reduced
Cast boundaries:     input -> matmul_accum -> post -> output
Device:              a2 | a5
Topology:            cube-only | cube->vec | vec->cube | cube->vec->cube | ...
Tail behavior:       which axes may be non-aligned, at what alignment
Accumulation:        replace | atomic_add
```

## Per-field rule: ASK vs MAY ASSUME

| Field | Default handling |
|-------|------------------|
| input tensor shape when only a dtype is given | ASK |
| broadcast intent for any aux tensor (bias, mask, scale vector) | ASK |
| aux tensor staging (kernel-side broadcast vs caller-side pre-expand) | ASK |
| output dtype when different from the inputs | ASK |
| interior post-matmul cast boundary (for example `p.half()` before a later cube stage) | ASK |
| tail behavior (non-aligned axes) | ASK |
| topology for 2+ matmuls, softmax, or online reduction | ASK unless the user already named the loop / reduction order |
| accumulation semantics across repeated launches | ASK |
| device when a referenced kernel path already fixes it (`agent/example/kernels/a2/` or `agent/example/kernels/a5/`) | MAY ASSUME that device |
| `matmul` accumulation dtype | MAY ASSUME `float` |
| tile family, split thresholds (`>= 32`), and `L0C` row offset | MAY ASSUME repo defaults; document the choice |

## Standardized question block

Keep all clarifying questions in one upfront block:

```text
Before I write the kernel, I need to confirm N things that the formula alone does not fix:
  1. <field>: possibilities are (a)..., (b)..., (c).... Which matches your intent?
  2. <field>: ...
I will wait for your answer before starting.
```

Why this format:
- one handoff is cheaper than scattered follow-up questions
- listing `(a) / (b) / (c)` proves you enumerated the real interpretations
- "I will wait" makes the stop explicit instead of silently defaulting

## When no user answer is available

In automated runs where the user cannot reply in real time:
- still write the question block
- then pick the best-guess default for each unresolved field
- record each guess next to the question
- echo the same assumptions in the kernel file header comment

## After clarifications

Once every field is settled, continue with:
- `agent/playbooks/kernel-authoring.md`
