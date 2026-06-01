# Contract Intake Reference

Use this file when `agent/playbooks/clarify-first.md` needs help extracting the contract from a reference instead of a direct user spec.
This page owns the intake rules, device-default rules, and contract anti-patterns.

## Input forms

### Form A: structured spec in the user message

If the user already states formula, shapes, dtypes, device, or topology intent directly in text:
- go straight back to `clarify-first.md`
- fill the Contract Template

### Form B: pointer to a PyTorch function file

Examples:
- `"实现 /path/to/ref.py 里的 fn"`
- `"基于 a5 实现 xxx.py 的函数"`

Intake rule:
1. Read the reference file end-to-end. Do not skim the `__main__` block.
2. Extract what the file actually fixes: formula, shapes, dtypes, scalar values, and any device hints.
3. Verify PyTorch type promotion when Python scalars are involved. For example, `half_tensor * 0.5` may promote to `float32` while `half_tensor * half_tensor` stays `half`.
4. Fill the Contract Template.
5. ASK on fields the file does not fix, especially device, broadcast intent, topology for multi-stage flows, and tail behavior.

### Form C: pointer to an existing kernel file

Examples:
- `"参考 kernels/a5/foo.py 帮我写个 kernel"`
- `"根据 xxx.py 写一个"` where `xxx.py` already contains `@kernel`

Interpretation rule:
1. The default meaning is "reproduce the reference in a new file".
2. Never invent a new computation the user did not describe.
3. If the user described an explicit change, apply exactly that change on top of the reference contract.
4. Treat the reference kernel's device, topology, and tile shape as evidence of what is legal, not as mandatory copy targets.

### Form D: verbal description only

Use the Contract Template directly and ASK on every unresolved field.

## Device when unspecified

When the user did not name a device and no direct text fixes one:
- if the referenced kernel lives under `agent/example/kernels/a2/`, target a2
- if it lives under `agent/example/kernels/a5/`, target a5
- if the reference is a PyTorch file with no NPU hint, ASK

Do not silently default to a5.

## Reference-derived rules that are easy to miss

- Example shapes in `__main__` are part of the contract evidence.
- A shape that happens to be aligned does not prove the kernel may ignore tails.
- If the formula has 2+ matmuls, softmax, or online reduction, topology is part of the contract, not a late implementation detail.
- If the value path intentionally casts before a later cube stage, that cast boundary is semantic and must be confirmed.

## Anti-patterns

- filling the Contract Template silently and moving on
- inventing a new formula when the user only said "参考 xxx.py"
- assuming output dtype equals input dtype without checking Python-scalar promotion
- treating an aligned example shape as permission to skip tail reasoning
- silently defaulting to a5 when the reference file does not fix the device
