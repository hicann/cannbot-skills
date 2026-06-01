# Kernel Authoring Playbook

Use this playbook when writing a new kernel or replacing a kernel body in a major way.
Keep the path short, tool-first, and incremental.

## Step 0 (prerequisite): settle the contract first

Before anything below, do:
- `agent/playbooks/clarify-first.md`

If the contract is still ambiguous, stop and ask the user before you start coding.

## Goal

Produce a kernel that:
- matches the exact PyTorch contract
- uses a justified topology
- stays within device, sync, and precision rules
- is validated stage by stage instead of guessed into existence

## 1. Fast path: prefer tools before large prose

For most new kernels, the cheapest route is:

1. shortlist examples with the selector tool
2. pick the topology and print a fresh scaffold
3. estimate tile/core split only when the shape is non-trivial
4. open one source example only after the selector or index narrowed it down

Recommended commands:

```bash
conda run -n torch210npu python agent/scripts/select_kernel_example.py --query "qk softmax pv" --topology 'cube->vec->cube->vec' --limit 3 --catalog
conda run -n torch210npu python agent/scripts/gen_kernel_skeleton.py --name preview_kernel --topology 'cube->vec' --print
conda run -n torch210npu python agent/scripts/estimate_matmul_datamove.py --help
```

If the selector tool is not enough, fall back to:
1. `agent/references/examples/kernel-index.md`
2. one matching `agent/references/examples/kernel-catalog.md` entry
3. one source file under `agent/example/kernels/`

## 2. Minimal read map by scenario

If you need raw values before choosing a deeper route, jump straight to one focused facts page:
- `agent/references/facts-device-runtime.md` for device caps, pipe pairs, and mutex signatures
- `agent/references/facts-authoring.md` for hard rules, DBuff formulas, and a2 bridge reminders
- `agent/references/facts-simulator-opexec.md` for simulator / `shape_bindings` / `OpExec` gotchas

Then branch to only one focused follow-up when possible.

| Situation | Read next | Open more only if... |
|-----------|-----------|----------------------|
| pure cube matmul or layout rewrite | `agent/references/patterns/cube-only.md` | split choice, unusual layout, or family tile is still unclear |
| a5 kernel with any vec-side stage | `agent/references/constraints/a5-device.md` | `@vf()` is not enough and you need `micro` / sort specifics |
| a2 cube -> vec | `agent/references/patterns/a2-cube-vec.md` | reduction, tail, or workspace behavior is still unclear |
| a2 cube -> vec -> cube | `agent/references/patterns/a2-cube-vec-cube.md` | delayed stage ownership still feels ambiguous |
| a2 normalized online softmax (`score -> p -> pv -> final divide`) | `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md` | you hit a special failure mode that page does not already cover |
| sync / counter warning | `agent/references/constraints/autosync.md` or `agent/references/constraints/counters.md` | the warning persists after you traced the producer / consumer lifetime |
| lowering or simulator behavior looks wrong | `agent/references/code-paths.md` | the problem is clearly in runtime / parser behavior, not the kernel math |

## 3. Authoring rules that still matter on every kernel

- Keep the file layout simple:
  1. imports and constants
  2. helper `@vf` / `@func` blocks
  3. main `@kernel`
  4. visible `__main__` validation story
- Prefer normal Python `if` / `elif` / `else` inside kernels; the repository rewrites them before instruction capture.
- On a5, vec-side math belongs in `@vf()` / `micro`; do not copy an a2-style direct vec body into an a5 kernel.
- Keep local buffers full-tile sized. Use `valid_m` / `valid_n` only at GM read/write boundaries unless a focused constraint file says otherwise.
- `auto_sync()` is same-side ordering only. Cross-side ownership changes still need `CvMutex` or `VcMutex`.
- Keep matmul accumulation in `float` unless the contract or an established family says otherwise.
- If scalar dimensions may alias at runtime, add `shape_bindings=` at the `OpExec(...)(...)` call site before rewriting the kernel body.

## 4. Implementation loop

Use the same small loop for every serious kernel:

1. build the smallest honest slice of the formula
2. validate one aligned case
3. add the next stage or tail path
4. validate again
5. only then optimize or fuse further

Practical rules:
- keep the PyTorch reference inline in `__main__`
- keep one aligned case and one tail case visible
- for large fused kernels, keep standalone stage runners alive until the merged version passes
- study examples for legal patterns, not for copy-paste

## 5. When to leave the fast path

Open lower-level implementation files only when the smaller guidance layers stop being enough.
Typical escalation path:

1. `easyasc/stub_functions/`
2. `easyasc/parser/`
3. `easyasc/simulator_v2/`

Do not treat passing output with unresolved warnings as good enough.
Warnings usually mean the model is incomplete.
