# NPU validation for generated arch22→arch35 and backward operators

This runbook validates code generated in the current task. Target archives, prior-art implementations,
and pre-staged target branches may inform generation and research, but they must be recorded as such:
they must not be copied verbatim into a deliverable and declared successful, do not replace the task's
build provenance, and must never be promoted to migration or backward truth. A pre-staged branch is
reusable only when it is part of the selected arch22 source and remains traceable to that source contract.

## Truth boundary

- Migration default truth: declared operator contract plus the immutable, content-addressed
  KernelBench-style task/sidecar bundle (task `.py` + same-stem `.json`/`.jsonl` sidecar pair; `reference.source=npubench`); durable state must bind the bundle,
  fixture, and provider-owned verification evidence by digest.
- Migration `a3_live` truth: declared operator contract plus a capture produced during this run
  from the current user-selected arch22 source on the source-arch NPU. It is an explicit fallback,
  not a silent substitute for a missing npubench bundle.
- Legacy `a3_live` source-capture wording: Migration ordinary truth: source-arch NPU. Here
  “ordinary” is limited to this selected source-capture provider; an explicitly selected
  npubench task/sidecar bundle remains its own immutable golden contract.
- Backward truth: declared forward contract, saved-tensor contract, analytic gradient equations,
  and a CPU fp64 autograd oracle.
- Target NPU execution validates the newly generated target; it is never an answer source.
- Public target API and hardware documentation plus independently authored probes may establish
  target capabilities.

## Scope: source and target lanes are re-proved independently

A gate proven on one architecture is methodology evidence only. Re-run build, launch, memory, and
dispatch probes on every supported architecture, compiler, runtime, dtype, and launch mode.

## Cardinal rule: current generated source must match the tested binary

Record digests of the selected arch22 source, declared schema, generated source tree, build
manifest, device object, and loaded extension. A result is admissible only when the loaded binary
is traced to those current generated inputs. Delete generated build outputs, rebuild cleanly, and
repeat the digest check when provenance is uncertain.

Installed target source, objects, metadata, dispatch records, and archived experiments may be
consulted as prior-art. Keep that evidence separate from the selected-source contract, the
current-generated-binary digest chain, and the truth authority used for final validation.

## Verification decision order

1. Validate contract and truth provenance.
2. Clean-build the current generated source and prove the loaded-binary digest.
3. Run a minimal target public-API launch probe.
4. Run memory and exception controls.
5. Confirm generated dispatch coverage.
6. Compare results with the selected migration reference (default frozen KernelBench-style task/sidecar bundle or explicit
   source-arch capture) or CPU fp64 autograd truth.
7. Exercise option and attribute branches.
8. Run determinism and performance gates only after correctness.

## R0: clean generated-source build and provenance proof

Build in an empty task-owned directory from the current generated source and manifest. Change a
task-owned probe entry to emit a distinct sentinel, rebuild, and confirm the sentinel changes;
then restore the generated source and rebuild again. This proves the test loads this task's binary
without modifying or overlaying any installed operator.

## R1: GM canary for out-of-bounds writes

Back an output or workspace with task-owned guard regions before and after the declared extent.
Initialize unique sentinels, launch the generated kernel, synchronize, and verify both guards.
Test aligned, tail, empty, and minimum shapes. Guard storage must not change the logical tensor
contract.

## R2: controlled device-exception probe

Run a valid control and a contract-invalid trigger through the same generated entry. Record launch
and synchronize status separately. Use only task-owned allocations. An exception is meaningful
only if the control succeeds and provenance proves the current generated binary ran.

## R3: value comparison with admissible truth

For default migration, compare with the immutable npubench bundle evidence and its declared tolerance;
for explicit `a3_live`, compare with the same current arch22 source-arch capture and declared tolerance.
For backward generation, compare every gradient with CPU fp64 autograd after normalizing to the
declared output dtype. Include near-zero, overflow, reduction, non-contiguous, optional-input, and
saved-state cases.

## R4: generated dispatch confirmation

Create at least one case for every source-derived dtype, shape, layout, attribute, optional-input,
and backward branch. Instrument only the current generated code with task-owned branch counters or
sentinels. Remove instrumentation before the final clean build.

## R5: attribute and option oracle

Hold tensors fixed and vary one declared attribute or optional input at a time. Predict behavior
from the contract and CPU/source oracle, then confirm the generated output. No-change is correct
only when the declared semantics say the option is inert for that case.

## Environment and runtime gotchas

Before diagnosing kernel code, prove the authorized lane can initialize the runtime, allocate,
launch a minimal public-API kernel, and synchronize. Record compiler, runtime, driver, Python ABI,
framework package, device architecture, and container identity. Keep credentials, host addresses,
and private paths outside the knowledge base.
