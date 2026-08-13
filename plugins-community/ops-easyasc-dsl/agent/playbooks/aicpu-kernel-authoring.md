# Device-side AICPU Kernel Authoring

Use this workflow only when the user explicitly requests AICPU, an AI CPU
kernel, or a `.aicpu` implementation in the active request. AICPU is an opt-in
backend: an existing AICPU file, a precision failure, an unsupported AI Core
operation, or a possible speedup does not activate this route.

If the active request does not explicitly authorize AICPU, stop this workflow
and return to the Router-selected EasyASC DSL workflow. Do not create, compile,
package, dispatch, benchmark, or run an AICPU candidate speculatively.

## Required evidence

Before implementation:

1. Read [`authoring-preflight.md`](../references/authoring-preflight.md) in full.
2. Read [`aicpu-authoring.md`](../references/aicpu-authoring.md) in full.
3. Read the source contract end to end and record public inputs, outputs,
   dtypes, shapes, attributes, numerical boundaries, and target SoC.
4. Inspect the target CANN installation for the AICPU compiler module, headers,
   libraries, and supported architecture. Documentation for another SoC is not
   sufficient evidence.
5. For a hardware run, use the task-scoped access path in the git-ignored
   `machine_specs.md`, inspect device occupancy first, and never interfere with
   another process.

## Choose the delivery boundary

Select one topology before writing code:

- **AICPU-only operator:** the AICPU kernel owns all contract computation.
- **AICPU fallback:** public dtype/shape metadata selects either the existing
  AI Core implementation or a complete AICPU implementation.
- **Mixed AI Core and AICPU pipeline:** each engine owns an explicit stage,
  intermediate tensors live in GM, and cross-stream dependencies are part of
  the contract.

Do not hide multiple runtime launches behind a single-kernel claim. A mixed
pipeline is a multi-launch implementation even when it has one Python entry.
Dispatch may depend only on public runtime metadata, never case ids, filenames,
tensor values, seeds, or golden outputs.

## Implement incrementally

1. Start with one `.aicpu` source and one minimal `.asc` launcher. Use the ABI
   and build forms from the focused reference instead of extending the EasyASC
   DSL.
2. Validate argument layout and a trivial GM write before implementing the
   formula.
3. Add one numerical stage at a time. Record every cast, accumulation dtype,
   activation, state boundary, and final output conversion.
4. When internal `double` is requested, keep the public tensor dtype unchanged
   unless the contract itself exposes `DT_DOUBLE`. Test both the high-precision
   computation and the required output rounding boundary.
5. For a mixed pipeline, validate each engine separately before adding
   cross-stream events. Bridge completion back to the caller's stream before
   returning from an ACLNN/plugin entry.
6. Append tracked production-kernel changes to `agent/diary.md` while making
   them. Temporary task-local probes remain under `tmp/<task>/`.

## Build and integration

- Prefer the installed CANN AICPU CMake language support. It owns the AArch64
  sysroot, Device libraries, and compiler flags that a plain C++ build lacks.
- Keep the `.aicpu` implementation outside EasyASC `@kernel` syntax. EasyASC
  parser, simulator, and codegen changes are not required for a manually
  integrated AICPU path unless the user separately requests first-class DSL
  support.
- For an ACLNN or CANN Bench operator, update the smallest owning CMake target,
  launcher/plugin, operator metadata, and package manifest. Do not add unrelated
  operators or dependencies.
- Follow the official separate-stream contract for mixed AI Core/AICPU launches.
  A runtime accepting both on one stream is not portable evidence that the
  combination is supported.

## Validation gate

Run, in order:

1. compile the `.aicpu` Device source and `.asc` launcher for the exact SoC;
2. load and execute a trivial real-device GM write;
3. validate every public output's shape, dtype, value, NaN/Inf behavior, and
   deterministic repeatability;
4. validate aligned, tail, empty/optional, and maximum supported shapes;
5. for mixed execution, validate both engine outputs, event ordering, memory
   visibility, caller-stream completion, and repeated invocation;
6. run the complete affected evaluator without changing its golden or
   thresholds;
7. measure performance only on real hardware, with warmup and stable repeated
   statistics; and
8. inspect profiler timelines for actual AICPU/AI Core execution and overlap.

The Python EasyASC simulator does not execute AICPU code and cannot provide
AICPU performance evidence. Report unsupported toolchains, runtime timeouts,
and unmeasured performance explicitly.

## Stop and report

Report the explicit AICPU authorization, target SoC/CANN version, delivery
topology, source and binary owners, public dispatch rule, numerical boundaries,
stream/event design, correctness results, profiler source, latency statistics,
and remaining portability risks. Do not generalize one accepted runtime behavior
into a cross-version AICPU contract.
