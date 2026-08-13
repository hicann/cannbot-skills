# PyTorch or Contract to One EasyASC Kernel

Use this route for every new single-runtime-kernel request that starts from
PyTorch, ONNX, a golden, a formula, a settled contract, or a reference kernel.
Do not open a generic authoring overview or clarification template first.

Do not use this route when the user explicitly asks for multiple runtime kernels,
or when the task is debugging/optimizing an existing kernel.

## Fixed outcome

Deliver one production `@kernel` and one production `OpExec`.

The host may inspect shapes, allocate outputs, apply shape-only views, dispatch,
and return outputs. Formula arithmetic, reductions, normalizers, activations,
broadcast materialization, and other contract semantics stay in the kernel.

Probe kernels under `tmp/<task>/` may test one uncertain primitive. They are evidence,
not extra production stages. If one kernel is infeasible within budget, preserve
the evidence and report the boundary; do not silently change the contract.

## Start before implementation

1. Read `agent/references/authoring-preflight.md` and one target section of
   `agent/references/facts-device-runtime.md`.
2. Read the source contract end to end. Infer formula, shapes, dtypes,
   broadcasts, reductions, casts, device, and the shape domain.
3. Record the formula stages and likely dataflow needs, such as row reduction,
   runtime broadcast, tail-safe elementwise work, or a mixed cube/vec bridge.
   Do not open pattern or example documents yet; query them in the work loop
   only when a focused match is needed.
4. Create one short `tmp/<task>/workflow.json` before editing a
   candidate. Keep only:

   ```json
   {
     "route": "single_kernel",
     "phase": "contract",
     "contract": {
       "formula": "...",
       "inputs_outputs": "...",
       "device_mode": "...",
       "shape_domain": "...",
       "patterns": ["..."],
       "delivery": "one @kernel / one OpExec",
       "host": "shape, allocation, dispatch only",
       "semantic_coverage": ["formula stage -> kernel owner"]
     },
     "candidate": {"path": null, "sha256": null, "count": 0},
     "boundary": {"name": null, "fixes": 0, "evidence": null},
     "passed": [],
     "next": "select one reference or probe the least-proven boundary"
   }
   ```

5. Ask only when a missing fact would change semantics and cannot be recovered
   from the source, examples, repository defaults, or task environment. Use
   `agent/references/clarification-template.md` only for that unresolved
   question.

## Work loop

1. Select one focused reference with `agent/scripts/select_kernel_example.py`, the
   pattern index, or the example index. Prefer `--pattern <id>` when the
   dataflow is known. Open only the selected pattern, one primary runnable
   example, and the constraint sections that pattern names.
2. For nontrivial vec storage, record `logical shape | instruction footprint |
   physical allocation` before implementation. Do not infer a physical UB
   allocation from the logical live-lane count alone.
3. Identify the least-proven DSL/hardware boundary. If no repository example
   proves it, make the smallest aligned plus tail/multi-tile probe first.
4. If no pattern fits, evidence conflicts, or a capability-absence claim is
   forming, follow `agent/references/evidence-escalation.md`. Return with one
   source-backed invariant or minimal probe instead of loading source broadly.
5. Let one parent own the candidate. Scouts are optional and read-only; use one
   only when a separate evidence question would otherwise block the parent.
6. Implement the smallest honest end-to-end candidate and validate it before
   optimizing. Archive/hash before replacing its architecture.
7. Update `workflow.json` and the runtime checkpoint before a long task,
   candidate replacement, pause, or context compaction. Store full source and
   logs on disk, not in the checkpoint.

Default budget: three materially different candidates and at most two fixes at
the same failing boundary. The counters in `workflow.json` are authoritative.

## Claim gate

Run in this order:

1. map every formula stage to the final kernel body;
2. audit the module for exactly one production `@kernel`, one production
   `OpExec`, and no host tensor math/readback;
3. run self-generated aligned, tail, multi-core, and multi-tile simulator cases
   required by the inferred shape domain;
4. run the unfiltered public simulator evaluator;
5. run required hardware correctness;
6. preserve candidate SHA-256, exact commands/exit codes, evaluator JSON, and
   log paths.

A public numerical pass does not override the delivery or shape-domain gates.
Candidate failure also does not prove a DSL capability gap. Any such claim must
pass the capability-absence gate in `agent/references/evidence-escalation.md`.

## Stop and report

Stop when all gates pass or the bounded search is exhausted. Report the inferred
contract, references with path/line evidence, candidate count/path/hash, exact
validation evidence, and the first unresolved boundary. Never relabel
"one Python entry point" as "one runtime kernel".
