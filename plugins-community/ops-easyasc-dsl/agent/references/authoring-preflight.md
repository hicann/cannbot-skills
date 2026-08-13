# Kernel Authoring Preflight

Read this once before implementing or materially changing a kernel. Device
details still belong to `facts-device-runtime.md`; this page owns cross-cutting
authoring traps.

## AICPU opt-in gate

- AICPU authoring is disabled by default. Do not create, compile, package,
  dispatch, benchmark, or run a `.aicpu` kernel unless the user explicitly
  requests AICPU in the active request.
- Existing AICPU files, earlier conversation context, precision pressure,
  unsupported AI Core behavior, or an apparent performance opportunity do not
  grant that authorization.
- When the active request explicitly names AICPU, route through
  [`aicpu-kernel-authoring.md`](../playbooks/aicpu-kernel-authoring.md) and use
  [`aicpu-authoring.md`](aicpu-authoring.md) as the focused implementation
  reference. Otherwise remain on the selected EasyASC DSL workflow.

## Structure and public ABI

- A kernel is a function decorated with `@kernel`. User-visible inputs and
  outputs map strictly to its `GMTensor` arguments and returned `GMTensor`
  objects.
- Ordinary Python helpers called from `@kernel` or `@vf` must use `@func()` so
  the AST naming transform remains active.
- Import only the target facade: `easyasc.a2`, `easyasc.a3`, `easyasc.a5`, or `easyasc.a5pr`.
  Do not mix targets or import private vector stubs.
- Dispatch structural variants from plain Python into separate `@kernel` or
  `@vf` functions. Keep the traced kernel body structurally unambiguous.

## Conditions and loops

- Every normal Python `if`/`elif` in a decorated body is a runtime condition.
  The parser rejects a condition that traces to a plain `bool` or `int`; do not
  use `if FLAG`, `if True`, or dtype/config booleans for specialization.
- An `if` in a `@vf` becomes a hardware branch and can materially reduce vector
  throughput. Use separate branch-free VFs for structural or precision variants.
- `if` is forbidden inside `unroll()` loops. Hoist a genuine runtime decision
  outside the unroll.
- `range(..., name="X")` emits `X` as a C++ loop counter. It must not collide
  with a live Tensor, Var, Reg, RegList, MaskReg, buffer, or nested loop name.
  Parser validation rejects known collisions; still use descriptive counter
  names such as `arow`, `krow`, and `nblk`.
- Explicit low-level `with If(...)` remains supported for parser-level code, but
  it is not a specialization mechanism.

## Buffers, transfers, and synchronization

- Derive each buffer size and layout from the formula, tile, dtype, and target
  capacity. Do not inherit an allocation solely from another sample.
- A2 BT has two 256-byte slots (64 fp32/int32 elements per slot). A5/A5PR BT has
  two 2KB slots (512 elements per slot). Shortcut bias width must fit one slot.
- A2 `gm_to_ub_pad` and `ub_to_gm_pad` require `n_burst` in `[0, 4095]`;
  `0` is an empty transfer. The stub checks known values and the simulator checks
  the resolved runtime value.
- Synchronization follows producer completion, consumer readiness, and buffer
  reuse. A passing single-tile simulation is not a reuse proof.
- For a lifetime or synchronization fix, add a regression shape that forces the
  same active core to process more than one reused outer tile. For M-tiled
  attention, verify `ceil(BH * ceil(S1 / TILE_M) / core_count) > 1`.
- Treat `auto_sync` warnings as correctness signals. Resolve the ownership model
  or make a concrete parser proposal; do not waive them.

## Numerical contract

- Record where accumulation, casts, saturation, rounding, and lossy layout
  conversions occur. Validate against the golden at those boundaries.
- Before golden inputs enter a kernel, allow only `squeeze`, `unsqueeze`, and
  `reshape`. `expand`, `tile`, `permute`, and value-changing preprocessing alter
  the contract unless explicitly requested.
- A simulator pass proves modeled execution, not generated C++ compilation or
  real-device bit-exactness. State that limitation unless a board build/run also
  passed.

## Completion check

- public signature, placeholders, and `shape_bindings` agree;
- all tile and buffer bounds hold on the selected device;
- constant conditions and generated identifiers pass parser validation;
- baseline, tail, alignment-boundary, and same-core reuse shapes pass;
- warnings are explained and resolved;
- the owner catalog/reference and `agent/diary.md` are updated for kernel work.
