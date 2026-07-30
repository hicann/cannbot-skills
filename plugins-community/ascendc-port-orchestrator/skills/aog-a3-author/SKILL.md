---
name: aog-a3-author
description: >
  Generate per-op `run_a3_reference.py` + `input_gen.py` + `manifest.json` for
  an arch22→arch35 migration workspace by parsing the upstream ops-nn op directory's
  `examples/test_aclnn_{op}.cpp` (aclnn signature) and `op_host/{op}_proto.cpp`
  (output shape inference). Eliminates the per-op manual-authoring step that
  blocks `phase_o25_a3_ref` from running on new ops.
  Use when Phase O2.4 needs to construct a missing A3 reference runner.
  Usage:
    /aog-a3-author {op_dir} {workspace}     # parse op_dir, emit scripts to workspace
---

# /aog-a3-author


> **Anti-pressure protocols (MANDATORY, load first)**: read `${CLAUDE_PLUGIN_ROOT}/kb/shared/ANTI_PRESSURE_PROTOCOLS.md` before any decision. P1–P8 are LLM-pressure drift modes that override technical rules under load.

## When to use

- Phase O2.4 (orchestrator-managed) — automatically invoked by `phase_o25_a3_ref` when:
  - `opgen_mode == port_a3_to_a5` AND
  - `workspace/<op>/run_a3_reference.py` is missing
- Manually for a fresh arch22→arch35 migration op when the upstream provides `examples/test_aclnn_<op>.cpp`

## When NOT to use

- Any non-`port_a3_to_a5` mode
- When `run_a3_reference.py` already exists (skill is idempotent — refuses to overwrite)
- For ops without `examples/test_aclnn_<op>.cpp` (no upstream aclnn reference → can't extract signature)

## Inputs

- **`op_dir`** — absolute path to ops-nn op directory containing `examples/test_aclnn_<op>.cpp` + `op_host/<op>_proto.cpp`
- **`workspace`** — absolute path to `workspace/<op>/` where output scripts land

## Outputs (written to `<workspace>/`)

1. **`run_a3_reference.py`** — calls torch_npu API on A3 NPU per case, captures outputs + per-case timings. Conforms to invariants:
   - `from __future__ import annotations` + import torch / torch_npu / time / json / pathlib
   - `def main():` entrypoint
   - Reads `_HERE / "edge_inputs.pt"` (list[dict] with `case_id`, `inputs`, `attrs`)
   - For each case: `torch.npu.set_device(0)`, transfer inputs to npu, call relevant `torch_npu.<op>` or `torch.<op>` API mapped from aclnn signature, time with `torch.npu.synchronize()` + `time.perf_counter()`, median of 3
   - Writes `edge_dataset.pt` = list of `{...case, "a3_outputs": <tensor or dict>}` + `a3_baseline_perf.json` = `{case_id_str: median_ms}`
   - `if __name__ == "__main__": main()`

2. **`input_gen.py`** — generates edge_inputs.pt fixture covering dtype × shape × edge cases. Invariants:
   - Uses `case_gen` at `sign_off` coverage with one source-supported representative rank by default
   - Uses `SCHEMA["ranks"]` only when the admitted arch22 source proves that the interface is rank-polymorphic
   - Each case is `{"case_id": int, "inputs": {<tensor_name>: torch.Tensor}, "attrs": {<scalar_name>: value}}`
   - Writes `edge_inputs.pt` (torch.save) + `manifest.json` with case count + per-case meta
   - Keeps each case's combined tensor payload at or below 100 MiB by default; reduce
     representative shapes or rank variants if the canonical pre-allocation guard rejects it
   - Keeps the complete generated tensor payload at or below 1 GiB before hashing, saving,
     or source-NPU staging; reduce the source-supported representative case set if rejected
   - `if __name__ == "__main__": main()`

3. **`manifest.json`** — emitted by `input_gen.py` at run, containing case metadata for later cross-reference.

## Hard invariants — must hold or skill returns FAIL

- Both produced .py files MUST parse with `ast.parse` (skill validates before write)
- Both MUST contain `if __name__ == "__main__": main()` block
- Both MUST define `def main()` function
- `run_a3_reference.py` MUST mention `torch_npu` import + `torch.npu.set_device(` + `torch.save` for `edge_dataset.pt`
- `input_gen.py` MUST `torch.save` to `edge_inputs.pt` + write `manifest.json`
- `input_gen.py` MUST retain executable canonical `MAX_CASE_TENSOR_BYTES` (100 MiB)
  and `MAX_DATASET_TENSOR_BYTES` (1 GiB) guards; comments do not satisfy either gate

## Output validation gates (skill-side; before declaring DONE)

1. AST parse both Python files
2. Executable AST checks for the byte-budget constants, `generate_cases(...)` call,
   and raising single-case/dataset limit gates
3. (Optional, if locally runnable) `python3 -c "import ast; ast.parse(open('input_gen.py').read())"` smoke

If any gate fails: skill emits `.a3_author_FAILED-<run_id>` marker with failure reason + stays out — does NOT write partial scripts.

## Idempotence

- If both `run_a3_reference.py` and `input_gen.py` already exist + parse cleanly, skill returns NO-OP with verdict "ALREADY_AUTHORED"
- Never overwrites existing scripts (caller can delete them to force re-author)

## Parser strategy (LLM-assisted, not regex-only)

The aclnn signature in `examples/test_aclnn_<op>.cpp` varies:
- Some ops use `CreateAclTensor` (tensor inputs via C++ vector) — easy
- Some use `aclIntArray` / `aclFloatArray` (scalar arrays) — different creation pattern
- Some have output tensors created blank + filled by the kernel — need to read proto.cpp for shape
- Scalar params can be `const float lr = 0.1` or function arg directly

LLM (claude --print) reads BOTH `test_aclnn_<op>.cpp` + `<op>_proto.cpp` and emits the two Python files. The validation gates above catch malformed output.

## Two emission paths (selected automatically by the LLM)

### Path A — torch_npu Python wrapper (preferred, ~70% of ops)

Used when `torch_npu.npu_<op>` (or `torch_npu._C._npu_<op>` / `torch.ops.npu.<op>`) exposes the op. The runner directly calls the torch wrapper:

```python
result = torch_npu.npu_<op>(input1, input2, scalar_param=...)
```

Simple, fast, no extra build step.

### Path B — cpp-binary wrapper (fallback, for ops with no torch wrapper)

Used when the LLM determines all three torch_npu paths are absent (e.g. `apply_adam_w_quant` 2026-05-13). The skill emits FIVE artifacts instead of two:

1. **`input_gen.py`** — same as Path A (writes edge_inputs.pt + manifest.json)
2. **`run_a3_reference.py`** — subprocess-style; calls the compiled binary + parses output binary files into torch tensors
3. **`<op>_runner.cpp`** — modified copy of upstream `examples/test_aclnn_<op>.cpp`:
   - Replaces hard-coded `std::vector<float> input...HostData = {...}` with `aclrtMemcpy` from per-case `.bin` files written by `run_a3_reference.py` (which torch.tensor + tensor.numpy().tofile() converts edge_inputs.pt cases to .bin)
   - Writes output device buffers back to `.bin` files after `aclrtSynchronizeStream`
   - Adds per-case timing via `aclrtSynchronizeStream` + `std::chrono::high_resolution_clock`
4. **`CMakeLists.txt`** — minimal CMake to build `<op>_runner` linking:
   - the runtime CANN libraries `ascendcl`, `nnopbase`, and `opapi`
   - the configured CANN runtime `lib64` directory; do not add driver/devlib-only libraries
   - C++17 standard
5. **`build_runner.sh`** — one-line script: `cmake -S . -B build && cmake --build build --clean-first`.
   The clean-first build prevents a corrected CMake file from reusing a stale link command. The orchestrator's
   `phase_o25_a3_ref` invokes this before exec.

Path-B C++ helpers must not define generic macros such as `CHECK_RET` because CANN headers
already define them. Prefer ordinary functions, or use an op-specific macro prefix such as
`<OP>_CHECK_RET` whose argument handling is valid for the call sites.

run_a3_reference.py main loop in Path B:
```python
for case in cases:
    # Write inputs to .bin (per-case dir to allow parallel ops)
    case_dir = workspace / f"case_{case['case_id']}"
    case_dir.mkdir(exist_ok=True)
    for name, tensor in case["inputs"].items():
        tensor.contiguous().cpu().numpy().tofile(case_dir / f"{name}.bin")
    # Run the compiled binary
    subprocess.run(["./build/<op>_runner", str(case_dir)], check=True)
    # Load outputs back
    outputs = {name: torch.from_numpy(numpy.fromfile(case_dir / f"out_{name}.bin",
              dtype=...)) for name in <output_names>}
```

**Path B selection signals** (the LLM checks during emission):
- `aclnnXXX` function name doesn't match any `torch_npu.npu_*` pattern after snake_case mapping
- The op_dir's `op_host/op_api/<op>.cpp` doesn't expose a torch-side aten/_npu/_aten/_ops registration
- Skill's existing `run_a3_reference.py` would have to fail-loud at the 3-fallback wrapper check

When in doubt: skill emits **both Path A and Path B artifacts** + run_a3_reference.py tries Path A first, falls through to Path B subprocess. Validation gates accept either set of artifacts as AUTHORED.

## Failure modes + recovery

| Verdict | Meaning | Recovery |
|---|---|---|
| ALREADY_AUTHORED | both files exist + parse | skip, return success |
| MISSING_CPP | `examples/test_aclnn_<op>.cpp` not in op_dir | op needs manual authoring; not a port_a3_to_a5 fit |
| PARSE_FAILED | LLM-emitted script doesn't parse | retry once with the error text in prompt; second failure → FAIL |
| INVARIANT_FAILED | script parses but missing required imports/symbols | retry once with specific gate failure in prompt; second failure → FAIL |
| LLM_TIMEOUT | claude --print took > 300s | FAIL |
| LLM_PROXY_ERROR | Huawei proxy error from subprocess | FAIL (caller can refresh proxy + retry) |

## References

- Trigger gate: [engine/src/scripts/orchestrator/phase_o25_a3_ref.py](../../engine/src/scripts/orchestrator/phase_o25_a3_ref.py) → `RUNNER_MISSING` branch
- Subprocess wrapper: [engine/src/scripts/orchestrator/a3_author_invoke.py](../../engine/src/scripts/orchestrator/a3_author_invoke.py)
- Golden references (existing pre-authored scripts that newly-generated should structurally match):
  - `workspace/ctc_loss_v3/run_a3_reference.py` + `workspace/ctc_loss_v3/input_gen.py`
  - `workspace/gather_elements_v2/run_a3_reference.py` + `workspace/gather_elements_v2/input_gen.py`
- ROADMAP §1.5 — arch22→arch35 migration mode
