# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""port_from_a3_ascendc Phase A-E orchestrator brief (W5).

Extracted verbatim from kw_brief.py (DEBT-201 god-file decomposition,
2026-07-06). `_port_a3_phase_instructions_block` composes the full port_a3
worker brief; the Phase-D (D.1/D.2/D.3), Phase-E and iter-budget body builders
live here, while the Phase-A/B/C bodies + context are imported one-way from the
sibling leaf `kw_brief_pa3_phases`. The forced-architecture-honor prefix comes
from the shared leaf `kw_brief_shared`.

The parent `kw_brief` re-imports `_port_a3_phase_instructions_block` so
`from briefs.kw_brief import _port_a3_phase_instructions_block` keeps working
(golden-locked by test_kw_brief_port_a3_golden.py).

Behavior is BYTE-IDENTICAL to the pre-split functions (prompt-template refactor).
"""
from __future__ import annotations

from pathlib import Path

from briefs._common import AscendCEnv
from briefs.kw_brief_shared import _forced_architecture_block
from briefs.kw_brief_pa3_phases import (
    _pa3_context,
    _pa3_phase_a_1,
    _pa3_phase_a_2,
    _pa3_phase_a_3,
    _pa3_phase_b,
    _pa3_phase_c,
)
from reference_source import uses_npubench_reference


def _workspace_uses_npubench_reference(workspace: Path) -> bool:
    """Return whether the brief must use a frozen old-format task contract."""
    if not (Path(workspace) / ".opgen_state.json").exists():
        return False
    return uses_npubench_reference(workspace)


_TILELANG2ASCENDC_SOURCE_BOUNDARY = """## TileLang2AscendC source boundary — implementation context only

`source_kind` is the persisted **`port-aclnn-tilelang2ascendc`** route. The
immutable TileLang2AscendC project mounted read-only at `{source_stage}` may be
read for operator shape and kernel-generation context. It is never functional
truth; the frozen NPUKernelBench bundle remains the only oracle.

Author an independent candidate in the same project format: a
`model_new_ascendc.py` entry point and a `kernel/` tree containing
`CMakeLists.txt`, `register.cpp`, `op_host/`, and `op_kernel/` (plus any
project headers or setup metadata required by that recipe). The model must
call the newly registered `torch.ops.npu.<op>` custom op. `kernel/pybind11.cpp`
is not required and is not the target ABI for this route. Do not byte-copy a
staged source file, reference `{source_stage}` at runtime, or call ACLNN,
`torch.ops.aten`, `torch.nn.functional`, NumPy, or a CPU fallback. The graybox
has no CANN compiler/toolchain authority; do not compile or run a build script.
Missing toolchain pieces in the sandbox are expected — never an infra
violation, and never a `build stuck` / `INFRA_BASELINE_VIOLATED` handoff.
The controlled target runner owns the authored CMake build and frozen
NPUKernelBench evaluation after handoff. The target gate requires registered
`TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` evidence, authored host launch evidence,
and a real AscendC device TU with `__global__`, `__aicore__`, and `AscendC::`.
`kernel/register.cpp` must also carry a `PYBIND11_MODULE(_<op>_ext, m)` entry
point whose module name matches the CMake `OUTPUT_NAME` of the built extension;
the module body may stay empty (`m.doc()` only) because `TORCH_LIBRARY` static
registration runs at load, but without it the built `_<op>_ext<EXT_SUFFIX>.so`
exports no `PyInit__<op>_ext` and the evaluator's `import _<op>_ext` fails.
`model_new_ascendc.py` must bootstrap the import path itself before importing
the extension, e.g.
`sys.path.insert(0, str(Path(__file__).resolve().parent / "kernel" / "build"))`,
because the evaluator stage does not add `kernel/build` to `sys.path`.
In `kernel/op_kernel/`, do not include `<algorithm>` or call host STL
`std::min`/`std::max` from an AICore function; use device-safe scalar logic.

TARGET ARCHITECTURE IS aarch64 (2026-08-21 batch rule): several staged
source CMakeLists hardcode `x86_64-linux` CANN paths (generated on x86
hosts). NEVER copy that pattern into your `kernel/CMakeLists.txt` — the
target build machine is aarch64 and any `x86_64-linux` path fails the
controlled build. Write all CANN toolchain paths against
`aarch64-linux` (e.g. `${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/asc/include`
and `${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/lib64/...`), or better use the
architecture-agnostic variables provided by the CANN cmake toolchain.

CMake PATH PROBES MUST BE SILENT (2026-08-21 batch rule): when
`execute_process` runs `import torch_npu` (or torch) to discover an include
path, the import can print device/console-log noise to stdout in a
container without full device init; that text is captured INTO the path
variable and corrupts the generated flags.make ("missing separator").
Silence stdout at the FD level during the import — the device-init noise
is printed through C-level fd writes that `sys.stdout` redirection does NOT
catch. Use
`python3 -c "import os; _s=os.dup(1); os.dup2(os.open(os.devnull,
os.O_WRONLY), 1); import torch_npu; os.dup2(_s, 1);
print(os.path.dirname(torch_npu.__file__))"` — and apply the same to any
`import torch` probe (backend autoload prints the same noise). Never
capture raw import output into a path.
"""


def _tilelang2ascendc_npubench_worker_context(workspace: Path) -> str:
    """Return the persisted TileLang2AscendC implementation contract."""
    state_path = Path(workspace) / ".opgen_state.json"
    if not state_path.exists():
        return ""
    try:
        import json

        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        # A malformed state is handled by the common durable-state gates.  Do
        # not let this optional Tile context break brief building.
        return ""
    if not isinstance(state, dict):
        return ""
    if state.get("source_kind") != "port-aclnn-tilelang2ascendc":
        return ""
    port_source = state.get("port_source") if isinstance(state, dict) else None
    if not (
        isinstance(port_source, dict)
        and port_source.get("kind") == "port-aclnn-tilelang2ascendc"
    ):
        raise RuntimeError("TileLang2AscendC durable state has no matching port_source binding")
    source_stage = state.get("port_a3_source")
    if not isinstance(source_stage, str) or not source_stage:
        raise RuntimeError("TileLang2AscendC durable state has no source-stage path")
    return _TILELANG2ASCENDC_SOURCE_BOUNDARY.format(source_stage=source_stage)


_NPUBENCH_TL_PHASE_A_INTRO = (
    """A.1. Read the frozen task only to understand its callable API and input-group
semantics. Read the staged TileLang2AscendC project only as implementation
context, then author an independent candidate in the same `model_new_ascendc.py`
plus `kernel/` project format. Keep `register.cpp`, `kernel/op_host/`,
`kernel/op_kernel/`, and the authored `kernel/CMakeLists.txt` coherent. The
model must call a matching local `torch.ops.npu.<op>` registration. Do not
byte-copy the staged project, call ACLNN or framework ATen operators, add a
CPU/NumPy fallback, or point the model at the read-only source stage. The
graybox has no CANN compiler/toolchain authority, so do not compile or run a
build script; missing toolchain pieces there are expected, never an infra
violation, and never a `build stuck` / `INFRA_BASELINE_VIOLATED` handoff. The
controlled target runner owns build, candidate snapshot, and
NPUKernelBench evaluation after handoff; exit with
`→ orchestrator: build-ready — <one-line summary>`. This route does not require
`kernel/pybind11.cpp`; it requires
`TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`, host launch evidence, and real
`__global__` + `__aicore__` + `AscendC::` device implementation evidence.
Keep AICore code device-safe: do not include `<algorithm>` or call host STL
`std::min`/`std::max` in `kernel/op_kernel/`; use scalar comparisons instead.
Additionally, `kernel/register.cpp` must define a
`PYBIND11_MODULE(_<op>_ext, m)` entry point matching the CMake `OUTPUT_NAME`
of the built extension (an empty body with only `m.doc()` is enough; the
`TORCH_LIBRARY` static registration completes at load). Without it the built
`_<op>_ext<EXT_SUFFIX>.so` has no `PyInit__<op>_ext` symbol and the evaluator
fails with "dynamic module does not define module export function".
`model_new_ascendc.py` must add `kernel/build` to `sys.path` itself before
importing the extension, e.g.
`sys.path.insert(0, str(Path(__file__).resolve().parent / "kernel" / "build"))`;
the isolated evaluator stage cannot find the module otherwise."""
    + """ The runner-selected `ModelNew` (or compatibility
fallback `Model`) in `model_new_ascendc.py` must subclass `torch.nn.Module`,
`nn.Module`, or imported `Module`, because the harness moves it with `.to(device)`
and calls `.eval()` before frozen evaluation."""
)

_NPUBENCH_TL_PHASE_A_HANDOFF = """A.3. Report static readiness and unsupported shapes honestly. Do not claim a
build or functional PASS from the graybox. The target runner builds the
authored TileLang2AscendC CMake project and evaluates it against the frozen
NPUKernelBench bundle."""

_NPUBENCH_PHASE_A_INTRO = (
    """A.1. Read the frozen task only to understand its callable API and input-group
semantics. Implement and build the target kernel in the current workspace. The
candidate must be exposed through `model_new_ascendc.py` and invoke the newly
built kernel directly. A framework dispatcher, CPU fallback, subprocess
delegation, lookup table, or replayed reference tensor is not an implementation."""
    + """ The runner-selected `ModelNew` (or compatibility
fallback `Model`) in `model_new_ascendc.py` must subclass `torch.nn.Module`,
`nn.Module`, or imported `Module`, because the harness moves it with
`.to(device)` and calls `.eval()` before frozen evaluation."""
)

_NPUBENCH_PHASE_A_HANDOFF = """A.3. Build failures and unsupported callable shapes must be reported honestly
in the handoff. Never mark a functional result PASS from a local self-report."""

_NPUBENCH_PHASES_TEMPLATE = """# PHASES (NPUKernelBench port)

## Reference contract — immutable original task bundle

`reference.source` is **`npubench`**. Functional truth is the frozen,
old-format NPUKernelBench task selected in `.opgen_state.json`: its real task
Python path, same-stem JSON/JSONL sidecar, and bundle manifest. Do not
translate it into the repository's generic model/test reference representation,
rename it, rewrite it, regenerate inputs, or use a caller-supplied path after
staging. The migration source tree is implementation context only; it is never
a functional oracle.

The orchestrator has already run the target-side NPUKernelBench preflight
before this worker starts and records the immutable result in
`npubench_evidence/preflight_target_receipt.json`. Inspect that receipt and
preserve its binding. Do **not** run `python3 -m npubench_runner preflight`
from the graybox worker: it has no A5 target dependencies and may overwrite
the target evidence with a controller-side provider-installation error.

The PROGRESS.md tail `## OPERATOR NOTE` entries / attachment area are the
operator's injection channel — always read the tail on spawn.

The task may rely on `__file__`, sibling imports, package-relative imports, and
a `.json` file whose contents are JSONL. Preserve those semantics. Do not
write inside `reference_inputs/`, replace runner adapter files, or cache task
outputs as a candidate shortcut.

## Phase A — author a real target candidate

{phase_a_intro}

For a compiled extension under the candidate's `kernel/build/`, make
`model_new_ascendc.py` self-contained: it is loaded as a top-level frozen task,
not as a Python package. Resolve the build directory from `__file__`, add it
only when it exists, then use a normal extension import. Do **not** use
`from .kernel.build import ...`, and do not rely on the orchestrator's current
working directory or adapter paths. The candidate snapshot includes its own
built extension artifacts.

```python
from pathlib import Path
import sys

_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _BUILD.is_dir():
    sys.path.insert(0, str(_BUILD))
import _<op>_ext
```

A.2. Preserve the task's callable and initialization compatibility. The
harness seeds randomness, takes candidate `get_init_inputs()` when present
(otherwise the reference provider), creates each reference input group once,
and deep-copies it for comparison. It does not transfer reference state into
the candidate; do not alter the task or substitute unrelated random inputs.

{phase_a_handoff}

## Phase B — harness-owned precision and performance

B.1. The orchestrator owns final evaluation. After the candidate build is
complete it invokes `npubench_runner evaluate`, applying the frozen precision
contract and writing bound evidence. Do not author or overwrite
`verification.json` precision counts, runner evidence, profiler reports, or a
substitute grader.

B.2. The final performance gate is a separate harness measurement using the
repository quick profiler engine with warm-up 3, repeats 5, and retained raw
profile artifacts. Do not substitute a timing method or claim a speedup from
an unbound local run.

B.3. Precision and performance must bind the same frozen bundle and candidate
tree. Any source or candidate change invalidates prior evidence and requires a
fresh orchestrator evaluation.

## Phase C — provenance and handoff

C.1. Preserve current-workspace build lineage in
`verification.json.build_evidence.compiled_provenance` (workspace source,
deployed source, object, shared library, and SHA256 values). Do not use an
installed target tree as binary provenance.

C.2. Write `knowledge_update.md` with `## Context`, `## Findings`,
`## KB-promotable patterns`, `## Cited KB items`, and `## Anti-patterns avoided`.

C.3. `iter_cap_remaining = {iter_cap_remaining}`. Hand off only after the
candidate is buildable and its true entry point is available for the harness.
Evaluation failures are feedback for a new implementation iteration, not a
reason to weaken or replace the frozen task contract.
"""


def _npubench_phase_a_texts(has_tilelang_context: bool) -> tuple[str, str]:
    """Return the Phase-A intro and handoff bodies for the two provider routes.

    Extracted verbatim from ``_npubench_phase_instructions_block``; the two
    routes stay byte-identical to the pre-extraction branches.
    """
    if has_tilelang_context:
        return _NPUBENCH_TL_PHASE_A_INTRO, _NPUBENCH_TL_PHASE_A_HANDOFF
    return _NPUBENCH_PHASE_A_INTRO, _NPUBENCH_PHASE_A_HANDOFF


def _npubench_phase_instructions_block(
    op: str,
    workspace: Path,
    iter_cap_remaining: int,
    env: "AscendCEnv",
) -> str:
    """Worker contract for a frozen old-format NPUKernelBench provider.

    This is self-contained because the old task is the functional oracle and
    the migration source tree is implementation context only.
    """
    del env  # The provider has no source-runtime configuration dependency.
    forced_block = _forced_architecture_block(workspace)
    forced_prefix = (forced_block + "\n\n") if forced_block else ""
    tilelang_source_context = _tilelang2ascendc_npubench_worker_context(Path(workspace))
    tilelang_source_prefix = (tilelang_source_context + "\n") if tilelang_source_context else ""
    phase_a_intro, phase_a_handoff = _npubench_phase_a_texts(bool(tilelang_source_context))
    return forced_prefix + tilelang_source_prefix + _NPUBENCH_PHASES_TEMPLATE.format(
        phase_a_intro=phase_a_intro,
        phase_a_handoff=phase_a_handoff,
        iter_cap_remaining=iter_cap_remaining,
    )


def _pa3_phase_d_1(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Phase D — Verify against A3 ground truth

**⚠ MANDATORY (DEBT-NEW + P140, 2026-05-14/17):**
Phase D verification MUST execute OUR built kernel binary on A5 NPU.

**P140 (2026-05-17) — invocation pattern**: use the **AscendC official Pybind调用
pattern** (`atlas_ascendc_10_0057.html`) — `ACLRT_LAUNCH_KERNEL` macro inside
a `pybind11.cpp` wrapper. This pattern has long been used by the generic
cold-start path. Pre-P140, this
brief incorrectly mandated aclnn-direct C++ runner, which only works when
CANN install **already ships** the op's A5 binary; for ops CANN doesn't
ship (the actual port_a3 product target), aclnn-direct hits
`ACL_ERROR_OP_NOT_FOUND` and requires vendor opp registration which is a
9-hour rabbit hole. The correct path is `ACLRT_LAUNCH_KERNEL` + pybind11
— binds the kernel binary directly, no CANN registration required, no
PyTorch dispatcher fallback risk.

PyTorch dispatcher paths (`torch.nn.functional.*`, `torch._foreach_*`,
`torch_npu.npu_*`) are still **FORBIDDEN** for the verify path — they
fall back to AICPU when our kernel isn't reachable, masking failures.

**Canonical pattern (per `atlas_ascendc_10_0057.html`)**:

```cpp
// kernel/pybind11.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "aclrtlaunch_<op>_kernel.h"   // auto-generated by ascendc_library
#include "torch_npu/csrc/core/npu/NPUStream.h"

at::Tensor run_<op>(const at::Tensor &x, /* other args */) {{
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  at::Tensor z = at::empty_like(x);
  ACLRT_LAUNCH_KERNEL(<op>_kernel)(blockDim, stream, x.data_ptr(), ..., z.data_ptr(), totalLength);
  return z;
}}

PYBIND11_MODULE(_<op>_ext, m) {{ m.def("run_<op>", &run_<op>); }}
```

Required artifacts for Phase D PASS (port_a3 mode, P140 contract):
- `workspace/{op}/kernel/{op}_kernel.cpp` — AscendC kernel with
  `extern "C" __global__ __aicore__ void <op>_kernel_<dtype>(...)`
  signatures for fp16 / fp32 / bf16. **NOT** `<op>_apt.cpp` PR4778-style
  with `GET_TILING_DATA` — that requires host-side tiling struct + GM
  allocation. Pybind/ACLRT_LAUNCH_KERNEL takes raw pointer args directly.
- `workspace/{op}/kernel/{op}_kernel.h` — helpers / constants
- `workspace/{op}/kernel/pybind11.cpp` — pattern above
- `workspace/{op}/model_new_ascendc.py` — `nn.Module` wrapping the
  pybind ext: `from .kernel.build import _<op>_ext; class ModelNew(nn.Module): forward = _<op>_ext.run_<op>`

  **HARD BAN — cache-replay anti-pattern (OL-165 / P151, enforced 2026-05-18 by P135.CR brief directive)**:
  `model_new_ascendc.py` MUST NOT contain ANY of the following:
  - `import hashlib` or any hash-keyed lookup of cached tensors
  - `_tensor_digest` / `_LOOKUP_CACHE` / `_build_lookup` / similar
    digest-keyed dictionary built from `a5_capture.pt` or any
    prior-capture file
  - `a5_capture.pt` read (or any `torch.load("*.pt")` of prior outputs)
  - `.cpu()` data motion in forward path (data must stay on NPU; CPU
    transfer = wrong place for compute)
  - `try: kernel(...); except: return cached[...]` fallback to cached
    outputs when the kernel raises — that's the explicit "fallback to
    pretend it worked" cheat pattern
  - `subprocess` / `torch_npu.npu_<op>` / `torch.<op>` delegation —
    those run aclnn/PyTorch, not OUR kernel
  - **Using another `output/` artifact as executable truth or copying it raw.**
    A `.prior_art_scan.json` result, DEBT203 branch base, source/target archive
    provenance, or SHA-verified `.upstream_prestaged.json` entry may be read
    only as logged, read-only advisory context. Do not copy untracked files,
    transplant bodies, replay outputs, or use another archive to satisfy a
    truth gate. The source-architecture detector's fresh arch22 evidence and
    source-NPU capture remain authoritative; the generated arch35 result must
    still be built and verified on target NPU. `--optimize` / `--resume` may
    also read this op's own tracked history, subject to the same provenance.

  If `_ext.run_<op>(t)` raises, PROPAGATE the exception. Do NOT
  silently substitute cached outputs or PyTorch fallback. P149/P151
  finalize gates will REJECT any archive containing these markers,
  causing iter_cap death-loop. foreach_neg 2026-05-18: 9 worker
  iters wrote cache-replay model_new_ascendc.py because the brief
  didn't ban it explicitly — workspace had to be abandoned. Don't
  repeat that pattern. The forward body should be ≤5 lines:
  ```python
  def forward(self, *inputs):
      return _ext.run_<op>(*inputs)
  ```
  That's it. Any logic beyond this — alignment trim, padding,
  dtype convert, CPU offload — moves into the kernel or is a bug.

  **forward signature MUST be callable as `model(*group)` — NO keyword-only
  params (forward-idiomatic, P0-FWD 2026-06-09).** The generic gate
  (`precision_eval_two_tier` AND `phase_o5_perf_capture`) calls `model(*group)`
  where `group` is a `get_input_groups()` tuple — e.g. FA emits
  `(q, k, v, atten_mask, attrs, dims)`. A keyword-only signature like
  `def forward(self, q, k, v, atten_mask=None, *, attrs=None, dims=None)`
  raises `TypeError` under `model(*group)` (the `*` forces attrs/dims to be
  passed by keyword, but `model(*group)` passes them positionally). Effect: the
  op can ONLY be bespoke-verified (runners that pass `attrs=`/`dims=` explicitly),
  and is INVISIBLE to BOTH the generic precision gate and the generic perf gate —
  it silently never goes through generic CI. Write `def forward(self, *inputs)` or
  an all-positional-or-keyword signature (drop the `*`). The harness device-move
  (`phase_o5` `t.npu() if isinstance(t, torch.Tensor) else t`) leaves non-tensor
  attrs/dims dicts untouched, so positional dicts are safe. (FA whole-port
  da415a80 had `*, attrs, dims` keyword-only → bespoke-verified-only until the
  2026-06-09 remove-`*` fix made it generic-gate-able.)
- `workspace/{op}/edge_runner.py` — drives ModelNew on edge_inputs.pt,
  produces a5_capture.pt
- Build via `python3 vendor/AscendOpGenAgent/utils/build_ascendc.py {op} -v $SOC_VERSION --clean`
  (the standalone build helper auto-generates CMakeLists). NO
  hand-written `build_runner.sh` needed — script handles bisheng compile
  + pybind link.
- `verification.json.truth_source` must contain the substring `"a3_capture"`
  (edge_dataset.pt's A3 ground truth) — pybind path doesn't go through
  aclnn, so the prior `"aclnn"` substring requirement is RETIRED for
  port_a3 mode (post-P140 gate). orchestrator post-verify accepts
  `"a3_capture_via_pybind_aclrtlaunch_kernel"` or similar.

  A live, complete A3 capture is mandatory. If it is unavailable, stop and
  report the capture verdict; never synthesize or substitute the migration
  truth. `edge_dataset.pt["a3_outputs"]` must therefore carry measured A3
  outputs, and `verification.json.truth_source` must identify that live capture.

"""


def _pa3_phase_d_2(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""**Pattern authority**: use only the repository's independently authored
AscendC templates and the fixed-layout/build contracts cited above. Generated
archives are never a template source.

D.1. **Build the pybind module on A5** via `build_ascendc.py` (which
     compiles via bisheng on A5 then links pybind11.so).
D.2. **Run `edge_runner.py` against `workspace/{op}/edge_inputs.pt`** —
     each case calls `ModelNew.forward` (pybind), captures outputs to
     `a5_capture.pt`.
D.3. **NATIVE TWO-TIER verdict (task#82, 2026-06-14, owner-directed)** — your
  `pass_a_runner.py` MUST produce a two-tier (T1 + T2) verdict via the SHARED
  harness judge. Do NOT emit a
  single-tier max_abs_diff verdict and let a precision-probe back-fill T2.

  The verdict needs THREE per-case tensors. Your runner ALREADY computes all
  three inside its per-case loop — you just feed them to the shared judge:
    - ours      = your A5 kernel output — `ModelNew()(inputs...)` run on NPU
                  (the tensor you already capture per case; NOT a pre-saved file)
    - a3_cann   = A3-CANN NPU captured output — `case["a3_outputs"]["<out>"]`
                  read straight from `edge_dataset.pt` (per-case dict)
    - cpu_truth = CPU fp64 ground truth — `Model()(inputs...)` from `model.py`
                  run on CPU per case (the tensor you already compute for pass_a)

  Standard (owner-directed 2026-06-30): the DEFAULT forward verdict is the 生态 (ecosystem)
  VERBATIM-vendored cann-bench `compare.py` — NOT a worker-defined tolerance. The grader is a
  HARNESS function; your job is to EMIT TENSORS, NOT to decide pass/fail.
    - **生态 T1 (DEFAULT)**: golden=cpu_truth(fp64), output=ours@native dtype, native_output=the
      reference re-run at the op's NATIVE dtype on CPU. compare.py applies MERE/MARE + small-value/
      cancellation carve-outs (this is what faithfully PASSes fp32 near-zero — the prior tolerance
      classifier false-FAILed it). Integer/index ops = bit-exact.
    - **T2 (OPTIONAL 商用)**: ours fails T1 but ours_MARE ≤ a3_cann_MARE — opt-in only, NOT the default.

  **WORKER EMITS TENSORS; HARNESS GRADES (anti-reward-hack — owner 2026-06-30).** Your
  `pass_a_runner.py` MUST save these per-case tensor files (do NOT decide the verdict yourself; the
  harness `precision_eval_port_a3_two_tier.load_and_classify` independently re-grades them with the
  vendored compare.py, author≠measurer):
    - `a5_capture.pt`        — ours: ModelNew on NPU, per-case first out tensor (at the op's native dtype)
    - `cpu_truth_outputs.pt` — cpu_truth: model.py Model on CPU at fp64, per-case first out tensor
    - `native_capture.pt`    — **REQUIRED for the fp32-near-zero fix**: model.py Model on CPU re-run at
                               the op's NATIVE dtype (fp16/bf16/fp32), per-case first out tensor. This is
                               the REAL CPU-same-precision baseline compare.py needs. NEVER fabricate it
                               (do NOT just cast cpu_truth) — it must be the reference fn at native dtype.
    - `edge_dataset.pt`      — {{"inputs", "a3_outputs"}}: the A3-CANN capture (optional 商用 competitor)
  ```python
  import torch
  ours_list, cpu_list, native_list = [], [], []
  for c in ds:                                   # ds = edge_dataset.pt (list of case dicts)
      ours_list.append(run_kernel_on_npu(c["inputs"]))               # ModelNew on NPU, native dtype
      cpu_list.append(run_model_on_cpu(c["inputs"], dtype="fp64"))   # model.py on CPU, fp64 truth
      native_list.append(run_model_on_cpu(c["inputs"], dtype=NATIVE))# model.py on CPU, NATIVE dtype
  torch.save(ours_list,   "a5_capture.pt")
  torch.save(cpu_list,    "cpu_truth_outputs.pt")
  torch.save(native_list, "native_capture.pt")   # CPU-same-precision baseline (REAL, not a cast)
  # The HARNESS grades: load_and_classify(workspace) → compare.py 生态 verdict (default).
  ```
  You MAY ALSO print a self-report summary (`classify_port_a3_case`/`summarize`) for convenience, but
  the AUTHORITATIVE verdict is the harness re-grade of the tensors above — emit correct tensors.
  `verification.json.precision.pass_a` carries: `tier1_pass`, `tier2_pass`, `tier1_pass_inclusive`,
  `total`, `tier2_status` ∈ {{`N/A_ALL_T1`, `N/A_T1_PASS`, `N/A_ECOSYSTEM`, `PASS`, `FAIL`,
  `A3_UNAVAILABLE`}}, plus `native_capture_present` / `n_native_used`.

  **phase_o5 ENGAGEMENT GATE (task#82)**: phase_o5_runner FAILS the verify (RUNNER_FAILED) if the
  pass_a summary lacks `tier2_status`. The gate logs `[phase_o5 task#82] port_a3 two-tier ENGAGED`.

  **A3 capture guard**: both grading routes require a genuine, complete A3-NPU
  capture. Missing or aliased `a3_outputs` is a blocking provenance failure; do
  not substitute another implementation or auto-pass either tier.

  **D.3.fmt — FORMAT-ROBUST truth load (MANDATORY, DEBT-137 2026-05-31)**:
  `edge_dataset.pt` from the live A3 capture is a top-level **LIST** of
  per-case records `[{{**case, "a3_outputs": {{...}}, ...}}, ...]`. Reject any
  other top-level shape; it is not valid migration truth. Load it as follows:
    ```python
    data = torch.load(HERE / "edge_dataset.pt", weights_only=False)
    assert isinstance(data, list), "live A3 capture must be a list of case records"
    truth_outputs = [r["a3_outputs"] for r in data]
    ```
  Keep the capture-provenance checks active before grading.

D.3a. **FAILURE-ANALYSIS SOP — re-read A3 source BEFORE claiming
  reference-side anomaly** (mandatory, port_a3 only).

  When ANY case fails D.3 with `max_abs_diff > 10× tier1_atol` (T1) /
  `max_abs_diff > 1.0` (T2) / `max_rel_diff > 100×`: **READ FULL SOP
  AT `kb/shared/GATE_CONTRACT.md §D.3a-port_a3`**.

  TL;DR (steps you MUST execute in order):
  1. Re-read A3 op_kernel for the case's specific code path; cite file:line for
     every transformation.
  2. Check `model.py` matches A3 semantics line-by-line; fix model.py first if
     mismatch.
  3. Re-run Phase D after model.py fix; if pass → reference bug, close loop.
  4. Only if step 3 still fails: investigate A5-kernel-side.
  5. OL-109 hardware-floor claim allowed ONLY after steps 1-3 + residual ≤ 2 LSB
     + cite specific primitive.

  FORBIDDEN under trigger: claiming OL-109/floor/non-canonical without source
  citation; hypothesis sweep on scale variations (reward-hacking); declaring
  PARTIAL_PERSIST + shipping. See GATE_CONTRACT.md §D.3a-port_a3 for full
  rationale (2026-05-21 fused_quant_mat_mul case 6 incident).

D.4. **Perf measurement — METHOD-SYMMETRIC contract (P141, 2026-05-17)**.

     ⚠ **CRITICAL — methodology asymmetry is fraud-equivalent**:
     a3 path measures `torch_npu.npu_<op>` / aclnn pair (`GetWorkspaceSize` +
     `Execute` + workspace alloc + executor build) wrapped in
     `perf_counter`+`synchronize`. a5 path measures `ACLRT_LAUNCH_KERNEL`
     macro directly (no aclnn host overhead). Published aclnn host overhead
     30-80µs alone explains observed 50-70µs A3-A5 gap at host-overhead scale
     — the resulting ratio is **a methodology artifact, NOT kernel speedup**.

     Per P102 cross-arch byte-equal contract + P141 retraction precedent
     (clipped_swiglu 4.5× / expand_into_jagged_permute 3.77× retracted
     2026-05-17), perf claim MUST satisfy ONE OF:

     ⚠ **CRITICAL — read this before deciding which option**:
     Option 1 and Option 2 are about TIMING SYMMETRY, not API SYMMETRY.
     A3 and A5 will almost always use DIFFERENT call shapes (A3 calls
     aclnn; A5 calls ACLRT_LAUNCH_KERNEL macro since most ops aren't
     shipped on A5). That asymmetry is the WHY of the port — not a
     blocker for measurement. **Do NOT escape to Option 3 just because
     the API surfaces differ.** Option 3 is the LAST resort, reserved
     for situations where BOTH Option 1 AND Option 2 are demonstrably
     infeasible (e.g., A3 stream model fundamentally can't expose
     aclrtEvent — extremely rare).

     **Option 1 — torch_npu.profiler + operator_details.csv (PRIMARY)**:

       This uses the same device-kernel timing primitive across source and target —
       see `vendor/AscendOpGenAgent/utils/performance.py:343-369`
       (`_measure_single_with_profiler`). Pure on-NPU per-kernel
       device duration extracted from profiler dump's
       `operator_details.csv` `"Device Self Duration(us)"` column.
       Excludes host runtime, pybind dispatch, aclnn host prep, AND
       stream stall between per-tensor kernel pushes (the trap that
       breaks `torch.npu.Event` wrap on V220 foreach-list ops).
       Handles single-tensor / foreach-list / fused-multi-kernel
       op shapes uniformly via `groupby("Name").sum()`.

       Concrete pattern (both sides — A3 via SSH runner, A5 locally):
       ```python
       import torch_npu, pandas as pd
       from torch_npu.profiler import (profile, ProfilerActivity,
"""


def _pa3_phase_d_3(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""                                       schedule, _ExperimentalConfig,
                                       ProfilerLevel,
                                       tensorboard_trace_handler)
       def measure(test_fn, warmup=5, repeats=5, out_dir="/tmp/prof"):
           experimental = _ExperimentalConfig(
               aic_metrics=None, profiler_level=ProfilerLevel.Level1,
               l2_cache=False, data_simplification=False)
           # initial warm-up to amortize lazy init
           test_fn(); torch.npu.synchronize()
           total_steps = (1 + warmup) + repeats
           with profile(
               activities=[ProfilerActivity.NPU, ProfilerActivity.CPU],
               schedule=schedule(wait=0, warmup=warmup, active=repeats,
                                 repeat=1, skip_first=1+warmup),
               on_trace_ready=tensorboard_trace_handler(out_dir),
               experimental_config=experimental,
           ) as prof:
               for _ in range(total_steps):
                   test_fn(); prof.step(); torch.npu.synchronize()
           # parse operator_details.csv from out_dir
           df = pd.read_csv(find_csv(out_dir, "operator_details.csv"))
           total_us = df.groupby("Name")["Device Self Duration(us)"].sum().sum()
           return total_us / repeats  # avg device-self us per iter
       ```

       Both A3 and A5 support `torch_npu.profiler`. SAME function +
       SAME parsing on both sides. The ratio = `measure(a3_test_fn)
       / measure(a5_test_fn)` is purely device-time, symmetric by
       construction.

       Method label MUST mention: `"torch_npu.profiler.profile"` +
       `"operator_details.csv"` + `"Device Self Duration(us)"` so the
       P141 gate recognizes this as a clean methodology.

     **Option 1B — Device-event wrap (DEPRECATED — fragile on V220 foreach)**:

       ⚠ **Use Option 1 (profiler-CSV) instead unless infeasible**.
       Device-event wrap (aclrtEvent on A3 / torch.npu.Event on A5)
       was the prior "Option 1" but has a known regime split: for
       foreach-list ops on V220 (A3), per-tensor host-side aclnn
       dispatch latency causes stream STALL between kernels which
       `elapsed_time` captures as device time. Empirically
       foreach_sqrt 2026-05-18: A3 Event 0.081-0.150ms > A3
       perf_counter wall-clock 0.048-0.068ms (wrong direction —
       Event should be SMALLER if it excluded host). Single-tensor
       ops (e.g., `foreach_neg` with one tensor) work correctly
       (A3 Event 39-71µs < perf_counter 58-89µs). **Profiler-CSV
       avoids this entire regime split** — it captures pure
       kernel-execution time per-kernel-row in the CSV, no
       elapsed-time-window leakage.

       If you MUST use Option 1B (e.g., profiler unavailable), this
       is the pattern:
       ```python
       e0 = torch.npu.Event(enable_timing=True)
       e1 = torch.npu.Event(enable_timing=True)
       e0.record()
       model_new_ascendc.forward(x)
       e1.record()
       torch.npu.synchronize()
       elapsed_ms = e0.elapsed_time(e1)
       ```
       A3 runner MUST construct executor + workspace BEFORE the
       event-wrapped region.

       Worker MUST cite which regime (`single-tensor` or `foreach-list`)
       the op is in if using Option 1B, AND verify
       `Event_median < perf_counter_median` on the SAME inputs as a
       sanity check before accepting the Option 1B ratio.

     **Option 2 — Wrapper-inclusive symmetric (acceptable, end-to-end)**:

       Both sides use Python `perf_counter` + `torch.npu.synchronize`
       (or stream sync), each wrapping its OWN natural user-visible call
       shape end-to-end. The wrappers DO NOT need to be byte-equivalent
       — they only need to be the user-visible entry point on each side
       (the call shape a real consumer would use). Both sides report
       wall-clock end-to-end time including their respective host
       overheads (A3=aclnn host prep + kernel; A5=pybind binding +
       launch + kernel). The ratio is then "how fast is user-visible
       end-to-end on A5 vs A3" — a real measurement, even if A3's
       host overhead has a different composition than A5's.

       Concrete pattern (both sides):
       ```python
       # A3 side (subprocess to cpp runner, OR pybind-wrapped aclnn):
       torch.npu.synchronize()  # or aclrtSynchronizeStream
       t0 = time.perf_counter()
       call_a3()  # natural A3 entry — subprocess / pybind / torch_npu API
       torch.npu.synchronize()
       t1 = time.perf_counter()
       a3_us = (t1 - t0) * 1e6

       # A5 side:
       torch.npu.synchronize()
       t0 = time.perf_counter()
       model_new_ascendc.forward(x)  # pybind+ACLRT_LAUNCH_KERNEL
       torch.npu.synchronize()
       t1 = time.perf_counter()
       a5_us = (t1 - t0) * 1e6
       ```

       MUST declare `performance.method.option2_wrapper_composition`
       explicitly — what each side's wrapper contains — so a future
       reader can judge whether the host overhead difference is fair
       (e.g., subprocess-launch overhead on A3 side dominates and
       makes ratio noisy).

     **Option 3 — NOT_VERIFIED_SAME_METHOD (LAST RESORT)**:

       Reserved for cases where BOTH Option 1 AND Option 2 are
       demonstrably infeasible. The `performance.retraction.reason` MUST:
       - Explain why Option 1 doesn't work (e.g., A3 aclnn entry has no
         exposed stream parameter, can't record event) — NOT just "API
         differs".
       - Explain why Option 2 doesn't work (e.g., A3 runner cannot be
         wrapped in Python at all) — NOT just "wrappers different".
       - Cite specific code/CANN-doc evidence for each infeasibility.

       Status: `"NOT_VERIFIED_SAME_METHOD"`. Ratio: ABSENT (not None).
       Per-side absolute timings MAY be reported but ratio MUST NOT.

       ⚠ **Default-to-Option-3 is the cheating mode this contract
       exists to prevent**. Worker MUST attempt Option 1 first; if
       blocked, attempt Option 2; only then Option 3.

     **PREFERRED ORDER**: Option 1 (rigorous) → Option 2 (acceptable)
     → Option 3 (escape). Emit BOTH Option 1 ratio AND Option 2 ratio
     when feasible; the ratio field uses Option 1 (primary), Option 2
     goes in `performance.option2_wrapper_inclusive` (secondary cross-check).

     **Hard floor 0.6× still applies ONLY when status == PASS via Option 1 or 2.**
     With Option 3 (NOT_VERIFIED), no floor check — finalize gate skips perf.

D.5. Write `workspace/{op}/verification.json` with tier verdict per case
     and the ratio (only if status == PASS per D.4). **`truth_source` MUST
     contain `"aclnn"`, `"a3_cann"`, OR `"a3_capture_via_pybind_aclrtlaunch_kernel"`
     (P140 pattern).**

     **`performance.method` MUST explicitly name BOTH sides' invocation**:
     e.g. `"a3=aclrtEventElapsedTime around aclnn execute call (executor+workspace built outside timed window); a5=torch.npu.Event around ACLRT_LAUNCH_KERNEL; method=Option1 device-event"`
     — finalize gate parses this to detect asymmetric perf_counter+wrapper claims.

     When emitting Option 2 as secondary cross-check, ALSO populate
     `performance.option2_wrapper_inclusive` = {{`a3_us`, `a5_us`,
     `ratio`, `a3_wrapper_composition`, `a5_wrapper_composition`}}.

**Anti-pattern (BANNED per DEBT-NEW)**: a `pass_a_runner.py` whose only
reference computation is `tensor.<op>()` on CPU + comparing to A3-captured
NPU output (e.g. CPU `tensor.abs()` vs A3 `torch._foreach_abs` NPU output)
— this NEVER touches A5 hardware, let alone our kernel. If your op doesn't
have a CANN aclnn entry (e.g. dynamic_rnnv2 / index_put_with_sort), emit
`verdict: MISSING_ENTRY` and stop — do NOT fall back to CPU reference.

D.6. **pass_b schema in port_a3 mode (MANDATORY shape)** (P96, 2026-05-15).

In port_a3_to_a5 mode, pass_b is **degenerate by design** — `edge_dataset.pt`
IS the ground truth, `pass_a` IS the A5-vs-A3-edge_dataset comparison via
the runner.cpp. There is no separate benchmark-style "Pass A on benchmark
cases + Pass B on edge_dataset" two-tier shape. So:

- **DO NOT write `workspace/{op}/run_pass_b.py`** — this file is the
  legacy generic verifier; migration does not use it. Writing it triggers
  the P94 anti-cycle gate (verifier reading verification.json).
- **DO write `verification.json.precision.pass_b` with this canonical N/A shape**:
  ```
  {{
    "status": "N/A",
    "reason": "port_a3_to_a5 mode: pass_b is subsumed by pass_a — edge_dataset.pt['a3_outputs'] IS the truth source per ROADMAP §1.5 Path-B contract; pass_a IS the A5-vs-A3-edge_dataset comparison. pass_b would be degenerate.",
    "method": "n/a — port_a3 mode pass_b not applicable"
  }}
  ```
- **Schema checklist for port_a3 mode** (replaces benchmark Phase E checklist):
  - [ ] `verification.json.precision.pass_a.status` in {{PASS / PASS_WITHIN_TOLERANCE / FAIL}}
  - [ ] `verification.json.precision.pass_a.method` references the runner.cpp + edge_dataset, NOT `precision_eval_two_tier.py` or `Model.forward vs ModelNew.forward`
  - [ ] `verification.json.precision.pass_b.status` == "N/A" with the canonical reason above
  - [ ] `workspace/run_pass_b.py` MUST NOT exist
  - [ ] `verification.json.truth_source` contains `"aclnn"` or `"a3_cann"`
  - [ ] `verification.json.performance.independent_re_measure` exists with non-null `ran`/`ratio` (or explicit skip reason)

This catches the 2026-05-15 gather_elements_v2 cycle-gate trip (worker wrote
run_pass_b.py with self-citation of verification.json). C-PORT-A3-PASS-B-SCHEMA
in aog-self-critic catalog enforces this at gate level too; the brief here
sets the expectation up-front so the worker doesn't ship the wrong shape.

"""


def _pa3_phase_e(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Phase E — Knowledge Update + handoff

E.1. Write `workspace/{op}/knowledge_update.md` (≥ 100 bytes, target
    500-2000) — deliverable ③ of the COMPLETE ARCHIVABLE DELIVERABLE contract
    above. It MUST carry the canonical Phase E 5-section structure with literal
    headers (`GATE_CONTRACT.md` §"Phase E Knowledge Update"): `## Context`,
    `## Findings`, `## KB-promotable patterns`, `## Cited KB items`,
    `## Anti-patterns avoided`. The finalize KB_WRITEUP gate structurally
    rejects a writeup that lacks `## Findings`. Highlight any arch22→arch35 port-specific
    findings in the `## Findings` section:
    - Was there an unexpected `ToFloat<>` adjustment? (KB W11 candidate)
    - Did the cross-op router edit pattern apply cleanly? (KB W9 confirms)
    - Did `__CCE_AICORE__ == 220` strip introduce any FP16/BF16 deviation?
      (KB W10 candidate)

E.1.bis. **BINARY PROVENANCE** (deliverable ② — gate `binary_provenance` /
    DEBT-091, REQUIRED on PASS). Emit only current-workspace build lineage in
    `verification.json.build_evidence.compiled_provenance`: workspace-relative
    `source`, copied-back `deployed_source`, `object`, and dispatched `shared_lib`,
    plus their SHA256 values. `workspace_source_sha256`, `deploy_source_sha256`,
    and `built_from_source_sha256` must be identical. Never inspect or hash an
    installed CANN target tree; such evidence is outside this migration boundary.

E.2. Handoff per `_exit_handoff_block`:
    - PASS T1 + perf ≥ 0.6× → `→ orchestrator: done — <summary>`
    - PASS T2 only (L2/L3 floating) → still done; T2 is valid for non-L1
    - perf < 0.6× → `@aog-kernel-optimizer`
    - precision FAIL after 4 iter → `@aog-precision-probe`

"""


def _pa3_iter_budget(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Iter budget

iter_cap_remaining = {iter_cap_remaining}. Port ops often complete in
1 iter if the L1 / L2 path is clean. L3 fused ops may need 2-3 iters.
If you exhaust the budget, exit with handoff to orchestrator (do NOT
keep iterating)."""


def _port_a3_phase_instructions_block(
    op: str,
    workspace: Path,
    iter_cap_remaining: int,
    env: "AscendCEnv",
) -> str:
    """W5 (2026-05-12, ROADMAP §1.5): port_from_a3_ascendc Phase A-E prose.

    Differs from a generic cold-start in:
    - Source = ops-nn op_dir (op_host/ + op_kernel/), not Model.forward
    - Reference = A3-CANN outputs captured in Phase O2.5 a3-ref variant
      (workspace/edge_dataset.pt["a3_outputs"]), not CPU-PyTorch truth
    - Output = arch35 kernel + apt.cpp + ascend950 binary config (NOT a
      pure workspace/kernel/<op>.cpp; mirror the ops-nn layout)
    - Cross-op check = if peer_op_dependencies is non-empty (e.g. ctc_loss_v3
      depends on ctc_loss_v2), the peer's op_api/<peer>.cpp may need 3
      surgical router edits to route the current op on A5 (per KB W9
      cross-op router pattern)
    """
    # Each non-live provider branches before loading live-capture metadata.
    # This prevents its worker context from consuming source-runtime artifacts.
    if _workspace_uses_npubench_reference(workspace):
        return _npubench_phase_instructions_block(
            op, workspace, iter_cap_remaining, env
        )

    import json as _json
    port_source = env.port_a3_source or "(env.port_a3_source missing; orchestrator misconfig)"
    a3_runnable_path = workspace / "a3_reference_runnable.json"
    aclnn_entry = "(W4 a3_reference_runnable.json not present yet)"
    gen_data_source = "(unknown — read a3_reference_runnable.json after Phase O2.5)"
    peer_deps_line = "(unknown — read a3_reference_runnable.json after Phase O2.5)"
    if a3_runnable_path.is_file():
        try:
            payload = _json.loads(a3_runnable_path.read_text())
            aclnn_entry = payload.get("aclnn_entry") or "(no aclnn entry — MISSING_ENTRY verdict)"
            gen_data_source = payload.get("gen_data_source") or "(absent — input_gen must be hand-authored)"
            deps = payload.get("peer_op_dependencies") or []
            peer_deps_line = ", ".join(deps) if deps else "(none — single-op port)"
        except Exception as e:
            aclnn_entry = f"(failed to read a3_reference_runnable.json: {e!r})"

    # Forced-architecture honor block (2026-06-16): also applies to port_a3 —
    # any forced-architecture op. Empty for
    # non-forced ops (brief stays byte-identical).
    _forced_block = _forced_architecture_block(workspace)
    _forced_prefix = (_forced_block + "\n\n") if _forced_block else ""
    phase_args = {
        "op": op,
        "workspace": workspace,
        "iter_cap_remaining": iter_cap_remaining,
        "port_source": port_source,
        "aclnn_entry": aclnn_entry,
        "gen_data_source": gen_data_source,
        "peer_deps_line": peer_deps_line,
        "env": env,
    }

    return _forced_prefix + (
        _pa3_context(**phase_args)
        + _pa3_phase_a_1(**phase_args)
        + _pa3_phase_a_2(**phase_args)
        + _pa3_phase_a_3(**phase_args)
        + _pa3_phase_b(**phase_args)
        + _pa3_phase_c(**phase_args)
        + _pa3_phase_d_1(**phase_args)
        + _pa3_phase_d_2(**phase_args)
        + _pa3_phase_d_3(**phase_args)
        + _pa3_phase_e(**phase_args)
        + _pa3_iter_budget(**phase_args)
    )
