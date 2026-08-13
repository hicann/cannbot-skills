# CANN Bench ACLNN Evaluation Playbook

Use this playbook when the job is to take an EasyASC-generated ACLNN operator,
migrate it into `cann-bench/agent/example/demo/aclnn_launch_example`, and run
`scripts/run_evaluation.sh` for an existing benchmark task. It also owns the
optional submission-zip delivery route described below.

This is the validated shared-box workflow for the root-operated A2 / 910B3 path.
It is not a generic upstream `cann-bench` setup guide, and it is not the first
stop for ordinary `OpExec(..., simulator=False)` bring-up inside this repository.
For EasyASC runtime generation details, follow `agent/references/code-paths.md`
and `doc/06_codegen_and_runtime.md`.

All CANN Bench working material lives in the `easyasc_cannbench_kernels`
extension repository, expected as a sibling checkout (`../easyasc_cannbench_kernels`):
production kernels under `agent/example/kernels/*/cann_bench/`, contract tests, evaluation
reports, submission tooling, and the open-work diary. This playbook stays in
EasyASC and routes into that checkout; paths below marked "extension repo" are
relative to it.

## Success criteria

- a direct `cann_bench.<op>(...)` smoke from the source tree matches the host reference
- one in-process sweep attempts every target case; isolated single-case success is not sufficient
- `scripts/run_evaluation.sh` writes a report and JSON for the target task
- if performance scoring is enabled, profiler export also produces `kernel_details.csv`
- every unsupported or inaccurate case is reported as a failure instead of being
  replaced by an official operator, host implementation, or relaxed threshold

## Non-negotiable choices

- use the root A2 / 910B3 access path from local `machine_specs.md`; do not guess the box route
- keep the EasyASC mirror and the `cann-bench` checkout separate
- generate migration templates from `OpExec(..., simulator=False, debug=False, gen_only=True)`
- do not migrate from the `debug=True` standalone workspace
- keep public-contract adaptation in the wrapper and keep the generated kernels on the narrow ABI that EasyASC already proved
- count only EasyASC-generated device rows as evaluated work; an official ACLNN,
  PyTorch, CPU, or host-computed fallback invalidates the affected result
- keep the benchmark's CPU-fp64 golden and thresholds unchanged; if the hardware
  path cannot meet them, preserve the failure and its error metrics

If the requested operator does not exist yet, this route owns both phases:
author and validate the single EasyASC kernel first, then begin ACLNN migration.
Do not assume an existing generated kernel as a precondition when the router sent
an authoring-and-scoring request here.

## Optional route: package a submission zip

When the user asks to "generate a CANN Bench submission zip", "package a
CANN Bench submission", or equivalent, follow the A2 CANN Bench submission
packaging guide at `easyasc_cannbench_kernels/kernels/a2/cann_bench_submission/README.md`.
The guide and its adjacent `package_submission.py` are the owners for archive
layout, SoC defaults, host normalization, source-only filtering, manifest
generation, and zip validation.

This is an optional delivery branch, not a requirement for every evaluation:

- package every supplied task under `csrc/ops/<task>/` in one submission zip;
- if the user requests packaging only, do not require benchmark scoring or a
  full evaluation run before delivering the archive;
- keep the A2 versus A5 target boundary stated by the packaging guide; and
- deliver the zip together with its manifest, SHA-256 digest, task count, and
  physical-kernel count.

**Submission discovery invariant:** for every full or subset package, the
following three sorted sets must be exactly equal:

1. physical task directories directly under `csrc/ops/`, excluding `_common`
   and every other underscore-prefixed directory;
2. the strings in the one module-level, literal `cann_bench/__init__.py::__all__`
   assignment; and
3. the names of public module-level wrapper functions in that initializer.

Rewrite the initializer when packaging a subset. Remove wrappers for omitted
tasks instead of retaining them behind `globals().pop`, runtime filtering, or a
later dynamic `__all__` reassignment. The scorer may inspect source without
executing it, so the initializer must contain exactly one `__all__` assignment
whose value is accepted by `ast.literal_eval`.

Enforce this invariant twice: once on the staged source tree and again by
reading the finished ZIP. The archive check must parse `cann_bench/__init__.py`
from the ZIP, derive physical tasks from archive member paths, and fail on any
extra or missing task, wrapper, or export. Do not deliver an archive based only
on a pre-ZIP check.

Use the normal migration workflow below first when an operator directory has
not yet been assembled. Once all requested operator directories exist, switch
to the packaging guide for the final archive and its focused smoke checks.

## Optional route: submit and read results through the BenchSite MCP

`https://cannbench.com` scores submission zips on its own NPU pool and exposes
the whole flow as an MCP server, so a packaged archive can be submitted and its
per-case results read without leaving the agent. Official documentation lives at
`https://cannbench.com/docs/mcp`; this section records only what that page does
not say and what cost time to discover.

### Install

The official launcher installs `benchsite-mcp` into the default interpreter.
Prefer a dedicated virtualenv instead: `benchsite-mcp` requires `mcp` with no
upper bound, and the mcp SDK 2.x moved `mcp.server.fastmcp`, so installing beside
an existing 2.x gives `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`
at startup. Pin `mcp<2` in the venv and leave the ambient environment alone.

```bash
VENV="$HOME/.local/share/benchsite-mcp/venv"
python3 -m venv "$VENV"
DL_URL=$(curl -fsSL https://cannbench.com/api/meta/mcp-version \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['download_url'])")
"$VENV/bin/python" -m pip install "mcp>=1.2,<2" "$DL_URL"
```

Keep the token out of the agent config. `~/.claude.json` is not a secret store
and a project-level `.mcp.json` is version controlled, so read the token from a
file in the launcher instead — rotating it then means editing one file.

```bash
# ~/.local/share/benchsite-mcp/launch.sh  (chmod +x)
set -e
VENV="$HOME/.local/share/benchsite-mcp/venv"
export BENCHSITE_API_TOKEN="$(cat "${BENCHSITE_TOKEN_FILE:-$HOME/cannbench_token.txt}")"
export BENCHSITE_URL="${BENCHSITE_URL:-https://cannbench.com}"
# Same upgrade contract as the official launcher: one version probe per start,
# reinstall only when the remote version differs, fail loudly rather than
# silently running a stale build.
REMOTE=$(curl -fsSL "$BENCHSITE_URL/api/meta/mcp-version" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['version'],d['download_url'])")
set -- $REMOTE
LOCAL=$("$VENV/bin/python" -c "import benchsite_mcp;print(benchsite_mcp.__version__)" 2>/dev/null || echo 0)
[ "$LOCAL" = "$1" ] || "$VENV/bin/python" -m pip install --quiet --upgrade "mcp>=1.2,<2" "$2"
exec "$VENV/bin/python" -m benchsite_mcp.server
```

```bash
claude mcp add --scope user cann-bench-site -- ~/.local/share/benchsite-mcp/launch.sh
```

A server added mid-session is not visible to the running agent — Claude Code
binds MCP servers at session start. Either restart, or drive the server over
stdio from a throwaway JSON-RPC client for the remainder of the session. Such a
client must keep the child process alive and read each response before sending
the next request; closing stdin after writing all requests makes slow tool calls
return nothing, which looks exactly like a broken tool.

### Tools

`get_version`, `whoami`, `get_credits`, `list_runners`, `list_benchmarks`,
`get_benchmark_task`, `submit_kernel`, `rerun_hidden_cases`,
`get_operator_rank`, `get_job`, `get_job_logs`, `list_jobs`.

`submit_kernel` requires `submission_path`, `target_hardware` and
`selected_operators`; `benchmark_slug`, `aggregation_token` and `submission_tag`
are optional. Fill `aggregation_token` anyway -- **it**, not `submission_tag`, is
the key the account's submissions are grouped under, and a job without it still
scores but lands outside the intended entry. The token is account state, not
something the archive or the MCP responses reveal:

    aggregation_token = "agg_xvtgdlvdwytr9mgpbs3l"

`submission_tag` is a separate, free-form leaderboard label (Agent/Harness/
Model). Leaving it unset inherits the aggregation scheme's existing tag, which is
what the account's past jobs did -- so an untagged job is only mislabelled if
`aggregation_token` was also omitted.

### Which runner the number came from

The 950pr pool is not uniform. **`runner-950pr-simen` scores low and drifts on
small values**, so a score from it is a lower bound and its precision numbers
near zero are not trustworthy. Take **`runner-950pr-multi` as the reference**
and treat simen only as a smoke signal -- a regression seen only on simen is not
a regression until multi reproduces it, and a score improvement measured on
simen cannot be compared against one measured on multi.

`job.results.setup_info.environment` names the runner that actually ran a job, so
check it before comparing two scores.

### Reading a result

Per-case diagnostics live in `get_job`, not in `get_job_logs`. `get_job_logs`
returns stage events only (queued, claimed, build, archive). The useful payload
is `job.results.operators[i].cases[j]`, where `error_msg`, `failure_type`,
`failure_reason` and `accuracy` carry the evaluator's own message — including the
`MERE=` / `MARE=` numbers and markers such as `NaN位置不匹配`. `job.results.setup_info.environment`
names the runner's chip, CANN version and container image, and
`job.submission.original_filename` confirms which archive a job actually ran.

### Costs and limits

`get_credits` reports a daily quota (`remaining`, resetting at 16:00 UTC /
midnight Asia/Shanghai) and `max_active_jobs_per_user`, default 3. A submission
spends a credit whether it scores or fails, so spend them on runs that can
distinguish hypotheses. The control plane deduplicates by zip content SHA-256:
resubmitting an identical archive returns the *old* job instead of running again,
so any new run needs a real content change, even a comment.

### Gotchas paid for

- `build.sh` must sit at the zip root, not in a nested directory, or the
  submission is rejected before it reaches a runner.
- A job that stays `queued` usually means no online runner for the target
  hardware. Check `list_runners` before assuming the submission is at fault.
- The runner's environment is not the environment the archive was built against.
  One 950pr runner reported `cann: 9.1.0-beta.1` inside `docker: cake-ci / CANN 9.0.0`.
- **Do not introduce a torch operator the runner has no binary for.** A
  diagnostic build that allocated its output with `at::full` failed all 20 cases
  with `Op Fill does not has any binary` / `FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed`
  before the kernel under test ever ran. Host-side allocation plus `.to(device)`
  avoids the dependency entirely: build the tensor with `x.options().device(at::kCPU)`
  and copy it down. Prefer `at::empty` in shipped builds for the same reason.

## Workflow

### 1. Prepare the remote shell

- open the root login shell described in `machine_specs.md`
- resolve the EasyASC and benchmark checkout paths from `machine_specs.md`;
  tracked documentation must not assume a workstation or remote-home layout
- from the EasyASC checkout, run with `PYTHONPATH=.`

Quick sanity check:

```bash
cd <easyasc-repo-root>
PYTHONPATH=. python -c "import torch, torch_npu; print(torch.__version__)"
```

### 2. Generate non-debug EasyASC templates

Write or reuse a small driver under `tmp/<task>/` that calls:

```python
OpExec(kernel_fn, out_dir=..., simulator=False, debug=False, gen_only=True)(...)
```

Rules:

- if the public benchmark op needs dtype dispatch, generate one EasyASC kernel per dtype path
- keep the kernel-side ABI simple, usually a flattened `[1, n]` tensor contract plus scalar `Var`s
- do wrapper-only formula adaptation on the host side, not in the generated kernel body

TensorList and reduction rules discovered during `foreach_norm` bring-up:

- generate with `debug=False`; the debug workspace path does not support `GMTensorList`
- pass an explicit host `item_count` scalar when list length drives kernel loops; do not rely on the current A2 `GMTensorList.size()` ACLNN path
- use `builtins.range` in plain-Python runners because `from easyasc.a2 import *` shadows Python `range`
- bind the dynamic item and shape axes explicitly with `shape_bindings`
- drain the vector pipeline before rebinding a dynamic list item at the next loop iteration
- for a two-stage reduction, derive the number of initialized partial rows from the exact stage-1 assignment, not `min(tile_count, core_count)`:

```text
tiles_per_core = ceil(tile_count / core_count)
active_cores = ceil(tile_count / tiles_per_core)
```

  The simpler expression over-counts workers whenever ceil-div leaves empty
  trailing cores. Reading those unwritten rows can pass in isolation and fail
  only after earlier cases leave nonzero allocator contents.
- use a finite placeholder scalar for a mode-owned `+inf` path; emitting a raw
  C++ initializer such as `float p = inf` is not portable generated code
- if a one-launch reduction publishes per-AIV GM partials and then uses
  `allvec_ready/allvec_wait`, keep every participant resident. On A2, the
  validated arrangement is `@kernel(mode="mix")`: blockDim is 20 and task ratio
  1:2 supplies all 40 AIV lanes, while the cube body may remain empty. Do not use
  `mode="vec"` for this pattern: it emits `KERNEL_TYPE_AIV_ONLY`, which can
  compile but is outside the `CrossCoreSetFlag/WaitFlag` task-mode contract.
  Manually switching that pure-vector task to `MIX_AIV_1_0` also reduces the
  resident vector lanes to 20 and loses throughput. The Python simulator models
  20 cores x 2 vec lanes and therefore will not expose the invalid AIV-only
  device task declaration; require a real-HW run for this check.

Validated `Exp` split:

- `ExpHalfKernel`
- `ExpFloatKernel`
- `ExpBfloat16Kernel`

After generation, build each emitted template once so the formal autogen ACLNN
API files appear:

```bash
cd <template_root>
bash build.sh
```

When templates are transferred from macOS with a tar pipe, disable metadata
sidecars at archive creation time. Excluding an existing `._*` glob is not
sufficient: BSD tar can synthesize AppleDouble entries from extended
attributes while it writes the stream.

```bash
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude='._*' \
  -C <template-parent> <template-name> \
  | ssh <target> 'tar xzf - -C <task-scoped-destination>'
```

Before invoking CMake, require both the transferred template and the assembled
source tree to pass `find <root> -name '._*' -print -quit` with no output. Apply
the same rejection to a submission ZIP. An AppleDouble `._foo.cpp` can be
auto-discovered as C++ source and fail `opbuild`, so this check is part of the
source boundary rather than an ignorable transfer warning.

Collect:

- `build_out/autogen/aclnn_<name>.h`
- `build_out/autogen/aclnn_<name>.cpp`
- the matching generated `ophost` / `opkernel` files from the custom-op project

### 3. Migrate into `aclnn_launch_example`

Use `<cann-bench-root>/agent/example/demo/aclnn_launch_example` as the benchmark-facing source project.

**Merge all dtype kernels into a single operator directory.** Do not create one
standalone `csrc/ops/<op>_<dtype>_kernel/` directory per dtype. Instead, place
everything under `csrc/ops/<op>/` and register each kernel from the same
`CMakeLists.txt` through multiple `register_aclnn_op` calls. The outer
`csrc/ops/CMakeLists.txt` auto-discovers subdirectories via `add_subdirectory`
(skipping `op_kernel/` and `_`-prefixed helpers), so a single `exp/` directory
is sufficient — no evaluation-script changes needed.

`register_aclnn_op` macro signature (defined in `cmake/func.cmake`):

```cmake
register_aclnn_op(
    OP_TYPE           # kernel name as registered in the op info cfg
    HOST_SRCS         # host-side tiling source file(s)
    API_SRCS          # L2 ACLNN API source file(s)
    KERNEL_DIR        # kernel source directory, relative to csrc/ops/
    KERNEL_FILE       # main kernel .cpp filename
    TILING_INCLUDE_DIR # subdir for tiling includes (expanded to abs path)
    API_INCLUDE_DIR   # subdir for API includes (expanded to abs path)
)
```

### Pre-build deployment checklist

Before running `bash build.sh --soc=ascend910b`, verify these five items
against the validated `csrc/ops/exp/CMakeLists.txt` template:

| # | Check | File |
|---|-------|------|
| 1 | diff `csrc/ops/<op>/CMakeLists.txt` vs `csrc/ops/exp/CMakeLists.txt` — every `register_aclnn_op` must pass `"${...}"` variable references, never literal `""` | two `CMakeLists.txt` |
| 2 | Host/API source paths use `${CMAKE_CURRENT_SOURCE_DIR}/op_host/...`, not bare `/op_host/...` | `CMakeLists.txt` |
| 3 | `register_aclnn_plugin` passes `"${<OP>_PLUGIN_SRCS}"` not `""` | `CMakeLists.txt` |
| 4 | After a successful build, the wheel + `.run` sizes visibly grow vs the previous build — flat size after adding new operator files is a red flag | `dist/` |
| 5 | The generated source/header stem matches the CANN op type's snake-case name. Acronym groups can differ (`FinalizeDG...` -> `finalize_dg_...`, not `finalize_d_g_...`), so inspect generated host/API/kernel filenames before changing math | generated template + operator directory |

These five checks are the minimum barrier before every `build.sh`. Skipping
them is how Sigmoid (2026-07-04) shipped with empty-string HOST_SRCS/API_SRCS
and scored 0/20; the fix was a 1:1 diff against `exp/CMakeLists.txt`.

Key rules for the merged layout:

- `KERNEL_DIR` must be relative to `csrc/ops/` (e.g. `exp/op_kernel`), NOT
  relative to the current `CMakeLists.txt` — the top-level build uses
  `SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops` and appends `KERNEL_DIR`.
- `TILING_INCLUDE_DIR` and `API_INCLUDE_DIR` are expanded relative to
  `CMAKE_CURRENT_SOURCE_DIR` (the operator's own directory), so use bare
  subdir names like `op_kernel` / `op_api`.
- Shared kernel headers (e.g. `tensorutils.h`) keep one copy under
  `op_kernel/`.

Validated `Exp` merged layout:

```
csrc/ops/exp/
├── CMakeLists.txt          # 3× register_aclnn_op + 1× register_aclnn_plugin
├── op_host/                # one host .cpp + _tiling.h per dtype
│   ├── exp_float_kernel.cpp
│   ├── exp_float_kernel_tiling.h
│   ├── exp_half_kernel.cpp
│   ├── exp_half_kernel_tiling.h
│   ├── exp_bfloat16_kernel.cpp
│   └── exp_bfloat16_kernel_tiling.h
├── op_api/                 # one ACLNN L2 API .cpp + .h per dtype
│   ├── aclnn_exp_float_kernel.cpp / .h
│   ├── aclnn_exp_half_kernel.cpp / .h
│   └── aclnn_exp_bfloat16_kernel.cpp / .h
├── op_kernel/              # all kernel sources, shared tensorutils.h
│   ├── exp_float_kernel.cpp / _vec.h / _cube.h
│   ├── exp_half_kernel.cpp / _vec.h / _cube.h
│   ├── exp_bfloat16_kernel.cpp / _vec.h / _cube.h
│   └── tensorutils.h
└── op_plugin/
    └── exp_plugin.cpp      # dtype dispatch + torch binding
```

`exp/CMakeLists.txt` skeleton:

```cmake
set(EXP_FLOAT_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/exp_float_kernel.cpp)
set(EXP_FLOAT_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_exp_float_kernel.cpp)
register_aclnn_op(ExpFloatKernel "${EXP_FLOAT_HOST_SRCS}"
    "${EXP_FLOAT_API_SRCS}" exp/op_kernel exp_float_kernel.cpp
    op_kernel op_api)

# ... repeat for ExpHalfKernel, ExpBfloat16Kernel ...

set(EXP_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/exp_plugin.cpp)
register_aclnn_plugin("${EXP_PLUGIN_SRCS}" op_plugin)
```

Plugin include paths: in `exp_plugin.cpp`, include API headers with relative
paths from the plugin directory (e.g. `#include "../op_api/aclnn_exp_float_kernel.h"`)
— the cmake `register_aclnn_plugin` adds `${CMAKE_CURRENT_SOURCE_DIR}/op_plugin`
to the plugin include dirs, so `../op_api/` resolves correctly.

Wrapper responsibilities only:

- flatten the input to `[1, n]`
- precompute host-side scalar coefficients
- dispatch by dtype to the generated ACLNN entrypoint
- reshape the result back to the original public shape

For `Exp`, the wrapper precomputed `scale_coeff` / `shift_coeff` and dispatched
to `aclnnExpHalfKernel`, `aclnnExpFloatKernel`, or
`aclnnExpBfloat16Kernel`.

### 4. Preload runtime libraries inside `cann_bench/__init__.py`

This is the single most important patch. The evaluation framework installs the
wheel via `pip install` and then imports `cann_bench` from site-packages.
The subprocess doing the import is a fresh Python process — it does **not**
automatically inherit parent-shell environment variables unless they are
explicitly forwarded. The only reliable injection point is `__init__.py` itself.

In `<cann-bench-root>/agent/example/demo/aclnn_launch_example/cann_bench/__init__.py`, add the following
block **before** `from . import _C`:

```python
import ctypes, os as _os

# 4a. Always preload libopapi.so with RTLD_GLOBAL
_ascend_home = _os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
_opapi_handle = ctypes.CDLL(
    _os.path.join(_ascend_home, "lib64/libopapi.so"), mode=ctypes.RTLD_GLOBAL
)

# 4b. Load libcust_opapi.so from the caller-provided custom OPP root.
#     Do not embed a checkout- or machine-specific fallback path.
_custom_opp = _os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
if _custom_opp:
    _cust_lib = _os.path.join(_custom_opp, "op_api/lib/libcust_opapi.so")
    if _os.path.exists(_cust_lib):
        # Keep both handles alive for later lazy ACLNN symbol lookup.
        _custom_opapi_handle = ctypes.CDLL(_cust_lib, mode=ctypes.RTLD_GLOBAL)

        # 4c. Auto-set environment variables so child processes and subsequent
        #     ACLNN runtime lookups can find the extracted custom OPP tree.
        _os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _custom_opp)

        _ld = _os.environ.get("LD_LIBRARY_PATH", "")
        for _sub in ["op_api/lib", "op_impl/ai_core/tbe/op_tiling",
                      "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"]:
            _d = _os.path.join(_custom_opp, _sub)
            if _d not in _ld:
                _ld = _d + ":" + _ld if _ld else _d
        _os.environ["LD_LIBRARY_PATH"] = _ld

        _py = _os.environ.get("PYTHONPATH", "")
        for _sub in ["op_impl/ai_core/tbe/custom_ops_impl",
                      "op_impl/ai_core/tbe"]:
            _d = _os.path.join(_custom_opp, _sub)
            if _d not in _py:
                _py = _d + ":" + _py if _py else _d
        _os.environ["PYTHONPATH"] = _py
```

**Why all three parts are needed:**

- **(a) libopapi.so preload**: generated custom-op libraries depend on symbols
  such as `l0op::Contiguous` from `libopapi.so`; without the global preload,
  `dlopen("libcust_opapi.so")` fails even though the build succeeded.
- **(b) libcust_opapi.so preload**: loads the custom ACLNN operator registry
  so `torch.ops.cann_bench.<op>` resolves.
- **(c) env-var auto-set**: the evaluation harness spawns subprocesses that
  import `cann_bench` from the installed wheel. Those subprocesses do **not**
  automatically inherit the parent's `ASCEND_CUSTOM_OPP_PATH`, `LD_LIBRARY_PATH`,
  or `PYTHONPATH`. Setting them inside `__init__.py` guarantees they are present
  as soon as `cann_bench` is imported, regardless of how the process was launched.

**After editing `__init__.py`, you MUST rebuild the wheel** (see §5) so the
changes are baked into the installed package.

### 5. Build the launch example

```bash
cd <cann-bench-root>/agent/example/demo/aclnn_launch_example
bash build.sh --soc=ascend910b
```

Expected artifacts under `dist/`:

- the wheel package (`cann_bench-*.whl`)
- the `.run` package (`cann_bench_*.run`)

**Rebuild after any `cann_bench/__init__.py` change.** The evaluation framework
installs the wheel via `pip install`; a source-tree-only edit to `__init__.py`
is invisible to the installed package until you rebuild.

### 6. Make the custom OPP runtime visible to evaluation subprocesses

Five things must be in place before `cann_bench._C` can call a custom kernel on
NPU. Without them a direct smoke or `run_evaluation.sh` will fail — usually with
"not found in libcust_opapi.so" or "binary_info_config.json does not support
opType".

#### 6a. Runtime library preload + env-var auto-set (`cann_bench/__init__.py`)

This is the critical patch — see §4 for the full code block. The `__init__.py`
must:

1. Preload `libopapi.so` and `libcust_opapi.so` with `ctypes.RTLD_GLOBAL` before
   `from . import _C`
2. After a successful `libcust_opapi.so` load, auto-set `ASCEND_CUSTOM_OPP_PATH`,
   `LD_LIBRARY_PATH`, and `PYTHONPATH` in `os.environ` so that child processes
   and the ACLNN Python runtime can locate the extracted custom OPP tree
3. Require `ASCEND_CUSTOM_OPP_PATH` to be exported or forwarded by the caller;
   never bake a machine-local extraction path into the package

Without this, `dlopen("libcust_opapi.so")` sees unresolved `l0op::Contiguous`
and other symbols from `libopapi.so`, and `_C` import fails or custom kernels
don't register. Even if the preload succeeds, without the env-var auto-set the
ACLNN runtime cannot find `binary_info_config.json` and the kernel scripts.

#### 6b. Custom OPP path (`ASCEND_CUSTOM_OPP_PATH`)

Must point at the extracted `vendors/custom_ops` tree so the ACLNN runtime can
find `binary_info_config.json` and kernel binaries.

#### 6c. Linker path (`LD_LIBRARY_PATH`)

Prepend three directories so `libcust_opapi.so`, `liboptiling.so`, and
`libcust_opmaster_rt2.0.so` are resolvable:

- `$ASCEND_CUSTOM_OPP_PATH/op_api/lib`
- `$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling`
- `$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64`

#### 6d. Python path for kernel scripts (`PYTHONPATH`)

The ACLNN Python framework imports per-kernel scripts (e.g. `exp_float_kernel.py`)
from the op_impl tree. Without this the runtime can't locate the kernel's
`binary_info_config.json` entry and rejects the opType at launch:

- `$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/custom_ops_impl`
- `$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe`

#### 6e. Manual extraction as fallback

```bash
cd <cann-bench-root>/agent/example/demo/aclnn_launch_example/dist
mkdir -p run_extract
bash cann_bench_*.run --noexec --extract=run_extract
CUSTOM="$PWD/run_extract/packages/vendors/custom_ops"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM"
export LD_LIBRARY_PATH="$CUSTOM/op_api/lib:$CUSTOM/op_impl/ai_core/tbe/op_tiling:$CUSTOM/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CUSTOM/op_impl/ai_core/tbe/custom_ops_impl:$CUSTOM/op_impl/ai_core/tbe:${PYTHONPATH:-}"
```

#### Required cann-bench patches

If `run_evaluation.sh` fails because these values are not forwarded, patch the
following benchmark-repo files:

- `cann_bench/__init__.py` — preload `libopapi.so` and `libcust_opapi.so` with
  `ctypes.RTLD_GLOBAL` before `from . import _C`; add Python wrapper functions
  for any new operators (e.g. `cann_bench.exp(...)`).
- `src/kernel_eval/data/package_manager.py`
  - extract the `.run` instead of executing `--install`
  - resolve the run path absolutely
  - after extraction, set **all four** environment variables:
    `ASCEND_CUSTOM_OPP_PATH`, `LD_LIBRARY_PATH` (op_api + op_tiling),
    `PYTHONPATH` (custom_ops_impl + tbe)
  - record these in `self._runtime_env` so they persist across process life
- `src/kernel_eval/eval/subprocess_runner.py`
  - forward `ASCEND_CUSTOM_OPP_PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH` into
    child processes
  - forward the active CLI knobs such as `--device-id`, `--warmup`, `--repeat`,
    `--profiler-level`, and `--no-perf`
  - use the configured `source_dir` instead of a hardcoded project root
- `src/kernel_eval/eval/evaluator.py`
  - pass the full config into `SubprocessRunner`

### 7. Run a direct smoke before scoring

Before invoking the benchmark harness, compare one small case per dtype against
the host reference. The important part is to prove that:

- the wrapper imports cleanly
- all runtime requirements from §6 are satisfied
- the wrapper-level shape and dtype adaptation is correct

**Critical import order:** `torch_npu` (and `torch`) must be imported **before**
`cann_bench`. Reversing the order causes `torch_npu` to detect its backend as
already loaded and raise `RuntimeError: Two accelerators cannot be used at the
same time in PyTorch: npu and npu`.

Manual smoke skeleton (if the `__init__.py` preload from §4 is already in place,
the script does **not** need to duplicate the `ctypes.CDLL` calls — the
`__init__.py` handles them automatically):

```python
import os, sys

# 1. Ensure the .run has been extracted and env vars are set
DIST_DIR = "<cann-bench-root>/agent/example/demo/aclnn_launch_example/dist"
EXTRACT_DIR = os.path.join(DIST_DIR, "run_extract")
CUSTOM_OPS = os.path.join(EXTRACT_DIR, "packages/vendors/custom_ops")
if not os.path.isdir(CUSTOM_OPS):
    import subprocess
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    run_files = [p for p in os.listdir(DIST_DIR) if p.endswith(".run")]
    subprocess.run(["bash", os.path.join(DIST_DIR, run_files[0]),
                    "--noexec", f"--extract={EXTRACT_DIR}"], check=True)

os.environ["ASCEND_CUSTOM_OPP_PATH"] = CUSTOM_OPS
# (LD_LIBRARY_PATH and PYTHONPATH will be auto-set by cann_bench/__init__.py)

# 2. Import torch_npu BEFORE cann_bench
import torch, torch_npu
sys.path.insert(0, "<cann-bench-root>/agent/example/demo/aclnn_launch_example")
import cann_bench

# 3. Test one case per dtype
for dt in (torch.float16, torch.float32, torch.bfloat16):
    x = torch.randn(1024, 1024, dtype=dt, device="npu:0")
    y = cann_bench.sigmoid(x)  # replace with your op
    ref = torch.sigmoid(x.float())  # replace with your reference
    err = (y.float() - ref).abs().max().item()
    assert err < 1e-3, f"{dt}: error {err} too large"
    print(f"{dt}: max_abs_diff={err:.6e}")
```

### 8. Performance-tuning loop (fast iteration)

The full `run_evaluation.sh` takes ~200 s because it evaluates all registered
operators across all 20 cases. For kernel tuning you need seconds, not minutes.
Three progressively faster options:

#### 8a. Single-operator CLI call (`~17 s` per case)

After the first `build.sh` has produced the wheel and `.run`, you can re-run a
single case **without rebuild** by calling the evaluation CLI directly:

```bash
cd <cann-bench-root>
PYTHONPATH=<cann-bench-root>/src python -m kernel_eval.cli eval \
  --source-dir agent/example/demo/aclnn_launch_example \
  --operator Sigmoid \
  --case-id 1 \
  --device-id 0 \
  --skip-install \
  --no-subprocess-isolation \
  --no-perf \
  -v
```

| Flag | Effect |
|------|--------|
| `--operator Sigmoid` | only this operator |
| `--case-id 1` | only this test case (omit to run all 20) |
| `--skip-install` | skip `build.sh` + `pip install`; uses the already-built wheel |
| `--no-subprocess-isolation` | run in the same process, avoids subprocess spawn + re-import overhead |

**Accuracy of `--no-subprocess-isolation` timing:** the `--no-perf` path falls
back to Python `time.perf_counter()` wall-clock measurement, which includes
Python dispatch, tensor conversion, and ACLNN launch overhead. For small
elementwise ops this overhead can dominate, making the reported time
insensitive to kernel-level changes. Use `OpExec` profile (§8b) for accurate
kernel timing.

> **When `--skip-install` is NOT enough:** if you changed any C++ source under
> `csrc/ops/<op>/`, you must rebuild:
> ```bash
> cd <cann-bench-root>/agent/example/demo/aclnn_launch_example
> bash build.sh --soc=ascend910b   # ~100 s
> # re-extract .run, pip install new wheel, then re-run
> ```

> **Rebuild trap — `run_evaluation.sh` skips the build if `dist/` has a wheel.**
> When you change kernel sources and want `run_evaluation.sh` to pick them up,
> you must delete the cached wheel first:
> ```bash
> rm -f dist/cann_bench*.whl dist/cann_bench*.run
> rm -rf dist/run_extract build/
> ```
> Otherwise `prepare_from_source` finds the existing wheel and uses it as-is,
> ignoring your source changes — the scores will be identical to the previous run.

> **Profiler-cache trap — even after deleting the wheel, the score can stay
> bit-for-bit identical.** `perf_eval.py` reuses the OLDEST `kernel_details.csv`
> under `reports/prof_data/{op}/{case}/` (keyed by case id only, never cleaned in
> archive mode, `os.listdir` picks the first entry — not newest-by-mtime), so the
> score stays frozen at the FIRST kernel version ever evaluated. Clear it before
> every scored run:
> ```bash
> rm -rf reports/prof_data/level1/<op>   # or mv aside; per-operator — clear the one you eval
> ```
> Tell-tale you hit this: `overall_score` repeats to ~1e-13 across independent
> runs. See Failure routing ("stale profiler-csv reuse") for the full explanation.

#### 8b. EasyASC `OpExec` profile — bypass cann-bench entirely (`~30 s`)

When you are only changing the **EasyASC DSL kernel** (not the generated C++
code), the fastest loop is to profile directly on real hardware through
`OpExec(..., simulator=False, debug=False, profile=True)`. This skips
cann-bench's build → extract → install → subprocess chain entirely.

Add a `profile_*_case` helper to your kernel file following the pattern in
`agent/example/kernels/a2/vec_only/exp_general.py:199-252`:

```python
def profile_sigmoid_case(shape, dtype, seed):
    ...
    y_out = OpExec(
        kernel_fn,
        simulator=False, debug=False, profile=True,
        out_dir=f"tmp/<task>/sigmoid_profile_{tag}_{numel}",
        cann_path=os.environ.get("ASCEND_HOME_PATH") or "",
    )(x_flat, y_flat, numel, shape_bindings={...}).reshape(shape)
    # validate correctness, extract avg_us from r.sh.log
```

Gate it behind `--profile` in `__main__`:

```python
if __name__ == "__main__":
    if "--profile" in sys.argv[1:]:
        profile_sigmoid_case((1024, 1024), torch.float16, seed=0)
        sys.exit(0)
    # ... normal checks ...
```

Run on the remote box:

```bash
cd <easyasc-repo-root>
PYTHONPATH=. python agent/example/kernels/a2/vec_only/sigmoid_general.py --profile
```

**Trade-off:** this measures kernel-only wall time, not the full
cann-bench scoring metric. Use it for rapid iteration; run the CLI single-case
call (§8a) for a score check before claiming a result.

#### 8b2. A/B several candidates against a same-session control

`easyasc_cannbench_kernels/tools/cannbench_ab_bench.sh` builds N submission zips into
**private** OPP roots and privately unpacked wheels, then benchmarks them
interleaved on one device:

```bash
PY=$CONDA/bin/python VARIANTS="ctrl v2 v3" DEVICE=npu:1 ROUNDS=4 COUNT=20 \
  bash agent/scripts/cannbench_ab_bench.sh both      # or: build | bench
```

Read the **median per-repeat speedup against the same-session baseline**, and the
min-max spread next to it. A remembered number from earlier in the day is not a
baseline: on a shared board the same unchanged package measured 12.9-15.9x slower
in the afternoon than in the morning while the `torch` baseline it runs against
moved by less than 2%, because the CANN built-in is not competing for the same
vector cores. "The baseline is stable" does **not** license "the device is quiet".

When the spread is wider than the difference you are chasing, the board cannot
answer the question that day — say so rather than reporting the median. The
sharpest test for that: if a case whose *code is byte-identical* between the two
variants moves by tens of percent, every number in that run is noise.

Three build traps the script exists to route around, all paid for once:

* the shared conda `site-packages` is rewritten by whichever session installed
  last, so each variant imports `cann_bench` from its own unpacked wheel;
* the aclnn scalar list is compiled into **both** the `.run` and the wheel, so a
  variant that changes the host tiling plan needs its own wheel — reusing one
  gives `AclNN_Parameter_Error`, or a silently wrong plan;
* `build.sh` runs under the system python whose torch segfaults, so the wheel is
  rebuilt with the conda interpreter afterwards.

And two device traps:

* the runner must call `torch_npu.npu.set_device()` before timing — `synchronize()`
  and `Event` act on the *current* device, so timing `npu:1` without it measures
  an unsynchronised stream (a 268 MB reduction "in 36 us");
* do **not** wrap an `OpExec`-based script in `flock` on the shared NPU lock
  (`easyasc_npu0.lock` in the board's system temp dir). `OpExec` takes that same
  lock itself, and the outer `flock` deadlocks it into the 1800 s "NPU busy:
  another easyasc on-board run has held ..." bailout. Only wrap scripts that run
  a pre-built package.

Per-operator companions live next to the kernel; for `adaptive_avg_pool_3d` they
are `board_floor.py` (the fixed per-call cost — half the public set finishes
inside it on some boards, so measure it before reading any geomean) and
`board_profile.py` (the AI-vector pipe breakdown out of `op_summary.csv`).

#### 8c. Exclude unrelated operators from the build

If you want every `build.sh` and `run_evaluation.sh` to touch only your
operator, rename the directories of other custom operators with a `_` prefix:

```bash
cd <cann-bench-root>/agent/example/demo/aclnn_launch_example/csrc/ops
mv exp _exp        # skip Exp in builds
mv mish _mish      # skip Mish (if you added it)
```

First record the exact directory list and reserve unique temporary names. Restore
them explicitly immediately after the build, even when the build fails, and
verify the before/after lists. Do not rely only on a remote-shell `trap`: a lost
SSH/session boundary can leave the shared checkout partially renamed.

The outer `csrc/ops/CMakeLists.txt` skips directories starting with `_`, so
they won't be compiled or registered. Rename back when you're done.

> **Default rule — only evaluate the operator you are working on.** Unless the
> user explicitly asks for a full-suite run (all operators), always exclude
> unrelated operators via the `_`-prefix rename before `build.sh` and
> `run_evaluation.sh`. A full run takes ~200 s per operator and wastes time
> — plus the stale-wheel trap (§5) and subprocess-environment issues (§6) can
> drown real failures in noise from operators you didn't touch.
> After the evaluation, restore the directory names so the repo is clean for
> the next session.

> **Restore-and-rebuild trap — a `_`-excluded build leaves a PARTIAL wheel.**
> Renaming the dirs back is NOT enough: `dist/` still holds the wheel/`.run`
> from the single-op build, which contains ONLY the operator you tested. The
> next full-suite run reuses that partial wheel (stale-wheel trap, §5/§8a) and
> every other operator scores `0.00` — their kernels aren't in
> `libcust_opapi.so` at all. Tell-tale: `run_extract` has `.o` for a single op,
> and `strings .../libcust_opapi.so | grep -oE 'aclnn[A-Za-z]*Kernel'` lists
> only that op's entrypoints. Fix — force a clean full rebuild after restoring
> the names, before any scored full run:
> ```bash
> rm -f dist/cann_bench*.whl dist/cann_bench*.run
> rm -rf dist/run_extract build/ build_py/
> bash build.sh --soc=ascend910b
> ```
> Verified 2026-07-06: a SwiGlu-only build left the other 5 ops at `0.00`
> (suite 176.69) until a full rebuild restored the complete wheel (445.34).

---

### 9. Run the benchmark evaluation

**Prerequisite:** the wheel must have been rebuilt after the §4 `__init__.py`
preload patch (see §5). Without the rebuild, the installed `cann_bench` package
lacks the preload and all custom-operator cases will fail with `0.00 μs` timing
and `0.000000` accuracy error (the subprocess cannot locate the custom OPP).

Pre-extract the `.run` and export `ASCEND_CUSTOM_OPP_PATH` in the calling shell
so the parent evaluation process can find `libcust_opapi.so` before spawning
subprocesses:

```bash
DIST_DIR=<cann-bench-root>/agent/example/demo/aclnn_launch_example/dist
CUSTOM_OPS=$DIST_DIR/run_extract/packages/vendors/custom_ops
if [ ! -d "$CUSTOM_OPS" ]; then
    cd $DIST_DIR && mkdir -p run_extract
    bash $(ls *.run | head -1) --noexec --extract=run_extract
fi
export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_OPS
# (LD_LIBRARY_PATH and PYTHONPATH will be auto-set by cann_bench/__init__.py)
```

On a shared checkout, do not replace an unrelated globally installed
`cann_bench` package for an A/B run. Extract each candidate `.run` into its own
directory and install its wheel with `pip --target` into an isolated
site-packages directory. Launch the benchmark with that directory first in
`PYTHONPATH` and with candidate-specific `ASCEND_CUSTOM_OPP_PATH` and
`LD_LIBRARY_PATH`. Record those paths and package hashes in the manifest; this
keeps two candidates reproducible and prevents one operator's package from
silently contaminating another evaluation.

Functional sweep:

```bash
cd <cann-bench-root>
PYTHONPATH=src python -m kernel_eval.cli eval \
  --operator <OperatorName> \
  --device-id 0 \
  --no-perf \
  -v
```

Use the exact operator filter on checkouts where the wrapper's `--task-dir`
expands to the whole level. Confirm the log reports the expected case count.

Quick profiler sanity check:

```bash
cd <cann-bench-root>
PYTHONPATH=src python -m kernel_eval.cli eval \
  --operator <OperatorName> \
  --case-id 1 \
  --device-id 0 \
  --warmup 1 \
  --repeat 2 \
  -v
```

Read the generated results under `<cann-bench-root>/reports/`.

### 10. Record findings and produce the batch summary

Treat `easyasc_cannbench_kernels/doc/13_cann_bench_evaluation_findings.md` as
the canonical evaluation ledger. A later summary must be derived from the current ledger and archived
artifacts, not from memory or selected successful cases.

For every evaluated operator:

1. Freeze the evaluated boundary: generated wheel and run package hashes,
   hardware target, case inventory and checksums, threshold mode, functional
   report, and profiler manifest.
2. Audit provenance. Confirm profiler CSVs contain the expected custom kernel
   rows and explicitly count any official ACLNN rows. A result with an
   official, PyTorch, CPU, or host-computed fallback is invalid/non-counting.
   For an optimization rerun, compare profiler row names/counts before and
   after as well as package hashes, so a timing improvement cannot come from a
   changed dispatch boundary.
3. Record the exact pass count, score, failed case IDs, error metrics, missing
   baseline rows, and unsupported dtypes or shapes. Do not describe an
   unsupported case as skipped when it contributes a benchmark failure.
4. Classify each issue as one or more of: kernel algorithm/numerics, EasyASC
   DSL/API gap, parser/codegen gap, simulator-versus-device discrepancy,
   generated ABI/wrapper integration, runtime/package loading, or benchmark
   harness/profiler defect.
5. Separate deterministic failures from sample-dependent behavior. For a
   suspected race, rerun fixed inputs several times and record whether the
   device outputs are bit-identical. Do not erase the original scored failure.
6. Preserve historical entries. A DSL fix, kernel optimization, or harness
   correction creates a new dated rerun entry linked to the old one; it never
   rewrites the score that was actually observed.
7. End a multi-operator batch with a compact comparison table and a follow-up
   list ordered by ownership: EasyASC framework, kernel, integration, then
   benchmark harness. Link each item to its source path or upstream document.
   For an optimization rerun, include exact before/after case timings from two
   immutable archives, the aggregation method, and whether missing benchmark
   baselines prevent formal performance points.

Before publishing the summary, verify that report paths and hashes still
resolve on the evaluation host, `git diff --check` passes locally, and every
claimed custom row has profiler evidence. If an artifact is unavailable, mark
it unavailable instead of reconstructing a stronger result.

## Validated examples

### Exp (2026-07-04)

- source project: `<cann-bench-root>/agent/example/demo/aclnn_launch_example`
- benchmark task: `kernel_bench/level1/exp`
- result: `20 / 20` passed
- score: `84.40`

### Sigmoid (2026-07-04)

- kernel: `agent/example/kernels/a2/vec_only/sigmoid_general.py` (3 dtypes: float32/float16/bfloat16)
- source project: `<cann-bench-root>/agent/example/demo/aclnn_launch_example` (merged into `csrc/ops/sigmoid/`)
- benchmark task: `kernel_bench/level1/sigmoid`
- result: `20 / 20` passed
- score: `68.01`
- gotcha: `csrc/ops/sigmoid/CMakeLists.txt` originally had empty-string `""`
  HOST_SRCS/API_SRCS — the host and API .cpp were never compiled, producing
  0/20 with `0.00 μs` every case. Fixed by diffing against
  `csrc/ops/exp/CMakeLists.txt` and substituting `"${...}"` variable refs.
  The pre-build deployment checklist in §3 was added as a direct result of
  this bug.

End-to-end workflow validated both times: gen_only → build templates → merge →
build cann-bench → manual smoke → full evaluation. The `__init__.py` preload +
env-var auto-set patch from §4 was the critical fix that made evaluation
subprocesses find the custom OPP tree.

## Failure routing

- `undefined symbol` from `libcust_opapi.so` during import:
  missing `libopapi.so` preload in `cann_bench/__init__.py`
- `aclnnXxxKernel not found in libcust_opapi.so`:
  `libcust_opapi.so` was loaded without `RTLD_GLOBAL` or not loaded at all;
  add `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` preload in `__init__.py`
- `binary_info_config.json does not support opType`:
  `PYTHONPATH` does not include the kernel script directories; the ACLNN
  Python framework cannot locate `exp_float_kernel.py` etc. to resolve the
  binary info config entry
- **All custom-op cases show `0.00 μs` timing and `0.000000` accuracy error,
  but built-in ops (e.g. Mish) pass:** the evaluation subprocess cannot find
  the custom OPP tree. The `__init__.py` preload block from §4 is either
  missing or was not rebuilt into the wheel. Verify: (a) the preload block
  exists in the source tree's `__init__.py`, (b) the wheel was rebuilt after
  the edit, (c) the `.run` has been extracted and `ASCEND_CUSTOM_OPP_PATH`
  points to the extracted `custom_ops` directory.
- `RuntimeError: Two accelerators cannot be used at the same time in PyTorch:
  npu and npu` during smoke: `torch_npu` was imported after `cann_bench`.
  Always `import torch, torch_npu` **before** `import cann_bench`.
- direct smoke passes but `run_evaluation.sh` fails:
  the child-process environment is missing `ASCEND_CUSTOM_OPP_PATH`,
  `LD_LIBRARY_PATH`, `PYTHONPATH`, or the benchmark subprocess runner
  ignored the active config; the `__init__.py` env-var auto-set from §4
  fixes the most common root cause
- **Small scalar-unit GM accesses are intermittently stale:** first distinguish
  `GMTensor.GetValue/SetValue` scalar access from DMA transfers. For a genuine
  scalar/DCache boundary, own at least one full 64-byte cacheline in the host
  allocation and use
  `clean_dcache(..., EntireType.single, DcciDst.out, mode="vec")`. Do not add
  DCCI based on output size alone. Huawei documents DMA GM traffic as bypassing DCache,
  `CACHELINE_OUT` as the output consistency target, and `CACHELINE_ATOMIC` as
  reserved/unsupported; do not cargo-cult `DcciDst.atomic`. See the
  [Huawei DataCacheCleanAndInvalid API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0177.html).
- **Some reduction cases fail only in the full ordered sweep:** inspect the
  producer's exact ceil-div worker assignment. If stage 2 consumes
  `min(tile_count, core_count)` rows while stage 1 actually initializes
  `ceil(tile_count / ceil(tile_count/core_count))`, trailing stale partial rows
  explain allocator-history-dependent errors. Add a regression tile count such
  as 256 over 40 cores (37 active rows), and keep the sequential all-case sweep.
- **TensorList cases ignore their declared value range:** inspect
  `DataGenerator._normalize_value_ranges`; a TensorList nested shape may be
  counted as one public input and collapse the range to `None`, producing the
  default `[0, 1]`. Treat this as a harness defect and verify the actual generated
  distribution before drawing value-range conclusions.
- **A YAML attribute written as bare `None` disagrees with the golden:** YAML
  parsers keep bare `None` as the string `"None"` (YAML null is `null` or `~`).
  If the public contract says Python `None`, `"None"`, and `"update"` are the
  same mode, normalize all three in both the wrapper and the benchmark golden.
  First print `repr(case.attrs["reduce"])` and prove the device output's changed
  positions against the indices; do not make a correct kernel return identity
  merely to match a golden that skipped every branch.
- **Functional kwargs pass, but profiler output is `NoneType` and every
  per-case profiler directory is empty:** compare the parameter-dict insertion
  order with the public schema. `ParamBuilder` may insert all tensors before
  attrs (for example `data, indices, updates, dim, reduce`), while `OpRunner`
  converts that dict directly to positional arguments even though the schema is
  `data, dim, indices, updates, reduce`. The functional path hides the defect by
  calling with kwargs; the profiler catches the positional type error inside
  `PerfResult.error` and leaves `last_outputs=None`. Preserve the golden
  signature order when building the parameter dict (and assert it in an
  isolated harness check) before trusting a scored rerun.
- profiler run completes but no `kernel_details.csv` appears:
  the profiler ownership fix must happen on the writable source path behind the
  read-only `/usr/local/Ascend/...` bind mount, not only inside the mount itself
- `--task-dir` still evaluates extra operators:
  inspect the benchmark-side operator matching logic and tighten it before
  claiming that the task filter is authoritative
- **One specific custom operator shows all `0.00 μs` / `0.000000` while other
  custom operators pass (e.g. Sigmoid fails but Exp passes):** the operator's
  `CMakeLists.txt` has `register_aclnn_op` calls with empty-string `""`
  HOST_SRCS or API_SRCS, so the host and API sources are never compiled into
  `libcust_opapi.so`. Compare the broken `CMakeLists.txt` against
  `csrc/ops/exp/CMakeLists.txt` (the validated template): every
  `register_aclnn_op` must pass `"${...}"` variable references — not `""` —
  and every path must use `${CMAKE_CURRENT_SOURCE_DIR}/op_host/...` instead
  of bare `/op_host/...`. Also check `register_aclnn_plugin` the same way.
  After fixing, the build output (wheel / `.run` size) should noticeably grow;
  a flat size after a CMakeLists-only edit means the fix didn't take.
- **Scores unchanged after kernel source changes:** `run_evaluation.sh` skips
  the build when it finds a wheel in `dist/`. Delete `dist/cann_bench*.whl`,
  `dist/cann_bench*.run`, and `build/` before re-running, or the old binary
  is used regardless of source changes.
- **Scores STILL bit-for-bit identical after deleting the wheel — stale
  profiler-csv reuse (the more insidious cause, and usually the real one behind
  "the optimization didn't move the score"):** `perf_eval.py` archives every
  profiler run under a directory keyed ONLY by case id
  (`reports/prof_data/{rel_path}/{caseid}/`) and never cleans it in archive mode,
  then locates `kernel_details.csv` via `os.listdir(...)` taking the FIRST entry
  (not newest-by-mtime). Every eval therefore reuses the OLDEST csv, freezing the
  score at the FIRST kernel version ever evaluated — recompiles, plugin fixes, and
  wheel rebuilds all have no effect. **Tell-tale:** `overall_score` reproduces to
  ~1e-13 across independent full runs (e.g. `442.4320764040093` four times), or an
  operator's per-case timing is byte-identical across reports — real-HW timing
  never repeats to 1e-13. **Fix without touching the framework:** before each
  scored run, `rm -rf reports/prof_data/level1/<op>` (or `mv` it aside). prof_data
  accumulates per operator, so clear the one you evaluate. Verified 2026-07-06:
  clearing SwiGlu's prof_data moved the score 76.11 → 76.69.
- **`--no-perf` timing won't reflect small kernel changes:** the wall-clock
  fallback includes Python dispatch overhead. For elementwise ops this overhead
  often dominates — kernel-level improvements may not change the reported score.
  Use `OpExec(profile=True)` (§8b) for accurate kernel timing. (Note: an
  unchanged score with the profiler ON — i.e. NOT `--no-perf` — is the stale-csv
  bug above, not a timing-method issue. Don't misattribute it to wall-clock.)
- **Score barely moves even after a genuine kernel speedup:** the per-case perf
  score is hardware-anchored — `(T_base−T_HW)/((T_cand−T_HW)+(T_base−T_HW))`
  (`results.py:get_perf_score`). Once the kernel is at/under baseline (speedup
  ≥1x), shaving a few percent barely changes the score, and a single-shape
  `OpExec(profile=True)` win (e.g. "bf16 ↓28%") is diluted across the 20-case
  distribution. Expect low-single-digit score deltas from tile/DBuff tuning.
- **A re-migrated kernel SIGSEGVs at runtime (subprocess rc=-11, every case
  0.00) — the DBuff / `tile_len` migration triple-trap:** when you change an
  easyasc kernel that uses `tile_len` or DBuff and migrate it back, replacing
  only `op_kernel/*` (or even `+ op_host/*`) is NOT enough. The process dies
  with no aclnn error — the eval harness swallows the stack; reproduce with a
  manual smoke (`import cann_bench; cann_bench.<op>(x)`) under
  `ASCEND_SLOG_PRINT_TO_STDOUT=1` to see the core dump. Three fixes, each cost
  a real-HW round (verified 2026-07-06, sigmoid DBuff, 68.34 → 71.53):
  1. **Regenerate `op_api` too.** `op_api/aclnn_<op>_kernel.cpp` embeds the
     workspace size + attr descriptors, which change when buffers double
     (DBuff) or the tile changes. A stale op_api computes the wrong workspace →
     device overrun.
  2. **Make the plugin pass `tile_len`** — copy `gelu_plugin.cpp`:
     `tile_len = n < TILE_MAX ? n : TILE_MAX;
     ACLNN_CMD(aclnn<Op>Kernel, xf, n, tile_len, yf);` — four logical args
     (input, attrs, output). A plugin passing only `(xf, n, yf)` mis-binds the
     args → crash. Use each dtype's own TILE_MAX.
  3. **Gen with `n ≫ tile_len`.** If the gen driver uses `n == tile_len`
     (e.g. n=2048, `pick_tile_len`→2048), easyasc treats it as single-tile and
     DROPS the tile_len attr from op_api (`attrDesc[]={1}` instead of `{1,1}`) →
     the plugin passes 4 args to a 3-arg API → SIGSEGV. Verify with
     `grep -c tileLen op_api/aclnn_<op>_*.cpp` == 2. Safest: copy op_api from an
     `OpExec(..., profile=True)` run's `build_out/autogen/` — its n is millions
     (≫ tile_len) and it is already real-HW-validated.
  **Diagnosis shortcut:** `OpExec(..., profile=True)` runs the kernel on real HW
  bypassing cann-bench entirely — if that passes, the bug is in the migration,
  not the kernel; then `diff` the profile-run tree vs the gen tree's
  `op_kernel/op_host/op_api` to pin the single differing file (it will be
  op_api).

## Cross-references

- `agent/references/code-paths.md`
- `doc/06_codegen_and_runtime.md`
- `doc/api/op_exec.md`
