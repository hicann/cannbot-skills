- The user-facing skill entrypoint is `skill/SKILL.md`; the reusable workflow lives under `agent/`.
- Before reading archived runtime/docs or examples, run `bash agent/scripts/init.sh` to restore `easyasc/`, `doc/`, `doc_cn/`, and `agent/example/`.
- Start every conversation by reading `agent/ROUTER.md` first
- Follow the router-first reading order:
  1. read `agent/ROUTER.md`
  2. read only one matching playbook for the current task when possible
  3. read only the focused constraint, pattern, example, or map file that the playbook/router points to
  4. read source files only after the smaller guidance layers stop being enough
- Do not start by loading giant summaries when a smaller route already answers the task
- High-level owner files:
  - `agent/references/repo-map.md`: top-level repository layout and ownership map
  - `agent/references/code-paths.md`: implementation-path lookup for operations and behaviors
  - `doc/11_architecture_for_contributors.md`: contributor-level architecture and subsystem ownership fallback (restored on demand)
  - `agent/scripts/tools_summary.md`: tool fallback; prefer `agent/references/examples/tool-catalog.md` and `agent/playbooks/tool-authoring.md` first
- For kernel authoring tasks:
  - **first** do `agent/playbooks/clarify-first.md` to settle the Contract Template before any code — this is a hard prerequisite, not an optional step
  - then start `agent/playbooks/kernel-authoring.md`
  - then read only the specific constraint files you need
  - then use `agent/references/examples/kernel-catalog.md` to choose source examples
- For kernel debugging tasks:
  - start with `agent/playbooks/kernel-debugging.md`
  - then read only the matching constraint file for the suspected failure mode
- For tool changes:
  - start with `agent/playbooks/tool-authoring.md`
  - use `agent/references/examples/tool-catalog.md` before opening tool code
- For documentation changes:
  - start with `agent/playbooks/doc-authoring.md`
  - then use `README.md`, `README_CN.md`, `agent/references/repo-map.md`, and `doc/11_architecture_for_contributors.md` as needed
- For repo-structure questions:
  - start with `agent/references/repo-map.md`
  - then use `agent/references/code-paths.md` if the question is really about implementation location
  - then use `doc/11_architecture_for_contributors.md` only when the smaller maps are not enough

## Repository working rules

- Before guessing, inspect the real implementation path for the target behavior inside this repository.
  - check the relevant files under `easyasc/stub_functions/`
  - then parser lowering under `easyasc/parser/`
  - then simulator behavior under `easyasc/simulator/` when execution semantics matter
- Existing kernels are for study, not copy-paste. Use them to understand repository semantics, constraints, and synchronization patterns.
- Unless the user explicitly requests otherwise or the task is already tied to an existing kernel file, implement a new kernel in a new file by default.
- In this repository, "kernel" means a function decorated with `@kernel`. User-visible inputs and outputs must map strictly to the `GMTensor` arguments and returned `GMTensor` objects of that `@kernel` function.
- Build kernels incrementally: reason about tile strategy first, then implement and validate stage by stage instead of trying to finish the whole kernel in one jump.
- Justify each operation, cast, buffer, synchronization edge, and datamove from the target formula and hardware behavior. Do not keep a step only because another kernel happened to use it.
- Before golden inputs enter the kernel, only shape-only transforms are allowed, such as `squeeze`, `unsqueeze`, and `reshape`. Do not use `expand`, `tile`, `permute`, or other layout-changing transforms unless the user explicitly wants that behavior.
- If any part of the reasoning is ambiguous or cannot be justified from repository evidence, stop and ask instead of guessing. For kernel authoring specifically, the Contract Template in `agent/playbooks/clarify-first.md` must be fully settled before you begin Step 1 of `agent/playbooks/kernel-authoring.md`; do not substitute "reasonable assumptions" for questions on fields the playbook marks ASK.
- Treat warnings as signals that the mental model is incomplete. Investigate the root cause instead of accepting a passing result with unresolved warnings.
- For `auto_sync` warnings specifically, understand the synchronization model first, then either adjust the kernel or propose a concrete parser/autosync change.

## Update owner docs, not giant summaries

When stable repository knowledge changes, refresh the owner files that actually describe that area:
- repository structure or subsystem ownership -> `agent/references/repo-map.md` and/or `doc/11_architecture_for_contributors.md`
- implementation-path lookup -> `agent/references/code-paths.md`
- kernel additions or meaningful kernel changes -> `agent/references/examples/kernel-catalog.md`
- tool additions or tool behavior changes -> `agent/scripts/tools_summary.md`
- demo boundaries -> `agent/example/demo/README.md`
- documentation entry maps -> `README.md`, `README_CN.md`, and the affected `doc/` or `doc_cn/` pages

- If a `torch210npu` conda environment is available it is a convenient example environment; otherwise use the default environment with the required dependencies.
- Typehints should be compatible with python3.8.
- All codes, error messages, and readme files should be written in English.
- Refer to the `agent/example/kernels` folder for templates.

## Environment setup

The simulator-only path (`OpExec(..., simulator=True)`) needs only a working Python + PyTorch environment. The non-simulator path (`OpExec(..., simulator=False)`, `cannsim` smoke runs, and real on-device validation) additionally needs a Linux host and a local CANN install.

### Python / conda environment

- Create (or reuse) a Python environment. A conda env named `torch210npu` is the conventional example; any environment with the listed dependencies works.

  ```bash
  # example only — adjust to your local setup
  conda create -n torch210npu python=3.10 -y
  conda activate torch210npu
  ```

- Install the Python dependencies listed in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  Current `requirements.txt` contents: `numpy`, `torch`, `torch-npu`, `einops`, `prettytable`, `rich`, `sympy`, `setuptools`, `decorator`, `scipy`, `attrs`, `psutil`. Keep `torch` / `torch-npu` aligned with your local CANN version. CANN itself is not a pip package and must be installed separately (see below).

- Typehints in this repository must stay compatible with Python 3.8.

### CANN install (Linux only)

`cannsim` and real on-device validation require a Linux host plus a local CANN toolkit install. macOS / Windows are **not** supported for the non-simulator path.

- Download the CANN package matching your target architecture from:
  `https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260415000324328/`
- Install the toolkit following the official CANN install guide, then export the install root so generated `b.sh` / `r.sh` scripts and restored helpers can find it:

  ```bash
  export ASCEND_HOME_PATH=<path-to-your-CANN-install>   # the directory that contains bin/setenv.bash
  # optional: prepend a Python-with-NumPy for CANN opbuild
  # export EASYASC_PYTHON_BIN=<conda-env>/bin
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
  ```

- Do not hardcode workstation-specific paths such as `/home/<user>/...` or `/usr/local/Ascend/...` into the delivered repository. Delivered scripts must stay environment-driven via `ASCEND_HOME_PATH` / `ASCEND_INSTALL_PATH`.

### What runs without CANN vs. what requires it

- Simulator-only kernel validation (`OpExec(..., simulator=True)`) — works on any OS with PyTorch installed; no CANN needed.
- `OpExec(..., simulator=False)`, `cannsim` smoke runs, and real NPU validation — **require** a Linux host and a local CANN install with `ASCEND_HOME_PATH` set.

## Python formatting

For Python code, do not automatically rewrite function signatures or calls into a one-argument-per-line style.

- Prefer compact formatting.
- Keep arguments on the same line if the result is still easy to read.
- Wrap only when necessary for readability or line length.
- If wrapping is needed, use a compact multi-line layout instead of placing every argument on its own line unless there is a clear readability benefit.
- Avoid unnecessary trailing commas that encourage vertical expansion.
