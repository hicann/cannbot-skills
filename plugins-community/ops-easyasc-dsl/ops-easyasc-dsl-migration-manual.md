# Legacy Repo Migration Playbook

Use this playbook when you need to transform the original `easyasc` repository layout into the current `ops-easyasc-dsl` skill-oriented layout.

Read this file first for migration tasks.
Do not start by guessing folder moves from memory.

## Goal State

The migrated repository should end in this shape:

- repository root name: `ops-easyasc-dsl`
- `agent/` is the primary skill-facing surface
- `agent/SKILL.md` is the entrypoint
- `agent/ROUTER.md` drives progressive disclosure
- `agent/scripts/` holds repository-maintenance scripts
- `agent/example/kernels/` holds curated kernel examples
- `agent/example/demo/` holds manual demo programs
- `agent/assets/ops-easyasc-dsl-runtime.tar.gz` stores the archived `easyasc/`, `doc/`, and `doc_cn/` payload
- `agent/scripts/init.sh` restores `easyasc/`, `doc/`, and `doc_cn/` on demand
- `testcases/` is removed from the delivered skill bundle
- delivered `.py` / `.sh` / `.bash` files carry the required script license header
- delivered `.h` / `.cpp` files carry the required C/C++ license header

## Migration Order

Apply the transformation in this order so paths stay valid while you edit the repository:

1. Inspect the original tree.
2. Remove obsolete top-level content.
3. Move repository tools into the skill surface.
4. Move runnable examples into the skill surface.
5. Convert `agent/` into a real skill entrypoint.
6. Archive the runtime/docs payload and add an initialization script.
7. Rename the repository root.
8. Repair documentation, routes, and path references.
9. Apply the required license headers.
10. Validate the migrated repository from a fresh-user workflow.

## Detailed Steps

### 1. Inspect the original tree

Confirm the original repository still has the legacy layout:

- top-level `easyasc/`
- top-level `doc/`
- top-level `doc_cn/`
- top-level `kernels/`
- top-level `demo/`
- top-level `tools/`
- top-level `testcases/`
- existing `agent/` content that is not yet a complete skill entrypoint

Record which files describe structure or commands before moving anything:

- `README.md`
- `README_CN.md`
- `AGENTS.md`
- `CLAUDE.md`
- `agent/ROUTER.md`
- `agent/references/repo-map.md`
- `agent/references/code-paths.md`

### 2. Remove obsolete top-level content

- Delete `testcases/` from the delivered repository.
- Treat removed tests as historical context only; if they need to be mentioned, describe them as removed from the skill bundle rather than still available.

### 3. Move repository tools into the skill surface

- Create `agent/scripts/` if it does not exist.
- Move every file from top-level `tools/` into `agent/scripts/`.
- Preserve script filenames so existing references can be updated with simple path rewrites.
- Move or regenerate the tool summary file under `agent/scripts/tools_summary.md`.

Expected path rewrite:

- `tools/<name>` -> `agent/scripts/<name>`

### 4. Move runnable examples into the skill surface

- Create `agent/example/`.
- Move top-level `kernels/` into `agent/example/kernels/`.
- Move top-level `demo/` into `agent/example/demo/`.
- Remove the original top-level `kernels/` and `demo/` directories after the move.

Expected path rewrites:

- `kernels/<path>` -> `agent/example/kernels/<path>`
- `demo/<path>` -> `agent/example/demo/<path>`

### 5. Convert `agent/` into a real skill entrypoint

Add or rewrite these files:

- `agent/SKILL.md`: user-facing skill entrypoint
- `agent/ROUTER.md`: first routing layer with progressive disclosure

`agent/SKILL.md` should:

- explain that `agent/scripts/init.sh` must run before reading archived runtime/docs content
- point readers to `agent/ROUTER.md`
- describe the preferred read order: router -> one playbook -> one focused reference -> one source example
- list `agent/example/`, `agent/scripts/`, and `agent/assets/`
- avoid machine-specific absolute paths in environment guidance

`agent/ROUTER.md` should:

- route kernel authoring to `agent/playbooks/kernel-authoring.md`
- route kernel debugging to `agent/playbooks/kernel-debugging.md`
- route tool work to `agent/playbooks/tool-authoring.md`
- route documentation work to `agent/playbooks/doc-authoring.md`
- route repo-structure questions to `agent/references/repo-map.md`

### 6. Archive the runtime/docs payload and add an initialization script

Create a compressed archive that contains exactly:

- `easyasc/`
- `doc/`
- `doc_cn/`

Recommended artifact path:

- `agent/assets/ops-easyasc-dsl-runtime.tar.gz`

Then create:

- `agent/scripts/init.sh`

`init.sh` should:

- resolve the repository root relative to itself
- verify that the archive exists
- restore `easyasc/`, `doc/`, and `doc_cn/` only when missing
- be safe to run multiple times

After the archive and init script are in place:

- delete the unpacked `easyasc/`, `doc/`, and `doc_cn/` trees from the repository snapshot
- rely on `init.sh` to restore them when needed
- whenever you later edit restored files under `easyasc/`, `doc/`, or `doc_cn/`, rebuild `agent/assets/ops-easyasc-dsl-runtime.tar.gz` and delete the unpacked trees again so the delivered repository returns to the skill-only state

### 7. Rename the repository root

- Rename the top-level checkout directory from `easyasc` to `ops-easyasc-dsl`.
- Update every document and script that still mentions the old repository root name when the new name matters.

### 8. Repair documentation, routes, and path references

Update all docs and scripts that referenced removed or moved paths.

Mandatory rewrite categories:

- command examples that referenced `kernels/...`
- command examples that referenced `demo/...`
- tool references that referenced `tools/...`
- repository maps that still describe `testcases/`
- skill/router docs that do not yet mention `init.sh`
- environment guidance that hardcodes a machine-specific absolute path
- generated scripts or runtime helpers that hardcode a machine-specific CANN install root

Preferred style after the migration:

- use repo-relative paths such as `agent/example/kernels/...`
- use environment variables such as `ASCEND_HOME_PATH` instead of fixed host paths
- if an environment name is mentioned, present it as an example rather than a required default
- check both the visible repository files and the archived `easyasc/`, `doc/`, `doc_cn/` payload before declaring the migration complete

Practical repair checklist:

- update visible docs such as `README.md`, `README_CN.md`, `AGENTS.md`, `CLAUDE.md`, and `agent/SKILL.md`
- if the archive has already been restored once, inspect `easyasc/`, `doc/`, and `doc_cn/` for hardcoded host paths such as `/home/ubuntu/...` or `/usr/local/Ascend/...`
- update `easyasc/kernelbase/kernelbase.py` so generated scripts prefer environment variables over a fixed CANN path
- update any restored helper scripts under `easyasc/resources/` that still assume a fixed local install path
- normalize every validation example to `python agent/example/kernels/a5/matmul_float_mmad.py`
- after fixing restored files, rebuild `agent/assets/ops-easyasc-dsl-runtime.tar.gz` and remove the unpacked trees again

### 9. Apply the required license headers

Before final delivery, add the required license headers to all delivered source and script files.

For `.py`, `.sh`, and `.bash` files, prepend this header.
If a file has a shebang, keep the shebang as the first line and insert the header immediately after it:

```text
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
```

For `.h` and `.cpp` files, prepend this header:

```text
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
```

License-header checklist:

- apply headers to the delivered repository tree, not just to manually edited files
- include files under `agent/`, restored `easyasc/`, restored `doc/` helper scripts, generated `b.sh` / `r.sh`, and generated runtime source files if they are part of the delivered snapshot
- make the insertion idempotent so rerunning it does not duplicate headers
- after any validation run that creates new `.py`, `.sh`, `.bash`, `.h`, or `.cpp` files, rerun the license-header pass before finalizing the migration

### 10. Validate the migrated repository

Validate from the perspective of a fresh user:

1. start from the migrated repository root
2. run `bash agent/scripts/init.sh`
3. confirm that `easyasc/`, `doc/`, and `doc_cn/` are restored
4. run `python agent/example/kernels/a5/matmul_float_mmad.py`
5. confirm that the example validates both:
   - `OpExec(..., simulator=True)`
   - `OpExec(..., simulator=False)`
6. if the validation run generates fresh `b.sh`, `r.sh`, or runtime source files, rerun the license-header pass
7. rebuild `agent/assets/ops-easyasc-dsl-runtime.tar.gz` if the restored trees were edited during validation or repair
8. remove the unpacked `easyasc/`, `doc/`, and `doc_cn/` trees again so the final delivered repository stays skill-only

If the non-simulator path is slow, let it complete naturally instead of forcing a shell timeout into the repository script unless the user explicitly asks for that behavior.

## Files Commonly Updated During This Migration

- `README.md`
- `README_CN.md`
- `AGENTS.md`
- `CLAUDE.md`
- `agent/SKILL.md`
- `agent/ROUTER.md`
- `agent/references/repo-map.md`
- `agent/references/code-paths.md`
- `agent/references/examples/kernel-catalog.md`
- `agent/references/examples/tool-catalog.md`
- `agent/index/kernels.json`
- `agent/index/tools.json`
- `agent/playbooks/*.md`
- `agent/scripts/tools_summary.md`
- restored `easyasc/kernelbase/kernelbase.py`
- restored `easyasc/resources/debug/build.sh`
- restored `easyasc/resources/debug/run.sh`
- restored `doc/01_quickstart.md`
- restored `doc_cn/01_quickstart.md`
- restored `doc_cn/index.md`

## Anti-Patterns

Avoid these mistakes during the migration:

- leaving duplicate live copies of `kernels/` or `demo/` at the repository root
- keeping `testcases/` in the delivered skill bundle after claiming it was removed
- hardcoding a workstation path such as `<user-home>/...` into user-facing docs
- hardcoding a generic fallback such as `/usr/local/Ascend/...` into delivered scripts when the repository is supposed to be environment-driven
- describing archived `easyasc/`, `doc/`, or `doc_cn/` as always present
- forgetting to update validation commands after moving examples
- fixing restored archive contents but forgetting to rebuild `agent/assets/ops-easyasc-dsl-runtime.tar.gz`
- running validation, generating new source files, and forgetting to add license headers to those generated files
- introducing a shell `timeout` into repository scripts just to simplify one local verification run

## Completion Check

The migration is complete when all of the following are true:

- `agent/SKILL.md` is the repository entrypoint
- `agent/ROUTER.md` supports progressive disclosure
- `agent/scripts/` owns the former tool scripts
- `agent/example/` owns the former `kernels/` and `demo/`
- `agent/assets/ops-easyasc-dsl-runtime.tar.gz` exists
- `agent/scripts/init.sh` restores the archived trees
- `testcases/` is gone
- visible files and archived payloads no longer hardcode machine-specific absolute paths
- delivered `.py` / `.sh` / `.bash` files carry the required script license header
- delivered `.h` / `.cpp` files carry the required C/C++ license header
- docs no longer tell users to use stale paths
- a fresh-user validation passes from `python agent/example/kernels/a5/matmul_float_mmad.py`
