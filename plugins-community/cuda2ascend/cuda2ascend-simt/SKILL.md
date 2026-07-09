---
name: cuda2ascend-simt
description: "将 CUDA 算子/Kernel 迁移到 Ascend C SIMT。触发：CUDA 转 Ascend、CUDA 迁移昇腾、cuda kernel porting to Ascend、CUDA to Ascend SIMT、CUDA 代码移植 NPU、GPU 算子转 NPU、.cu 转 .asc、CUDA 转 NPU。Use when migrating CUDA operators, kernels, or related CUDA implementations to Ascend C SIMT. Suitable for CUDA-to-Ascend migration, CUDA kernel porting, CUDA runtime or device API translation, or delivering a migrated result as a standalone sample, a torch_npu extension project, or a pybind extension project. This skill produces Chinese migration execution documents."
---

# cuda2ascend-simt

## Overview

Use this skill to turn an existing CUDA operator into an Ascend C SIMT implementation that can be built, tested, and, when the environment allows, validated on Ascend 950 PR hardware.

Default to one-to-one migration fidelity. Preserve reusable abstraction layers, dtype coverage, shape-specialized paths, kernel-selection behavior, algorithm structure, and user-facing invocation paths unless the local Ascend C SIMT grammar or constraint references show that the original CUDA form cannot be preserved directly.

Before writing `plan.md` or implementation code, always determine exactly one migration mode:

- `sample`
- `torch_npu`
- `pybind`

This skill is intended for publishing and reuse. Keep the workflow portable, explicit, and document-driven.

Preserve the original engineering form and invocation path whenever possible. Never silently downgrade a task to `sample`; if a downgrade is required, record the reason in both `plan.md` and `README.md`.

Output code and project files under:

```text
ported-ops/<operator_name>/
```

The generated engineering documents must be written in Chinese:

- `plan.md`
- `README.md`

Keep code, APIs, file names, and error messages in their original technical language where appropriate.

## Non-Negotiable Rules

- Before writing `plan.md`, always determine exactly one migration mode: `sample`, `torch_npu`, or `pybind`.
- Never guess Ascend C SIMT APIs, runtime behavior, or compiler requirements.
- Always consult local references before replacing CUDA APIs or rewriting CUDA syntax.
- Preserve the original engineering form and invocation path whenever possible.
- Never default to `sample` only because it is easier to implement.
- If mode downgrade is unavoidable, document the reason before implementation starts. Silent downgrade is forbidden.
- Do not silently reduce dependency paths, dtype coverage, behavior branches, or test scope.
- Every omitted capability must be explicitly classified as `downgrade`, `blocked`, or `excluded` with reason, impact, and recovery conditions.
- Use `downgrade` only when a supported but reduced implementation is intentionally accepted.
- Use `excluded` for unsupported removable subpaths that are left out while supported source-visible behavior remains.
- Use `blocked` when the selected migration path cannot continue without missing environment support, unsupported required behavior, or explicit user approval.
- Preserve source behavior one-to-one whenever feasible. If the source abstraction shape, launch policy, kernel split, helper layering, or dispatch structure can be preserved, preserve it.
- For built-in torch operators, preserve reusable upper-layer abstractions such as `gpu_kernel` whenever feasible: directly include shared public abstractions that are reusable across CPU, CUDA, and HIP, and recursively migrate CUDA-specific dependencies by providing Ascend C SIMT counterparts instead of flattening them into per-operator special-case code.
- For built-in torch operators, if a reusable CUDA-specific backend layer such as `TensorIterator` execution, `gpu_kernel`, `Loops.cuh`, shape-specialized launch policy, or shared math helpers is required by multiple sibling operators, treat implementation of an Ascend C SIMT counterpart for reuse as the default first-choice plan.
- Only rewrite syntax, qualifiers, or execution form when `reference/grammar.md`, `reference/constraints.md`, or their rule files show that the original CUDA form is unsupported or restricted on Ascend C SIMT. When a workaround exists, choose the narrowest documented workaround that preserves source-visible behavior as closely as possible.
- `out-of-tree` delivery is the default accepted path in this repository and does not by itself trigger a major-downgrade approval gate. Still document the reason if the source artifact form differs from the delivered form.
- The following are major downgrades and require explicit user approval before implementation starts:
  - failing to preserve a one-to-one reusable abstraction layer and instead flattening it into operator-local special-case code
  - collapsing multiple shape/layout/kernel-selection/hardware-specialized paths into one generic simple kernel
  - replacing device execution for a core path with host fallback
  - removing source-visible fast paths driven by shape, layout, stride, dtype, or hardware information
  - narrowing reuse scope from a shared backend counterpart to a one-off operator patch
- In this repository, the following major downgrades are always hard-stop approval gates:
  - abstraction-layer downgrade: source reusable abstractions such as `TensorIterator`, `gpu_kernel`, `Loops.cuh`, shared math helpers, or shape-specialized dispatch helpers are not migrated one-to-one and the plan becomes "just make the operator work"
  - kernel-selection downgrade: source multi-path kernel dispatch based on shape, stride, contiguousness, vector width, last-dim/general-dim split, problem size, or hardware information is simplified into a single path
- When a hard-stop approval gate is triggered:
  - stop after feasibility analysis and before implementation
  - do not create or modify implementation files
  - do not treat documenting the downgrade in `plan.md` or `README.md` as approval
  - present options to the user and wait for explicit approval
- When a major downgrade seems likely, stop after feasibility analysis and present options to the user. The options must compare migration cost, migration difficulty, migration benefit, impacted operators, validation implications, and the recovery path back to the preferred reusable design.
- Silent fallback to a simple flat implementation is forbidden unless the user explicitly accepts that downgrade in the interaction.
- The user-facing approval message must contain these sections:
  - `Detected downgrade`
  - `Why this is a downgrade`
  - `Option A: preserve one-to-one abstraction/kernel split`
  - `Option B: downgraded fallback`
  - `Impact on reuse`
  - `Impact on validation`
  - `Recovery path`
  - `Please choose`
- Use `assets/approval-message-template-zh.md` as the required Chinese skeleton when a hard-stop approval gate is triggered.
- Preserve the original algorithm structure, variable naming, and testing intent whenever possible.
- No implementation change without a verification target. Define what will prove correctness before changing code.
- Prefer test-first migration work. For each migrated behavior, add or update the relevant test before finalizing implementation whenever the environment allows it.
- No repeated blind fixes. If validation fails, investigate the root cause before changing more code.
- Define scope and non-goals before implementation. Do not expand scope during migration unless a blocker forces a documented plan update.
- If `reference/constraint_rules.yaml` classifies an unsupported item with `action: stop_and_report`, stop and report `UNSUPPORTED_FEATURE_ERROR`.
- If `reference/constraint_rules.yaml` classifies an unsupported item with `action: remove_and_record`, treat it as an unsupported removable subpath: exclude it from implementation and validation, record the exclusion in `plan.md`, and continue only if supported source-visible behavior remains.
- Native JIT compilation or loading paths are unsupported in the current migration model. Do not preserve, generate, or validate `nvrtc`, runtime compilation, extension JIT loading, or other on-the-fly native-code compilation/loading paths in migrated outputs.
- Do not classify PyTorch dispatcher metadata, `torch.library.register_fake`, FakeTensor/meta kernels, or Python wrapper code kept only for `torch.compile` compatibility as native JIT compilation/loading paths unless they compile or load native code at runtime.
- Complex dtypes are unsupported removable dtype subpaths in the current migration model. For torch migration work, detect complex dtype branches explicitly and exclude complex dtype implementation and validation paths from the migrated result. If no supported non-complex behavior remains, report the migration as blocked.
- Device-side `double` paths are unsupported removable dtype subpaths in one-to-one work. Detect dtype dispatch branches, kernel overloads, device helpers, FP64 intrinsics, and validation cases that require device-side `double`; exclude those paths from implementation and validation, and record the exclusion in `plan.md`. Do not rewrite such paths to `float` as an implicit downgrade. Host-side `double` scalar parameters may be preserved only when they are not a device-side double execution path and the device computation remains in a supported dtype. If no supported non-double behavior remains, report the migration as blocked.
- Never report `success` unless the migrated operator was actually validated on Ascend 950 PR hardware.
- If hardware validation is missing, incomplete, or failed, report the result as incomplete, blocked, or failed, not successful.
- Always place migration outputs in `ported-ops/<operator_name>/`.
- Always generate `plan.md` and `README.md` in Chinese using the provided templates.
- Always keep final status wording aligned with `reference/validation-checklist.md`.
- Evidence before claims. Any completion, success, pass, or fix claim must be backed by fresh build or test evidence.

## Resource Loading Rules

Load only the resource needed for the current step.

- Treat `reference/api-mapping/device_api.yaml` and `reference/api-mapping/runtime_api.yaml` as the source of truth for API mapping status.
- Use `reference/api-mapping/device_api.md` and `reference/api-mapping/runtime_api.md` as human-readable browse views, not as the primary mapping database.
- When working on runtime API references, use the subcategories in `reference/api-mapping/runtime_api.md` to narrow the current edit scope, such as `Device Management`, `Stream Management`, `Memory Management`, or `Peer Device Memory Access`.
- For `reference/api-mapping/runtime_api.yaml`, encode the matching runtime subcategory in `category` using the form `runtime/<subcategory-kebab-case>`, such as `runtime/device-management` or `runtime/memory-management`.
- Load `reference/api-mapping/device_api.yaml` when translating CUDA device APIs, intrinsics, thread-level primitives, or math helpers.
- Load `reference/api-mapping/runtime_api.yaml` when translating host runtime behavior such as memory allocation, memcpy, stream management, launch setup, and error handling.
- Load `reference/grammar_rules.yaml` as the source of truth for syntax rewrite rules.
- Load `reference/grammar.md` as the human-readable browse view for syntax rewrites.
- Load `reference/constraint_rules.yaml` as the source of truth for unsupported or restricted feature rules.
- Load `reference/constraints.md` as the human-readable browse view for migration constraints.
- Load `reference/validation-checklist.md` before making any final success or completion claim.
- Load `reference/schema.md` when updating or extending the mapping references.
- Load `reference/rule_schema.md` when updating or extending syntax rules or constraint rules.
- Use `reference/example/` as a pattern library of manually migrated operator pairs when you need a concrete migration example.
- For `sample` mode, prefer `reference/example/cuda-sample/` and `reference/example/simt-sample/`.
- For `torch_npu`-style custom operator work, prefer `reference/example/cuda-torch/` and `reference/example/simt-torch/`.
- When using `reference/example/simt-torch/`, preserve the original project layout where possible and treat the SIMT-side `extension_cpp/` structure as the primary reference shape.
- CUDA-side source projects may use different registration styles, but for SIMT-side torch migration work, follow the SIMT torch registration style demonstrated by the target example.
- Do not treat `reference/example/` as a substitute for API mapping references, constraints, or official documentation.
- Load `assets/approval-message-template-zh.md` when a hard-stop approval gate is triggered and a user-facing approval message must be produced.
- Load `assets/plan-template-zh.md` before drafting `plan.md`.
- Load `assets/readme-template-zh.md` before drafting `README.md`.

## Workflow

Execute the following steps in order. Do not skip steps or merge them into a vague summary.

### Step 0: Confirm the migration mode

After reading the source entry points and project structure, determine exactly one migration mode:

- `sample`
- `torch_npu`
- `pybind`

Selection rules:

1. Use `sample` when the source artifact is already a standalone CUDA sample or a small standalone project that can be built and run directly.
2. Use `torch_npu` when the source artifact is a torch extension, a custom operator, or the requested delivery must be invokable from Python through `torch_npu`.
3. Use `pybind` when the source artifact is already a `pybind` extension project.

Execution rules:

- Do not start `plan.md` before the mode is decided.
- Do not downgrade `torch_npu` or `pybind` work into `sample` merely for implementation convenience.
- If the current repository, dependency chain, build chain, or registration path cannot preserve the original engineering form, document the blocker before deciding on any downgrade.
- `sample` mode must keep `main` as the final execution entry.
- `torch_npu` mode must not use `main` as the final delivery entry and must preserve a Python invocation path.
- `pybind` mode must not use `main` as the final delivery entry and must preserve a Python invocation path.
- If a downgrade is required, `plan.md` must explicitly record:
  - source artifact form
  - target migration mode
  - actual output mode
  - downgrade reason
  - current blocker
  - what is required to recover the target mode later

### Step 1: Inspect the CUDA operator

Read the CUDA source files, build files, launch path, and tests.

Identify:

- Host entry points
- `__global__` kernels
- `__device__` helper functions
- CUDA runtime dependencies
- Existing UT/ST structure
- Source artifact form: `sample`, `torch_npu`, `pybind`, or unknown until resolved
- Source invocation form: `main`, Python invocation, extension binding, or another host entry path
- Dependency classes required by the selected path:
  - public shared abstractions that may be reused by direct include
  - CUDA-specific dependencies that may require recursive migration or an Ascend C SIMT counterpart
- Capability surface:
  - dependency path coverage
  - dtype coverage
  - behavior branches such as alternate host paths or multi-branch logic
  - layout and stride assumptions
  - shape-driven or size-driven fast paths
  - specialized kernel selection rules such as last-dim path, contiguous path, vectorized path, or small/large problem split
  - test entry paths and validation shape

Record the initial analysis and the mode-selection inputs as planning notes to be transferred into Chinese `plan.md`. Do not create the formal `plan.md` until Step 3.

### Step 2: Run feasibility analysis

Before writing code:

- define the migration scope and explicit non-goals
- build a CUDA-to-Ascend API mapping list
- group runtime APIs by the matching subcategory in `reference/api-mapping/runtime_api.md` before investigating or updating them
- identify syntax changes required by Ascend C SIMT
- check unsupported features and workaround limits
- detect whether the source contains native JIT-related code paths such as `nvrtc`, runtime compilation, extension JIT loading, or other on-the-fly native-code compilation/loading entry points
- distinguish native JIT compilation/loading paths from PyTorch dispatcher metadata such as `torch.library.register_fake`, FakeTensor/meta kernels, or Python wrapper code kept only for `torch.compile` compatibility
- detect whether the source contains complex dtype branches, complex tensor checks, or complex-number helper paths
- detect whether the source contains device-side `double` execution paths, including double dtype dispatch, double tensor checks, double kernel overloads, double device helpers, FP64 intrinsics, or double-specific validation cases
- for every planned syntax rewrite or execution-form rewrite, record the exact unsupported or restricted item from `reference/grammar*` or `reference/constraint*` that justifies the change
- record device-side `double` paths as excluded from implementation and validation instead of planning a float-rewrite downgrade
- confirm whether the selected mode preserves the original engineering form
- identify whether registration logic or binding logic belongs to the required migration scope
- determine whether Python invocation is a required validation target
- decide explicitly whether any native JIT path exists and record whether it will be removed with `remove_and_record` or blocks the selected migration path
- for `torch_npu` mode, decide explicitly whether complex dtype logic exists and record that complex dtype code paths will not be migrated
- for `torch_npu` mode, decide explicitly whether double dtype logic exists and record that double dtype code paths will not be migrated
- classify each dependency as `reuse`, `migrate`, `downgrade`, `blocked`, or `excluded`
- classify each capability item as `preserve`, `reuse`, `migrate`, `downgrade`, `blocked`, or `excluded`
- decide explicitly for each reusable abstraction layer whether it will be kept by direct include, reimplemented as an Ascend C SIMT counterpart, or marked as `downgrade`, `blocked`, or `excluded` with reason
- decide explicitly whether reusable backend work should be lifted above the current operator so sibling operators can reuse it later; examples include unary math helpers, `TensorIterator` execution layers, and shape-specialized dispatch helpers
- analyze whether different shape, stride, contiguousness, vector width, last-dim/general-dim, or small/large problem regimes require distinct kernel implementations or launch policies on Ascend
- confirm whether dtype coverage, branch coverage, and test scope still match the source-visible behavior
- explicitly answer before coding:
  - did I preserve the original reusable backend abstraction one-to-one
  - did I preserve the source multi-path kernel-selection behavior one-to-one
  - if either answer is no, did the user explicitly approve the downgrade

If a hard blocker is found, stop implementation and document it in Chinese.
If the likely implementation would require a major downgrade, stop after feasibility analysis and ask the user to choose a migration option before coding.

### Step 3: Build the migration plan

Create `ported-ops/<operator_name>/plan.md` in Chinese.

Use `assets/plan-template-zh.md` as the required structure.

The plan must include:

- migration mode selection
- source artifact form and target delivery form
- form-difference note if the actual output form differs from the source form
- downgrade reason, current blocker, and recovery conditions when a real downgrade exists
- dependency closure analysis
- capability coverage matrix
- unsupported-path handling for native JIT-related code
- unsupported-dtype handling for complex dtype code when source torch code contains such branches
- unsupported-dtype handling for device-side `double` paths when source code contains such branches
- CUDA function inventory
- call chain analysis
- API mapping analysis
- per-function migration strategy
- reusable Ascend counterpart strategy and reuse candidates across sibling operators
- downgrade decision gate when applicable
- execution order
- risks and blockers
- validation plan

Do not invent sections outside the template unless the user explicitly asks for them.

Before implementation starts, review the plan critically:

- check that the selected mode matches the source artifact form and the required user-facing invocation path
- check that every required dependency is classified as `reuse`, `migrate`, `downgrade`, `blocked`, or `excluded`
- check that every source-visible capability is classified as `preserve`, `reuse`, `migrate`, `downgrade`, `blocked`, or `excluded`
- check that any detected native JIT path is classified explicitly as `remove_and_record` or `blocked` with the unsupported reason
- check that any PyTorch fake/meta or `torch.compile` compatibility registration is not mislabeled as native JIT unless it compiles or loads native code at runtime
- check that any detected complex dtype path in torch migration work is classified explicitly as `remove_and_record` or `blocked` with the unsupported reason
- check that any detected device-side `double` path is classified explicitly as `remove_and_record` or `blocked`, excluded from implementation and validation when removable, and recorded with affected source files, functions, dtype branches, and validation cases
- check that shape/layout/kernel-selection behavior is classified explicitly rather than implicitly flattened away
- check that any planned major downgrade is converted into a user-facing option decision instead of being assumed
- check that every intended migration step maps to a concrete file or function
- check that every planned code change has a validation method
- check that blockers and assumptions are explicit
- check that `out-of-tree` delivery is not mistakenly treated as a major downgrade by itself
- check that the plan records explicit yes/no decisions for abstraction one-to-one preservation and multi-path kernel preservation
- if either decision is `no`, check that the 10.0 approval gate records an explicit user approval before any implementation work starts

If the plan has critical gaps, fix the plan first instead of pushing uncertainty into implementation.

### Step 4: Run the environment gate

Before treating the migration as runnable, verify the target environment.

Minimum checks:

- Ascend device nodes are present
- Ascend environment can be initialized
- required compiler and build tools are available

If the environment is not ready:

- keep the analysis and migration plan
- do not claim runnable success
- clearly state that Ascend 950 PR validation could not be completed

### Step 5: Implement the migration

Create the migrated operator under:

```text
ported-ops/<operator_name>/
```

Expected artifacts depend on the selected migration mode.

If mode is `sample`, expected artifacts usually include:

- one or more `.asc` files
- host code with `main`
- `CMakeLists.txt`
- `README.md`
- `test/UT` and `test/ST` when needed

If mode is `torch_npu`, expected artifacts usually include:

- one or more `.asc` files
- host wrapper or launch code
- registration code
- Python invocation entry
- installable standalone project files
- build and install instructions
- test instructions separated from packaging instructions
- Python-side verification scripts or examples
- `README.md`
- existing project UT/ST or new tests when missing

If mode is `pybind`, expected artifacts usually include:

- one or more `.asc` files
- host wrapper or launch code
- `pybind` binding code
- installable standalone project files
- build and install instructions
- Python-side verification scripts or examples
- `README.md`
- existing project UT/ST or new tests when missing

Implementation rules:

- prefer direct API replacement when valid
- rewrite only the incompatible parts
- if the selected path depends on a public upper-layer abstraction shared across CPU, CUDA, and HIP and directly includable from the installed package, prefer reuse by include instead of remigration
- if the selected path depends on a CUDA-specific file, helper, or backend-private path, analyze the dependency recursively and complete migration of the required chain
- if the selected path depends on a CUDA-specific abstraction that is likely reusable by sibling operators, prefer implementing an Ascend counterpart at that abstraction layer before writing operator-local kernels
- preserve source kernel partitioning, shape-based dispatch, and launch-policy branching by default; only merge or simplify them when the user explicitly approves that downgrade
- when a syntax rewrite is necessary, apply the minimum rewrite required by the documented Ascend C SIMT grammar or constraint limitation instead of redesigning the operator around that limitation
- do not replace a reusable CUDA abstraction with ad hoc per-operator code unless the plan explicitly records why direct reuse or an Ascend C SIMT counterpart is not viable
- do not collapse shape-specialized or layout-specialized source behavior into a single simple kernel unless the user has explicitly approved that downgrade after reviewing the tradeoff
- do not reduce dtype coverage, branch coverage, or dependency closure without documenting a per-item downgrade
- remove native JIT-related migration targets instead of trying to preserve runtime compilation, extension JIT loading, or other unsupported native-code compilation/loading forms
- preserve PyTorch dispatcher fake/meta registrations or `torch.compile` compatibility metadata when they are part of the Python invocation path and do not compile or load native code at runtime
- when source torch code contains complex dtype branches, keep dtype guards explicit and do not emit complex dtype kernels, registrations, tests, or validation cases in the migrated output
- when source code contains device-side `double` paths, keep guards explicit where needed and do not emit double kernels, double device helpers, double registrations, double tests, or double validation cases in the migrated output
- migrate leaf device helpers before higher-level callers where possible
- preserve observable behavior before considering optimization
- keep the build and test flow explicit
- when `torch_npu` or `pybind` mode is selected, kernel migration alone is insufficient; the user-facing invocation path must also be migrated
- for each migrated behavior, prefer a red-green cycle inside this implementation step:
  - define or update the relevant test or targeted validation
  - run it to observe the expected failure, missing coverage, or unsupported state when the environment allows it
  - implement the minimal migration change
  - rerun the targeted validation before moving to the next unrelated rewrite
- do not batch many unrelated rewrites before running validation

### Step 6: Validate the migrated result

Final validation order after implementation:

1. build the migrated target
2. run UT or targeted native-side validation
3. run end-to-end tests when available or required
4. run ST when available or required
5. run mode-specific Python, import, install, or package validation when required
6. re-read the actual command output before making any claim
7. record outcomes, commands, and blockers in Chinese documents

Mode-specific validation requirements:

- `sample` mode may use `main` as the validation path when `main` already contains test data, result checking, and failing exit behavior; otherwise add UT/ST
- `torch_npu` mode must verify build, install, and Python invocation behavior
- `pybind` mode must verify build, install, import, and Python invocation behavior
- `torch_npu` and `pybind` modes should reuse existing project UT/ST when available
- if no project UT/ST exists for `torch_npu` or `pybind`, add `gtest`-based tests and validate through runtime APIs directly
- `torch_npu` and `pybind` modes must add Python-side tests
- for `torch_npu`, keep CMake test commands and packaging commands as separate subcommands; do not merge them into one command path
- do not add validation commands for removed native JIT paths
- treat preserved `torch.compile` compatibility checks as Python-side compatibility tests, not as evidence that native JIT compilation/loading was migrated
- for torch migration work with detected complex dtype branches, validate only the supported non-complex dtype paths and record the complex dtype exclusion in `plan.md`
- validate only supported non-double device paths when device-side `double` paths are detected, and record the double-path exclusion in `plan.md`

For `torch_npu` and `pybind` modes, running only a standalone executable is not sufficient evidence of completion.

If validation fails, return to implementation and continue until the result is either validated or clearly blocked.

When validation fails, use a root-cause loop instead of trial-and-error:

1. read the exact error output and identify the failing step
2. reproduce the issue consistently
3. isolate the failing layer, such as API mapping, syntax migration, build setup, runtime environment, or result checking
4. form one concrete hypothesis
5. test that hypothesis with the smallest possible change
6. rerun the relevant verification command

If multiple fix attempts fail, revisit the migration plan, API mapping, and overall migration strategy before changing more code.

### Step 7: Produce the final report

The final outcome must fall into exactly one of these states:

- validated on Ascend 950 PR
- migrated but not validated on Ascend 950 PR
- blocked by environment or unsupported feature
- build or validation failed

Use the final status labels exactly as listed in `reference/validation-checklist.md`. Do not collapse these states into a generic success message.

The final Chinese documents must explicitly record:

- environment readiness
- executed build and validation commands
- validation evidence
- final status label
- unresolved items and next action when the work is incomplete
- root-cause notes for any failed or blocked validation

Before final reporting, perform a consistency check:

- make sure the reported status matches the validation evidence
- make sure the recorded commands match what was actually executed
- make sure the Chinese documents and the final summary do not contradict each other
- make sure no undocumented blocker or unverified claim remains

## Output Contract

The fixed output directory is:

```text
ported-ops/<operator_name>/
```

The minimum deliverables depend on the selected migration mode.

Common minimum deliverables:

- `plan.md` in Chinese
- `README.md` in Chinese
- migrated source files
- validation artifacts when applicable

Additional minimum deliverables by mode:

- `sample`
  - executable project files
  - host code with `main`

- `torch_npu`
  - standalone installable project files
  - host code
  - registration code
  - Python invocation path
  - installation and invocation verification artifacts

- `pybind`
  - standalone installable project files
  - host code
  - binding code
  - Python invocation path
  - installation and invocation verification artifacts

Chinese engineering documents should read like execution records, not marketing summaries.

`plan.md` should focus on:

- what will be migrated
- how it will be migrated
- what is blocked
- how verification will be executed
- which native JIT compilation/loading paths were removed because they are unsupported
- which complex dtype paths were excluded because they are unsupported
- which device-side `double` paths were excluded because they are unsupported in one-to-one migration

`README.md` should focus on:

- what was migrated
- how to build and run it
- what was verified
- what remains limited or unverified
- what the next engineering action should be if the result is blocked or incomplete

## Validation Gate

Read `reference/validation-checklist.md` before any final conclusion.

You may report a validated success only if all required conditions in that checklist are satisfied.

If any required condition is missing, you must explicitly downgrade the status.

Examples of invalid success claims:

- code compiles locally but was never run on Ascend 950 PR
- tests were prepared but never executed on device
- device execution happened but results were not checked
- environment setup failed before runtime validation
- a `torch_npu` task returned only a standalone sample without registration logic
- a `pybind` task returned only a standalone sample without binding code
- only kernel code was migrated while the final user-facing invocation path was not migrated

## Failure Reporting

When blocked or failed, report the reason precisely.

Preferred failure categories:

- `UNSUPPORTED_FEATURE_ERROR`
- `ENVIRONMENT_NOT_READY`
- `API_MAPPING_BLOCKED`
- `BUILD_FAILED`
- `VALIDATION_FAILED`

For each failure, specify:

- affected file or function
- triggering feature or step
- evidence from the environment, build, or test result
- whether a workaround exists
