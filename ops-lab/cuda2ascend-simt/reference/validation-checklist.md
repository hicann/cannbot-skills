# Validation checklist

Use this checklist before making any final completion claim.

## You may report validated success only if all required conditions are true

- The target run happened on Ascend 950 PR hardware.
- Ascend device nodes were present during validation.
- The Ascend environment was initialized successfully.
- The migrated target built successfully in the target environment.
- The required validation path for the selected migration mode actually ran.
- The output was checked against an expected result, reference implementation, or numerical acceptance rule.
- The validation outcome was recorded in the generated Chinese documents.

Mode-aware validation path requirements:

- For `sample`, the executable, `main`-based validation path, or required native-side test path actually ran.
- For `torch_npu`, the required build, install, and Python invocation path actually ran.
- For `pybind`, the required build, install, import, and Python invocation path actually ran.

## You must not report success if any of the following is true

- The code was only analyzed or planned.
- The code was rewritten but never built.
- The code built, but the required mode-specific validation path never ran.
- The code executed, but result correctness was not checked.
- The environment check failed before validation.
- The target device was not confirmed to be Ascend 950 PR.
- A required native-side, Python-side, import-side, install-side, or package-side validation failed.
- A `torch_npu` task returned only a standalone sample without registration or Python invocation validation.
- A `pybind` task returned only a standalone sample without binding, import, or Python invocation validation.
- Only kernel code was migrated while the required user-facing invocation path was not validated.

## Minimum evidence to record

Record the following in the generated Chinese documents:

- selected migration mode
- source artifact form and final delivered form
- environment readiness summary
- dependency closure or reuse summary when relevant
- explicit `downgrade`, `blocked`, or `excluded` capability items when present
- explicit native JIT compilation/loading removal record when native JIT-related source paths exist
- explicit complex-dtype exclusion record when torch source contains complex dtype paths
- explicit device-side double-path exclusion record when source code contains double dtype dispatch, double kernels/helpers, FP64 intrinsics, or double validation cases
- build status
- executed validation commands
- native/C++ side validation result
- Python-side validation result when required
- install/import/package validation result when required
- final validation conclusion

## Allowed final status labels

Use one of these status families in the final report:

- validated on Ascend 950 PR
- migrated but not validated on Ascend 950 PR
- blocked by environment or unsupported feature
- build or validation failed
