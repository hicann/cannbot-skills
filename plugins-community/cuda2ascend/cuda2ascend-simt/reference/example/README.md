# Example Library

This directory stores manually prepared CUDA and Ascend C SIMT example pairs for reference.

## Purpose

Use these examples as a pattern library:

- to understand how one CUDA operator was migrated into Ascend C SIMT
- to compare host-side launch flow, build setup, and basic implementation structure
- to borrow layout and organization ideas for new migrations

## Layout

The current example library is organized by delivery shape:

- `cuda-sample/`
  - standalone CUDA samples
- `simt-sample/`
  - standalone Ascend C SIMT samples
- `cuda-torch/`
  - CUDA-side PyTorch custom operator examples
- `simt-torch/`
  - Ascend C SIMT-side PyTorch custom operator examples

Current example entries:

- `cuda-sample/vector-add/`
- `simt-sample/vector-add/`
- `cuda-torch/muladd/`
- `simt-torch/muladd/`

Example notes:

- `simt-torch/muladd/` is the finalized Ascend C SIMT torch extension reference for this operator pair.
- The SIMT side keeps the original `extension_cpp/` project layout as the primary comparison target.
- CUDA-side examples may use different registration styles, but the SIMT-side migration target should follow the SIMT torch registration style used by this example.
- Use this example to study minimal structural changes from the CUDA `extension_cpp/` project into a standalone Ascend C SIMT torch-style project.
- The Python wrapper and fake/meta registration may remain intentionally close to the CUDA-side wrapper when that preserves the user-facing Python API. `torch.library.register_fake`, FakeTensor/meta kernels, and wrapper code kept only for `torch.compile` compatibility are not native JIT compilation or loading paths unless they compile or load native code at runtime.
- Native JIT compilation, runtime compilation, extension JIT loading, and `nvrtc`-backed paths remain unsupported migration targets and should be removed or reported according to `reference/constraint_rules.yaml`.

## Limits

Do not treat these examples as the source of truth for API compatibility.

For actual migration decisions, always prioritize:

1. `reference/api-mapping/*.yaml`
2. `reference/constraints.md`
3. `reference/grammar.md`
4. official local documentation

## Maintenance

When adding new examples:

- keep each operator self-contained
- keep CUDA and SIMT examples easy to compare
- prefer one operator per directory if multiple examples are added later
- document any important simplifications or assumptions near the example
- place the example under the directory that matches its delivery shape
