# Constraints

Use this file to identify CUDA features that are unsupported or only partially supported in the current Ascend C SIMT migration model.

Unsupported items are split by action:

- `stop_and_report`: the selected migration path cannot continue. Report `UNSUPPORTED_FEATURE_ERROR` and stop the migration.
- `remove_and_record`: the feature is an unsupported source subpath. Remove or exclude that subpath from the migrated result, record the exclusion in `plan.md`, and continue only if supported behavior remains.

If an unsupported removable subpath is the only source-visible behavior or the only user-facing invocation path, treat the migration as blocked instead of silently producing a reduced result.

## Blocking unsupported features with no migration path

### OpenGL / GL

Action: `stop_and_report`.

No workaround in the current migration model.

### Cooperative Groups

Action: `stop_and_report`.

No workaround in the current migration model.

### Texture

Action: `stop_and_report`.

No workaround in the current migration model.

### MMAP

Action: `stop_and_report`.

No workaround in the current migration model.

### `nvrtc`

Action: `stop_and_report` when `nvrtc` is required by the selected migration path.

No Ascend C SIMT runtime-compilation replacement is available in the current migration model. Optional `nvrtc`-backed source subpaths should be handled under the JIT-related subpath rule below.

## Unsupported subpaths to remove or exclude

### Native JIT-related compilation or loading paths

Action: `remove_and_record`.

This includes native JIT compilation, runtime compilation, extension JIT loading, or other on-the-fly native-code compilation or loading paths in migration targets. Remove those paths from the migrated result and record the removal in `plan.md`.

Do not classify PyTorch dispatcher metadata, `torch.library.register_fake`, FakeTensor/meta kernels, or Python wrapper code kept only for `torch.compile` compatibility as native JIT compilation or loading paths unless they compile or load native code at runtime.

### Complex dtype in migrated results

Action: `remove_and_record`.

For torch migration work, detect complex dtype branches explicitly, do not migrate complex dtype code paths, and record the exclusion in `plan.md`. If no supported non-complex behavior remains, report the migration as blocked.

### Device-side `double` paths in one-to-one migration

Action: `remove_and_record`.

No float-rewrite workaround in one-to-one migration.

Detect and exclude source paths that require device-side `double`, including double dtype dispatch branches, double tensor checks, double kernel overloads, double device helpers, FP64 intrinsics, and double-specific validation cases. Record the exclusion in `plan.md` and validate only the remaining supported non-double device paths.

Host-side `double` scalar parameters may be preserved only when they are not a device-side double execution path and the device computation remains in a supported dtype.

## Restricted features with workaround

### Occupancy query APIs

Workaround:

- when the source uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, do not hard-code the local CANN installation path; include the occupancy header through the project's normal CANN/SIMT include configuration
- include `<simt/acl/occupancy/acl_occupancy_query.h>` and migrate the occupancy query to `aclOccupancyMaxActiveBlocksPerVectorCoreFromDevice` when the symbol is available
- map CUDA `blockSize` to Ascend `threadsPerBlock` and CUDA dynamic shared-memory bytes to Ascend `dynamicUbufBytesPerBlock`
- record semantic differences explicitly: CUDA reports active blocks per multiprocessor; the Ascend workaround reports active blocks per vector core
- if the header or function is missing from the CANN/SIMT include configuration, record the missing item in `plan.md`, then avoid the occupancy dependency by using a documented fixed launch policy, measured tuning table, or conservative launch bound
- do not report `UNSUPPORTED_FEATURE_ERROR` for `cudaOccupancyMaxActiveBlocksPerMultiprocessor` until the occupancy header lookup and fallback analysis have been recorded

Other CUDA occupancy APIs remain unsupported unless a local CANN SIMT replacement is found and documented.

### Device-side `std::` usage

Workaround:

- when device-side code uses `std::xxx` forms, use CMake package discovery instead of hard-coded CANN include paths:

```cmake
find_package(simt_stl CONFIG REQUIRED)
target_link_libraries(<target> PRIVATE simt::std)
```

- prefer the `simt::std` replacement when the semantic match is clear, and include the matching SIMT header through the imported target include directories
- if no replacement exists, record the missing `std::` symbol and search evidence in `plan.md`, then avoid the dependency with a narrow local helper, direct language construct, or host-side precomputation
- do not silently keep device-side `std::` code that is unsupported by the Ascend SIMT compiler

### Source files with `.cpp` suffix compiled as Ascend code

Workaround:

Set source-file properties in `CMakeLists.txt`, for example:

```cmake
set_source_files_properties(
    main.cpp PROPERTIES LANGUAGE ASC
)
```

### User-defined struct passed by value to `__global__` kernel parameters

Workaround:

- avoid passing multi-field user-defined structs by value to `__global__` kernels
- prefer splitting the struct fields into ordinary scalar kernel parameters
- when a grouped parameter object is still useful for device code organization, pass scalars into the kernel and construct the struct locally inside the kernel body
- if a migration source uses functor-style or parameter-object-style kernel arguments, record the rewrite in `plan.md` instead of preserving the by-value struct kernel parameter form

Observed limitation:

- current `bisheng` SIMT compilation may crash during instruction selection when a `__global__` kernel accepts a multi-field user-defined struct parameter by value
- the same logical fields may still compile successfully when passed as separate scalar parameters
