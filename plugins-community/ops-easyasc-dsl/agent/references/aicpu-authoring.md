# AICPU Authoring Reference

This reference owns Device-side AICPU authoring and integration facts. Read it
only after the Router selects the AICPU workflow.

## Opt-in policy

AICPU is disabled unless the user explicitly requests AICPU, AI CPU, or a
`.aicpu` implementation in the active request. Do not infer authorization from:

- an existing AICPU source or generated artifact;
- earlier conversation turns that are no longer the active request;
- a numerical failure, unsupported AI Core primitive, timeout, or performance
  hypothesis; or
- the availability of an AICPU compiler on the target machine.

Without explicit authorization, use the normal EasyASC DSL route or report the
unresolved boundary. Do not author, compile, package, dispatch, benchmark, or
run an AICPU probe.

## Execution model

AICPU runs on a Device-side Arm processor, not on the Host CPU. Its computation
body resembles C++, and the runtime exposes libc, the C++ standard library, and
STL facilities, but it is not an ordinary Host C++ target:

- compile Device source as `.aicpu` or with `-x aicpu` using the installed
  Bisheng AICPU toolchain;
- declare the kernel `__global__ __aicpu__`;
- return a non-void status type such as `uint32_t`;
- accept exactly one pointer argument carrying a packed argument structure;
- do not define the kernel as a class member or inside an anonymous namespace;
- launch from `.asc` Host code with `<<<blockDim, nullptr, stream>>>`, passing
  both the argument pointer and `sizeof(KernelArgs)`; and
- use `blockDim = 1` unless target-specific evidence establishes a supported
  partitioning model. Multiple blocks do not provide normal AI Core-style work
  division.

Only Device addresses may be dereferenced by the AICPU kernel. Use fixed-width
fields in `KernelArgs`, keep the Host and Device definitions identical, and
audit `sizeof`, alignment, pointer ownership, and lifetime.

Minimal Device source:

```cpp
#include "aicpu_api.h"

struct KernelArgs {
    const float *x;
    float *y;
    int64_t count;
};

__global__ __aicpu__ uint32_t scale_aicpu(void *raw_args)
{
    auto *args = reinterpret_cast<KernelArgs *>(raw_args);
    for (int64_t i = 0; i < args->count; ++i) {
        args->y[i] = 2.0f * args->x[i];
    }
    return 0;
}
```

Minimal launcher declaration and call:

```cpp
extern __global__ __aicpu__ uint32_t scale_aicpu(void *args);

KernelArgs args = {x_device, y_device, count};
scale_aicpu<<<1, nullptr, aicpu_stream>>>(&args, sizeof(args));
```

Follow the official plain `extern` declaration form. Do not add C linkage to the
AICPU declaration unless the installed compiler and matching definition prove
that ABI explicitly.

## Internal FP64

An FP16, BF16, or FP32 public tensor may be converted to `double` internally
without exposing `DT_DOUBLE` in the operator schema:

```cpp
double acc = 0.0;
for (int64_t k = 0; k < width; ++k) {
    acc += static_cast<double>(x[k]) * static_cast<double>(w[k]);
}
out[index] = static_cast<OutputType>(acc);
```

Internal FP64 cannot recover information already lost in the public input. It
also does not automatically match a low-precision golden. Record whether state
and intermediate expressions remain FP64, round to FP32, or round to the public
dtype at each step. For recurrent operators, validate the complete sequence;
single-timestep agreement is insufficient.

Use the target's double implementations of `exp`, `tanh`, division, and fused
operations deliberately. Compiler contraction and fast-math settings can change
rounding, so keep them part of the recorded precision path.

## Build contract

Prefer CANN's CMake language modules:

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
find_package(Threads REQUIRED)

project(aicpu_probe LANGUAGES ASC AICPU CXX)

add_executable(aicpu_probe
    kernel.aicpu
    main.asc
)

set_target_properties(aicpu_probe PROPERTIES LINKER_LANGUAGE ASC)
target_link_libraries(aicpu_probe PRIVATE Threads::Threads)
target_compile_options(aicpu_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=<target-architecture>>
)
```

Add the installed Ascend C CMake module root to `CMAKE_PREFIX_PATH`. Resolve the
architecture from the target installation; do not copy another SoC's
`--npu-arch` value.

Direct Bisheng compilation is possible, but AICPU Device compilation needs the
installed AArch64 sysroot, target C++ headers, `aicpu_api` include directory,
Device-library search path, and `aicpu_api` library. Prefer CMake rather than
copying those version-specific flags. If a direct build reads x86 Host headers
or reports a missing `gnu/stubs-32.h`, the AICPU sysroot is incomplete. Some
CANN installations also require the final ASC link to include pthread support.

Before implementation, verify that the exact CANN installation contains:

- an AICPU compiler language module or equivalent Bisheng support;
- `aicpu_api.h`;
- the AICPU Device libraries and cross-compilation sysroot; and
- an architecture accepted for the target SoC.

A successful compile for another SoC does not establish target support. Prove
load and execution with a trivial GM write on the requested hardware.

## EasyASC integration boundary

Manual AICPU integration does not require extending the EasyASC DSL. Keep:

- EasyASC `@kernel` sources on the public target facade and existing AI
  Core/Vector codegen path; and
- AICPU sources as separate `.aicpu` translation units compiled by CANN's
  AICPU toolchain.

The owning CMake/plugin layer must still compile, link, install, dispatch, and
package the AICPU artifact. A `.aicpu` file cannot be dropped into an EasyASC
`@kernel`, and the Python simulator does not execute it.

Only add first-class AICPU stubs, parser lowering, simulator behavior, or an
`@aicpu_kernel` surface when the user explicitly requests that framework work.
Otherwise keep the change at the smallest integration boundary.

## AI Core and AICPU streams

The official heterogeneous-launch contract requires AI Core and AICPU kernels
to use different streams. Establish every dependency explicitly:

```text
AI Core stream:  producer -> record ready event ........ wait done event
AICPU stream:               wait ready event -> consumer -> record done event
```

Host enqueue order across streams is not a dependency. When an ACLNN/plugin
entry receives a caller stream, bridge final AICPU completion back to that
caller stream before returning. Reuse managed streams and events; creating and
destroying them per invocation can dominate small operators.

A specific compiler/runtime may accept AI Core and AICPU launches on the same
stream. Treat that as an implementation observation, not a supported portable
contract. Same-stream execution is ordered rather than concurrent and does not
remove true producer/consumer dependencies.

For shared control or tiling structures, use the target-supported memory
visibility primitive in addition to execution ordering when required. Do not
replace a cross-stream dependency with a Device busy-wait unless the product
documentation explicitly supports that protocol.

## Performance and timeout boundary

AICPU is appropriate for explicitly requested branch-heavy or irregular work,
not as an automatic precision or unsupported-op escape hatch. Large dense GEMM
and long scalar recurrence can be much slower than AI Core and may hit the
AICPU execution timeout.

For mixed recurrent dataflow, avoid switching engines every timestep when one
larger boundary is possible. A common candidate is:

```text
AI Core: precompute all input projections
cross-stream dependency
AICPU: run the complete recurrent sequence and final conversion
```

This is a candidate, not a default. Compare it with the requested baseline on
real hardware. Simulator elapsed time is not AICPU performance evidence.

## Validation checklist

- explicit AICPU authorization is present in the active request;
- exact SoC and CANN version are recorded;
- `.aicpu` Device source and `.asc` launcher compile with the installed
  toolchain;
- argument layout, Device pointers, shapes, dtypes, outputs, and errors match
  the public contract;
- aligned, tail, optional, maximum-size, NaN/Inf, deterministic-repeat, and
  timeout cases are covered as applicable;
- internal FP64 and every narrowing boundary are compared with the real golden;
- mixed paths validate both engines, cross-stream events, memory visibility,
  repeated invocation, and caller-stream completion;
- profiler evidence identifies the actual AICPU and AI Core kernels;
- warmup, repeat count, min/median statistics, and compile/load exclusion are
  documented; and
- package contents contain only the required AICPU/AI Core artifacts and public
  operator dependencies.

## Primary documentation

- [AI CPU programming model](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00049.html)
- [AI CPU compilation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00050.html)
