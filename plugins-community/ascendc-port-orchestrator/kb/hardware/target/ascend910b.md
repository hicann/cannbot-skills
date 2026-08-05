---
platform: Ascend910B
type: target
verified: false
cann_version: 9.0.0_beta.2
soc_version: Ascend910B3
soc_family: ascend910b_list
npu_arch: 2201
arch_codename: DaVinci V220
chip_codename: Hi1980BV100
marketing_name: A2 / Ascend 910B
date_verified: TBD (no A2 hardware available at config time)
---

# Ascend910B (DaVinci V220) — Target Platform

> **NOT YET VERIFIED ON HARDWARE.** This file is a forward-looking spec
> derived from CANN source + public references. When an A2 server becomes
> available, run smoke tests (build + `acl.get_soc_name()` + a SIMD `Add`
> kernel) and update `verified: true` + `date_verified`.
>
> A2 = single die DaVinci V220. CANN treats it as `ascend910b_list` family,
> compiles with `BUILD_MODE=c220`, arch dir `arch22`. Same arch as A3 but
> single-die (so half the AIV count of A3).
>
> **SIMD-only.** Same SIMT-restriction list as A3 — see `ascend910c.md`.

## SOC version reference

| Variant | SOC string | Marketed as |
|---------|-----------|-------------|
| B1 | `Ascend910B1` | 910B1 |
| B2 | `Ascend910B2` / `Ascend910B2C` | 910B2 |
| B3 | `Ascend910B3` | 910B3 |
| B4 | `Ascend910B4` / `Ascend910B4-1` | 910B4 |

`acl.get_soc_name()` returns the active SOC string at runtime — query before
choosing build flags. Most public A2 deployments are B3.

## Compute (per chip)

| Parameter | B2 | B3/B4 | Source |
|-----------|----|----|--------|
| AICore total | 24 | 20 | HAMi guide, public refs |
| AIV per AICore | 2 | 2 | CANN `GetAicAivTaskRation()` |
| **AIV total** | **48** | **40** | derived |
| AIC total (Cube) | 24 | 20 | derived |
| **Programming model** | SIMD only | SIMD only | DavidV100 manual §25.1.1 |
| `threadIdx` / `blockDim` SIMT | NOT available | NOT available | arch22 |
| `Simt::` namespace | NOT available | NOT available | arch22 |

What works / fails on A2 = same as A3 (see `ascend910c.md`).

## Memory (per chip)

| Parameter | B2 | B3/B4 | Source |
|-----------|----|----|--------|
| HBM type | HBM2e | HBM3e (B3/B4) | public refs |
| **HBM capacity** | 64 GB | 32 GB (B4) / 64 GB | HAMi guide |
| **HBM bandwidth** | ~400 GB/s | ~800 GB/s | public refs |
| **UB / AIV** | **192 KB** | **192 KB** | CANN `hardware.h` |
| L1 / AIC | 512 KB | 512 KB | CANN `arch.h` |
| L0A / AIC | 64 KB | 64 KB | CANN `arch.h` |
| L0B / AIC | 64 KB | 64 KB | CANN `arch.h` |
| **L0C / AIC** | **128 KB** | **128 KB** | CANN `arch.h` (`__NPU_ARCH__==3510`→256, else→128 — **was "3101", corrected 2026-06-18 vs arch.h:29**) |
| **BT (bias table) / AIC** | **1 KB (1024)** | **1 KB (1024)** | cannbot npu-arch §2.3 DAV_2201 `bt_size` (A5/950PR=4KB — differs) |
| **Structured sparsity 4:2** | **supported** (`sparsity=1`) | **supported** | cannbot npu-arch §2.3 DAV_2201 (A5/950PR `sparsity=0` — NOT supported; don't assume 4:2 on A5) |
| L2 cache | TBD | TBD | needs probe |

## Atomics

Same TBD list as A3 — see `ascend910c.md` § Atomics. A2 and A3 share the
V220 architecture, so atomic capability should be identical, but verify
on first A2 port.

## Cross-platform porting summary

| What | A5 (950PR) | A3 (910C) | A2 (910B) |
|------|-----------|-----------|-----------|
| Programming model | SIMT + SIMD | SIMD only | SIMD only |
| `__NPU_ARCH__` | 3510 | 2201 | 2201 |
| arch dir | arch35 | arch22 | arch22 |
| AIV count | 56 | 80~96 | 40~48 |
| UB per AIV | 256 KB | 192 KB | 192 KB |
| L0C per AIC | 256 KB | 128 KB | 128 KB |
| HBM bandwidth | 1.6 TB/s | ~1.6 TB/s (dual-die) | 0.4–0.8 TB/s |

## Build flags (CMake)

```cmake
set(SOC_VERSION "Ascend910B3" CACHE STRING "...")  # or B1/B2/B4 to match your chip
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE STRING "...")
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
```

## Open verification tasks (when A2 server becomes available)

1. Confirm SOC string via `acl.get_soc_name()` and update this doc's
   `soc_version` frontmatter
2. Confirm AIV count via `GetCoreNumAiv()` (40, 48, or other)
3. Smoke test: build the SIMD `Add` test kernel with the right SOC and run it
4. Probe: FP16/BF16 atomicAdd support (same probe as A3)
5. Probe: HBM bandwidth measurement (compare against the public 400 GB/s figure)
6. Update `verified: true` and `date_verified` in frontmatter

## References

- `docs/design/PLUGIN_PARADIGM_NOTES.md#ascend-chip-comparison` — A2/A3/A5 comparison
- CANN source (`code_channel_infer.py`, `arch.h`, `hardware.h`)
- HAMi Ascend 910B virtualization guide (public)
- HiAscend AscendC manual 9.0.0 (Chinese version)
- `~/workspace/a3/doc/Hi1980B&CV100 Davinci Cloud芯片 用户指南_01.docx` —
  primary hardware doc covering both Hi1980B (A2) and Hi1980C (A3)
