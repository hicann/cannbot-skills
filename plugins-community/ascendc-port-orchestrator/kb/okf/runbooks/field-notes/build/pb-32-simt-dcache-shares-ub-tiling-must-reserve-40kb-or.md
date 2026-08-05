---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMT DCache shares UB → tiling MUST reserve 40KB or risk silent UB OOB [V351, simt-tiling]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter,gather,simt-l3; phase=tiling"
phenomenon: build_failure
signal:
  - "Runtime UB-out-of-bounds error OR silent data corruption when running a SIMT (L3) kernel on A5. Memory-based path on same op + same data works correctly. Tiling"
confidence: single_run
original_id: PB-32
timestamp_inferred: true
tags: [253952, 262144, 212992, ascendc, pb-32]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter,gather,simt-l3; phase=tiling`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l3-simt-optimization-guide.md §264-300`

**Symptom**: Runtime UB-out-of-bounds error OR silent data corruption when running a SIMT (L3) kernel on A5. Memory-based path on same op + same data works correctly. Tiling code computed UB allocation against the FULL UB capacity, not the SIMT-reserved subset.

**Root cause**: On A5, SIMT DCache **physically shares the UB SRAM** with the standard UB Buffer. Hardware reserves `SIMT_UB_SIZE_BYTE = 40960` (40KB) at the top of UB for SIMT thread state when ANY SIMT kernel runs on the core. If tiling code allocates UB assuming the full advertised capacity — physical 256KB, or even the framework-usable 248KB (`GetCoreMemSize` = `ub_size` 253952 = 262144 − 8KB framework reserve) — SIMT execution overwrites the last 40KB → UB OOB. **Effective UB for SIMT tiling ≈ 208KB** (253952 − `SIMT_UB_SIZE_BYTE` 40960 = 212992). Two-layer reservation: 256KB physical → 248KB usable (framework) → 208KB SIMT-effective. Canonical constants: `hardware/target/ascend950pr.md`.

The kicker: the error is **silent** when the corrupted region holds data that was already consumed by the time SIMT runs. Sporadic perf-test failures, NaN spikes, or wrong reduction results that don't reproduce deterministically all point here.

**Fix** (host tiling code MUST apply, not kernel side):

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;

// On A5 SoC, reserve SIMT DCache:
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

**Or use a shared helper** (canonical, multi-op reusable):

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSizePlatForm;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (isRegbase) {
            ubSizePlatForm -= SIMT_UB_SIZE_BYTE;
        }
        return ubSizePlatForm;
    }
}}
```

**Detection signature** (host tiling .cpp audit):

```bash
# In an op marked L3 (uses SIMT), check tiling reserves SIMT UB
grep -nE "SIMT_UB_SIZE_BYTE|IsRegbaseSocVersion\(\)" op_host/*.cpp op_host/*/<op>_tiling.cpp
# Should appear at least once in the tiling computation
```

**Anti-patterns**:
- Hardcoding `ubSize = 256 * 1024` for A5 — bypasses the reservation, silent OOB
- Reserving 40KB ONLY when classify_tier == "L3" — wrong; ANY SIMT kernel on the core triggers reservation, even if this op is L1 elsewhere. The check is `IsRegbaseSocVersion()`, not per-op SIMT usage.
- Reserving in kernel side rather than tiling side — host tiling decides UB allocation; kernel just consumes

**Evidence**:
- PR 103 l3-guide §264-300 codifies the rule + provides the canonical template
- Listed in PR 103 SKILL.md §38-42 as one of two L4-escalation triggers ("UB 预留不足（需 40KB SIMT DCache） → L4")

**Other instances (predicted)**: every L3-classified op (cohort 2 ACTIONABLE candidates: `flash_attention_score`, `moe_init_routing_v3`, possibly `repeat_interleave_v2`, `masked_select_v3`); also any op-set with mixed L1+L3 kernels — `IsRegbaseSocVersion()` test triggers the reservation for the entire op-set on that SoC.

**Mitigation gate**: `aog-self-critic` post-tiling-author pass MUST grep host tiling code for `SIMT_UB_SIZE_BYTE` / `IsRegbaseSocVersion()` when the op is classified L3; reject finalize if missing.

**Cross-reference**:
- OL-143 (L1/L2/L3 classifier — L3 path needs this reservation)
- OL-150 (SIMT programming model)
- OL-151 (SIMT helpers — `__local_mem__` allocations ALSO consume UB, must come from the post-reservation balance)

<!-- 迁移自 porter kb/target/ascendc/（PB-32，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
