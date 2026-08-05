# KB Writing Discipline — class-level, verifiable, customer-impactful

> Derived from EC-62/63 case study (DS 2026-05-25). Applies to ALL KB entries regardless of backend.

## 5-step discipline

### 1. Root cause must be white-box discovered

❌ "Container crashed, infrastructure issue" — not a KB entry
✓ "TBuf Get<T>(n) on unallocated UB → 507035. Fix: pipe_.InitBuffer(name, size)" — white-box diagnosed

**Test**: Can a fresh agent reading this entry identify the EXACT code pattern that causes the problem? If not, you haven't found the root cause.

### 2. Entry must apply to op CLASS, not single op

❌ "1_RotaryMul crashes because rotBuf_ not initialized" — op-specific, useless for next op
✓ "Every TBuf<...> member in AscendC kernel class MUST have pipe_.InitBuffer()" — class-level, applies to ALL AscendC kernels

**Test**: Would a worker generating a DIFFERENT op benefit from this entry? If not, the scope is too narrow.

### 3. Entry must have automated detection

❌ "Check manually before build" — won't be checked
✓ "v220_prebuild_check.py blocks build if TBuf without InitBuffer" — automated enforcement

EC-62: `grep 'TBuf<.*> \w+_;' kernel.h | while read decl; do grep -q "InitBuffer($name," kernel.h || echo MISSING; done`
EC-63: `grep 'std::string' kernel/pybind11.cpp && echo BLOCKING`

**Test**: Can a script catch this BEFORE the kernel crashes on NPU? If detection requires human attention, it's a gap.

### 4. Entry must state measurable customer impact

❌ "Performance improvement opportunity"
✓ "507035 vector core exception on V220 — kernel compiles but never executes. All cases fail."

**Test**: What observable failure does the CUSTOMER see on fresh harness install? State it concretely.

### 5. Entry must cross-reference related entries

EC-62 cross-refs: EC-60 (blockDim=0), EC-61 (scalar-pipe), PB-22 (DataCopy 32B limit)
EC-63 cross-refs: OL-180 (CANN env init)

**Test**: Does a worker following cross-refs understand the FULL error class (not just one symptom)? If cross-refs are missing, the worker sees the symptom but not the pattern.

## Anti-patterns (what NOT to write)

| Anti-pattern | Why wrong | Fix |
|---|---|---|
| "Op X crashes on container" | Op-specific, no root cause | White-box diagnose the code pattern |
| "Consider checking alignment" | Vague, no enforcement | Add automated check script |
| "Perf could be improved" | No measurable customer impact | State exact crash/error/failure |
| "See also: <list of 20 entries>" | Indiscriminate cross-refs | Only cross-ref entries in same error CLASS |
| Single-op evidence without class scope | Next op repeats same mistake | Document the pattern, not the instance |

## Evidence strength classification

| Level | Criteria | Example |
|---|---|---|
| HIGH | Root cause confirmed via white-box kernel code analysis + A3 NPU test + automated detection exists | EC-62: TBuf InitBuffer → 507035, verified on 1_RotaryMul, v220_prebuild_check catches it |
| MEDIUM | Pattern observed across ≥2 ops but root cause not fully isolated OR no automated detection | DataCopy alignment warnings (non-blocking in prebuild check) |
| LOW | Single-op observation, no class-level generalization, no automated detection | "Container crashed, retry helped" |

**Rule**: LOW entries MUST be classified as `OTHER` or `UNVERIFIED` — never promoted to canonical OL/EC/PB. MEDIUM entries can be in candidates with explicit "needs automated detection" tag.

## Case study: EC-62 (TBuf InitBuffer) — 5-step discipline applied

1. **White-box root cause**: 1_RotaryMul kw-1 kernel — 5 TBuf workspace buffers (rotBuf_, tmpBuf_, bf16Buf1/2/3) had no pipe_.InitBuffer(). Using TBuf::Get<T>(n) on unallocated UB → 507035 vector core exception.
2. **Class-level scope**: Applies to ALL AscendC kernels using TBuf for workspace/scratch. NOT specific to rotary embedding or fp16.
3. **Automated detection**: `v220_prebuild_check.py` check_tbuf_initbuffer() — grep for TBuf declarations, verify each has InitBuffer call. BLOCKING on missing.
4. **Customer impact**: "Kernel compiles cleanly (10/10 static check) but crashes 507035 on every NPU launch. All cases fail. Customer sees zero PASS."
5. **Cross-refs**: EC-60 (blockDim=0 → 107000), EC-61 (scalar-pipe → 0.047× perf), PB-22 (DataCopy 32B → 507035). All four are "compiles cleanly, crashes at runtime" V220 error classes.

## Case study: EC-63 (std::string ABI) — 5-step discipline applied

1. **White-box root cause**: 28_Interpolate pybind11.cpp used `const std::string&` parameter → SIGSEGV 139 on V220 ARM64 (bisheng/GCC ABI mismatch).
2. **Class-level scope**: Applies to ALL pybind11 wrappers on V220 ARM64. NOT specific to interpolate or resize ops.
3. **Automated detection**: `v220_prebuild_check.py` check_std_string() — grep for std::string in pybind11.cpp. BLOCKING.
4. **Customer impact**: "pybind11 module loads but any function call crashes SIGSEGV. Kernel never executes. Customer sees exit code 139."
5. **Cross-refs**: OL-180 (CANN env init for .so loading) — both are pybind11-level V220 host issues.

## When to PROMOTE vs HOLD

| State | Criteria |
|---|---|
| **Promote to canonical OL/EC/PB** | All 5 steps satisfied + evidence_strength HIGH + ≥2 independent verifications (different ops/containers/agents) |
| **Hold in candidates** | Steps 1-4 satisfied but no automated detection (Step 3) OR evidence_strength MEDIUM |
| **Archive/deprecate** | Entry superseded by newer finding OR root cause found to be wrong (e.g., PB-28/MIX arch-guard falsified) |
| **Do NOT write** | Step 1 (root cause) not established — "infra issue" or "container problem" without code-level diagnosis |
