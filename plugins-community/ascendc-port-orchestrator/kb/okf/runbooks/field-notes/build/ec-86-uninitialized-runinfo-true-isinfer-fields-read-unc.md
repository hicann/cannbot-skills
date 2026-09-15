---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "uninitialized `RunInfo<true>` (isInfer) fields read UNCONDITIONALLY in `CalcS1Coord` → stack-garbage query/kv-row shift — LATENT in the shared fa_class template until an inference-mode op instantiates it; fix = struct-default at declaration (one point beats per-consumption-site `if constexpr` guards)"
description: "applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.x; op_class=fa_class (isInfer=true path — MLA / sparse-FA / any inference-mode FA); kernel_type=fa_class template (wholeport)"
phenomenon: build_failure
signal:
  - "an inference-mode FA-class op (a kernel instantiated with isInfer=true — MLA, sparse-FA, decode) produces a +1 query-row (or kv-row) coordinate shift → wrong ou"
confidence: single_run
original_id: EC-86
timestamp_inferred: true
tags: [calcs1coord, ascendc, ec-86]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.x; op_class=fa_class (isInfer=true path — MLA / sparse-FA / any inference-mode FA); kernel_type=fa_class template (wholeport)`
`verified_on: static source-trace vs latest main e3e4b051 (2026-07-24); originally hit at runtime in the SFA forward a3→a5 port (2026-06-22, case-0 +1 query-row shift, branch scan/home commits 89a5241e→ce372ccb)`

- **Symptom**: an **inference-mode** FA-class op (a kernel instantiated with `isInfer=true` — MLA, sparse-FA, decode) produces a **+1 query-row (or kv-row) coordinate shift** → wrong output, and it is **build-fragile**: one lucky build passes, an independent clean rebuild fails (classic uninitialized-read signature). A pure forward/training FA op (`isInfer=false`) does NOT exhibit it.
- **Root cause**: in `fa_class/.../op_kernel/wholeport/wp_util_regbase.h`, `struct RunInfo<true>` (the isInfer specialization) declares four coordinate fields with **NO default initializer**: `preTokensPerBatch` / `nextTokensPerBatch` (via the `COMMON_RUN_INFO` macro, `wp_util_regbase.h:177-178`) and `queryLeftPaddingSize` / `kvLeftPaddingSize` (`wp_util_regbase.h:201-202`). `CalcS1Coord` (`wp_block_cube.h:493/495/506`) then reads all four **unconditionally** to offset `s1Coord`/`s2Coord`. But the ONLY assignment (`wp_kernel_train.h:69-70`) is fenced under `if constexpr (hasAtten) { if ASCEND_IS_AIV { … } }` and covers **only** `preTokens/nextTokens` — `queryLeftPaddingSize`/`kvLeftPaddingSize` are **NEVER assigned anywhere in the template tree**. So on the AIC (or any `hasAtten=false` inference op), those reads are pure stack garbage → garbage coordinate offset.
- **Why LATENT in current main**: as of `e3e4b051` **no op in the repo instantiates `isInfer=true`** — `RunInfo<true>` appears only as its struct definition (`wp_util_regbase.h:196`), so the shared template ships this bug dormant. It bites the FIRST inference-mode FA op built on the template (that is exactly how the SFA a3→a5 port surfaced it).
- **Fix**: give the fields a default **at the struct/macro declaration** — `int64_t queryLeftPaddingSize = 0;` etc. (and `preTokensPerBatch`/`nextTokensPerBatch` a semantically-correct default). One declaration-site default covers **every** consumption site at once and is codegen-neutral. This **supersedes** the tempting per-consumption-site `if constexpr (hasAtten)` guard (which is fragile: it has to be replicated at every read and misses the AIC / hasAtten=false paths — the SFA port's partial guard `89a5241e` was itself superseded by the struct-default `ce372ccb`).
- **Detection**: grep the fa_class wholeport template for `RunInfo<true>` / `COMMON_RUN_INFO` fields that are read in `CalcS1Coord`/attenmask/pse but lack a declaration default; specifically flag any struct field read unconditionally on a template branch that is only assigned under an `if constexpr(...)` guard. **Two INDEPENDENT clean builds expose it where within-build determinism does not** — a single build can leave lucky-zero stack, so require bit-identical output across two fresh builds (a sharper form of the "verify the md5 of the source the build actually compiled" rule, OL — statically-verified-build family).
- **Evidence**: SFA forward a3→a5 port (2026-06-22, Ascend950PR, `scan/home` branch): case-0 query-row shift traced to these uninitialized fields; per-site `if constexpr` guard (`89a5241e`) was partial → struct-default-init of 4 fields (`ce372ccb`) grounded a committed-byte 6/6 floor-PASS. Re-verified structurally against latest main `e3e4b051` (2026-07-24): the four no-default declarations + unconditional reads + never-assigned `query/kvLeftPaddingSize` are all still present in the shipped fa_class template. **The SFA operator itself was never merged (superseded), but this latent template hazard is live in main today.**
- **Cross-ref**: EC-36 (`Cast<T,T>` no-op uninit family), EC-37 (K2 workspace uninit), OL-161 (conditional-write tail uninit) — same uninitialized-read class, but none cover the "struct-declared-no-default field read on a not-yet-instantiated `isInfer=true` template branch"; `patterns/domains/fa_class_template.md` K1/K2 (host-rule↔kernel-instance consistency — this is the RunInfo-field analogue on the coordinate path). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/（EC-86，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
