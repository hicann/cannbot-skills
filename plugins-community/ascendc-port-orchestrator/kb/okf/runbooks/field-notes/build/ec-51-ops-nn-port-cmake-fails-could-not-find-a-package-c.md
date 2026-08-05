---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "ops-nn-port cmake fails `Could not find a package configuration file provided by \"ASC\"` — ASCEND_CANN_PACKAGE_PATH not auto-exported"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5"
phenomenon: build_failure
signal:
  - "ops-nn-port build.sh --pkg or --opkernel fails at cmake configure with Could not find a package configuration file provided by \"ASC\". Trace points to cmake/depe"
confidence: single_run
original_id: EC-51
timestamp_inferred: true
tags: [ascendc, ec-51]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: ops-nn-port `build.sh --pkg` or `--opkernel` fails at cmake configure with `Could not find a package configuration file provided by "ASC"`. Trace points to `cmake/dependencies.cmake` line ~75: `find_package(ASC REQUIRED HINTS ${ASCEND_CANN_PACKAGE_PATH}/.../tikcpp/ascendc_kernel_cmake)` resolving to an empty hint path.

**Root cause**: CANN's `set_env.sh` sets `ASCEND_HOME_PATH` (and `ASCEND_TOOLKIT_HOME`, etc.) but does NOT export `ASCEND_CANN_PACKAGE_PATH`. The ops-nn-port build chain reads `ASCEND_CANN_PACKAGE_PATH` directly, with no fallback to `ASCEND_HOME_PATH`. With it unset, the `HINTS` clause becomes a literal `/.../tikcpp/ascendc_kernel_cmake` path and `find_package` cannot locate the actually-shipped `asc-config.cmake`.

**Fix** — explicit pre-build export:
```bash
source /data/cann_b103/cann-9.0.0/set_env.sh    # NOT pipelined — see anti-pattern below
export ASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH
# Then run build.sh
```
The CANN install at `/data/cann_b103/cann-9.0.0` ships `asc-config.cmake` at BOTH:
- `${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc-config.cmake`
- `${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc-config.cmake`

Either path resolves; the `HINTS` lookup walks both.

**Anti-pattern** (separate bug class — observed in same incident): `source /data/cann_b103/cann-9.0.0/set_env.sh | tail -1`. The pipe puts `source` into a subshell — env vars set by `set_env.sh` (including `ASCEND_HOME_PATH`) NEVER reach the parent shell. Use `source ... 2>&1; tail` or `{ source ...; } && echo done` if filtering output is needed. **Symptom is identical** to the missing-export above: `ASCEND_HOME_PATH` is also unset, so `ASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH` exports an empty string.

**Evidence**: fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17, A5 host 198.51.100.35 container npu_dev3, CANN 9.0.0 build b103). Iter-1 cmake failure trace; fix in iter-2 produced `Built target ascendc_impl_gen` and proceeded through `--opkernel` to 3× kernel ELF.
- GDN `chunk_gated_delta_rule` catlass/bisheng build (A5/V351, CANN 9.1.T500, 2026-06-16): an ICE-looking catlass build failure was actually CANN `set_env.sh` never sourced at all (`ASCEND_HOME_PATH` entirely unset, so the toolchain/`ASC` lookup resolved to empty hint paths) — same root class, one-line fix (`source .../set_env.sh` before build). A build error that LOOKS like a compiler ICE should first be triaged as "is the CANN env sourced?" before suspecting bisheng.

**Other instances (predicted)**: any ops-nn-port build on a fresh container/shell session; any orchestrator-spawned build subshell that re-sources `set_env.sh` without explicit `ASCEND_CANN_PACKAGE_PATH` follow-up export. Add to the canonical port_a3 build pre-step list.

**Cross-reference**: OL-158 (Phase C build interpretation — this EC is the prerequisite for getting to a state where `--pkg`/`--opkernel` artifact-set inspection is meaningful).

<!-- 迁移自 porter kb/target/ascendc/（EC-51，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
