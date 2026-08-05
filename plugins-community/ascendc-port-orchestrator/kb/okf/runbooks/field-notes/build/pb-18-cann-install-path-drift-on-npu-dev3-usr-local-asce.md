---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CANN install path drift on npu_dev3 — `/usr/local/Ascend/cann-9.0.0` empty, real install at `/data/cann_b103/cann-9.0.0`"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: PB-18
timestamp_inferred: true
tags: [npu_dev3, ascendc, pb-18]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (17_EmbeddingWithInitialLayernormBackward). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22, op#28)
- **Affected**: A5 container `npu_dev3` on host `198.51.100.35` as of 2026-04-22. Likely recurs after any host-side CANN reinstall that moves the working install to a non-standard path.
- **Symptom (CMake)**:
  ```
  CMake Error at CMakeLists.txt:19 (message):
    ascendc_kernel_cmake does not exist, please check whether the cann package is installed.
  ```
  at `cmake -S _autogen_cmake -B build -DSOC_VERSION=... -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0`.
- **Symptom (runtime)**:
  ```
  ImportError: libhccl.so: cannot open shared object file: No such file or directory
  ...
  RuntimeError: Failed to load the backend extension: torch_npu
  ```
  when running `utils/verification_ascendc.py` / `utils/performance.py` without properly sourcing the correct `set_env.sh`.
- **Root cause**: `/usr/local/Ascend/cann-9.0.0/` and `/usr/local/Ascend/cann-9.0.T501/` are empty directories. `/usr/local/Ascend/cann → cann-9.0.T501` broken symlink. `/usr/local/Ascend/ascend-toolkit/latest → /usr/local/Ascend/cann` also dead. The real working install is at `/data/cann_b103/cann-9.0.0/` (contains `x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake`, `compiler/`, `opp/`, `set_env.sh`, etc.). `/root/.bashrc` tries to `source /usr/local/Ascend/ascend-toolkit/set_env.sh` — that path is dead, default shell env is wrong.
- **Additional gotcha — libhccl path**: Even after `source /data/cann_b103/cann-9.0.0/set_env.sh`, `LD_LIBRARY_PATH` only includes `/data/cann_b103/cann-9.0.0/lib64` but `libhccl.so` lives at `/data/cann_b103/cann-9.0.0/x86_64-linux/lib64/libhccl.so`. Must manually `export LD_LIBRARY_PATH=/data/cann_b103/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH`.
- **Workaround**:
  - **Build**: update `workspace/.ascendc_env` → `CANN_PATH=/data/cann_b103/cann-9.0.0`. `deploy_to_a5.sh` (2026-04-22 patched) picks up via `$ASCEND_HOME_PATH` / `$LD_LIBRARY_PATH` exports.
  - **Verification / perf runs**: prefix with `bash -lc 'source /data/cann_b103/cann-9.0.0/set_env.sh >/dev/null; export LD_LIBRARY_PATH=/data/cann_b103/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH; ...'`.
- **Detection**: `find / -type d -name ascendc_kernel_cmake 2>/dev/null` in container — if only `/data/` hits, drift confirmed. Or `ls /usr/local/Ascend/cann-9.0.0/` empty → confirmed.
- **Long-term fix**: `/aog-preflight` should probe `$CANN_PATH/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake` before writing `.ascendc_env`; scan `/data/` + `/usr/local/Ascend/` for a working install when probe fails. Orchestrator's build step could add the same pre-check.
- **Status**: OPEN (environment drift, not a code bug). May auto-resolve on next container rebuild.
- **Evidence**: op#28 MultimodalRopePositionComputationWithGridBasedIndexing Phase C iter 1 (2026-04-22) — CMake fail on orchestrator's default `/usr/local/Ascend/cann-9.0.0` path. Worker traced via `find` to `/data/cann_b103/cann-9.0.0`. Updated `.ascendc_env` → iter 2 configure succeeded. Phase D: `torch_npu` import failed with `libhccl.so: cannot open` until explicit `LD_LIBRARY_PATH` fix applied (orchestrator independent perf re-run).

<!-- 迁移自 porter kb/target/ascendc/（PB-18，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
