---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A build-only A5/V351 host (toolkit-only host CANN) needs a clean own-container device-runtime — ABI-matched torch_npu + a complete CANN, not the host's toolkit CANN"
description: "Toolkit-only A5 hosts compile .so but have no device runtime (chipType=0 / allocator-ABI faults at device-init); run precision/perf in your OWN container with an ABI-matched torch_npu wheel against a complete build+run CANN."
phenomenon: build_failure
signal:
  - "An A5/V351 host compiles .so fine but a linked kernel will not run — chipType=0 or allocator-ABI fault at device-init, or the base image's stock torch_npu free()-crashes"
confidence: single_run
original_id: OL-234
classified_by: llm-assisted
timestamp_inferred: true
tags: [build, a5, torch-npu, abi-match, device-runtime, ol-234]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
`applies_to: soc=Ascend950PR; cann=9.1.0; host_class=build-only-A5 (toolkit-only host CANN, no device runtime)`. `verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060 (host .141)`.

Some A5/V351 hosts ship a TOOLKIT-ONLY CANN — it compiles `.so` but has no device runtime, so a linked kernel will not run (chipType=0 / allocator-ABI faults at device-init). The base image's STOCK torch_npu may additionally `free()`-crash at import/run.

## 根因 / 教训
To run precision/perf on such a host, build your OWN container with a self-consistent device-runtime combo, NOT the host's toolkit CANN. The load-bearing constraint is the **ABI match**: the torch_npu wheel <-> the CANN runtime version. Install the matched daily wheel against a COMPLETE (build+run) CANN.

Concrete recipe — host `.141` (`tbe`, 7x Ascend950PR_9579), validated:
- **image**: `a5_base:1122_autotest` — but DO NOT use its stock torch_npu `.post1` (it `free()`-crashes).
- **torch_npu**: the on-host daily `.post5` wheel `/data/ci_env/daily/torch_npu-2.7.1.post5.dev20260614-cp311*.whl` -> `pip install --no-deps --force-reinstall <wheel>` (a5_base's torch `2.7.1+cpu` matches its Requires-Dist; `.post5` <-> B060).
- **CANN runtime**: mount host `/data/pri/Ascend/9.1.0.B060`; its `cann-9.1.0` is a COMPLETE CANN (build+run unified). `source .../set_env.sh` AFTER `unset ASCEND_OPP_PATH ASCEND_HOME_PATH` (else the image's stale 8.x install hijacks both).
- **container**: `--privileged` + NPU device mounts (`/usr/local/Ascend/driver`, `/usr/local/dcmi`, `/dev/davinci*`, `/usr/local/bin/npu-smi`) + `/data:/data` (1:1) + `ASCEND_RT_VISIBLE_DEVICES=<idle NPU>`.
- **build toolchain** (if compiling): `/home/npu_user/cann-9.1.0`; the orch `build_ascendc` torch-probe must use `importlib.find_spec` (NOT `import torch_npu`, which ABI-trips on `.post5`) — landed as the `src/scripts/patches/build_ascendc.py` DEBT-20 overlay.
- **A5 safety**: build your OWN container (clone the image), pick an idle card, NEVER touch any `peer_CI_*`/`atk_*`/peer container.

The `.post5`<->B060 ABI-match and the complete-vs-toolkit-CANN distinction are the transferable lessons, not the specific paths — swap host/wheel/CANN paths for any build-only / toolkit-only-CANN A5/V351 host.

## 证据
- FA-grad fused backward slice 24/30 bit-equivalent-to-vendor on `.141` (2026-06-20, whitebox-proxy + an independent graybox, both clean re-verified); the orch ran precision+build end-to-end through this runtime. SFA precision lane reused this recipe (2026-06-20).
- Predicted: any build-only / toolkit-only-CANN A5/V351 host (`.171` when device-wedged; future hosts) — same principle.
