# a3 c220 GM-comm-slot LOOPED bidirectional MIX probe — RUN_LOG

DECISIVE QUESTION: does the shipped c220 GM-comm-slot mechanism (single shared broadcast
flag + per-block GM slot + dcci/dsb) make a LOOPED (>=8 rounds) bidirectional multi-core
MIX cube<->vec handshake run DEADLOCK-FREE on a3/220x, where the naive per-sub DISTINCT
flag-id loop DEADLOCKED (mode3, 507014)?

Env: host a3-host, a3 CANN 9.0.0 container, NPU **the a3 NPU ONLY** (ASCEND_RT_VISIBLE_DEVICES=<n>).
the a3 NPU (the a3 NPU, ) verified CLEAN at start: AICore 0%, HBM 3098MB, no proc.

## Prior (mode4_probe/RUN_LOG.md, device-measured)
- mode0 1-shot distinct-id: PASS deadlock-free
- mode1 1-shot shared-id: PASS deadlock-free
- mode3 LOOPED distinct-id (8 rounds): DEADLOCK 507014 (device latched AICore=100%)
- mode2 LOOPED shared-id: NOT RUN (device latched first)

## KB findings this session (cann-9.0.0 a3 headers) — measured
### pto/npu/a2a3/custom/TSyncCVID.hpp — TSYNC_CVID (c2v ONLY)
  CV_COMM_CTRL=12 (13,14,15 reserved). kCvCommSlotBytes=512. kCvMaxCores=25.
  CVSyncMode: C_ALL_CORE_SYNC=0, V_SUBCORES_SYNC=1, CV_CORE_SYNC=2.
  _getFFTSMsg(mode,flag,base=1) = (base&0xf) + ((mode&3)<<4) + ((flag&0xf)<<8).
  cube: write comm_slot -> GM[block_idx*512]; dcci(slot,SINGLE_CACHE_LINE); dsb(DSB_DDR);
        ffts_cross_core_sync(PIPE_MTE2, msg(CV_CORE_SYNC=2, CV_COMM_CTRL=12)).
  vec:  dcci(slot); wait_flag_dev(12); read GM slot.
  => single SHARED broadcast flag id 12, disambiguation by GM SLOT (block_idx), NOT distinct ids.

### pto/npu/a2a3/custom/TSync_Custom.hpp — bidirectional record/wait/allocate/free
  c2v record: cube ffts(PIPE_FIX, msg(CV_CORE_SYNC, flag)); vec wait_flag_dev(flag).
  v2c record: vec ffts(PIPE_MTE3, msg(CV_CORE_SYNC, flag)); cube wait_flag_dev(flag).
  backward allocate/free use flag+1. All SHARED broadcast (CV_CORE_SYNC=2).

### pto/npu/a2a3/TPush.hpp — full shipped FIFO (canonical FA bidirectional)
  "both" dir uses 4 flags: FlagID (c2v fwd), FlagID+1 (c2v free), FlagID+2 (v2c fwd), FlagID+3 (v2c free).
  Each a SINGLE SHARED CV_CORES_SYNC=2 broadcast. SyncPeriod credit-batching (shouldWaitFree/shouldNotifyFree)
  => syncs only every SyncPeriod tiles, NOT every round -> avoids per-round orphan-credit accumulation.
  KEY INSIGHT: shipped recipe = shared broadcast per direction + GM-slot data + PERIODIC (batched) credit sync.

## Design of GM-slot looped mode (this probe)
Reuse mode4_probe/kernel base. Add mode 5 = LOOPED bidirectional, GM-slot recipe:
  per round r in 0..T-1 (T>=8):
    c2v: cube does trivial matmul -> GM data slot; dcci+dsb; ffts(PIPE_MTE2, msg(CV_CORE_SYNC, 12)); vec wait_flag_dev(12).
    vec: trivial vec op; reverse: intra-vec sub barrier, subblock0 ffts(PIPE_MTE3, msg(CV_CORE_SYNC, 13)); cube wait_flag_dev(13).
  Single shared broadcast flags 12 (c2v) + 13 (v2c). Both AIVs consume the c2v broadcast (balanced fanout-2);
  reverse aggregated to 1 via subblock0-only issue (balanced 1:1). GM-slot per block_idx*512.
Keep mode3 (distinct-id looped) as known-DEADLOCK CONTROL.

## Timeline
23:47:58Z RUN_LOG created, headers grepped

$(date -u +%H:%M:%SZ) Kernel design finalized. Building mode 5.
KEY INSIGHT (grounded in headers): the DECISIVE anti-deadlock variable is single SHARED
broadcast flag id per direction with BALANCED consumption:
  - c2v: cube ONE set (CV_CORE_SYNC=2 broadcast) -> BOTH subs wait same id (broadcast delivers
    a credit to each recipient; mode1 1-shot proved 1 set satisfies 2 waits). Zero orphan.
  - v2c: each sub sets same id -> cube waits it N(=2) times. Balanced.
  vs mode3 DISTINCT ids: cube's broadcast of id X physically reaches BOTH subs but only 1 waits X
  -> sibling orphan accumulates each round -> FFTS stall. That is the 507014 root.
GM-slot's role = lets you use a SHARED flag (data disambiguated by block_idx*512 slot, not by
distinct flag id). Faithful mode5 = shared flags CV_COMM_CTRL=12 (c2v) + 13 (v2c) + per-block
GM slot write + DataCacheCleanAndInvalid (AscendC API; raw dcci/dsb are bisheng c220 builtins,
using the portable AscendC cache API instead). mode2 (shared, no gm-slot) = flag-equivalent control.
RUN ORDER (device budget: hang LATCHES): candidate mode5 FIRST on clean device. If PASS ->
decisive positive. Then mode3 control LAST (expected to latch = accepted end state).

23:56:17Z BUILD SUCCESS: mode5 GM-slot kernel compiles clean on bisheng arch22 (Ascend910_9382).
  DataCacheCleanAndInvalid<int32_t,SINGLE_CACHE_LINE,CACHELINE_OUT>, CV_COMM_CTRL=12/13, GM slot all OK.
  .so = _mix2_ext.cpython-311-aarch64-linux-gnu.so (9.2MB). Running mode5 (CANDIDATE) on clean the a3 NPU.

## RE-RUN verdict (gmslot-rerun-kw-2)
Host=a3-host , a3 CANN 9.0.0 container.
Probe files synced at <scratch>/mode4_probe (build_gmslot.sh, test_mix2.py, kernel/ present, Jul 20 07:54).
STEP1 the a3 NPU clean gate: npu-smi the a3 NPU () AICore=0%, HBM=3097/65536MB (baseline),
  "No process in device", no python/aclrt/test_mix procs. => the a3 NPU CLEAN/IDLE, NOT latched. Proceeding.
STEP2 build: BUILD SUCCESS. _mix2_ext.cpython-311-aarch64-linux-gnu.so (9276648 bytes) at Jul 20 08:19.
STEP3 running mode5 CANDIDATE (hard 90s timeout), ASCEND_RT_VISIBLE_DEVICES=<n>...

### FINAL RE-RUN VERDICT
STEP3 mode5 CANDIDATE: **HANG** (EXIT_CODE=124, timeout 90s). stdout captured:
  "torch_npu DEV=npu:0 visible=2 mode=5 (LOOPED GM-comm-slot ...) CANDIDATE"
  "[mode=5] launching blockDim=2 (2 MIX groups)..."
  -> then NO further output; no "RESULT mode=5" line. Host sync blocked >90s = deadlock signature.
  max_rel=n/a, latency=n/a (never returned).
the a3 NPU end-state AFTER python kill: AICore=0%, HBM=3098MB (baseline), "No process in device"
  => device RECOVERED cleanly on process teardown this run (no persistent 507014 latch observed;
     the timeout-kill of python tore down the aclrt context and freed the device). No >120s recovery needed.
STEP4 mode3 control: **SKIPPED** (brief: only run if mode5 PASSED; it hung).

INTERPRETATION: mode5 GM-comm-slot recipe (single SHARED broadcast flag 12/13 per direction +
per-block GM slot + DataCacheCleanAndInvalid, LOOPED >=8 rounds bidirectional, blockDim=2) ALSO
DEADLOCKS on a3/220x (Ascend910_9382). The shared-flag + GM-slot disambiguation does NOT escape the
LOOPED-bidirectional MIX deadlock limit. Consistent with prior device witness: single-shot concurrent
MIX works, LOOPED bidirectional (full-FA class) hangs. => GM-slot recipe is NOT the missing key;
the deadlock is deeper than distinct-vs-shared flag-id orphan-credit. Full-FA multicore on a3 via this
path remains UNACHIEVED. Device did recover cleanly (good for budget), but verdict is negative.

STRUCTURED: MODE5=HANG; max_rel=n/a; latency_ms=n/a; MODE3_CONTROL=SKIPPED; device2_end_state=clean(recovered-post-kill)
