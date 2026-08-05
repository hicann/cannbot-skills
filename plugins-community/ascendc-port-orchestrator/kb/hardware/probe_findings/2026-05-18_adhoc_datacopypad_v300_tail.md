# PROBE REPORT — V351 DataCopyPad UB→GM non-aligned tail

## Verdict
ACCEPT_CORRECT

**One-line summary**: On Ascend950PR (V351, `__CCE_AICORE__ == 300`) with
CANN 9.0.0 + bisheng 15.0.5, `DataCopyPad` UB→GM with a non-32B-aligned
byte-level `blockLen` (tested 31, 33, 47, 63) writes **exactly the
requested number of bytes**, does **not overflow** into adjacent GM, and
does **not raise** any runtime error. EC-23 (V220 crash) does NOT apply
to V351. OL-167 / P-P98 are confirmed valid on V351.

## Environment

```
- date: 2026-05-19T04:48Z
- host: 198.51.100.35 (npu_dev3 container)
- kernel: 6.6.110-42.13.tl4.x86_64
- OS: openEuler 22.03 (LTS-SP4)
- CANN: /data/cann_b103/cann-9.0.0 (sourced via bin/setenv.bash)
- bisheng (ccec): clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)
                  build 2026-04-03T18:03:06+08:00
                  Target: x86_64-unknown-linux-gnu
- npu-smi: 25.7.rc1
- torch: 2.7.1.post4.dev20260312
- torch_npu: shipped with /data/cann_b103/cann-9.0.0
- SOC_VERSION: Ascend950PR_9579 (V351, __CCE_AICORE__ == 300)
- NPU 4 utilization before run: 0%; HBM 4568/114688 MB; no other processes
- NPU 4 utilization after run: 0%; HBM 4780 MB; no orphaned processes
```

## Question

On V351/Ascend950PR with bisheng A5, does `DataCopyPad` UB→GM for a
non-32B-aligned tail (blockLen=47 bytes from a fp16 UB) write cleanly,
or does it crash like V220 (per EC-23 "DataCopyPad UB→GM crashes on
V220 — use plain DataCopy + manual tail pad")?

## Hypothesis vs Observation

- **Hypothesis** (per BRIEF.md): Verdict (a) `ACCEPT_CORRECT` — V351 is
  newer than V220 and `DataCopyPad` is the documented primitive for
  non-aligned writes.
- **Observation**: Confirmed. All four non-aligned byte counts (31, 33,
  47, 63) wrote exactly `blockLen` bytes; the immediately adjacent guard
  bytes preserved the 0xCAFE sentinel intact; no `507035 / EZ9999 /
  EE9999` runtime errors; no kernel launch failure; NPU 4 remained
  idle and healthy after each invocation.

## Evidence

### Probe design

- **GM buffer**: 256 bytes (128 × fp16), host-allocated then pre-filled
  with raw bit pattern `0xCAFE` per fp16 slot (little-endian bytes
  `0xFE 0xCA 0xFE 0xCA ...`) as overflow sentinel.
- **UB scratch**: 64 × fp16 = 128 bytes; first 24 slots set to bit
  pattern `0xBEEF` (little-endian bytes `0xEF 0xBE 0xEF 0xBE ...`); the
  remaining 40 slots set to `0xDEAD` (bytes `0xAD 0xDE`) — these are
  the "would-overflow-if-bug" bytes we'd see leak into GM if the API
  rounded `blockLen` up to a 32B-aligned write.
- **Kernel**: single AIV, `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`,
  one call to `DataCopyPad(gmDst, ubLocal, {1, blockLen, 0, 0, 0})`.
- **Host check** (after `torch.npu.synchronize()`):
  - bytes `[0, blockLen)` must equal the expected payload sliced from
    `(0xBEEF×24 || 0xDEAD×40)` little-endian byte stream
  - bytes `[blockLen, GM_BYTES)` must each remain at their pre-init
    sentinel (`0xFE` even index, `0xCA` odd index)
  - any Python-level exception or non-zero `aclrtlaunch_*` return code
    fails the run

### Build

`build_ascendc.py` configured CMake under
`/root/probe_work/probe_datacopypad_v300_tail/kernel/build/` with
`-DSOC_VERSION=Ascend950PR_9579 -DASCEND_CANN_PACKAGE_PATH=/data/cann_b103/cann-9.0.0`.

- First attempt: compile error — `KERNEL_TASK_TYPE_DEFAULT` was
  mistakenly prefixed with `AscendC::` namespace; macro lives in
  global scope. One-line fix → unprefixed.
- Second attempt: **clean build**. No warnings of substance. Library
  produced at
  `/root/probe_work/probe_datacopypad_v300_tail/kernel/build/_probe_datacopypad_ext.cpython-311-x86_64-linux-gnu.so`.

### Runtime

`run_results.json` (full, captured from container):

```json
[
  {
    "blockLen": 47,
    "exception": null,
    "valid_match": true,
    "overflow_detected": false,
    "first_64_bytes_hex": "efbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefcafecafecafecafecafecafecafecafeca"
  },
  {
    "blockLen": 33,
    "exception": null,
    "valid_match": true,
    "overflow_detected": false,
    "first_64_bytes_hex": "efbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefcafecafecafecafecafecafecafecafecafecafecafecafecafecafecafeca"
  },
  {
    "blockLen": 63,
    "exception": null,
    "valid_match": true,
    "overflow_detected": false,
    "first_64_bytes_hex": "efbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeaddeaddeaddeaddeaddeaddeaddeadca"
  },
  {
    "blockLen": 31,
    "exception": null,
    "valid_match": true,
    "overflow_detected": false,
    "first_64_bytes_hex": "efbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefbeefcafecafecafecafecafecafecafecafecafecafecafecafecafecafecafecafeca"
  }
]
```

### Byte-by-byte trace for blockLen=47 (the headline case)

```
byte index:  0  1  2  3 ... 44 45 46 | 47 48 49 50 51 ...
expected  : EF BE EF BE ... EF BE EF | CA FE CA FE CA ...   <- valid 47B then sentinel
observed  : ef be ef be ... ef be ef | ca fe ca fe ca ...
                                       ^^
                            first byte past the 47-byte tail is the SENTINEL
                            (0xCA = high byte of 0xCAFE, odd index 47).
                            If DataCopyPad had over-written to the next 32B
                            boundary (byte 63 inclusive), we would instead
                            see 'be' (low byte of 0xBEEF[23]) or 'ad/de'
                            (0xDEAD for UB slots >=24) here.
```

For blockLen=63 the pattern transitions from 0xBEEF region into the
0xDEAD region within the valid window (consistent with UB layout —
slot 24 onwards is 0xDEAD), then resumes 0xCAFE sentinel at byte 63:
`...efbe[23rd 0xBEEF] addead addead ad de ad ca` — the final `de ad`
pair is the UB[31] low/high bytes (bytes 62/63 of payload — wait, byte
63 is past the 63-byte tail: index 63 is the 64th byte). Re-decoding
the 63 case:

```
bytes 0..45 : 0xBEEF × 23 elements = 'efbe' × 23   (46 bytes)
byte  46    : low byte of UB[23] (0xBEEF)          = 0xEF
byte  47    : high byte of UB[23] (0xBEEF)         = 0xBE
bytes 48..61: UB[24..30] (0xDEAD × 7) = 'addead'×... actually
              fp16 size 2 → 14 bytes 'ad de ad de ad de ad de ad de ad de ad de'
byte  62    : low byte of UB[31] (0xDEAD)          = 0xAD
                                                     ^
                                                     63rd byte of write
bytes 63..  : 0xCAFE sentinel, first byte at odd index 63 = 0xCA  ✓
```

That precisely matches `...adde adde adde ad ca fe ca fe...` in the
hex dump (low-nibble first per byte): tail of `...addeaddeaddeaddeadca`
where the final `ad` is byte 62 (last byte written) and `ca` is byte 63
(first untouched sentinel byte).

### Raw files

- Kernel: `workspace/probe_datacopypad_v300_tail/kernel/probe_kernel.cpp`
- Pybind: `workspace/probe_datacopypad_v300_tail/kernel/pybind11.cpp`
- Host driver: `workspace/probe_datacopypad_v300_tail/run_probe.py`
- Run log (full stdout/stderr): `workspace/probe_datacopypad_v300_tail/run.log`
- Machine-readable: `workspace/probe_datacopypad_v300_tail/run_results.json`

## Recommendation for orchestrator

- **Amend OL-167 + P-P98** with `verified_on: Ascend950PR (V351, CANN
  9.0.0, bisheng 15.0.5)` plus a pointer to this probe report. The
  byte-exact `blockLen` semantic is confirmed empirically — `DataCopyPad`
  IS the correct primitive for non-aligned UB→GM writes on V351.
- **Confirm EC-23's scope as V220-only** (no widening needed). The
  EC-23 `applies_to: __CCE_AICORE__ == 220` tag is correct; do NOT
  propagate the V220 crash carve-out into V351 kernels.
- **Unblock foreach_neg cold-start (task #22)**: the `aog-kernel-worker`
  that cited EC-23 as a conflict for V351 was over-applying a
  V220-specific defect rule. Direct instruction: "use `DataCopyPad`
  with byte-level `blockLen` per P-P98; no pybind narrow+contiguous
  workaround required on V351".
- **No new PB entry needed.** No new finalize-pipeline gate needed.

## Caveats

- Verified at `__CCE_AICORE__ == 300` only — not at 310/320/220 etc.
- Verified for single-AIV launch, single-row `blockCount=1`,
  `srcStride=dstStride=0`. Pathological cases (`blockCount > 1` with
  small `blockLen`; sub-32B GM offsets via `gm_dst + odd_offset`) were
  NOT exercised by this probe — they remain unverified.
- bisheng build date 2026-04-03; if bisheng <2026-04 is in use the
  result may not transfer (DataCopyPad codegen has changed across
  versions historically — e.g. EC-23 itself is a regression note).
- The GM buffer was a fresh `torch::empty + view-as-int16-fill`
  allocation; pre-existing dirty GM scenarios were not tested.
- No multi-stream / multi-AIV contention scenarios were tested.
