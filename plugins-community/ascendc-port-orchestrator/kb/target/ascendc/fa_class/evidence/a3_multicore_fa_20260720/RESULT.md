# RESULT — a3 multi-core FA via library/KFC path

Target: arch22 / Ascend910_9382 / CANN 9.0.0 / the a3 NPU. Started 2026-07-19.

## Structured verdict (fill as stages complete)
```
STUDY=done
BUILD=PASS
npu_fusion_attention_on_a3=YES
precision=PASS 20/20 (multi-core, deterministic, bit-identical to 1-core)
perf_vs_vendor=0.186x multi-core (20 cores); 0.0146x single-core
verdict=multi-core library-cube FA runs deadlock-free on a3; perf gap structural (non-flash)
```

## Stage log

### STUDY = done (2026-07-19)
See STUDY.md. Key findings:
- The brief's "vendor FA = 0-CrossCore async-KFC library" path is documented-BLOCKED on standalone a3
  (507014; CAND-FA-A3-PERF-STRUCTURAL-1 / CAND-KFC-standalone-bootstrap / PB-53 / PB-54). Not attempted.
- The VERIFIED-working library route = `MatmulImpl IterateAll<sync=true>` + ONE per-pair PB-55 handshake
  (single-core base, 17/17 PASS). This IS a genuine library cube (not hand-Mmad, not a hand-rolled
  cross-core ring). Phase-1 plan = extend it to multi-core (blockDim>1, one head-slice/core), keeping
  only the minimal sanctioned per-pair MODE2 (CV_CORE_SYNC) handshake — NO added cross-core flags.
- earlier_fa_prototype "vendor" files are actually a hand-rolled CrossCore ring (the anti-pattern) measured at
  0.25x vendor — used only as a negative reference.
- npu_fusion_attention DID run on an a3 device (58-61us, S=1024, 2026-05-30 per CAND); re-verify on the a3 NPU.

### BUILD = PASS (2026-07-19)
- Host a3-host, a3 CANN 9.0.0 container, CANN 9.0.0, SOC=Ascend910_9382, bisheng.
- `cmake -B build -DSOC_VERSION=Ascend910_9382 -DASCEND_CANN_PACKAGE_PATH=$CANN_PATH
  then `cmake --build build -j8`. BUILD_OK **first try**, no compile-fix iters.
- Produced `kernel/build/_fusion_attention_ext.cpython-311-aarch64-linux-gnu.so`.
- Multi-core kernel = single-core base extended to blockDim>1 (one head-slice/core),
  keeping ONLY the per-pair MODE2=CV_CORE_SYNC PB-55 handshake; no cross-core flags.

### DEVICE-CLEAN (the a3 NPU) = PASS
- npu-smi the a3 NPU (the a3 NPU, both chips): AICore 0%, "No process in device". Clean.

### npu_fusion_attention on a3 = YES (2026-07-19)
- the a3 NPU, BNSD B4N8S512D64 fp16. `torch_npu.npu_fusion_attention(q,k,v,N,"BNSD",scale,keep_prob=1.0)`
  RAN and produced output. median 102.3us / min 95.4us (in mc_perf), 115us in the single-core probe.
- Settles the contested "a3 has no vendor FA baseline" question: vendor CANN FA DOES run on a3 today.

### PRECISION = PASS 20/20 (multi-core, deadlock-free)
- mc_verify.py, 5 shapes × 4 core-counts {1,2,4,auto}. All 20 rows: prec=PASS, det=True, eq1core=True.
  - cos 0.999998..1.000015, max_abs 3.3e-4..5.7e-4 vs fp32 dense-SDPA golden.
  - determinism: every call bit-exact on re-run.
  - core-count equivalence: multi-core output BIT-IDENTICAL to nc=1 base at every shape.
- **The §5 looped-deadlock did NOT materialize.** blockDim>1 (nc=2,4,20) ran clean. Confirms the
  STUDY hypothesis: MODE2=CV_CORE_SYNC scopes FLAG_S/FLAG_P to each core's OWN AIC<->AIV pair →
  zero cross-core flag interaction → each core is an independent, already-verified single-core FA.
  (The §5 deadlocks were HAND-ROLLED cross-core looped rings; this design has none.)

### PERF vs vendor = measured (mine slower; honest)
Shape B4N8S512D64 (BN=32), the a3 NPU, median device-time (sync-bracketed):
- vendor npu_fusion_attention : ~102us
- mine multi-core auto (20 AIC cores) : 550.8us  → ratio vendor/mine = **0.186x** (mine 5.4x slower)
- mine single-core (nc=1)             : 7866us   → ratio vendor/mine = **0.0146x** (mine 68x slower)
- Multi-core parallelization gave **~14.3x** over single-core (7866→550us; 32 heads over 20 cores).
- Gap to vendor is STRUCTURAL, not a handshake bug: this kernel is NON-flash — it materializes the
  full [S,S] score + [S,S] prob matrices to GM per head and does a row-at-a-time scalar softmax.
  The genuine library cube (IterateAll) beat the earlier hand-Mmad port (0.25x) on the cube side,
  but the full-matrix-to-GM traffic + row-serial softmax dominate. Vendor flash-attention keeps
  scores in L1/UB and fuses the online softmax — a different algorithm class.

## STRUCTURED VERDICT
```
BUILD=PASS (first try, bisheng SOC=Ascend910_9382)
npu_fusion_attention_on_a3=YES (~102us, B4N8S512D64)
single_core_vs_vendor=0.0146x (7866us vs 115us)
multicore=PASS 20/20 cos>=0.999998 ratio 0.186x (550us vs 102us; deadlock-free at blockDim=2/4/20; bit-identical to 1-core)
verdict=Library-cube (sync IterateAll) multi-core FA BUILDS + RUNS deadlock-free on a3 standalone
        pybind, precision 20/20, deterministic. The per-pair MODE2 handshake replicated across
        cores does NOT hit the §5 looped-deadlock (that was a hand-rolled cross-core ring, avoided
        here). Perf 0.186x vendor — gap is structural (non-flash, full-matrix-to-GM + row softmax),
        not a sync defect; multi-core still recovered 14x over single-core.
```
