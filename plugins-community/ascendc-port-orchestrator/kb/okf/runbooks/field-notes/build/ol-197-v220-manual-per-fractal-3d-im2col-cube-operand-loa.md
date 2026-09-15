---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 manual per-fractal / 3D-im2col cube operand loads are A5-invalid — port_a3→A5 must use a single 2D `LoadData2DParamsV2` (mStep-encoded)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all; mode=port_a3_to_a5; scope=cube-matmul operand load (L1→L0)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all; mode=port_a3_to_a5; scope=cube-matmul operand load (L1→L0)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-197
timestamp_inferred: true
tags: [470000, loaddata2dparamsv2, ascendc, ol-197]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all; mode=port_a3_to_a5; scope=cube-matmul operand load (L1→L0)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: op_class beyond attention-fwd+chunk-recurrent (FA + GDN confirmed; principle is API-surface-level so expected to transfer to other cube classes — confirm on next non-attention port_a3 cube op)`

**Cross-op corroboration (transfer beyond FA)**: GDN `chunk_gated_delta_rule` regbase (A5/V351 arch35, CANN 9.1.T500, 2026-06-16) — the 2D fractal `LoadDataToL0A<MK>` / `LoadDataToL0B<KN>` (`LoadData2DParamsV2`-based) produced correct math at every GDN matmul site (key@key^T, query@key^T, attn@v_beta, and the transpose-A state updates) on A5; 118/122 通过. Confirms OL-197 transfers from attention-fwd to a chunk-recurrent (gated-delta) op — the 2D fractal load is the A5-valid path, no per-fractal/im2col needed. Cross-ref OL-224 (the manual-cube tail-K / transpose-A correctness gotchas surfaced by the same GDN cube).

**Principle**: A V220 cube-matmul operand load (L1 NZ → L0) expressed as EITHER (a) a manual per-M-fractal `LoadData2DParams`(V1) i-loop (`startIndex=i`, `srcStride=mStep`, per-band `dst`), OR (b) a 3D im2col `LoadData3DParamsV2` helper (colC0Stride-form), is **A3/V220-valid but A5/arch35-INVALID**. On the arch35 NZ layout it mis-maps operand fractals → per-C0(16)-band d-fragment drift + out-of-bounds reads. The failure is **value-magnitude-preserved**: it passes all-ones / single-onehot probes (contraction count-correct) but fails dense-signed-random (~470000× relative) — the wrong-fragment pairing only manifests with non-degenerate values. (A5 = `__CCE_AICORE__==310` / `__NPU_ARCH__==5102` → the `LoadData2DParamsV2` codegen path.)

**A5 fix (decision rule — applies to A AND B operands, at EVERY load site incl. tail paths)**: replace with a SINGLE 2D `LoadData2DParamsV2` (mStep-encoded) — no per-fractal i-loop, no 3D im2col:
```cpp
LoadData2DParamsV2 p;
p.mStartPosition = 0; p.kStartPosition = 0;
p.mStep   = ceil(M/16);          // all M-fractals encoded in ONE call
p.kStep   = GetBlockNum<T>(K);   // fp16: ceil(K/16)
p.srcStride = p.mStep;  p.dstStride = p.mStep;
p.ifTranspose = /* A-operand: isLeftTranspose; B-operand: !isRightTranspose */;
LoadData(l0, l1[off], p);        // single call
```
**Load-bearing non-obvious bit — the B-operand NEGATES**: `A-operand ifTranspose = isLeftTranspose` (direct); `B-operand ifTranspose = !isRightTranspose`. A mathematically non-transposed B (`isRightTranspose=false`) STILL needs LoadData `ifTranspose=TRUE` to arrange the Zn L0B fragment. A hand-roll easily misses this — it was specifically the MM2 (PV) fix that took attn FAIL→8/8.

**Coverage caveat (test-shape trap)**: convert EVERY operand-load site, including `kRemain>0` / `D%BASE_K≠0` tail paths. Shapes with `D%BASE_K==0` (e.g. D=128, BASE_K=128) leave the tail branch dormant and MASK the residual bug. Test a `D%BASE_K≠0` shape to exercise tail loads. (Caught by main's #253 merge-audit: main-path 6× converted, tail Q-load still V1 i-loop, dormant at D=128.)

**Evidence**: FA `flash_attention_score` port_a3_to_a5 on A5/Ascend950PR (2026-05-29). MM1 QK^T (Q-operand V1 per-fractal i-loop → 2D-V2): `sm_max`/`sm_sum` 8/8 (pre-fix FAIL). MM2 PV (P+V 3D-im2col → 2D-V2, incl. B `ifTranspose=!isRightTranspose`): `attn` 8/8, max_abs=2.4e-4 (pre-fix 0/8 garbage, ~470000×). DS source-derive (direct arch35 CANN read, port_a3-authorized) × independent prototype A5 dump-harness + pass_a, **3 cycles after 4 refuted hypotheses** (rounding / block-permutation / operand-row-offset / i-loop-alone), zero wrong-fix shipped. Complete canonical base = independent prototype `d81db863` (main+tail, all sites 2D-V2); the tail-path conversion independently re-validated the same fix on the `D%BASE_K≠0` path (generality corroboration).

**Provenance (REFERENCE — folds under the membase-A5-invalid OL, task#5; do not duplicate here)**: cv-agent is an A3/CPU-truth-verified reference, **NOT A5-verified** (F10.A.1: A3 16/16; cv-agent-on-A5 blocked on #53, never ran; membase/V220 tier). Its per-fractal/3D-helper loads are A3-valid/A5-invalid; any cv-agent-derived A5 port inherits this.

**Cross-ref / dedup (Mode-2, main 2026-05-29)**: (1) **Resolves PR #250** (the "A5 MM1 QK^T cube membase-invalid — fractal addressing, fix-semantics pending #250" scaffold) — OL-197 IS those pending fix-semantics; PR #250 closed superseded-by-OL-197 (its scaffold candidate not promoted — OL-197 is the canonical entry). (2) Validated resolution of the **superseded `LoadData2DParams ifTranspose=true` candidate** (`patterns/unverified/candidates.md` ~L1127 — the historical compile-clean-but-wrong signal that masked this bug). (3) **Complementary** to `CAND-V220-V351-FA-DIFF-1` (V220 monolithic-class vs V351 per-engine block-types — orthogonal *structural* axis; cross-link, NOT merged into this load-form OL) and the `qk_load_form` decision (`fa_class/cv_reference_concrete_params.md`). (4) Companion: **PB-43** (the membase-A5-invalid PHENOMENON — the symptom + detection-trap "build-success ≠ A5-validation"; this OL is the concrete load-form FIX that PB-43 points to). Source-derived from arch35 `LoadDataToL0A`/`LoadDataToL0B` (matmul.h L101-152 / L218-269). Origin draft: DS `2565d7f1`.

**Other instances (predicted)**: any port_a3_to_a5 cube-heavy op lifting V220 manual fractal-loads (BMM / grouped-matmul / other attention variants); independent prototype GQA (cv-agent-derived) IF A5-ported (separate localization per FA-A5≠GQA-A910 guardrail).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-197（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
