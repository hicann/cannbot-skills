---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Manual ascend950 kernel registry install — 3-file merge into running CANN runtime"
description: "Source: workspace/top_k_top_p_sample/knowledge_update.md F3 (kw-1, 2026-05-14) Scope: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5; phase=Phase C install Principle: after build.sh --pkg --ops=<"
phenomenon: build_failure
signal:
  - "Source: workspace/top_k_top_p_sample/knowledge_update.md F3 (kw-1, 2026-05-14)"
confidence: inferred
status: stub
original_id: CAND-A3A5-15
timestamp_inferred: true
tags: [candidate, inferred, find, binary_info_config.json, apply_top_k_top_p_with_sorted.json, topktoppsample, cand-a3a5-15]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: workspace/top_k_top_p_sample/knowledge_update.md F3 (kw-1, 2026-05-14)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5; phase=Phase C install`

**Principle**: after `build.sh --pkg --ops=<op> --soc=ascend950` produces `.o`+`.json` artifacts in `build/binary/ascend950/bin/ascend950/<op>/`, registering them into the running CANN runtime requires THREE separate filesystem edits, NOT one. Missing any of the three leaves the kernel "built but not callable" — verify reports EZ1001 even though `find` shows the binary on disk. Many port_a3_to_a5 sessions misdiagnose "Phase C done" because step 1 alone (binary install) was performed without steps 2 and 3 (registry merge + per-op JSON copy).

**3-step install procedure**:

1. **Install per-binary `.o` + `.json` into kernel directory**:
   ```bash
   cp build/binary/ascend950/bin/ascend950/<op>/*.o   $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   cp build/binary/ascend950/bin/ascend950/<op>/*.json $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/<op>/
   ```

2. **Merge op entry into runtime `binary_info_config.json`** (single top-level key per op_type):
   ```python
   import json, shutil
   live = "$CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/binary_info_config.json"
   built = "build/binary/ascend950/bin/config/ascend950/binary_info_config.json"
   shutil.copy(live, f"{live}.bak")
   merged = {**json.load(open(live)), **json.load(open(built))}  # op_type unique → no schema conflict
   json.dump(merged, open(live, "w"), indent=2)
   ```

3. **Copy per-op registration JSON**:
   ```bash
   cp build/binary/ascend950/bin/config/ascend950/<op>.json \
      $CANN/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_nn/<op>.json
   ```
   (Follows the existing pattern for `apply_top_k_top_p_with_sorted.json` and other pre-shipped registry entries.)

**Concrete anchor (top_k_top_p_sample, 2026-05-14)**: Step 2 merged 251 → 252 keys; new key `TopKTopPSample` added cleanly (no schema conflict — op_type is unique). After all three steps the V1 kernel is callable from any aclnn entry that targets `TopKTopPSample`.

**Anti-pattern**: doing step 1 only and assuming the runtime auto-discovers binaries by directory scan. CANN's runtime indexes by `binary_info_config.json` keys — a binary on disk WITHOUT a registry entry is invisible at op-resolve time.

**Promotion gate**: needs ≥ 1 additional op confirmation that the same 3-step procedure works for another op_type without further filesystem edits. If a second op surfaces a 4th required step (e.g., `simplified_key.ini` re-generation, `opp_kernel_list.txt` append), this candidate should be revised before promotion.

**Cross-ref**:
- OL-158 (Phase C activation criteria — when to attempt build+install at all)
- CAND-A3A5-14 (variant-aliasing trap — install can succeed yet verify still FAILS due to routing)
- OL-132 (port strategy `regbaseCfg` vs flat `AddConfig` — determines the `op_def.cpp` shape that produced these artifacts)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-15，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
