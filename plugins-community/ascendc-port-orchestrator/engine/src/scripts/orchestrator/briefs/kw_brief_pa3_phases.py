# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""port_a3 Phase A/B/C body builders + context for the port_from_a3 brief.

Extracted verbatim from kw_brief.py (DEBT-201 god-file decomposition,
2026-07-06). Cohesive cluster: the migration-level reference block, the
`_pa3_context` header, the Phase-A (A.1/A.2/A.3), Phase-B and Phase-C prose
builders, plus the CUBE-class-mix and complete-deliverable helper blocks that
`_pa3_context` composes.

LEAF module (depends only on `_common`, imported inline where used). The
orchestrator `kw_brief_port_a3._port_a3_phase_instructions_block` imports these
one-way (no cycle).

Behavior is BYTE-IDENTICAL to the pre-split functions (prompt-template refactor;
golden-locked by test_kw_brief_port_a3_golden.py + the pa3 phase goldens).
"""
from __future__ import annotations
import logging

from pathlib import Path
from typing import Optional


def _port_a3_cube_class_mix_block(workspace: Optional[Path]) -> str:
    """Layer 2 forcing-function (design `PORT_A3_CUBE_CLASS_MIX_ENFORCEMENT_DESIGN.md`
    §3.2), injected when the port_a3 op carries the `CUBE_MIX` tag in
    `op_classification.json` (seeded by `_cmd_port_a3` Layer 1, orchestrator
    :3963-3978, when `_classify_reference_arch == "cube-required"`).

    Mirrors `_backward_perf_c2_block`: tag-gated, returns "" for non-cube-class
    ops so their briefs stay byte-identical. Drives REAL content — it tells the
    worker up front that a pure-VEC kernel is a HACK (the finalize gate
    `_check_architecture_class` REJECTS it as ARCHITECTURAL_HACK, PR #316) and
    hands the concrete MIX scaffold + the understand→KB→regenerate methodology
    so the worker starts from the MIX template, not from a pure-vec fallback.

    Worked-example knowledge is reached primarily via KB (§3.4
    `cube_vector_fusion.md`). Provenance-tracked prior archives may provide
    advisory context, never migration truth or raw implementation bodies.
    """
    if workspace is None:
        return ""
    cls = workspace / "op_classification.json"
    if not cls.is_file():
        return ""
    try:
        import json as _json
        tags = (_json.loads(cls.read_text()).get("op_class_tags") or [])
    except Exception:
        return ""
    if "CUBE_MIX" not in tags:
        return ""
    return (
        "## CUBE-CLASS MIX (MANDATORY — OL-188, finalize gate REJECTS pure-VEC)\n"
        "\n"
        "This op is **cube-class** (`op_classification.json` carries `CUBE_MIX`, derived "
        "from the CANN reference family/markers at classify-time). A pure-VEC kernel for a "
        "cube-required op is a **HACK** (same anti-cheat tier as CPU fallback, OL-188): the "
        "finalize gate `_check_architecture_class` will return **ARCHITECTURAL_HACK** and "
        "block ship (PR #316). You MUST emit a MIX (cube + vector) kernel.\n"
        "\n"
        "**Concrete MIX scaffold** (the worked-example pattern lives in KB "
        "`kb/target/ascendc/patterns/domains/cube_vector_fusion.md` — "
        "read it; any prior-archive observation must be provenance-logged, advisory only, "
        "and independently reconstructed and reverified):\n"
        "- **File split**: `<op>_cube.h` (cube class) + `<op>_vec.h` (vec class) + "
        "`<op>_kernel.h`/`.cpp` orchestrator; class names contain literal `Cube`/`Vec`.\n"
        "- **Cube primitive**: manual `AscendC::Mmad` (FA-verified; `matmul::Matmul<>` was "
        "numerically wrong ~500× on V220, so the verified A5 path picks `Mmad`).\n"
        "- **Vec epilogue**: the op's vector reduction/activation (FA: `SoftmaxFlashV2`).\n"
        "- **Cross-core sync**: `WorkspaceQueue<T, RING_SLOTS=3>` ring buffer, ONE per "
        "producer↔consumer direction, paired flag IDs, raw `PIPE_FIX`/`PIPE_MTE3`/`PIPE_MTE2` "
        "literals (CANN 9.0.0 forbids templated `pipe_t`). NOT inline `CrossCoreSetFlag` in "
        "the loop (Antipattern A → 507015).\n"
        "- **Task type**: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (V220-native, "
        "PB-28 FALSIFIED 2026-05-25 — do NOT arch-guard it out).\n"
        "\n"
        "**Generation methodology — understand→KB→research→regenerate** (NOT line-port / "
        "NOT pure-vec fallback):\n"
        "1. **Understand**: read the A3/V220 algorithm source (`op_kernel/*.h` + `op_host/*.h`) "
        "for WHAT cube op + epilogue + dataflow. Do NOT line-port.\n"
        "2. **KB**: consult `cube_vector_fusion.md` (§3.4) + L-tier RegBase MicroAPI (OL-143/144) "
        "for the A5 mapping.\n"
        "3. **Research (only if you cannot generate)**: consult trusted public API documentation "
        "and interface-level KB evidence; target/prior artifacts remain advisory only.\n"
        "4. **Generate from understanding** using the scaffold above.\n"
        "5. **Regenerate** on gate rejection / precision-perf gap by looping back to "
        "understand/KB/research — **NEVER** to a pure-VEC fallback (the gate will reject it)."
    )


def _port_a3_complete_deliverable_block() -> str:
    """The port_a3 COMPLETE-ARCHIVABLE-DELIVERABLE contract (fix/port-a3-complete-deliverable,
    2026-06-16). Applies to EVERY port_a3 op (FA or not).

    Background (celu a3_port incident): a non-FA port_a3 op reached precision
    PASS but `finalize` could NOT promote it because it hit 3 port_a3
    COMPLETENESS gates, each rollback exposing the next missing artifact:
      ① op_host_completeness (PB-33): workspace/op_host/ missing the GE op_host
         (<op>_def.cpp + _tiling.cpp + _tiling.h + CMakeLists.txt).
      ② binary_provenance: current workspace build SHA256 lineage missing.
      ③ KB_WRITEUP (Phase E): knowledge_update.md missing `## Findings`.

    Root cause: the FA path mandated all 3 (the FA-scoped `_fa_ge_host_gen_block`),
    but a non-FA port_a3 op was never told to emit them UP-FRONT. These gates
    are NOT FA-specific (① universal across AscendC modes / ② universal for
    port_a3+PASS / ③ universal for all modes+PASS). The customer doesn't care
    about failure reasons — they care whether they get a COMPLETE, customer-
    usable A5 operator. A complete deliverable = a REAL CANN operator (the GE
    op_host is what registers it as an aclnn-callable op); the pybind +
    ACLRT_LAUNCH_KERNEL .so is ONLY our test-harness verify path.

    The gates are unchanged (anti-reward-hacking — they correctly reject an
    incomplete port). This contract is the PRODUCING side: emit the complete
    deliverable up-front so finalize promotes in ONE pass, no gate churn.

    This is the GENERAL contract; the FA-class path adds its specialization
    (the `wp_fa_host_tiling.h` / `wfh::` shared arch35 tiling layer + the
    GE_OPHOST_RAW_CANN_COPY gate) on top via `_fa_ge_host_gen_block`. FA stays
    a specialization of this general recipe.
    """
    return (
        "## COMPLETE ARCHIVABLE DELIVERABLE (port_a3 — REQUIRED, all op classes)\n"
        "\n"
        "A port_a3 op is only DONE when it ships a COMPLETE, customer-usable A5\n"
        "operator — not just a kernel .so that verifies. `finalize` enforces this with\n"
        "3 completeness gates (each rejects an INCOMPLETE port; they are correct —\n"
        "do NOT try to route around them). Emit ALL THREE deliverables UP-FRONT so\n"
        "finalize promotes in ONE pass (the celu 2026-06-16 incident churned through\n"
        "them one gate at a time because they were missing):\n"
        "\n"
        "### ① GE op_host (gate `op_host_completeness` / PB-33) — REQUIRED, GENERATED from A3 source\n"
        "Ship the COMPLETE GE op_host in `workspace/{op}/op_host/` (≥3 non-config,\n"
        "non-patch files): `<op>_def.cpp` + `<op>_tiling.cpp` + `<op>_tiling.h` +\n"
        "`CMakeLists.txt` (plus `<op>_infershape.cpp` / `op_api/` if upstream has them).\n"
        "The GE op_host is what registers the op as a REAL aclnn-callable CANN operator;\n"
        "the pybind + ACLRT_LAUNCH_KERNEL .so (Phase D) is ONLY our test-harness verify\n"
        "path, NOT the customer deliverable. **GENERATE** the GE op_host by following the\n"
        "per-file recipe — **do NOT byte-copy CANN arch35 source** (a byte-copy = a\n"
        "customer with no CANN source can't reproduce it; that is the anti-copy red line).\n"
        "Recipe (op-CLASS-general — the CARRY / CARRY+PATCH / REPLACE-HOOK rule applies\n"
        "to every port_a3 op; only the op-specific *specifics* differ):\n"
        "`kb/target/ascendc/patterns/domains/fa_class/templates/op_host/`\n"
        "→ `GE_HOST_TRANSFORM_RECIPE.md`. The three transform classes (derive from YOUR\n"
        "op's A3 (arch22) op_host source at `<port_source>/op_host/`, which IS available):\n"
        "1. **`<op>_infershape.cpp` = CARRY** — if the A3 infershape has 0 arch refs\n"
        "   (grep `arch22|arch35|regbase|dav`), carry it verbatim (shape/dtype inference\n"
        "   is arch-invariant). Most simple ops carry cleanly.\n"
        "2. **`<op>_def.cpp` = CARRY + PATCH** — carry the A3 op IR (input/output/attr\n"
        "   names + order + dtype rows already present); PATCH the SOC config string to\n"
        "   add the A5 `AddConfig(\"ascend910_95\", ...)` block. Add dtype rows ONLY if the\n"
        "   A3 def lacks them AND the target space declares them.\n"
        "3. **`<op>_tiling.cpp` + `_tiling.h` = REPLACE-HOOK (or CARRY for arch-agnostic\n"
        "   tiling)** — if the A3 tiling is arch-agnostic (most simple elementwise / norm\n"
        "   ops: split-by-element, no regbase core-split algorithm), CARRY + light PATCH\n"
        "   (`IsRegbaseSocVersion()` branch per OL-143 if needed). If the op has a\n"
        "   fundamentally different A5 regbase tiling architecture (FA-class), REPLACE-HOOK\n"
        "   onto the KB shared layer — that is the FA SPECIALIZATION below.\n"
        "RED LINE: host C++ only — NO `#include \"arch35/\"` device headers in op_host/, NO\n"
        "aclnn/aclop calls. Derive from the A3 op_host input + this recipe, NEVER from\n"
        "arch35 (the ARCH35_WRAP_CHEAT gate scans op_host/*.cpp too).\n"
        "\n"
        "### ② binary provenance (gate `binary_provenance`) — REQUIRED on PASS\n"
        "Record only artifacts produced by THIS workspace build; never inspect, search, or\n"
        "hash an installed CANN target tree. Emit `build_evidence.compiled_provenance` with:\n"
        "- workspace-relative `source`, copied-back `deployed_source`, `object`, and\n"
        "  dispatched `shared_lib` paths (all four files must exist under the workspace);\n"
        "- `workspace_source_sha256`, `deploy_source_sha256`,\n"
        "  `built_from_source_sha256`, `object_sha256`, and `shared_lib_sha256`.\n"
        "The first three source digests MUST be identical 64-hex SHA256 values; each file\n"
        "digest must match its current bytes. The object basename must be `<source>.o`.\n"
        "Installed CANN binaries, hashes, and path inventories are forbidden evidence.\n"
        "\n"
        "### ③ knowledge_update.md `## Findings` (gate `KB_WRITEUP` / Phase E) — REQUIRED on PASS\n"
        "`workspace/{op}/knowledge_update.md` MUST be ≥100 bytes AND carry the canonical\n"
        "Phase E 5-section structure with literal headers: `## Context`, `## Findings`,\n"
        "`## KB-promotable patterns`, `## Cited KB items`, `## Anti-patterns avoided`.\n"
        "The finalize gate structurally checks for the `## Findings` header — a writeup\n"
        "without it is rejected even if non-trivial. See `GATE_CONTRACT.md` §Phase E.\n"
        "\n"
    )


def _migration_level_block(op: str, workspace: Path) -> str:
    """P114: invoke migration_level plugin → emit per-level KB references.

    Per Zheng 2026-05-16: plugin form, no if-else. Heuristic dispatch lives
    in migration_level.py; this block just consumes the LevelDecision and
    formats it as worker-facing KB ref list. Migration KB content lives in
    kb/target/ascendc/migration/ (imported from PR #103
    `ascendc-operator-A5-migration` skill — P113).
    """
    import json as _json
    import sys as _sys
    plugin_dir = Path(__file__).resolve().parent.parent / "plugins" / "port_a3"
    _sys.path.insert(0, str(plugin_dir))
    try:
        from migration_level import decide_migration_level
    except ImportError:
        return ""  # plugin missing → no injection (caller path still works)

    # Build minimal op_meta from workspace files (best-effort; heuristics
    # default to L1 on missing signals — safe degradation).
    op_meta = {"op_name": op, "op_class": "", "dtypes": []}
    op_cls_path = workspace / "op_classification.json"
    if op_cls_path.is_file():
        try:
            cls = _json.loads(op_cls_path.read_text())
            tags = cls.get("op_class_tags") or []
            op_meta["op_class"] = " ".join(tags) if tags else op
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    d = decide_migration_level(op_meta)

    guides_md = "\n".join(f"  - `kb/target/ascendc/migration/{g}`"
                         for g in d.guides)
    subdirs_md = "\n".join(f"  - `kb/target/ascendc/migration/{s}` (whole subdir)"
                          for s in d.extra_subdirs) if d.extra_subdirs else "  (none)"
    escalation = (
        "\n\n**⚠ ESCALATION SIGNAL**: this op matched an L4 heuristic. "
        "L4 is OUT OF SCOPE for the standard port_a3 worker path — escalate "
        "to architectural-tiling researcher path (TBD)."
        if d.needs_escalation else ""
    )
    return f"""## MIGRATION LEVEL (per P114 migration_level plugin)

Per PR #103 `ascendc-operator-A5-migration` decision tree:
- **Resolved level**: **{d.level.value}**
- **Rationale**: {d.rationale}

**MUST-READ guides for this level**:
{guides_md}

**MUST-READ KB subdirs**:
{subdirs_md}

L1 = mechanical port (all ops). L2 = RegBase MicroAPI rewrite (perf-critical /
quant Cast / overflow / FP8). L3 = SIMT (Scatter/Gather + simple-index +
high-parallel). L4 = escalate (tiling needs IsRegbaseSocVersion / UB shortage).

**If your op's actual characteristics differ from the resolved level** (e.g.
the heuristic missed a signal), document the mismatch in `analysis.md
§Migration level audit` and the orchestrator will retry with corrected
op_meta on next iteration.{escalation}
"""


def _pa3_context(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""# PHASES (port_from_a3_ascendc — arch22→arch35 port mode, ROADMAP §1.5)

This is an arch22→arch35 PORT, not a from-scratch kernel write. The algorithm
already exists in `op_kernel/<op>.h` (A3/V220 variant) and the migration
to A5 (arch35/V351) involves:

1. A small number of surgical changes to produce `op_kernel/arch35/<op>.h`
   (typically: strip V220-only includes, remove `__CCE_AICORE__ == 220`
   conditionals, audit `ToFloat<>` usage for A5 BF16/FP8 restriction).
2. New host-side artifacts: `op_kernel/<op>_apt.cpp` (A5 entry point),
   `op_host/config/ascend950/{{<op>_binary.json, _simplified_key.ini}}`,
   modified `op_host/<op>_def.cpp` to add `ASCEND950` config block.
3. **Possibly cross-op router patch** (see Phase A.3 below).

## CONTEXT (read these first)

- **OPS-NN SOURCE DIR**: `{port_source}`
- **ACLNN ENTRY**: `{aclnn_entry}` (from Phase O2.5 a3-ref provider)
- **CANN UT GEN_DATA**: `{gen_data_source}`
- **PEER-OP DEPENDENCIES**: `{peer_deps_line}`
- **A3 REFERENCE DATASET**: `workspace/edge_dataset.pt["a3_outputs"]`
  (captured by phase_o25_a3_ref; ground truth for precision verify)
- **A3 BASELINE PERF**: `workspace/a3_baseline_perf.json` (target for perf
  ratio — aim ≥ 1.0×, hard floor 0.6×)

{_migration_level_block(op, workspace)}

{_port_a3_cube_class_mix_block(workspace)}

{_port_a3_complete_deliverable_block()}

"""


def _pa3_phase_a_1(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Phase A — Source analysis

  **⚠ Step A.1.−1 — GENERATE FROM V220 SOURCE (single uniform mode)**
   (mandatory independent-generation contract):

  Your job is to **GENERATE** the arch35 kernel from the arch22 algorithm source +
  the codified KB templates. You author every translation unit.

  Optional advisory context may be present in `.prior_art_scan.json`, a
  provenance-tracked DEBT-203 branch base, or files covered by a verified
  `.upstream_prestaged.json` manifest when `OPGEN_PRESTAGE_ARCH35=1`. You may
  inspect these inputs to recover migration knowledge, but MUST log every read
  in `reference_manifest.jsonl`. They are never truth: the current arch22 source
  and fresh source-NPU capture remain authoritative, and target-NPU verification
  is still mandatory. Do not raw-copy an untracked target implementation.

  Uniform authoring contract (all port_a3 ops):
  - **Source of truth = V220 algorithm**: read upstream `<port_source>/op_kernel/*.h`
    (top-level `.h` files) + `<port_source>/op_host/*.h` for the algorithm,
    dataflow, and TilingData struct layout. Do NOT line-port; understand → generate.
  - **Author `kernel/<op>_kernels.cpp`** (build TU): define the tiling-data struct
    matching upstream `op_host/<op>_tiling.h` BEFORE any algorithm `#include`
    (the standalone verifier doesn't process op_host/ tiling registration — caught swi_glu
    kw-1 2026-05-21: 15 compile errors on missing `SwiGluTilingData` def).
  - **Author `kernel/pybind11.cpp`** ACLRT_LAUNCH_KERNEL shim.
  - **NEVER submit a raw, untracked copy of a prior archive or upstream entry-point**, and NEVER
    `#include "arch35/..."` — that is wrapping upstream V351, not a V220→V351 port
    (ARCH35_WRAP_CHEAT gate rejects `#include "arch35/..."` in kernels.cpp /
    pybind11.cpp at finalize). Apply A5 knowledge differentially (RegBase MicroAPI,
    CAST_RINT, VECCALC TBuf, etc.) — a pure V220 line-port that only strips the
    `__CCE_AICORE__ == 220` gate misses the A5 opportunity.
  - Build + verify precision vs `edge_dataset.pt["a3_outputs"]` + measure perf.

  Proceed to Step A.1.0.

  **Step A.1.0 — MANDATORY FIRST: verify the detected arch22 input.** Read
  `.opgen_state.json.source_arch_detection`, then enumerate only its arch22
  evidence and arch22-reachable top-level/common dependencies. Record those
  paths in `analysis.md §arch22 source inventory`. If evidence is absent,
  inconsistent, empty, or points only at target code, stop; do not guess.

  **Step A.1.1 — read the rest**:
  - `<port_source>/op_host/<op>_def.cpp` — understand existing
    `AddConfig("ascend910b")` + `AddConfig("ascend910_93")` shape. Your
    job is to add a `regbaseCfg` block + `AddConfig("ascend950", regbaseCfg)`
    with `ExtendCfgInfo("opFile.value", "<op>_apt")`.
  - **`<port_source>/op_kernel/<op>.h`** — read only when it is part of the
    detector's arch22 evidence or an arch22-reachable top-level dependency.
    This is the arch22 algorithm and the semantic source of truth. Advisory
    target context may inform the migration, but cannot replace this input.
    Identify: V220-specific includes (`impl/dav_c220/*`), `__CCE_AICORE__ == 220`
    blocks, `ToFloat<>` calls on FP16 (need ReinterpretCast<bfloat16_t> on A5).
  - `<port_source>/op_host/<op>_tiling.cpp` + `_tiling.h` — verify the tiling
    is target-agnostic; usually no change needed for arch35.
  - `<port_source>/examples/test_aclnn_<op>.cpp` — understand the aclnn API
    surface this op exposes.

A.1.4. **MANDATORY architecture-class classification** (NEW 2026-05-25, OL-188, owner directive 02:16Z).
  Before writing kernel: classify the architectural REQUIREMENT for this op:
  cube-required (uses matmul/conv/RNN/attention) vs vec-only (pure
  element-wise / norm / reduction without GEMM).

  **Decision** — two-signal combine:

  1. **Op-family signal**: check if `<port_source>` path matches any
     cube-required family in `kb/target/ascendc/cann_classification/cube_required_ops.txt`
     (109 ops batch-learned from CANN 2026-05-25). Families:
     - `ops-transformer/attention` (30 ops) — FA / MLA / sparse / quant
     - `ops-nn/matmul` (14) — batch_mat_mul_v3 / quant_batch_matmul_v3 / etc.
     - `ops-transformer/experimental` (13), `ops-nn/conv` (8),
       `ops-transformer/mhc` (5), `ops-transformer/gmm` (5),
       `ops-transformer/ffn` (4), `ops-nn/rnn` (3), `ops-nn/experimental` (2),
       `ops-transformer/posembedding` (1), `ops-nn/quant` (1 — flat_quant boundary)

  2. **Content signal**: inspect only the detector's recorded arch22 evidence
     and its explicit arch22-reachable dependencies for cube primitives:
     `matmul::Matmul<` / `MatmulImpl<` / `REGIST_MATMUL_OBJ` / `KERNEL_TYPE_MIX_AIC_[12]_[12]` / `cube_block`. Any match → cube-required.

  **Verdict** (combine):
  - Both signals say cube → **cube-required**: kernel MUST use cube primitives (matmul library OR MatmulImpl with REGIST_MATMUL_OBJ + KFC-internal sync). Pure-VEC kernel for cube-required op = **HACK class** (same anti-cheat tier as CPU fallback per owner 02:16Z). Finalize gate `_check_architecture_class` will REJECT.
  - Op-family in BOUNDARY (e.g. `ops-nn/quant/flat_quant`) → multi-variant emission per OL-187 (cube path + vec path, dispatched per upstream tiling rule). See OL-185 calibration anchor.
  - Both signals say vec → **vec-only**: pure-VEC AscendC primitives correct.

  **Cube-required correct paradigm** (CANN canon):
  - Declare `matmul::Matmul<>` or `MatmulImpl<>` as **class member** (long-lived, NOT function-local)
  - Wire `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, ...)` at kernel entry
  - Use `bmm1.IterateBatch(...)` + `bmm1.WaitBmm1Result()` (KFC-internal sync, NOT manual `CrossCoreSetFlag<0x2, ...>`)
  - PB-28 is FALSIFIED 2026-05-25 — `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` IS V220-native; do NOT arch-guard it out

  **Anti-pattern HACK signatures** (REJECTed by finalize gate):
  - Pure-VEC kernel for cube-required op
  - Arch-guarded cube path (`#if __NPU_ARCH__ >= 3510` wrap → V220 falls through to vec-only at runtime)
  - `MatmulImpl<>` + manual `CrossCoreSetFlag<0x2, PIPE_*>` (Path X — works for some shapes but diverges from CANN canon, hits cube limits at scale, customer-non-portable)

  There is no architecture-class waiver; a cube-required source must produce a cube-capable implementation.

  **Standalone check** (run before declaring done):
  ```bash
  python3 src/scripts/orchestrator/checks/architecture_class_check.py \
      --workspace workspace/{op} --op-name {op} \
      --port-source <port_source>
  ```
  Exit 0 = OK; exit 1 = ARCHITECTURAL_HACK detected.

A.1.5. **MANDATORY L-tier classification** (NEW 2026-05-13, OL-143).
  Before writing analysis.md, classify the port effort into one of
  L1 / L2 / L3 / L4. The tier decides which KB references to load and
  determines the amount of independently-authored target redesign required;
  no tier permits reading or copying a CANN arch35 implementation.

  **Decision tree** (evaluated TOP-DOWN, mutually exclusive; first match wins):

  ```
  L2 if ANY of these is true:
    - op_class is perf-critical hotpath (rmsnorm / rope / softmax / attention)
    - kernel uses Cast<fp8_e4m3fn_t,...>, Cast<fp8_e5m2_t,...>, Cast<hifloat8_t,...>
      (A5 new narrow-float types per OL-144)
    - kernel uses ReduceSumCustom or DataCopyPad (A3 idioms; A5 substitutes
      via OL-152 mapping)
    - kernel needs overflow-mode control (per OL-148 — RMSNorm/Softmax SPR<60> toggle)
    - op_def.cpp declares DT_FLOAT8_E*, DT_HIFLOAT8, DT_FLOAT4_E* (forces MicroAPI)

  L3 if NOT-L2 and ALL of these are true:
    - kernel has Gather / Scatter / IndexCopy / IndexPut / index-arith
      (index-based GM r/w)
    - index logic is simple (no nested per-element computation)
    - data volume is large (> a few 2048-element batches)

"""


def _pa3_phase_a_2(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""  L4 (escalate to researcher, OUT of kw scope) if ANY:
    - tiling needs IsRegbaseSocVersion() decision in host code
    - kernel needs > 40KB UB beyond SIMT DCache reservation (PB-32)

  L1 if none of the above — lower-complexity independent port (apply the
  interface/API changes in EC-49 and PB-29/30/31, then create target config).
  ```

  **Record the classification** in `workspace/{op}/analysis.md` §"L-tier
  judgment" with explicit citation of which trigger fired. Example:
    "L2 — quant Cast chain detected: `Cast<fp8_e5m2_t, float>` at line 142
     of op_kernel/<op>.h. Per OL-144 + OL-146, apply L2 MicroAPI rewrite
     of the cast/store pipeline."

  **Load corresponding KB references** (cite in analysis.md §"KB Manifest LOADED"):
    - L1: OL-142, EC-47, EC-48, EC-49, PB-29, PB-30, PB-31
    - L2: above + OL-143, OL-144, OL-145 (if MX-format), OL-146, OL-147
          (if math-fn), OL-148 (if bounded-output), OL-152, P-P95, P-P96
    - L3: L1 set + OL-150, OL-151, OL-152, PB-32
    - L4: STOP and escalate; analysis.md should explain WHY this is L4

  **Anti-pattern**: treating every port_a3_to_a5 as low-complexity L1 wastes
  iterations when an op is actually L2/L3; classify from arch22 evidence first.

A.2. Write `workspace/{op}/analysis.md` covering:
  - Op signature (inputs/outputs, dtypes, shape constraints) — read from
    `op_host/<op>_def.cpp` and `op_host/config/ascend910b/<op>_binary.json`
  - A3 algorithm summary (1-paragraph)
  - **L-tier judgment** (from A.1.5 above) with the trigger citation
  - **KB Manifest LOADED** (the OL/EC/PB/P-P IDs from A.1.5 reference list)
  - Per-file arch22→arch35 diff plan (kernel changes + host changes + binary.json)
  - Cross-op dep analysis — if peer_op_dependencies non-empty above,
    list each peer + which router function(s) need patching + which
    files to edit (e.g. `loss/ctc_loss_v2/op_api/ctc_loss_v2.cpp` lines
    around `IsregBaseAiCoreSupport`)

  ⚠ **Stop-gate contract (check_worker.sh, rc=2 on violation)**: the three
  fields `algorithm_family`, `choice`, `dtypes` MUST appear as BULLET LINES
  with the exact prefixes `- algorithm_family: ...`, `- choice: ...`,
  `- dtypes: ...` (hook greps `^- *<field>`). A `## algorithm_family` heading
  or a `- **algorithm_family**:` bold line does NOT count and rejects the
  worker. Example: `- algorithm_family: fused-attention (Softmax(QK^T)V)`.

A.2.5. **Source-analysis-summary** (NEW 2026-05-13, closes Task #39).
  Write `workspace/{op}/<op>_source_summary.md` per the 8-section template
  below. This serves as authoritative input for Phase D precision test-case
  generation. Test cases derive from THIS document, not from any external
  doc (per the source-as-truth rule).

  ```markdown
  # Source-analysis summary: <op_name>

  ## 1. Function description
  - Math formula / compute logic (reverse-engineered from kernel code if no doc)
  - Op class: elementwise / reduction / attention / matmul / scatter-gather / ...

  ## 2. Interface signature
  - Inputs (name, type, shape constraints, dtype)
  - Outputs (name, type, shape compute rule)
  - Optional / attribute parameters

  ## 3. Supported dtypes
  - From _def.cpp DataType config
  - From kernel template specialization / conditional-compile paths
  - List explicitly: fp16 / bf16 / fp32 / int8 / fp8 / hifloat8 / etc.

  ## 4. Compute logic pseudocode
  - From kernel .cpp/.h Compute / Process function
  - Annotate up-precision paths (fp16 → fp32 → fp16 etc.)
  - Annotate conditional-compile branches (BF16 path / FP32 path)

  ## 5. Tiling strategy
  - Block-level split (formerNum / formerLength / tailNum / tailLength)
  - UB-level split (tileLength compute, bufferCoefficient)
  - Special constraints (alignment, tail-tile handling)

  ## 6. Input domain constraints
  - Mathematical domain (e.g. acosh requires x≥1, log requires x>0)
  - Data range constraints (e.g. fp16 max 65504)
  - Shape constraints (dim limits, alignment requirements)

  ## 7. Boundary conditions & special values
  - Zero / min / max value behavior
  - NaN / Inf propagation
  - Subnormal handling (FTZ vs preserve — see OL-147)

  ## 8. Doc-vs-source diff (when external doc exists)
  - Doc descriptions inconsistent with source
  - Doc-missing but source-implemented features
  - Doc-described but source-not-implemented features
  - **NOTE**: test case generation uses SOURCE as truth, doc as reference
  ```

  When generating precision test cases (Phase D), bind each case to a
  specific section (e.g. boundary value tests bind to §7; dtype matrix
  tests bind to §3). This document IS the spec for ≥30-case coverage.

A.2.6. **DUAL-INPUT MODEL.PY CONSTRUCTION RULE** (mandatory, port_a3 only —
  2026-05-21, post fused_quant_mat_mul case 6 reward-hacking incident).

  When you write `workspace/{op}/model.py` (the CPU-truth reference that
  drives Phase D precision comparison), you have TWO inputs and MUST USE
  BOTH:

  **Input 1 — arch22 source op_kernel/** (`<port_source>/op_kernel/arch22/`
  plus top-level/common files reached by that source; exclude `arch35/`). This tells you
  the **specific algorithm sequence** the reference implementation
  actually executes:
    - Dequant order: is `x2_scale` applied first then `x1_scale`, or
      reverse? `AscendDequant(fp32_scale, ...)` vs raw multiply?
    - Cast modes: `RoundMode::CAST_RINT` (round-half-to-even on a
      different LSB than PyTorch default), `CAST_ROUND`, or `CAST_NONE`?
    - Polynomial coefficients: GELU-erf typically has 7+ Horner
      coefficients; preserve the arch22 numeric values exactly, do not substitute PyTorch's
      `torch.nn.functional.gelu(approximate='none')` which uses
      `0.5*x*(1+erf(x/√2))` directly (different float-arith order).
    - Bias add timing: pre-quant (fp32 path) vs post-quant (int path)?
    - Pertoken vs perchannel scale broadcasting axes
    - Intermediate dtype promotion points (fp16 → fp32 → fp16 vs all-fp32)

  **Input 2 — Math formula** (from doc, paper, or op-class general
  knowledge). This is the semantic check — confirms the A3 source is
  computing the documented function and lets you sanity-check edge
  cases (NaN propagation, divide-by-zero, domain bounds).

  **The rule**:
  - `model.py` MUST reproduce the A3 source's specific algorithm
    sequence — NOT the math-formula default. If A3 source does
    `AscendDequant(fp32) → multiply(scale) → bias_add → GELU → CAST_RINT`,
    then `model.py` does the SAME order with `_cast_rint(...)` matching
    the hardware rounding semantics. Do not substitute
    `(int32 * scale).to(bf16)` because "it should be the same" — it is
    NOT bit-equal under fp16/bf16 LSB.
  - The math formula is the **semantic spec the A5 kernel must satisfy**.
    The A5 kernel itself is designed via math + hardware adaptation
    (NOT a byte-by-byte copy of A3 source). But the **reference we
    compare against** (`model.py`) MUST match what A3 source actually
    does, because A3 source IS the per-case ground truth Phase D loads
    from `edge_dataset.pt["a3_outputs"]`.

"""


def _pa3_phase_a_3(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""  Why this rule exists: 2026-05-21 fused_quant_mat_mul case 6 (bf16 +
  GELU-erf) — worker built `model.py` from the math formula only,
  ignored A3's specific `EpilogueDequantGeluErf::ComputeGeluErf` Horner
  sequence + `CAST_RINT`, then ran a 14-hypothesis sweep trying to
  match A3's pre-captured output. The 14 hypotheses were all
  formula-derivable variations; the actual sequence (`AscendDequant
  bf16-scale path → x1_scale pertoken multiply → fp32 bias → ComputeGeluErf
  with library coefficients → CAST_RINT to fp16`) was never in the
  sweep because the worker never read CANN source. Worker then claimed
  "my kernel matches my model.py bit-exact, A3 is non-canonical" —
  that's reward hacking: building a friendly reference to make the
  kernel pass.

  Anti-pattern (forbidden): writing `model.py` from formula only +
  appealing to OL-109 Tier-2 "hardware floor" or "A3 non-canonical"
  when the case fails. Phase D verifies against captured A3 outputs,
  so a `model.py` that doesn't match A3 source semantics IS THE BUG
  in the reference, not a property of A3.

  Read only the source-architecture detector's arch22 evidence and its
  arch22-reachable top-level/common dependencies. Never traverse, open,
  search, hash, compare, or summarize a co-located arch35 specialization.
  Cite the allowed arch22 file and line for each algorithm detail used.

A.2.8. **CASE-COUNT — port_a3 manifests MUST use case_gen (NOT hand-author)**
  (port_a3 mode — 2026-05-24 owner directive 22:10Z, supersedes 2026-05-22 ≥20-case heuristic).

  **Owner direction 2026-05-24T22:10Z**: 必须用 case_gen 系统化展开，已经
  pass 的 op 用修复的 harness 重新 cold start 验证。Audit caught all 20
  prior port_a3 archives hand-authored 8 cases → under-coverage. Hand-rolled
  `_make_case()` loops are FORBIDDEN going forward.

  **Required pattern**:
  ```python
  # input_gen.py top:
  from case_gen import dataset_data_sha256, generate_cases
  SCHEMA = {{
      "op_name": "<op>",
      "formula": "<pseudo-code>",
      "tensor_inputs": [{{"name": ..., "role": "operand"}}, ...],
      "scalar_inputs": [...],   # if op has scalar attrs
      "tensor_output": "<output_name>",
      "rank": <N>,              # tensor rank
      # OPTIONAL fused-op extensions: base_shape_filter, shape_derive, value_gen
  }}
  COVERAGE_TIER = "sign_off"    # ~40 cases; or "production" (~60)
  ```

  workflow_critic gates `O2_5.B.inv1` + `O2_5.B.inv3` ENFORCE this — spawn
  rejected if `from case_gen import` / `SCHEMA` / `COVERAGE_TIER` missing in
  input_gen.py, or `data_sha256` / `coverage_tier` missing in manifest.json.

  **SCHEMA-author recipe** (from V220 source):
  1. Open `<port_source>/op_host/<op>_def.cpp` — input/output names + dtypes
  2. If admitted in `.opgen_state.json.source_arch_detection.analyzed_paths`,
     open top-level `<port_source>/op_kernel/<op>.cpp` — rank from `GM_ADDR`
     parameter list + tiling attrs
  3. Fill `SCHEMA` dict per case_gen contract — `src/scripts/reference_provider/input_gen.template.py` has worked examples
  4. For fused/multi-input ops with interdependent shapes (e.g.
     sparse_indices ≤ S2), use `base_shape_filter` callback — NEVER fall
     back to hand-rolling cases

  **Coverage axes case_gen handles automatically**:
  - Tail/non-aligned shapes (alignment, partition, tile boundary, prime)
  - Distribution stress (uniform, constants, large/small_mag, denormal, mixed_sign)
  - Scalar probe variants (per scalar's `probe_values` list)
  - Multi-dtype expansion (if scalar_inputs declares dtype probe)

  **Exception** (very narrow): if SCHEMA truly cannot express an op's
  semantics (e.g. graph-level op with no tensor inputs), file a DEBT entry
  + use `.workflow_exception_O2_5` user-signed waiver. Do NOT silently
  hand-author 8 cases.

  **Why it matters** (post-incident 2026-05-24 + post 2026-05-22 fused-op
  bias-column-band catch): with 8 hand-authored cases, per-shape coverage
  is 1-2% of customer workload variance. case_gen sign_off tier ~40 cases
  distributes coverage across alignment / distribution / scalar probe →
  catches first-iter on bugs that affect only specific tail/edge shapes.

A.2.7. **TILING STRUCT — READ FROM ARCH22 SOURCE, DO NOT L4-ESCALATE ON SIZE**
  (mandatory, port_a3 only).

  The TilingData struct definition lives in the provided V220 op_host source.
  **Before declaring L4 "TilingData struct too large to reimplement", you MUST
  first read the arch22 struct definition and reproduce it in your
  build TU** — a many-field struct is NOT grounds for L4 escalation.

  Concrete check (mandatory at end of Phase A, before any L4 escalation):
  ```bash
  # Enumerate top-level arch22 host headers only; never recurse into target dirs.
  find <port_source>/op_host/ -maxdepth 1 -type f \\( -name '*tiling*.h' -o -name '*regbase.h' \\) | head -10
  # Grep for the struct name your kernel files reference
  grep -l "FlashAttentionScoreSimplifiedTilingData\\|<your-op-TilingData-struct-name>" <port_source>/op_host/*.h 2>/dev/null
  ```

  Reproduce the struct definition (matching the detected arch22 field layout) at the
  top of your build TU, BEFORE the algorithm `#include`s:
  ```cpp
  // workspace/{op}/kernel/{op}_kernels.cpp:
  struct <Op>TilingData {{ /* fields matching arch22 op_host/<op>_tiling.h */ }};
  ```
  Do NOT `#include` a target `op_host/.../arch35/*.h` into the build TU. Read
  only the top-level arch22 struct, then author your own definition.

  **Anti-pattern (caught 2026-05-22 FA evening retry)**: worker counted
  TilingData struct field-count + LOC of tilingkey logic, declared
  "30-50 iter level / structural_rewrite_needed L4 escalation", aborted
  Phase A — without reading the struct header from upstream source.
  Result: 4 sequential cold-start attempts all reached the same L4
  judgment. Worker's Phase A "Source & references" section must cite the
  actual allowed top-level `op_host/*.h` files it read for the struct layout.

  L4 escalation is reserved for cases where: (a) the struct is NOT present
  in the arch22 source at all, OR (b) tilingkey computation is fundamentally
  non-portable (requires host CANN runtime API that has no in-workspace
  equivalent), OR (c) hand-written tilingkey logic exceeds worker iter
  budget. NOT for "the arch22 struct has many fields" alone — reproducing
  the struct definition solves that.

A.3. **CROSS-OP ROUTER CHECK** — if peer_op_dependencies is non-empty:
  - Read peer's `op_api/<peer>.cpp` and locate the dispatcher function
    (typically named like `<PeerOp>()` returning the entry function).
  - Identify the 3 router-edit primitives per KB W9 cross-op router pattern:
    (a) **branch redirect**: `IsregBaseAiCoreSupport` branch needs to call
        the current op's entry (`<CurrentOp>AiCore`), not the peer's.
    (b) **support-gate extend**: any `IsV<N>AiCoreSupport` that excludes
        Ascend950 needs `!Ops::NN::AclnnUtil::IsRegbase()` added.
    (c) **alignment unify**: if peer's aclnn allocates output shape with
        platform-conditional alignment (e.g. `IsRegbase() ? raw :
        8-aligned`), strip the conditional so all platforms agree.
  - List the planned peer edits in `analysis.md §Cross-Op Edits`.

"""


def _pa3_phase_b(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Phase B — Write A5 (arch35) artifacts

B.1. **Kernel** — independently write
  `workspace/{op}/kernel/arch35/<op>.h` and any required sibling headers.

  **Source boundary (RFC §15.2, mandatory):** derive the authoritative interface,
  tensor contracts, algorithm order, and constants from the detected arch22
  source. Trusted API/KB documentation and provenance-tracked target prior art
  may be consulted as advisory migration context, but cannot become truth.

  - Record the arch22 input/output contract and algorithm steps in `analysis.md`.
  - Design the arch35 tiling, queueing, synchronization, and entry class from
    those semantics and documented arch35 interfaces.
  - Replace arch22-only includes and guarded primitives with verified arch35
    APIs; do not perform a line-by-line textual transformation.
  - Audit `ToFloat<T>` calls against the verified target API restrictions (KB W11).
  - Before verification, run the allowed-input provenance gate. It records the
    generated output, declared arch22 evidence, and any target/prior-art context;
    verified prestage entries must still match their recorded SHA256.

  **B.1.bis — OL-103 hw-transcendental check (MANDATORY, 2026-05-19, task #42)**:

  Before finishing kernel write, grep your kernel files for these hw
  transcendental ops (OL-103 forbidden list in fp32 output paths):

      `AscendC::Exp` / `AscendC::Log` / `AscendC::Sigmoid` /
      `AscendC::Tanh` / `AscendC::Reciprocal` / `AscendC::Sqrt` /
      `AscendC::Rsqrt` / `AscendC::GeluV2` / `AscendC::Div`

  If ANY of these appear in a kernel path that produces fp32 output (or
  is verified at fp32 threshold against CPU/A3 reference), **the kernel
  precision verdict will hit max_abs_diff ~ 1 fp32 ULP** (NOT bit-exact)
  because hw transcendentals are ~fp16 mantissa precision per OL-103.

  **Required action when hw transcendental detected in fp32 path**:
  1. Read OL-103 §"Tier 1 software fp32 sigmoid (canonical implementation)"
     for the bit-exact ops palette (Mul/Adds/Cast/ShiftLeft/ReinterpretCast/Newton).
  2. Read the software-transcendental recipe in the codified KB
     (software exp via range-reduction + Horner polynomial + 2^k
     reconstruction; software 1/x via Newton iteration; precision ~2^-21 relative).
  3. Adapt the pattern to your op's transcendental:
     - Sqrt(x) → Rsqrt via Newton then Mul: `y_{{n+1}} = y_n*(1.5 - 0.5*x*y_n^2)`,
       then sqrt = x * y (3 Newton iters saturates fp32).
     - Exp(x) → range-reduce by ln2: k=round(x*log2e), r=x-k*ln2, exp(x) = poly(r) * 2^k.
     - Div(x,y) → 1/y via Newton then Mul: `y_{{n+1}} = y_n*(2 - y*y_n)`, 3 iters.
     - Reciprocal(x) → 1/x via Newton same as Div with x=1.
  4. **fp16/bf16 output paths**: hw transcendental is usually OK since the dtype
     itself absorbs the residual error. Decision: if all dtype cases are
     fp16/bf16 → keep hw ops; if ANY fp32 cases AND reference is bit-precise
     (migration uses fresh arch22 source-NPU truth) → use a software implementation.

  **Tradeoff** (must document in `analysis.md`): software impl costs perf
  (op#2 SwiGLU evidence: ~2× slowdown on fp16/bf16 paths because the
  software fp32 sigmoid runs even when output dtype absorbs residual).
  Optimizer may revisit this after Phase D verifies precision PASS.

  **Skip software impl ONLY IF** ALL ≥30 cases (per §A.2 coverage spec)
  satisfy v2.1 §4.5.1 max_rel_diff
  ≤ 2^-21 (fp32) / 2^-9 (fp16) / 2^-6 (bf16) with the hw path AND the
  per_case `max_abs_diff` is the residual NOT a bug. v2.1 v21_classification
  reclassifier `src/scripts/precision_eval_v21.py` will report PASS in that
  case — but it leaves performance headroom for software-impl future upgrade.
  **Coverage discipline**: if `input_gen.py` emits fewer than 30 cases, the
  skip decision is UNDER-DECIDED — proceed to software impl rather than
  claiming coverage you don't have (per feedback_input_requirements_immutable).

B.2. **A5 entry point** — write `workspace/{op}/kernel/<op>_apt.cpp`:
  Body shape per KB W8 (`ops_nn_layout/apt_entry_point.md`):
  ```cpp
  #include "kernel_operator.h"
  #include "kernel_tiling/kernel_tiling.h"
  #include "arch35/<op>.h"
  using namespace <OpNS>;
  extern "C" __global__ __aicore__ void <op>(
      GM_ADDR <input_1>, ..., GM_ADDR workspace, GM_ADDR tiling)
  {{
      GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
      GET_TILING_DATA(tilingData, tiling);
      <OpClass><TemplateArgs> op;
      TPipe pipe;
      op.Init(&tilingData, <inputs>, <outputs>, usrWorkspace, &pipe);
      op.Process();
  }}
  ```

B.3. **Binary config** — write
  `workspace/{op}/op_host/config/ascend950/<op>_binary.json` and
  `workspace/{op}/op_host/config/ascend950/<op>_simplified_key.ini`.
  Schema per KB W8 (`ops_nn_layout/op_def_ascend950.md`). Reuse the
  ascend910b binary.json as starting point; adjust `bin_filename` hash,
  preserve `op_type` + `inputs[]` + `outputs[]` + `attrs[]`.

B.4. **op_host COMPLETE GE DELIVERABLE** (NEW 2026-05-14 / generalized 2026-06-16
  — fixes archive gap reported by user 02:55Z, PB-33; this is deliverable ① of the
  COMPLETE ARCHIVABLE DELIVERABLE contract above). Per PR4778 spec, the archive MUST
  ship complete files (not patches). Patches are useful for review but lose all
  context once detached from the master they were generated against. GENERATE each
  op_host file by following the per-file CARRY / CARRY+PATCH / REPLACE-HOOK recipe
  (`GE_HOST_TRANSFORM_RECIPE.md`, see the deliverable-① block above) deriving from
  YOUR op's A3 (arch22) op_host source — do NOT byte-copy CANN arch35. Write the
  MODIFIED-FOR-A5 version as a COMPLETE FILE in `workspace/{op}/op_host/`:

  - `workspace/{op}/op_host/<op>_def.cpp`  — full file with `regbaseCfg` block added
  - `workspace/{op}/op_host/<op>_tiling.cpp` — derive from the arch22 tiling
    contract and target KB recipe (see OL-143 / PB-32 where applicable)
  - `workspace/{op}/op_host/<op>_tiling.h` — independently declare the target
    tiling interface required by that implementation
  - `workspace/{op}/op_host/CMakeLists.txt` — author from the required source list
    and verified target build contract (PB-29/30)
  - `workspace/{op}/op_host/<op>_infershape.cpp` — derive from the arch22 interface
    semantics when the source exposes inference behavior
  - `workspace/{op}/op_host/op_api/<op>.cpp` + `<op>.h` — independently implement
    the public interface from the arch22 op_api contract when one exists
  - Plus the binary config files from B.3

  **Hard rule**: after Phase B, the generated op_host must contain every
  required customer deliverable justified by the detector-admitted, top-level
  arch22 interfaces and repository templates. Do not recursively enumerate or
  mirror `<port_source>/op_host/`; target subdirectories and unrelated files
  are outside the input boundary.

  **Also write the patch** (for review trail): `workspace/{op}/op_host/<op>_def.cpp.patch`
  is STILL produced (diff between upstream's _def.cpp and your modified version).
  But the COMPLETE file is the ship artifact, not the patch.

B.5. **Cross-op peer router patch** (if A.3 listed peer edits) — write
  `workspace/{op}/peer_router.patch` (diff vs each peer's
  op_api/<peer>.cpp). One patch file lists all peer edits.

  ALSO write the complete modified peer file(s) to
  `workspace/{op}/peer_complete/<peer_op>/op_api/<peer>.cpp` (same reason
  as B.4 — ship complete files, patch is for review only).

"""


def _pa3_phase_c(
    *, op, workspace, iter_cap_remaining, port_source, aclnn_entry, gen_data_source, peer_deps_line, env
) -> str:
    return f"""## Phase C — Build + smoke

C.1. Sync writes to A5 host (`{env.host}` container `{env.container}`).
C.2. Apply patches:
  - Apply `<op>_def.cpp.patch` against the on-host ops-nn checkout
  - Apply `peer_router.patch` if present
  - Deploy the independently generated arch35 kernel + apt.cpp into the on-host op_kernel/ dir
  - Copy ascend950 config files
C.3. Build: `cd <on_host_ops_nn>/<category>/{op} && bash build.sh
     --pkg --ops={op} --soc=ascend950` (note: DOUBLE dash on --soc per
     KB W8 "ops_nn_build.md"; check `build/gen_bisheng_dir/bisheng`
     shebang per KB W10).
C.4. Verify .so produced.

"""
