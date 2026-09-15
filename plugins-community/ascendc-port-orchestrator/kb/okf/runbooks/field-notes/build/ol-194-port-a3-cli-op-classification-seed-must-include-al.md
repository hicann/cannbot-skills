---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`--port-a3` CLI op_classification seed MUST include algorithm-class tags + complexity, not just the mode tag — otherwise FA-class detection cannot fire for FA-class ports"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=port_a3_to_a5 (CLI-flag-driven classification)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=port_a3_to_a5 (CLI-flag-driven classification)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-194
timestamp_inferred: true
tags: [is_fa_class, ascendc, ol-194]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=port_a3_to_a5 (CLI-flag-driven classification)`
`verified_on: soc=Ascend950PR; cann=9.0.0`

> **Note (task#17)**: this OL originally described the (now-removed) historical IL routing gate. The **seed principle is still live** — the FA-class tags it seeds now drive `is_fa_class` detection → the kw **template-assembly** path (owner 2026-06-07), not any IL chain. Rewritten below to the current routing.

**Principle**: When `_cmd_port_a3` seeds `workspace/<op>/op_classification.json` with the CLI-flag-authoritative shortcut (`source: "cli_flag_port_a3"`), the seed MUST include algorithm-class tags + `op_complexity` IN ADDITION to the mode tag. Otherwise FA-class detection (`is_fa_class`, which needs FUSED+SOFTMAX tags) cannot recognize the op, because the mode-only seed (e.g. `["a3_to_a5_port"]`) lacks FUSED+SOFTMAX, and `phase_o17_classify._read_cached` short-circuits to the seed (intentional design: CLI flag IS the classification, no LLM re-classify needed). The state machine's `snap["op_taxonomy"]` then comes from `schema_norm._detect_op_class` reading the same seeded `op_classification.json` — same hole, same outcome: the op is not recognized as FA-class and the kw FA template-assembly recipe is not selected.

**Concrete fix anchor** (orchestrator.py `_cmd_port_a3`):
```python
op_class_tags = ["a3_to_a5_port"]
op_complexity = None
op_lower = op_name.lower()
if "flash_attention" in op_lower or "fused_attention" in op_lower or "fusion_attention" in op_lower:
    op_class_tags.extend(["FUSED_SOFTMAX", "fa_class"])
    op_complexity = "L4"
# seed op_class_tags + op_complexity in op_classification.json
```

After fix: a FA-named `--port-a3` cold-start seeds FUSED_SOFTMAX + fa_class + L4, so `is_fa_class` recognizes the op and the kw worker receives the FA template-assembly recipe (arch22 spec + P-P103 KB templates).

**Other instances (predicted)**: any future `--port-a3` op whose FA-class recognition depends on the seeded tags — any op_name containing `flash_attention` / `fused_attention` / `fusion_attention` keywords. Extend the `_cmd_port_a3` seed as new algorithm classes get op-class-specific handling.

**Cross-ref**: OL-176 (empirically-attempt-build-before-L4-declare anti-escalation discipline).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-194（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
