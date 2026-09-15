---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class FORWARD operators with softmax/attention require codified AscendC tile templates, not a naive single-spawn rewrite [V220; arch22→arch35 scope narrowed by OL-185]"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=flash_attention_score_forward, incre_flash_attention, flash_decoding, mha_forward — only when the kernel contains softmax + tile-scheduled Q×K@V"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=flash_attention_score_forward, incre_flash_attention, flash_decoding, mha_forward — only when the kernel co"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-159
timestamp_inferred: true
tags: [ascendc, ol-159]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=flash_attention_score_forward, incre_flash_attention, flash_decoding, mha_forward — only when the kernel contains softmax + tile-scheduled Q×K@V`
`do_not_apply_to: backward / gradient operators (including lightning_indexer_grad and flash_attention_score_grad); classify those as L2/L3 per OL-185 unless their own source proves the same forward online-softmax schedule`
`verified_on: 3_FusionAttention cold start on V220 plus direct AscendC probes on V351`

**Principle**: FA forward correctness couples the algorithm, tile schedule, cube/vector split,
online-softmax carry state, and cross-core synchronization. A worker must assemble the kernel
from the codified arch22 specification and FA template blocks (P-P103 and related entries).
Generating the full kernel from a Python reference in one unconstrained pass is not a reliable
path.

**Cold-start trigger** — select the standard kw worker's FA template-assembly recipe after
Phase A when all of the following hold:

1. The op is a forward FA-class operator and source inspection confirms softmax plus
   tile-scheduled Q×K@V.
2. There is no prior workspace kernel with meaningful coverage, or the prior archive remains
   at no more than 5 passing cases.
3. The upstream arch35 implementation is absent or cannot be staged as a self-contained
   kernel because its dependency closure and tiling-key registration are unavailable.

Do not treat high LOC, Mmad call count, or CrossCore call count alone as this signal. Those are
surface indicators; OL-185 requires checking the algorithmic class first. Backward and
non-attention cube+vec operators stay on the normal migration path.

**Why template assembly is required**:

- Online softmax makes tile scheduling part of the algorithm: every K tile updates a running
  max and sum, and the rescale must stay consistent with the Q-row/K-column partition.
- Cube and vector stages share workspace, event IDs, and visibility rules that must be emitted
  as one validated schedule.
- FA variants add mask, layout, GQA, and auxiliary-output deltas. The reusable template keeps
  those deltas explicit instead of asking each worker to reconstruct a 1500–2500-line kernel.

**Evidence**:

- A V220 3_FusionAttention cold start produced only 1/61 coverage after repeated free-form
  attempts; the reusable block/tile structure was the missing asset, not another local edit.
- An arch22→arch35 flash_attention_score migration showed that an apparent op-local arch35
  directory can depend on a much larger shared-common closure and binary-build-only tiling-key
  registration. Phase A must audit that closure before deciding the reference is stageable.
- On V351, a direct AscendC single-launch Pattern A probe ran correctly and deterministically.
  Therefore this entry does not claim that another IR is architecturally necessary; it requires
  a validated AscendC template when the ordinary staging path is unavailable.

**Worker action**: write the Phase A evidence, choose the FA template blocks, assemble a
self-contained AscendC deliverable, then run the normal precision/determinism gates. If the
required template block or source contract is missing, report that concrete missing asset and
effort estimate; do not route to a removed backend or emit a placeholder kernel.

**Cross-references**: OL-141 (upstream-reference preflight), OL-156 (structural signature),
OL-176 (build before declaring a ceiling), OL-185 (forward-vs-backward scope), P-P103
(FA template skeleton and block inventory).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-159（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
