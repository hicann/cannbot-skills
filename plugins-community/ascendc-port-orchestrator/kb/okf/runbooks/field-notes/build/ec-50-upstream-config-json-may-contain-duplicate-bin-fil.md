---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Target prior-art JSON may contain duplicate bin_filename keys — regenerate a valid task-owned schema"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5"
phenomenon: build_failure
signal:
  - "While mirroring upstream op_host/config/ascend950/<op>_binary.json into a port_a3_to_a5 workspace, the file contains a JSON object with TWO bin_filename entries"
confidence: single_run
original_id: EC-50
timestamp_inferred: true
tags: [bin_filename, ascendc, ec-50]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2.5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: workspace/foreach_reciprocal/knowledge_update.md (kw-1, 2026-05-16)`

**Symptom**: While mirroring upstream `op_host/config/ascend950/<op>_binary.json` into a port_a3_to_a5 workspace, the file contains a JSON object with TWO `bin_filename` entries on adjacent lines for the same dtype variant:

```json
{
    "bin_filename": "ForeachReciprocal_8d9f799857af3a32fcb6092255dfdab9",
    "bin_filename": "ForeachReciprocal_10f6ed20a89d7d8d379ca7132257bfa5",
    "inputs": [...]
}
```

**Root cause**: This is a JSON spec violation (RFC 8259 forbids duplicate keys in the same object) shipped in upstream's prebuilt config. The first hash is a stale entry from a prior build run that wasn't pruned; the second hash is the current build's effective bin_filename. Most JSON parsers (Python `json.loads`, C++ `nlohmann::json` default config, jq) implement **last-key-wins** semantics, so the runtime picks the second hash and the binary loads correctly.

**Fix**: treat the target JSON as advisory evidence and generate an RFC-8259-valid task-owned schema
from the selected contract and current build outputs. If the effective value is ambiguous, fail and
require an explicit canonical value. Validate `bin_filename` against the current clean-build binary;
target byte identity is not an acceptance gate.

**Anti-patterns**:
- Copying the duplicate-key target file because a default parser happens to use last-key-wins.
- Normalizing it silently without recording the canonical value and its build provenance.

**Escalation**: if the selected contract and current build cannot determine one canonical filename,
return a visible schema/provenance failure. Do not substitute the target file as a workaround.

**Historical evidence**: foreach_reciprocal 2026-05-16 target prior art carried two bf16
`bin_filename` keys; the old pipeline accepted last-key-wins. This demonstrates the ambiguity and
motivates strict task-owned schema generation; it does not justify copying the invalid JSON.

**Other instances (predicted)**: any upstream `<op>_binary.json` for ports where the build farm has multiple historical hashes. Quick detector: `python -c "import json; json.loads(open('<f>').read(), object_pairs_hook=lambda p: [None for k,_ in p if list(zip(*p))[0].count(k)>1])"` — flags duplicate keys at parse time.

**Cross-reference**: OL-141 (target artifacts are advisory, never a skip or byte-mirror verdict),
OL-157 (foreach unary family packaging patterns).

<!-- 迁移自 porter kb/target/ascendc/（EC-50，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
