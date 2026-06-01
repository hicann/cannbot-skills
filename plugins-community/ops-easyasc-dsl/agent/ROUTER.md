# Agent Router

First routing layer. Pick one route; each route names ONE starting file. The starting file (usually a playbook or the facts router) then points at the smaller files you actually need. Do not preload every drill-down listed — read only what the starting file tells you to, and stop once that file or the focused fact page it points to already answered your question.

## Primary Routes

| Task | Start here | Then (only if needed) |
|------|-----------|-----------------------|
| **Quick fact / value look-up** (device caps, pipe pairs, mutex signatures, hard rules, simulator gotchas) | `agent/references/facts.md` (quick chooser) | one focused facts page; then `agent/references/constraints/<topic>.md` for the *why* |
| **Write a new kernel** | `agent/playbooks/clarify-first.md` (mandatory Step 0) → `agent/playbooks/kernel-authoring.md` (tool-first fast path) | one focused facts / constraint / pattern file named by the playbook |
| **Debug an existing kernel** | `agent/playbooks/kernel-debugging.md` (match the Symptom-to-check map first) | one focused facts page; then the constraint file named by the playbook section |
| **Find an example kernel** | `agent/scripts/select_kernel_example.py` or `agent/references/examples/kernel-index.md` (filter to ≤3 candidates) | `agent/references/examples/kernel-catalog.md` (read ONLY the matching `###` entry) → source file |
| **Practice problems (write-from-scratch)** | `agent/references/examples/kernel-practice.md` | — |
| **Modify or add a tool** | `agent/playbooks/tool-authoring.md` | `agent/references/examples/tool-catalog.md` |
| **Modify or extend docs** | `agent/playbooks/doc-authoring.md` | `README.md`, `README_CN.md`, `doc/`, `doc_cn/` |
| **Repo structure question** | `agent/references/repo-map.md` | `doc/11_architecture_for_contributors.md` |
| **Which code path implements X?** | `agent/references/code-paths.md` | `agent/references/simulator-v2.md` for sim-specific paths; then source/tests |

## Reading Rule

1. `facts.md` is the chooser for quick factual questions — jump from there to one focused facts page, not all of them.
2. For authoring/debugging, read one playbook and let it pick which constraint / pattern / example files you actually need. Do not preload the full list.
3. For examples, prefer the selector tool first; otherwise filter with the index, then open only the one catalog entry that matches.
4. Open source files only when the guidance layers stop being enough.
5. `Files to study` / `Fallback references` lists at the bottom of a constraint or pattern file are depth pointers, NOT mandatory follow-ups. If the file you are in already answered your question, stop. In particular, do not bounce back through the facts router once you already reached the focused facts page that owns the value you needed.

## Machine-readable fallbacks

- `agent/index/kernels.json`, `agent/index/tools.json` — for programmatic selection tools only; agents should prefer the markdown catalogs.

## Deeper references (addressed by the starting files above)

- `agent/references/constraints/` — topic-focused constraint files (tiling, autosync, counters, precision, tail-safety, online-softmax-tail, datamove, reduction, mask, vec-stride, a2-device, a5-device, a2-vec-kernel, vec-reduction-a2)
- `agent/references/patterns/` — topology-specific pipeline shapes (a2-* and generic cube/vec combinations)
- `agent/references/simulator-v2.md` — V2 runtime and bridge map
