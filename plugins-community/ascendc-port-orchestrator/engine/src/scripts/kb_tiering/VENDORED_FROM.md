# Vendored contract provenance

`interface.py` is **byte-identical** to a5_ops `8165e991:src/scripts/kb_tiering/interface.py`
(the three-way-APPROVED stable contract — Entry / KBProvider / gate / Arbiter + hard_key/full_sig/
jaccard helpers). Implements `KB_TIERING_DESIGN §14` (v0.5 `a64fd91a`).

**Do NOT edit `interface.py` here.** It is the frozen 3-party contract; any change goes through
main (a5_ops), who keeps the a5ops / autoport / cannbot adapters consistent. This vendoring is the
"retirement-clean interim" of design §9: when a5_ops-core lands its wiring, cannbot re-syncs as a
code-swap, not a data-migration.

`adapter_a5ops.py` is ALSO vendored byte-identical from a5_ops
`origin/main:src/scripts/kb_tiering/adapter_a5ops.py` — it is the b-tier reader over the
`references/` KB_INDEX + OL/EC/PB format, which cannbot bundles verbatim. Same do-not-edit rule;
re-sync is a code-swap. `adapters/cannbot_b.py` is a thin cannbot factory pointing it at the
bundled `src/skills/references/`.

cannbot-local pieces (NOT vendored — cannbot's own): `adapters/cannbot_c.py` (Markdown user_kb →
Entry) + `adapters/cannbot_b.py` (factory over the vendored a5ops reader) + the read/write wiring
into the engine. See notes CANNBOT_KB_TIERING_GAPMAP.md.
