# Vendored contract provenance

`interface.py` is **byte-identical** to a5_ops `8165e991:src/scripts/kb_tiering/interface.py`
(the three-way-APPROVED stable contract — Entry / KBProvider / gate / Arbiter + hard_key/full_sig/
jaccard helpers). Implements `KB_TIERING_DESIGN §14` (v0.5 `a64fd91a`).

**Do NOT edit `interface.py` here.** It is the frozen 3-party contract; any change goes through
main (a5_ops), who keeps the a5ops / autoport / cannbot adapters consistent. This vendoring is the
"retirement-clean interim" of design §9: when a5_ops-core lands its wiring, cannbot re-syncs as a
code-swap, not a data-migration.

~~`adapter_a5ops.py`~~（曾 vendored byte-identical 自 a5_ops，b-tier KB_INDEX 读取器）与
~~`adapters/cannbot_b.py`~~（指向 bundled `kb/KB_INDEX.md` 的工厂）已随 **OKF-only 迁移
（2026-08）删除**：bundled 知识即 kb/okf，不再有 KB_INDEX provider，b-tier 的存在基础消失。
`demo/`、`poc/`（legacy b-tier 演示代码）同批删除。

cannbot-local pieces (NOT vendored — cannbot's own): `adapters/cannbot_c.py` (Markdown user_kb →
Entry) + the read/write wiring into the engine (`read_bridge.py` 现为 c-only 组装).
See notes CANNBOT_KB_TIERING_GAPMAP.md.
