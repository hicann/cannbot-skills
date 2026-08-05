# porter OKF 知识库

porter 从 `kb/` 迁移而来的 OKF 卡片库。三个内容板块：
* [reference](reference/index.md) — API/指南参考卡（本期从 migration 逐 API 卡 + hardware 迁入）
* [runbooks](runbooks/index.md) — 跨算子优化点（OPT-*/AP-*）+ field-notes 实战现象卡（EC/PB/OL 迁入）
* [ops](ops/index.md) — 按算子设计卡（porter 一般不产此层）

引擎：外部社区插件 **`cannbot-knowledge`**（运行期 `knowledge-query`；维护者可选 *knowledge-lint*，RFC #381 —— 不再随本插件 vendored）。本插件经 `engine/src/scripts/okf/okf_kb.sh` 调用，检索用 `--knowledge-root kb/okf`。卡片按 **`okf.v1`** 规范由本插件自维护（迁移脚本 `engine/src/scripts/okf/migrate_cards_to_okf_v1.py`）。
