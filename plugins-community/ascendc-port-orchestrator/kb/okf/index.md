# porter OKF 知识库

porter 从 `kb/` 迁移而来的 OKF 卡片库。三个内容板块：
* [reference](reference/index.md) — 仅保留少量 porter 本地资产；官方知识由 cannbot-knowledge 提供
* [runbooks](runbooks/index.md) — inferred 121 张 stub 保留；operator-optimization 282 张、field-notes 348 张、hardware 9 张已迁移至 cannbot-knowledge
* [ops](ops/index.md) — 按算子设计卡（porter 一般不产此层）

引擎运行时只通过独立 **`cannbot-knowledge`** checkout 的 `knowledge-query` 消费官方知识；本目录不再作为官方知识检索根。
