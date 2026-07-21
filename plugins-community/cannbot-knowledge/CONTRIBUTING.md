# Knowledge Contribution Guide

本文说明外部贡献者如何向独立 cannbot-knowledge 知识库提交 PR、勘误和新增知识。`cannbot-skills` 插件仓只接收工具、流程和规范改动；真实知识正文应提交到独立知识库仓库。

## 可接受的贡献

- `reference/`：官方文档、API、指南、Profiling、术语的蒸馏卡片。
- `ops/`：基于 fixed commit golden 源码的算子设计 wiki。
- `runbooks/`：跨算子优化点、反模式、field notes、版本迁移经验。
- `contrib/`：社区贡献的知识卡片、勘误、补充案例和待验证经验。
- 勘误：修正错误 API、平台限制、Tiling、Kernel、UB 布局、优化结论或失效边界。
- 治理修复：frontmatter、source、index、graph、search、log、死链、status/confidence。
- 知识编译能力：新增或改进把上游材料编译成 OKF 卡片的 skill、adapter 或工具。
- 知识检索能力：新增或改进 `knowledge-query` 检索模式、召回/排序策略、alias/tag/type 映射。
- 工具修复：lint、query、graph、compile 脚本和相关验证方案。

## 不接受的贡献

- 无 source 的高风险 API、平台、精度、性能或默认注入结论。
- 大段复制官方文档、源码、日志或第三方材料。
- 密钥、token、账号、内网地址、私有路径、不可公开日志。
- 只适用于个人环境且无法复现、无法追溯证据的经验。
- 把单个算子的私有技巧误写成通用 runbook 优化点。
- 静默删除错误知识；错误知识应优先 `retracted` 或转成负知识。

## PR 必须说明

- 变更类型：新增、更新、勘误、治理修复、知识编译能力、知识检索能力、工具修复。
- 影响范围：涉及哪些内容根、卡片、图谱、索引或工具。
- 证据来源：官方文档 URL、GitCode fixed commit 链接、开发轨迹证据或复现实验。
- 处理结论：哪些 applied、deferred、rejected，以及原因。
- 验证结果：列出已运行的 lint、query、graph、ops/runbook 分层校验；涉及测试时补充 UT/ST 结果。

## 贡献知识编译 skill

知识编译 skill 的职责是把上游材料编译成符合 OKF 规范的知识卡片。插件仓只接收编译能力本身，不接收真实大体量知识正文。

新增或改造知识编译能力时，必须满足：

- 先定义输入来源、目标知识类型和落点目录，例如 `reference/`、`ops/`、`runbooks/` 或 `contrib/`。
- 产物必须是蒸馏后的 OKF 卡片，不能复制原文、源码或日志。
- 卡片 schema 必须说明 `type`、`resource` / `sources`、`tags`等字段如何生成。
- git 源必须固定到 commit；文档源必须能追到稳定 URL 或版本 pin。
- 对已有卡片必须做增量更新，不整体覆盖。
- 实质知识变更必须同步 `index.md`、`graph/`、`log/` 和必要的 query verify。

编译入口统一由 `ops-knowledge-ingest` 路由：

- 如果只是接入新的文档、golden 源码或 trace 形态，优先扩展已有生产 skill。
- 如果出现新的知识类型或生成链路，可以新增 `skills/<name>/SKILL.md`，必要时在该 Skill 的 `scripts/` 下增加唯一脚本入口。
- 新增 skill 后必须同步更新 `skills/ops-knowledge-ingest/SKILL.md`，说明触发条件、输入参数、目标目录、委派到哪个生产 skill，以及收尾验证命令。
- `ops-knowledge-ingest` 只做入口判断和路由编排，不应在其中直接写具体卡片正文。

知识编译类 PR 必须额外说明：

- 新增或修改了哪类知识卡，目标目录是什么。
- 为什么现有生产 skill 不能覆盖，或为什么选择扩展现有 skill。
- 生成链路如何保证 source、frontmatter、index、log、graph 和 query verify。
- 与该知识类型匹配的检索模式是否已经存在；不存在时必须同时补充检索模式，或说明暂不需要检索扩展的理由。

## 贡献知识检索 skill

知识检索扩展的默认入口是 `knowledge-query`。新增知识类型或新增社区贡献卡片后，应优先扩展检索模式，而不是在主检索逻辑中临时硬编码。

新增检索能力时，优先修改：

- `skills/knowledge-query/modes/<mode>.md`：新增或更新检索模式卡片。
- `skills/knowledge-query/modes/README.md`：登记模式用途。
- `skills/knowledge-query/SKILL.md`：如果该模式属于常用入口，同步加入模式列表。
- 独立知识库中的 `metadata/type_map.yaml`、`metadata/tag_registry.yaml`、`metadata/aliases.yaml`：当新增 type、tag 或别名时同步说明或提交对应知识库改动。

每个检索模式应说明：

- 适用场景：什么问题触发这个模式。
- 目标知识：覆盖哪些目录、type、tags、paradigms、status。
- 召回路线：使用 bm25、tfidf、tagtype、graph、dense 中哪些方法。
- 排序路线：使用 bm25f、tagidf、quality、reranker 或 llm-judge 中哪些方法。
- 图谱扩展：是否需要 `neighbors` 多跳补充 API、算子、runbook 或 field note。
- 注入方式：命中后如何 `get` 整卡，如何引用 doc-id、title、resource 和 confidence。
- 验证场景：给出至少一个 query verify 场景或可复现命令。

检索类 PR 必须额外说明：

- 新模式解决什么检索缺口，是否已有模式可以覆盖。
- 是否新增 type/tag/alias，以及这些字段如何进入 metadata 或 frontmatter。
- 默认是否只检索 `verified`，是否排除 `stub`、`deprecated`、`retracted`。
- 是否会改变已有模式的排序结果；如果会，说明影响范围和回归验证方式。

只有在检索模式无法表达新能力时，才考虑新增独立检索 skill。新增独立 skill 必须解释为什么不能复用 `knowledge-query` 的 mode、recall、rerank、get 和 neighbors 组合。

## 单卡准入标准

- frontmatter 完整，至少包含 `type`、`title`、`description`、`resource` 或 `sources`、`tags`、`status`、`confidence`。
- git 源必须使用 fixed commit 的 `blob` 或 `tree` URL。
- 多源卡必须有且只有一个 `primary`，且 `resource == primary.url`。
- 正文必须是蒸馏后的使用知识，不是原文或源码搬运。
- `reference/` 写官方事实；`ops/` 写单算子设计事实；`runbooks/` 只写跨算子可复用规律、失效边界和坏实践；`contrib/` 写社区贡献、勘误、补充案例和待验证经验。
- 每级 `index.md` 只列本层直接子项。

## 勘误规则

- 勘误必须指出错误位置、错误原因、正确证据和替代结论。
- 过时知识标 `deprecated`，错误知识标 `retracted`，并写明原因、时间和替代关系。
- 撤除正知识时，优先沉淀成负知识或坏实践，避免后续重复踩坑。
- 平台差异、版本差异、精度差异不能互相覆盖，必须写清适用条件。

## 合入门槛

普通知识变更至少通过：

```bash
python3 skills/knowledge-query/scripts/knowledge_query.py --knowledge-root <knowledge-root> verify
python3 skills/ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <knowledge-root> verify
python3 skills/knowledge-lint/scripts/knowledge_lint.py --knowledge-root <knowledge-root>
```

涉及 `ops/` 或 `runbooks/` 的变更额外通过：

```bash
python3 skills/ops-knowledge-vv-ingest/scripts/validate_layered_knowledge.py --knowledge-root <knowledge-root>
```

涉及知识编译 skill 的 PR 还必须说明 `ops-knowledge-ingest` 路由如何触发，并给出 dry-run、样例输入或最小复现场景。

涉及知识检索模式的 PR 还必须给出至少一个可复现查询，并说明该查询如何命中新增或修改后的知识类型。

当前插件目录不预置 `tests/` 路径。涉及工具逻辑、新增 skill 或改变 skill 行为的 PR，需要说明验证方案；UT/ST 目录和 fixture 组织后续按单独测试方案落地。

## Review 分级

- R0：导航、索引、术语。需要 lint 和 link check。
- R1：文档蒸馏事实。需要 source 校验和抽查。
- R2：算子设计事实。需要 fixed commit 源码核验。
- R3：通用优化点。需要适用条件、失效边界、坏实践和语义抽查。
- R4：API、平台、精度约束。需要强 source 或领域 owner 确认。
- R5：跨任务默认注入知识。需要 source、status、confidence、conflict 和 trace 检查。
