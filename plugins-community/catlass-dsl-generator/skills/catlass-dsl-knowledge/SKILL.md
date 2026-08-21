---
name: catlass-dsl-knowledge
description: Use when querying, validating, indexing, initializing, or appending evidence-backed CATLASS DSL kernel knowledge for Ascend NPU operator development in an OKF v0.2 project bundle
---

# CATLASS DSL OKF 知识管理

本 Skill 是公共知识入口。插件根 `knowledge/` 是只读的内置
Open Knowledge Format v0.2 bundle；运行时只读取和写入目标项目的
`.catlass-dsl/knowledge/`。内置 concept 不覆盖，项目经验只追加；不得在目标项目根
目录创建运行时 `knowledge/`。

项目 bundle 使用以下渐进发现目录：

- `.catlass-dsl/knowledge/dsl/`：API 与编程概念；
- `.catlass-dsl/knowledge/operator/`：有固定提交源码支持的算子样例；
- `.catlass-dsl/knowledge/debug/`：构建、IR、runtime 与正确性调试；
- `.catlass-dsl/knowledge/profiler/`：msOpProf 上板与仿真采集、指标和可视化分析；
- `.catlass-dsl/knowledge/optimization/`：带适用条件和证据门禁的优化候选；
- `.catlass-dsl/knowledge/learned/`：项目完整测试或 profiling 证明的结论。

## 初始化

```bash
python skills/catlass-dsl-knowledge/scripts/record_knowledge.py initialize \
  --project-root <confirmed-worktree>
```

初始化只复制缺失文件，不覆盖目标项目已有内容。

## 查询

```bash
python skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root <confirmed-worktree> \
  [--type <OKF-type>] [--tag <tag>] [--status <status>] \
  [--operator-family <family>] [--arch c310] [--text <term>] [--limit <count>] \
  [--compact]

python skills/catlass-dsl-knowledge/scripts/record_knowledge.py get \
  --project-root <confirmed-worktree> \
  --path <path-returned-by-query>
```

查询先使用正式 YAML frontmatter 解析并校验整个 bundle，再按结构化字段过滤。
`--tag` 可重复。完整结果包含项目相对路径、标题、描述、状态、验证记录和来源；
`--compact` 只返回选择所需的摘要、评分、命中字段、命中词和最多三条带行号的正文片段。
Agent 应先用 compact 查询缩小范围，再用 `get --path` 仅读取选中的完整 concept；`get`
接受 query 返回的项目相对路径，也接受 bundle 相对路径，并拒绝越界、符号链接和保留文件。
算子族和文本先通过 `query-vocabulary.yaml` 做大小写、下划线、连字符与显式别名归一化；
结构化过滤始终严格，文本按标题、算子族、标签、描述、路径、正文和 source title
确定性加权排序。全部词零命中时才允许至少半数词命中的受控放宽，结果通过 `score`、
`matched_fields`、`matched_terms` 解释原因；CLI 默认返回前 20 条并报告总数、匹配模式和
零命中建议。建议不自动参与召回。
concept 必须包含执行所需的接口、代码模式、约束和验证方法。`knowledge/operator/`
concept 保留这些通用章节，并在其中分别加入“算子算法”“分核策略与基本块切分”
“数据路径与存储层级”“流水排布、同步关系与数值精度”四个二级主题。离线时使用内置concept，
并结合 `verified` 与可选的 `stale_after` 判断可信度和新鲜度。

## 入库标准路由

新增、重写或审核 `dsl/`、`operator/`、`debug/`、`profiler/`、`optimization/`、
`learned/` concept 前，必须先完整阅读并执行
[公共入库标准](references/common-entry-standards.md)，再按目标目录完整阅读并执行对应的
专用标准：

- `knowledge/operator/`：
  [Operator 入库标准](references/operator-entry-standards.md)，包括固定源码取证范围、
  “算子算法”“分核策略与基本块切分”“数据路径与存储层级”
  “流水排布、同步关系与数值精度”四个强制主题，以及优化指导的最低标准；
- `knowledge/optimization/`：
  [Optimization 入库标准](references/optimization-entry-standards.md)；
- `knowledge/dsl/`：[DSL 入库标准](references/dsl-entry-standards.md)；
- `knowledge/profiler/`：
  [Profiler 入库标准](references/profiler-entry-standards.md)；
- `knowledge/debug/`：[Debug 入库标准](references/debug-entry-standards.md)；
- `knowledge/learned/`：[Learned 入库标准](references/learned-entry-standards.md)。

跨目录创建、修改、迁移或审核 concept 时，只读取一次公共标准，但必须读取所有相关目录的
专用标准。只执行查询、初始化、校验或重建索引时无需读取这些 references。

## 校验与索引

```bash
python skills/catlass-dsl-knowledge/scripts/record_knowledge.py validate \
  --project-root <confirmed-worktree>

python skills/catlass-dsl-knowledge/scripts/record_knowledge.py reindex \
  --project-root <confirmed-worktree>
```

`query-vocabulary.yaml` 是版本化 bundle 元数据；`validate` 会拒绝重复或冲突别名、非法
schema，以及没有任何 concept 声明的规范算子族。

每个非保留 Markdown concept 必须有合法 YAML frontmatter，并包含
`type`、`title`、`description`、`tags`、`status`、`generated`、`verified`
和 `sources`。静态 concept 的事实必须通过与 `sources[].id` 相同的 Markdown
footnote 归因；actor、ISO 时间、source resource 和 index 链接也会被校验。
`.catlass-dsl/knowledge/index.md` 是 bundle 入口；`index.md`、`log.md` 是 OKF 保留文件。
任一 `index.md` 索引子目录时必须显式链接 `<子目录>/index.md`，不能只链接目录路径
或使用 `<子目录>/` 尾斜杠形式。
`.catlass-dsl/knowledge/learned/index.md` 是可重建发现视图，
不是事实数据库。

## Learned 准入

`record --entry` 接受一个对象或对象数组。每条候选必须包含：

- 安全的 `operator_family`、`topic`，架构 `c310` 和 CATLASS/CANN 版本；
- shape、dtype、layout、仓库集成条件；
- 假设、实际修改、修改前后正确性和性能；
- 修改后正确性 `passed`；
- 至少一个项目内可访问的 test 或 profiling 普通文件；
- 最终 best kernel 的完整 SHA-256；
- `有效`、`无效` 或 `条件有效`。

```bash
python skills/catlass-dsl-knowledge/scripts/record_knowledge.py record \
  --project-root <confirmed-worktree> --entry <candidate-or-batch.json>
```

finish 应一次提交本次任务全部候选。只有完整测试或 profiling 直接支持的结论
可以进入 `learned/`；证伪假设、静态推断、不可访问证据和
`not_run` 结论不得入库。批次先整体校验；同名 concept 或批次内冲突会失败。

## 写入保证

- learned concept 使用 `O_EXCL | O_NOFOLLOW` 创建，拒绝覆盖。
- 并发写入使用 POSIX `flock`；不支持时安全失败。
- 批量 record 在同一锁内预检全部目标；任一冲突或写入失败都不保留本批次的部分 concept。
- index 使用临时文件、`fsync` 和原子替换。
- 新证据与旧结论冲突时追加新的条件化 concept，不修改历史 concept。
- `project-evidence:` source 同时绑定项目相对证据路径和完整 kernel SHA-256。

任何路径、仓库身份、证据、frontmatter 或 OKF 版本问题都返回结构化 `failed`，
不会把 `not_run` 汇总为 `passed`。
