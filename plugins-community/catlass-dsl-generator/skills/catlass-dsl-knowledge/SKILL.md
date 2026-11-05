---
name: catlass-dsl-knowledge
description: Use when querying, validating, or appending evidence-backed CATLASS DSL kernel knowledge for Ascend NPU operator development in an OKF v0.2 project bundle; this version reads and writes Markdown files directly without scripts.
---

# CATLASS DSL OKF 知识管理（独立阶段）

> **入口**：作为独立任务入口时，先加载 `catlass-dsl-workflow` 并输出阶段卡。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待长跑任务时直接等待并轮询，不结束回合。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块，
> 一次只做一类动作，禁止串成一条长命令。编译独立 300s，单例运行 `timeout -k 5 60s`，
> 批量按分片逐段执行，禁止整批套一次性超时，禁止裸跑。超时、无输出增长或卡住时立即终止
> 命令，不得死等或加大超时。
> 终止后先做卡健康判断：同一张卡的小探针或官方算子能过就判定为本算子实现的死锁，直接转
> debug 定位；探针也失败才判定为设备问题并换卡重试一次。

本 Skill 是公共知识入口。插件根 `knowledge/` 是只读的内置 Open Knowledge Format v0.2
bundle；项目经验写入目标项目的 `.catlass-dsl/knowledge/`。内置 concept 不覆盖，项目经验
只追加；不得在目标项目根目录创建运行时 `knowledge/`。

**本版本没有 `record_knowledge.py`。** 查询通过直接读取 Markdown 索引与条目完成，记录
通过按模板新建 Markdown 文件完成。

## 目录结构

```text
knowledge/
├── index.md                  # bundle 入口
├── query-vocabulary.yaml     # 算子族别名与检索词
├── dsl/                      # API 与编程概念
├── operator/                 # 有固定源码支持的算子样例
├── debug/                    # 构建、IR、runtime 与正确性调试
├── profiler/                 # 上板与仿真采集、指标和可视化分析
├── optimization/             # 带适用条件和证据的优化候选
└── learned/                  # 项目完整测试或 profiling 证明的结论
```

项目覆盖层使用相同目录结构，位于 `<project>/.catlass-dsl/knowledge/`。同名 concept
以内置为准，项目层只追加新文件；`index.md` 与 `log.md` 是保留文件，不要当作事实条目。

## 查询

不要扫描整个知识库。按以下顺序渐进发现：

1. 读 `knowledge/index.md`，确定相关分类；
2. 读目标分类的 `index.md`，按标题、标签和描述筛选候选；
3. 读候选条目的 YAML frontmatter，确认 `type`、`tags`、`status`、`verified`、
   `stale_after` 与来源；
4. 只对实际采用的条目读取全文，记录项目相对路径、适用条件和版本。

检索词先按 `query-vocabulary.yaml` 做大小写、下划线、连字符和别名归一化。结构化过滤
（category、tag、status、operator family）保持严格；文本匹配按标题、算子族、标签、描述、
路径和正文加权。零命中时调整关键词或分类重查，不得仅凭零命中断言没有相关知识。

条目必须包含执行所需的接口、代码模式、约束和验证方法。`operator/` 条目还应包含
“算子算法”“分核策略与基本块切分”“数据路径与存储层级”“流水排布、同步关系与数值精度”
四个主题。离线时使用内置条目，并结合 `verified` 与 `stale_after` 判断可信度和新鲜度。

查询记录写入当前阶段的 `ITERATIONS.md`：查询词、命中的条目路径、采用理由；不要只写
“查过知识库”。

## 入库标准

新增、重写或审核 concept 前，必须先完整阅读并执行
[公共入库标准](references/common-entry-standards.md)，再按目标目录读取对应的专用标准：

- `operator/`：[Operator 入库标准](references/operator-entry-standards.md)
- `optimization/`：[Optimization 入库标准](references/optimization-entry-standards.md)
- `dsl/`：[DSL 入库标准](references/dsl-entry-standards.md)
- `profiler/`：[Profiler 入库标准](references/profiler-entry-standards.md)
- `debug/`：[Debug 入库标准](references/debug-entry-standards.md)
- `learned/`：[Learned 入库标准](references/learned-entry-standards.md)

跨目录创建、修改或审核时，只读一次公共标准，但必须读取所有相关目录的专用标准。只做
查询时无需读取这些 references。

## 校验（人工）

没有脚本校验时，按以下清单人工检查：

- 每个非保留 Markdown concept 有合法 YAML frontmatter，包含 `type`、`title`、
  `description`、`tags`、`status`、`generated`、`verified` 和 `sources`；
- 静态事实通过 `sources[].id` 对应的 Markdown footnote 归因；
- `index.md` 索引子目录时显式链接 `<子目录>/index.md`，不能只链接目录路径；
- `query-vocabulary.yaml` 中不存在重复或冲突别名，也没有未被任何 concept 声明的规范
  算子族；
- 新增文件不覆盖已有 concept；冲突结论以新的条件化 concept 追加。

## Learned 准入

项目实测经验写入 `.catlass-dsl/knowledge/learned/`，每条至少包含：

- 安全的 `operator_family`、`topic`、架构 `c310` 和 CATLASS/CANN 版本；
- shape、dtype、layout、仓库集成条件；
- 假设、实际修改、修改前后正确性和性能；
- 修改后正确性 `passed`；
- 至少一个项目内可访问的 test 或 profiling 普通文件路径；
- 最终 best kernel 的完整 SHA-256；
- 结论标记为 `有效`、`无效` 或 `条件有效`。

只有完整测试或 profiling 直接支持的结论可以入库；证伪假设、静态推断、不可访问证据和
`not_run` 结论不得入库。新证据与旧结论冲突时追加新的条件化 concept，不修改历史 concept。

模板见 [`templates/learned-entry.md`](templates/learned-entry.md)。

## 写入约定（人工）

- 新建文件使用“文件名冲突即失败”的方式：写入前检查目标路径不存在，不使用覆盖写入；
- 批量记录前先列出全部目标路径，确认无冲突后再逐个写入；
- 一个 concept 一个文件，文件名使用安全 ID；
- 写入后更新对应目录的 `index.md`，并在 `log.md`（不存在时创建）增加一条变更记录；
- 证据路径使用项目相对路径，并绑定 kernel SHA-256。

任何路径、仓库身份、证据、frontmatter 或 OKF 版本问题都应记录为 `failed` 并停止写入，
不得把 `not_run` 汇总为 `passed`。
