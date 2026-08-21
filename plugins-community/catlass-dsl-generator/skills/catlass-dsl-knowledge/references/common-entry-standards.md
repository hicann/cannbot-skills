# CATLASS DSL 知识公共入库标准

创建、修改、迁移或审核 concept 时，先执行本文，再执行目标目录的专用标准。

## 证据能力与目录选择

| 目录 | 只记录什么 | 不得声称什么 |
| --- | --- | --- |
| `knowledge/dsl/` | 固定提交中的 DSL 接口、语义和可执行模式 | API 一定更快 |
| `knowledge/operator/` | 固定提交中的算法、分核、数据路径、同步和优化地图 | 候选已提速 |
| `knowledge/debug/` | 可复现症状、诊断路径、根因证据和修复验证 | 未复现问题的唯一根因 |
| `knowledge/profiler/` | 可复现采集方法、字段含义和归因方法 | 单个 ratio 已证明瓶颈 |
| `knowledge/optimization/` | 带前提、代价和门禁的可复用优化候选 | 未实测的性能收益 |
| `knowledge/learned/` | 项目证据直接支持的条件化结论 | 超出 workload/版本的规律 |

跨证据等级必须拆分：固定源码机制进入静态 concept，项目正确性和性能结果进入 learned。

## OKF 与来源门禁

- frontmatter 必含 `type/title/description/tags/status/generated/verified/sources`；相关条目补
  `operator_families` 和 `arch`。
- 静态 concept 只引用固定 commit 的实现、文档、测试或官方工具说明；每个 source id 在相邻事实
  处使用同名脚注。实现、文档、测试冲突时记录漂移，以实现和可执行测试界定行为。
- `learned` 只接受 `project-evidence:`，并绑定最终 kernel 的完整 SHA-256。
- 来源不能直接证明的内容标为未知或待验证假设，不从命名、经验或相邻算子补全。

## 静态 concept 的公共正文

静态 concept 保留 `# 接口与概念`、`# 用法`、`# 代码模式`、`# 约束`、`# 失败表现`、
`# 验证方法`。正文必须足以离线判断适用条件、执行代码/命令、识别失败并完成验证。

## 事实、推断和实测分层

- **源码事实**：固定提交直接可见，并紧邻来源脚注。
- **静态推断**：写成可证伪的“候选/首查”，包含条件和预期观测。
- **工具观测**：绑定工具版本、设备、命令、采集边界和原始 artifact。
- **项目结论**：仅由同配置完整正确性和 benchmark/profile 支持，进入 learned。

编译成功、局部 smoke、理论量、fast 名称、单次快值或单个 ratio 均不能证明收益。

## 索引与演进

- 新 concept 加入所属 `index.md`；父索引显式链接 `<子目录>/index.md`。
- 算子实现名需要跨 concept 复用时，在根 `query-vocabulary.yaml` 追加到唯一规范算子族；
  不把未经确认的名称相似性写成别名。
- 静态 concept 更新时同步来源、`verified` 和漂移说明；learned 只追加，不覆盖历史。
- 新鲜度敏感内容使用 `stale_after`。

## 统一入库工作流

1. 选目录并固定来源版本，区分事实、假设、观测和项目结论。
2. 按专用标准写最小完整正文，为建议补条件、代价、回退和验证。
3. 更新索引并运行 `validate` 与相关测试；learned 仅通过 `record` 批量追加。
4. 删除来源不支持、不能执行或不影响决策的内容。

## 统一验收问题

- 关键事实是否有直接来源，假设是否可证伪？
- 读者能否离线执行并判断适用/失败/回退？
- 结论是否越出 workload、版本、layout、dtype 或架构？
- 是否保留了与使用、诊断、优化或验证无关的内容？
