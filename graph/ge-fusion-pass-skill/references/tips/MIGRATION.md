# tips 迁移映射（MIGRATION）

R1（知识架构与统一开发范式）建立了两个**导航层**文档，把散落在 15 条 tip 里的开发主流程与诊断步骤串成顺序：

- `references/pass-development-paradigm.md` —— 开发主流程导航（输入识别 → 片段边界 → 选型 → 语言 → 注册/开关 → 签名核实 → 实现/加载/验证 → 样例指针）。
- `references/fusion-troubleshooting.md` —— 融合诊断树导航（加载 → 执行 → 匹配 → 守卫 → replacement → InferShape/format/engine → 图与输出）。

**这两个文档只做导航，不复制规范性事实。** 每条 tip 仍是该条事实的**唯一权威定义源**，正文未改动；本表登记每条 tip 的导航落点，供"旧 tip 跳转文件保留周期"决策时查阅。

## 迁移映射

| tip（权威定义源） | 性质 | 主导航落点 | 仍被 skill 直接引用处 |
|---|---|---|---|
| `dump-first-op-type.md` | 核实 / 诊断 | `pass-development-paradigm.md` §1、`fusion-troubleshooting.md` §3 | 阶段一（目标片段）、阶段二（实现规则）、阶段三（诊断） |
| `api-signature-gate.md` | 开发门禁（G2 字段定义源） | `pass-development-paradigm.md` §6 | 阶段二（门禁 G2） |
| `cpp-style-naming.md` | 命名 / 日志风格 | `pass-development-paradigm.md` §7 | 阶段二（实现规则）、example-map §二 |
| `ge-fusion-namespace.md` | API 命名空间 | `interface-catalog.md` §二/§三（导航指向 tip） | 阶段二（实现规则）、interface-catalog §二 |
| `infershape-util-only.md` | API 知识 | `interface-catalog.md` §二（导航指向 tip） | requirements-analysis-template §6、阶段二（实现规则）、阶段三（诊断） |
| `format-sensitive-nchw.md` | 诊断（format） | `fusion-troubleshooting.md` §5/§6 | 阶段二（实现规则）、阶段三（诊断）、interface-catalog §三 |
| `compliant-node-builder-ir-order.md` | API / 诊断（IR 顺序） | `fusion-troubleshooting.md` §5、`interface-catalog.md` §三 | 阶段二（实现规则）、阶段三（诊断）、fragment-spec §四 |
| `es-all-no-version-rename.md` | 决策树（唯一） | `pass-development-paradigm.md` §7 | 阶段二（实现规则）、阶段三（诊断） |
| `kernel-registration-mismatch.md` | 诊断背景（不构成预换许可） | `fusion-troubleshooting.md` §6 | 阶段二（实现规则）、阶段三（诊断）、ge-repo-map §7.1 |
| `python-remove-node-lifecycle.md` | API 知识 | `interface-catalog.md` §二（导航指向 tip） | 阶段二（实现规则） |
| `dump-log-diff-checklist.md` | 验证清单 | `fusion-troubleshooting.md` §7 | 阶段三（验证顺序与证据、诊断） |
| `stale-pass-artifact-cleanup.md` | 验证（加载基线） | `fusion-troubleshooting.md` §1/§7 | 阶段三（验证顺序与证据） |
| `failure-attribution.md` | 归因纪律（诊断树根） | `fusion-troubleshooting.md`（开篇纪律） | 阶段三（诊断）、gen_es_api.sh |
| `subagent-search-path-scope.md` | 检索纪律 | `knowledge-base.md` §四（导航指向 tip） | knowledge-base §四 |
| `tf1-graph-mode-compat.md` | 输入脚本兼容 | `pass-development-paradigm.md` §1 | 阶段二（实现规则） |

## 维护规则

- **新增 tip** 时（由统一 skill 的阶段四显式触发）：在本表补一行，登记其导航落点；落点若不进现有两份导航文档，说明理由。
- **删除 tip** 前（旧 tip 保留周期决策落地后）：先确认本表「仍被 skill 直接引用处」列已全部改指向导航文档或其它 tip，再删；删除后从本表移除该行，并更新宿主仓库的相关索引或结构校验。
- 本表自身不是事实源，不记录 tip 正文内容；要查某条事实，读对应 tip。
