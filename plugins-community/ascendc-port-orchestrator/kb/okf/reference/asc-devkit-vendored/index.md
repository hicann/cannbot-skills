# asc-devkit-vendored

从 `gitcode.com/cann/asc-devkit` 搬运的上游 API 文档副本，511 张卡。

* [api](api/index.md) — 逐 API 参考卡，23 个功能族，490 张
* [guide](guide/index.md) — 长文指南（编程模型 / API 概述 / 兼容性迁移），21 张

---

## 溯源：为什么叫 `-vendored`，以及 `resource` 钉的是哪个 commit

本插件**不是**直接从 asc-devkit 摄入的。链条是两跳：

```
cann/asc-devkit （某个未确定的修订）
      ↓  PR #103 `Ascend/agent-skills` 的 ascendc-operator-A5-migration skill
      ↓  2026-05-16 导入本插件（P113）
kb/okf/reference/asc-devkit-vendored/
```

### 确切的导入源 commit 未能定位（已做的查找，如实记录）

| 尝试 | 结果 |
|---|---|
| `Ascend/agent-skills` PR #103 记录的 `66637919` | 该仓 920 个 ref 中无以此开头者；`git fetch --depth 1 <sha>` → `couldn't find remote ref`（PR 分支已删） |
| `cann/asc-devkit` HEAD | `docs/api/context/` 已不存在（文档树改版为 `docs/zh/api/`），全部路径失效 |
| 四个版本**分支**（9.0.0 / 9.0.0-beta.1 / 8.5.0 / 8.5.0-beta.1） | 最好的 9.0.0：515 张中 389 条路径存在、仅 19 张逐字节相同 |
| **全部 16 个 tag** | 见下 |

逐 tag 比对路径命中率（515 张有上游声明的卡）：

```
v9.1.0-beta.1   479/515   ← 最佳
v9.0.0-rc.1     389/515      v9.0.1        389/515      v9.0.0-beta.2  388/515
v9.1.0 / v9.1.0.pre / v9.1.0-beta.2 / -beta.2.pre / -beta.3 / v9.1.1 / v9.2.0-*   0/515（已改版）
```

对最佳候选 **`v9.1.0-beta.1` = `792bc49f7ea06312bdb8964d22b8753e6a5cea30`** 做全量内容比对：

```
可比对 454 张   逐字节相同 109   有差异 345（median 2 行 / p90 14 / max 144；≤4 行的 250/345）
```

**结论（不要读成比这更强）**：`v9.1.0-beta.1` 是**已知最接近**的上游修订，
**但未被证实为导入源** —— 109/454 逐字节相同、其余差异与 PR #103 已知的搬运变换
（剥死链、删图、去行尾空白）形态一致，但这既不能证明是它、也不能排除是它相邻的某个未打 tag 的提交。

### 因此本 bundle 不叫 `asc-devkit`

`knowledge_lint.py:81` 的 `GIT_BUNDLES = {"asc-devkit", "ops"}` 要求该 bundle 每张卡的
`resource` 是 GitCode 40 位 sha 永久链接，且 bundle 根 index.md 带 `upstream_commit` pin。
把一个**未证实**的 sha 填进 511 张卡的 `resource` 并以 pin 形式声明，等于把「最接近」
写成「就是」。命名为 `asc-devkit-vendored` 使本 bundle 落在所有规则表之外，
`resource` 只需 `^https?://`。

**代价**：失去 `retrieval/relevance.py:131-133` 对 `path.startswith("asc-devkit/api/")` 的
**+0.08**，且仅在检索意图为 `api_usage` 时生效。

### `resource` 当前指向

各卡 `resource` 已从原先的 `blob/master/`（**移动引用，且 master 上该路径已不存在，链接是死的**）
改为钉住 `792bc49f7ea06312bdb8964d22b8753e6a5cea30`。实测该 commit 下：

```
511 条 resource 中，路径存在的 479，不存在的 32
```

那 32 条（集中在 `api/cube_datamove/asc_set_gm2l1_loop*`、`api/tensor_layout/*LayoutFormat` 等）
是本插件快照里有、而该 tag 上没有的文件，与「快照取自两个 release 之间的某个提交」一致。
它们的 `resource` 保留所记录的原始路径，**读者需知这 32 条在该 pin 上打不开**。
