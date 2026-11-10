# docs-skills-benchmark 家族基准回归

> 目的：以 `common_bench.md` 为基线（golden），固定输入样例 `cann_sim工具指南.md`，
> 在 **skill 修改后** 重跑家族检查管线，验证检查结果是否仍与基线**对齐**；
> 不对齐时把差异明细反馈给用户，由用户判断是"预期改进（更新基线）"还是"非预期回归（修正 skill）"。

## 基准构成

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `cann_sim工具指南.md` | 固定输入样例 | 内置 63 处已知问题（标签、链接、拼写、标点、合规、语义等），**禁止修改**——改它等于换考卷，基线随即失效 |
| `common_bench.md` | 基线（golden） | 当前 skills 对样例的标准检出结果（63 条，问题1~63 块格式）；**只能通过"基线重建"流程更新**（见下） |
| `scripts/bench.js` | 基准工具 | `convert` / `compare` / `selftest` 三个子命令（依赖 node） |
| `runs/` | 运行产物 | 最近一次重跑的 `common_bench.md` 与对齐报告 `align_report.md`（覆盖式写入） |
| `output/` | 聚合产物 | 全量模式检查的最终 JSON（待检文档同级 output/ 约定，由 hub 生成） |

基线引擎分布（共 63 条）：punctuation-check 31、aicheck 16、compliance-check 11、
tag-closed-check / link_validity-check / resource-existence-check / codespell-check / filename-check 各 1。

## 标准执行流程（skill 修改后执行）

执行者为 Agent（在 docs-common-review 会话中按本流程编排），四步：

### 第 1 步：全量模式重跑

按 [hub 全量模式](../workflows/full-review.md) 对基准样例执行 9 项检查（含 3 个模型项 subagent）：

- 检查对象：`docs-skills-benchmark/cann_sim工具指南.md`（绝对路径）
- 项目根目录：`docs-common-review`（中间产物落其 `output/`，运行前先按门禁模式第 1 步预清理，避免历史遗留污染聚合）
- 基准场景不询问 csv/md 导出、不询问模式（固定全量模式），聚合完成后直接清理中间产物
- 最终产物：`docs-skills-benchmark/output/docs-common-review-result.json`

### 第 2 步：转换为 bench 格式

```bash
node docs-skills-benchmark/scripts/bench.js convert \
  --input=docs-skills-benchmark/output/docs-common-review-result.json \
  --output=docs-skills-benchmark/runs/common_bench.md
```

产出与基线同构的"问题N" Markdown（引擎名使用基线写法，可直接与 `common_bench.md` diff 肉眼比对）。

### 第 3 步：对齐比对

```bash
node docs-skills-benchmark/scripts/bench.js compare \
  --golden=docs-skills-benchmark/common_bench.md \
  --new=docs-skills-benchmark/runs/common_bench.md \
  --report=docs-skills-benchmark/runs/align_report.md
```

退出码：`0`=对齐；`1`=不对齐（存在阻断性差异）；`2`=输入错误。
（`--new` 也可直接指向聚合 JSON，跳过第 2 步；需要更严口径时加 `--strict`。）

### 第 4 步：反馈

- **对齐**（exit 0）：向用户汇报"基线回归通过"，附各引擎检出数。
- **不对齐**（exit 1）：**必须**把 `align_report.md` 的差异明细逐类反馈给用户：
  漏检（基线有、新结果无）、多检（新结果有、基线无）、关键字段变化（严重程度/问题类型/行号/文件）、
  文案差异（默认仅提示）、网络相关差异（link 项需人工确认）。
  禁止替用户直接判定"对齐/可忽略"。

## 对齐判定口径

| 差异类别 | 判定 | 说明 |
| --- | --- | --- |
| 漏检 / 多检 | ❌ 阻断 | 检出集合变化，skill 行为改变的核心信号 |
| 严重程度、问题类型、行号、文件变化 | ❌ 阻断 | 同条问题的关键属性改变 |
| 上下文/描述/建议措辞差异 | ⚠️ 默认不阻断 | LLM 表述波动；`--strict` 升级为阻断；脚本类引擎的措辞变化会附提示 |
| 基线外引擎（如 markdownlint） | ℹ️ 不参与判定 | 基线生成早于该引擎接入；如需纳入，先走基线重建 |
| link-validity-check 差异 | ℹ️ 网络注记 | 外链可访问性随环境变化，反馈时标注"需人工确认" |

匹配算法：先按（引擎 + 行号）分组配对，组内按上下文/描述相似度贪心匹配；
再对剩余项做同引擎跨行兜底（相似度 ≥ 0.5，记为行号漂移）。引擎历史口径差异
（`AICheck`/`aicheck`、`link_validity-check`/`link-validity-check`、`punctuation-check`/
`StaticCheck_punctuation`、`compliance-check`/`StaticCheck_compliance` 等历史写法）
在比较前统一归一，不产生伪差异。

## 基线重建（golden 更新）流程

仅当用户确认差异为**预期改进**（如规则升级、误报修复、检出能力增强）时执行：

1. 人工审阅 `runs/common_bench.md` 与 `align_report.md`，确认每条差异都是预期行为；
2. 用 `runs/common_bench.md` 覆盖 `common_bench.md`，并在提交说明中注明差异原因；
3. 重跑 `node scripts/bench.js selftest` 确认工具与新基线自洽。

禁止在未确认差异性质前静默覆盖基线。

## 工具自检

```bash
node docs-skills-benchmark/scripts/bench.js selftest
```

对工具本身（解析、别名归一、匹配、差异判定、convert 往返）做 7 项回归，`bench.js` 自身改动后必跑。

## 已知噪声源（反馈时向用户说明）

- **LLM 表述波动**：模型项（punctuation/compliance/aicheck）描述措辞逐次不同——已归入非阻断文案差异；
  但漏检/多检/严重程度变化是真实信号，不视为噪声。
- **网络环境**：link-validity-check 的外链结论随网络变化（内网/外网），差异需人工复核；
  无外网环境跑基准时可为 run_check.js 加 `--disable-external`，但结果与含外链结论的基线不可直接对齐。
- **compliance 术语源**：术语/敏感词参考数据每次运行重新拉取（预清理会删除 `output/compliance/temp` 缓存），
  上游数据更新可能引入合规项差异，反馈时注明。
- **subagent 非确定性**：同一 skill 两次全量重跑可能出现少量行号漂移或检出数 ±1 的波动；
  连续两次都不对齐且差异一致，才可判定为 skill 行为变化。
