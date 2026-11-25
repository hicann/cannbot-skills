# docs-skills-benchmark 家族基准回归

> 目的：以人工维护的 `common_bench.md` 为测试集基线、`perf_bench.md` 为耗时基线，固定输入样例
> `bench测试数据集.md`，在 **skill 修改后** 重跑家族检查管线，验证两件事：
> ① 检查结果是否仍与测试集基线**对齐**；② 各检查项耗时相对耗时基线的变化（提示级）。
> 不对齐时把差异明细反馈给用户，由用户判断是"预期改进（按保护机制更新基线）"还是"非预期回归（修正 skill）"。

## 基准构成

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `bench测试数据集.md` | 固定输入样例 | 内置已知问题（标签、链接、拼写、标点、合规、语义等），**禁止修改**——改它等于换考卷，基线随即失效 |
| `common_bench.md` | 测试集基线（人工维护） | 79 条人工维护的标准问题集（缺陷目录口径：一缺陷一条；引擎对同一缺陷的重复报告由 compare 吸收通道收敛）；更新受"测试集保护机制"约束：**只允许修改与新增，删除须逐条证据并通过 `bench.js protect` 校验**（见下） |
| `perf_bench.md` | 耗时基线（perf golden） | 各检查项标准耗时（逐项毫秒）；**2026-11-17 已按建档流程完成首建档**（基准刷新轮产物，9 项计时、无失败检查项；此前的预置参考值已退役）；头部若含"参考性质"标志行会被 `perf compare` 拒绝；**仅同环境可比** |
| `scripts/bench.js` | 基准工具 | `convert` / `compare` / `selftest` + `perf convert` / `perf compare` + `protect`（测试集保护校验）/ `format-check`（格式规范校验），均依赖 node；`compare` 在 `--new` 传聚合 JSON 时附带执行完备性硬校验 |
| `runs/` | 运行产物 | 最近一次重跑的 `common_bench_test.md`、对齐报告 `align_report.md`、耗时快照 `perf_bench_test.md`、耗时对比报告 `perf_report.md` 与保护校验报告 `protect_report.md`（覆盖式写入） |
| `output/` | 聚合产物 | 全量模式检查的最终 JSON（待检文档同级 output/ 约定，由 hub 生成） |

> 测试集基线引擎不包括markdownlint，该能力源于[markdownlint项目](https://github.com/DavidAnson/markdownlint/blob/main/README.md)，本项目暂不做变更。

## 标准执行流程（skill 修改后执行）

执行者为 Agent（在 docs-common-review 会话中按本流程编排），五步：

### 第 1 步：全量模式重跑

按 [hub 全量模式](../workflows/full-review.md) 对基准样例执行 9 项检查（含 3 个模型项 subagent）：

- 检查对象：`docs-skills-benchmark/bench测试数据集.md`（绝对路径）
- 项目根目录：`docs-common-review`（中间产物落其 `output/`，运行前先按门禁模式第 1 步预清理，避免历史遗留污染聚合）
- 基准场景不询问 csv/md 导出、不询问模式（固定全量模式），聚合完成后直接清理中间产物
- 最终产物：`docs-skills-benchmark/output/docs-common-review-result.json`
- **预热（仅当本次目的含耗时对比时）**：首跑含 cspell 词典安装、术语源拉取、外链首访等一次性成本，
  会污染计时——先完整跑一次并**丢弃其结果**，取第二次运行的产物做耗时对比；仅做检出对齐时无需预热。

### 第 2 步：转换为 bench 格式

```bash
node docs-skills-benchmark/scripts/bench.js convert \
  --input=docs-skills-benchmark/output/docs-common-review-result.json \
  --output=docs-skills-benchmark/runs/common_bench_test.md
```

产出与基线同构的"问题N" Markdown（引擎名使用基线写法，可直接与 `common_bench.md` diff 肉眼比对）。

### 第 3 步：对齐比对（含执行完备性硬校验）

```bash
node docs-skills-benchmark/scripts/bench.js compare \
  --golden=docs-skills-benchmark/common_bench.md \
  --new=docs-skills-benchmark/output/docs-common-review-result.json \
  --report=docs-skills-benchmark/runs/align_report.md
```

退出码：`0`=对齐；`1`=不对齐（存在阻断性差异，**或**存在未成功执行的检查项）；`2`=输入错误。
`--new` 也可指向第 2 步的 `runs/common_bench_test.md`，但**执行完备性硬校验仅在 `--new` 直接传聚合 JSON 时
生效**（失败清单只存在于 JSON 的 `summary.failed_check` 中），标准流程建议按上例传 JSON。
需要更严口径时加 `--strict`。

**执行完备性硬校验**：`--new` 为聚合 JSON 且 `summary.failed_check` 非空时，`compare` 强制判不对齐
（exit 1），对齐报告首部逐项列出失败检查项与原因——存在失败检查项时漏检数与问题总数不代表全量结论，
须先修复检查项执行后重跑基准，再进行对齐判定。

### 第 4 步：耗时对比（提示级）

```bash
node docs-skills-benchmark/scripts/bench.js perf convert \
  --input=docs-skills-benchmark/output/docs-common-review-result.json \
  --output=docs-skills-benchmark/runs/perf_bench_test.md
node docs-skills-benchmark/scripts/bench.js perf compare \
  --golden=docs-skills-benchmark/perf_bench.md \
  --new=docs-skills-benchmark/runs/perf_bench_test.md \
  --report=docs-skills-benchmark/runs/perf_report.md
```

退出码：`0`=已生成耗时对比报告（**提示级，不代表基线对齐结论**）；`2`=输入错误
（如耗时基线 `perf_bench.md` 尚未建立——首次使用先按下方"耗时基线建档"建立，再进入本步）。
`--new` 也可直接传聚合 JSON 跳过 `perf convert`；快照文件用于留档，建议保留。

### 第 5 步：反馈

- **对齐结论**（compare）：对齐（exit 0）时汇报"基线回归通过"，附各引擎检出数。
- **不对齐**（exit 1）时，**必须**逐类反馈：
  - 执行完备性：失败检查项逐项（名称 + 原因），本次基准不可信；
  - 检出差异：漏检（基线有、新结果无）、多检（新结果有、基线无）、关键字段变化（严重程度/问题类型/行号/文件）、
    文案差异（默认仅提示）、网络相关差异（link 项需人工确认）。
- **耗时对比**（perf）：逐项耗时对比表、显著回归/显著改善标记、本次无计时与基线未收录项——
  **提示级**，不改变基准回归通过判定，禁止仅凭耗时差异判基准不通过。
- 禁止替用户直接判定"对齐/可忽略"。

## 对齐判定口径

| 差异类别 | 判定 | 说明 |
| --- | --- | --- |
| 漏检 / 多检 | ❌ 阻断 | 检出集合变化，skill 行为改变的核心信号 |
| 严重程度、问题类型、行号、文件变化 | ❌ 阻断 | 同条问题的关键属性改变 |
| 执行完备性（failed_check 非空） | ❌ 阻断 | 存在未成功执行的检查项，本次结果不完整、漏检数不可信 |
| 上下文/描述/建议措辞差异 | ⚠️ 默认不阻断 | LLM 表述波动；`--strict` 升级为阻断；脚本类引擎的措辞变化会附提示 |
| 基线外引擎（如 markdownlint） | ℹ️ 不参与判定 | 现行基线未收录该引擎；如需纳入，走"基线更新流程"以新增合入 |
| link-validity-check 差异 | ℹ️ 网络注记 | 外链可访问性随环境变化，反馈时标注"需人工确认" |
| 重复检出吸收 | ℹ️ 不阻断 | 同一缺陷的重复报告并入已配对基线，不计多检（同一缺陷一票制），报告单列"吸收"节供人工复核 |

### 匹配算法（6 维度加权评分，多阶段配对）

引擎为前置门槛（同引擎才可比）；同引擎两两评分，满分 9：

| 维度 | 分值 | 匹配规则 |
| --- | --- | --- |
| 问题类型 | 1 | 完全相同得 1（硬门：不同即不匹配） |
| 严重程度 | 1 | 完全相同得 1（硬门：不同即不匹配） |
| 位置行号 | 0～2 | 相同得 2；差值 ≤2 得 1；差值 ≤5 得 0.5；否则 0（区间行号取起始行） |
| 问题上下文 | 0～2 | 归一化后相同得 2；包含关系得 1.5；bigram 重合 ≥3 得 1、≥1 得 0.5 |
| 问题描述 | 0～2 | bigram 重合 ≥5 得 2；≥3 得 1.5；≥1 得 0.5 |
| 修改建议 | 0～1 | bigram 重合 ≥3 得 1；≥1 得 0.5 |

中文关键词提取采用 2 字 n-gram（bigram，另含 ≥2 字符的英文/数字 token，小写归一），避免整句匹配失效。

配对阶段（按序，前序已配对的不再参与）：

1. **P1 主匹配**：双门通过且总分 ≥6 → 按总分降序一对一贪心配对；
2. **P2 严重程度变化诊断**：仅类型门通过、严重程度不同、其余分值（满分 8）≥5.5 → 配对成功并记为"严重程度"字段变化（阻断）；
3. **P2b 低置信度兜底**：双门通过且总分 ≥5.5、行号分 ≥1、描述分 ≥0.5 → 配对（对齐），报告标注低置信度供人工复核；
4. **P3 重复检出吸收**：仍未配对的新结果，若对某条**已配对基线**双门通过、总分 ≥5.5、行号分 ≥0.5、描述分 ≥1.5 → 并入该基线（吸收，不阻断、不计多检）；
5. 剩余：基线侧 → 漏检；新结果侧 → 多检；漏检 ↔ 多检间按相似度 ≥0.5 给出"疑似迁移"提示（不阻断）。

引擎历史口径差异（`AICheck`/`aicheck`、`link_validity-check`/`link-validity-check`、
`punctuation-check`/`StaticCheck_punctuation`、`compliance-check`/`StaticCheck_compliance`
等历史写法）在比较前统一归一，不产生伪差异。

## 耗时对比口径（perf，提示级）

| 差异类别 | 判定 | 说明 |
| --- | --- | --- |
| 脚本项显著回归 | ❌ 提示标记 | Δ>+50% **且** Δ>+10s 同时满足才标记（脚本项确定性强，抖动主要来自环境） |
| 模型项显著回归 | ❌ 提示标记 | Δ>+100% **且** Δ>+60s 才标记（模型项为 subagent 墙钟：LLM 出 token 速度 + API 延迟，天然波动大故放宽） |
| 显著改善 | ✅ 提示标记 | 与显著回归对称的阈值 |
| 本次无计时（基线有） | ⚠️ 提示 | 该检查项本次未执行成功或被跳过，结合执行完备性确认 |
| 基线未收录（本次有） | ℹ️ 提示 | 新增计时项；如需纳入，走"耗时基线建档/重建" |

三条硬约束：

1. **提示级不阻断**：耗时差异只呈现在 `perf_report.md`，不改变 `compare` 的对齐结论与退出码；
   仅当同环境连续多次出现"显著回归"且差异可复现时，才怀疑 skill 性能劣化。
2. **仅同环境可比**：耗时基线绑定机器、CPU 负载、网络与缓存状态；换环境（含换机器、CI 与本地互比）
   后必须先重建耗时基线，旧基线数字不迁移。
3. **预热后取第二跑**：首跑的一次性成本（cspell 词典安装、术语源拉取、外链首访）
   不计入标准耗时。

计时口径：逐项耗时取自聚合 JSON 的 `summary.check_timing_ms`（hub 执行原则 8：脚本项为命令计时、
模型项为 subagent 墙钟）；多目标计时键（`filename-check-1/2`）在提取时折叠求和归一到 `filename-check`。

## 测试集保护机制（基线版本演进守则）

适用对象：测试集基线 `common_bench.md` 及任何 bench 同构问题集文件。基线更新遵循：

- **只允许"修改"与"新增"**：修正条目字段、合入预期改进、扩充问题集均为允许操作，无须经由。
- **不允许删除前一版本已有条目**：确需删除（确认误报、重复条目、数据集变更等）时，必须逐条提供
  ① 问题定位（检查引擎 + 位置行号）② 删除原因 ③ **必要证据**——误报须给出源码/规则依据，
  重复须给出重复对象条目，数据集变更须给出 diff。
- **机械校验**（更新必过）：

```bash
node docs-skills-benchmark/scripts/bench.js protect \
  --old=docs-skills-benchmark/common_bench.md \
  --new=<更新候选文件> \
  [--reasons=<删除理由.json>] [--report=docs-skills-benchmark/runs/protect_report.md]
```

  退出码 `1`（存在未证据化删除）时**禁止以候选覆盖基线**，逐条补齐证据后重跑；
  `0`（无删除或全部删除已证据化）方可进入覆盖步骤。

删除理由文件为 JSON 数组，每项四要素齐全才有效（缺任一项告警并忽略）：

```json
[
  {
    "engine": "compliance-check",
    "line": 287,
    "reason": "确认误报：该场景规则不适用",
    "evidence": "tag_checker.js 无此判定分支，复跑实测 0 检出"
  }
]
```

基线沿革：2026-11-12 基线由"87 条管线采集版"切换为"63 条人工维护版"。切换依据（对比分析结论）：
采集版含疑似误报簇（代码块符号配对 18 条）、同问题双侧复报 4 组，且缺失 26 条必选项；
人工维护版条目已对数据集逐条核验。切换由维护者决策执行。
2026-11-17 基线由 63 条更新为 75 条（经 protect 校验，报告见 runs/protect_report.md）：收编 13 条
（8 条同行一拆二/多报、5 条同类跨位置复现，字段逐字取自引擎真实输出），证据化删除 1 条
（原问题51 AICheck L218"为为"：位于 ```text 围栏代码块内，与中文错别字检查及重复字脚本的代码块
排除范围冲突，属基线误收）。
2026-11-17 基线由 75 条修剪为 72 条（缺陷目录口径，经 protect 校验）：删除 3 条同缺陷重复收编条目
（L289 结束标记 id1 重复于 L287~289 标签对条目——287 行即 289 行的开始标记；L293 无效NPU值与
L293 910B 不在支持范围为同一标签的多报；L298 ID重复与 L296 id3 重复为同一缺陷）。引擎对同一缺陷
的重复报告改由 compare 的吸收通道（P3）自动并入，不再单独占基线条目。同日 compare 匹配算法升级为
6 维度加权评分（类型/严重程度硬门 + 行号/上下文/描述/建议评分，命中线 ≥6），含严重程度诊断、
低置信度兜底与重复吸收三通道，自检 16 项全过。
2026-11-17 恢复 AICheck L218"为为"条目（v1 问题51，经 protect 校验：修改 0 处、新增 1 条）：
当日早前的证据化删除判定有误——中文错别字检查排除条款为"代码块、代码段、命令行中的代码内容
（**不含注释**）"，注释文字不在排除范围，该重复字位于 ```text 块内的注释行属可检出内容；经维护者
复核恢复，基线 72 → 73 条。
2026-11-17 基准刷新轮（全量重跑）：维护者将 v1 备份重置回基线位（63 条起点），9 项检查全量重跑
（aicheck 按执行原则 6 降级路径由主会话执行，代码块注释行纳入检查范围后 L218"为为"检出对齐），
按 protect 流程从本轮新鲜产物采纳 16 条：原收编 10 条以本轮引擎最新措辞重新提取（punct L9/L29/L94/L148、
compliance L13/L39/L43/L215×2、tag L64），加 aicheck 连续两轮稳定新检出 6 条（L9 CANN/L29 MC2/L29 HCCL/
L110 注释标点/L116 aclnn/L176 表格标点；aicheck L30"标点误用"与基线 punctuation 条目为同一缺陷跨引擎，
不采纳），63 → 79 条（修改 0、删除 0）。同日耗时基线完成首建档（覆盖预置参考值；脚本项为本轮实测，
punct/compliance 为 subagent 墙钟兜底、aicheck 为主会话墙钟）。

## 基线更新流程（common_bench.md，受保护机制约束）

仅当用户确认差异为**预期改进**（如规则升级、误报修复、检出能力增强）时执行：

1. 按"标准执行流程"完成重跑与对齐比对，人工审阅 `runs/common_bench_test.md` 与 `align_report.md`，
   确认每条差异的预期性质；
2. 将预期差异合入更新候选（新文件）：修改对应条目或新增条目；候选中相对前版被删除的条目，
   逐条准备删除理由文件（`engine` + `line` + `reason` + `evidence`）；
3. 保护校验：执行上方 `protect` 命令，exit 1 时禁止覆盖，补齐证据后重跑直至通过；
4. 通过后以候选覆盖 `common_bench.md`，提交说明注明合入内容与删除证据；
5. 格式校验：`node scripts/bench.js format-check --input=common_bench.md --strict-ctx` 须通过；
6. 重跑 `node scripts/bench.js selftest` 确认工具与新基线自洽。

禁止在对齐比对与保护校验完成前静默覆盖基线。测试集基线更新不必同步耗时基线，两者独立维护。

## 问题集格式规范（format-check）

`common_bench.md` 及任何 bench 同构问题集须满足：

- 问题编号从 1 起连续递增（`## 问题N`）；
- 每块 8 字段**齐全、按序且非空**：检查引擎、问题类型、文件路径、位置行号、问题上下文、问题描述、修改建议、严重程度；
- 检查引擎在合法白名单（tag-closed-check / link-validity-check / resource-existence-check /
  codespell-check / filename-check / punctuation-check / compliance-check / aicheck / markdownlint，
  历史别名比较前自动归一）；
- 位置行号为非负整数，或形如 `287~289` 的**闭区间**（起始 ≤ 结束；用于跨行问题锚定，比对时取起始行参与匹配）；
- 问题类型 ∈ {规范性 / 正确性 / 可读性 / 完备性}；严重程度 ∈ {严重 / 一般 / 提示}；
- 本基线的本地约定：问题推荐按检查引擎集中分类排列（引擎分组顺序与组内位置可按需调整，调整后须与基线头部「问题分布」表保持一致）；问题上下文用反引号包裹
  （上下文自身含反引号时用双反引号定界），此包裹约定由 `--strict-ctx` 校验。

```bash
node docs-skills-benchmark/scripts/bench.js format-check --input=docs-skills-benchmark/common_bench.md --strict-ctx
```

退出码 `0`=合规 / `1`=存在违规 / `2`=输入错误。

## 耗时基线建档 / 重建（perf_bench.md）

仅当环境未变且需要建立或刷新标准耗时时执行（2026-11-17 已完成首建档；换环境或 skill 性能
优化后按以下流程重建，重建前可先看现行基线预判各检查项量级：秒级脚本项 / 分钟级模型项）：

1. 确认本次运行环境即后续对比的固定环境（机器、负载、网络）；
2. 按第 1 步预热一次全量模式并丢弃结果，再完整跑第二次；
3. 对第二次产物执行 `perf convert` 生成 `runs/perf_bench_test.md`，确认计时无一次性成本混入、
   无失败检查项（快照头部"执行完备性"应为"无失败检查项"）；
4. 用其覆盖 `perf_bench.md`，提交说明注明重建原因（如"性能优化后刷新标准耗时"或"更换运行环境"）；
5. 重跑 `node scripts/bench.js selftest` 确认工具自洽。

基线头部含"参考性质"标志行时 `perf compare` 退出码 2 并提示先建档（首建档已解除）；
按上述流程建档/重建覆盖后即可正常作为 golden 对比。
注：建档时 `perf_bench.md` 与本轮 `runs/perf_bench_test.md` 完全一致属预期（基线即由建档轮快照
覆盖而来）；两者自**下一次重跑**起开始分化——runs 快照随每轮覆盖刷新，基线保持固定，
`perf compare` 展示逐项差值与显著回归/改善标记。
禁止在计时不可信（含失败检查项、未预热）时建档。

## 工具自检

```bash
node docs-skills-benchmark/scripts/bench.js selftest
```

对工具本身（解析、别名归一、匹配、差异判定、convert 往返、perf 提取与快照往返、耗时对比判定、
执行完备性硬校验、测试集保护判定、格式规范校验）做回归，`bench.js` 自身改动后必跑。

## 已知噪声源（反馈时向用户说明）

- **LLM 表述波动**：模型项（punctuation/compliance/aicheck）描述措辞逐次不同——已归入非阻断文案差异；
  但漏检/多检/严重程度变化是真实信号，不视为噪声。
- **网络环境**：link-validity-check 的外链结论随网络变化（内网/外网），差异需人工复核；
  无外网环境跑基准时可为 run_check.js 加 `--disable-external`，但结果与含外链结论的基线不可直接对齐。
- **compliance 术语源**：术语/敏感词参考数据每次运行重新拉取（预清理会删除 `output/compliance/temp` 缓存），
  上游数据更新可能引入合规项差异，反馈时注明。
- **subagent 非确定性**：同一 skill 两次全量重跑可能出现少量行号漂移或检出数 ±1 的波动；
  连续两次都不对齐且差异一致，才可判定为 skill 行为变化。
- **耗时噪声**：模型项墙钟随 LLM 出 token 速度与 API 延迟波动（宽阈值缓解）；脚本项受 CPU 负载与
  缓存冷热影响（同环境 + 预热缓解）；换机器/换环境的耗时不可直接对比（耗时基线不迁移）。
