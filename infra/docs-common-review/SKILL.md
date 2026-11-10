---
name: docs-common-review
description: |
  通用文档质检统一入口（hub）：按 门禁 / 全量 / 自定义 三种模式编排 9 项检查能力（markdownlint、HTML 标签闭合、链接有效性、资源存在性、单词拼写、文件命名、标点符号、合规、AI 语义检查），统计各检查项执行耗时，聚合为统一 JSON 质检报告（级别统计 + 门禁判定 + 耗时统计），产物落在待检文档同级 output/ 目录，互不依赖的检查（脚本项与模型项）在同一条消息中并发执行，汇报后按用户选择导出 csv/md。当用户提到以下任一场景时必须使用本技能：通用质检 / 综合质检 / 全面检查 Markdown 文档质量、文档质检报告、门禁预检 / CI 门禁项本地检查 / 提交前自检、全量模式 / 门禁模式 / 自定义模式检查文档、多个检查项组合检查同一文档、家族基准回归 / benchmark 对齐检查 / 基线对齐（docs-skills-benchmark，见"基准回归"节）。用户仅点名单项检查（如"只查链接"、"只查标点"）时不走本入口，直达对应原子检查技能。
version: 1.0.0
metadata:
  author: Huawei Technologies Co., Ltd.
  category: utility-doc
---

# 通用文档质检统一入口（hub）

本 skill 是 docs-common-review 家族的编排入口：自身不实现任何检查，职责限定为 模式路由 → 调度原子检查能力 → 聚合统一报告。各检查能力的规则真源在原子 skill / 脚本处（知识依赖单向：本 skill 只消费，不复制、不修改它们的规则定义）。

## 质检能力项（canonical 命名）

| 检查项 | 功能说明 | 执行方式 | 执行入口（相对本 skill 根目录） |
| --- | --- | --- | --- |
| markdownlint | Markdown 基础语法检查 | 脚本（node，一条命令可组合多项） | docs-tool-based-check/scripts/run_check.js |
| tag-closed-check | HTML 标签闭合检查 | 同上 | 同上 |
| link-validity-check | 链接可访问性检查 | 同上 | 同上 |
| resource-existence-check | 资源存在性检查 | 同上 | 同上 |
| codespell-check | 单词拼写检查 | 同上 | 同上 |
| filename-check | Markdown 文件命名规范检查 | 脚本（python，黑盒直调） | docs-filename-check/docs-filename-check.py |
| punctuation-check | 标点符号检查（空格缺失/冗余、中英混排等） | 原子 skill（模型+脚本） | docs-punctuation-check/SKILL.md |
| compliance-check | 合规性检查（术语、敏感词、标签、编码、table、目录层级） | 原子 skill（模型+脚本） | docs-compliance-check/SKILL.md |
| aicheck | 基于上下文语义的 AI 检查（一致性、冗余、可读性、结构完整性） | 原子 skill（模型） | docs-basic-aicheck/SKILL.md |

前 5 项通过 `run_check.js` 的 `--checks` 参数执行；为获得单项耗时与独立结果文件，各检查项独立调用一次（每项结果文件 `tool-based-<检查项>-result.json`）。该脚本另支持 toc-check、file-naming-check 等，本家族不暴露，视为非法输入。

### 别名映射（用户口语 / 历史写法 → canonical）

| 用户写法 | canonical |
| --- | --- |
| markdownlint、md lint、语法检查 | markdownlint |
| tag_closed、tag_closed_check、标签闭合 | tag-closed-check |
| link_check、链接检查、死链检查 | link-validity-check |
| resource_check、资源检查 | resource-existence-check |
| codespell、拼写检查 | codespell-check |
| filename、文件命名检查 | filename-check |
| punctuation、标点检查、空格检查 | punctuation-check |
| compliance、合规检查、术语检查 | compliance-check |
| AICheck、AI 检查、语义检查 | aicheck |

命中别名按映射执行；未命中任何 canonical / 别名的名称视为非法输入：停止执行并向用户列出 9 项合法名，禁止模糊猜测执行。

## 质检维度 × 检查项映射

| 维度 | 覆盖检查项 |
| --- | --- |
| 写作规范性 | markdownlint、punctuation-check、filename-check、compliance-check（标签/编码/table/目录层级） |
| 正确性 | tag-closed-check、link-validity-check、resource-existence-check、codespell-check、compliance-check（术语/敏感词）、aicheck |
| 完备性 | aicheck（结构完整性） |
| 可读性 | aicheck（一致性/冗余/可读性） |

内容正确性与完备性/可读性目前主要由 aicheck 的语义规则承载；所有检查当前与文档类型（API/指南/工具类）无关，类型化差异为后续路线。

## 三种质检模式

| 模式 | 覆盖检查项 | 特点 |
| --- | --- | --- |
| 门禁模式（ci-gated） | markdownlint、codespell-check、tag-closed-check、link-validity-check、resource-existence-check + compliance-check 的两个脚本步骤（芯片/位置标签、MD 编码格式） | 纯脚本管线，脚本项并发执行，秒级完成；报告给出门禁 pass/fail |
| 全量模式（full） | 9 项全部 | 覆盖最全；脚本项与模型项同消息全并发（含 aicheck；唯一例外是 markdownlint 首跑缓存预热先行），耗时与成本最高 |
| 自定义模式（custom） | 用户勾选的任意组合 | 确定性执行勾选项；用户未给清单时必须先弹 9 项多选框（见"模式选择"硬约束） |

门禁集合依据：CANN 社区 docs-ci 实际启用且拦截 PR 的项为 link / resource / tag-closed / compliance（标签+编码）；markdownlint 与 codespell 在 CI 中仅告警不拦截——因此门禁判定只看前四类，后两项按告警单独计数。标点、术语、文件命名不在社区 CI 内，归全量/自定义模式。

## 模式选择

- 用户显式指定模式（"门禁模式 / 全量 / 自定义"）→ 直接路由对应 workflow。
- 未指定模式 → 必须先以可勾选（单选）选项形式询问用户（全量 / 门禁 / 自定义，各附一句话特点说明），禁止默认选某个模式直接执行。
- 仅当用户明确跳过选择时，才按门禁模式执行（耗时最短）并告知用户。
- **自定义模式硬约束（检查项必须先问后执行）**：用户选择自定义模式但未在对话中给出检查项清单时，**必须**先以可勾选（多选）选项形式向用户发出询问并等待回复，勾选项为 9 项 canonical 检查项（各附一句话功能说明，取"质检能力项"表）；**禁止**未经用户勾选反馈就自行确定检查项集合（含全选 9 项）直接执行，**禁止**"先全选执行、再告知可重跑子集"的补救式交互；仅当用户明确跳过选择时，才按全选 9 项执行并告知用户。该约束由 custom-review.md 第 2 步的 `[CHECKPOINT: 检查项确定完成]` 门控（未收到用户清单/勾选/明确跳过前禁止输出、禁止进入执行步骤）。

## 输入

支持四种形态，统一解析后透传给各执行单元：

1. 单个 Markdown 文档路径
2. 多个路径（英文逗号分隔）
3. 一个文档目录（hub 先展开为 `.md` 文件清单再派发；展开时排除任何 `output/` 目录下的文件，避免本技能历史报告被再次检查）
4. 直接粘贴一段 Markdown 内容（先写入 `<项目根>/output/temp/auto_temp.md` 后按文件路径处理）

不自动从 git diff 提取变更文件；用户需要 PR 预检时自行提供变更文件列表。

## 输出（聚合产物）

所有模式最终由 `scripts/aggregate.js` 聚合为一份报告：

- **产物目录（最终报告落点）＝待检查文档同级目录下的 `output/`**：
  - 单文件目标：该文件所在目录（如 `docs/readme.md` → `docs/output/`）
  - 多文件目标：所在目录相同时取该目录；跨目录时取公共父目录
  - 目录目标：该目录自身（`output/` 建在其下）
  - 粘贴内容：`<项目根>/output/`
- 最终产物：`<产物目录>/docs-common-review-result.json`（csv / md 为可选导出，见"通用收尾"）
- **中间产物一律落在 `<项目根>/output/` 运行工作区**（各检查项分项结果文件、耗时记录、temp/）：各原子 skill 的默认输出路径不因本 hub 改变；聚合与导出完成后统一清理，聚合失败时保留现场供排查
- 结构：`{"summary": {...}, "issues": [...]}`；issues 为 11 字段统一 JSON（rule_group、rule_item、rule_type、rule_name、snippet、description、rule_level、file_name、line_number、location、suggestion）
- 聚合层统一规范化（各原子能力输出存在口径差异，全部在此收敛，不改动原子 skill）：
  - `rule_group` 统一改写为"通用规则"
  - `location` 去除 `vscode://file/` 等协议前缀、反斜杠统一为 `/`，格式为 `绝对路径:行号`；无行号的检查项（如 filename-check）仅绝对路径
  - `line_number` 无行号时为空串
- summary 字段：模式、问题总数、按级别计数（严重/一般/提示）、按检查项计数、`check_timing_ms`（各检查项执行耗时，毫秒）、输入文件使用/跳过情况；门禁模式额外含 `gate`（passed、blocking_issues、warning_issues）；**可选字段 `failed_check`（仅当存在未成功执行的检查项时存在）**：数组，每项 `{"check": "<canonical 检查项名>", "reason": "<具体失败原因>"}`，其非空时问题总数不代表全量结论（呈现规范见"检查项失败提示"节）
- 排序固定为 文件名 → 行号 → 检查项 → 描述，同输入产出逐字节一致（评测可复现）

### 通用收尾（所有模式，汇报完成后必须执行）

1. 询问用户是否导出 csv / md（多选：CSV / Markdown / 两者都要 / 不导出；用户明确跳过询问视同不导出）。
2. 选择导出时，以相同 `--inputs / --mode / --timing / --output`（及本次聚合使用过的 `--failed-file`，如有）重跑 aggregate.js 并追加 `--format=<csv / md / csv,md>`，向用户汇报新产物路径；产物与 JSON 同目录同名异扩展名。
3. **清理中间产物（必做，不可跳过）**：删除 `<项目根>/output/` 下本次生成的分项结果文件（tool-based-*-result.json、filename-check-result.json、compliance-gate-result.json、punctuation-markdown-check-result.json、compliance-markdown-check-result.json、general-markdown-check-result.json）、`filename-*/`、`timing/`（计时目录）、`temp/`、`failed_checks.json`（失败清单）、各原子 skill 的中间文件与临时目录（名称集合见各 workflow 第 1 步的预清理命令），以及产物目录中本次未生成的历史 csv/md；**唯一例外**是聚合失败时保留现场供排查。
4. 完成后输出 `[CHECKPOINT: 收尾完成]`（汇报未完成或中间产物未清理时禁止输出）。

## 执行原则

1. **脚本项直调**：所有模式中的脚本项直接调用脚本命令，不经过原子 skill 的交互流程（范围询问等是它们独立使用时的行为）。工具 5 项逐项独立调用 run_check.js、每项独立结果文件；filename-check 逐目标运行后由 `scripts/merge_json.js` 合并为单一结果文件。
2. **模型项派 subagent**：punctuation-check / compliance-check / aicheck 以 subagent 执行——subagent 读取并严格遵循对应原子 SKILL.md 的完整流程；subagent 提示词中必须预先指定检查对象、项目根目录、检查范围（免交互）。
3. **全并发启动**：互不依赖的检查在单条消息中同时发出——工具项各自一条 Bash、filename 各目标一条 Bash、三个模型项（punctuation / compliance / aicheck）各一个 subagent（Agent 派发与 Bash 调用可混在同一条消息）；禁止把互不依赖的检查排成先后步骤串行执行。模型三项之间**无顺序约束**（aicheck 不等 compliance：经源码核实，本家族 spoke 的脚本不产出共享中间文件 `output/tag_check_result.json`——该文件仅社区 CI 版本产生，spoke 的 `tag_checker.js` 直接把结果写入调用方传入的输出路径，无从互删；第 1 步预清理历史中间产物进一步保证并发起点干净）。唯一例外：工具 5 项中 markdownlint 先行单独执行（社区脚本首次运行需下载缓存到固定路径，并发下载同一文件会互踩），其返回后其余 4 项再并发（缓存确认存在时 5 项同时发出）。codespell-check 在本次范围内时，须先完成"环境依赖 → cspell 词典"的检查/安装前置（已装则跳过），再进入上述并发编排。宿主不支持同消息多工具调用时降级为顺序执行并告知用户。
4. **完整性校验与失败隔离**：按模式维护期望结果文件清单，全部执行单元返回后核对落盘情况；缺失则重派一次；仍缺失则按第 5 条计入失败清单并在最终汇报中显著标注，禁止用空结果冒充。**任一检查项异常中断（命令失败、超时、subagent 崩溃、重派后仍失败、环境缺失跳过）只影响该项本身，其余检查项必须正常执行并完成聚合，禁止因单项失败中止整个质检或放弃产出报告**。
5. **失败记录与显著提示（failed_check）**：存在未能成功执行的检查项时：(1) 在聚合前将失败清单写入 `<项目根>/output/failed_checks.json`（JSON 数组，每项 `{"check": "<canonical 检查项名>", "reason": "<具体原因，含错误输出摘要>"}`）；(2) 聚合命令追加 `--failed-file=<项目根>/output/failed_checks.json`，由 aggregate.js 写入 `summary.failed_check`（可选字段，有失败才存在）；(3) 汇总输出时**必须**在汇报最前面输出显著的检查项错误提示块（模板与可能原因映射见"检查项失败提示"节）；(4) `failed_checks.json` 属中间产物，收尾清理。
6. **宿主降级**：运行环境不支持 subagent 时，改为按顺序自行执行各原子 skill，并告知用户耗时将增加。
7. **零编造**：报告只来自各执行单元的真实输出；执行单元报 0 问题时如实呈现；失败检查项如实记录，禁止编造其结果。
8. **计时统计**：每个检查项一个计时文件 `<项目根>/output/timing/<检查项>.txt`（内容一行 `检查项名 毫秒数`；并发执行下禁止多进程共写同一文件，重派时直接覆盖同名文件；filename 多目标时每目标一条，名为 `filename-check-<序号>`）。脚本项在各自 Bash 命令内用 `date +%s%3N` 前后差值写入（毫秒时间戳不可用时以秒级 ×1000 近似）；模型项在收到 subagent 完成通知后写入（优先取通知中的 duration_ms，缺失时用当前时间戳减派发时间戳）。聚合时 `--timing=<项目根>/output/timing`（支持目录/单文件/逗号分隔多个）读取全部计时文件，嵌入 summary.check_timing_ms。

## 检查项失败提示（failed_check 呈现规范）

存在失败检查项（summary.failed_check 非空）时，各 workflow 的汇总输出步骤**必须**把以下提示块置于汇报最前面（禁止藏在正文中间或省略），逐项列全：

```text
❌ ============ 检查项执行失败（<N> 项）============
❌ 失败项 <i>/<N>：<检查项 canonical 名>
     原因：<failed_checks.json 中记录的具体原因>
     可能原因与建议：<按下方映射表给出>
❌ 本次质检结果不完整：以上检查项无检出结果，问题总数不代表全量结论
❌ 失败清单已记入 summary.failed_check（docs-common-review-result.json）
====================================================
```

可能原因映射表（按失败检查项选取，可多选）：

| 失败检查项 | 可能原因与建议 |
| --- | --- |
| markdownlint / tag-closed-check / link-validity-check / resource-existence-check / codespell-check（run_check.js 类） | node 不可用；社区脚本拉取失败（网络/代理受限，建议检查网络或配置 `--proxy`）；外链检查网络超时（内网建议 `--disable-external`） |
| codespell-check | cspell 词典未安装或安装失败（npm 源不可达，处理见"环境依赖 → cspell 词典"） |
| filename-check | python / python3 不存在（安装后重跑该检查项） |
| 合规脚本块（标签 / MD 编码格式） | node 不可用；目标文件读取失败（路径含特殊字符或权限不足） |
| punctuation-check / compliance-check / aicheck（模型项 subagent） | subagent 执行超时或异常中断；上下文超限；原子 skill 内部错误 |
| 全部检查项均失败 | node / python 环境缺失，或项目根目录判定错误（重跑前先确认环境与路径） |

修复建议统一附加：修复后可用自定义模式单独重跑失败项，再与本次结果合并判断。

## 环境依赖

| 依赖 | 用途 | 缺失时 |
| --- | --- | --- |
| node | run_check.js、compliance 脚本、aggregate.js | 停止相关检查项并提示安装（Windows：`winget install OpenJS.NodeJS.LTS`；Ubuntu/Debian：`sudo apt install -y nodejs npm`；EulerOS：`sudo yum install -y nodejs npm`） |
| python 3 | docs-filename-check.py | 跳过 filename-check 并提示安装，禁止编造结果 |
| npm（随 Node.js 安装） | 首次安装 cspell 词典（见下方"cspell 词典"） | 词典无法安装时跳过 codespell-check 并明示，禁止编造结果（处理规则见下方"cspell 词典"） |

外链检查受网络影响：内网/无外网环境为 run_check.js 加 `--disable-external`，避免外链误报。

### cspell 词典（codespell-check 执行前置）

社区脚本为自包含打包产物，但 codespell-check 所需 cspell 词典未随脚本内联，运行时从脚本缓存目录解析；词典缺失时普通英文单词（如 quick、normal）会被大量误报为"拼写错误"。**本次检查范围包含 codespell-check 时，必须先检查词典是否已安装，缺失才安装，已安装则跳过**；三种 workflow 均在进入并发编排前执行本前置。

词典与脚本共用缓存目录 `<系统临时目录>/docs-tool-based-check`（该目录下存在 `node_modules/@cspell/cspell-bundled-dicts` 即已安装），依赖与社区 [docs-ci 依赖清单 package.json](https://gitcode.com/cann/.gitcode/blob/master/actions/staticcheck/dist/staticcheck/docs-ci/package.json) 保持一致（仅 `cspell-lib@9.2.1`）。检查与安装命令：

```bash
DICT_DIR="$(node -e "console.log(require('path').join(require('os').tmpdir(),'docs-tool-based-check'))")"
if [ ! -d "$DICT_DIR/node_modules/@cspell/cspell-bundled-dicts" ]; then
  mkdir -p "$DICT_DIR"
  printf '%s\n' '{' '  "name": "docs-ci",' '  "version": "0.0.1",' '  "dependencies": { "cspell-lib": "9.2.1" }' '}' > "$DICT_DIR/package.json"
  (cd "$DICT_DIR" && npm install --no-fund --no-audit)
fi
```

安装失败（无法访问 npm 源或 npm 不存在）时跳过 codespell-check：不产生其结果文件与计时条目，并在汇总中明示原因——门禁模式该缺失文件计入 skipped（门禁判定为未通过/不可信，须引导用户修复 npm 与网络后重跑）；全量/自定义模式按环境缺失计（不计缺失，仅明示）。禁止词典缺失时仍执行 codespell-check：大量误报会污染聚合报告。

## 工作流文件

| 文件 | 何时读取执行 |
| --- | --- |
| workflows/ci-gated-review.md | 门禁模式 |
| workflows/full-review.md | 全量模式 |
| workflows/custom-review.md | 自定义模式（含检查项清单解析与多选交互） |

模式确定后先读对应 workflow 文件再执行，禁止凭本文件的记忆直接编排。

## 基准回归（docs-skills-benchmark）

用户要求 **跑基准回归 / benchmark 对齐检查 / 基线对齐 / 检测 skill 修改后的结果对齐** 时：读取并严格遵循 [docs-skills-benchmark/README.md](docs-skills-benchmark/README.md) 的标准执行流程（固定样例全量模式重跑 → `bench.js convert` 转换 → `bench.js compare` 对齐比对 → 差异反馈）。要点：

- 检查对象固定为 `docs-skills-benchmark/cann_sim工具指南.md`，模式固定全量，不询问模式与 csv/md 导出；
- 对齐判定与差异反馈口径以 README"对齐判定口径"为准，不对齐时逐类反馈差异明细，禁止替用户判定可忽略；
- 基线 `common_bench.md` 只能按 README"基线重建"流程更新，禁止静默覆盖。

## 文件结构

```
docs-common-review/
├── SKILL.md                    # 本文件（入口：模式路由与原则）
├── workflows/                  # 三种模式的执行流程
│   ├── ci-gated-review.md
│   ├── full-review.md
│   └── custom-review.md
├── scripts/aggregate.js        # 聚合：规范化 + summary + 耗时/门禁判定 + csv/md 转换
├── scripts/merge_json.js       # JSON 数组合并（filename-check 多目标 → 单一结果文件）
├── evals/                      # 评测用例与共享数据集
├── docs-tool-based-check/      # 原子：5 项工具检查（脚本）
├── docs-filename-check/        # 原子：文件命名检查（python 脚本）
├── docs-punctuation-check/     # 原子：标点检查（skill）
├── docs-compliance-check/      # 原子：合规检查（skill）
├── docs-basic-aicheck/         # 原子：AI 语义检查（skill）
└── docs-skills-benchmark/      # 家族级基准回归（固定样例 + common_bench.md 基线 + scripts/bench.js 对齐工具，见其 README.md）
```
