# 全量模式（full-review）

## 触发

用户选择全量模式；或要求全面 / 彻底检查文档质量。

## 检查范围

9 项全部：

- 脚本直调：markdownlint、tag-closed-check、link-validity-check、resource-existence-check、codespell-check（run_check.js 每项一条 Bash）、filename-check（python 脚本逐目标运行后合并为单一结果文件）
- 原子 skill（subagent 执行）：punctuation-check、compliance-check、aicheck

## 并发模型

- **互不依赖的检查全部并发**：工具 5 项（各自一条 Bash）、filename 各目标（各自一条 Bash）、punctuation / compliance / aicheck（各自一个 subagent）在同一条消息中发出；Agent 派发与 Bash 调用可混合同发。禁止把互不依赖的检查排成先后步骤串行执行。
- **无顺序约束**：三个模型项之间不需要任何先后等待（aicheck 不等 compliance）。曾存在"aicheck 等 compliance"约束，其前提经源码核实不成立——本家族 spoke 的 `tag_checker.js` 直接把结果写入调用方传入的输出路径，spoke 侧无人产出 `output/tag_check_result.json`（该文件仅社区 CI 版本产生），无从互删；第 1 步预清理历史中间产物进一步保证并发起点干净。
- **缓存预热**：工具 5 项中 markdownlint 先行单独执行（社区脚本首次运行需下载缓存到固定路径，并发下载同一文件会互踩），其返回后其余 4 项再并发（缓存确认存在时 5 项全部同消息发出）。

## 产物

- 最终产物：`<产物目录>/docs-common-review-result.json`（+用户选择导出的 csv/md）。产物目录 = 待检文档同级 `output/`（规则见 SKILL.md 输出节）。
- 中间产物：`<项目根>/output/` 下的分项结果文件与 `timing/` 计时目录，收尾时清理（见 SKILL.md 通用收尾）。

## 执行流程

执行任何步骤前先输出 `[CHECKPOINT: 开始]`；前一个 CHECKPOINT 未输出禁止进入下一步。

### 第 1 步：输入解析

同门禁模式第 1 步（四种形态；工作区预清理与预创建命令见门禁模式第 1 步，先于 auto_temp.md 落盘执行；内容先落盘 `<项目根>/output/temp/auto_temp.md`；目录展开排除 `output/`；确定项目根目录与产物目录）。

- 输出 `[CHECKPOINT: 输入解析完成]`

### 第 2 步：并发启动（单条消息发出全部互不依赖调用）

**codespell 词典前置**：全量范围固定含 codespell-check，先按 SKILL.md"环境依赖 → cspell 词典"完成词典检查/安装（已装则跳过、秒级完成；首次安装需访问 npm 源），完成后再进入下方消息编排。安装失败时按该节规则跳过 codespell-check（环境缺失，不计缺失）并在第 7 步明示。

**消息 1**（同一条消息，混合 Bash 与 Agent）：

1. Bash：markdownlint 工具项（预热缓存；缓存确认已存在时其余 4 项也并入本条消息）
2. Bash × N：filename-check 每目标一条（目标多于 5 个时可合并为一条 Bash 顺序执行，仍每目标独立输出目录与计时文件）
3. Agent：punctuation-check subagent
4. Agent：compliance-check subagent
5. Agent：aicheck subagent（无任何等待，与 compliance 同消息发出）

**消息 2**（markdownlint 返回、缓存就绪后）：Bash（tag-closed-check）+ Bash（link-validity-check）+ Bash（resource-existence-check）+ Bash（codespell-check），四条同消息并发。

**工具项调用模板**（`<检查项>` 替换为 canonical 名，每项一条 Bash）：

```bash
T0=$(date +%s%3N)
node <skill_base>/docs-tool-based-check/scripts/run_check.js \
  --targets="<目标清单>" \
  --checks=<检查项> \
  --output="<项目根>/output/tool-based-<检查项>-result.json"
T1=$(date +%s%3N)
mkdir -p "<项目根>/output/timing" && echo "<检查项> $((T1-T0))" > "<项目根>/output/timing/<检查项>.txt"
```

**filename 调用模板**（`<序号>` 从 1 起，每目标一条 Bash；该脚本每次运行会删除并重建结果文件，多目标必须各用独立输出目录）：

```bash
T0=$(date +%s%3N)
mkdir -p "<项目根>/output/filename-<序号>" "<项目根>/output/timing"
python <skill_base>/docs-filename-check/docs-filename-check.py "<目标>" "<项目根>/output/filename-<序号>"
T1=$(date +%s%3N)
echo "filename-check-<序号> $((T1-T0))" > "<项目根>/output/timing/filename-check-<序号>.txt"
```

- `python` 不存在时改试 `python3`；仍失败则跳过 filename-check 并按 SKILL.md 提示用户（不产生结果文件与计时条目），禁止编造结果。
- 重名文件检查按脚本语义覆盖文件所在目录的第一层 `.md`。
- 内网/无外网环境为 run_check.js 加 `--disable-external`；首次运行含脚本拉取缓存耗时属正常。

**subagent 提示词模板**（以 punctuation 为例；派发时记录当前时间戳，供耗时兜底计算）：

```text
你是原子检查技能的执行者。读取并严格遵循 <skill_base>/docs-punctuation-check/SKILL.md 的完整执行流程执行检查，输入指定如下：
- 待检查对象：<目标，多值逗号分隔>
- 项目根目录：<项目根>
- 检查范围：全部检查（无需向用户询问范围）
完成后仅返回：结果文件路径与检出问题总数；其余遵循该 SKILL.md 的静默要求。
```

compliance 同理（`<skill_base>/docs-compliance-check/SKILL.md`，不传 `--skip-*` 参数即全量执行；仓名按其 SKILL.md 第 1.1 步规则由 subagent 自行从项目根目录推导）。aicheck 用 `<skill_base>/docs-basic-aicheck/SKILL.md`，执行全部规则。

宿主不支持同消息多工具调用时，降级为逐项顺序执行并告知用户耗时将增加。

- 输出 `[CHECKPOINT: 检查已全部启动]`

### 第 3 步：完成通知与计时

- 每个模型项完成通知到达后，用一条 Bash 写计时文件（内容一行 `<检查项> <毫秒>`）：

```bash
echo "<检查项> <毫秒数>" > "<项目根>/output/timing/<检查项>.txt"
```

  毫秒数优先取完成通知中的 duration_ms；通知未含时用 当前时间戳（`date +%s%3N`）减派发时间戳。

- 各 Bash 项的计时文件已由模板命令写入，无需补记。
- 命令成功但结果文件缺失的工具项创建 `[]` 同名文件；**命令失败的项记录为失败检查项**（canonical 检查项名 + 错误输出摘要，供第 6 步写入 failed_checks.json），其缺失文件照常列入聚合 inputs（计 skipped）；**失败只影响该项本身，禁止因此中止其余检查项**。
- subagent（模型项）执行失败或超时：同样记录为失败检查项（原因含 subagent 错误信息），交由第 5 步重派；**不阻断其他检查项的执行与返回**。
- 全部任务返回后输出 `[CHECKPOINT: 检查完成]`

### 第 4 步：filename 结果合并（单一产物）

全部 filename 调用返回后执行：

```bash
node <skill_base>/scripts/merge_json.js \
  --inputs=<项目根>/output/filename-1/filename-markdown-check-result.json,<项目根>/output/filename-2/filename-markdown-check-result.json,... \
  --output=<项目根>/output/filename-check-result.json
rm -rf <项目根>/output/filename-*
```

- `--inputs` 按实际生成的 `filename-<n>/filename-markdown-check-result.json` 逐个列出；`rm -rf` 在 merge 成功后执行。
- 输出 `[CHECKPOINT: filename 合并完成]`

### 第 5 步：完整性校验

期望结果文件清单（相对 `<项目根>/output/`）：

| 检查项 | 结果文件 |
| --- | --- |
| markdownlint | tool-based-markdownlint-result.json |
| tag-closed-check | tool-based-tag-closed-check-result.json |
| link-validity-check | tool-based-link-validity-check-result.json |
| resource-existence-check | tool-based-resource-existence-check-result.json |
| codespell-check | tool-based-codespell-check-result.json |
| filename-check | filename-check-result.json（单一合并产物） |
| punctuation-check | punctuation-markdown-check-result.json |
| compliance-check | compliance-markdown-check-result.json |
| aicheck | general-markdown-check-result.json |

逐项核对落盘；缺失项重派对应执行单元一次（重派后覆盖同名计时文件）；仍缺失则在第 7 步汇报中明确标注"该检查项结果缺失"，禁止以空结果冒充。环境缺失而跳过的项（如 python 缺失跳过 filename-check、cspell 词典无法安装跳过 codespell-check）不计缺失。**重派后仍缺失的项与环境缺失跳过的项，连同第 3 步记录的命令/subagent 失败项，一并计入失败清单**（第 6 步写入 failed_checks.json）。

- 输出 `[CHECKPOINT: 完整性校验完成]`

### 第 6 步：聚合

**失败清单落盘（仅当存在失败检查项时）**：把全部失败检查项（命令失败、subagent 失败、重派后仍缺失、环境缺失跳过）写入 `<项目根>/output/failed_checks.json`：

```bash
node -e "require('fs').writeFileSync('<项目根>/output/failed_checks.json', JSON.stringify([{\"check\":\"<canonical 检查项名>\",\"reason\":\"<具体原因，含错误输出摘要>\"}/*,...*/], null, 2))"
```

```bash
node <skill_base>/scripts/aggregate.js \
  --inputs=<项目根>/output/tool-based-markdownlint-result.json,<项目根>/output/tool-based-tag-closed-check-result.json,<项目根>/output/tool-based-link-validity-check-result.json,<项目根>/output/tool-based-resource-existence-check-result.json,<项目根>/output/tool-based-codespell-check-result.json,<项目根>/output/filename-check-result.json,<项目根>/output/punctuation-markdown-check-result.json,<项目根>/output/compliance-markdown-check-result.json,<项目根>/output/general-markdown-check-result.json \
  --mode=full \
  --timing=<项目根>/output/timing \
  --output=<产物目录>/docs-common-review-result.json
```

- `--inputs` 逐文件列出实际生成的结果文件（环境缺失跳过的项剔除）；失败清单存在时聚合命令**必须**追加 `--failed-file=<项目根>/output/failed_checks.json`（aggregate.js 将其写入 `summary.failed_check`，无失败时不传该参数、字段不出现）；聚合失败时如实报告。
- 输出 `[CHECKPOINT: 聚合完成]`

### 第 7 步：汇总输出

**失败检查项显著提示（前置）**：summary.failed_check 非空时，**必须**先输出检查项错误提示块（模板与可能原因映射见 SKILL.md"检查项失败提示"节），再输出常规汇报。

summary（模式、总数、级别分布、检查项分布、各检查项耗时）+ 逐条问题（文件、行号、描述、建议；量大时按级别分组呈现）+ 缺失标注（如有）+ 误报处理引导（同门禁模式第 4 步）。

- 输出 `[CHECKPOINT: 结果输出完成]`

随后**必须**执行 SKILL.md"通用收尾"：询问是否导出 csv/md → 按选择重跑 aggregate（同参数 + `--format=...`）→ 清理中间产物（必做）→ 输出 `[CHECKPOINT: 收尾完成]`。

## 约束

- 不得跳过 9 项中的任何一项（环境缺失时须明示跳过原因）
- **任一检查项失败只影响该项本身**：记录失败清单并入 summary.failed_check（呈现见 SKILL.md"检查项失败提示"节），其余项正常执行并完成聚合，禁止因单项失败中止
- 全部 9 项无顺序依赖：同消息并发发出，禁止串行排队（唯一例外：markdownlint 首跑缓存预热先行）
- subagent 执行期间主 Agent 不重复执行原子检查
- 报告零编造，仅来自各执行单元真实输出与聚合结果
