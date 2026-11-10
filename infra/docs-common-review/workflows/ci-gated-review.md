# 门禁模式（ci-gated-review）

## 触发

用户选择门禁模式；或要求 CI 门禁项本地预检 / 提交前预检。

## 检查范围（固定，全部脚本级）

| 检查项 | 执行方式 |
| --- | --- |
| markdownlint、tag-closed-check、link-validity-check、resource-existence-check、codespell-check | run_check.js 每项一条 Bash，并发执行（每项独立结果文件 + 独立计时文件） |
| compliance-check 的标签检查（芯片/位置标签） | tag_checker.js 直调（合规脚本块内，与工具项并发） |
| compliance-check 的 MD 编码格式检查（UTF-8 无 BOM） | encoding_format_checker.js 直调（合规脚本块内，与工具项并发） |

不执行：filename-check、punctuation-check、aicheck、compliance-check 的模型规则部分（术语/敏感词/table/目录层级）。

依据：社区 docs-ci 拦截 PR 的项仅 link / resource / tag-closed / compliance（标签+编码）；markdownlint、codespell 在 CI 中仅告警。本模式定位是零模型参与、纯脚本管线。

## 并发模型

- 工具 5 项（各自一条 Bash）与合规脚本块（一条 Bash）**并发执行**，禁止串行排队。
- markdownlint 先行单独执行：社区脚本首次运行需下载缓存到固定路径，并发下载同一文件会互踩；其返回后其余 4 项与合规脚本块在同一条消息中并发发出（缓存确认已存在时，5 项工具 + 合规块全部同时发出）。
- tag/encoding 两脚本为"读-合并-写"语义，多文件在其循环内**串行**追加同一结果文件（不能拆开并发），但该块整体与工具项并发。

## 产物

- 最终产物：`<产物目录>/docs-common-review-result.json`（+用户选择导出的 csv/md）。产物目录 = 待检文档同级 `output/`（规则见 SKILL.md 输出节）。
- 中间产物：`<项目根>/output/` 下的分项结果文件与 `timing/` 计时目录，收尾时清理（见 SKILL.md 通用收尾）。

## 执行流程

执行任何步骤前先输出 `[CHECKPOINT: 开始]`；前一个 CHECKPOINT 未输出禁止进入下一步。

### 第 1 步：输入解析

- 工作区准备（先于一切写入执行，含 auto_temp.md 落盘）：按下方命令预清理 `<项目根>/output/` 历史中间产物并重建目录。
- 解析检查对象（单文件 / 多文件 / 目录 / 粘贴内容；内容形态先写入 `<项目根>/output/temp/auto_temp.md`）。
- 目录目标先展开为 `.md` 文件清单（排除任何 `output/` 目录下的文件）。
- 确定项目根目录（通常为当前工作目录）与仓名（项目根最后一级目录名，供标签检查使用）。
- 确定产物目录（按 SKILL.md 输出规则）。
- 输出 `[CHECKPOINT: 输入解析完成]`

**工作区准备命令**（清理集合＝本家族在 `output/` 下的全部已知产物名：分项结果文件、各 spoke 的中间文件与临时目录、`timing/`、`temp/`；目的：上次运行中断的遗留文件不污染本次聚合，也排除陈旧中间文件干扰并发运行）：

```bash
rm -rf "<项目根>/output/"tool-based-*-result.json \
  "<项目根>/output/"filename-* \
  "<项目根>/output/punctuation-markdown-check-result.json" \
  "<项目根>/output/compliance-markdown-check-result.json" \
  "<项目根>/output/compliance-gate-result.json" \
  "<项目根>/output/general-markdown-check-result.json" \
  "<项目根>/output/"tag_check_result.json \
  "<项目根>/output/"encoding_format_check_result.json \
  "<项目根>/output/"prohibited_words_check_result.json \
  "<项目根>/output/"null_cell_check_result.json \
  "<项目根>/output/"catalog_num_check_result.json \
  "<项目根>/output/"mix_language_check_result.json \
  "<项目根>/output/timing" "<项目根>/output/temp" \
  "<项目根>/output/failed_checks.json" \
  "<项目根>/output/punctuation" "<项目根>/output/compliance" "<项目根>/output/general"
mkdir -p "<项目根>/output" "<项目根>/output/timing"
```

### 第 2 步：并发执行全部脚本检查

**codespell 词典前置**：门禁范围固定含 codespell-check，先按 SKILL.md"环境依赖 → cspell 词典"完成词典检查/安装（已装则跳过、秒级完成；首次安装需访问 npm 源），完成后再进入下方编排。安装失败时按该节规则跳过 codespell-check：其结果文件缺失计入 skipped，门禁判定为未通过/不可信，须在汇报中明示。

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

**合规脚本块**（一条 Bash，块整体与工具项并发）：

```bash
mkdir -p "<项目根>/output/timing"
T0=$(date +%s%3N)
for f in <文件清单>; do
  node <skill_base>/docs-compliance-check/scripts/tag_checker.js "$f" "<项目根>/output/compliance-gate-result.json" <仓名>
done
T1=$(date +%s%3N); echo "compliance-tag-check $((T1-T0))" > "<项目根>/output/timing/compliance-tag-check.txt"

T0=$(date +%s%3N)
for f in <文件清单>; do
  node <skill_base>/docs-compliance-check/scripts/encoding_format_checker.js "$f" "<项目根>/output/compliance-gate-result.json"
done
T1=$(date +%s%3N); echo "compliance-encoding-check $((T1-T0))" > "<项目根>/output/timing/compliance-encoding-check.txt"
```

**执行编排**（互不依赖的调用必须放在同一条消息中并发发出）：

- 消息 1：Bash（markdownlint 工具项）+ Bash（合规脚本块）
- 消息 2（markdownlint 返回、缓存就绪后）：Bash（tag-closed-check）+ Bash（link-validity-check）+ Bash（resource-existence-check）+ Bash（codespell-check），四条同消息并发
- 社区脚本缓存确认已存在（此前运行过）时：5 个工具项 + 合规脚本块全部并入消息 1 并发

- 内网/无外网环境为 run_check.js 追加 `--disable-external`；需代理时 `--proxy=<地址>`；文件多可调 `--concurrency`。首次运行含脚本拉取缓存时间，该项耗时偏高属正常。
- 全部返回后核对：命令成功但结果文件缺失的工具项创建内容为 `[]` 的同名文件（0 问题也必须落盘）；命令失败的项禁止伪造空结果——**记录为失败检查项**（canonical 检查项名 + 错误输出摘要），保留缺失状态进聚合（计为 skipped，门禁判定按未通过/不可信处理，并经第 3 步写入 summary.failed_check）；**失败只影响该项，禁止因此中止其余检查项**。合规两脚本（tag_checker.js、encoding_format_checker.js）零问题时亦自行写入 `[]`，compliance-gate-result.json 在合规块整体成功后必然存在，无需补写。
- 输出 `[CHECKPOINT: 脚本检查完成]`

### 第 3 步：聚合

**失败清单落盘（仅当存在失败检查项时）**：把全部失败检查项（命令失败、环境缺失跳过如 cspell 词典无法安装）写入 `<项目根>/output/failed_checks.json`（格式与写盘命令见 full-review 第 6 步）。

```bash
node <skill_base>/scripts/aggregate.js \
  --inputs=<项目根>/output/tool-based-markdownlint-result.json,<项目根>/output/tool-based-tag-closed-check-result.json,<项目根>/output/tool-based-link-validity-check-result.json,<项目根>/output/tool-based-resource-existence-check-result.json,<项目根>/output/tool-based-codespell-check-result.json,<项目根>/output/compliance-gate-result.json \
  --mode=ci-gated \
  --timing=<项目根>/output/timing \
  --output=<产物目录>/docs-common-review-result.json
```

- 失败项的缺失文件照常列入 `--inputs`（聚合计为 skipped）；失败清单存在时聚合命令**必须**追加 `--failed-file=<项目根>/output/failed_checks.json`（写入 summary.failed_check，aggregate.js 同步把 gate.passed 置为不可信）；聚合整体失败（无任何可用输入）时停止并如实报告，禁止给出"通过"。
- 输出 `[CHECKPOINT: 聚合完成]`

### 第 4 步：汇总输出

读取聚合报告，向用户输出：

1. **失败检查项显著提示（前置）**：summary.failed_check 非空时，先输出检查项错误提示块（模板与可能原因映射见 SKILL.md"检查项失败提示"节，逐项列全），再输出下方各项
2. 门禁判定：通过 / 未通过（拦截项问题数、告警项问题数；输入被跳过或存在 failed_check 时判定不可信，须明示）
3. 拦截项问题逐条：文件、行号、问题描述、修改建议
4. 告警项（markdownlint / codespell）计数与要点
5. 各检查项耗时（summary.check_timing_ms，换算为秒呈现）
6. 误报处理引导：失效链接/资源可加入项目 `.doctools/url.whitelist.txt`；拼写误报加入 `.doctools/codespell.whitelist.txt`；不适用的 markdownlint 规则可建 `.doctools/markdownlint.config.json` 关闭

- 输出 `[CHECKPOINT: 结果输出完成]`

随后**必须**执行 SKILL.md"通用收尾"：询问是否导出 csv/md → 按选择重跑 aggregate（同参数 + `--format=...`）→ 清理中间产物（必做）→ 输出 `[CHECKPOINT: 收尾完成]`。

## 约束

- 零模型逐行检查、零 subagent；互不依赖的脚本项必须并发执行，禁止串行排队
- **任一脚本项失败只影响该项本身**：记录失败清单并入 summary.failed_check（呈现见 SKILL.md"检查项失败提示"节），其余项正常执行并完成聚合；门禁判定据此不可信
- 报告内容只来自脚本与 aggregate.js 输出，禁止增删改写与编造
- node 缺失时停止执行并提示安装（指引见 SKILL.md 环境依赖）
- 结果文件内容与汇总数字必须一致（以聚合报告为准）
