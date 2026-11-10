# 自定义模式（custom-review）

## 触发

用户选择自定义模式；或要求按指定检查项组合检查文档。

## 并发模型

同 full-review：勾选项互不依赖的全部并发（工具项各自一条 Bash、filename 各目标一条 Bash、模型项各自一个 subagent，单条消息混发，无任何顺序约束）；markdownlint 若勾选则先行预热缓存，其余项与其同消息或待其返回后并发。

## 产物

同 full-review：最终产物 `<产物目录>/docs-common-review-result.json`（+可选 csv/md），产物目录 = 待检文档同级 `output/`；中间产物在 `<项目根>/output/`（含 `timing/` 计时目录），收尾清理（见 SKILL.md 通用收尾）。

## 执行流程

执行任何步骤前先输出 `[CHECKPOINT: 开始]`；前一个 CHECKPOINT 未输出禁止进入下一步。

### 第 1 步：输入解析

同门禁模式第 1 步（四种形态；工作区预清理与预创建命令见门禁模式第 1 步，先于 auto_temp.md 落盘执行；内容先落盘 `<项目根>/output/temp/auto_temp.md`；目录展开排除 `output/`；确定项目根目录与产物目录）。

- 输出 `[CHECKPOINT: 输入解析完成]`

### 第 2 步：检查项清单确定（硬约束：先问后执行）

- 用户已在对话中给出检查项 → 逐项按 SKILL.md 的别名映射表转换为 canonical 名。
- **未给出 → 必须先以可勾选（多选）选项形式向用户发出询问并等待回复**：勾选项为 9 项 canonical 检查项（各附一句话功能说明，取 SKILL.md"质检能力项"表）。**禁止**未经用户勾选反馈就自行确定检查项集合（含全选 9 项）并开始执行，**禁止**"先全选执行、再告知可重跑指定子集"的补救式交互——正确行为是先问、收到勾选后再执行。
- 用户明确跳过选择（明确回复"跳过 / 全部 / 默认"等）→ 按全选 9 项执行，并在继续执行前告知用户本次为全选。
- 出现无法映射的名称 → 停止执行，向用户列出全部 9 项合法名，待修正后重试；禁止模糊猜测或静默忽略。

**CHECKPOINT 门控**：`[CHECKPOINT: 检查项确定完成]` 仅允许在以下两种情形输出，否则禁止输出、更禁止进入第 3 步：

1. 用户已给出检查项清单，且完成别名映射与合法性校验；
2. 已向用户发出多选询问，且已收到用户的勾选结果或明确跳过指令。

- 输出 `[CHECKPOINT: 检查项确定完成]`

### 第 3 步：并发执行

按 full-review 第 2-3 步的方式执行勾选项：

- **codespell 词典前置**：勾选项含 codespell-check 时，先按 SKILL.md"环境依赖 → cspell 词典"完成词典检查/安装（已装则跳过），再进入下方并发编排；安装失败时按该节规则跳过该勾选项（环境缺失，不计缺失）并明示。
- **脚本组**：勾选项 ∩ {markdownlint、tag-closed-check、link-validity-check、resource-existence-check、codespell-check}——每项一条 Bash（调用模板见 full-review），同一条消息并发发出；markdownlint 若在勾选中则先行预热缓存，其余脚本项（含 filename 各目标）与其同消息或待其返回后并发。
- **模型组**：勾选 punctuation-check / compliance-check / aicheck 时各自派 subagent（提示词模板见 full-review），与脚本项同消息并发，无任何顺序约束。
- 调用模板、计时文件写法、命令成功/失败的处理规则均同 full-review 第 2-3 步。
- **失败隔离**：任一勾选项异常中断（命令失败、超时、subagent 崩溃）只影响该项——记录为失败检查项（canonical 名 + 错误输出摘要），其余勾选项正常执行；用户勾选了 N 项、失败 M 项时，其余 N-M 项必须完成并照常聚合，禁止因单项失败中止整个质检。

- 输出 `[CHECKPOINT: 检查执行完成]`

### 第 4 步：filename 结果合并

勾选了 filename-check 时，按 full-review 第 4 步将各目标结果合并为单一 `filename-check-result.json` 并删除 `filename-<n>/` 目录。

- 输出 `[CHECKPOINT: filename 合并完成]`（未勾选 filename-check 时输出后直接进入下一步）

### 第 5 步：完整性校验

期望结果文件 = 仅本次勾选项对应的结果文件（取 full-review 第 5 步清单的子集）。缺失重派一次；仍缺失则标注。环境缺失跳过的项不计缺失。**重派后仍缺失、环境缺失跳过与第 3 步记录的失败项，一并计入失败清单**（第 6 步写入 failed_checks.json）。

- 输出 `[CHECKPOINT: 完整性校验完成]`

### 第 6 步：聚合

`--inputs` 只传本次实际执行项的结果文件，`--mode=custom`，`--timing=<项目根>/output/timing`，`--output=<产物目录>/docs-common-review-result.json`，其余同 full-review 第 6 步。**存在失败检查项时**：先把失败清单写入 `<项目根>/output/failed_checks.json`（格式见 full-review 第 6 步），聚合命令**必须**追加 `--failed-file=<项目根>/output/failed_checks.json`（写入 summary.failed_check；无失败时不传、字段不出现）。

注意：第 1 步的预清理命令已移除历史运行遗留的全部已知产物文件，正常情况下不存在本次未执行项的旧结果文件；聚合前仍逐项核对 `--inputs` 清单与本次勾选项一一对应，发现清单外文件立即剔除。

- 输出 `[CHECKPOINT: 聚合完成]`

### 第 7 步：汇总输出

**失败检查项显著提示（前置）**：summary.failed_check 非空时（用户勾选 N 项、失败 M 项），**必须**先输出检查项错误提示块（模板与可能原因映射见 SKILL.md"检查项失败提示"节，逐项列全 M 个失败项），再输出常规汇报。

常规汇报同 full-review 第 7 步（summary + 逐条问题 + 各检查项耗时 + 缺失标注 + 误报处理引导）。

- 输出 `[CHECKPOINT: 结果输出完成]`

随后**必须**执行 SKILL.md"通用收尾"，逐项落实：

1. 询问是否导出 csv/md（多选：CSV / Markdown / 两者都要 / 不导出；用户明确跳过询问视同不导出）。
2. 按用户选择导出（如需）。
3. **清理中间产物（必做，不可跳过）**：删除 `<项目根>/output/` 下本次分项结果文件（`tool-based-*-result.json`、`filename-check-result.json`、`punctuation-markdown-check-result.json`、`compliance-markdown-check-result.json`、`general-markdown-check-result.json`）、`filename-*/`、`timing/`、`temp/`、`failed_checks.json`（失败清单），以及产物目录中本次未生成的历史 csv/md（聚合失败时保留现场供排查）。
4. 输出 `[CHECKPOINT: 收尾完成]`。

## 约束

- 严格按用户勾选执行：未勾选的项禁止执行，勾选的项禁止跳过（环境缺失除外并明示）
- **未给清单必须先弹 9 项多选框**：收到用户勾选结果或明确跳过指令前，禁止输出"检查项确定完成"、禁止开始执行任何检查项；禁止"先全选执行、再提供重跑"的补救式交互
- **汇报后必须完成通用收尾**：csv/md 导出询问 + 中间产物清理缺一不可，未清理禁止结束
- 勾选项之间互不依赖的必须并发执行，禁止串行排队（无顺序约束）
- 非法检查项零容忍：报错并列出全部合法名，禁止静默忽略或近似执行
