
# assets 索引指南（给人和 ClaudeCode 助手）

`assets/` 放的是 **code-performance-advisor 技能运行所依赖的“静态资产”**：规则、模板、清单（manifest）、优质样例等。
目标是：**你要做什么 → 立刻知道去哪个目录、看哪个入口文件**。

---

## 1) 快速导航：按任务找东西

| 我要做什么 | 去哪个目录 | 优先看什么 |
|---|---|---|
| 新增/修改性能优化规则（Rule） | `rules/` | `rules/special_rules/R_*/R_*.md` + 对应 `*_tags.json` |
| 让新规则能被检索/打分（更新索引） | `manifests/` | `manifests/index.json`（由 CLI/`bootstrap.sh` 维护） |
| 找规则写作模板/字段规范 | `templates/` | `rule_template.md` |
| 写/补算子描述模板 | `templates/` | `op_description_template.md` |
| 写/补 profiling 解读模板 | `templates/` | `profiling_interpretation_template.md` |
| 放（或取）LLM 提示词资产 | `llm_prompts/` |（当前为空，预留）|
| 放运行日志/中间产物 | `logs/` |（当前为空，预留）|
| 放配置文件 | `conifgs/` |（当前为空，注意目录名拼写为 conifgs）|

---

## 2) 目录说明（当前结构）

### 2.1 `rules/`

规则资产目录，包含可检索的优化规则与其侧车标签。

- `rules/special_rules/`：按规则 ID 建目录，典型结构如下：
	- `R_xxx/R_xxx.md`：规则正文（触发信号、动作、验证、约束等）
	- `R_xxx/R_xxx_tags.json`：标签侧车（`tags`/`required_tags` 等）
	- `R_xxx/code_snippets/`：代码对照片段（可选但强烈建议）
		- `caseN/base_code/base_code.md`
		- `caseN/good_code/good_code.md`

- `rules/general_rules/`：通用规则预留目录（当前为空）。

### 2.2 `manifests/`

规则索引与评分结果的“机器可读清单”。

- `index.json`：规则索引（核心入口），结构要点：
	- `version`: 索引版本
	- `rules`: 规则条目数组（每项含 `rule_paths`、`tags`、`required_tags`、`source_hash`、`source_mtime` 等）
- `index.json.backup.*`：`bootstrap.sh` 自动生成的历史备份。
- `scored_results.json`：一次 `rules_search/score` 的输出样例（含 `query_tags`、每条 rule 的 `score/coverage_ratio/matched_tags/missing_tags`）。

### 2.3 `templates/`

写作/结构化模板（给人写，也给模型生成/补全用）。

- `rule_template.md`：规则正文模板
- `op_description_template.md`：算子描述模板
- `profiling_interpretation_template.md`：profiling 解读模板

### 2.4 `llm_prompts/` / `logs/` / `conifgs/`

这三个目录当前为空，作为后续扩展位：

- `llm_prompts/`：建议按“子任务/阶段”分文件，例如 `tagging.md`、`suggest.md`、`rule_update.md`。
- `logs/`：建议仅存放可复现实验的日志/中间产物，避免长期堆积。
- `conifgs/`：预留配置文件目录（注意拼写），建议统一使用 `.json`/`.yaml`，并在目录下补一个 `README.md`。

---

## 3) 新增/更新规则时的最短闭环

1. 在 `rules/special_rules/R_<RULE_ID>/` 下新增或修改：
	 - `R_<RULE_ID>.md`
	 - `R_<RULE_ID>_tags.json`
	 - （可选）`code_snippets/caseN/{base_code,good_code}/*.md`

2. 更新索引（让新规则可被检索/打分）：

```bash
cd skills/code-performance-advisor
bash bootstrap.sh
```

说明：`bootstrap.sh` 会调用 `scripts/analysis_engine/cli.py update-index --rule <rule_file>`，并写入 `assets/manifests/index.json`，同时保留 `index.json.backup.*`。

---

## 4) 扩展模板（可直接复制）

当你新增一个一级目录或引入新类型资产时，建议在该目录下创建 `README.md` 并包含：

```md
# <目录名>

## 用途
- 一句话说明该目录解决什么问题。

## 入口文件
- <文件A>：何时读、读什么。
- <文件B>：何时读、读什么。

## 命名约定
- 文件命名规则...
- 目录层级规则...

## 检索关键词
- 关键词1, 关键词2, 关键词3
```

这样可以保证：人类定位快，ClaudeCode/大模型检索也稳定命中。

