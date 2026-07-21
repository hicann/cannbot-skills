
# scripts 索引指南（给人和 ClaudeCode 助手）

`scripts/` 放的是 **code-performance-advisor 的可执行工具脚本**（用于构建规则索引、打分、初始化 workspace、校验 tags 等）。
默认约定：在 skill 根目录运行命令（即 `skills/code-performance-advisor/`）。

---

## 1) 快速导航：按任务选脚本

| 我要做什么 | 入口脚本 | 常用命令（在 skill 根目录执行） | 主要产物 |
|---|---|---|---|
| 跑完整闭环（状态机编排：INIT→…→DONE） | `analysis_engine/workflow.py` | `python3 scripts/analysis_engine/workflow.py run --op <op> --mode interactive` | `workspace/sessions/<session_id>/...` |
| 续跑/查看某个 session（断点恢复） | `analysis_engine/workflow.py` | `python3 scripts/analysis_engine/workflow.py resume --op <op>` / `python3 scripts/analysis_engine/workflow.py status --op <op>` | `workspace/sessions/<session_id>/workflow_state.json` |
| 把某条规则加入/更新到索引 | `analysis_engine/cli.py` | `python3 scripts/analysis_engine/cli.py update-index --rule <rule_md>` | `assets/manifests/index.json` |
| 从 tag 文件对规则打分排序 | `analysis_engine/cli.py` | `python3 scripts/analysis_engine/cli.py score --tag-file <tag_json>` | `assets/manifests/scored_results.json` |
| 初始化某个算子的 workspace 输入布局（从 CAKE2 输出搬运） | `analysis_engine/init_workspace.py` | `python3 scripts/analysis_engine/init_workspace.py --op <op> [--root <CAKE2_ROOT>]` | `workspace/inputs/<op>/...` |
| 清理 session/cache 工件（按需） | `analysis_engine/clear.py` | `python3 scripts/analysis_engine/clear.py --sessions [--cache-tags] [--dry-run]` | `workspace/sessions/`, `workspace/cache/tags/` |
| 校验规则 tags 是否符合 taxonomy（并提示纠错/新增） | `analysis_engine/tag_validator.py` | `python3 scripts/analysis_engine/tag_validator.py [rule_pattern]` | 退出码用于 CI；提示修改位置 |

---

## 2) 目录结构（当前）

### 2.1 `analysis_engine/`

这里是脚本主体，尽量保持“stdlib-only、可直接运行”。

- `workflow.py`：端到端工作流编排入口（推荐主入口）
	- 作用：按状态机跑完整闭环（INIT→TAG→SCORE→ROUTE→SUGGEST→…→DONE）
	- 说明：`SCORE` 阶段会调用 `cli.py score` 完成打分（因此 `cli.py` 不是另一套“冲突工作流”，而是被编排调用的工具）
	- 常用：
		- `python3 scripts/analysis_engine/workflow.py run --op <op> --mode interactive`
		- `python3 scripts/analysis_engine/workflow.py resume --op <op>`
		- `python3 scripts/analysis_engine/workflow.py status --op <op>`

- `cli.py`：规则索引与打分 CLI（工具入口，可单独运行/调试）
	- 默认索引：`assets/manifests/index.json`
	- 默认 tag 目录：`workspace/cache/tags/`
	- 默认输出：`assets/manifests/scored_results.json`
	- 用法（示例）：
		- `python3 scripts/analysis_engine/cli.py update-index --rule assets/rules/special_rules/R_xxx/R_xxx.md`
		- `python3 scripts/analysis_engine/cli.py score --tag-file workspace/cache/tags/tag_xxx.json`

- `init_workspace.py`：把 CAKE2 的算子产物整理成标准输入目录（给后续 tagging / scoring 使用）
	- 目标布局：`workspace/inputs/<op>/` 下的 `code/`、`profiling/`（及可选 `roofline/`、`flowchart/`）
	- 用法（示例）：
		- `python3 scripts/analysis_engine/init_workspace.py --op fastgelu`
		- `python3 scripts/analysis_engine/init_workspace.py --op fastgelu --root /path/to/CAKE2`

- `clear.py`：清理 session/cache 工件（不触碰 `workspace/inputs`）
	- 适用：批量回收旧 session 或清空 tag 缓存
	- 常用：`--sessions`、`--cache-tags`、`--dry-run`

- `tag_validator.py`：校验 `*_tags.json` 的 tags 是否在 `references/standards/tag_taxonony.md` 中
	- 适用：新增/修改规则 tags 后自检；也可用于 CI
	- 特性：模糊匹配提示拼写修正；区分“疑似新 tag”与“疑似 typo”

---

## 3) 常见工作流（最短闭环）

### 3.0 跑完整闭环（推荐）

如果你希望按 SKILL.md 描述的状态机“一路跑到建议/验证”，优先用：

```bash
python3 scripts/analysis_engine/workflow.py run --op <op> --mode interactive
```

### 3.1 新增/更新规则后，让它可被检索打分

1) 写规则与 tags：
- `assets/rules/special_rules/R_xxx/R_xxx.md`
- `assets/rules/special_rules/R_xxx/R_xxx_tags.json`

2) 校验 tags：

```bash
python3 scripts/analysis_engine/tag_validator.py R_xxx
```

3) 更新索引：

```bash
python3 scripts/analysis_engine/cli.py update-index --rule assets/rules/special_rules/R_xxx/R_xxx.md
```

（批量重建索引可直接用 `bash bootstrap.sh`，见 skill 根目录脚本。）

### 3.2 针对某个算子跑规则匹配（打分）

```bash
python3 scripts/analysis_engine/cli.py score --tag-file workspace/cache/tags/tag_<op>_<timestamp>.json
```

然后读取 `assets/manifests/scored_results.json` 中的 `results[*].score / coverage_ratio / matched_tags / missing_tags`。

---

## 4) 扩展模板（后续加新脚本时建议补充）

新增脚本请在本 README 的“快速导航”加一行，并在脚本头部 docstring 里提供：

```text
用途：一句话
输入：哪些文件/目录
输出：哪些文件/目录
用法：3 条最常用命令
失败信号：常见报错/退出码
```

