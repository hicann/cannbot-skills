# PR 代码获取

派发为 `general` 子 Agent 执行。

## 派发

```
Agent({
  subagent_type: "general",
  description: "PR 代码获取",
  prompt: "PR 代码获取

【输入】
- PR URL: {pr_url}

【执行步骤】

1. 识别 PR 托管平台（URL 含 `gitcode.com` → GitCode）
2. 获取 diff：
   - 脚本路径 = `{skill_base}/scripts/get_gitcode_pr_diff.py`
   - `mkdir -p ./operators/.pr_diff`
   - 执行脚本，传入完整 URL，`--output` 保存到 `./operators/.pr_diff/{pr_number}.diff`
3. 克隆完整源码：
   - `mkdir -p ./operators/.pr_repo`
   - 执行 `python {skill_base}/scripts/clone_pr_source.py --repo {repo_url} --pr {pr_number} --clone-dir ./operators/.pr_repo/{pr_number}/`
   - 克隆失败则终止

【输出】

返回以下结构化信息：
- `operator_name`: {提取的算子名}
- `diff_path`: `./operators/.pr_diff/{pr_number}.diff`
- `repo_path`: `./operators/.pr_repo/{pr_number}/`

禁止生成报告文件。"
})
```
