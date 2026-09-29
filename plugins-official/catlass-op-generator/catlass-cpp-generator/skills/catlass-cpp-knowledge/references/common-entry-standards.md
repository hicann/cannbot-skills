# 公共入库标准

1. concept 必须使用 OKF v0.2 frontmatter，并包含 `type/title/description/tags/status/generated/verified/sources`。
2. `sources` 必须固定到可复现的提交、版本或仓库相对文件；正文用同 ID 的 Markdown footnote 归因。
3. 不把名称推断、未运行的性能判断或对话记忆写成事实。不确定内容标为待核对条件。
4. 正文保留“接口与概念、用法、代码模式、约束、失败表现、验证方法”六章。
5. 新文件必须加入所在目录和全部父级索引；别名加入 `query-vocabulary.yaml` 前检查冲突。
6. 修改后运行 `validate`、`reindex`、再次 `validate`，并检查查询能命中预期 concept。
