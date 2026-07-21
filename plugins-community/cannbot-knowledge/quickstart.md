# Quickstart

## 安装到知识库项目

```bash
cd plugins-community/cannbot-knowledge
bash init.sh project claude /path/to/project --knowledge-root /path/to/knowledge-base
```

支持的工具参数：`claude`、`opencode`、`trae`、`cursor`、`copilot`。未指定时默认 project + opencode + 当前目录。

默认安装 `all` profile，即完整贡献者能力。也可以按用途安装部分 skill：

```bash
# 只消费知识库：只安装 knowledge-query
bash init.sh project claude /path/to/project --profile consumer --knowledge-root /path/to/knowledge-base

# 只提交/整理 Issue：只安装 knowledge-issue-report
bash init.sh project claude /path/to/cannbot-knowledge --profile issue

# 贡献知识库：安装完整知识编译、治理、检索、Issue 能力
bash init.sh project claude /path/to/cannbot-knowledge --profile contributor

# 高级用法：精确指定 skill 子集
bash init.sh project claude /path/to/cannbot-knowledge --skills knowledge-query,knowledge-issue-report
```

## 通过 Claude Code Plugin 安装

首次使用 marketplace 时先注册：

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
```

按用途选择安装包：

```text
# 只消费知识库：只安装 knowledge-query
/plugin install cannbot-knowledge-consumer-skills@cannbot

# 只提交/整理 Issue：只安装 knowledge-issue-report
/plugin install cannbot-knowledge-issue-skills@cannbot

# 贡献知识库：安装完整 Team 和全量 skills
/plugin install cannbot-knowledge@cannbot
```

安装后执行 `/reload-plugins`，然后新开会话或 `/clear`。

## 运行只读检索

```bash
python3 skills/knowledge-query/scripts/knowledge_query.py discover
python3 skills/knowledge-query/scripts/knowledge_query.py search --query "DataCopyPad 对齐"
```

## 提交前治理检查

```bash
python3 skills/knowledge-query/scripts/knowledge_query.py verify
python3 skills/ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root /path/to/cannbot-knowledge verify
python3 skills/knowledge-lint/scripts/knowledge_lint.py --knowledge-root /path/to/cannbot-knowledge
```

涉及 `ops/` 或 `runbooks/` 的变更，额外运行：

```bash
python3 skills/ops-knowledge-vv-ingest/scripts/validate_layered_knowledge.py --knowledge-root /path/to/cannbot-knowledge
```
