# cannbot-knowledge

`cannbot-knowledge` 是面向 AscendC / 昇腾 NPU 算子开发知识库的社区插件。它提供知识编译、治理、检索、Issue 材料整理和勘误流程；真实知识内容维护在独立知识库仓库中。

## 内容边界

本插件包含：

- `skills/`：知识编译、治理、检索相关 skills。
- `skills/*/scripts/`：随对应 Skill 安装的唯一脚本入口；生产脚本统一接受 `--knowledge-root <path>`，`knowledge-query` 还会读取安装时持久化的 root 配置。
- `CONTRIBUTING.md`：外部贡献和勘误合入规则。

本插件不包含：

- 真实 `reference/`、`ops/`、`runbooks/` 知识正文。
- 真实 `graph/` 判定缓存和 `log/` 审计日志。
- 私有轨迹、密钥、内网路径或不可公开日志。

## 快速使用

```bash
bash init.sh project claude /path/to/project --profile consumer --knowledge-root /path/to/knowledge-base
python3 skills/knowledge-query/scripts/knowledge_query.py discover
python3 skills/knowledge-query/scripts/knowledge_query.py search --query "DataCopyPad 对齐"
```

安装 profile：

- `consumer`：只安装 `knowledge-query`，用于只读检索和知识消费。
- `issue`：只安装 `knowledge-issue-report`，用于提交 Issue、整理 `needs-info` 和复现附件。
- `contributor` / `all`：安装全部 7 个 skill，用于知识编译、治理、检索、Issue 和贡献门禁。
- `--skills knowledge-query,knowledge-issue-report`：高级用法，精确安装指定 skill 子集。

如果通过 Claude Code Plugin marketplace 安装，可以直接选择对应包：

- `/plugin install cannbot-knowledge-consumer-skills@cannbot`：只读检索。
- `/plugin install cannbot-knowledge-issue-skills@cannbot`：Issue 提交材料整理。
- `/plugin install cannbot-knowledge@cannbot`：完整贡献者 Team，依赖全量 skills。

更多流程见 [quickstart.md](./quickstart.md) 和 [CONTRIBUTING.md](./CONTRIBUTING.md)。

安装时传入 `--knowledge-root /path/to/knowledge-base` 后，插件会把 root 持久化为：

- `CANNBOT_KNOWLEDGE_ROOT`
- `CANNBOT_KNOWLEDGE_ROOTS`
- `OKF_KNOWLEDGE_ROOT`
- `OKF_KNOWLEDGE_ROOTS`

`skills/knowledge-query/scripts/knowledge_query.py` 会按以下顺序解析上述变量（优先级从高到低）：

1. `--knowledge-root` / `--knowledge-roots` 参数
2. `CANNBOT_KNOWLEDGE_ROOT`、`CANNBOT_KNOWLEDGE_ROOTS`、`OKF_KNOWLEDGE_ROOT`、`OKF_KNOWLEDGE_ROOTS`
3. `~/.config/cannbot/knowledge.env`
4. 有限目录结构自动探测
