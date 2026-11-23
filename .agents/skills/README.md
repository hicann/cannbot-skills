# .agents/skills 兼容入口

本目录是跨工具兼容的 skill 发现入口（`.agents/skills/<name>/SKILL.md`，Agent Skills 开放规范），opencode 等兼容该规范的工具会在仓库内自动发现此处的 skill。

## 定位与边界（硬性约束）

本目录**只引用已有 skills 服务于 skills 仓自身维护，不存放 skills 本体**：

- 只允许存放指向 skill 本体的符号链接（目录级、相对路径）；本体一律维护在各领域目录（`infra/`、`ops/`、`model/` 等）下
- 禁止在本目录新增 skill 本体或任何实质内容（`references/`、`scripts/` 等）。新增 skill 请提交到对应领域目录，如需被本目录发现，再补一条链接
- 链接名与本体目录名一致，即 `.agents/skills/<name>` -> `../../<领域>/<name>`，本体内容原样透传，零复制、无需同步

## 当前入口

| 入口（本目录） | 本体 | 用途 |
| --- | --- | --- |
| `gitcode-issue-gen/` | `infra/gitcode-issue-gen/` | GitCode Issue 生成与提交（PR 关联 / 手动提单） |
| `gitcode-issue-handler/` | `infra/gitcode-issue-handler/` | GitCode Issue 分诊、跟进与修复交付 |
| `gitcode-pr-handler/` | `infra/gitcode-pr-handler/` | GitCode PR 标题 / 描述重新生成并写回 |
| `gitcode-toolkit/` | `infra/gitcode-toolkit/` | GitCode 协作共享基础库（内部参考，不直接触发） |
| `cannbot-skill-reviewer/` | `infra/cannbot-skill-reviewer/` | CANNBot Skill 入库审查（自动门禁+判例法评分+整改建议） |

## 维护约定

- 新增入口：`ln -s ../../<领域>/<name> .agents/skills/<name>`（必须用相对路径，适应任意 clone 位置）
- 本体移动 / 重命名 / 目录重构后，检查本目录链接未断：`ls -l .agents/skills/`，逐个 `readlink -f` 应解析到本体 `SKILL.md`
- 内容、`references/`、`scripts/` 的变更一律修改本体

> 依赖符号链接支持：Windows 下 clone 需开启 `core.symlinks=true` 及开发者模式，当前不作 Windows 兼容目标。
