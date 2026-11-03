# Changelog fragments

为避免多个 PR 同时追加 `CHANGELOG.md` 产生冲突，用户可见变更统一在本目录新增 fragment，**不要直接修改 `CHANGELOG.md`**。

## 新增 fragment

每个 PR 新增一个文件：

```text
changelog.d/<PR号>.<类型>.md
```

类型四选一：

- `added`：新特性；
- `fixed`：缺陷修复；
- `docs`：文档；
- `chore`：其他变更。

例如 `changelog.d/1234.fixed.md`，文件内容仅一条 Markdown 列表项，正文不超过 200 字：

```markdown
- 【安装部署】修复安装器重复执行时复用旧配置的问题。
```

PR 号必须与当前 PR 一致；CI 同时校验文件名、类型、单条内容和 PR 号。可先创建草稿 PR 获取编号，再补充 fragment。无用户可见变更的 PR 可不带 fragment，但需在 PR 描述中说明。

## 发版聚合

维护者手动运行 `changelog-release` workflow。workflow 调用 `scripts/changelog.py aggregate`：

1. 按类型和 PR 号将 fragment 插入 `CHANGELOG.md` 最新日期段；
2. 在文件中写入 `GENERATED FILE` 标记，明确其为生成物；
3. 写入成功后删除已聚合 fragment；
4. 仅提交 `CHANGELOG.md` 与 `changelog.d`，由 `cann-robot` 推回 `master`。

同一日期只允许聚合一次；没有 fragment 时 workflow 不产生提交。workflow 需要配置仅具有本仓 `contents write` 最小权限的 `CHANGELOG_RELEASE_TOKEN` secret，缺失时失败且不推送；凭据只通过进程级 HTTP header 使用，不写入远端配置。

本地校验与聚合：

```bash
python3 scripts/changelog.py validate --repo .
python3 scripts/changelog.py aggregate --repo . --date 2026-11-03
```

聚合命令会修改 `CHANGELOG.md` 并删除 fragment，仅在准备发布结果时执行。
