# Changelog fragments

为避免多个 PR 同时追加 `CHANGELOG.md` 产生冲突，用户可见变更统一在本目录新增 fragment，**不要直接修改 `CHANGELOG.md`**。

## 新增 fragment

每个 PR 新增一个文件：

```text
docs/changelog.d/<PR号>.<类型>.md
```

类型四选一：`added`（新特性）、`fixed`（缺陷修复）、`docs`（文档）、`chore`（其他变更）。

例如 `docs/changelog.d/1234.fixed.md`，内容仅一条 Markdown 列表项，正文不超过 200 字：

```markdown
- 【安装部署】修复安装器重复执行时复用旧配置的问题。
```

PR 号必须与当前 PR 一致；CI 校验文件名、类型、单条内容和 PR 号。可先创建草稿 PR 获取编号，再补充 fragment。无用户可见变更的 PR 可不带 fragment，但需在 PR 描述中说明。

## 发版聚合

维护者运行以下命令，将 fragment 按类型和 PR 号插入 `CHANGELOG.md` 最新日期段，并删除已聚合文件：

```bash
python3 scripts/changelog.py aggregate --repo . --date 2026-11-03
```

聚合结果写入 `GENERATED FILE` 标记。检查变更后通过普通 PR 提交，不由 CI 直接推送主线。同一日期只允许聚合一次；没有 fragment 时不产生改动。

仅校验 fragment：

```bash
python3 scripts/changelog.py validate --repo .
```

聚合命令会修改 `CHANGELOG.md` 并删除 fragment，仅在准备发布结果时执行。
