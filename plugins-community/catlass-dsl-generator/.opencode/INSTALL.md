# 在 OpenCode 中安装 CATLASS DSL

## 前置条件

- 已安装 OpenCode。
- Git 仓库地址必须指向完整的 `catlass-dsl-plugin` 仓库，不能只复制 `.opencode/`。

## 项目级安装

在目标项目的 `opencode.json` 中加入：

```json
{
  "plugin": ["catlass-dsl-generator@git+<git-repository-url>"]
}
```

重启 OpenCode，然后使用原生 `skill` 工具确认能看到 `catlass-dsl-design`、
`catlass-dsl-develop`、`catlass-dsl-bench`、`catlass-dsl-optimize` 和
`catlass-dsl-knowledge`。

## 用户级安装

在 `~/.config/opencode/opencode.json` 中加入相同的 `plugin` 声明，然后重启 OpenCode。项目级和用户级只选择一种，避免重复加载。

## 固定版本

在 Git URL 末尾添加 tag 或 commit，例如 `#v0.2.0` 或 `#<commit-sha>`。

## 更新

更新 tag/commit 或恢复未固定的 Git URL，再重启 OpenCode。如果仍使用旧版本，清理 OpenCode/Bun 对该包的缓存后重新安装；清理前先确认缓存目标只属于 CATLASS DSL。

## 卸载

从对应的项目级或用户级 `opencode.json` 的 `plugin` 数组中删除 CATLASS DSL 条目并重启。不要删除同一配置中的其他插件。

## Windows 回退

若 Bun 无法处理 Git URL，使用系统 npm 将仓库安装到专用目录，再在 `plugin` 数组中填写该本地包路径。不要覆盖现有 OpenCode 配置目录。

## 排错

1. 检查 OpenCode 日志中是否加载 `.opencode/plugins/catlass-dsl.js`。
2. 确认 transform hook `experimental.chat.messages.transform` 已执行，并在第一条用户消息中看到唯一标记 `CATLASS_DSL_OPENCODE_SUBAGENT_MAPPING_V1`；映射只注入一次。
3. 该映射将 `Subagent (general-purpose)` 解释为调用 `task`，并设置 `subagent_type: "general"`；实现与 review 必须使用不同的 task 身份。
4. 确认包内同时存在根目录 `skills/` 和 `.opencode/`。不要把 `skills/` 复制到平台目录，也不要安装或发布固定 Agent。
5. 使用原生 `skill` 工具列出 skill，并加载 `catlass-dsl-develop`。
6. 若错误包含 `CATLASS DSL skills directory not found`，说明安装包不完整，应重新安装完整仓库。
