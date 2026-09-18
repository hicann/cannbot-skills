# GitCode Issue Handler 快速上手

`gitcode-issue-handler` 可以帮你查看和回复 GitCode Issue，也可以继续排查代码、准备修复和创建 PR。
建议先拿一个 Issue 试用，熟悉后再处理整批 Issue。

## 开始前准备

你需要：

- Python 3.10 或更高版本；
- 一个 GitCode Token，用于读取或修改 Issue；
- 要处理 Issue 的代码仓库已经下载到本地。

先检查 Python 依赖是否已经安装：

```bash
python3 -c "import requests, yaml"
```

如果命令报错，再安装缺少的依赖：

```bash
python3 -m pip install "requests>=2.28.0" "PyYAML>=6.0"
```

登录 GitCode 后，在“个人设置 → 访问令牌”中创建 Token，并勾选 Issue 和 Pull Request 相关权限。使用前把 Token 放到环境变量中：

```bash
export GITCODE_TOKEN="你的 Token"
```

不要把 Token 写进配置文件或提交到代码仓库。

## 安装

先进入要处理 Issue 的代码仓库：

```bash
cd /path/to/your-repository
```

然后选择你正在使用的客户端。使用 `npx` 安装时需要 Node.js 20 或更高版本。

### Claude Code

在 Claude Code 中依次执行：

```text
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install infra-skills@cannbot
```

安装时选择适合自己的范围。只想在当前仓库使用，可以选择 local；希望团队共享，可以选择 project；希望所有仓库都能使用，可以选择 user。

### OpenCode

```bash
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool opencode --level project
```

### Codex

```bash
npx @cannbot-ai/install-helper install gitcode-issue-handler gitcode-toolkit \
  --tool codex --level project
```

### 让 Agent 帮你安装

如果本地已经有 `cannbot-skills` 源码仓，也可以直接在该仓库中启动 Agent，把下面这段话发给它：

```text
请把当前 cannbot-skills 仓库中的 gitcode-issue-handler 和 gitcode-toolkit
以 project 级安装到 /path/to/your-repository，目标客户端是 Codex，
使用本仓最新源码并完成必要的项目配置。
```

把目标目录和客户端名称换成自己的即可，例如 OpenCode、Claude Code 或 Codex。Agent 会检查本仓的安装方式并完成操作，不需要自己找 Skill 目录或创建软链接。

项目级安装会把 Skill 放到当前仓库，并自动准备配置文件。如果使用 Claude Marketplace 或全局安装，第一次在目标仓库中运行时也会自动准备配置。

## 配置

配置文件放在：

```text
.cannbot/gitcode-issue-handler/config/
├── classify_config.yaml
└── operator_owners.yaml
```

处理单个 Issue 时通常不用改配置。`repo` 留空即可，首次运行会根据 Issue 地址或当前仓库的 GitCode remote 自动填写。

如果要批量处理 Issue，先打开 `classify_config.yaml`，确认 `responsibility` 写的是你负责的范围。这里用正常文字描述即可。例如：

```yaml
responsibility:
  handle:
    - 我负责的算子和公共组件
  list-only:
    - 其他团队负责的算子
  ignore: []
```

`auto-response` 和 `auto-assign` 默认都是 `false`。刚开始使用时建议保持默认值：助手会先给出回复稿或询问是否执行，不会直接回复或指派 Issue。

如果已经有确认过的算子负责人，可以再填写 `operator_owners.yaml`。没有负责人名单也可以先用，助手会列出待确认项，不会把猜测的候选人直接当成负责人。

## 使用

下面的自然语言提示词适用于 OpenCode、Codex，以及通过 install-helper 安装的 Claude Code。
通过 Claude Marketplace 安装时，也可以先输入 `/infra-skills:gitcode-issue-handler`，再输入任务。

### 处理单个 Issue

最简单的方式是直接给出 Issue 地址：

```text
使用 gitcode-issue-handler 处理 https://gitcode.com/cann/ops-math/issues/1511
```

如果只想分析和准备回复，不想改代码，可以说：

```text
使用 gitcode-issue-handler，只回复 https://gitcode.com/cann/ops-math/issues/456，不改代码
```

默认配置不会直接回复或指派 Issue。助手完成分析后会给出回复稿，并在需要写入 GitCode 或提交代码时询问你。

### 批量拉取并处理

批量处理时，先从目标仓库根目录启动客户端。不要从 `cannbot-skills` 源码仓发起批量任务：

```bash
cd /path/to/your-repository
```

第一次运行前，打开 `.cannbot/gitcode-issue-handler/config/classify_config.yaml`，确认 `responsibility` 与你的实际职责一致。然后输入：

```text
使用 gitcode-issue-handler 批量拉取并处理当前仓库需要关注的 Issue
```

Handler 会拉取当前仓库的 open Issue，过滤掉不在责任范围内或已经处理完成的项目，再为需要关注的 Issue 准备回复、转交或代码处理方案。`auto-response` 和 `auto-assign` 保持默认的 `false` 时，先看助手给出的草稿和待办，再决定是否执行外部操作。

如果这次只想看看有哪些 Issue 需要处理，可以说：

```text
使用 gitcode-issue-handler 批量拉取当前仓库需要关注的 Issue，先列出处理计划，不要发送回复
```

以后继续处理时，仍然使用第一条批量提示词即可。Handler 会复用上次的进度，并检查等待中的 Issue 有没有新回复。处理报告保存在 `.cannbot/gitcode-issue-handler/reports/latest.md`。

## 遇到问题

- 提示缺少 `gitcode-toolkit`：把它和 `gitcode-issue-handler` 安装到同一个客户端、同一级别。
- 提示缺少 `requests` 或 `yaml`：重新执行上面的 Python 依赖安装命令。
- 找不到目标仓库：确认客户端从目标仓库根目录启动，并检查 GitCode remote 是否正确。
- 想查看处理结果：打开 `.cannbot/gitcode-issue-handler/reports/latest.md`。

不同客户端的全局安装、更新、卸载和完整配置说明见 [安装与配置指南](docs/installation-guide.md)。
