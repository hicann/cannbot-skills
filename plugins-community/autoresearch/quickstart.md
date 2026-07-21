# AutoResearch 插件快速开始

AutoResearch 会以独立项目的形式安装。插件根目录直接承载 `op-autoresearch` 项目，不存在额外的 `app/` 嵌套或第二套运行时；其中包含 `.claude/`、`.opencode/`、`scripts/`、`src/`、`tests/`、`ar_examples/`、`pyproject.toml`、`config.yaml` 和完整文档。

```bash
bash plugins-community/autoresearch/init.sh project claude /path/to/project
```

如果当前目录就是目标项目目录，可以省略最后的路径参数：

```bash
bash /path/to/cannbot-skills/plugins-community/autoresearch/init.sh project claude
```

如果从 bash/WSL 调用安装脚本，请传入当前 shell 能解析的目标路径。

## 安装与运行位置

`cannbot-skills` 仓库只负责提供安装脚本。安装完成后，不要在插件目录里运行 AutoResearch；请进入目标项目根目录运行 Claude Code：

```bash
cd /path/to/project
claude
```

安装依赖后，可以在 Claude Code 或 OpenCode 中使用 `/autoresearch` 命令。所有运行态都会留在目标项目根目录下，例如 `workspace/`、`ar_tasks/`、`.session_tasks/` 和 `.task_dir_pointers/`。

## 安装后的目录结构

```text
<project>/
|-- .claude/
|   |-- agents/
|   |   `-- ar-diagnosis.md
|   |-- commands/
|   |   `-- autoresearch.md
|   `-- settings.json
|-- .opencode/
|-- .session_tasks/
|-- .task_dir_pointers/
|-- ar_examples/
|-- ar_tasks/
|-- scripts/
|-- src/
|-- tests/
|-- workspace/
|-- .gitignore
|-- AGENTS.md
|-- AUTORESEARCH.md
|-- README.md
|-- config.yaml
|-- pyproject.toml
`-- requirements-worker.txt
```

安装脚本会备份目标目录已有的同名入口文件或目录，然后写入 AutoResearch 运行时入口。安装后从目标项目根启动 Claude Code，即可按 AutoResearch 的标准入口使用：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py --op-name <op> --devices 0
```

如果评测走已经启动好的远端 worker，用 `--worker-url host:port` 替代本地设备参数。
