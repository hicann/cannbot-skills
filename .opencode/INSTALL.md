# Installing CANNBot for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) v1.3+ installed
- Git installed
- Python 3 installed (for asc-devkit documentation cleanup)

## Option 1: Plugin Install (Recommended)

### Step 1: Install the plugin

```bash
opencode plugin cannbot@git+https://gitcode.com/cann/cannbot-skills.git
```

For global installation (available in all projects):

```bash
opencode plugin cannbot@git+https://gitcode.com/cann/cannbot-skills.git -g
```

### Step 2: Select a team (optional)

不指定 team 时，默认仅安装 `ops-direct-invoke` team。如需安装其他 team 或全部能力，编辑 `opencode.json`（项目级 `.opencode/opencode.json` 或全局 `~/.config/opencode/opencode.json`）：

**安装全部（team 参数为 all）**：

```json
{
  "plugin": [
    ["cannbot@git+https://gitcode.com/cann/cannbot-skills.git", {"team": "all"}]
  ]
}
```

**仅安装 Ascend C Kernel 直调开发（默认，不指定 team）**：

```json
{
  "plugin": [
    "cannbot@git+https://gitcode.com/cann/cannbot-skills.git"
  ]
}
```

**仅安装 PyPTO 算子开发**：

```json
{
  "plugin": [
    ["cannbot@git+https://gitcode.com/cann/cannbot-skills.git", {"team": "pypto-op-orchestrator"}]
  ]
}
```

Available teams:

| Team | Description | Agents | Skills |
|------|-------------|--------|--------|
| `all` | 安装全部 | 8 | 25 |
| `ops-direct-invoke`（默认） | Ascend C Kernel 直调开发 | 3 | 11 |
| `pypto-op-orchestrator` | PyPTO 算子开发 | 3 | 8 |

### Step 3: Restart OpenCode

Restart OpenCode to load the plugin.

## Option 2: Script Install

### 1. Clone the repository

```bash
git clone https://gitcode.com/cann/cannbot-skills.git ~/.config/opencode/cannbot
```

### 2. Run the installer

**Ascend C Kernel direct-invoke**:

```bash
cd ~/.config/opencode/cannbot/plugins-official/ops-direct-invoke
bash init.sh global opencode
```

**PyPTO operator development**:

```bash
cd ~/.config/opencode/cannbot/plugins-official/pypto-op-orchestrator
bash init.sh global opencode
```

### 3. Restart OpenCode

Restart OpenCode to load the new skills and agents.

## Verify Installation

After restarting OpenCode, ask:

```
你有哪些 skills？
```

You should see team-specific skills listed (e.g., `ascendc-*` for ops-direct-invoke, `pypto-*` for pypto-op-orchestrator).

You can also verify by asking:

```
你是谁？
```

The agent should identify itself as CANNBot.

## Updating

### Plugin mode

```bash
opencode plugin cannbot@git+https://gitcode.com/cann/cannbot-skills.git -f
```

### Script mode

```bash
cd ~/.config/opencode/cannbot && git pull
cd plugins-official/ops-direct-invoke && bash init.sh global opencode
```

## Troubleshooting

### Skills not found

1. Check if skills directory exists:
   - Plugin mode: `ls ~/.cache/opencode/packages/cannbot*/node_modules/cannbot/ops/`
   - Script mode: `ls ~/.config/opencode/skills/`
2. Verify AGENTS.md is accessible
3. Restart OpenCode

### asc-devkit not available

The asc-devkit is cloned automatically on first use. If it fails:

```bash
cd <cannbot-root>/plugins-official/ops-direct-invoke
git clone https://gitcode.com/cann/asc-devkit.git asc-devkit
```

### Tool mapping

When AGENTS.md references Claude Code concepts, use OpenCode equivalents:
- Subagent dispatch → `@agent-name`（具体名称取决于安装的 team）
- `Read`, `Write`, `Edit`, `Bash` → Your native tools
- `skill` tool → OpenCode's native `skill` tool

## Getting Help

- Repository: https://gitcode.com/cann/cannbot-skills
- Issues: https://gitcode.com/cann/cannbot-skills/issues
