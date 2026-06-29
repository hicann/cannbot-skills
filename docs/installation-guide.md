# 安装指南

本文档提供 CANNBot Skills 的完整安装命令参考与故障排查。快速上手见 [README 快速开始](../README.md)。

---

## 完整安装

通过 `install-helper` 安装完整插件内容（Skills + Agents + Workflows + 外部依赖）：

| 方式 | 命令 | 说明 |
|------|------|------|
| **curl** | `curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh \| bash` | 一键安装（推荐） |
| **npx** | `npx @cannbot-ai/install-helper` | 免安装运行 |
| **npm** | `npm install -g @cannbot-ai/install-helper` | 全局安装 |

### install-helper 命令参考

| 命令 | 说明 |
|------|------|
| `install-helper` | 交互式安装向导 |
| `install-helper install <name>` | 安装指定插件或 Skill（自动识别） |
| `install-helper install --list` | 按类别列出所有可用 Skills |
| `install-helper update [plugin]` | 更新已安装的插件 |
| `install-helper uninstall <plugin>` | 卸载指定插件 |
| `install-helper list` | 查看可用场景及安装状态 |
| `install-helper doctor --fix` | 健康检查 + 自动修复 |
| `install-helper lang set en_US` | 切换语言 |

> 完整命令参考和详细文档：[install-helper README](../plugins-community/install-helper/README.md)

### 手动执行安装脚本

如果不使用 install-helper，也可以进入对应插件目录手动执行 `init.sh` 安装脚本。以安装 AscendC Kernel 直调插件到 OpenCode 为例：

```bash
cd plugins-official/ops-direct-invoke
bash init.sh project opencode
```

`<tool>` 支持 `opencode` / `claude` / `trae` / `cursor` / `copilot`，各插件的详细安装步骤参见对应插件目录下的 `quickstart.md` 文档。

---

## 独立 Skill 安装命令

本仓库的 Skills 遵循 [Agent Skills](https://agentskills.io) 开放标准，可通过开源 [skills CLI](https://github.com/vercel-labs/skills) 安装到 70+ 种 AI 编程工具（OpenCode、Claude Code、Cursor、Codex、Trae 等）。

```bash
# 浏览可用 Skills
npx skills add https://gitcode.com/cann/cannbot-skills.git --list

# 安装单个 Skill（交互式选择目标工具）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-tiling-design

# 安装 Skill 到指定工具（支持 opencode / claude-code / trae / cursor 等）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-env-check --skill npu-arch --agent opencode

# 安装全部 Skill 到所有已检测到的工具（非交互式）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill '*' --agent '*' -y

# 查看已安装的 Skills
npx skills list

# 卸载
npx skills remove ascendc-tiling-design
```

> **此方式仅安装独立 Skills**。如需完整插件内容（Skills + Agents + Workflows + 外部依赖），请使用 `install-helper` 或 `init.sh` 脚本。

---

## 安装遇到问题？

运行 `install-helper doctor --fix` 自动检测并修复常见问题。

| 问题 | 解决方法 |
|------|---------|
| `install-helper` 报错 | 确认 Node.js >= 18：`node --version` |
| AI 工具无法识别 Skills | 重启工具或新开会话 |
| 软链接失效 | `install-helper doctor --fix` |
| 网络问题 | 配置 GitCode SSH Key 或设置代理 |
| CANN 环境未配置 | 仅影响代码编译/运行类 Skills，知识检索类不受影响 |

更多故障排查详见各场景对应的 quickstart 文档。
