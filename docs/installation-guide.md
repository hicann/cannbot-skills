# 安装指南

本文档提供 CANNBot Skills 的完整安装命令参考、使用入门与故障排查。快速上手见 [README 快速开始](../README.md)。

📖 [功能清单](feature-list.md) · [使用样例](skills-usage.md) · [架构设计](architecture-design.md) · [README](../README.md)

---

## 安装方式对比

本项目提供两种安装方式，按需选择：

| 维度 | 完整安装（推荐） | 独立 Skill 安装 |
|---|---|---|
| **包含内容** | Skills + Agents + Workflows + 外部依赖 | 仅 Skills |
| **安装工具** | install-helper / init.sh | install-helper / npx skills |
| **适用场景** | 需要完整开发流程编排 | 只需单个能力 |
| **支持工具** | OpenCode/Claude/Trae/Cursor/Copilot/CodeArts | 70+ 种 AI 编程工具 |

> [!NOTE]
> **外部依赖**指安装时由安装脚本动态克隆的外部仓库（如 `asc-devkit`、`pypto`、`tilelang-ascend`、`cann-samples` 等）。
>
> **Plugin** = Skills + Agents + Workflows 的组合包，提供端到端开发流程编排；**Skill** 是独立能力模块。标注 Plugin 的能力需通过完整安装获取，标注 Skill 的能力两种安装方式均可使用。

---

## 完整安装

通过 `install-helper` 安装完整插件内容（Skills + Agents + Workflows + 外部依赖）：

### Linux / macOS

```bash
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash
```

### Windows

```powershell
iwr -useb https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.ps1 | iex
```

### npm（全平台，需要 [Node.js 18+](https://nodejs.org/zh-cn/download)）

```bash
npm install -g @cannbot-ai/install-helper
```

> 也可以免安装直接运行：`npx @cannbot-ai/install-helper`

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

命令中的 `opencode` 可替换为 `claude` / `trae` / `cursor` / `copilot` / `codearts`。各插件的详细安装步骤见对应目录下的 `quickstart.md`，例如 [ops-direct-invoke 快速上手](../plugins-official/ops-direct-invoke/quickstart.md)。

---

## 验证安装

安装完成后，运行以下命令确认安装成功：

```bash
# 完整安装验证
install-helper list          # 查看已安装插件，应显示已安装的 Plugin

# 独立 Skill 安装验证
npx skills list              # 查看已安装 Skills，应显示已安装的 Skill
```

若命令未找到或列表为空，运行 `install-helper doctor --fix` 自动诊断修复。

---

## 更新与卸载

```bash
# 更新单个插件
install-helper update <plugin>

# 更新全部
install-helper update

# 卸载
install-helper uninstall <plugin>
```

独立 Skill 安装的更新与卸载使用 `npx skills` 命令，详见 [skills CLI 文档](https://github.com/vercel-labs/skills)。

---

## 独立 Skill 安装

本仓库 Skills 遵循 [Agent Skills](https://agentskills.io) 开放标准，可通过开源 [skills CLI](https://github.com/vercel-labs/skills) 安装到 70+ 种 AI 编程工具（OpenCode、Claude Code、Cursor、Codex、Copilot、Trae、CodeArts 等）。

```bash
# 浏览可用 Skills
npx skills add https://gitcode.com/cann/cannbot-skills.git --list

# 安装单个 Skill（交互式选择目标工具）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-tiling-design

# 安装 Skill 到指定工具（支持 opencode / claude-code / trae / cursor / copilot / codearts 等）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill ascendc-env-check --skill npu-arch --agent opencode

# 安装全部 Skill 到所有已检测到的工具（非交互式）
npx skills add https://gitcode.com/cann/cannbot-skills.git --skill '*' --agent '*' -y

# 查看已安装的 Skills
npx skills list

# 卸载
npx skills remove ascendc-tiling-design
```

---

## 安装后使用

安装完成后，在你的工作目录中启动 AI 编程工具，直接用自然语言描述需求即可：

| 你想做什么 | 可以这样说 | 用到的能力 |
|-----------|-----------|-----------|
| 开发一个算子 | "帮我开发一个 Abs 算子，输入 float16，输出 float16" | Plugin: `ops-direct-invoke` |
| 调试精度问题 | "我的 Add 算子精度不达标，帮我排查一下" | Skill: `ascendc-precision-debug` |
| 查阅 API 文档 | "aclnnAdd 接口的参数和返回值是什么" | Skill: `ascendc-docs-search` |
| 检查开发环境 | "帮我检查一下当前的 CANN 开发环境" | Skill: `ascendc-env-check` |
| 代码检视 | "帮我检视这段 Kernel 代码是否符合规范" | Plugin: `ops-code-reviewer` |

> [!TIP]
> 完整安装可运行 `install-helper status` 查看已安装插件，独立 Skill 安装可运行 `npx skills list` 查看已安装 Skills。

> 更多使用样例详见 [Skills 使用样例](skills-usage.md)。

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

更多故障排查详见各插件的 `quickstart.md`（如 [ops-direct-invoke/quickstart.md](../plugins-official/ops-direct-invoke/quickstart.md)）。
