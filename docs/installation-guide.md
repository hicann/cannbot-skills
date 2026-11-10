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
| **支持工具** | OpenCode/Claude/Trae/Cursor/Copilot/CodeArts/Qoder/DSH | 70+ 种 AI 编程工具 |

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

### npm（全平台，需要 [Node.js 20+](https://nodejs.org/zh-cn/download)）

先运行 `node --version` 确认版本；使用 Node.js 18 等旧版本时，请通过上述官方下载页升级到 Node.js 20 或更高版本，再执行安装命令。

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

### 安装 cannbot 仓插件

部分插件已迁移至 [cannbot 仓](https://gitcode.com/cann/cannbot) 维护，install-helper 直接支持：无需更换工具，`install-helper install <插件名>` 即可，产物与官方安装器直装完全一致（详见 [install-helper README](../plugins-community/install-helper/README.md#安装-cannbot-仓插件)）。

```bash
install-helper install ops-direct-invoke --tool opencode
```

cannbot 仓插件在 `list` / `status` 中带 `[cannbot]` 徽标；仅支持 project 级安装与 opencode / codex / claude / trae 四种工具。

### 手动执行安装脚本

如果不使用 install-helper，也可以进入对应插件所在仓库手动执行 `init.sh`。`ops-direct-invoke` 已迁至 cannbot 仓，要求 Node.js >= 20；以安装到现有项目 `/path/to/project` 为例：

```bash
git clone --recurse-submodules https://gitcode.com/cann/cannbot.git
cd cannbot
bash plugins-official/ops-direct-invoke/init.sh project opencode /path/to/project
```

该插件当前仅支持 `project` 级安装，工具为 `opencode` / `claude` / `codex` / `dsh` / `trae`；可用 `bash plugins-official/ops-direct-invoke/init.sh --help` 查看参数。旧仓脚本的 `global`、`--override` 等参数不适用于此入口。详细步骤见 [ops-direct-invoke 安装与使用说明](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke)。

---

## 验证安装

```bash
install-helper list     # 完整安装：应列出已安装的 Plugin
npx skills list         # 独立 Skill 安装：应列出已安装的 Skill
```

命令未找到或列表为空时，运行 `install-helper doctor --fix` 自动诊断修复。

Skill 以软链接安装到各工具的标准目录（指向 `~/.cannbot/repo/` 缓存）。OpenCode 与 Claude Code 的发现路径：

| 工具 | project 级（仅当前项目） | global 级（本机所有项目） |
|------|------------------------|--------------------------|
| OpenCode | `<项目根>/.opencode/skills/` | `~/.config/opencode/skills/` |
| Claude Code | `<项目根>/.claude/skills/` | `~/.claude/skills/` |

```bash
ls -l .opencode/skills/   # 检查软链接是否存在（Claude Code 对应 .claude/skills/）
```

> OpenCode 还兼容 `.claude/skills/`、`.agents/skills/`；其他工具（Trae / Cursor / Codex / Copilot / CodeArts / Qoder / DSH）的路径运行 `install-helper doctor` 查看，DSH 详见 [DSH（DeepSeek Harness）](#dshdeepseek-harness) 一节。

最后确认客户端已加载：**Claude Code** 输入 `/skills` 查看列表；**OpenCode** 直接询问"当前会话有哪些可用的 skills？"，或输入"检查环境"做触发词冒烟验证。

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

通过 install-helper 安装的独立 Skill 使用重复安装来刷新：

```bash
install-helper install <skill-name> --tool <tool> --level <level>
install-helper uninstall <skill-name> --tool <tool> --level <level>
```

通过 `npx skills` 安装的独立 Skill，其更新与卸载使用 `npx skills` 命令，详见
[skills CLI 文档](https://github.com/vercel-labs/skills)。

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

### DSH（DeepSeek Harness）

install-helper 原生支持 DSH 目标：项目级安装落到 `<项目根>/.dsh/skills/`，全局安装落到 `$DSH_HOME/skills/`（`DSH_HOME` 未设置时默认 `~/.dsh`）。DSH 会原生发现 `<项目根>/.dsh/skills` 下的 `SKILL.md`（frontmatter 解析、按名触发均正常）。

```bash
# 通过 install-helper 安装（以 ascendc-env-check 为例）
install-helper install ascendc-env-check -t dsh -l project
install-helper install ascendc-env-check -t dsh -l global
```

也可以手动复制 skill 目录到技能根（无 bash 环境的 Windows 用户可用资源管理器直接复制，效果等价于 `npx skills add`）：

```bash
# 项目级手动安装
mkdir -p .dsh/skills && cp -r ops/ascendc-env-check .dsh/skills/ascendc-env-check

# 全局手动安装
mkdir -p ~/.dsh/skills && cp -r ops/ascendc-env-check ~/.dsh/skills/ascendc-env-check
```

---

## 安装后使用

安装完成后，在你的工作目录中启动 AI 编程工具，直接用自然语言描述需求即可：

| 你想做什么 | 可以这样说 | 用到的能力 |
|-----------|-----------|-----------|
| 开发一个算子 | "帮我开发一个 Abs 算子，输入 float16，输出 float16" | Plugin: [ops-direct-invoke（cannbot）](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke) |
| 调试精度问题 | "我的 Add 算子精度不达标，帮我排查一下" | Skill: `ascendc-precision-debug` |
| 查阅 API 文档 | "aclnnAdd 接口的参数和返回值是什么" | Skill: `ascendc-docs-search` |
| 检查开发环境 | "帮我检查一下当前的 CANN 开发环境" | Skill: `ascendc-env-check` |
| 代码检视 | "帮我检视这段 Kernel 代码是否符合规范" | Skill: `ascendc-code-review` |

> [!TIP]
> 完整安装可运行 `install-helper status` 查看已安装插件，独立 Skill 安装可运行 `npx skills list` 查看已安装 Skills。

> 更多使用样例详见 [Skills 使用样例](skills-usage.md)。

---

## 安装遇到问题？

运行 `install-helper doctor --fix` 自动检测并修复常见问题。

| 问题 | 解决方法 |
|------|---------|
| `install-helper` 报错 | npm/npx 方式请确认 Node.js >= 20：`node --version` |
| AI 工具无法识别 Skills | 重启工具或新开会话 |
| 软链接失效 | `install-helper doctor --fix` |
| project 级 Skill 在其他项目不可见 | project 级仅当前项目生效，跨项目请安装 global 级（`--level global`） |
| 客户端看不到 project 级 Skill | OpenCode / Claude Code 从启动目录向上查找，请在项目根目录（或子目录）启动 |
| Skill 在自定义目录 | Claude Code 用 `claude --add-dir <目录>` 加载；OpenCode 软链接到标准路径即可（install-helper 不支持自定义安装根路径） |
| 网络问题 | 配置 GitCode SSH Key 或设置代理；HTTPS git 被防火墙拦截时改走 SSH，见下方[防火墙拦截 HTTPS git](#防火墙拦截-https-git) |
| CANN 环境未配置 | 仅影响代码编译/运行类 Skills，知识检索类不受影响 |

#### 防火墙拦截 HTTPS git

HTTPS git 被拦截但 SSH 可用时，任选一种处理（首次克隆与缓存刷新均生效）：

**方式一：一条命令全局改写（推荐）**

```bash
git config --global url."git@gitcode.com:".insteadOf "https://gitcode.com/"
```

**方式二：只为 install-helper 设置环境变量**

```bash
export CANNBOT_REPO_URL=git@gitcode.com:cann/cannbot-skills.git      # skills 仓
export CANNBOT_MAINLINE_URL=git@gitcode.com:cann/cannbot.git         # cannbot 主线仓
```

方式二只影响 install-helper 托管的缓存（`~/.cannbot/`）；方式一改写本机所有 gitcode.com 的 HTTPS 拉取（含 cannbot 安装器内部 clone），已检出仓库的配置不受影响。

更多故障排查详见对应插件文档；已迁移的直调插件见 [cannbot 中的 ops-direct-invoke](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke)。
