# CANNBot 代码检视快速入门指南

## 概述

CANNBot 代码检视模式适用于**Ascend C 算子代码检视**场景，采用"主 Agent 做大脑、子 Agent 做搜查"的分工模型，实现全量条例覆盖与高效并行检视。

### 与单独调用 ascendc-ops-reviewer 的区别

| 对比维度 | 本 Team（主 Agent + 子 Agent） | 单独调用 ascendc-ops-reviewer |
|---------|------------------------------|------------------------------|
| 检视架构 | 主 Agent 编排 + 子 Agent 并行执行 | 单 Agent 独立执行 |
| 条例覆盖 | 全量条例并行检视，确保 100% 覆盖 | 单次检视有限条例 |
| 报告生成 | 主 Agent 统一撰写，格式规范 | 单 Agent 直接输出 |
| 适用场景 | 大规模代码全量检视、PR 检视 | 快速检视指定条例 |
| 并行能力 | 多子 Agent 并行，波次调度 | 无并行能力 |

## 一、环境搭建

### 前置条件

- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### 操作步骤

#### 方式一：项目级安装（推荐）

在项目目录下安装，配置仅对当前项目生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入代码检视 Team 目录
cd cannbot-skills/plugins-official/ops-code-reviewer

# 3. 执行初始化脚本（项目级）
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
bash init.sh project trae       # TRAE 用户
bash init.sh project cursor     # Cursor 用户
```

#### 方式二：全局安装

在用户目录下安装，配置全局生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/cannbot-skills.git

# 2. 进入代码检视 Team 目录
cd cannbot-skills/plugins-official/ops-code-reviewer

# 3. 执行初始化脚本（全局）
bash init.sh global opencode    # OpenCode 用户（默认）
bash init.sh global claude      # Claude Code 用户
bash init.sh global trae        # TRAE 用户
bash init.sh global cursor      # Cursor 用户
```

### 安装内容

init.sh 脚本会完成以下操作：

| 内容 | OpenCode 项目级 | OpenCode 全局 | Claude 项目级 | Claude 全局 | TRAE 项目级 | TRAE 全局 |
|------|----------------|---------------|---------------|-------------|------------|----------|
| Skills 技能模块 | `.opencode/skills/ascendc-code-review/` | `~/.config/opencode/skills/ascendc-code-review/` | `.claude/skills/ascendc-code-review/` | `~/.claude/skills/ascendc-code-review/` | `.trae/skills/ascendc-code-review/` / `.marscode/skills/ascendc-code-review/` / `.traecli/skills/ascendc-code-review/` | `~/.trae-cn/skills/ascendc-code-review/` / `~/.marscode/skills/ascendc-code-review/` / `~/.traecli/skills/ascendc-code-review/` |
| Agents 子代理 | `.opencode/agents/ascendc-ops-reviewer.md` | `~/.config/opencode/agents/ascendc-ops-reviewer.md` | `.claude/agents/ascendc-ops-reviewer.md` | `~/.claude/agents/ascendc-ops-reviewer.md` | `.trae/agents/ascendc-ops-reviewer.md` / `.marscode/agents/ascendc-ops-reviewer.md` / `.traecli/agents/ascendc-ops-reviewer.md` | `~/.trae-cn/agents/ascendc-ops-reviewer.md` / `~/.marscode/agents/ascendc-ops-reviewer.md` / `~/.traecli/agents/ascendc-ops-reviewer.md` |
| 配置文件 | 项目根目录 `AGENTS.md` | `~/.config/opencode/AGENTS.md` | 项目根目录 `CLAUDE.md` | `~/.claude/CLAUDE.md` | 项目根目录 `AGENTS.md` | `~/.trae-cn/AGENTS.md` / `~/.marscode/AGENTS.md` / `~/.traecli/AGENTS.md` |

#### Cursor 安装路径

| 内容 | Cursor 项目级 | Cursor 全局级 |
|------|--------------|--------------|
| Skills 技能模块 | `.cursor/skills/ascendc-code-review/` | `~/.cursor/skills/ascendc-code-review/` |
| Agents 子代理 | `.cursor/agents/ascendc-ops-reviewer.md` | `~/.cursor/agents/ascendc-ops-reviewer.md` |
| 配置文件 | 项目根目录 `AGENTS.md` | `~/.cursor/AGENTS.md` |

#### Copilot 安装路径

| 内容 | Copilot 项目级 | Copilot 全局级 |
|------|---------------|---------------|
| Skills 技能模块 | `.github/skills/ascendc-code-review/` | `~/.copilot/skills/ascendc-code-review/` |
| Agents 子代理 | `.github/agents/ascendc-ops-reviewer.md` | `~/.copilot/agents/ascendc-ops-reviewer.md` |
| 配置文件 | 项目根目录 `AGENTS.md` | `~/.copilot/AGENTS.md` |

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/cannbot-skills/plugins-official/ops-code-reviewer/init.sh project claude

# 安装到指定项目
bash /path/to/cannbot-skills/plugins-official/ops-code-reviewer/init.sh project claude /path/to/your_project_path
```

### 环境校验

执行完上述步骤后，检查目录结构是否符合以下规范：

**项目级安装**：
```
cannbot-skills/plugins-official/ops-code-reviewer/
├── AGENTS.md                       # OpenCode 配置文件
├── .opencode/
│   ├── skills/                    # 技能模块
│   │   └── ascendc-code-review/   # 代码检视技能
│   ├── agents/                    # 子代理
│   │   └── ascendc-ops-reviewer.md
│   └── cannbot-manifest.json      # 安装清单
├── init.sh                        # 初始化脚本
└── quickstart.md                  # 本文档
```

**全局安装（OpenCode）**：
```
~/.config/opencode/
├── skills/                        # 技能模块
│   └── ascendc-code-review/       # 代码检视技能
├── agents/                        # 子代理
│   └── ascendc-ops-reviewer.md
├── AGENTS.md                      # Team 配置
└── cannbot-manifest.json          # 安装清单
```

**全局安装（Claude Code）**：
```
~/.claude/
├── skills/                        # 技能模块
│   └── ascendc-code-review/       # 代码检视技能
├── agents/                        # 子代理
│   └── ascendc-ops-reviewer.md
├── CLAUDE.md                      # Team 配置
└── cannbot-manifest.json          # 安装清单
```

**全局安装（TRAE）**：
```
~/.trae-cn/      # TRAE IDE 全局路径
~/.marscode/     # TRAE Plugin 路径（自动检测）
~/.traecli/      # TRAE CLI 路径（自动检测）
├── skills/                        # 技能模块
│   └── ascendc-code-review/       # 代码检视技能
├── agents/                        # 子代理
│   └── ascendc-ops-reviewer.md
├── AGENTS.md                      # Team 配置
└── cannbot-manifest.json          # 安装清单
```

**项目级安装（TRAE）**：
```
cannbot-skills/plugins-official/ops-code-reviewer/
├── AGENTS.md                      # Team 配置
├── .trae/          # TRAE IDE 项目路径
├── .marscode/      # TRAE Plugin 路径（自动检测）
├── .traecli/       # TRAE CLI 路径（自动检测）
│   ├── skills/                    # 技能模块
│   │   └── ascendc-code-review/   # 代码检视技能
│   ├── agents/                    # 子代理
│   │   └── ascendc-ops-reviewer.md
│   └── cannbot-manifest.json      # 安装清单
├── init.sh                        # 初始化脚本
└── quickstart.md                  # 本文档
```

**项目级安装（Cursor）**：
```
cannbot-skills/plugins-official/ops-code-reviewer/
├── AGENTS.md                      # Team 配置
├── .cursor/
│   ├── skills/                    # 技能模块
│   │   └── ascendc-code-review/   # 代码检视技能
│   ├── agents/                    # 子代理
│   │   └── ascendc-ops-reviewer.md
│   └── cannbot-manifest.json      # 安装清单
├── init.sh                        # 初始化脚本
└── quickstart.md                  # 本文档
```

## 二、快速上手

### 启动

在初始化完成的目录下执行：

```bash
opencode   # OpenCode 用户
claude     # Claude Code 用户
```
> **TRAE 用户**：TRAE 通过 IDE、VS Code 插件或 CLI 启动。init.sh 会自动检测 TRAE IDE（`~/.trae-cn`）、Plugin（`~/.marscode`）或 CLI（`~/.traecli`）并安装到对应目录。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。

### 检视示例

在交互界面中输入检视需求，CANNBot 会自动启动 4 阶段检视流程：

```
检视算子文件：moe_init_routing/op_kernel/moe_init_routing.h
```

或检视 PR：

```
检视 PR：https://gitcode.com/cann/ops-transformer/pull/3604
```

### 核心工作流

采用 4 阶段流程，确保检视质量：

```
阶段1：识别侧别 + 提取条例 → 阶段2：分组与派发子 Agent → 阶段3：行号校对 → 阶段4：撰写报告
```

每个阶段完成后才能进入下一阶段，详见 AGENTS.md。

### 产出物

检视报告路径：
- 文件检视：`./operators/{operator_name}/{source_file}_review_summary.md`
- PR 检视：`./operators/pr-{pr_number}/{pr_number}_review_summary.md`

## 三、检视模式

| 模式 | 触发词 | 检视范围 | 特点 |
|------|--------|----------|------|
| **单文件检视** | `检视算子文件：xxx` | 全量条例 | 主 Agent 自动识别代码侧别，过滤适用条例 |
| **PR 检视** | `检视 PR：xxx` | 全量条例 | 仅检视变更部分，支持 GitCode PR |

### 代码侧别自动识别

| 代码侧别 | 文件特征 | 适用条例 |
|---------|---------|---------|
| **Kernel 侧** | `op_kernel/*.cpp`, `*.asc`, 含 `AscendC::` API | `[适用: All]` |
| **Tiling 侧** | `op_host/*.cpp`, 文件名含 `tiling`/`infershape` | `[适用: All]` + `[适用: Tiling]` |

## 四、检视文档体系

主 Agent 自动选择适用的检视文档：

| 文档名称 | 适用侧别 | 核心检视内容 |
|---------|---------|-------------|
| **C++ 安全编码规范** | All | 数值安全、内存安全、输入验证 |
| **Ascend C API 最佳实践** | Kernel | API 黑名单、对齐要求、配对检查 |
| **Ascend C 高性能编程** | Kernel | 循环优化、DoubleBuffer、精度保护 |
| **TOPK 高频问题** | All/Host/Kernel | 野指针、特殊值处理、返回值校验 |

## 五、可用技能

| Skill | 用途 | 触发时机 |
|-------|------|---------|
| `ascendc-code-review` | 代码检视方法论、假设检验驱动 | 所有检视任务 |

| Agent | 用途 | 角色 |
|-------|------|------|
| `ascendc-ops-reviewer` | 条款级检视执行 | 子 Agent（搜查） |

## 六、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新技能模块？

重新执行 init.sh 即可，脚本会自动覆盖旧版本符号链接。

### Q: 如何增减检视条例？

全局安装和项目安装均采用软链接方式，检视条例定义在 CANNBot 仓库的 `cannbot-skills/skills/ascendc-code-review/references/` 目录下。

修改步骤：
1. 直接编辑 CANNBot 仓库中的规范文档（如 `cpp-secure.md`、`ascendc-api.md`）
2. 重启 OpenCode 或 Claude Code，修改即刻生效

### Q: Team 检视和单独调用 reviewer Agent 如何选择？

| 场景 | 推荐方式 |
|------|---------|
| 大规模代码全量检视 | Team 检视（主 Agent + 子 Agent） |
| 快速检视指定条例 | 单独调用 ascendc-ops-reviewer |
| PR 检视需要完整报告 | Team 检视 |
| 仅需逐条检视结果 | 单独调用 ascendc-ops-reviewer（快速检视模式） |

### Q: 检视报告保存在哪里？

- 文件检视：`./operators/{operator_name}/{source_file}_review_summary.md`
- PR 检视：`./operators/pr-{pr_number}/{pr_number}_review_summary.md`

---

## 总结

1. 代码检视 Team 采用"主 Agent 做大脑、子 Agent 做搜查"架构，实现高效并行检视
2. 环境搭建核心两步：克隆仓库 → 执行 init.sh
3. `opencode` / `claude` / `trae` / `cursor` 是核心交互指令
4. 4 阶段工作流确保全量条例覆盖和报告质量
5. 自动识别代码侧别，精准过滤适用条例
