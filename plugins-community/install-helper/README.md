# install-helper

> CANNBot Install Helper — CANNBot 交互式安装助手

[![npm version](https://img.shields.io/npm/v/@cannbot-ai/install-helper.svg)](https://www.npmjs.com/package/@cannbot-ai/install-helper)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org)

## 快速开始

```bash
# 全局安装（推荐）
npm install -g @cannbot-ai/install-helper
install-helper

# 或 npx 即用（无需安装）
npx @cannbot-ai/install-helper
```

交互式向导会自动检测已安装的 AI 编程工具（OpenCode / Claude Code / Trae / Cursor），通过箭头键选择场景即可完成安装。

## 主要特性

### 核心功能
- **动态 Skill 发现**：自动扫描仓库中的 SKILL.md 文件，实时发现可用 Skills
- **智能缓存机制**：扫描结果缓存 24 小时，大幅提升后续命令执行速度
- **备份与恢复**：安装新插件时自动备份 AGENTS.md，支持一键恢复
- **参数校验**：`--tool` 和 `--level` 参数支持校验，提供友好的错误提示（支持中英文）
- **智能记忆**：记住上次选择的工具和安装级别，下次运行自动填充

### 安装管理
- **交互式向导**：箭头键选择，空格切换，回车确认
- **插件安装**：支持 10 个官方插件，自动处理依赖
- **Skill 安装**：支持 89 个 Skills，按 11 个类别组织
- **智能卸载**：根据安装记录精确删除，自动清理空目录
- **健康检查**：检测安装状态，自动修复常见问题

## 命令一览

| 命令 | 说明 |
|------|------|
| `install-helper` / `install-helper init` | 交互式安装向导 |
| `install-helper list` | 列出可用场景及安装状态 |
| `install-helper install <name>` | 安装指定插件或 Skill（自动识别） |
| `install-helper install --list` | 按类别列出所有可用 Skills |
| `install-helper update [plugin]` | 更新已安装的 Skills |
| `install-helper uninstall <plugin>` | 卸载指定场景 |
| `install-helper status` | 查看已安装插件详情 |
| `install-helper doctor` | 健康检查 |
| `install-helper doctor --fix` | 健康检查 + 自动修复 |
| `install-helper info <plugin>` | 查看插件详情 |
| `install-helper lang show` | 显示当前语言 |
| `install-helper lang set en_US` | 切换语言 |

## 命令详解

### 交互式安装向导

```bash
install-helper
# 或
install-helper init
```

向导会引导你完成：
1. 自动检测已安装的 AI 编程工具
2. 选择安装级别（project / global）
3. 多选要安装的场景（空格切换，回车确认）
4. 确认并执行安装

**智能记忆**：向导会记住你上次选择的工具和级别，下次运行无需重新选择。已安装的插件会自动标记并默认不勾选。

### 安装指定场景

```bash
install-helper install ops-direct-invoke              # 安装单个
install-helper install triton --tool claude           # 指定工具
install-helper install ops-direct-invoke --yes        # 跳过确认
install-helper install ops-direct-invoke --level global  # 全局安装
```

**重要说明**：

每个插件都有独立的 `AGENTS.md` 配置文件。如果在同一目录下安装多个插件，后安装的会覆盖前一个的配置文件。

**推荐做法**：
- 每个项目只安装一个插件
- 如需使用多个插件，在不同目录下分别安装

**示例**：
```bash
# 项目 A：使用 Kernel 直调
cd project-a
install-helper install ops-direct-invoke --tool opencode

# 项目 B：使用 Triton
cd project-b
install-helper install triton --tool opencode
```

> **注意**：每个插件有独立的 `AGENTS.md` 配置文件。同一项目目录下安装多个插件时，后安装的会覆盖前一个的 `AGENTS.md`。如需同时使用多个插件，建议分别在不同项目目录中安装。

### 安装单个 Skill（轻量模式）

如果只需要让 AI agent 掌握特定领域知识，不需要完整工作流，可以直接安装单个 Skill：

```bash
install-helper install npu-arch                          # 安装 NPU 架构知识
install-helper install ascendc-precision-debug           # 安装精度调试技能
install-helper install model-infer-kvcache               # 安装 KVCache 优化技能
install-helper install --list                            # 查看所有可用 Skills
```

Skills 按 11 个大类组织：知识与参考、环境与工具、调试与诊断、测试与质量、AscendC 开发、PyPTO 开发、TileLang 开发、Triton 开发、模型推理优化、图模式、平台工具。

`install` 命令自动识别输入是插件还是 Skill，无需区分。

**选项：**

| 选项 | 说明 |
|------|------|
| `--tool <tool>` | 指定 AI 工具（opencode, claude, trae, cursor, copilot） |
| `--level <level>` | 安装级别（project, global），默认 project |
| `--all` | 安装全部场景 |
| `--yes` | 跳过所有确认提示 |

**已装检测**：如果插件已安装，会提示是否覆盖安装。使用 `--yes` 可跳过提示直接覆盖。

### 更新已安装的 Skills

```bash
install-helper update              # 更新所有已安装插件
install-helper update triton       # 更新指定插件
```

更新流程：`git pull` 拉取最新 Skills 仓库 → 重新执行安装脚本 → 软链接自动指向最新内容。

### 查看插件详情

```bash
install-helper info ops-direct-invoke
```

输出插件描述、包含的 Skills/Agents 列表、quickstart 文档路径、安装状态。

### 健康检查

```bash
install-helper doctor              # 仅检测
install-helper doctor --fix        # 检测 + 自动修复
```

检查项目：
- AI 工具安装状态
- 已安装插件状态
- 软链接完整性
- 配置文件存在性

`--fix` 会自动修复：清理失效软链接、重建缺失目录。

### 语言管理

```bash
install-helper lang show           # 显示当前语言
install-helper lang set zh_CN      # 设置为中文（默认）
install-helper lang set en_US      # 设置为英文
```

语言设置保存在 `~/.cannbot/config.yaml`，重启后自动生效。

### 卸载

```bash
install-helper uninstall ops-direct-invoke --tool opencode
```

卸载时根据安装记录（`~/.cannbot/installs/<plugin>.json`）精确删除所有安装产物：
- Skills 软链接
- Agents 软链接
- Workflows 链接
- 配置文件（AGENTS.md / CLAUDE.md）
- Manifest 文件
- 参考仓库链接（asc-devkit 等）
- 空目录清理

## 可用场景

install-helper 支持 **10 个官方插件** 和 **89 个 Skills**（动态发现，数量可能随仓库更新而变化）。

#### 官方插件

| # | 场景 | ID | 别名 |
|---|------|----|------|
| 1 | AscendC Kernel 直调 | ops-direct-invoke | ops-direct, direct, kernel |
| 2 | AscendC Kernel 从零构建 | ops-direct-invoke-flash | flash |
| 3 | AscendC 算子注册调用 | ops-registry-invoke | ops-registry, registry |
| 4 | PyPTO 算子 | pypto-op-orchestrator | pypto |
| 5 | Triton 算子生成 | triton-op-generator | triton |
| 6 | TileLang 算子 | tilelang-op-orchestrator | tilelang |
| 7 | NPU 推理优化 | model-infer-optimize | model-infer, infer |
| 8 | Catlass 算子直调 | catlass-op-generator | catlass |
| 9 | 代码检视 | ops-code-reviewer | code-review, reviewer |
| 10 | torch.compile 图模式 | torch-compile | torch, compile |

#### Skills 类别

Skills 按 11 个类别组织（动态发现，数量可能随仓库更新而变化）：
- 知识与参考
- 环境与工具
- 调试与诊断
- 测试与质量
- AscendC 开发
- PyPTO 开发
- TileLang 开发
- Triton 开发
- 模型推理优化
- 图模式
- 平台工具

使用 `install-helper install --list` 查看完整的 Skills 列表。

## 前置条件

- Node.js >= 18
- 已安装至少一个 AI 编程工具（OpenCode / Claude Code / Trae / Cursor）
- Git（用于克隆 Skills 仓库）

## 工作原理

install-helper 是一个编排工具：

1. **自动检测**已安装的 AI 编程工具
2. **动态扫描**仓库中的 SKILL.md 文件，发现可用 Skills
3. **缓存扫描结果**到 `~/.cannbot/scan-cache.json`（24 小时有效）
4. **克隆或更新** cannbot-skills 仓库（缓存到 `~/.cannbot/repo/`）
5. **备份现有配置**（如果检测到已安装的 AGENTS.md）
6. **调用**各插件的 `init.sh` / `install.sh` 脚本执行实际安装
7. **追踪**安装状态（`~/.cannbot/config.yaml` + manifest）

安装逻辑仍由各插件的 init.sh 脚本负责，install-helper 提供统一的交互式入口和状态管理。

## 配置文件

配置保存在 `~/.cannbot/` 目录：

```yaml
# config.yaml - 用户配置
language: zh_CN           # 界面语言
lastTool: opencode        # 上次使用的 AI 工具
lastLevel: project        # 上次安装级别
installedPlugins:         # 已安装的插件列表
  - ops-direct-invoke
  - pypto-op-orchestrator
```

```json
// scan-cache.json - 扫描缓存（自动生成，24 小时有效）
{
  "skills": [...],
  "timestamp": 1234567890,
  "repoPath": "/path/to/repo"
}
```

```json
// installs/<plugin>.json - 安装记录（每个插件一个文件）
{
  "pluginId": "ops-direct-invoke",
  "installTime": "2026-06-23T10:00:00Z",
  "configRoot": "/path/to/config",
  "files": [...],
  "directories": [...]
}
```

## 开发

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-community/install-helper

npm install
npm run dev          # watch 模式
npm run build        # 构建
npm link             # 全局注册
install-helper list  # 测试
```


