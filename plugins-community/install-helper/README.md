# install-helper

> CANNBot Install Helper — CANNBot 交互式安装助手

[![npm version](https://img.shields.io/npm/v/@cannbot-ai/install-helper.svg)](https://www.npmjs.com/package/@cannbot-ai/install-helper)

## 快速开始

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

---

安装完成后运行 `install-helper` 启动向导。

## 一键安装所有 Skills

```bash
# 安装全部 Skills 到当前项目
install-helper install --all --tool opencode --yes

# 安装全部 Skills 到全局
install-helper install --all --tool opencode --level global --yes
```

## 命令一览

| 命令 | 说明 |
|------|------|
| `install-helper` | 交互式安装向导 |
| `install-helper list` | 列出可用场景及安装状态 |
| `install-helper install <name>` | 安装指定插件或 Skill（自动识别） |
| `install-helper install --list` | 按类别列出所有可用 Skills |
| `install-helper install --all --tool <tool> --yes` | 一键安装全部 Skills |
| `install-helper update [plugin]` | 更新已安装的插件 |
| `install-helper uninstall <plugin>` | 卸载指定插件 |
| `install-helper uninstall` | 交互式批量卸载（checkbox 勾选） |
| `install-helper uninstall --all` | 卸载全部已安装 Skills 和 Plugins |
| `install-helper uninstall --recent` | 卸载最近一次安装批次的 Skills |
| `install-helper status` | 查看已安装插件详情 |
| `install-helper doctor` | 健康检查 |
| `install-helper doctor --fix` | 健康检查 + 自动修复 |
| `install-helper info <plugin>` | 查看插件详情 |
| `install-helper lang set en_US` | 切换语言 |

## 命令详解

### 交互式安装向导

```bash
install-helper
```

向导引导：选择类型（Plugin/Skill）→ 检测工具 → 选择位置 → 选择插件/Skill → 确认安装。

已安装的插件会自动标记，上次选择的工具和级别会被记住。

### 安装指定插件

```bash
install-helper install ops-direct-invoke              # 安装单个
install-helper install triton --tool claude           # 指定工具
install-helper install ops-direct-invoke --yes        # 跳过确认
install-helper install ops-direct-invoke --level global  # 全局安装
```

> 每个插件有独立的 `AGENTS.md`，建议每个项目只安装一个插件。

### 安装单个 Skill

```bash
install-helper install npu-arch                          # 安装 NPU 架构知识
install-helper install ascendc-precision-debug           # 安装精度调试技能
install-helper install --list                            # 查看所有可用 Skills
```

`install` 命令自动识别输入是插件还是 Skill。

交互式选择时支持：全选所有 Skills、全选当前类别、跨类别累积选择。

**选项：**

| 选项 | 说明 |
|------|------|
| `--tool <tool>` | 指定 AI 工具（opencode, claude, trae, cursor, copilot, codearts） |
| `--level <level>` | 安装级别（project, global），默认 project |
| `--all` `-a` | 安装全部 Skills |
| `--list` | 按类别列出所有可用 Skills |
| `--yes` `-y` | 跳过所有确认提示 |

### 更新

```bash
install-helper update              # 更新所有已安装插件
install-helper update triton       # 更新指定插件
```

### 卸载

```bash
install-helper uninstall ops-direct-invoke --tool opencode  # 卸载指定插件
install-helper uninstall npu-arch --yes                     # 跳过确认卸载 Skill
install-helper uninstall                                     # 交互式批量卸载（checkbox 勾选）
install-helper uninstall --all --yes                         # 一键卸载全部已安装内容
install-helper uninstall --recent --yes                      # 卸载最近安装的 Skills 批次
install-helper uninstall --all --tool opencode --level project  # 指定范围卸载
```

根据安装记录精确删除所有产物，自动清理空目录。

**交互式卸载**：

```bash
install-helper uninstall
```

- 自动检测已安装内容，跳过只有单一选项的步骤（如仅一个工具/级别有内容时自动选中）
- 若 Skills 和 Plugins 都已安装，先选择卸载类型
- ≤15 个已安装 Skill 时显示扁平列表，>15 个时按类别导航
- **勾选=卸载**（标准 checkbox 语义），勾选后确认即可批量卸载
- 支持跨类别累积选择

**`--all` 一键卸载**：

```bash
install-helper uninstall --all --yes
```

卸载指定范围内全部已安装的 Skills 和 Plugins。通过 `--tool` / `--level` 限定范围，未指定时默认 project 级别。

**`--recent` 批次卸载**：

```bash
install-helper uninstall --recent --yes
```

仅卸载最近一次安装批次中的 Skills。每次 `install-helper install` 会自动记录安装批次，`--recent` 精确卸载最后一批，不影响更早安装的内容。

**安全机制**：

- 仅删除 `install-helper` 安装的 symlink，不影响用户手动创建的目录或文件
- 无安装记录时自动回退到文件系统扫描（识别 symlink）和 manifest 文件，确保残留内容可清理
- Plugin 卸载时完整清理 skills、agents、配置文件（AGENTS.md/CLAUDE.md）、manifest、外部仓库链接

**选项：**

| 选项 | 说明 |
|------|------|
| `--tool <tool>` | 指定 AI 工具（opencode, claude, trae, cursor, copilot, codearts） |
| `--level <level>` | 安装级别（project, global），默认 project |
| `--yes` `-y` | 跳过确认提示 |
| `--all` `-a` | 卸载全部已安装 Skills 和 Plugins |
| `--recent` | 卸载最近一次安装批次的 Skills |

### 健康检查

```bash
install-helper doctor              # 仅检测
install-helper doctor --fix        # 检测 + 自动修复
```

### 语言管理

```bash
install-helper lang show           # 显示当前语言
install-helper lang set en_US      # 切换为英文
```

## 可用插件

| # | 场景 | ID | 别名 |
|---|------|----|------|
| 1 | AscendC Kernel 直调 | ops-direct-invoke | ops-direct, ascendc-direct, direct, kernel |
| 2 | AscendC Kernel 直调（快速版） | ops-direct-invoke-flash | flash, kernel-flash |
| 3 | AscendC 算子注册调用 | ops-registry-invoke | ops-registry, ascendc-registry, registry |
| 4 | Catlass 算子直调 | catlass-op-generator | catlass |
| 5 | NPU 推理优化 | model-infer-optimize | model-infer, infer, inference |
| 6 | PyPTO 算子开发 | pypto-op-orchestrator | pypto, pytorch |
| 7 | TileLang 算子开发 | tilelang-op-orchestrator | tilelang |
| 8 | Triton 算子开发 | triton-op-generator | triton |
| 9 | torch.compile 图模式 | torch-compile | torch, compile, graph |
| 10 | 代码检视 | ops-code-reviewer | code-review, reviewer, review |
| 11 | NPU 推理 SOTA 优化 | model-infer-sota-approach | sota, sota-approach, model-infer-sota |

## 前置条件

- 已安装至少一个 AI 编程工具（OpenCode / Claude Code / Trae / Cursor / GitHub Copilot / CodeArts）
- Git（用于克隆 Skills 仓库）

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
