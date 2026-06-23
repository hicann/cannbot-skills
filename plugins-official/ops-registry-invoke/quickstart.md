# CANNBot 注册调用模式快速入门指南

## 概述

CANNBot 注册调用模式适用于**生产级自定义算子开发**场景，通过完整的开发流程生成符合算子仓规范的算子代码，包含 aclnn 接口层、Kernel 实现、UT/ST 测试用例等，适合向算子仓贡献代码、企业级算子开发等场景。

### 与直调模式的区别

| 对比维度 | 注册调用（本模式） | 直调模式 |
|---------|-------------------|---------|
| 适用场景 | 生产级算子、算子仓贡献 | 快速验证、原型开发、学习研究 |
| 开发内容 | 完整 aclnn 接口 + Kernel + UT/ST 测试用例 | Kernel + Tiling + Host 验证代码 |
| 调用方式 | 通过 aclnn API 注册调用 | Host 端 `<<<>>>` 直调 Kernel |
| 目录结构 | 算子仓标准目录结构 | 独立工程目录 |
| 开发周期 | 长（天级） | 短（小时级） |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### Claude Code

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-registry-invoke
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-registry-invoke
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### TRAE

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-registry-invoke
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

### Cursor

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-registry-invoke
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

### Copilot

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-registry-invoke
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/cannbot-skills/plugins-official/ops-registry-invoke/init.sh project claude

# 安装到指定项目
bash /path/to/cannbot-skills/plugins-official/ops-registry-invoke/init.sh project claude /path/to/your_project_path
```

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到已安装的 skills 和 agents
ls .
# 应看到 CLAUDE.md 位于项目根目录

# OpenCode
opencode agent list
# 应看到 ascendc-ops-architect / ascendc-ops-developer / ascendc-ops-tester

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Cursor
ls .cursor/
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录
```

## 二、快速上手

### 启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

> **TRAE 用户**：TRAE 通过 IDE、VS Code 插件或 CLI 启动。init.sh 会自动检测 TRAE IDE（`~/.trae-cn`）、Plugin（`~/.marscode`）或 CLI（`~/.traecli`）并安装到对应目录。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。

### 开发算子示例

在交互界面中输入算子开发需求，CANNBot 会自动加载工作流技能并指导开发：

```
帮我生成一个AddCustom算子，适配 Ascend 910 芯片架构，支持 float16/bfloat16/float32 数据类型
```

### 核心工作流

CANNBot 采用四阶段开发工作流，确保算子开发质量：

```
阶段一：需求与设计
    ↓ ⛔CP1确认
阶段二：开发（双轨道并行：算子代码 + ST用例）
    ↓ ⛔CP2确认
阶段三：验收（性能，可选）
    ↓ ⚪CP3确认
阶段四：上库（代码检视 + 开发总结）
```

每个阶段完成后才能进入下一阶段，详见 AGENTS.md。

### 产出物示例

注册调用模式下，CANNBot 会在指定目录下生成符合规范的完整算子：

```
operators/add_custom/
├── CMakeLists.txt              # 编译配置文件
├── op_host/                    # Host 代码层
│   ├── add_custom_def.cpp      # 算子定义
│   ├── add_custom_infershape.cpp # Shape推导
│   └── arch22/                 # 芯片架构适配 (Ascend 910)
├── op_kernel/                  # Kernel 实现层
│   ├── add_custom.h
│   ├── add_custom_arch22.cpp   # Ascend 910 Kernel实现
│   └── arch22/                 # 芯片架构适配
└── tests/                      # 测试目录
    ├── ut/                     # 单元测试
    └── st/                     # 系统测试
```

## 三、可用技能

| Subagent | 用途 | 触发时机 |
|----------|------|---------|
| `ascendc-ops-architect` | 需求分析、方案设计 | 阶段一 |
| `ascendc-ops-developer` | 算子代码开发、UT 开发 | 阶段二 |
| `ascendc-ops-tester` | ST 用例开发、测试验收 | 阶段二、阶段三 |
| `ascendc-code-review`（skill） | 全量代码检视 + 设计一致性检查 | 阶段四 |
| `ascendc-docs-search` | 文档资源索引 | 查找 API 文档和示例 |
| `ascendc-env-check` | 环境检查 | NPU 设备查询 |

## 四、开发资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| Task 调用参数 | `workflow/resources/task-prompts.md` | 各阶段 Subagent 详细调用参数 |
| 数据流说明 | `workflow/resources/data-flow.md` | 各阶段输入输出文件说明 |
| 错误处理指南 | `workflow/resources/error-handling.md` | 各阶段错误类型、回退策略 |

## 五、常见问题

### Q: 如何更新技能模块？

重新执行 init.sh 即可，脚本会自动覆盖旧版本：

```bash
cd cannbot-skills/plugins-official/ops-registry-invoke && bash init.sh
```

### Q: 注册调用模式和直调模式如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 快速验证算法思路 / 学习 Ascend C | 直调模式 |
| 生产级算子开发 / 向算子仓贡献代码 | 注册调用模式 |
