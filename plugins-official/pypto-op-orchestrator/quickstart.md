# CANNBot PyPTO 算子开发快速入门指南

## 概述

CANNBot PyPTO 算子开发模式适用于通过 PyPTO 开发自定义算子。采用 7 阶段状态机驱动，覆盖从需求理解到性能调优的完整开发流程，支持断点续跑与失败恢复。

### 与 Kernel 直调开发的区别

| 对比维度 | PyPTO 算子开发（本模式） | Kernel 直调开发 |
|---------|------------------------|----------------|
| 适用场景 | PyPTO 框架算子开发 | Ascend C Kernel 直接开发 |
| 编程语言 | Python（PyPTO API） | C++（Ascend C API） |
| 开发内容 | PyPTO kernel + golden + test | Kernel + Tiling + Host 验证代码 |
| 阶段数 | 7 阶段（含 API 探索和 Golden 生成） | 7 步骤（含设计串讲和修复循环） |
| 精度验证 | 三态标记自动判定 | Reviewer 独立构建验证 |
| 性能调优 | 自动迭代调优（最多 10 轮） | Developer 手动采集 |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装 PyPTO，版本需与 CANN 配套。通过 PyPI 安装时，CANN 与 PyPTO 版本对应关系查阅 [PyPI 安装](https://pypto.gitcode.com/install/build_and_install.html#pypi)；CANN 9.1.0 版本推荐使用源码编译安装，参阅 [源码编译安装](https://pypto.gitcode.com/install/build_and_install.html)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install pypto-op-orchestrator@cannbot
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### TRAE

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

### Cursor

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

### Copilot

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project claude

# 安装到指定项目
bash /path/to/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project claude /path/to/your_project_path
```

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 pypto-op-orchestrator@cannbot ✔ enabled
ls .
# 应看到 CLAUDE.md 位于项目根目录

# OpenCode
opencode agent list
# 应看到 pypto-op-analyst / pypto-op-developer / pypto-op-perf-tuner

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

在交互界面中输入算子开发需求，CANNBot 会自动启动 7 阶段流程：

```
帮我开发一个 softmax 算子，支持 float16 数据类型，shape 主要是 [1,128]、[4,2048]、[32,4096]
```

### 核心工作流

采用 7 阶段状态机，确保算子开发质量：

```
Stage 1: 需求理解 → Stage 2: API 探索 → Stage 3: Golden 生成
    → Stage 4: 设计 → Stage 5: 代码实现 → Stage 6: 精度修复（按需）
    → Stage 7: 性能调优
```

每个阶段完成门禁校验后才能进入下一阶段。支持断点续跑和失败恢复，详见 AGENTS.md。

### 产出物示例

PyPTO 算子开发模式下，CANNBot 会在 `custom/{operator}/` 目录下生成以下文件：

```
custom/softmax/
├── SPEC.md                    # 需求规格
├── API_REPORT.md              # API 可行性报告
├── DESIGN.md                  # 设计文档
├── softmax_golden.py          # PyTorch 参考实现
├── softmax_impl.py            # PyPTO kernel 实现
├── test_softmax.py            # 测试入口
├── README.md                  # 实现说明
├── .orchestrator_state.json   # 流程状态（自动维护）
└── history_version/           # 版本备份
```

## 三、可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-api-explore` | API 可行性探索与分析 | Stage 2 |
| `pypto-golden-generate` | Golden 参考实现生成 | Stage 3 |
| `pypto-op-design` | 算子设计方案生成 | Stage 4 |
| `pypto-op-develop` | 算子代码实现与测试 | Stage 5 |
| `pypto-precision-debug` | 精度问题代码层排查 | Stage 6 |
| `pypto-precision-compare` | 精度中间结果对比分析 | Stage 6（辅助） |
| `pypto-op-perf-tune` | 算子性能分析与自动调优 | Stage 7 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-op-analyst` | Golden 生成与设计 | Stage 3-4 |
| `pypto-op-developer` | 代码实现与精度修复 | Stage 5-6 |
| `pypto-op-perf-tuner` | 性能分析与调优 | Stage 7 |

## 四、断点续跑与恢复

CANNBot 通过 `.orchestrator_state.json` 维护全局状态，支持：

| 场景 | 使用方式 |
|------|---------|
| 中断后继续 | 再次输入算子名，自动从上次中断处续跑 |
| 失败后重试 | 输入"继续开发 {算子名}"，从失败阶段恢复 |
| 查看状态 | 查看 `custom/{op}/.orchestrator_state.json` |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
# Claude Code
/plugin update pypto-op-orchestrator@cannbot

# OpenCode (init.sh 方式)
cd cannbot-skills/plugins-official/pypto-op-orchestrator && bash init.sh
```

### Q: PyPTO 和 Kernel 直调模式如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO 算子开发 |
| 使用 Ascend C API 开发算子 | Kernel 直调开发 |
| 快速验证算子可行性 | PyPTO 算子开发 |
| 需要精细控制硬件资源 | Kernel 直调开发 |
| 原型开发和概念验证 | PyPTO 算子开发 |
| 生产级高性能算子 | Kernel 直调开发 |

---

## 总结

1. PyPTO 算子开发模式通过 7 阶段状态机实现端到端自动化
2. Claude Code 用户用 `/plugin install` 一键安装，OpenCode/TRAE/Cursor 用户用 `init.sh` 脚本安装
3. `claude` / `opencode` 是核心交互指令
4. 所有阶段通过门禁驱动，支持断点续跑与失败恢复
5. 产出物包含完整的参考实现、设计文档、PyPTO 实现和测试入口
