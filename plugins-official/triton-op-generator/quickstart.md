# CANNBot Triton 算子生成快速入门指南

## 概述

CANNBot Triton-Ascend 算子生成模式适用于通过 Triton DSL 开发高性能 Ascend NPU 算子。采用 6 阶段工作流驱动，覆盖从任务构建到性能优化的完整生成流程，支持迭代修复与自动优化。

---

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

## Claude Code

### 安装

**方式一：install.sh 脚本（推荐，一键完成，含 CLAUDE.md 自动配置）**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/triton-op-generator
bash install.sh project claude     # 项目级
bash install.sh global claude      # 全局级
```

**方式二：Plugin Marketplace（技能自动管理）**

```bash
# 1. 注册 marketplace（首次，GitCode 仓库需完整 URL）
claude plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 2. 安装插件（安装 skills 依赖）
claude plugin install triton-op-generator@cannbot

# 3. 手动链接 AGENTS.md → CLAUDE.md（插件安装后必需步骤）
PLUGIN_DIR=$(ls -d ~/.claude/plugins/cache/cannbot/triton-op-generator/*/ 2>/dev/null | sort -V | tail -1)
ln -sf "${PLUGIN_DIR%/}/AGENTS.md" ~/.claude/CLAUDE.md
```

### 验证

```bash
# 查看已安装插件
claude plugin list
# 应看到 triton-op-generator@cannbot ✔ enabled

# 验证 CLAUDE.md 已正确链接
ls -la ~/.claude/CLAUDE.md
# 应显示为符号链接，指向插件缓存目录下的 AGENTS.md
```

### 启动

```bash
claude
```

### 更新

```bash
# install.sh 方式
cd cannbot-skills/plugins-official/triton-op-generator && bash install.sh

# Plugin Marketplace 方式
claude plugin update triton-op-generator@cannbot
# 更新后需重新链接 CLAUDE.md（缓存目录可能变化）
PLUGIN_DIR=$(ls -d ~/.claude/plugins/cache/cannbot/triton-op-generator/*/ 2>/dev/null | sort -V | tail -1)
ln -sf "${PLUGIN_DIR%/}/AGENTS.md" ~/.claude/CLAUDE.md
```

---

## OpenCode

### 安装

```bash
# 1. 克隆仓库
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/triton-op-generator

# 2. 运行安装脚本
bash install.sh project opencode   # 项目级（默认）
bash install.sh global opencode    # 全局级
```

### 验证

```bash
opencode agent list
# 应看到 triton-op-generator
```

### 启动

```bash
opencode
```

### 更新

```bash
cd cannbot-skills/plugins-official/triton-op-generator && bash install.sh
```

---

## TRAE

### 安装

```bash
# 1. 克隆仓库
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/triton-op-generator

# 2. 运行安装脚本（TRAE 仅支持项目级安装）
bash install.sh project trae
```

### 验证

检查项目目录下是否生成：
- `.trae/skills/` 目录，包含 6 个 skill 符号链接
- `CLAUDE.md` 符号链接

### 启动

通过 TRAE CLI 或 IDE 启动。

### 更新

```bash
cd cannbot-skills/plugins-official/triton-op-generator && bash install.sh project trae
```

## Cursor

### 安装

```bash
# 1. 克隆仓库
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/triton-op-generator

# 2. 运行安装脚本
bash install.sh project cursor     # 项目级
bash install.sh global cursor      # 全局级
```

### 验证

检查项目目录下是否生成：
- `.cursor/skills/` 目录，包含 6 个 skill 符号链接
- `AGENTS.md` 符号链接

### 启动

通过 Cursor IDE 启动。

### 更新

```bash
cd cannbot-skills/plugins-official/triton-op-generator && bash install.sh project cursor
```

## Copilot

### 安装

```bash
# 1. 克隆仓库
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/triton-op-generator

# 2. 运行安装脚本
bash install.sh project copilot    # 项目级
bash install.sh global copilot     # 全局级
```

### 验证

检查项目目录下是否生成：
- `.github/skills/` 目录（项目级）或 `~/.copilot/skills/` 目录（全局级），包含 6 个 skill 符号链接
- `AGENTS.md` 符号链接

### 启动

通过 VS Code Copilot CLI / IDE 启动。

### 更新

```bash
cd cannbot-skills/plugins-official/triton-op-generator && bash install.sh project copilot
```

---

## 安装路径说明

| 工具 | 安装方式 | 安装路径 | 说明 |
|------|---------|---------|------|
| Claude | Plugin Marketplace | `~/.claude/skills/` + 手动 `~/.claude/CLAUDE.md` | 技能自动安装，需手动链接 AGENTS.md |
| Claude | install.sh | `.claude/skills/` + `CLAUDE.md` | 一键完成，skills 和 CLAUDE.md 自动配置 |
| OpenCode | install.sh | `.opencode/skills/` + `AGENTS.md` | Skills 通过 symlink 安装，AGENTS.md 自动配置 |
| TRAE | install.sh | `.trae/skills/` + `CLAUDE.md` | 仅支持项目级安装 |
| Cursor | install.sh | `.cursor/skills/` + `AGENTS.md` | 项目级/全局级 |
| Copilot | install.sh | `.github/skills/` + `AGENTS.md`（项目级）/ `~/.copilot/skills/` + `AGENTS.md`（全局级） | 项目级/全局级 |

> **注意**：`claude plugin install` 仅安装技能（skills），不会自动创建 `CLAUDE.md`。因为 Claude Code 插件系统当前不支持安装后钩子（post-install hook），所以需要手动执行 `ln -s` 命令链接 `AGENTS.md`。

---

## 快速上手

### 生成算子示例

在交互界面中直接输入算子生成需求，CANNBot 会在当前会话中逐步执行 6 阶段工作流，**所有中间过程实时可见**。下面三个示例分别对应后文「输入模式说明」中的模式 A / B / C：

**示例 1：直接描述生成（对应模式 A ）**
```
生成一个 Triton-Ascend 框架的 softmax 算子实现，ASCEND_RT_VISIBLE_DEVICES=1，请将结果输出到 /path/output/
```

**示例 2：标准 torch 文件（对应模式 B）**
```
生成 Triton-Ascend 框架的算子，算子描述文件为 /path/NMS.py，ASCEND_RT_VISIBLE_DEVICES=1，请将结果输出到 /path/output/
```

**示例 3：GPU Kernel 输入（对应模式 C）**
```
基于 /path/gpu_softmax_kernel.py 中的 GPU Triton kernel 生成等价的 Triton-Ascend 实现，ASCEND_RT_VISIBLE_DEVICES=1，请将结果输出到 /path/output/
```

### 核心工作流

采用 6 阶段流水线，确保算子生成质量。所有阶段在当前会话中**实时显示**：

```
Phase 0: 参数确认 → Phase 1: 任务构建 → Phase 2: 算法设计
    → Phase 3: 代码生成与验证（迭代）→ Phase 4: 性能优化与验证（迭代）
    → Phase 5: 输出报告 → Phase 6: 会话导出
```

各阶段通过门禁校验后才能进入下一阶段。支持迭代修复和自动优化，详见工作流指令文档（Claude 为 `CLAUDE.md`，OpenCode 为 `AGENTS.md`）。

### 产出物示例

Triton-Ascend 算子生成模式下，CANNBot 会在工作目录下生成以下文件：

```
op_{op_name}_{timestamp}_{rid}/
├── {op_name}.py                          # Phase 1: 算子任务描述
├── {op_name}.json                        # Phase 1: 多 case 模式专属
├── sketch.txt                            # Phase 2: 算法草图
├── output/
│   ├── generated_code.py                 # Phase 3 最终通过验证的代码
│   ├── perf_result.json                  # Phase 3 最终性能报告
│   ├── optimized_code.py                 # Phase 4 最终优化代码（成功时）
│   ├── iter_0/                           # Phase 3 迭代记录
│   │   ├── generated_code.py
│   │   ├── verify/
│   │   │   ├── verify_result.json
│   │   └── perf_result.json
│   └── opt_iter_0/                       # Phase 4 优化记录
│       ├── optimized_code.py
│       └── ...
├── {op_name}_generated.py                # Phase 5: 最终代码
├── summary.json                          # 执行摘要
└── report.md                             # 最终报告
```

---

## 可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `triton-task-extractor` | 从用户代码中提取算子，构建算子任务格式任务文件 | Phase 1 |
| `triton-op-designer` | 设计高质量算法草图（sketch），指导代码生成 | Phase 2 |
| `triton-op-coding` | 根据任务描述生成 Triton Ascend 内核代码 | Phase 3 |
| `triton-op-verifier` | 验证代码正确性（精度比对）和性能测试 | Phase 3 / Phase 4 |
| `triton-latency-optimizer` | 逐步优化 Triton 代码性能 | Phase 4 |
| `npu-arch` | NPU 硬件架构参考（被 designer/coding 引用），区分 arch22/arch35 等规格 | Phase 2 / Phase 3 |

---

## 输入模式说明

### 模式 A：直接描述生成

用户直接描述算子需求（如"生成 softmax 算子"），系统在当前会话中自动：
1. 构建任务描述文件（`{op_name}.py`）
2. 设计算法草图
3. 生成并验证代码

### 模式 B：标准 torch文件（单 case / 多 case）

用户提供算子描述文件，系统调用 `triton-task-extractor` skill：
- **单 case**：`.py` 文件可包含 `get_inputs()`，返回单组输入
- **多 case**：`.py` 文件可包含 `get_input_groups()`，同目录有同名 `.json`

### 模式 C：GPU Kernel 输入模式

用户提供 GPU Triton kernel 源码（含 `@triton.jit`），系统自动检测并：
1. 构建 `Model` 类（返回预存 GPU 输出或手写 PyTorch 参考实现）
2. 提取输入数据
3. 生成 NPU Triton Kernel 等价实现

---

## 断点续跑与恢复

| 场景 | 使用方式 |
|------|---------|
| 查看历史结果 | 查看工作目录下的 `summary.json` 和 `report.md` |
| 重新生成 | 再次输入相同需求，系统会创建新的工作目录 |
| 查看迭代记录 | 查看 `output/iter_{N}/` 和 `output/opt_iter_{N}/` 目录 |

---

## 常见问题

### Q: 如何查看帮助信息？

```bash
bash install.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何指定输出目录？

在需求描述中明确指定：
```
生成 softmax 算子，输出到 /home/user/output/
```

### Q: 如何指定目标设备？

系统会自动检测 NPU 设备（通过 `npu-smi info -m`，避免解析主表格）。如需指定特定设备：
```bash
export ASCEND_RT_VISIBLE_DEVICES=1
```

### Q: 验证失败怎么办？

系统会自动进入迭代修复循环（最多 5 轮），所有修复过程**实时显示**：
1. 分析错误类型（A 类-代码逻辑 / B 类-环境 / C 类-重复失败）
2. 根据错误信息针对性修复
3. 重新验证

若达到最大迭代次数仍失败，会输出失败报告。

### Q: 性能优化做了什么？

Phase 4 会自动尝试以下优化（按严格顺序，每次只试一个），所有尝试**实时显示**：
- 向量化加载（tl.arange 替代标量访问）
- Grid 并行度优化（匹配物理核数）
- 内存访问模式优化（连续轴向量化）
- Pass 合并（减少数据遍历次数）
- 循环消除（小数据量场景）
