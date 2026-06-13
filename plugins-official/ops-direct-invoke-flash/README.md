# ops-direct-invoke-flash

Ascend C / Ascend950 Reg API Kernel **从零构建** Skill。从 CPU 函数、数学公式、代码片段或文本描述出发，端到端产出一个生产级的华为昇腾 NPU 核函数。

> 与 Team 版 [`ops-direct-invoke`](../ops-direct-invoke) 的区别：本插件以**一个自包含 Skill**为核心，仅附带一个评审子 Agent（`ops-direct-invoke-flash-reviewer`），工作流与 Hooks 全部内置、无需依赖共享 `ops/` Skills，上手更快（Flash）。首次安装时会拉取两个参考仓库（`asc-devkit`、`cann-samples`）作为 API 文档与算子样例；离线环境下会跳过并告警，不影响 Skill 安装。

**核心价值**：

| 维度 | 能力 |
|------|------|
| **文档先行** | 先写定义文档与设计文档，子 Agent 评审通过后再写代码 |
| **分阶段实现** | 以小步增量方式实现核函数，每步本地构建/测试 |
| **双重验证** | 本地构建 + 真实 NPU 硬件远程验证 |
| **Reg API 支持** | Ascend950 / `dav-3510` 默认生成原生 `AscendC::Reg` 计算代码 |
| **进度持久化** | `docs/{OP}/STATE.md` 作为 git 跟踪的唯一可信进度来源 |
| **失败防护** | 内置 Hooks 与常见失败模式清单，提前拦截典型错误 |

---

## QUICKSTART

```bash
# Step 1: 安装 Skill（在本插件目录下执行；默认 project 级 + OpenCode）
bash init.sh                       # 等价于 init.sh project opencode
bash init.sh project claude        # Claude Code，安装到当前项目 .claude/skills/

# Step 2: 启动工具并运行命令
/ops-direct-invoke-flash 帮我实现一个 abs 算子，float16，shape [1,128]/[4,2048]

# 也可以从源文件出发
/ops-direct-invoke-flash ./reference/abs_cpu.cpp
```

`$ARGUMENTS` 可以是：文件路径（C++、PyTorch、Numpy）、数学公式、规格说明文档或文字描述。若为空，Skill 会先请你补充算子来源。

---

## 安装

`init.sh` 以软链接方式安装三类内容：内置的 `ops-direct-invoke-flash` Skill 装入目标工具的 `skills/` 目录、评审子 Agent 装入 `agents/` 目录，并在项目 / 配置根目录放置规则文件（`AGENTS.md`，Claude Code 下为 `CLAUDE.md`）。此外会拉取 `asc-devkit`、`cann-samples` 两个参考仓库到插件根目录（global 模式下软链进配置根目录）。

```
Usage: init.sh [level] [tool] [install_path]

  level        - project（默认）| global
  tool         - opencode（默认）| claude | trae | cursor | copilot
  install_path - project 级安装目录（默认：当前工作目录）
```

**支持的工具与安装路径**：

| 工具 | project 级 | global 级 |
|------|-----------|-----------|
| OpenCode | `.opencode/skills/` | `~/.config/opencode/skills/` |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| Trae | `.trae/`、`.marscode/`、`.traecli/` 下 `skills/` | 对应 `~/` 路径 |
| Cursor | `.cursor/skills/` | `~/.cursor/skills/` |
| Copilot | `.github/skills/` | `~/.copilot/skills/` |

**示例**：

```bash
bash init.sh                              # project，OpenCode
bash init.sh global claude                # global，Claude Code
bash init.sh project cursor               # project，Cursor
bash init.sh project claude /path/to/proj # 安装到指定项目目录
bash init.sh --help                       # 查看完整帮助
```

安装特性：
- ✅ 软链接安装，源码更新后无需重装即可生效
- ✅ 幂等：重复执行只刷新本 Skill / Agent 的链接，不影响其他已存在的 skills、agents
- ✅ 已存在的真实 `AGENTS.md` / `CLAUDE.md` 会先备份（`*.bak.<时间戳>`）再替换
- ✅ 首装拉取 `asc-devkit` / `cann-samples` 参考仓库（浅克隆、取最新；离线则告警跳过，不阻塞安装）
- ✅ 自动为内置 Hooks 赋予可执行权限，并生成 `cannbot-manifest.json` 安装清单（含 `installed_skills` / `installed_agents` / `installed_repos`）

---

## 工作流概览

```
源（CPU 函数 / 公式 / 代码 / 描述）
        │
        ▼
 1. 分析来源、提取语义，选取一个已完成算子作结构参考
        │
        ▼
 2. 创建 docs/{OP}/STATE.md → 写定义文档 → 写设计文档
        │
        ▼
 3. 子 Agent 评审（math / semantics / ub / instr / reg-api）→ 打磨
        │
        ▼
 4. 分阶段实现核函数 → 本地构建/测试 → 远程 NPU 验证
        │
        ▼
 5. docs/{OP}/plans/troubleshooting.md 记录问题，经验反哺工作流
```

详见 [skills/ops-direct-invoke-flash/SKILL.md](skills/ops-direct-invoke-flash/SKILL.md)。

---

## 目录结构

```
ops-direct-invoke-flash/
├── init.sh                              # 安装脚本（本说明所述）
├── README.md                           # 项目说明文档
├── AGENTS.md                           # 主 Agent 规则文件（Claude Code 下安装为 CLAUDE.md）
├── .claude-plugin/
│   └── plugin.json                     # 插件元信息（声明 agents）
├── agents/
│   └── ops-direct-invoke-flash-reviewer.md  # 评审子 Agent（设计评审 / 验收）
├── asc-devkit/                          # 安装时拉取的 API 文档/示例仓库（git 忽略）
├── cann-samples/                        # 安装时拉取的算子样例仓库（git 忽略）
└── skills/
    └── ops-direct-invoke-flash/
        ├── SKILL.md                    # 工作流主文件（含 Hooks 声明）
        ├── state-template.md           # STATE.md 模板（进度持久化）
        ├── hooks/
        │   ├── pre-build-check.sh      # 构建前检查 device 侧 host-only 辅助函数
        │   └── post-edit-reminder.sh   # 编辑核函数后提示约定校验
        └── references/
            ├── agent-team-patterns.md      # 子 Agent 协作模式
            ├── implementation-patterns.md  # 实现代码模式
            ├── common-failure-modes.md     # 常见失败模式与规避
            ├── reg-api-guide.md            # Ascend950 Reg API 指南
            ├── reg-api-patterns.yaml       # Reg API 允许/禁止清单
            └── review-prompts.md           # 各评审 Agent 的提示词与判定格式
```

---

## 适用场景

**适合使用**：
- 手上有 CPU 函数、数学公式、代码片段或文字描述，需要产出生产级 Ascend C 核函数
- 需要用 `AscendC::Reg` 编写 Ascend950 / `dav-3510` 算子
- 从零构建并向算子工程添加一个新算子
- 需要端到端的算子工作：规格说明、实现、NPU 验证

**不适合使用**：
- 对现有核函数代码做快速修补
- 孤立的测试改动或纯粹的代码讲解
- 修改一个已经实现完成的算子

---

## 免责声明

> ⚠️ **本项目由 AI 辅助生成**

本插件生成的核函数代码与设计文档需自行验证正确性和安全性，重要结论请人工复核并在真实 NPU 硬件上验证。

本项目遵循 CANN Open Software License Agreement V2.0。
