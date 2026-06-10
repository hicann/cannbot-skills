# ops-qa-suite

Ascend C 算子仓库 QA 检测套装，为华为昇腾 AI 处理器的 Ascend C 算子仓库（ops-math/ops-nn/ops-transformer/ops-cv）提供自动化质量扫描与分析能力。

**核心价值**：

| 维度 | 能力 |
|------|------|
| **文档质量** | 扫描仓库文档正确性、易理解性、规范性（31项） |
| **测试覆盖** | 分析 UT 测试缺失情况（4种类型）、执行全量 UT 测试 |
| **配置验证** | 检测 CMake 配置问题（9种类型） |
| **列表一致性** | 验证 op_list.md/op_api_list.md 表格与实际实现一致性 |
| **链接有效性** | 扫描 Markdown 断链、自动修复并创建 PR |
| **Issue 管理** | 自动生成 GitCode Issue、批量创建与智能合并 |

---

## QUICKSTART

三步上手，5 分钟完成首次扫描：

```bash
# Step 1: Clone 目标算子仓库
git clone https://gitcode.com/cann/ops-math.git

# Step 2: 执行扫描命令
/scan-repo-docs ops-math          # 文档质量扫描

# Step 3: 查看报告
# 报告自动生成到 reports/{日期}/{仓库}/ 目录
# Issue 文件存放在 reports/{日期}/{仓库}/issues/ 目录
```

**输出示例**：
```
reports/
└── 20260601/
    └── ops-math/
        ├── scan-repo-docs_report_143022.md    # 扫描报告
        └── issues/
            └── cmake_error_issue_143022.md    # Issue 文件（可提交到 GitCode）
```

> **提示**：如需直接提交 Issue 需要 GitCode guide 权限，否则手动复制到 web 提交。

---

## 核心能力概览

本项目提供 **12 个 Skills + 8 个 Commands + 1 个 Agent**，覆盖全生命周期质量检测：

| 类别 | 数量 | 功能 |
|------|:---:|------|
| **扫描类 Skills** | 4 | 文档质量、CMake 配置、算子列表、接口列表 |
| **分析类 Skills** | 2 | UT 缺失分析、Examples 缺失分析 |
| **测试执行类 Skills** | 2 | UT 测试执行、Examples 测试执行 |
| **工具类 Skills** | 3 | Issue 创建、断链修复、断链检查 |
| **快捷 Commands** | 8 | `/scan-repo-docs`、`/scan-ut-test` 等 |
| **编排 Agent** | 1 | ops-scanner（一键全量扫描） |

**扫描覆盖维度**：
- 文档正确性：21项（环境部署、QUICKSTART、算子开发、调试定位等）
- 文档易理解性：7项（大纲脉络、版本配套、章节完整性等）
- 文档规范性：3项（安全声明、许可证、链接直达性）
- CMake 配置问题：9种（OPTYPE 错误、函数不存在、目标冲突等）

---

## 技术架构

```
项目架构层次
┌─────────────────────────────────────────┐
│        Agents（编排层）                  │  ops-scanner：统一扫描编排
├─────────────────────────────────────────┤
│        Commands（快捷入口）              │  8 个扫描命令入口
├─────────────────────────────────────────┤
│        Skills（能力模块）                │  12 个技能模块
├─────────────────────────────────────────┤
│        Scripts（执行层）                 │  4 个核心脚本
├─────────────────────────────────────────┤
│        Templates（模板层）               │  3 个报告模板
├─────────────────────────────────────────┤
│        Config（配置层）                  │  仓库配置中心
└─────────────────────────────────────────┘
```

---

## 目录结构

```
ops-qa-suite/
├── .opencode/                      # OpenCode 配置与技能
│   ├── skills/                     # 技能模块（扫描类、分析类、测试执行类、工具类）
│   ├── commands/                   # 快捷命令（9个扫描命令）
│   ├── scripts/                    # 公共脚本（仓库检测、配置加载等）
│   ├── templates/                  # 报告模板
│   ├── config/                     # 配置文件（仓库配置中心）
│   └── agents/                     # Agent 配置
├── README.md                       # 项目说明文档
└── AGENTS.md                       # Agent 配置文档
```

---

## 常用扫描命令

| 场景 | 命令 | 说明 |
|------|------|------|
| **文档质量扫描** | `/scan-repo-docs ops-math` | 扫描文档正确性、易理解性、规范性 |
| **UT 缺失分析** | `/scan-ut-analysis ops-math` | 分析 UT 测试覆盖情况 |
| **UT 测试执行** | `/scan-ut-test ops-math` | 执行全量 UT 测试并生成报告 |
| **Examples 缺失分析** | `/scan-examples-analysis ops-math` | 分析 examples 测试用例缺失情况 |
| **Examples 测试执行** | `/scan-examples-test ops-math` | 执行全量 examples 测试 |
| **CMake 配置扫描** | `/scan-cmake ops-math` | 检测 CMake 配置问题（9种） |
| **算子列表一致性** | `/scan-op-list ops-math` | 验证 op_list.md 表格一致性 |
| **接口列表一致性** | `/scan-op-api-list ops-math` | 验证 op_api_list.md 表格一致性 |
| **统一扫描（推荐）** | `请扫描 ops-math 仓库` | 一键执行全部扫描 |

> **详细参数说明**：见 [docs/commands-reference.md](docs/commands-reference.md) 或各 Command 的 SKILL.md。

---

## 统一扫描（ops-scanner）

一键执行全部扫描任务，生成汇总报告：

```opencode
请扫描 ops-math 仓库
请扫描 ops-cv 仓库，跳过 UT 测试执行
```

**扫描流程（10 个 Phase）**：
| Phase | 扫描类型 | 功能 |
|:-----:|---------|------|
| 1 | scan-repo-docs | 文档质量扫描（正确性21项+易理解性7项+规范性3项） |
| 2 | scan-cmake | CMake 配置问题扫描（9种类型） |
| 3 | scan-op-list | 算子列表一致性验证 |
| 4 | scan-op-api-list | 接口列表一致性验证 |
| 5 | scan-ut-analysis | UT 缺失分析 |
| 6 | scan-ut-test | UT 测试执行与报告 |
| 7 | scan-examples-analysis | Examples 缺失分析 |
| 8 | scan-examples-test | Examples 测试执行与报告 |
| 9 | tool-reports-to-issue | Issue 创建 |
| 10 | 汇总报告生成 | 生成统一汇总报告 |

**输出文件**：
- 汇总报告：`reports/{date}/{repo}/ops-scan_summary_{time}.md`
- 各类扫描报告：`reports/{date}/{repo}/{skill}_report_{time}.md`
- Issue 文件：`reports/{date}/{repo}/issues/`

---

## 支持的仓库

本项目支持**ops-* 算子仓库**，通过动态发现机制自动识别。

**识别规则**：
- 仓库名匹配模式：`ops-*`
- 仓库特征验证：至少包含 CMakeLists.txt、docs、op_host、op_kernel 等特征

**常见仓库**：
| 仓库名 | 说明 |
|-------|------|
| ops-math | 数学运算算子仓库 |
| ops-nn | 神经网络算子仓库 |
| ops-transformer | Transformer 算子仓库 |
| ops-cv | 计算机视觉算子仓库 |

**自动检测特性**：
- ✅ 从任意嵌套目录向上遍历查找 ops-* 仓库根目录
- ✅ 未找到仓库时自动从 GitCode clone
- ✅ 每次扫描前自动 `git pull` 更新仓库

---

## 详细文档

| 文档 | 说明 |
|------|------|
| [AGENTS.md](AGENTS.md) | Agent 配置与工作流程详细说明 |
| [docs/project_capa.md](docs/project_capa.md) | 项目能力分析报告（技术规格书） |
| [.opencode/commands/](.opencode/commands/) | 各 Command 的 SKILL.md（详细参数） |
| [.opencode/skills/](.opencode/skills/) | 各 Skill 的 SKILL.md（详细说明） |

---

## 更新日志

- **2026-05-28**: CANN 算子仓库扫描助手首次上线，提供文档质量、UT 缺失、Examples、CMake 配置等 9 种扫描能力

---

## 免责声明

> ⚠️ **本项目由 AI 辅助生成**

本项目部分代码由 AI 辅助生成。使用前请注意：
- 本项目代码需自行验证正确性和安全性
- AI 生成的扫描结果仅供参考，重要结论需人工复核

本项目遵循 CANN Open Software License Agreement V2.0。