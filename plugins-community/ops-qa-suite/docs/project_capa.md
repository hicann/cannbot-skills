# CANN 算子仓库扫描助手项目能力分析报告

**生成时间**: 2026-05-28
**项目路径**: `plugins-community/repository-scan`
**项目类型**: OpenCode Skills/Commands/Agents 项目

---

## 一、项目概述

### 1.1 项目定位

本项目是一个 **CANN 算子仓库扫描助手**，专为华为昇腾 AI 处理器的 Ascend C 算子仓库（ops-math/ops-nn/ops-transformer/ops-cv）提供自动化质量扫描与分析能力。

### 1.2 核心价值

| 维度 | 能力 |
|------|------|
| **文档质量** | 扫描仓库文档正确性、易理解性、规范性（31项） |
| **测试覆盖** | 分析 UT 测试缺失情况（4种类型）、执行全量 UT 测试 |
| **配置验证** | 检测 CMake 配置问题（9种类型） |
| **列表一致性** | 验证 op_list.md/op_api_list.md 表格与实际实现一致性 |
| **链接有效性** | 扫描 Markdown 断链、自动修复并创建 PR |
| **Issue 管理** | 自动生成 GitCode Issue、批量创建与智能合并 |

### 1.3 技术架构

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

## 二、Agents 能力分析

### 2.1 Agents 总览

| Agent 名称 | 类型 | 描述 |
|-----------|------|------|
| **ops-scanner** | subagent | CANN 算子仓库统一扫描 Agent |

### 2.2 ops-scanner 详细能力

**定位**: 一次性执行全部扫描任务，生成汇总报告。

**核心职责**:
- 执行 10 种扫描类型（文档质量、UT、Examples、CMake、列表一致性等）
- 收集所有扫描报告并生成统一汇总
- 自动为问题创建 Issue 文件

**扫描流程（10 个 Phase）**:

| Phase | 扫描类型 | 功能 |
|:-----:|---------|------|
| 0 | 参数解析与环境检查 | 仓库路径自动检测、环境预检 |
| 1 | scan-repo-docs | 文档质量扫描（正确性21项+易理解性7项+规范性3项） |
| 2 | scan-cmake | CMake 配置问题扫描（9种类型） |
| 3 | scan-op-list | 算子列表一致性验证（op_list.md，含 README 检查） |
| 4 | scan-op-api-list | 接口列表一致性验证（op_api_list.md，含 aclnn 文档检查） |
| 5 | scan-ut-analysis | UT 缺失分析（infershape/tiling/kernel/api） |
| 6 | scan-ut-test | UT 测试执行与报告 |
| 7 | scan-examples-analysis | Examples 缺失分析 |
| 8 | scan-examples-test | Examples 测试执行与报告 |
| 9 | tool-reports-to-issue | Issue 创建 |
| 10 | 汇总报告生成 | 生成统一汇总报告 |

**调用方式**:
```opencode
请扫描 ops-math 仓库
请扫描 ops-cv 仓库，跳过 UT 测试执行
```

**输出文件**:
- 汇总报告: `reports/{date}/{repo}/ops-scan_summary_{time}.md`
- 各类扫描报告: `reports/{date}/{repo}/{skill_name}_report_{time}.md`
- Issue 文件: `reports/{date}/{repo}/issues/`

---

## 三、Skills 能力分析

### 3.1 Skills 总览

本项目包含 **12 个 Skills**，分为 5 大类：

| 类别 | Skills | 数量 |
|------|--------|:---:|
| **扫描类** | scan-repo-docs, scan-cmake, scan-op-list, scan-op-api-list | 4 |
| **分析类** | scan-ut-analysis, scan-examples-analysis | 2 |
| **测试执行类** | scan-ut-test, scan-examples-test | 2 |
| **工具类** | tool-reports-to-issue, fixer-broken-link, tool-link-checker | 3 |
| **基础设施** | gitcode-toolkit（软链接 infra） | 1 |

### 3.2 扫描类 Skills

#### 3.2.1 scan-repo-docs（仓库文档质量扫描）

**描述**: 扫描仓库文档的正确性、易理解性、规范性。

**扫描维度**:
| 类别 | 扫描项数 | 内容 |
|------|:-------:|------|
| 资料正确性 | 21 | 环境部署、QUICKSTART、源码下载、算子调用、算子开发、调试定位、性能调优等 |
| 资料易理解性 | 7 | 大纲脉络、Latest News、版本配套、章节完整性等 |
| 资料规范性 | 3 | 安全声明、许可证、链接直达性 |

**输出**: `reports/{date}/{repo}/scan-repo-docs_report_{time}.md`

---

#### 3.2.2 scan-cmake（CMake 配置问题扫描）

**描述**: 检测 9 种 CMake 配置问题。

**问题类型**:
| # | 问题类型 | 说明 |
|:---:|---------|------|
| 1 | OPTYPE 与目录名不一致 | CMakeLists.txt 中 OPTYPE 参数值与算子目录名不匹配 |
| 2 | 函数不存在 | 使用未定义的函数如 add_modules_llt_sources |
| 3 | 变量不存在 | 使用不存在的变量如 OPTEST_NAME |
| 4 | 参数名错误 | 函数参数名与仓库定义不一致 |
| 5 | if 语句语法错误 | 变量未用引号包裹导致空值语法错误 |
| 6 | 目标名称冲突 | 多个算子共享相同 CMake 目标名称 |
| 7 | 缺少源文件错误 | 测试模块未正确添加源文件 |
| 8 | 条件判断缺失 | 缺少 OP_HOST_UT 条件判断 |
| 9 | 第三方依赖错误 | FetchContent 解压失败 |

**问题性质分类**:
| 仓库 | 问题性质 | 影响 |
|------|---------|------|
| ops-transformer, ops-nn | **BUG** | ⚠️ 高 - 必须修复 |
| ops-math, ops-cv | 规范问题 | 📋 低 - 不影响构建 |

**输出**: `reports/{date}/{repo}/scan-cmake_report_{time}.md`

---

#### 3.2.3 scan-op-list（算子列表一致性扫描）

**描述**: 验证 docs/zh/op_list.md 表格与实际算子实现的一致性。

**检查项**:
| 检查项 | 说明 |
|-------|------|
| 算子目录存在性 | op_list 表格中的算子目录是否实际存在 |
| 算子分类正确性 | 分类列是否与实际父目录一致 |
| 实现状态标记一致性 | op_kernel/op_host/op_api/op_graph 的√×标记是否与实际文件一致 |
| 硬件单元一致性 | AI Core/AI CPU 说明是否与实际实现一致 |

**输出**: `reports/{date}/{repo}/scan-op-list_report_{time}.md`

---

#### 3.2.4 scan-op-api-list（接口列表一致性扫描）

**描述**: 验证 docs/zh/op_api_list.md 表格与实际 aclnn 接口的一致性。

**检查项**:
| 检查项 | 说明 |
|-------|------|
| 接口名一致性 | aclnn 接口名是否与实际接口对应 |
| 接口链接跳转 | 链接是否能正常跳转到 aclnn API 文档 |
| 接口说明一致性 | 说明列是否与 aclnn 文档功能说明一致 |
| 确定性说明一致性 | 确定性说明是否与 aclnn 文档约束说明一致 |

**输出**: `reports/{date}/{repo}/scan-op-api-list_report_{time}.md`

---

### 3.3 分析类 Skills

#### 3.3.1 scan-ut-analysis（UT 缺失分析）

**描述**: 分析算子的 UT 需求，判断何时需要 infershape/tiling/op_kernel/op_api UT。

**UT 类型说明**:
| UT 类型 | 测试内容 | 源文件位置 | 需要 IR 原型 |
|---------|---------|-----------|:-----------:|
| infershape | shape 推导测试 | op_host/*_infershape.cpp | ✅ 必须有 op_graph/*_proto.h |
| tiling | tiling 参数测试 | op_host/*_tiling.cpp | ❌ 不需要 |
| op_kernel | kernel 实现测试 | op_kernel/*.cpp | ❌ 不需要 |
| op_api | aclnn 接口测试 | op_api/aclnn_*.cpp | ❌ 不需要 |

**分析模式**:
| 模式 | 说明 |
|------|------|
| basic | 快速扫描，仅检查目录结构 |
| detailed | 深入分析源文件，识别纯模板调用等不需要 UT 的场景 |

**输出**: `reports/{date}/{repo}/scan-ut-analysis_report_{time}.md`

---

#### 3.3.2 scan-examples-analysis（Examples 缺失分析）

**描述**: 分析算子的 examples 需求，判断何时需要 examples 测试用例。

**判断规则**:
| 场景 | 是否需要 examples |
|------|:----------------:|
| 无调用接口（无 aclnn + 无 proto） | ❌ 不需要 |
| 仅 aclnn 接口（无 kernel） | ❌ 不需要 |
| 有 kernel + 有调用接口 | ✅ 需要 |

**分析模式**:
| 模式 | 说明 |
|------|------|
| basic | 快速扫描 examples 目录是否存在 |
| smart | 深入分析实现类型（仅 aclnn vs 有 kernel） |

**输出**: `reports/{date}/{repo}/scan-examples-analysis_report_{time}.md`

---

### 3.4 测试执行类 Skills

#### 3.4.1 scan-ut-test（UT 测试执行与报告）

**描述**: 执行全量 UT 测试（op_host/op_api/op_kernel/aicpu_kernel），生成测试报告。

**UT 类型说明**:
| UT 类型 | 执行内容 | 依赖环境 | 预计耗时 |
|---------|---------|---------|:-------:|
| full | 全量 UT（4种） | NPU（可选） | 30-75分钟 |
| op_host | tiling + infershape | BUILD_PATH | 10-15分钟 |
| op_api | aclnn 接口测试 | ACLNN 库 | 10-20分钟 |
| op_kernel | AscendC kernel | NPU 硬件 | 15-30分钟 |
| aicpu_kernel | AICPU kernel | AICPU 环境 | 5-10分钟 |
| cpu_only | op_host + op_api | 无需 NPU | 20-35分钟 |

**参数**:
| 参数 | 说明 |
|------|------|
| --scope full/sample | 执行范围 |
| --ut_type | UT 类型选择 |
| --skip_prompt | 跳过用户询问（agent触发时使用） |
| --soc={version} | 指定芯片版本 |

**输出**: `reports/{date}/{repo}/scan-ut-test_report_{time}.md`

---

#### 3.4.2 scan-examples-test（Examples 测试执行）

**描述**: 执行全量 examples 测试（test_aclnn_*.cpp/test_geir_*.cpp），生成测试报告。

**测试类型**:
| 测试类型 | 测试内容 | simulator 支持 |
|---------|---------|:-------------:|
| eager | ACLNN API 调用 | ✅ 支持 |
| graph | GEIR 图引擎调用 | ❌ 不支持 |

**执行模式**:
| 模式 | 当前 NPU 支持 | 当前 NPU 不支持 | simulator |
|------|:---:|:---:|:---:|
| 默认（无参数） | 直接测试 | 自动 simulator | ✅ 自动启用 |
| --simulator --soc=X | 强制 simulator | 跳过 | ✅ 强制启用 |
| skip-sim | 直接测试 | **跳过测试** | ❌ 禁用 |

**输出**: `reports/{date}/{repo}/scan-examples-test_report_{time}.md`

---

### 3.5 工具类 Skills

#### 3.5.1 tool-reports-to-issue（扫描报告转 Issue）

**描述**: 根据扫描报告或问题列表批量生成 GitCode Issue。

**核心能力**:
- 问题类型 → 模板匹配
- 同类问题智能合并
- Issue 内容自动填充

**模板匹配规则**:
| 问题类型 | 推荐模板 | 标签 |
|---------|---------|------|
| README缺失 | Documentation | documentation |
| aclnn文档缺失 | Documentation | documentation |
| CMake配置错误 | Bug-Report | bug-report |
| UT缺失 | Bug-Report | bug-report |
| Examples缺失 | Requirement | requirement |
| 断链问题 | Documentation | documentation |

**智能合并策略**:
| 问题数量 | 策略 |
|:-------:|------|
| ≥10 | 自动合并（仅提示） |
| 3-9 | 询问用户 |
| 1-2 | 不合并 |

**引用 gitcode-toolkit**: 环境预检、模板查询、API创建、日志记录。

---

#### 3.5.2 fixer-broken-link（断链修复）

**描述**: 扫描 Markdown 文件断链，自动修复并创建 PR。

**断链类型与修复策略**:
| 断链类型 | 可自动修复 | 修复策略 |
|---------|:---:|---------|
| 路径错误 | ✅ | 修正路径 |
| 链接换行 | ✅ | 合并单行 |
| 文件不存在 | ❌ | 删除链接，标注待补充 |
| 外部链接失效 | ❌ | 跳过，记录报告 |

**使用方式**:
```bash
# 仅扫描报告
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math

# 扫描并修复
python scripts/scan_links.py --repo ops-math --fix

# 修复并创建 PR
python scripts/scan_links.py --repo ops-math --fix --create-pr
```

**输出**: `reports/broken-link-fixer/{repo}_broken_link_report_{time}.md`

---

#### 3.5.3 tool-link-checker（断链检查）

**描述**: 检测 Markdown 文件链接有效性，分类统计。

**功能**: 扫描所有 .md 文件的内部链接，检测断链并分类。

---

### 3.6 基础设施 Skill

#### 3.6.1 gitcode-toolkit（软链接 infra）

**描述**: GitCode 协作通用基础参考（内部参考，不直接触发）。

**提供的能力**:
| 能力 | 说明 |
|------|------|
| 环境预检 | token / git / curl / /tmp 检查 |
| 模板查询 | PR/Issue 模板 API 查询 |
| PR 创建工作流 | Step 1-7 完整流程 |
| Issue 创建工作流 | Step 1-7 完整流程 |
| Git 操作 | clone/分支/diff/log/remote |
| 日志规范 | 统一日志命名与记录 |

**引用文档**:
- references/env-check.md
- references/gitcode-api.md
- references/token-config.md
- references/url-parsing.md
- references/logging-conventions.md

---

## 四、Commands 能力分析

### 4.1 Commands 总览

本项目提供 **8 个 Commands**，作为 Skills 的快捷入口：

| Command | 对应 Skill | 功能 |
|---------|-----------|------|
| `/scan-repo-docs` | scan-repo-docs | 仓库文档质量扫描 |
| `/scan-cmake` | scan-cmake | CMake 配置问题扫描 |
| `/scan-op-list` | scan-op-list | 算子列表一致性扫描（含 README 检查） |
| `/scan-op-api-list` | scan-op-api-list | 接口列表一致性扫描（含 aclnn 文档检查） |
| `/scan-ut-analysis` | scan-ut-analysis | UT 缺失分析 |
| `/scan-ut-test` | scan-ut-test | UT 测试执行与报告 |
| `/scan-examples-analysis` | scan-examples-analysis | Examples 缺失分析 |
| `/scan-examples-test` | scan-examples-test | Examples 测试执行 |

### 4.2 Commands 参数说明

#### scan-ut-test 参数
| 参数 | 说明 | 示例 |
|------|------|------|
| --scope | 执行范围 | full, sample |
| --ut_type | UT 类型 | full, op_host, op_api, op_kernel, aicpu_kernel, cpu_only |
| --skip_prompt | 跳过询问 | agent触发时使用 |
| --soc | 芯片版本 | ascend910b, ascend950 |

#### scan-examples-test 参数
| 参数 | 说明 | 示例 |
|------|------|------|
| --mode | 测试模式 | eager, graph, all |
| --simulator | 使用仿真 | 需配合 --soc |
| --soc | 仿真芯片 | ascend950, ascend910b |
| skip-sim | 跳过仿真 | 禁用 simulator |

#### scan-repo-docs 参数
| 参数 | 说明 | 示例 |
|------|------|------|
| --scope | 扫描范围 | full, correctness, understandability, normative |
| --skip-exec | 跳过执行验证 | 仅静态分析 |

---

## 五、Scripts 能力分析

### 5.1 Scripts 总览

| Script | 功能 | 用法 |
|--------|------|------|
| **repo_detector.py** | 仓库路径自动检测 | 支持任意嵌套目录向上遍历查找 ops-* 仓库 |
| **repo_discovery.py** | 仓库发现与克隆 | 自动 clone GitCode 仓库 |
| **config_loader.py** | 配置加载 | 加载 repo_config.yaml |
| **cmake_profiler.py** | CMake 性能分析 | CMake 配置问题辅助分析 |

### 5.2 repo_detector.py（核心脚本）

**功能**: 从任意嵌套目录自动检测 ops-* 仓库根目录。

**检测规则（优先级）**:
1. 用户指定 `--repo-root` → 使用指定路径
2. 当前目录是 ops-* 仓库 → 直接使用当前目录
3. 当前目录包含 ops-* 子目录 → 使用子目录
4. 向上遍历父目录查找 ops-* → 支持任意嵌套深度

**检测特征**:
- 目录名匹配模式: `ops-*`
- 必须包含特征文件: CMakeLists.txt, docs, op_host, op_kernel 等

**输出示例**:
```
仓库路径: /path/to/ops-math
检测方法: parent_directory_detection
Reports 输出目录: /path/to/ops-math/repository_scan/reports/
```

---

### 5.3 各 Skill 内嵌脚本

部分 Skills 包含专用脚本：

| Skill | 脚本 | 功能 |
|-------|------|------|
| scan-cmake | cmake_scan.py | CMake 配置扫描执行 |
| scan-ut-analysis | ut_missing_scan.py | UT 缺失扫描执行 |
| scan-op-list | op_list_scan.py | op_list.md 验证执行 |
| fixer-broken-link | scan_links.py | 断链扫描执行 |
| tool-reports-to-issue | generate_issue_md.py | Issue 文件生成 |

---

## 六、Templates 能力分析

### 6.1 Templates 总览

| Template | 用途 | 说明 |
|----------|------|------|
| **unified_report_template.md** | 统一报告模板 | 所有扫描报告遵循的结构模板 |
| **issue_workflow_spec.md** | Issue 创建规范 | Issue 文件命名、合并策略、提交流程 |
| **examples_test_report_template.md** | Examples 测试报告 | examples 测试专用报告模板 |

### 6.2 unified_report_template.md（核心模板）

**报告结构**:
```markdown
# {repo_type} {report_type}报告

## 报告元信息
## 执行摘要
## 问题分类与统计
## 问题详情记录
## {skill_special_section}  ← 各 Skill 特殊字段区域
## 修复建议
## 附录
```

**各 Skill 特殊字段**:
| Skill | 特殊字段 |
|-------|---------|
| scan-cmake | GitCode Issue 文件列表 |
| scan-repo-docs | 执行验证详情、断链详细清单 |
| scan-ut-test | 测试统计结果、UT测试类型说明 |
| scan-ut-analysis | UT覆盖情况详情、缺失UT文件算子 |
| scan-examples-analysis | 分类统计、有examples算子列表 |
| scan-examples-test | 测试统计结果、芯片兼容性记录 |

---

## 七、Config 配置分析

### 7.1 repo_config.yaml 配置结构

**配置中心管理**:
- 仓库匹配模式
- GitCode URL/API 配置
- 标准目录结构约定
- 文档路径和格式

### 7.2 配置模块

| 模块 | 配置项 | 说明 |
|------|--------|------|
| **repository** | pattern, markers, excluded_dirs | 仓库识别配置 |
| **gitcode** | clone_template, api_base, web_base | GitCode 平台配置 |
| **structure** | directories, ut_structure, architecture_dirs | 标准目录结构 |
| **docs** | zh_root, repo_docs, operator_docs | 文档路径约定 |
| **link_patterns** | op_list_link, op_api_list_link | 链接路径模式 |
| **naming** | implementation, ut_test, examples | 文件命名约定 |
| **reports** | output_dir, filename_format | 报告输出配置 |

### 7.3 GitCode URL 模板

```
克隆 URL: https://gitcode.com/cann/{repo}.git
Issue 页面: https://gitcode.com/cann/{repo}/issues/new
API 基础: https://api.gitcode.com/api/v5
```

---

## 八、支持的仓库

### 8.1 仓库识别

**动态发现机制**: 支持**任意 ops-* 格式的仓库**，无需预定义列表。

**识别规则**:
- 仓库名匹配: `ops-*` 模式
- 特征验证: 至少包含 CMakeLists.txt, docs, op_host, op_kernel 等特征（≥2个）

### 8.2 常见仓库示例

| 仓库名 | 目录特征 | 扫描命令 |
|-------|---------|---------|
| ops-math | math/, CMakeLists.txt, docs/ | `/scan-repo-docs ops-math` |
| ops-nn | nn/, activation/, matmul/ | `/scan-repo-docs ops-nn` |
| ops-transformer | transformer/, attention/, moe/ | `/scan-repo-docs ops-transformer` |
| ops-cv | cv/, image/, detection/ | `/scan-repo-docs ops-cv` |

---

## 九、输出规范

### 9.1 报告目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── {skill}_report_{HHMMSS}.md    # 各类扫描报告
        ├── {skill}_data.json             # JSON 数据文件
        └── issues/
            ├── {issue_type}_issue_{time}.md  # Issue 文件
            └── his/                       # 历史报告归档
```

### 9.2 Issue 创建流程

**核心原则**:
| 原则 | 说明 |
|------|------|
| 所有问题都创建 Issue | 不考虑级别，所有问题都生成 Issue 文件 |
| 报告后询问提交 | 生成报告后询问是否提交到 GitCode |
| 同类问题合并选项 | 同类问题可合并为一个 Issue |

**Issue 文件命名**:
- 合并模式: `{repo}_{issue_type}_merged_issue_{time}.md`
- 单算子模式: `{repo}_{op_name}_{issue_type}_issue_{time}.md`

---

## 十、项目特色

### 10.1 核心特色

| 特色 | 说明 |
|------|------|
| **统一编排** | ops-scanner Agent 一键执行全部扫描 |
| **智能合并** | Issue 创建支持智能合并策略（≥10自动合并） |
| **路径自动检测** | 支持任意嵌套目录向上遍历查找仓库 |
| **仓库自动克隆** | 未找到仓库时自动从 GitCode clone |
| **配置化设计** | URL 模板、识别规则可配置修改 |
| **模板标准化** | 统一报告模板，Issue友好格式 |
| **断链自动修复** | 扫描并自动修复断链，可创建 PR |

### 10.2 技术亮点

- **软链接引用**: gitcode-toolkit 通过软链接引用 infra，避免重复实现
- **分层架构**: Agents → Commands → Skills → Scripts → Templates
- **渐进式披露**: SKILL.md 只写核心内容，详细参考放 references/
- **自动化执行**: ops-scanner 调用时默认合并创建 Issue

---

## 十一、快速使用指南

### 11.1 统一扫描（推荐）

```opencode
请扫描 ops-math 仓库
请扫描 ops-cv 仓库，跳过 UT 测试执行
```

### 11.2 单独扫描

```opencode
# 文档质量
/scan-repo-docs ops-math

# UT 缺失分析
/scan-ut-analysis ops-math

# UT 测试执行（交互式）
/scan-ut-test ops-math

# UT 测试执行（agent 自动触发）
/scan-ut-test ops-math --skip_prompt --ut_type full --scope full

# Examples 测试
/scan-examples-test ops-math

# CMake 配置扫描
/scan-cmake ops-math

# 算子列表一致性
/scan-op-list ops-math

# 接口列表一致性
/scan-op-api-list ops-math
```

### 11.3 断链修复

```bash
# 仅扫描
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math

# 扫描并修复
python scripts/scan_links.py --repo ops-math --fix

# 修复并创建 PR
python scripts/scan_links.py --repo ops-math --fix --create-pr
```

---

## 十二、总结

本项目是一个功能完备、架构清晰的 CANN 算子仓库扫描助手，提供：

| 维度 | 数量 | 覆盖范围 |
|------|:---:|---------|
| Agents | 1 | 统一扫描编排 |
| Skills | 12 | 文档/UT/Examples/CMake/列表一致性/Issue管理 |
| Commands | 8 | 快捷扫描入口 |
| Scripts | 4+ | 核心执行脚本 |
| Templates | 3 | 报告与 Issue 模板 |
| Config | 1 | 统一配置中心 |

**项目定位**: 为华为昇腾 AI 处理器的 Ascend C 算子仓库提供全面的质量扫描与问题管理能力，支持自动化扫描、Issue 创建、PR 提交的完整工作流。

---

**报告生成时间**: 2026-05-28
**版本**: v1.0