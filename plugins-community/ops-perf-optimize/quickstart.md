git # AscendC 算子性能调优快速入门指南

## 概述

PerformanceOptimizationAgent 是一套 AscendC 算子性能调优工具，用户只需提供可运行的 demo 代码和测试用例文件，系统自动完成全量 case 性能数据采集、逐 case 分析、输出多个调优方案并分别实施验证，可选接入 CANN-Bench 评测：

```
用户提供 demo 代码 + cases.csv → 全量 case 采集数据 → 逐 case 性能分析 → 输出《性能调优方案》（含逐 case 覆盖清单）→ 多方案实施 → 全部 case 精度验证 → 输出《性能调优报告》（逐 case 瓶颈→手段→加速比）→ [可选] CANN-Bench A5 评测
```

## 一、环境搭建

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/skills.git

# 安装插件
/plugin install ops-perf-optimize@cannbot
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/hanrui22/cannbot-skills_workflow.git

# 项目级安装（在目标项目目录下执行）
cd /path/to/your/project
bash /path/to/cannbot-skills_workflow/plugins-community/ops-perf-optimize/init.sh project claude

# 全局级安装（任意目录执行）
bash /path/to/cannbot-skills_workflow/plugins-community/ops-perf-optimize/init.sh global claude
```

### OpenCode

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/hanrui22/cannbot-skills_workflow.git

# 项目级安装（在目标项目目录下执行）
cd /path/to/your/project
bash /path/to/cannbot-skills_workflow/plugins-community/ops-perf-optimize/init.sh project opencode

# 全局级安装（任意目录执行）
bash /path/to/cannbot-skills_workflow/plugins-community/ops-perf-optimize/init.sh global opencode
```

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 ops-perf-optimize@cannbot ✔ enabled

# OpenCode
opencode agent list
# 应看到 ascendc-perf-analysis-expert / ascendc-perf-impl-expert
```

## 二、快速上手

### 启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

### 调优算子示例

在交互界面中提供待调优的 demo 代码和测试用例即可：

```
请帮我优化 matmul 算子的性能，源码目录：/path/to/0_naive/，测试用例：/path/to/cases.csv
```

如需接入 CANN-Bench 评测（可选）：

```
请帮我优化 matmul 算子的性能，源码目录：/path/to/0_naive/，测试用例：/path/to/cases.csv。优化后接入 cannbench 评测，cann_bench 路径：/path/to/cann-bench_A5/，跑 A5 加速比。
```

### 必须提供的信息

| 信息 | 必需 | 说明 |
|------|------|------|
| 可运行的待调优 demo 代码 | **是** | 完整的源码目录，能够直接编译、运行、验证精度 |
| 测试用例文件 (cases.csv) | **是** | CSV 格式，包含所有测试用例的 shape、dtype、attrs 等参数。**每个 case 都会被全量分析和报告** |

**CANN-Bench 评测接入（可选）**：

| 信息 | 必需 | 说明 |
|------|------|------|
| cann_bench 代码路径 | 仅 Step 3 | CANN-Bench 仓库根目录 |
| 评测环境信息 | 仅 Step 3 | 目标 A5 芯片型号、CANN 版本等 |
| 接入 cannbench 的决策 | 仅 Step 3 | 明确要求"接入 cannbench 评测"或"跑 A5 加速比" |

无需额外提供算子名称、输入参数、性能数据或 TilingData — 系统会自动采集和分析。

## 三、工作流程

```
Step 1: 性能数据采集与分析（性能调优分析专家）
  → 运行 demo → 对全部 case 用 ops-profiling 采集数据
  → Tiling 建模 + 逐 case 性能分析（瓶颈相同的归组）
  → 输出《性能调优方案》（可含多个方案，含逐 case 覆盖清单）

Step 2: 方案实施（性能调优方案实施专家）
  → 按每个方案分别实施 → 全部 case 精度验证
  → 主 agent 统一 msprof 采集全部 case 的 aiv_time
  → 优先查 ascendc-performance-best-practices 货架
  → API 查询用 ascendc-docs-search，案例参考 cann-samples
  → 输出《性能调优报告》（逐 case 瓶颈→手段→加速比）

Step 3: CANN-Bench 评测接入（可选）
  → 确认/采集 A5 基线数据
  → 以 direct_launch_example 方式接入 cann_bench
  → 运行 A5 评测，计算加速比和评分
  → 输出《CANN-Bench 评测报告》（逐 case A5 加速比 + 评分）
```

## 四、可用技能

| Skill | 用途 | 触发时机 |
|-------|------|---------|
| `ops-profiling` | 上板 profiling 数据采集与分析 | Step 1：性能数据采集 + Step 2：性能对比 |
| `ascendc-performance-optimization` | Tiling 理论建模 + Bound / 负载均衡方法库 | Step 1：性能分析阶段 |
| `ascendc-performance-best-practices` | 性能优化货架知识（优先查阅） | Step 1 策略制订 + Step 2 实施阶段 |
| `ascendc-docs-search` | API 文档查询 | Step 2：实施阶段 |

## 五、产出物

| 阶段 | 产出物 | 说明 |
|------|--------|------|
| Step 1 | 性能数据采集产物 | profiling 目录（覆盖全部 case） |
| Step 1 | 《性能调优方案》 | 可含多个方案，每个含优化的 Tiling 参数和策略；含逐 case 覆盖清单；无需优化时输出说明 |
| Step 2 | 《性能调优报告》 | 各方案相对统一基线的性能对比（全部 case 的 aiv_time）、逐 case 瓶颈→手段→加速比 |
| Step 3 | 《CANN-Bench 评测报告》 | 逐 case A5 加速比和评分（可选） |

## 六、知识参考

实施优化时，系统会按优先级查阅以下知识源：

1. **ascendc-performance-best-practices**：优先查询货架设计文档
2. **[cann-samples](https://gitcode.com/cann/cann-samples)**：官方样例的优化实践参考
3. **[asc-devkit](https://gitcode.com/cann/asc-devkit)**：算子开发工具链信息

## 七、常见问题

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
# Claude Code
/plugin update ops-perf-optimize@cannbot

# OpenCode (init.sh 方式，在目标项目目录下执行)
cd /path/to/your/project && bash /path/to/cannbot-skills_workflow/plugins-community/ops-perf-optimize/init.sh
```

---

## 总结

1. 用户需提供可运行的 demo 代码和测试用例文件（cases.csv），无需手动提供性能数据或 TilingData
2. 系统自动运行 demo、对全部 case 采集性能数据
3. 《性能调优方案》可包含多个方案，含逐 case 覆盖清单
4. 实施时优先查货架知识、API 查询用 ascendc-docs-search
5. 《性能调优报告》含全部 case 的逐 case 瓶颈→手段→加速比
6. 可选接入 CANN-Bench 评测，以 direct_launch_example 方式跑 A5 加速比和评分
