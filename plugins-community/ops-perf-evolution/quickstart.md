# CANNBot Ascend C 算子性能进化优化快速入门指南

## 概述

Ascend C 算子性能进化优化模式适用于**已有 AscendC 内核或 ops 仓库算子的性能提升**场景，通过世界模型驱动的多轮并行进化，在保持精度的前提下逐步优化性能，最终输出 `evolution-report_*.html` 汇总报告。

### 工作流

```
步骤 0: 参数确认 + 路径判断  (Ops 仓库 / 基线内核)
步骤 1: 环境检测 + 设备绑定  (NPU、CANN、目标路径)
步骤 2: 基线性能评估        (ops-profiling / ops-evaluation)
步骤 3: 世界模型初始化      (evolution-world-model)
步骤 4: 多轮进化优化        (每轮：策略选择 → 并行生成变体 → 构建/评估 → 证据积累)
步骤 5: 最佳方案确定 + 泛化验证
步骤 6: 生成 evolution-report_*.html
```

### 路径分类

| 路径 | 输入类型 | 调用的 Subagent |
|------|---------|----------------|
| Ops 仓库路径 | ops-nn/cv/math/transformer/omni-ops 仓中的算子路径 | `ops-evo` |
| 基线内核模式 | 其他现有 AscendC 内核目录 | `lingxi-evo`（默认） |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 Claude Code
- 已准备待优化的基线内核目录或 ops 仓库算子路径

### Claude Code

**方式一：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install ops-perf-evolution@cannbot
```

**方式二：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-community/ops-perf-evolution
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

## 二、快速上手

### 使用方式

#### 场景一：Ops 仓库算子进化优化

在交互界面中输入：

```
对 ops-nn 仓的算子进行进化优化，npu=0，算子名称为 softmax，
算子路径为 /path/to/ops-nn/softmax/，目标加速比 2x，进化轮数 3，并行数 5
```

`ops-perf-evolution` Agent 会调度 `ops-evo` Subagent，直接在 ops 仓库内进行定向优化。

#### 场景二：基线内核进化优化

在交互界面中输入进化需求：

```
对基线内核进行进化优化，npu=0，基线内核路径为 /path/to/baseline/kernel/，
目标加速比 3x，进化轮数 3，并行数 5
```

CANNBot 会自动调度 `ops-perf-evolution` Agent，由其内部调度 `lingxi-evo` Subagent 执行：
1. 解析参数，判断路径类型
2. 环境检测与设备绑定
3. 评估基线性能
4. 初始化世界模型
5. 多轮并行进化（选择策略 → 生成变体 → 构建/评估 → 证据积累）
6. 选择最佳方案，进行泛化验证
7. 生成 `evolution-report_*.html`

### 产出物示例

进化完成后的 `output/<op_name>_<mode>_<timestamp>/` 目录：

```
output/add_layer_norm_custom_ops-evo_20260520_141500/
├── evolution-report_add_layer_norm_custom_20260520_141500.html  ← 浏览器打开看可视化报告
├── state.json                                    ← 当前进度（自动维护）
├── world_model.json                              ← 决策树 + 评测数据
├── shared/                                       ← 共享文件
├── round_1/
│   ├── parallel_0/
│   │   ├── kernel/                               ← AscendC 内核代码
│   │   ├── evaluation_results.json               ← compile/precision/speedup
│   │   └── profiling/                            ← msprof 数据
│   └── parallel_1/
│       └── ...
└── round_2/
    └── ...
```

### 关键查看点

| 想看 | 看哪里 |
|------|--------|
| 进化整体效果 | `evolution-report_*.html`（浏览器打开） |
| 最优变体的 kernel 代码 | `world_model.json` 找 best_score 节点 → 它的 `solution_ref` 字段 → 对应的 `round_N/parallel_K/kernel/` |
| 每个变体的精度/性能 | `round_N/parallel_K/evaluation_results.json` |
| 当前进度（任务跑到哪了） | `cat state.json` |
| 为什么进化停滞 | `world_model.json` 的 `stagnation_count` + `open_questions` 字段 |

### Tips

1. **进化轮数内未达标时如何快速迭代收敛**：性能优化过程中，鉴于优化轮数不足或者大模型能力原因，会遇到在设定的进化轮数中无法达到指定目标的情况，这时可以采用两种方案继续尝试：
   - **让 agent 自我反思性能瓶颈，定向优化**，例如：
     - "分析性能瓶颈，列出当前最优变体三个最严重的性能劣化的因素"
     - "针对劣化最严重的因素，在最优变体的基础上再优化 1 轮"
   - **人工查看算子实现，找出优化点，指示 agent 定向优化**，例如：
     - "基于最优变体，在此基础上修正 xxxx 的实现问题，再进化一轮"

2. **优化完成后建议执行全量 case 测试**：性能优化完会给出最优变体的最终性能，但该性能仅能作为参考——优化过程中为了快速定位问题通常会对测试 case 做精简。因此完成优化后，建议强制 agent 用脚本执行全量 case 测试或手动执行性能测试，获得准确性能结果：
   - **交互方式**：直接指示 agent，例如 "采用 ops-profiling --quick 模式重新采集一遍性能"
   - **手动测试方式（推荐，最准确）**：
     ```bash
     ASCEND_RT_VISIBLE_DEVICES=5 bash .claude/skills/ops-profiling/scripts/msprof_profile_run.sh --quick --output-dir=path/to/output/kernel
     ```

## 三、可用技能

| Skill | 用途 | 适用路径 |
|-------|------|---------|
| `npu-arch` | NPU 架构与芯片规格查询（数据源：CANN 包 platform_config、白皮书） | 所有 |
| `ops-profiling` | 性能采集与分析 | 所有 |
| `ops-evaluation` | ops 仓库算子构建与评估 | Ops 仓库路径 |
| `evolution-knowledge` | 优化知识与模式参考 | 所有 |
| `evolution-strategies` | 优化策略库查询 | 所有 |
| `evolution-world-model` | 世界模型初始化与更新 | 所有 |
| `evolution-report` | 生成进化优化汇总报告 | 所有 |

## 四、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 进化优化会修改原始基线代码吗？

不会。进化优化在 `{output_dir}/` 下生成新的变体副本，原始基线目录保持只读。

### Q: 目标加速比已经达标时会怎样？

当基线性能或某一轮变体已达成目标加速比时，进化会提前结束，并向用户汇报结果，不会继续无意义轮次。

### Q: Ops 仓库路径构建失败怎么办？

连续构建失败 ≥ 3 次会暂停并上报用户，由用户检查环境或 ops 仓库状态。

### Q: 进化轮次上限是多少？

由用户传入的 `max_rounds` 参数决定（默认 2）。每轮并行候选数由 `parallel_num` 决定（默认 3）。

---

## 总结

1. 从已有 AscendC 内核或 ops 仓库算子出发，进行多轮并行性能进化优化
2. 使用 `/plugin install ops-perf-evolution@cannbot` 一键安装，或用 `init.sh` 脚本安装 Claude Code 环境
3. 世界模型驱动、证据驱动的策略选择，保证优化方向的可积累性
4. 产出物包含世界模型、状态游标、各轮变体、最佳方案和 `evolution-report_*.html`
