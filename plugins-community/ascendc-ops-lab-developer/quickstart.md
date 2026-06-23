# CANNBot Ascend C 算子自动生成快速入门指南

## 概述

Ascend C 算子自动生成模式适用于**从 PyTorch Model 自动生成 Ascend C Kernel**场景，支持双路径：简单算子走 ops-direct-invoke 工作流（Architect 设计 → Developer 实现 → Reviewer 审查），复杂算子走 TileLang 设计表达 → AscendC 转译优化，端到端完成算子开发。

### 工作流

```
Phase 0: 参数确认 + 算子分类  (解析输入，判定简单/复杂路径)
Phase 1: 环境准备 + 工程初始化  (复制算子文件 + 初始化 kernel 工程)
Phase 2: Case 精简           (tilelang2ascend-case-simplifier)
Phase 3: 设计表达            (分支)
  ├─ 简单算子: 架构设计 + 设计串讲 (ops-direct-invoke: DESIGN.md + PLAN.md + WALKTHROUGH.md)
  └─ 复杂算子: TileLang 设计  (tilelang2ascend-tilelang-designer + 退化检测 + 迭代)
Phase 4: AscendC 生成与验证   (分支)
  ├─ 简单算子: 开发实现 + 代码审查 + 修复循环 (ops-direct-invoke: 渐进式开发 + REVIEW.md + 最多3轮修复)
  └─ 复杂算子: 转译          (tilelang2ascend-translator + 退化检测 + 迭代)
Phase 5: 性能分析            (ops-profiling --compare 模式)
Phase 6: 全量验证
Phase 7: Trace 记录          (tilelang2ascend-trace-recorder)
```

### 算子分类路由

| 路径 | 算子类型 | Skill 链 |
|------|---------|---------|
| 简单算子 | Index, IndexPut, Gather, Scatter, Nonzero, RepeatInterleave, EmbeddingDenseBackward | ops-direct-invoke（Architect → Developer → Reviewer） |
| 复杂算子 | Attention, MatMul 变体, Norm 变体, Sort, TopK, 多输入融合 | tilelang2ascend-tilelang-designer → tilelang2ascend-translator |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install ascendc-ops-lab-developer@cannbot
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/plugins-community/ascendc-ops-lab-developer
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/plugins-community/ascendc-ops-lab-developer
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### TRAE

仅支持项目级安装。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd skills/plugins-community/ascendc-ops-lab-developer
bash init.sh project trae
```

## 二、快速上手

### 使用方式

#### 场景一：单算子生成
在交互界面中输入算子开发需求：

```
生成ascendC算子，npu=0，算子描述文件为 /path/to/op/model.py，输出到 /path/to/output/op_name/
```

CANNBot 会自动调度 ascend-kernel-developer 按 7 Phase 流程执行：
1. 解析参数、判定算子类型（简单/复杂）
2. 准备环境、初始化 kernel 工程
3. 精简测试用例（≤ 10 个代表性 case）
4. 设计表达（简单算子走 ops-direct-invoke 架构设计 + 设计串讲，复杂算子做 TileLang 设计）
5. AscendC 生成与验证（简单算子走开发实现 + 代码审查 + 修复循环，复杂算子走转译 + 迭代验证）
6. 性能分析（对比 reference 和 ascendc）
7. 全量用例验证
8. 生成 trace.md 记录完整过程

#### 场景二：批量性能测试

适用于对多个已生成算子进行批量性能对比测试，支持单 NPU 串行或多 NPU 并行执行。

**操作步骤**：

1. 确保每个算子目录下已有 `model.py`、`model_new_ascendc.py` 和测试用例文件
2. 执行批量测试脚本：

```bash
bash .claude/skills/ops-profiling/scripts/msprof_profile_run.sh --batch \
    --base-dir /path/to/output_performance \
    --max-jobs 7 \
    --device-start 1
```

**参数说明**：
- `--base-dir`: 包含多个算子输出子目录的根目录（必填）
- `--max-jobs`: 最大并发数（默认 7）
- `--device-start`: 起始 NPU 设备 ID（默认 1）
- `--warm-up`: 预热次数（默认 3）

### 产出物示例

```
{output_dir}/
├── model.py                  # 算子描述文件（只读）
├── <op_name>.json            # 原始测试用例文件（备份保留）
├── <op_name>.json.bak        # 原始 .json 备份
├── docs/
│   ├── DESIGN.md             # 架构设计文档（简单算子路径）
│   ├── PLAN.md               # 开发计划（简单算子路径）
│   ├── WALKTHROUGH.md        # 设计串讲记录（简单算子路径）
│   └── REVIEW.md             # 代码审查报告（简单算子路径）
├── design/
│   ├── block_level/          # Block-level 设计（复杂算子路径）
│   └── tile_level/           # Tile-level 设计（复杂算子路径）
├── kernel/                   # AscendC kernel 实现
├── model_new_tilelang.py     # TileLang 实现（复杂算子路径）
├── model_new_ascendc.py      # AscendC 实现
├── performance.json          # 性能数据
└── trace.md                  # 执行记录
```

## 三、可用技能

| Skill | 用途 | 适用路径 |
|-------|------|---------|
| `tilelang2ascend-case-simplifier` | 测试用例精简 | 所有 |
| ops-direct-invoke (Architect + Developer + Reviewer) | 简单算子架构设计、开发实现、代码审查 | 简单算子 |
| `tilelang2ascend-tilelang-designer` | TileLang kernel 设计表达 | 复杂算子 |
| `tilelang2ascend-translator` | TileLang→AscendC 转译 | 复杂算子 |
| `tilelang2ascend-operator-project-init` | Kernel 工程初始化 | 所有 |
| `ops-profiling` | 性能测试分析（--compare 模式） | 所有 |
| `tilelang2ascend-trace-recorder` | 执行记录生成 | 所有 |

## 四、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 简单算子和复杂算子有什么区别？

简单算子（Index, Gather, Scatter 等）走 ops-direct-invoke 工作流：Architect 架构设计 → Developer 渐进式开发 → Reviewer 代码审查（100分制评分 + 修复循环）。复杂算子（Attention, MatMul, Norm, Sort 等）需要先做 TileLang 设计表达，再转译为 AscendC 代码。

### Q: TileLang 验证失败了怎么办？

TileLang 当前主要用于设计表达，不是 correctness gate。若 TileLang 验证失败但设计意图正确，可跳过 TileLang 验证直接进入 AscendC 转译阶段。在 trace.md 中会记录跳过原因。

### Q: AscendC 验证迭代上限是多少？

最多 3 轮迭代。超过上限后停止并报告当前状态。

---

## 总结

1. 从 PyTorch Model 出发，支持双路径（简单/复杂）端到端完成算子开发
2. Claude Code 用户用 `/plugin install` 一键安装，OpenCode/TRAE 用户用 `init.sh` 脚本安装
3. 7 Phase 工作流，自动算子分类路由，退化检测，迭代修复
4. 产出物包含设计文档/设计表达、kernel 代码、性能报告和 trace 记录
