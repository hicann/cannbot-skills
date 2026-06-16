# AscendC Ops Developer 开发指南

## 概述

本 Team 实现从 PyTorch Model 到 AscendC Kernel 的端到端自动开发流程。

## 工作流架构

```
用户输入 (npu, op_file, output_dir)
    │
    ▼
CANNBot Primary Agent (AGENTS.md)
    │  解析参数、调度 Subagent
    ▼
ascend-kernel-developer Subagent
    │
    ├── Phase 0: 参数确认 + 算子分类（简单/复杂）
    ├── Phase 1: 环境准备
    ├── Phase 2: Case 精简 (tilelang2ascend-case-simplifier)
    ├── Phase 3: 设计表达（分支）
    │     ├─ 简单: ops-direct-invoke 架构设计 + 设计串讲
    │     └─ 复杂: TileLang 设计 (tilelang2ascend-tilelang-designer) ← 迭代循环
    ├── Phase 4: AscendC 生成与验证（分支）
    │     ├─ 简单: ops-direct-invoke 开发实现 + 代码审查 + 修复循环（3轮上限）
    │     └─ 复杂: AscendC 转译 (tilelang2ascend-translator) ← 迭代循环（3轮上限）
    ├── Phase 5: 性能分析 (ops-profiling --compare 模式)
    ├── Phase 6: 全量验证
    └── Phase 7: Trace 记录 (tilelang2ascend-trace-recorder)
```

## Phase 3-4 迭代机制

### Phase 3: TileLang 设计迭代

```
generation → AST退化检测 → [通过] → 功能验证 → [通过] → 完成
                ↓ 失败                        ↓ 失败
            Conductor分析 → 生成修复建议 → 重新生成
                                     ↑
                        最多 5 轮迭代 ─┘
```

退化子类型：
- Type1: 无 TileLang kernel 导入（纯 PyTorch）
- Type2: 有导入但 forward() 未调用
- Type3: 调用 kernel 但部分计算仍用 PyTorch
- Type4: forward() 中存在逐元素 for 循环

### Phase 4: AscendC 转译迭代

```
generation → AST退化检测 → [通过] → 功能验证 → [通过] → 完成
                ↓ 失败                        ↓ 失败
            Conductor分析 → 生成修复建议 → 重新生成
                                     ↑
                        最多 3 轮迭代 ─┘
```

退化子类型：
- Type1: 无 AscendC 扩展导入（纯 PyTorch）
- Type2: 有导入但 forward() 未调用 kernel
- Type3: 调用 kernel 但部分计算仍用 PyTorch
- Type4: forward() 中存在逐元素 for 循环

## 错误分类

| 分类 | 含义 | 处理 |
|------|------|------|
| A 类 | 代码逻辑/算法错误（可修复） | 生成修复建议，继续迭代 |
| B 类 | 环境/基础设施错误（不可修复） | 立即终止 |
| C 类 | 同一 A 类子类型连续 ≥ 3 次 | 立即终止 |

## 产出物

```
{output_dir}/
├── model.py                  # 算子描述（只读）
├── <op_name>.json            # 原始测试用例文件（备份保留）
├── <op_name>.json.bak        # 原始 .json 备份
├── design/
│   ├── block_level/          # Block-level 设计
│   └── tile_level/           # Tile-level 设计
├── kernel/                   # AscendC kernel
├── model_new_tilelang.py     # TileLang 实现
├── model_new_ascendc.py      # AscendC 实现
├── preformance.json          # 性能数据
└── trace.md                  # 执行记录
```

## 关键约束

- TileLang 仅用于设计表达，不作为 correctness/performance gate
- model_new_*.py 中禁止使用 torch 算子
- 必须将核心计算融合成单个算子
- 文件操作范围限制在 {output_dir}/ 内
- 优先使用块级/向量化操作，避免标量逐元素写法
