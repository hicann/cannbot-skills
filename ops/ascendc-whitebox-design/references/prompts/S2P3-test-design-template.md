# S2P3_test_design.md 模板

## 用途

源码分析完成后，根据此模板生成 S2P3_test_design.md。
输入来源：参数推导结果（`S2P2_param_def.json`）+ 常见网络 shape（`low_configs`）+ 用户确认结果。

## 必填章节

```markdown
# {OpName} 白盒测试设计

## 1. 输入概览
| 输入类别 | 是否提供 | 说明 |
|----------|----------|------|
| torch 接口 | 是/否 | |
| tiling 代码 | 是/否（框架代理） | |
| kernel 代码 | 是/否 | |
| 资料描述 | 是/否 | |

## 2. 事实摘要
| 项目 | 结论 | 来源类别 | 类型 |
|------|------|----------|------|

## 3. 代码路径全景

从 tiling 入口到 kernel 叶子节点的完整分支树（主 Agent 根据 S2P1_path_list.json 的 paths 数组和 source_constraints 重建）：

---```
{op_name} ({平台}路径)
├── 条件 ...
│   └── [路径名] ...
│       ├── 子条件 ... → 函数/指令
│       └── 子条件 ... → 函数/指令
└── 条件 ...
    └── ...
---```

共 {N} 条路径（N 必须等于 S2P1_path_list.json paths 数组长度），分为 {M} 个 group（M 必须等于 S2P2_param_def.json groups 数组长度）。

**约束**：
- 每条路径必须单独列出，禁止折叠（如"内部复用各 tiling mode"的概述写法不允许）
- 路径 ID 必须与 S2P1_path_list.json 中的 id 一一对应
- 路径总数 N = S2P1_path_list.json paths 数组长度（不含 disputed）
- group 总数 M = S2P2_param_def.json groups 数组长度

## 4. 关键派生变量
| 变量 | 公式 | 依赖项 | 是否参与分支 | 来源 |
|------|------|--------|--------------|------|

## 5. 测试关注点（groups）

每个 group 自包含：路由条件 + 约束 + 维度表 + 预估组合数。

**约束**：
- group 列表必须与 S2P2_param_def.json groups 数组完全一致（数量、id、顺序）
- 每个 group 中的维度值（dtype、epsilon、ndim 等）必须与 S2P2_param_def.json 中对应 group 的 params 一致
- 禁止在 S2P3_test_design.md 中引入 S2P2_param_def.json 中不存在的维度值
- 每个 group 的 constraints 描述必须与 S2P2_param_def.json 中对应 group 的 constraints 语义一致

### 5.N {group_id}
**路由条件**：{什么条件进入此路径}
**约束**：{此路径下的参数限制}

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| {dim_name} | {values or thresholds} | {core_split / ub_tile / instruction_align / attr} | {source:line} |

**预估组合数**：估算 ~{N}

（重复 5.1, 5.2, ... 每个 group）

## 6. 执行模式分析

### 轴映射
| 执行层级 | 映射轴 | 控制变量 | 来源 |
|---------|--------|---------|------|
| 分核 | {axis_name} | {CeilDiv(dim, coreNum)} | {source} |
| UB 切分 | {axis_name} | {UbFactor 公式} | {source} |
| 指令对齐 | {axis_name} | {BLOCK_ELEM / VL 常量} | {source} |

### 三层覆盖策略
| 层级 | 模式 | 触发条件 | 对应维度取值 |
|------|------|---------|-------------|
| 分核 | 未开满核 | dim < coreNum | {dim}=1 |
| 分核 | 开满核无尾核（单倍） | dim == coreNum | {dim}={coreNum} |
| 分核 | 开满核无尾核（多倍） | dim == k*coreNum | {dim}=2*coreNum, 4*coreNum |
| 分核 | 开满核有尾核 | dim % coreNum != 0 | {dim}={coreNum+1} |
| UB | 单 pass | 数据量 <= UbFactor | {dim} 取小值 |
| UB | 多 pass + 尾块 | 数据量 > UbFactor 且不整除 | {dim} 取大值 |
| 指令 | 对齐 | dim % vectorWidth == 0 | {dim}={aligned_value} |
| 指令 | 非对齐 | dim % vectorWidth != 0 | {dim}={k*v-1 展开值} |

## 7. 未确认项

以下内容无法从当前输入中确认，**需要用户决定**：

| # | 问题 | 原因 | 建议处理 |
|---|------|------|---------|
| 1 | {问题} | {为什么不确定} | 忽略 / 需要补充信息 / 需要额外测试 |

Step 2 完成后，将此表展示给用户，等用户逐条确认或补充后再继续。

## 8. 设计估算
| 项目 | 值 | 说明 |
|------|----|------|

## 9. 验证结论
（Step 3 完成后由 verifier 填写）
```

## 通用维度指引

### data_range

每个 group 应包含 `data_range` 维度（除非算子有特殊限制），控制输入 tensor 的数据值域：

```json
"data_range": ["normal", "zero", "extreme", "negative", "tiny_pos", "all_ones", "near_zero", "with_inf", "with_nan"]
```

| 标签 | 含义 | 测什么 |
|------|------|--------|
| normal | torch.randn 正态随机 | 一般场景 |
| zero | 全零 | 零值传播、除零保护 |
| extreme | 接近 dtype 最大值 | 溢出、饱和 |
| negative | 全负数 | sigmoid/silu 负值分支 |
| tiny_pos | 极小正数（~1e-6） | 精度损失、scale 除零 |
| all_ones | 全 1 | 恒等验证 |
| near_zero | 接近零的正负混合 | 符号翻转、舍入 |
| with_inf | 正常数据中混入 inf | inf 传播处理 |
| with_nan | 正常数据中混入 nan | nan 传播处理 |

不需要全部包含——根据算子语义选择有意义的。量化算子至少需要 normal/zero/extreme/negative。

### ndim

如果算子支持多种 rank（如 ndim 2~8），加为维度：

```json
"ndim": [2, 3, 4]
```

pytest 代码根据 ndim 构造不同 rank 的 tensor（如 ndim=2 → `[batch, D]`，ndim=4 → `[B, N, S, D]`）。如果算子固定 ndim（如必须 4D），不加此维度。
