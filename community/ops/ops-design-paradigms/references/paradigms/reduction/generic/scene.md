# Reduction 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/overview.md §1/§2/§3/§4（原 reduction-template-overview.md）、原范式入口 patterns.md 范式总览。TilingKey/TilingData 的具体实现（模板宏、struct 定义）在适配层。

## 1 准入条件

本范式覆盖的算子需满足：

1. **跨元素归约语义**——输出元素由输入沿归约轴（R 轴）折叠计算得到（sum/mean/max/min/prod/any/all 及其复合），归约结果沿非归约轴（A 轴）保留
2. **归约轴可标记**——输入张量的每一根轴可按算子语义标记为 A 轴（preserve）或 R 轴（reduce）
3. **输出 shape 由 A 轴决定**——归约后 R 轴消失（keep_dims 场景 R 轴变 1，仍按消失处理）

典型场景：单/多轴 reduce、全轴归约（All Reduce）、含归约的复合计算（reduce 前/后有 elementwise 变换）。仅 element-wise 无归约的算子走其他范式。

## 2 坐标系与术语（语言中立）

| 术语 | 含义 |
|------|------|
| **A 轴** | 非 reduce 轴（preserve），归约后保留 |
| **R 轴** | reduce 轴，沿此轴归约 |
| **pattern** | 合轴预处理后轴的类型序列，形如 `A R A R …`（A 起头、严格交替） |
| **TailA / TailR** | 合轴后尾轴是 A（如 `ARA`）/ R（如 `ARAR`），决定片上数据排布与归约方向 |
| **All Reduce** | 所有轴都 reduce 的算子，合轴后 pattern 恒为 `AR` |
| **空 tensor** | shape 中含 0 的输入，分 EMPTY_A（A 轴含 0）/ EMPTY_R（R 轴含 0 且 A 全非 0） |
| **post_reduce_input** | 归约后阶段的额外输入（如 mean/variance 等统计量），shape 与输出相同（A 轴，size = aTotal） |
| **aTotal** | ∏(所有 A 轴 size)，输出元素总数 |
| **MAX_PATTERN_RANK** | 本算子合轴后 pattern 的最大可能轴数（推导见 [axis-preprocessing.md](axis-preprocessing.md) §5） |

坐标系经合轴预处理归一（见 [axis-preprocessing.md](axis-preprocessing.md)）：A 起头、A/R 严格交替、除补位轴外每轴 size ≥ 2。轴类型由位置奇偶唯一决定（i 偶 → A，i 奇 → R）。

## 3 模板分类

按多核切分轴分三类模板（每算子选 1~3 个）：

| 模板 | 使用场景 | 准入条件 | 多核切分 |
|------|---------|---------|---------|
| **Base** | 通用归约主路径 | 非空 tensor，A 轴并行度足够开多核 | 仅 A 轴参与分核，核间无依赖，核内完成全部 R 轴归约 |
| **Group** | A 轴小时借 R 轴补并行度 | 非空 tensor，A 用不满核且 R 有并行度 | A×R 2D 分核，核间有依赖，经中间存储二次归约 |
| **Empty** | 空 tensor 简化路径 | 空 tensor（EMPTY_A / EMPTY_R 共用） | EMPTY_A 全核早退；EMPTY_R 按 a 元素数切分 |

**模板选择键**只有两个维度：是否 Group、是否 Empty tensor。**dtype 不进选择键**——由框架按输入 dtype 注入编译实例；axisNum 不进选择键——运行时分支处理。选择键的落地机制（模板参数/分发宏）由语言适配层定义。

**统一树形累加**：所有归约算子（sum/mean/max/min/prod/any/all）统一使用二分缓存树累加（见 [tree-reduction.md](tree-reduction.md)）。对 sum/mean 有精度收益（浮点累加误差 O(M)→O(log M)）；对 max/min/any/all 结果不变（满足结合律）；prod 因浮点舍入不满足结合律、结果可能有差异，但统一路径简化模板设计。

## 4 数据流骨架

计算图按三段式划分，两个桥接节点连接：

```
┌─ 归约前段（preReducePhase）──────────────────────────────┐
│  输入 → 搬入 → PreElewise → preReduceResult（可选旁路输出）│
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌─ 归约段（reducePhase）───────────────────────────────────┐
│  preReduceResult → 片上归约 → 树缓存 → reduceResult（树根）│
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌─ 归约后段（postReducePhase）─────────────────────────────┐
│  reduceResult + 可选 post 输入 → PostElewise → 搬出        │
└───────────────────────────────────────────────────────────┘
```

- **preReduceResult / reduceResult** 是两个桥接节点；归约前段与归约后段**独立**做存活节点分析与融合分析，不跨归约段融合
- **Base 模板**单阶段完成三段；**Group 模板**拆两阶段：Phase 1 = 归约前段 + 局部 R 段归约（partial 结果写中间存储），Phase 2 = 剩余 R 维归约 + 归约后段，两阶段经跨核栅栏衔接（见 [sync-model.md](sync-model.md) §5）
- **Empty 模板**无归约：EMPTY_A 零计算零 IO；EMPTY_R 以固定值填充（`empty_r_output_value`，由算子语义决定，如 sum=0 / max=identity）替代归约，后接可选 PostElewise

## 5 编译期特化

- 特化维度只有**模板选择键**（是否 Group / 是否 Empty），与 dtype 无关（dtype 由框架注入）
- 合轴后 pattern 轴数上限 MAX_PATTERN_RANK 决定 pattern 描述数组的定长

## 6 能力边界

1. 归约精度：树形累加对结合律运算（max/min/any/all）结果不变；对 sum/mean 有精度收益；prod 有浮点舍入差异——产品语义若要求严格顺序累加，不在本范式 standard 路径内
2. 本范式不做切分之外的调度优化（Group 触发判定即 A 方向并行度不足时的借 R 补并行度策略）
3. 空 tensor 语义由算子级判定：空 axes（axesNum==0）按 all-reduce 还是 noop 展开必须给出单一结论（见 [axis-preprocessing.md](axis-preprocessing.md) §1）
4. 确定性：固定切分 + 固定树形累加顺序 → 同设备同输入多次执行 bitwise 可复现

## 7 依赖谓词

| 谓词 | 用途 |
|------|------|
| `REDUCE_OP` | 片上归约指令实体（三段式中归约段的执行者） |
| `TREE_CACHE_CAPACITY` | 树形累加缓存的容量约定 |
| `COMPUTE_PRECISION` | 归约统一计算精度 |
| `ALIGN_GRANULARITY` | 片上数据对齐粒度 |
| `CROSS_CORE_FENCE` | Group 两阶段跨核栅栏 |
| `WORKSPACE_MODEL` | Group 中间存储（跨核 partial 结果传递） |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
