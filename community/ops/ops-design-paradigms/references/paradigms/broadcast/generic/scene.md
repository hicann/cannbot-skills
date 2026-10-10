# Broadcast 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/overview.md §1/§2/§6（原 broadcast-template-overview.md）、原 patterns.md 范式总览。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **element-wise 计算语义**——输出元素由输入元素的逐点函数计算得到
2. **多输入间存在 broadcast**——输入 shape 不一致，需广播对齐后计算
3. **计算公式不含 Reduce**——无跨元素归约（归约类见 Reduction 范式）

典型场景：输入输出 shape 不一致需广播对齐的 element-wise 算子。

## 2 坐标系理论

所有输入/输出统一到同一个**广播上界坐标系**下描述：

| 术语（语言中立） | 含义 |
|------|------|
| **effective_shape** | 去 1 补 1 后的归一化 shape，所有输入/输出的统一坐标系 |
| **maximumBroShape** | broadcast 上界 shape，作为片上内存切分与多核切分的坐标系 |
| **B 轴（广播轴）** | 某张量在该维大小为 1 而坐标系大小 >1 的轴；偏移计算中该轴 **stride=0**，实现随路广播 |
| **A 轴（非广播轴）** | 两端等长的轴 |
| **RANK** | effective_shape 的维度数，编译期特化的唯一参数 |
| **split 轴（ubSplitIdx）** | 片上内存单切分轴，将 effective_shape 分为内轴与外轴 |
| **perBufElems / perBufBytes** | 单个片上 buffer 可容纳的元素数 / 字节数 = 片上容量 / P |
| **L（逻辑存活节点）** | 计算图 tensor 层面的峰值同时驻留数 |
| **P（物理存活节点）** | 片上 buffer 层面的峰值同时驻留数，决定 buffer 槽位数 |

广播语义通过 **stride=0** 表达：坐标到偏移的映射 `offset = Σ coord[d] × stride[d]` 中，B 轴 stride 为 0，则该轴坐标变化不产生偏移——源数据只读一份，目标形态按坐标系展开。

## 3 数据流骨架

本范式 standard 路径的数据流为三段式，多输入并行搬入、多输出独立搬出：

```
[段 1: 搬入（随路广播）]   每个输入: GM → 片上 buffer，B 轴 stride=0 随路展开
[段 2: 计算]             逐元素计算；若 CAP_REG_CHAIN_FUSION=true，链式中间值留在寄存器
[段 3: 搬出]             每个输出: 片上 buffer → GM，独立搬出
```

- 随路广播依赖谓词 `CAP_BROADCAST_IN_TRANSFER`：广播在搬入阶段完成，不占额外 buffer，不产生独立的广播步骤
- buffer 采用**单缓冲按 P 均分**策略：buffer 数 = P，每块 `perBufBytes = floor(片上容量/P)` 按 `ALIGN_GRANULARITY` 对齐；角色在搬入 / 中间结果 / 搬出之间轮转复用（详见 [liveness-fusion.md](liveness-fusion.md)）

## 4 编译期特化

- 特化维度只有 **RANK**，与数据类型无关；dtype 由框架按主输入类型注入，不在特化键中编码
- RANK 分两档：常态档（低维）与保底档（高维），实际 rank 落入低档即选低档
- 特化的落地机制（模板参数、分发宏等）由语言适配层定义

## 5 能力边界

1. **广播仅发生在搬入阶段**（GM → 片上）。若数据已在片上（中间计算结果）且需要广播形态，不在本范式 standard 路径内——此时需按算子整体数据流重新归入范式（AscendC 下的替代路线见 `paradigms/common/broadcast-rules.md` 路线 2）
2. 本范式不做片上内存切分与多核切分之外的调度优化（切分微调预留扩展点，见 [splitting.md](splitting.md) §5）
3. 广播不改变计算语义：所有精度、舍入行为与 element-wise 基础范式一致

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_BROADCAST_IN_TRANSFER` | 数据流路线成立的前提（§3 段 1） |
| `CAP_REG_CHAIN_FUSION` | 段 2 融合策略与 P 值压缩的前提 |
| `ALIGN_GRANULARITY` | buffer 预算对齐 |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
