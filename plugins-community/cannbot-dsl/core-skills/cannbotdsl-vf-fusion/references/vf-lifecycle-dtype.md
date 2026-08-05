# VF、生命周期和 dtype 门禁

## 适用场景

- 写或重构 `with vf()`、AIV 侧向量融合、UB 内 cast/expand/mul/add/reduce。
- 混合 AIC/AIV kernel 中存在 GM、UB、L1、L0C/FIXPIPE handoff。
- 输出或中间 workspace 使用 BF16/FP16，但 L0C 计算为 FP32，需要判断最终 GM 累加 dtype。

> **与 Channel 的关系**：跨迭代状态 / 跨核 handoff（AIC↔AIV 交接）**优先用 Channel（channel-first）**。本文的手写 M→FIXPIPE / FIXPIPE→M / FIXPIPE→MTE2 sync 适用于 Channel 推不出的场景（编译器不认识的 op、一次性 handoff）。

## 写 VF 前必须确认

1. 先读 `../cannbotdsl-vf-fusion/SKILL.md`。
2. 一个 `with vf()` 应表达一个逻辑向量公式，优先把同一公式内可连续执行的 vector op 放进同一个 VF；同类成熟算子的写法只能作为风格参考，不能替代当前公式分析。
3. `with vf()` 内只放 vector op 和支持的 UB<->UB `mem_copy`（如 nd2nz format convert，被 `decompose-mem-copy` 展成 `ub_copy`/`ub_format_convert`）；**GM↔UB / UB↔L1 等跨 mem_loc 搬运**、跨 PIPE 同步 op、sync、matmul 必须在 VF 外。即"跨 mem_loc 搬运在外，UB↔UB 可进"。
4. VF 读写到的 owning buffer 带跨 PIPE 同步语义；view 只作为 op operand，不作为生命周期 owner。
5. `outputs=[...]` 包含 VF 中被写的 buffer owner。不要只列最终输出，遗漏中间写入 buffer。
6. 不要把一个公式拆成多段 VF 来“保守”。只有已经证明存在别名、依赖或 lowering 限制时才拆分，并在注释或调试记录里写明原因和验证结果。

## UB 别名和输入输出

- 如果同一个 BF16 UB 同时承载原始输入和最终输出，先确认 lowering 不会把读写重排成别名风险。
- 有风险时使用角色清晰的两个 UB，例如 `ub_<name>_input_bf16`、`ub_<name>_output_bf16`，中间计算 buffer 按实际 dtype 命名。
- 命名按数据语义和生命周期，不用 `lhs/rhs` 这类纯代数名，除非当前文件就是通用 algebra helper。
- 注释说明 producer/consumer：谁把 GM 数据搬到输入 UB，VF 计算什么，谁把输出 UB 写回 GM/L1。

## dtype 和 GM 累加表

写代码前把下表补齐，尤其是 L0C->GM add：

| 项 | 必填内容 |
|----|----------|
| 公开输入 dtype | 调用方传入 dtype 和 layout |
| on-chip 计算 dtype | UB/VF、L1/L0A/L0B、L0C 的实际 dtype |
| workspace/GM dtype | 每个 GM 输出或 scratch 的元素类型 |
| 最终输出 dtype | 调用方看到的输出 dtype |
| add/atomic dtype | L0C2GM add 的 atomic setter 或等价语义 |
| CPU reference dtype | golden 每一步 cast/round 的位置 |

注意：

- 硬件 matmul 的 L0C 结果通常是 FP32，不代表最终 GM 输出或 GM atomic add 也必须是 FP32。
- CANNBotDSL 当前 L0C->GM add 翻译中，copy engine dtype 描述 L0C 源路径；atomic setter 由 GM 目标元素类型决定，BF16 GM 对应 `set_atomic_bf16()`。
- 改 BF16/FP16 GM add 路径后，必须用 codegen 单测确认 atomic setter、`set_atomic_add()`、`set_atomic_none()` 三者都存在且顺序正确。

## L0C/FIXPIPE/GM 生命周期

- L0C 结果被 FIXPIPE 读出前，先用 `M -> FIXPIPE` sync 保护生产完成。
- FIXPIPE 写 GM/UB/L1 后，如果同一个 L0C 缓冲区后续会被 M 复用，必须用 `FIXPIPE -> M` sync 保护 L0C 复用。
- 如果 FIXPIPE 写 GM 后同 kernel 内还要用 MTE2 读同一 GM 区域，另加 `FIXPIPE -> MTE2` sync；这只保护 GM 可见性，不等价于保护 L0C 缓冲区复用。
- 间歇精度失败如果只出现在 L0C2GM add 的输出上，优先审计上述三条同步，再检查数学公式。
- 同一 kernel 内多个 L0C 视图的总物理字节数必须按硬件容量核算。一个 128x128 FP32 tile 已经占满常见 L0C 容量时，不能再用 `Channel(MemLoc.L0C, ..., depth=2)` 申请双级或额外 L0C；应用 depth=1 顺序复用，由 Channel 保护 M/FIXPIPE 生产消费顺序。单块无同步 L0C scratch 才使用 `Buffer`。

> **raw-VF 操作数序签名里看不出来**（`vmadd`/`vexp_sub`/窄化 `vcast`/`vreduce`/`vload_brc`），猜错即静默算错。写 VF 前先把要用的 raw op 语义用 20 行探针钉死 —— 测操作数序用三个互不对称的值。详见 `../SKILL.md` 陷阱 10。

## 注释要求

- 注释用中文说明当前计算流程、公式、buffer 生命周期和分支职责。
- 分支注释说明这个分支生产或消费什么，例如某个 AIV 分支写状态基底，某个 AIC 分支等待 ready 后执行矩阵更新或 GM 累加。
- 不用“参考某实现”“对齐某实现”替代当前代码自己的语义说明。来源可以写在 commit/todo 中，代码注释只保留当下数据流。
- 融合 VF 前写公式注释，说明 VF 中每个主要输出的含义。

## 验证顺序

1. `py_compile` 或最小导入。
2. codegen 单测，检查 VF 是否按预期聚合，L0C2GM add 的 atomic dtype 是否正确。
3. 小 shape NPU correctness。
4. 目标 shape NPU correctness。
5. 如果曾出现间歇精度失败，用唯一 cache tag 重复跑目标 shape；不要只凭一次 pass 下结论。
