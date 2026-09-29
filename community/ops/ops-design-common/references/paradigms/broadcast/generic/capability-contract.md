# Broadcast 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法中依赖硬件/语言能力的量，统一抽象为**能力谓词**。通用层文档只引用谓词名，不出现任何具体语言 API。每个语言适配层必须在其 `capability-mapping.md` 中对全部谓词给出真值。

## 1 使用规则

1. **通用层**：算法描述中遇到能力依赖点时，引用本文档中的谓词名，不写具体数值或 API 名
2. **适配层**：`adapters/<lang>/capability-mapping.md` 逐项填写谓词真值与对应的 API 实体
3. **可填性检验**：新语言适配层必须能对每个谓词给出真值。填不出真值说明通用层抽象有缺陷，应修订抽象而非绕过
4. **冲突处理**：同一能力点若适配层真值与通用层假设冲突（如某语言搬运不支持随路广播），以适配层真值为准，并在该能力点标注设计影响（如退化为独立广播步骤、P 值增加）

## 2 谓词定义表

| 谓词 | 类型 | 含义 | 影响的通用层知识点 |
|------|------|------|------------------|
| `CAP_BROADCAST_IN_TRANSFER` | bool | 搬入单元是否支持广播轴随路展开（源侧 stride=0 原地重读，搬运时直接展开到目标形态） | scene 数据流路线；liveness-fusion 搬运随路广播不占额外 buffer |
| `CAP_REG_CHAIN_FUSION` | bool | 计算单元是否支持寄存器链融合（链式中间值留在寄存器，不落片上 buffer） | liveness-fusion 中 P < L 的合法性 |
| `CAP_FUSED_ARITH` | list | 硬件融合指令集合（一条指令完成两步运算、目的操作数就地覆写、省 1 个中间 buffer） | liveness-fusion 融合决策的候选集 |
| `ALIGN_GRANULARITY` | bytes | 片上 buffer 大小的对齐粒度 | splitting perBufBytes 公式 |
| `TRANSFER_MAX_DIMS` | int | 多维搬运单条指令的维数上限 | splitting 维度分层；搬运策略分叉（全维单次 vs 逐段） |
| `FUSION_CHAIN_MAX_LEN` | int | 单条融合链的操作数上限 | liveness-fusion 融合判据 1 |
| `CONVERT_BREAKS_CHAIN` | bool | 精度转换操作是否打断融合链 | liveness-fusion 融合判据 2；compute-flow dtype 分路径步骤数 |
| `CONVERT_INPLACE` | bool | 精度转换可否原地完成（src/dst 同一块 buffer） | liveness-fusion 额外 buffer 需求（转换中转 buffer） |
| `COMPUTE_PRECISION` | type | 范式统一采用的计算精度类型 | splitting perBufElems 公式；liveness-fusion 转换策略 |
| `BUFFER_MODEL` | enum | 片上 buffer 管理模型（显式单缓冲 / 队列式管理） | sync-model 手动同步需求；liveness-fusion 单缓冲策略 |
| `SYNC_PRIMITIVE` | desc | 跨执行单元同步原语及其事件管理规则 | sync-model 同步点落地方式 |
| `EXEC_UNITS` | desc | 执行单元划分（搬入单元 / 计算单元 / 搬出单元） | sync-model RAW/WAR 事件对 |

## 3 谓词使用示例

通用层 splitting 文档中的写法：

> `perBufBytes = floor(片上容量 / P)` 按 `ALIGN_GRANULARITY` 向下对齐

AscendC 适配层（capability-mapping.md）中的填值：

> `ALIGN_GRANULARITY` = 32 字节（`ONE_BLK_SIZE`）

DSL 适配层若对齐粒度为 16 字节，则填 16，通用层公式无需任何修改。
