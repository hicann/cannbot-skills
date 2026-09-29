# Reduction 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法中依赖硬件/语言能力的量，统一抽象为**能力谓词**。通用层文档只引用谓词名，不出现任何具体语言 API。每个语言适配层必须在其 `capability-mapping.md` 中对全部谓词给出真值。

## 1 使用规则

1. **通用层**：算法描述中遇到能力依赖点时，引用本文档中的谓词名，不写具体数值或 API 名
2. **适配层**：`adapters/<lang>/capability-mapping.md` 逐项填写谓词真值与对应的 API 实体
3. **可填性检验**：新语言适配层必须能对每个谓词给出真值。填不出真值说明通用层抽象有缺陷，应修订抽象而非绕过
4. **冲突处理**：同一能力点若适配层真值与通用层假设冲突（如某语言归约指令不支持 src 复用），以适配层真值为准，并在该能力点标注设计影响（如 buffer 份数增加、需显式清零策略）

## 2 谓词定义表

| 谓词 | 类型 | 含义 | 影响的通用层知识点 |
|------|------|------|------------------|
| `REDUCE_OP` | desc | 片上归约指令实体：支持按 pattern 方向（AR=沿内层归约 / RA=沿外层归约）做二维块归约的指令族，含其 scratch buffer 约定与 src 可否复用语义 | scene 三段式数据流；liveness-fusion 归约段 buffer 份数（3 份：src + scratch + dst） |
| `REDUCE_OP_PRECISION` | desc | 归约指令内部的累加方式（顺序累加 vs 内部树形折叠），决定单次归约的精度上界 | tree-reduction 两级树精度论证 |
| `TREE_CACHE_CAPACITY` | bytes | 树形累加缓存的容量约定（范式统一固定值），树缓存的预算上界 | splitting UB 预算（先扣固定树缓存）；tree-reduction R 全载钳制 |
| `COMPUTE_PRECISION` | type | 归约统一采用的计算精度类型（低精度输入先扩位到此精度再归约） | splitting buffer 容量（max dtype 宽度）；liveness-fusion Cast 参与链 |
| `ALIGN_GRANULARITY` | bytes | 片上数据搬运/存储的对齐粒度 | splitting 尾轴 CeilAlign/FloorAlign；valid/padded 双字段 |
| `CACHELINE_SIZE` | bytes | 片上访问 cache line 宽度，决定单次驻留片上的目标元素量上界 | splitting UB 切分联合爬坡上界 |
| `SINGLE_BUF_MAX` | bytes | 单个片上 buffer 的容量上限 | splitting Empty 切分约束 c |
| `FUSION_CHAIN_MAX_LEN` | int | 单条融合链的操作数上限 | liveness-fusion 融合判据 1 |
| `CONVERT_IN_CHAIN` | bool | 精度转换可否串入融合链（寄存器内完成，不额外占片上 buffer） | liveness-fusion Cast 对 P 的影响；融合链头部/尾部的扩位/缩位 |
| `FUSED_ARITH` | list | 硬件融合指令集合（一条指令完成两步运算、目的操作数就地覆写） | liveness-fusion 融合决策候选集 |
| `REG_CHAIN_FUSION` | bool | 计算单元是否支持寄存器链融合（链式中间值留在寄存器不落片上） | liveness-fusion 中 P < L 的合法性 |
| `BUFFER_MODEL` | enum | 片上 buffer 管理模型（显式单缓冲 / 队列式管理）与 double buffer 支持情况 | sync-model 手动同步需求；splitting UB 预算倍数 |
| `SYNC_PRIMITIVE` | desc | 跨执行单元同步原语及其事件管理规则（事件对、事件编号获取方式） | sync-model 同步点落地方式 |
| `EXEC_UNITS` | desc | 执行单元划分（搬入单元 / 计算单元 / 搬出单元） | sync-model RAW/WAR 事件对 |
| `CROSS_CORE_FENCE` | desc | 跨核栅栏原语（全部核到齐才放行）及其调度模式要求 | scene Group 两阶段衔接；sync-model §5；tuple-reduce Group 轮末栅栏 |
| `WORKSPACE_MODEL` | desc | 跨核中间存储（workspace）的申请约定（系统部分 + 用户部分）与布局 | splitting Group workspace 分配 |
| `MULTICORE_BALANCE` | desc | 多核负载均衡模型（大小核式：前 k 核多担 1 份，最大负载差 ≤ 1） | splitting A 方向分核与 Empty 切分 |
| `SCHEDULE_2D` | bool | 是否支持 A×R 2D 分核调度模式 | splitting Group 2D 分核 |
| `BLOCK_DISPATCH_MIN` | int | 核调度发起的最小 block 数（使用核数为 0 时仍须按此值发起，kernel 侧短路） | splitting Empty：EMPTY_A 的发起方式 |

## 3 谓词使用示例

通用层 splitting 文档中的写法：

> `ubAvailable = 片上容量 − TREE_CACHE_CAPACITY`（先扣固定树缓存）

AscendC 适配层（capability-mapping.md）中的填值：

> `TREE_CACHE_CAPACITY` = 16 KB（cacheBuf，二分缓存树专用）

DSL 适配层若树缓存为 8 KB，则填 8 KB，通用层预算公式无需任何修改。
