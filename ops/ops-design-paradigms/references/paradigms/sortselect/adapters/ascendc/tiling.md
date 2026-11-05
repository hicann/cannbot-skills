# SortSelect AscendC Tiling 与分核

本页说明 AscendC Sort 的 Host 策略。dtype 档位、轴长阈值和批次偏好是平台调参，不是算法硬约束。平台核数 C、UB/block/VReg 大小须查询；容量检查见 [Buffer 预算](dag-buffers.md)，字段映射见 [Tiling 参数](tiling-parameters.md)。令 N 为轴长、M=O×I 为序列数、P=ceil(M/C)。

## 全局规划流程

公共参数和五套模板见[整体模型](overview.md#按模板阅读)。本页维护路线优先级和经验门槛；每个模板的 Markdown 分别承载完整切分、内存和计算步骤，不需要先展开 example 才能推导方案。

```text
归一化 axis、输出字段、first_rank/L 和连续尾轴布局
若输出为空：调用层返回；否则取得 M、N、dtype、J、C、UB 与对齐
按 dtype 读取小轴经验档位：
  先枚举两阶段批次，逐候选查 scratch、对齐和紧凑键范围
  未选中再计算插入批次，检查行 pitch、容量和搬运 blockCount
未选中小轴：尝试长度门槛内的归并，规划行批次及 proposal 模式
未选中归并：检查完整单行 Radix 的 scratch、索引宽化和单/双缓冲
未选中单核：为多核 Radix 联合搜索 T/G/P_act，生成清零及 workspace 计划
只在候选副本中计算；选定后提交最终模板键、blockDim、局部内存及字段
```

| 路线 | 独立推导文档 | 必须产出的量 |
|---|---|---|
| 插入 | [templates/insertion/tiling.md](templates/insertion/tiling.md) | value/index 行 pitch、B、Q、C_act、尾批 |
| 两阶段 | [templates/two-stage/tiling.md](templates/two-stage/tiling.md) | 恢复分支、精确 scratch、B 搜索、轮数/空闲槽 |
| 归并 | [templates/merge/tiling.md](templates/merge/tiling.md) | 对齐轴长、行批次、队列份数、合批 proposal 条件 |
| 单核 Radix | [templates/radix-one-core/tiling.md](templates/radix-one-core/tiling.md) | 队列字节数、halfIndex、scratch、单/双缓冲 |
| 多核 Radix | [templates/radix-more-core/tiling.md](templates/radix-more-core/tiling.md) | 计数宽度、T/G/P_act、清零分工与六段 workspace |

## 小轴路由

下表 `(n,p)` 档位取自 [Sort Tiling 头文件](example/sort/op_host/arch35/sort_tiling_arch35.h) 的 `kSmallAxisRules`，属于当前参考实现的经验性能参数，用于选择排序路线，不代表 API 支持范围或算法正确性边界。调整档位须结合目标 dtype、轴长和行数进行性能比较，并独立满足容量及表示约束。

其中 n 是轴长 N 的上限，p 是每核行数指标 P=ceil(M/C) 的下限；P 是选路指标，不是实际批次行数 B。N=1 走插入路线，不受下表并行度门槛限制；仍须通过资源检查。其余小轴先找到首个 N≤n 的档位，再要求 P≥p。不能越过该档位，借后续档位更低的 p 放行。`SelectSmallAxisRouteImpl` 优先尝试两阶段，再尝试插入，均须通过容量和批次合法性检查。

| dtype | 插入 N 上限 | 插入档位 `(n,p)` | 两阶段 N 上限 | 两阶段档位 `(n,p)` |
|---|---:|---|---:|---|
| int64 / uint64 | 16 | (8,1), (16,4) | 512 | (15,8), (128,4), (512,8) |
| int32 / uint32 | 11 | (8,2), (11,4) | 384 | (11,8), (64,4), (384,8) |
| int16 / uint16 | 8 | (4,2), (8,4) | 192 | (7,8), (64,4), (192,12) |
| int8 / uint8 | 8 | (4,2), (8,7) | 128 | (3,8), (64,7), (128,16) |
| float32 | 8 | (4,16), (8,48) | 24 | (24,64) |
| float16 / bfloat16 | 8 | (4,16), (8,48) | 54 | (54,64) |

恢复分支与选路档位分开理解。紧凑键为何受 B²N 约束，见 [键范围推导](../../generic/splitting.md#两阶段紧凑键的范围)：

| 类别 | 条件 |
|---|---|
| 恢复分支策略 | `UseTwoStageRankInverse` 对所有 dtype 统一采用 N≤64；更长轴使用 Stage2 紧凑键。 |
| 打包键硬约束 | B 行、每行 N 个元素时，最大键 B²N−1 必须可用 uint16 表示。 |
| Host 上限 | `MaxTwoStageU16SafeBatch` 要求 B²N≤65536，允许最大键恰为 65535；按当前表示核对，不将键空间大小与最大键值混用。 |
| 资源与表示约束 | B×N、任务数及偏移不溢出目标字段；精确查询排序 scratch 后总 UB 可容纳。粗略 bytes/element 只能求候选上界。 |

插入排序使用 sch5，每批行数 B 取插入选择上限、UB 可容纳行数和 DataCopy blockCount 上限 4095 的最小值，并满足 B≤P；任务数 tasks=ceil(M/B)。

## 两阶段排序

完整的容量上界、精确查询、候选伪码、字段和算例见[两阶段 Tiling](templates/two-stage/tiling.md)。下面保留全局选路时需要的批次策略摘要。

两阶段排序使用 sch6。L2 完成布局规范化后，按连续行确定每批行数 B 的上限和任务数：B≤P，tasks=ceil(M/B)。原轴位置不再参与批次映射或搜索策略选择。

批次选择从候选上界递减到 1，逐个检查布局、紧凑键和 scratch；final index UB 按输出索引宽度预算。非末轴的布局转换不增加排序模板键。

两阶段排序的批次搜索以首个可行 B 的轮数 R=ceil(tasks/activeCores) 为基准，只在相同 R 内最小化空闲槽 `R×activeCores−tasks`；遇到不同 R 结束搜索，同分保留较大 B，不能简化成“总取最大 B”。实现锚点为 [Sort Tiling](example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `PrepareTwoStageBatchCandidate`、`SearchTwoStageBatchPlan`，以及 [Sort Host](example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `TrySmallAxis`。

## 归并与 Radix

小轴未命中后，连续末轴依次尝试 `TryMerge → TryRadixOneCore → RadixMoreCore`。以下是 Host 候选门槛，不代表 API 的完整支持域；每一项还须通过具体资源规划。

| 候选 | 门槛与选择 |
|---|---|
| MergeSort（sch0） | float32 N≤4096；float16 N≤2048；bfloat16 N≤1536。N≤32 也使用同一 MergeSort 模板，不单列 Sort32 路由。`TryMerge` 扩展了公共 helper 对半精度的 1024 阈值，以实际调用链为准。 |
| RadixOneCore（sch1） | 要求 `isInt32` 非零；原生 Sort scratch、value/index 队列与浮点零符号恢复区可容纳。再决定单/双缓冲；输出索引仍可为 int64。 |
| RadixMoreCore（sch2） | 前述路线未命中且分块/工作区规划成功即进入，不要求 N≥2^30。N≤0x3fffffff 时采用 uint32 计数/前缀（isInt32=1，key 258/65794），更长轴采用 int64（isInt32=0，key 2/65538）；UB、256 桶状态、GM workspace 与清零参数随计数宽度 S=4/8 计算。 |

`IsRadixUint32CounterRange` 以 `0x3fffffff` 为单核 uint32 候选上界，同时决定多核计数宽度。多核 lookback 将桶计数或前缀与同步状态放在同一个状态字中：高两位记录状态，低位保存非负数值。因此 32 位只剩 30 位计数，最大为 `2^30−1`；单桶计数或前缀可能达到整行 N，故 N 超过此值时切换到 64 位。64 位同样保留高两位，数值上限为 `2^62−1`（`0x3fffffffffffffff`）。状态定义和掩码见 [radix_sort_constants.h](example/sort/op_kernel/arch35/common/radix_sort_constants.h)。这是当前状态编码的边界，不是硬件轴长上限或进入多核路线的门槛；实际轴长还受资源与其他字段范围限制。输出索引 dtype 独立选择。选路完成后以最终 `info.isInt32` 编码 key，不能复用进入选路前按 N 得到的标志。

FP16/INT64 的 MergeSort 合批还要求 B≥4，且 `B×align32(N)≤255×32=8160`，对应 Sort32 的 uint8 repeat 次数容量；另须满足两份提案、索引和 I/O 队列的 UB 预算。未满足时仍可使用逐行 MergeSort。见 `SetMergeSortTiling` 与 `SortBatchedRows`。

### 多核 Radix 的工作粒度与行并行

完整联合搜索、重查 scratch 和清零计划见[多核 Radix Tiling](templates/radix-more-core/tiling.md)；六段实际分配见[对应 Buffer](templates/radix-more-core/dag-buffers.md)。

example 的 `SMALL_TILE_DATA_NUM=1024` 是每核轴内工作量的经验起点，生成实现须保留这一调参含义；256 是桶数及 tile 对齐单位，不能将它当作默认每核工作量。1024 不是 API 支持下限，也不要求每个尾 tile 都达到该长度。

本节用 Kt 表示每行 tile 数、Lt 表示每核每批处理的 tile 数，避免与 TopK 的 K、输出长度 L 混淆。联合选择每行核数 G、并发行数 P_act 和 tile 长度 T。先按 UB 及真实 uint8 Sort scratch 求可容纳的最大 tile T_cap；每行核数只在 `1..min(C,max(1,floor(N/1024)))` 内取候选。每个候选令 `T=min(T_cap,align256(ceil(N/G)))`、`Kt=ceil(N/T)`，实际 G 不超过 Kt；再计算 `P_act=min(M,floor(C/G))`、批次数 `R=ceil(M/P_act)`、每核每批最多处理的 tile 数 `Lt=ceil(Kt/G)`。

example 的默认启发式先最小化串行 tile 轮次 `R×Lt`；相同时选择较小的 T，以减少相同轮次的补齐元素数，再选择较少的 R。它同时考虑单核串行工作和跨行批次，不把行优先或占满单行核数设为固定规则。选中 T 后重新查询实际 scratch 并检查预算；不满足则缩小容量上限后重选。`activeCores=P_act×G`，核组内 lane 按 `tileId=lane; tileId<Kt; tileId+=G` 遍历，沿用既有 histogram/lookback/scatter 协议。

例如 C=56 时，单行 N=8192 最多分配 8 个轴内核，目标 T=1024；M=37 的长轴则同时比较每行 1、2、4 等核数对应的单核 tile 数和总批次，不能直接固定为每行 1 核。不要为让 tile 数接近总核数而继续缩小 T。计算函数见 [Sort Tiling](example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `SelectRadixCorePlan`、`ComputeRadixTileData` 和 `FillRadixMoreCoreInfo`。

UB 容量优先于经验粒度：容量只允许更小 T 时保留合法分块，整行不足 1024 或不足整 tile 时只处理有效元素，不能用虚构数据量满足粒度。并发行数改变后，按最终 P_act 按计数宽度重算状态清零、双侧 N 项中间区和 workspace；这些临时区只为活跃核组分配并跨批复用，不为全部 M 行常驻分配。

## 非末轴布局适配

所有接口共用布局规范化，不按长轴/短轴增加非末轴专用排序模板；连续尾轴直接使用，等价布局必须 reshape，仅改变元素排列时执行物理 transpose。输入与输出分别按 [布局判定](../../generic/shape-preprocessing.md#物理转置的判定与验收) 处理，主体输出长度 L 后按需还原所需字段。实际 API 参考 [L2 包装](kernel-entrance.md#api-与非末轴转置包装)，L0 只接受规范化后的末轴。实际需要的输入转置区、长度 L 的输出布局区、最终输出和 scratch 分别预算。P_act 是实际活跃行数，与本页路由量 P=ceil(M/C) 不同，临时区按 P_act 分配复用，见 [输出适配](output-mapping.md)。

## 多核 Radix 的验证规模

常规多核输入在 N≤2^30−1 时自然命中 32 位 key 258/65794；64 位 key 2/65538 的自然轴长边界是 N=2^30。覆盖 dtype/方向、输出区间的首项/中间/末项/全长、整 tile/尾 tile、多行/跨批次、特殊值和同地址更新重跑。测试专用构建可用小输入强制实例化 64 位流水，验证其算法路径，但不能计作自然选路或真实高索引覆盖。

单列宽计数与长轴义务：Host/算术单测检查 2^30 前后、状态位数值域、地址/字节乘积和资源拒绝；能够分配的大规模代表用例验证真实高索引和长轴设备行为。小规模同模板测试不冒充高位行为验证，描述符/注入测试也不计设备精度。每种资源证据及未完成项分别记录。两种计数宽度的 UB/GM 开销和性能分别测量。若白盒流程要求每个 key 自然命中，恢复双宽度后仍须履行大轴设备验证义务；资源不足应明确记录未覆盖项，不能以小输入替代。
