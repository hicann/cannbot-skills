# SortSelect AscendC Buffer 与依赖

以下预算对应 AscendC Sort 的 Host 与 Kernel。令 U 为查询到的 UB 字节数，A(x) 为向上对齐 UB block 字节数，D 为输入元素字节数，J 为输出索引字节数，B 为批次行数。R 为按目标 SDK 和实际调用链核对的 SIMT 保留量，参考 Host 的 `SIMT_UB` 取 32768 字节；仅使用 SIMT 的路线扣除 R，Merge/单核 Radix 的 Vector 路线取 0。参考常量不作为所有 SDK 的固定保证。

## 按模板读取

下面的汇总用于候选比较；具体设计按命中模板逐项展开物理分配和存活区间：

| 模板 | 完整内存说明 | 关键复用点 |
|---|---|---|
| 插入 | [Buffer](templates/insertion/dag-buffers.md) | BF16 搬运/转换区、独立行 pitch、批末搬出完成 |
| 两阶段 | [Buffer](templates/two-stage/dag-buffers.md) | input→finalValues、key2/负零标记→finalIndices、Gather 偏移 |
| 归并 | [Buffer](templates/merge/dag-buffers.md) | Queue 与 proposal 的同时存活、合批模式下 proposal 扩容 |
| 单核 Radix | [Buffer](templates/radix-one-core/dag-buffers.md) | 索引后半区→int64、sourceOrder 不随双缓冲倍增 |
| 多核 Radix | [Buffer](templates/radix-more-core/dag-buffers.md) | 六段 GM、清零覆盖长度、P_act 行槽及 Sort scratch→scatter 参数 |

## 小轴与 Radix 容量

| 路线 | UB 检查 | 实现位置 |
|---|---|---|
| 插入 | 普通每行 `A(ND)+A(NJ)`；BF16 每行 `A(2N)+2×A(2N)+A(NJ)`，包含原始 BF16 和 fp32 转换区。`B×每行字节数+R≤U` | `ComputeInsertionBytesPerSeg` |
| 两阶段 | E=B×N，`2×A(ED)+q×A(4E)+A(FE)+scratch+R≤U`；逆映射 q=2，否则 q=3。F 为 final index UB 元素宽度，按输出索引类型预算 | `EstimateTwoStageUbBytes`、`ComputeTwoStageSortTmpUb` |
| RadixOneCore | X=A(ND)，Y=A(4N)×(J/4)，浮点 Z=A(4N)、整数 Z=0；`b×(2X+Y)+Z+scratch≤U`，b 为 1 或 2。Z 保存浮点 sourceOrder，不随队列倍增 | `IsRadixSortOneCore`、`sort_radix_one_core.h::Init` |
| RadixMoreCore | S 为计数宽度：N≤2^30−1 取 4，否则取 8（中间索引队列按 S 保守分配），T 为 256 的倍数；`T×(D+S+6)+256×(3S+4)+scratch≤U−R`。逐步缩减 T 并重新查询 uint8 排序 scratch，再平衡多核分块 | `FillRadixMoreCoreInfo` 及其资源计算辅助函数 |

scratch 使用目标 SDK 的 `GetSortMaxMinTmpSize`，按实际 dtype、排序算法、shape、显式/隐式索引配置查询，采用实际调用所需空间并对齐。两阶段查询 Stage1 原 dtype；紧凑键分支还查询 Stage2 uint16，复用区取两者最大值。不能用固定 bytes/element 代替查询，也不能把粗略候选上界作为最终预算。

## 归并的峰值预算

`p0..p5` 指 [Tiling 参数](tiling-parameters.md) 中的 `keyParams0..5`。proposal 为 8 字节记录；普通 tensor 偏移不能替代 proposal offset。

| 路线 | 同时存活的空间与复用规则 |
|---|---|
| MergeSort（sch0） | 队列 `p4×(2p1+p2)`；索引及双 proposal 区 `H×(4+2×8)`。通常 H=p3，half/int64 且 batchMerge 时 H=p3×p0。BF16 另加 `p3×p0×4` 转换区。Host 公共候选计算还可能保守预留其他输出空间，以 Sort 最终候选和实际 `InitBuffer` 双向核对。 |

Host 预算入口：[Sort Host](example/sort/op_host/arch35/sort_tiling_arch35.cpp)。Kernel 分配见 [MergeSort](example/sort/op_kernel/arch35/sort_merge_sort.h)。改变 buffer 数量、转换类型或复用关系时，同步修改 Host 预算。

## GM workspace 与依赖

多核 Radix 分别预算 exclusive bins、global histogram/status、全局索引、tile histogram/prefix、字节键和中间 values；计数/前缀及中间索引工作区按 S=4/8 字节预算；中间索引实际读写宽度为 min(S, 输出索引字节数)，局部 uint8 键、uint16 桶计数和 uint32 局部排列不因此加宽。元素宽度及对齐由 `ComputeRadixSortWorkspace` 确定。常规核内 MergeSort 的 proposal ping/pong 放在 UB，按实际行数与对齐轴长预算。框架另有 16 MiB workspace 预留，它是框架约定，不是算法固有需求，也不能算作可用 UB。

复用前必须满足最后读写依赖：两阶段输入区在 Stage1 最后读后才别名为最终 values；布局转换缓冲须等待最后消费者完成后再复用；归并 ping/pong 的输入输出不能同层别名。counts/validBit 描述实际尾 run。Radix 的状态位占用从计数范围扣除。

SyncAll、事件和 GM 状态发布有不同作用，不能互相替代。通用依赖见 [sync-model](../../generic/sync-model.md)。

## 统一输出与批次工作区

所有配置按 [统一批次内存](../../generic/liveness-fusion.md#统一批次内存) 分开列出最终 M×L 输出、transpose 布局缓冲和排序 scratch。上述原语预算仍按完整 N 项计算；核内持有当前 UB 批次，多核外存按 P_act 活跃行复用。模板以最终输出兼作中间存储时，生成实现须将 scratch 基址与最终输出行偏移分离，两侧实际存活区均计入。

仅在最后一个消费者结束后复用空间；末轮最终名次确定后才按输出区间写回。Merge/单核 Radix 保持 Vector 选择、压紧和 Queue/MTE3 写回，新增 UB 区计入 adaptBytes 后重选容量；原有 SIMT 路线计入各自运行保留。实际表达见 [输出适配](output-mapping.md)。
