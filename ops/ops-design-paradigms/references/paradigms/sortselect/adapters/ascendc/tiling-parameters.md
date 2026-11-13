# SortSelect AscendC Tiling 字段对应

本表解释 [Sort 示例](example/sort/README.md) 的 Host/Kernel 协议，不要求派生算子沿用这些字段名。`p0..p5` 对应 `keyParams0..5`；“—”表示该路线不消费该字段，不保证 Host 写入值为零。所有字节数和元素数必须分开处理。

## 按 schedule 解释复用字段

| schId / 路线 | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| 5 / 插入 | 每批行数 | 批次任务数 | — | 0 | 0 | — |
| 6 / 两阶段 | 每批行数 B | 批次任务数 | rank-inverse 标志 | 0 | 0 | — |
| 0 / MergeSort | 每批行数 | 单个输入/输出 value 队列字节数 | 单个 index 队列字节数 | 对齐后轴长（元素） | 队列 buffer 数 1/2 | batchMerge 标志，仅 half/int64 分支使用 |
| 1 / RadixOneCore | 单个 value 队列字节数 | 单个 index 队列字节数 | halfIndex：以 uint32 元素计的偏移 | 队列 buffer 数 1/2 | — | — |
| 2 / RadixMoreCore | global histogram 清零核数 | exclusive bins 清零核数 | 每次清零的 CounterT 元素数 | histogram 清零循环次数 | 每核 bins 清零元素数 | 每核 histogram 清零元素数 |

sch1 的 halfIndex 是 `alignBlock(N×4)/4`，不能误当字节数；输出为 int64 时 index 队列扩容，偏移仍按 uint32 元素解释。sch2 的 p2 是每次清零的 CounterT 元素数，消费时乘 `sizeof(CounterT)` 转为字节数。多核 CounterT 为 uint32/int64，p2 因而以 4/8 字节元素计；输出索引类型仍独立选择。

## 公共字段与布局字段

- `(schId,isInt32,isDescend)` 是模板键；多核的 `isInt32=1/0` 分别表示 32/64 位计数和前缀，按 N 是否超过 2^30−1 选择；其他 schedule 保留各自约束，不表示输出 indices 一定是 int32。
- `tmpUbSize` 为该路线实际分配的临时 UB 字节数：两阶段取 Stage1/Stage2 查询需求的对齐最大值，单核 Radix 按匹配的 SDK 查询；多核 Radix 在查询需求之上按剩余 UB 扩容，还供清零和 scatter 参数复用；Merge/插入的参考实现为 0。查询需求与最终分配值分别记录，不跨路线复用。
- `numTileDataSize`、`lastDimTileNum` 等公共字段也由路线解释。按实际路线的 Host 写入与 Kernel 读取核对，不能只凭字段名推断单位。
- 非末轴布局由调用层规范化；全部输出配置共用连续尾轴主体，输出区间、长度 L 和批次基址按生成实现的统一合同传递。

## Host 写入与 Kernel 消费位置

路径相对 [Host 目录](example/sort/op_host/arch35/) 和 [Kernel 目录](example/sort/op_kernel/arch35/)。核对时成对读取，避免只看共用结构或旧注释。

| 路线 | Host 锚点 | Kernel 文件 / 消费点 |
|---|---|---|
| 插入、两阶段 | `FillSmallAxisBatched` 与 `TrySmallAxis` | `sort_small_axis_insertion.h`、`sort_small_axis_two_stage.h` 的 Init |
| MergeSort | `FillMergeSortInfo → SetMergeSortTiling` | `sort_merge_sort.h::Init / ParseTilingData` |
| RadixOneCore | `IsRadixSortOneCore` | `sort_radix_one_core.h::ParseTilingData` |
| RadixMoreCore | `FillRadixKernelParams / FillRadixKernelResources` | `sort_radix_more_core.h::ParseTilingData`、`common/radix_more_core_base.h` 清零代码 |

改动布局或分阶段复用时，一并核对 [容量预算](dag-buffers.md) 与 [Kernel 骨架](kernel-template.md)。
