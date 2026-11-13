# SortSelect 实现导航

按已选语言定位算法、Host 参数和 Kernel 的实现位置；接口与资源约束仍以对应适配层为准。

设计时先读 [AscendC 模板导航](adapters/ascendc/overview.md#按模板阅读) 或 [DSL 模板导航](adapters/dsl/index.md#按模板读取)：每条路线的 Tiling、Buffer、Kernel 已分别给出公式、伪码、参数含义及算例。本页在需要核对具体 API 或实现绑定时使用，不替代模板文档。

## AscendC

下表路径相对 [Sort 示例目录](adapters/ascendc/example/sort/)。

| 文件 | 核心锚点 | 对应功能 |
|---|---|---|
| `op_host/arch35/sort_tiling_arch35.cpp` | `SelectSortSchedule`、`TryMerge`、`IsRadixSortOneCore` | 路由优先级、容量约束与算法选型 |
| `op_host/arch35/sort_tiling_arch35.cpp` | `SelectSmallAxisRouteImpl`、`PrepareTwoStageBatchCandidate`、`ComputeInsertionBytesPerSeg`、`SelectRadixCorePlan` | dtype/轴长/并行度选型、紧凑键、UB 预算与 Radix 行/轴分核 |
| `op_host/arch35/sort_tiling_arch35.h` | `kSmallAxisRules`、`SortKthTileInfo`、`RADIX_INT64_VALUE_MAX` | 路由档位、参数和计数状态位上限 |
| `op_kernel/` 入口文件、`op_kernel/arch35/sort_tiling_key.h` | `sort`、`LaunchSortKernel`、`SORT_TPL_KEY_DECL` | schId、计数宽度、方向与 Kernel 实例分派 |
| `op_kernel/arch35/common/small_axis_insertion_base.h` | `SimtInsertionSortSegments` | 段内插入排序、相等键次序与索引移动 |
| `op_kernel/arch35/common/small_axis_two_stage_base.h` | `RankInverseScatter`、`BuildStage2KeysSimt`、`RunTwoStageSort` | 批次排序后的逆映射与紧凑键恢复 |
| `op_kernel/arch35/sort_merge_sort.h` | `FlipSignBit`、`SortRowsBf16`、`InitMergeIndicesVF`、`SortBatchedRows` | 批量向量构键/转换、索引初始化与归并行批；前后处理也按此复用 |
| `op_kernel/arch35/common/ping_pong_merge_sort.h` | `SortToProposal`、`MergeStage` | Sort32 初始 run、四路归并、ping/pong 与尾组 |
| `op_kernel/arch35/common/radix_more_core_base.h` | `ComputeOnePass`、`LookbackGlobal`、`ProcessRadix` | LSD 轮次、直方图、前缀发布、稳定散射与栅栏 |
| `op_kernel/arch35/common/radix_sort_simd_utils.h` | `TwiddleInFp16Impl`、`TwiddleInFp32Impl` | 排序键转换、零等键与条件 NaN 处理 |
| `op_kernel/arch35/common/signed_zero_sort_utils.h` | `HasNegativeZeroVec`、`PrepareSignedZeroKeysVec`、`RestoreSignedZeroValuesByIndexVec` | Reg 零符号检测、构键与按索引恢复；适用路线按 caller 核对 |
| `op_api/aclnn_sort.cpp` | `SortProcess`、`BuildSortGraph` | L2 连续化、条件转置、末轴排序与输出还原 |

选路条件见 [Tiling](adapters/ascendc/tiling.md)，字段单位和 Host/Kernel 对应关系见 [Tiling 参数](adapters/ascendc/tiling-parameters.md)。L2 的布局能力与单个 Kernel 的支持域分别核对；稳定性和 NaN 行为按实际调用参数确认。

## CANNBotDSL

下表路径相对 [Sort 示例目录](adapters/dsl/example/sort/)。

| 文件 | 核心锚点 | 对应功能 |
|---|---|---|
| `entry.py` | `PreparedSort`、`two_stage_batch` | 公开接口、选路、批次和缓冲分配 |
| `sort_host.py` | `sort_entry` | Host 模板分派与启动 |
| `radix.py` | `radix_permutation`、`sort_local_keys` | 字节直方图、稳定分组、局部排列与 LSD 排序 |
| `sort_small_axis_insertion.py` | `InsertionSort.__call__`、`contiguous`、`strided` | 连续路径 DMA 输入、每线程行内排序、合作写回；跨步路径协作加载与三次 barrier |
| `sort_small_axis_two_stage.py` | `SmallAxisTwoStage.__call__` | 合批键/索引 LSD、inverse 计秩或稳定行号分组（公开单字节，低层 B>256 为两字节）、SIMT 原值回读 |
| `sort_radix_one_core.py` | `RadixOneCore.__call__`、`NarrowRadixOneCore.__call__` | 32/64 位键/索引双缓冲，以及 1/2 字节局部排列与原值恢复 |
| `sort_radix_more_core.py` | `clear_states`、`prepare_digits`、`load_value_tile`、`extract_value_digit`、`load_digit_word`、`histogram`、`scatter` | DMA/SIMD 构键、统计、无等待 lookback、稳定散射与同步 |
| `sort_transpose.py` | `SortTranspose`、`transpose_entry` | 调用层输入布局规范化和输出还原 |

模板组织见 [实现指南](adapters/dsl/implementation-guide.md)，dtype 档位、恢复策略与容量公式见 [Tiling](adapters/dsl/tiling.md)。Sort、Argsort、TopK、KthValue 按 [统一输出映射](generic/output-mapping.md) 配置输出字段和名次区间。
