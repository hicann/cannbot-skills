# AscendC Sort 示例

本例沿指定轴返回排序后的 values 和原轴内 indices。L2 接口处理连续化、轴变换及输出还原，排序 Kernel 只处理连续末轴。示例包含五种算法路线，不包含 AiCPU 回退。

这里实现完整 Sort（`first_rank=0、L=N`，同时输出值和索引）。Argsort、TopK、KthValue 的生成方法见 [公共输出映射](../../../../generic/output-mapping.md) 和 [AscendC 输出适配](../../output-mapping.md)；本目录未提供这三种独立接口。

## 建议阅读顺序

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | [aclnn_sort.cpp](op_api/aclnn_sort.cpp)、[sort.cpp](op_api/sort.cpp) | L2 如何规范化布局，L0 如何提交末轴排序 |
| 2 | [Sort Tiling](op_host/arch35/sort_tiling_arch35.cpp)、[对应头文件](op_host/arch35/sort_tiling_arch35.h) | 参数校验、性能档位、UB 预算、分核和工作区；均集中在 Tiling 实现中 |
| 3 | [Kernel 入口](op_kernel/sort.cpp) | TilingKey 如何分派到五种模板 |
| 4 | 下表中的算法模板 | 数据加载、排序、原轴索引恢复及写回 |

`op_host/arch35/sort_tiling.cpp` 负责注册和平台资源查询；每次 Tiling 读取当前设备的核数与 UB，CompileInfo 不缓存这些资源。`op_host/sort.cpp` 的 `opFile.value=sort` 与 Kernel 入口文件名一致。

## 五种算法路线

以下 schedule 编号用于本例分派，详细门槛见 [Tiling 说明](../../tiling.md#小轴路由)。

| schedule | 模板 | 核心思路 |
|---|---|---|
| 0 | [MergeSort](op_kernel/arch35/sort_merge_sort.h) | Sort32 生成初始有序段，再用 MrgSort 四路归并 |
| 1 | [RadixOneCore](op_kernel/arch35/sort_radix_one_core.h) | 每核处理可放入 UB 的行，调用局部 Radix 排序 |
| 2 | [RadixMoreCore](op_kernel/arch35/sort_radix_more_core.h) | 多核协作处理长轴，按字节计数、求前缀并稳定散射 |
| 5 | [Insertion](op_kernel/arch35/sort_small_axis_insertion.h) | 批量处理极短轴；轴长为一也走此路线，比较循环为空 |
| 6 | [TwoStage](op_kernel/arch35/sort_small_axis_two_stage.h) | 将一批行展平排序，再用逆排名或 uint16 紧凑键恢复各行顺序 |

选路先检查经验性能门槛，再检查实际缓冲容量；不能把某个阈值当作硬件支持上限。没有独立的 Sort32 路线，Sort32 仅作为 MergeSort 内部原语。

## 多核 Radix 如何阅读

从 `SortRadixMoreCore::Init` 看工作区分配，再读 [RadixMoreCoreBase](op_kernel/arch35/common/radix_more_core_base.h) 的 `Process`。设 M 为行数、N 为轴长、T 为分块元素数，P_act 为同时处理的行数：每批复用 P_act 行工作区，每行由若干核协作。

每个字节轮次先生成块内桶计数和独占前缀，再通过全局桶偏移与 lookback 得到跨块前缀，最后散射。目标位置由「全局较小桶元素数 + 此前块的同桶元素数 + 本块桶内排名」组成，因此相同键保持输入次序。首轮用块起点生成原轴索引，后续轮次随值一起传递该索引。

多核计数与前缀按轴长选择：N≤2^30−1 使用 uint32，否则使用 int64。两种宽度共享流水，32 位全局桶偏移使用 DMA 原子加，64 位使用 SIMT 原子加。中间索引宽度不超过最终输出索引宽度，末轮再转换；输出 dtype 独立决定。边界与测试限制见 [多核选路及验证](../../tiling.md)。`ScatterOutput<OutputIndex, FirstPass>` 集中维护散射调用，类型和首轮差异由模板表达。散射的三个偏移暂存在 Sort 已释放的 scratch 中，以共享具名槽位读写；Vector→Scalar→Vector 事件及散射后的等待保护这段复用，不能仅为简化代码删除。具体约束见 [辅助处理与同步](../../kernel-helpers.md)。

## 接口与布局

公开签名见 [aclnn_sort.h](op_api/aclnn_sort.h)：支持 FLOAT16/BF16/FLOAT32、INT8/16/32/64、UINT8/16/32/64，以及在 L2 转为 UINT8 的 BOOL；公开 indices 输出为 INT64。L2 还处理空张量、标量和非连续输入。底层 OpDef 支持 INT32/INT64 索引，不能将其支持域直接等同于公开 API。

连续末轴直接排序。rank≤8 且轴后仅有 singleton 维度时，通过 reshape 暴露尾轴；其余非末轴由置换转入、排序后还原 values/indices。rank>8 先折叠维度再处理。索引数值始终是原排序轴坐标，不需要转换为扁平地址。生成新接口时，应按 [布局判定](../../../../generic/shape-preprocessing.md#物理转置的判定与验收) 分别检查输入和输出，尤其不能把 Sort 的输出转置条件直接套给 L=1 的选择操作。

## 接入与验证

本目录是算法与接口组织参考，不提供独立构建、打包工程，也不作为跨芯片工程模板。生成算子时应接入目标仓库的构建系统，根据目标能力核对 OpDef 注册、Host 选路和 Kernel 原语；不能仅修改 compute unit 就推定可移植。

接入需提供 CANN SDK、`aclnn_kernels`、opdev 及 Transpose/Cast/ZerosLike/TensorMove 等公共算子依赖。使用 `Ops::Base` 的日志/类型格式化函数时链接 `ops_base`；公共算子按目标 SDK 的实际符号归属链接相应分库。公开 API 使用手写 `op_api/aclnn_sort.h` 的签名，不能直接用通用 codegen attr 签名替换。

本机工程适配、版本、原始日志及个案结果保留在库外。NPU 验证不在仓库普通 CI 中执行。接入后的验收至少包括：

- 各可达 schedule 的 values、稳定原轴 indices 和输入未修改；以实际 TilingKey/schedule 证明命中，不能仅凭 shape 或用例名推断。
- two-stage 的两种恢复分支，以及非末轴和尾随 singleton 布局。
- L0 非法参数的拒绝与调用方错误传播；直接编译 L0 的源码级测试不能代替安装库的拒绝路径证据。
- 加载的 API/tiling 库和设备产物属于当前实现，避免同名系统 Sort 回退掩盖问题。

Argsort、TopK、KthValue 未提供独立接口或设备验收；生成后须单独覆盖 first_rank/L 边界及输入/输出布局。完整 dtype/shape、超大索引及性能矩阵也须按目标规格验证。
