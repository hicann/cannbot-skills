# SortSelect AscendC 输出适配

所有接口使用 [公共输出映射](../../generic/output-mapping.md)，本页仅给出语言表达。现有 [Sort example](example/sort/README.md) 是 first_rank=0、L=N 的配置；以下为生成实现中的自定义代码片段，不是新增 SDK API，也不是可直接运行的完整算子。

## 布局与排序主体

调用层按 [布局判定](../../generic/shape-preprocessing.md#物理转置的判定与验收) 规范化 `[O,N,I]→[O,I,N]`，主体只读连续尾轴。连续尾轴直接使用，等价布局用 reshape；确需改变排列时才物化转置。输出 `[O,I,L]` 各字段分别按实际 L 判定如何恢复 `[O,L,I]`，最后按接口处理 keepdim。实际条件转置和逆置换参考 example `op_api/aclnn_sort.cpp::SortProcess`；该 example 的 L=N，生成 L≠N 的实现不能直接沿用同一个输入/输出转置标志。沿用其高维交换置换时须核对实际非选择维顺序，不能机械套用三维折叠的等价条件。

排序主体及实际原语按 [Kernel 骨架](kernel-template.md) 复用。例如单核 Radix 仍调用参考 `AscendC::Sort` overload，并由 Host 查询完整 N 项 scratch；不能因为 L 小而按 L 规划排序容量。DMA 搬入使用匹配队列或 MTE2_V 依赖，最终索引恢复后再进入下列写回。

## 最终区间写回

输出区间统一，执行方式沿用各排序模板：

| 模板 | 区间选择与写回 |
|---|---|
| Merge、单核 Radix | Vector/Reg 读取目标名次、恢复原值和索引宽度，压紧到输出 UB，输出 Queue 交给 MTE3；保持纯 Vector 路线 |
| 插入、两阶段 | 保留模板原有的 SIMT 插入、逆映射或 Stage2 构键分工，以及 Vector 恢复和按批写回；按最终 B×L 域适配 |
| 多核 Radix | 在原有 SIMT 散射的末轮按最终 rank 写出区间，保留完整跨核交接 |

### Vector 路线：UB 选择、压紧与 DMA

Merge 与单核 Radix 的有序值、原轴索引已在 UB 中。先完成模板要求的原值恢复与名次映射，再按本批有效行数 v 生成紧凑输出。以下 `Vector*` 是算法操作名，具体 Reg API 按实际 dtype、索引宽度及 SDK 绑定：

```text
L=N：沿用模板的连续/多block DMA写回，不增加区间压紧
0<L<N：
  按Reg lane分块遍历p∈[0,v*L)，分别生成值/索引的有效mask
  b=p//L; rank=first_rank+p%L
  VectorGather有序索引[b*indexPitch+rank] → 目标输出索引宽度
  VectorGather已恢复的有序原值[b*valuePitch+rank] → 等宽位模式
  VectorStore到紧凑输出UB的第p项
  values/indices分别EnQue → DeQue → MTE3写出v*L项 → FreeTensor
  DMA最后读完后复用本批源区与压紧区
```

压紧目的区与尚有消费者的来源区分开；可复用已结束的物理角色，新增对齐区计入 `adaptBytes`。上述下标是逻辑元素坐标；实际 Gather 的字节偏移按各自源区元素宽度生成，中间索引宽度与输出索引宽度分别处理。值的特殊位模式在排序/恢复阶段保留；确需按原轴索引恢复时，在存活期安全的 UB 原值区上做 Vector Gather，并计入相应容量。单核 Radix 每核每次一行，v=1；Merge 保留 v 行合批。`L=1` 只减少输出项数，写回仍为 Vector/Queue/MTE3 链，`R_run=0`。

两种值来源的寻址分别为 `sortedValues[b*valuePitch+rank]` 与 `originalValues[b*originalPitch+j]`，其中 `j=sortedIds[b*indexPitch+rank]`。rank 是排序后的名次，j 是原轴位置；生成 Gather 字节偏移时先选定来源，不能混用二者。

### SIMT 路线：既有线程域内写出区间

下面片段仅表达原本使用 SIMT 的模板在最终 UB 索引恢复行归属后的批次写回，不是 Merge、单核 Radix 的共享写回入口。x 是规范化后的输入，values/indices 是长度 L 的尾轴输出；Carrier 为原值的等宽无符号类型，保留位模式。L>0、0≤firstRank≤N−L，indexPitch 足以覆盖完整行，均由 Host 校验。多核 Radix 直接在已有 scatter 中应用区间，不另调用本片段。

```cpp
struct OutputRange { int64_t n, firstRank, length; };

template<class Carrier, class Idx, bool emitValues, bool emitIndices>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void WriteRange(
    __ubuf__ Idx* ids, __gm__ Carrier* x,
    __gm__ Carrier* values, __gm__ int64_t* indices,
    OutputRange view, int64_t rowStart, int64_t rows, int64_t indexPitch)
{
    for (int64_t p = threadIdx.x; p < rows * view.length; p += 256) {
        int64_t b = p / view.length, t = p % view.length;
        int64_t r = rowStart + b;
        int64_t axis = ids[b * indexPitch + view.firstRank + t];
        int64_t dst = r * view.length + t;
        if constexpr (emitValues) values[dst] = x[r * view.n + axis];
        if constexpr (emitIndices) indices[dst] = axis;
    }
}
```

以 uint32 原值/中间索引、两个输出字段为例，`pipe` 是本核 TPipe；相应头文件为 `kernel_operator.h`、`simt_api/asc_simt.h`：

```cpp
OutputRange view{N, firstRank, L};
event_t e = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(e);
WaitFlag<HardEvent::V_S>(e);  // SIMD/段恢复完成，VF 才能读索引。
asc_vf_call<WriteRange<uint32_t, uint32_t, true, true>>(dim3(256),
    (__ubuf__ uint32_t*)finalIds.GetPhyAddr(), (__gm__ uint32_t*)x,
    (__gm__ uint32_t*)values, (__gm__ int64_t*)indices,
    view, rowStart, validRows, indexPitch);
SetFlag<HardEvent::V_S>(e);
WaitFlag<HardEvent::V_S>(e);  // 读者完成后，复用当前批次 UB。
```

其他 dtype、输出索引类型与原语布局须分别适配。若直接携带原值，可在最终搬出中应用相同区间，无需强制按索引回读。写回不改变或修复比较、NaN、零符号与稳定性语义。

### 保持批次写回粒度

对原本 B>1 合批的路径，Vector 选择或既有 SIMT VF 都按本批有效行数展开 B×L 工作域，每批或每个必要的容量块处理一次。不得仅为复用单行函数，在标量 row 循环内反复以 `rows=1` 或 `count=L` 调用 VF，再逐行搬出和等待；L=1 时，这会使每次调用可能只有一个线程处理有效数据。B=1、尾批、容量分块及有设备测量依据的粒度调整按 [通用批处理约束](../../generic/output-mapping.md#保持批处理粒度) 处理。

- L=N 时优先保留 example 的连续批量 DMA，不能因统一接口适配而无依据地改成逐行写回。
- L<N 时，Merge/单核 Radix 按 Vector 路径先在 UB 中选择、压紧再 DMA；原本使用 SIMT 的模板按各自线程域恢复和写出区间。压紧区不得覆盖其他线程尚未读取的源数据，新增缓冲须计入 Host 预算。
- UB 源行距按 N 或实际 padding 计算，GM 目标行距按 L 计算；等宽 carrier 保留原值位模式。不能通过截短排序域、丢弃尾行或放宽索引/特殊值语义减少开销。
- 同步保护整个输出块的真实依赖。VF 直写 GM 时等待 VF 完成再复用源 UB；使用 MTE3 时等待对应 DMA 消费者。不得用逐行同步替代可以保留的批次依赖，也不得为了合批删除必要同步。

按一个容量可覆盖的有效批次 v 核对：两阶段短输出先一次覆盖 v×L 的压紧，再由每个输出字段的一次连续 DMA 搬出；Merge 的 Vector 选择/压紧同样覆盖 v×L，再经 Queue/MTE3 合批写回。L=1 时域为 v 项，提交次数不随行数变成 v 次；API 长度上限确需分块时按容量块划分并保留同样的生产消费关系。

设计及代码检视需核对每批的 VF 调用、DMA 提交和同步次数，包括最后一个不满批次；不能只检查 VF 内部是否使用 `threadIdx.x`。输出适配改变这些粒度时，按 [比较替换后的代价](design-spec.md#比较替换后的代价) 记录差异并验证受影响场景。

### Scratch 与 UB 由 Host 规划

先确定排序、恢复和最终写回的完整调用链，再计算当前路线的运行保留与可用 UB。`R_simt` 按目标 SDK 和实际调用链核对；参考 Host 的 `SIMT_UB=32768` 字节是该实例的预算值，不作为通用 SDK 保证：

```text
R_run = R_simt（调用链包含 SIMT）或 0（纯 Vector 路线）
U_work = U - R_run
sum(实际对齐的非 scratch 物理 buffer 字节数) + sortScratchAllocated ≤ U_work
```

`R_run` 来自上面的模板执行方式，按整条路线计一次：Merge、单核 Radix 保持 Vector 写回，取 0；插入、两阶段和多核 Radix 按自身实际 SIMT 调用计入保留。各 buffer 的份数和别名按存活期求值。下文模板公式中的 `adaptBytes` 仅指输出适配新增、尚未计入既有物理区和 scratch 的对齐字节数。先把这些量代入模板的选批、proposal 模式和单双缓冲条件，再下发字段。参考 Host 将扣除保留后的 `U_work` 传给 `SetLocalMemorySize`，对应直调的 `dynamicUbBytes` 也使用该可用预算；Kernel 分配与 Host 容量计划一致，保留区不重复扣除。

Host 按实际 dtype、shape 和排序配置查询临时空间：

```cpp
std::vector<int64_t> dims{static_cast<int64_t>(n)};
ge::Shape shape(dims);
AscendC::SortConfig cfg;
cfg.type = AscendC::SortType::RADIX_SORT;
cfg.isDescend = true;
cfg.hasSrcIndex = false;
cfg.hasDstIndex = true;
uint32_t maxTmp = 0, minTmp = 0;
AscendC::GetSortMaxMinTmpSize(shape, ge::DT_UINT32, ge::DT_UINT32,
                            false, cfg, maxTmp, minTmp);
```

Host 使用资源接口查询的 UB 大小，令 A 为目标 block 向上对齐。以单行 uint32 值/索引为例，Vector 路线满足 `3×A(4N)+A(maxTmp)+adaptBytes≤UB`，浮点恢复区另按模板计入；其他实际含 SIMT 的路线再计相应 `R_run`。同时检查 N/输出区间、总偏移与 API 字段范围；scratch 采用查询所需最大量并按要求对齐。完整一行放不下时，应另选长轴路线，不能越界运行。

注册式 example 从 `context->GetPlatformInfo()` 构造 `PlatformAscendC`，使用 `platform/platform_info.h`。下述为独立 Host 查询方式：先初始化设备/运行时；资源查询入口为 `platform_ascendc::PlatformAscendCManager::GetInstance()`，使用 `GetCoreMemSize(CoreMemType::UB, ubBytes)` 和 `GetCoreNumAiv()`，并检查空查询对象、零容量及 API 返回结果。相关头文件为 `tiling/tiling_api.h`、`platform/platform_ascendc.h`；独立 Host 查询模块须按实际 SDK 链接 tiling/platform/register 等依赖，不能只因 C++ 编译成功就认定可以加载。

SIMT 保留由实际执行机制决定，不是所有路线的保险性扣减。改变模板的执行机制按 [实现复用与替换约束](design-spec.md#实现复用与替换约束)处理；共享输出区间参数不改变各模板的写回方式。Sort 的 dtype 路由门槛不等于公共 API 支持域。

## 长轴批次复用

长轴临时区按 P_act 行分配一次，以批次内行号寻址；输入与最终输出使用 `batchBase+r_local` 的全局行号。每批重置状态与 ping/pong 基址，保留完整 N 项的所有 digit 和原有 histogram/lookback/同步协议，上一批最后读写完成后复用 scratch。

末轮统一条件为 `firstRank≤rank<firstRank+L`，目标位置 `r_global*L+(rank-firstRank)`；仅在完整前缀和最终名次确定后判断，未输出者仍参与必要同步。多核参考按 N 选择 uint32/int64 计数和前缀，按 4/8 字节预算。中间索引宽度取计数宽度与输出索引宽度中的较小者，末轮才转换为最终输出类型；复用输出缓冲时必须匹配这两种行距。具体空间见 [Buffer 预算](dag-buffers.md)，验证见 [统一批次内存](../../generic/liveness-fusion.md#统一批次内存)。

## 同步与直调配置

- 上述 `SetFlag/WaitFlag<HardEvent::V_S>` 等待 SIMD/VF 的生产者或读者完成，再由标量控制流推进；它只覆盖对应的 VF/Vector 路径。
- MTE2 搬入使用 `MTE2_V` 或匹配的队列同步；若改用 MTE3 写回，还须在覆盖/复用前等待相应 DMA 消费者。不能照抄一个 `V_S` 代替所有路径事件。
- 直调入口使用 `aclrtLaunchKernelWithArgsArray` 和 `ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`，传入与 Tiling 一致的动态 UB 预算。注册式 Host 的对应配置为 `SetLocalMemorySize`，不能假设普通直调自动继承。
- 多核协作路线还须按实际模板配置调度模式；上述核内 API 用法不包含多核 Radix 启动配置。
- 系统 workspace 预留与 UB 分开预算。使用 `GetUserWorkspace` 时，按其实现核对 `RESERVED_WORKSPACE` 用户区偏移；分配须覆盖系统预留与用户区，不能仅缩小分配或传空指针。

下面列的是目标 SDK 内部的 API 定义位置，不是要求复制到示例的 include 路径；example 通过 `basic_api/kernel_vec_intf.h` 等入口头文件间接包含部分接口，接入时按实际 SDK 搜索路径核对。API 真值位置：目标 CANN 的 `kernel_operator.h`、`simt_api/asc_simt.h`、`adv_api/sort/sort_tiling_intf.h`、`acl/acl_rt.h`；对应实现为 [RadixOneCore](example/sort/op_kernel/arch35/sort_radix_one_core.h)、[小轴搬入](example/sort/op_kernel/arch35/common/small_axis_insertion_base.h) 和 [Radix 依赖](example/sort/op_kernel/arch35/common/radix_more_core_base.h)。运行时保留区核对安装树的 `impl/basic_api/.../kernel_operator_common_impl.h` 与 `impl/basic_api/utils/kernel_utils_constants.h`。

直调 Host 中通过真实运行时 API 设置动态 UB，不另提供示例启动库。`kernelEntry` 为本算子 kernel 地址，`args` 按其实际签名准备，`stream`、`blocks` 和 `dynamicUbBytes` 均由本工程提供：

```cpp
aclrtLaunchKernelAttr attr{};
attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
attr.value.dynUBufSize = dynamicUbBytes;
aclrtLaunchKernelCfg cfg{&attr, 1};
aclError ret = aclrtLaunchKernelWithArgsArray(kernelEntry, blocks, stream, &cfg, args);
// 检查 ret；启动成功不等于异步执行已完成，按工程运行时协议同步并检查错误。
```

Host 使用 `acl/acl_rt.h`。调用方须准备合法输入/输出和运行时要求的系统 workspace；资源查询/tiling 与 kernel 按各自模块的实际 SDK 依赖构建，不使用 example 的 include 路径，也不硬编码 UB、核数或 scratch 数值。
