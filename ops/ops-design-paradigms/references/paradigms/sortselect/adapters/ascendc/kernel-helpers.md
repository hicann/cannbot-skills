# SortSelect AscendC 辅助约束

排序主体按规范化后的连续行寻址，UB 内各值/索引行距按实际对齐分别计算；索引始终表示原轴位置。分块排序、归并和 Radix 每轮必须同时维护排序键与索引的对应关系。输出索引类型与内部计数类型分别核对，不能沿用其他选择算子的固定 int64 假设。

原生 proposal 布局使用 `GetSortOffset`；四路 MrgSort 的 counts/validBit 反映真实有效 run。尾块 padding 与合法极值、无穷和 NaN 的区分按 wrapper 实现，不能套用仅对有限值成立的哨兵规则。

## 辅助处理并行度

排序前后处理默认保留 example 的批量并行方式，不能只复制排序原语，再将构键、转换和恢复重写为逐行单线程循环。MergeSort 的直接参考为 [sort_merge_sort.h](example/sort/op_kernel/arch35/sort_merge_sort.h)：

| 处理 | 复用起点 |
|---|---|
| 升序键转换 | `FlipSignBit` 使用批量 Vector `Adds`；不是逐元素标量翻转 |
| BF16 比较类型转换 | `SortRowsBf16` 按当前批次 `Cast`，再进入相同归并主体 |
| 索引与行批 | `InitMergeIndicesVF` 使用 Reg；`SortBatchedRows` 保留其适用条件和批内行归属恢复 |
| 零符号恢复 | [signed_zero_sort_utils.h](example/sort/op_kernel/arch35/common/signed_zero_sort_utils.h) 提供 Reg 检测、构键与按索引恢复；按实际 caller 核对，不能直接套用到未覆盖的路线 |

已有路径不满足目标 NaN、padding 或原值位模式合同时，只适配对应步骤。独立转换、分类和 gather 按当前批次分配给 Vector/Reg lane 或多个 SIMT 线程；SIMT 使用 `p=threadIdx.x; p<total; p+=threads` 分工，不能启动多个线程后让除 thread 0 外的线程全部返回。常数规模标量控制和已有小轴插入的线程分段算法保留。

构键时先用批量 `Cast` 完成必要的比较类型转换，再用位运算、比较 mask 和 Select 处理符号、零与目标 NaN 键；原值位模式保留在独立存活区或原输入。采用 Reg 时按后文使用具名 VF，SIMT 独立转换也按元素分工。已有原语输出满足合同则直接截取；确需回读原值时，按最终区间的 B×L 项并行 gather，不为输出值恢复再扫描完整 B×N。输出截取和索引宽化本身不要求收尾排序。

### Merge 的 NaN 与 padding

`MergeSort::CopyIn` 在升序时用 `NAN` 补尾、降序时用 `-INFINITY` 补尾。升序经 `FlipSignBit` 转换后执行 Sort32/MrgSort，再恢复符号；Extract 结果按 H 行距读取前 N 项。补齐只发生在 UB，原输入 GM 保持不变。

L<N 时从排序结果取 `[first_rank,first_rank+L)`，恢复该区间的原轴索引并按需回读原值，输出工作域为 B×L。

批次内独立步骤尽量合并为批量 VF，保留依赖所需的事件；不得未经依赖分析删除 barrier。改变处理方式时同步重算资源与 buffer 存活。

并行度检查同时覆盖 VF 内部和外层调用者。最终写回为 B 行、每行 L 项时，按 B×L 展开线程工作域，核对是否由标量 row 循环反复启动仅处理 L 项的 VF。`LAUNCH_BOUND`、声明线程数或函数内部的分工循环不能证明整批并行；B>1、L=1 尤其需要检查。搬出还须核对每批的 DMA 提交和事件等待次数，避免每行一次 VF、两次小块 DMA、逐行等待的通用适配。尾批或容量分块按实际有效输出范围处理，具体规则见 [批次写回粒度](output-mapping.md#保持批次写回粒度)。

## Mask、索引与调用约束

Reg 的 `UpdateMask` 会更新传入的计数变量。同一逻辑长度还用于寻址、后续搬运或生成其他类型的 mask 时，保留原长度，为各次 `UpdateMask` 使用独立的局部计数；值与索引位宽不同时分别生成对应 mask。参考 [signed_zero_sort_utils.h](example/sort/op_kernel/arch35/common/signed_zero_sort_utils.h) 的 `RestoreSignedZeroValuesByIndexVecVF`，不要在移植时合并其计数变量。

输出恢复阶段的索引读取、Gather 与写回按各自实际有效元素生成 mask，不能把覆盖整个向量的 active mask 当作逻辑有效 mask。将无效 lane 的索引替换成安全的 index0 后再读取，仍发生了该 lane 的读取；若要求无效 lane 不读取，须限制实际消费者的 mask。排序原语所需、已分配并正确初始化的 padding 可以参与计算，不能据此扩展为“所有 padding 均禁止读取”；padding 不得成为有效输出或引起越界。

`AscendC::Sort` 的模板参数、隐式/显式索引 overload 和 scratch 查询从安装头文件核对，不凭名称猜测稳定性参数。浮点零符号恢复与 NaN canonicalize 的实际模板实参分别核对。接口约束见 [capability-mapping](capability-mapping.md)。

本例向量片段使用具名 `__simd_vf__` 和 `asc_vf_call`，内部向量辅助函数使用 `__simd_callee__`；原有 SIMT 插入/scatter 使用 `__simt_vf__`，改变函数组织不等于替换排序算法。SIMD VF 及其 callee 的运行时判断留在调用壳层，向量内使用 mask/Select 或已证明等价的无分支计算，类型特化使用 `if constexpr`。`LAUNCH_BOUND` 的线程数使用编译期常量；按目标 SDK 核对每个实际 VF 实参的字节数及 ABI 对齐。

example 通过 [PrepareScatterArgs / ScatterOutput](example/sort/op_kernel/arch35/sort_radix_more_core.h) 缩减 scatter 的 VF 实参；允许的参数区上限仍以目标 SDK 为准。三个 64-bit 偏移存入已结束的 Sort scratch：24 字节有效数据，12 个指针加一个 uint32 实参共 100 字节、自然对齐后 104 字节。Host 确认 scratch 容量，Vector→Scalar→Vector 事件保护参数写入，scatter 完成后才可复用。复用其他临时区也须证明相同的存活和同步条件，不能直接占用仍活跃的 buffer。
