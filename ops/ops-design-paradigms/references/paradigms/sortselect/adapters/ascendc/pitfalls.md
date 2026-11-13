# SortSelect AscendC 陷阱与定位

- 两阶段的注释类型须与真实声明核对：当前 Stage2 使用 uint16 紧凑键，容量校验在 Host。
- 稳定降序不能通过反转稳定升序实现；Stage2 恢复序列归属仍按恢复键升序。
- 多核 `isInt32=1/0` 分别使用 uint32/int64 计数和前缀；高两位是状态位，32 位有效计数只到 2^30−1；生成 key 时读取最终选路标志，不能沿用按 N 初算的标志。输出索引与局部排列的宽度另行核对。
- proposal 归并的 offsets、counts 和 validBit 均有物理含义，不能替换成普通数组偏移或整块长度。
- 原值恢复只需 L 项时，第二次 DMA 搬入整批 B×N 原输入会引入冗余搬运；排序输出已满足合同时直接截取，确需回读时按最终索引 gather B×L 项。见 [区间定位与原值恢复](templates/merge/kernel-template.md#区间定位与原值恢复)。
- 使用 Reg/Gather 不代表具备轴内并行：一个 lane 对应一行、串行扫描 H 项，B=1 时仍只有一个有效 lane。完整轴的分类、计数与过滤须沿轴分配 lane/线程；最终 L=1 的一次短 gather 可以保留。见 [辅助处理并行度](kernel-helpers.md#辅助处理并行度)。
- 两阶段与 Radix 的浮点零符号恢复、NaN 模板参数必须核对实际 caller，不能仅凭辅助函数存在推断契约。
- 多核 lookback 的计数与状态发布顺序、轮间 SyncAll 和所有参与核的路径须一起审核。
- 审查意见须区分正确性缺陷、工程规范与建议项。已配对且满足目标平台约束的存量 `SetFlag/WaitFlag` 可以保留；已正确对齐的存量 `DataCopy` 也不因“推荐 DataCopyPad”而必须重写，UB→UB helper 尤其须先确认替代接口适用。迁移不能破坏原同步和搬运语义。
- 移植源码时同步修改入口文件、OpDef 的 `opFile.value` 与工程引用；example 使用 `op_kernel/sort.cpp`。不要只改文件名而遗留动态源码或预编译模块路径。
