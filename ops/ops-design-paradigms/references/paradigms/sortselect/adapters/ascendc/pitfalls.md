# SortSelect AscendC 陷阱与定位

- 两阶段的注释类型须与真实声明核对：当前 Stage2 使用 uint16 紧凑键，容量校验在 Host。
- 稳定降序不能通过反转稳定升序实现；Stage2 恢复序列归属仍按恢复键升序。
- 多核 `isInt32=1/0` 分别使用 uint32/int64 计数和前缀；高两位是状态位，32 位有效计数只到 2^30−1；生成 key 时读取最终选路标志，不能沿用按 N 初算的标志。输出索引与局部排列的宽度另行核对。
- proposal 归并的 offsets、counts 和 validBit 均有物理含义，不能替换成普通数组偏移或整块长度。
- 两阶段与 Radix 的浮点零符号恢复、NaN 模板参数必须核对实际 caller，不能仅凭辅助函数存在推断契约。
- 多核 lookback 的计数与状态发布顺序、轮间 SyncAll 和所有参与核的路径须一起审核。
- 审查意见须区分正确性缺陷、工程规范与建议项。已配对且满足目标平台约束的存量 `SetFlag/WaitFlag` 可以保留；已正确对齐的存量 `DataCopy` 也不因“推荐 DataCopyPad”而必须重写，UB→UB helper 尤其须先确认替代接口适用。迁移不能破坏原同步和搬运语义。
- 移植源码时同步修改入口文件、OpDef 的 `opFile.value` 与工程引用；example 使用 `op_kernel/sort.cpp`。不要只改文件名而遗留动态源码或预编译模块路径。
