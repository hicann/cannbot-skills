# SortSelect AscendC Tiling 预处理

按 [shape-preprocessing](../../generic/shape-preprocessing.md) 归一化 axis 和 O/N/I，核对实际接口 dtype、索引类型、方向及稳定性。L2 的连续化、末轴规范化和转置路径在 `op_api/aclnn_sort.cpp`；L0 参数检查不能代替完整 L2 契约。

元数据乘法以足够宽类型求值并检查范围，再转换为 TilingData 字段。全部接口按 [统一映射](../../generic/output-mapping.md) 配置输出区间与长度 L；输入仍为每行 N 项，输出索引仍是原轴坐标。输出 shape 与输入 shape 独立推导。输出索引、Radix 计数和状态编码分别核对范围，不把状态位占用误算成完整整数数值域。

调用层完成布局规范化、平台资源查询后，按 dtype、轴长 N、序列数 M 及容量评估各排序路线；UB 预算包含对齐、转换、原生排序临时空间及路线所需的 SIMT 保留区。详细分派见 [tiling](tiling.md)。

## 规范化后的设计输入

| 量 | 计算或来源 | 后续消费者 |
|---|---|---|
| O/N/I | 归一化轴前乘积 / 轴长 / 轴后乘积 | 布局转换和最终 shape 还原 |
| M=OI | 独立序列数 | 小轴并行度、核分配和尾批 |
| D/J | 输入值及最终输出索引类型的字节数 | pitch、Queue、GM 输出 |
| C/U/a | 实际核数、UB 字节数、block 字节数 | 各模板独立容量和并行规划 |
| first_rank/L | 目标接口语义推导 | 只控制最终写回及输出 shape |

例如原输入 `[2,3,5,7]` 沿 axis=2：O=6、N=5、I=7、M=42；物理规范化后主体看到 `[42,5]`。L=1 时最终尾轴输出为 `[42,1]`，再按 `[6,7,1]→[6,1,7]` 的等价 reshape 展开原维度；输入、输出转置条件分别判断。

Host 读取 dtype/shape 使用 `GetInputDesc`、`GetInputShape`；只有实际依赖控制输入的值时才读取对应 `GetInputTensor`。核数、UB 与对齐从本次 Tiling 的 platform 查询，不经 CompileInfo 缓存或用固定默认值兜底。指针、零资源和字段转换范围在使用前检查，axis 先按原 int64 值校验再窄化；workspace 分配及有状态返回值的 Tiling setter 失败须向上传递。分派入口统一检查 workspace 容器，各候选在副本中计算，选定并完成配置后再写回方案与 workspace 大小。Host 拷贝使用带真实目标容量的安全函数并检查返回值。

## 最终方案与提交边界

提交 workspace、TilingKey、blockDim 和序列化字段前，确认最终选中方案与已验证的 N/M、输出区间、路线及平台资源一致。输出区间按 `first_rank≥0、L≥0、first_rank≤N、L≤N-first_rank` 检查，避免用可能溢出的 `first_rank+L` 求和；合法空输出由调用层返回，不启动非空排序主体。example 的 Sort 配置为 `first_rank=0、L=N`，派生配置须核对自身输出长度与寻址，不能沿用完整排序的输出大小。

只检查所选路线实际消费的计数、批次、偏移和资源关系：例如小轴批次数与 M/B、归并有效 run 长度、多核 tile 数与活跃核数。未消费字段允许为零，原生 API 的合法零 scratch 也不能一律判为失败。既有参数检查和候选规划已保证的不变量可直接复用；正常路径确有缺口时补在最终提交之前，无需再建一套重复规划或逐字段非零检查器。参考 example 的 `SelectSortSchedule`、各 `Fill*Info` 和 `PlanToTilingData` 的实际调用链；后续 setter/序列化失败须返回错误，不能提交失败后再换路线冒充正常候选回退。
