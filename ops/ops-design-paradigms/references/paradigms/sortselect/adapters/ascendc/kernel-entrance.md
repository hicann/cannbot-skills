# SortSelect AscendC Kernel 入口

[Sort 示例](example/sort/README.md) 的 `op_kernel/sort.cpp` / `arch35/sort_tiling_key.h` 将 `(schId,isInt32,isDescend)` 映射到具体 kernel 模板。多核 radix 按轴长选择 isInt32：N≤2^30−1 时为 1，生成 key 258/65794，计数/前缀为 uint32；更长轴为 0，生成 key 2/65538，计数/前缀为 int64。输出索引 dtype 独立确定。其他 schedule 保持各自的模板键条件。复用模板时按 [Tiling 参数对应](tiling-parameters.md) 逐字段核对 `keyParams0..5` 的含义与单位。

三个字段各占 8 位。SDK 对 UINT 声明列表按序号编码：`schId={0,1,2,5,6}` 对应 `schIndex={0,1,2,3,4}`，故 `key=schIndex | (isInt32<<8) | (isDescend<<16)`。多核 64 位降序的 65538 即 `2+(1<<16)`，32 位降序的 65794 再加 `(1<<8)`；插入路线 isInt32=1、升序时为 259。声明列表的顺序属于键编码约定，不可随意重排。

入口通过 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 读取 TilingData，按路线声明 AIV-only 或 MIX_AIV 任务类型。Kernel 入口、Host 注册和 API 接入分别组织；构建依赖见示例说明。

## API 与非末轴转置包装

example 同时提供 [L2 接口](example/sort/op_api/aclnn_sort.cpp) 和 [L0 调用封装](example/sort/op_api/sort.cpp)。`aclnnSortGetWorkspaceSize → BuildSortGraph` 在创建执行图时决定布局；`aclnnSort` 执行已构建的图。不能直接调用 `sort.cpp` 后，把缺少 L2 包装的非末轴用例视为完整 Sort 不支持。

- 本例的 AiCore L0 只接受末轴；连续末轴直接排序，输入和输出均不得启动布局转置 kernel。
- 非末轴由 `GetPermResult` 交换排序轴与尾轴；`SortProcess` 先调用 `l0op::Transpose`，再以 `dim=-1` 调用保留的排序模板，最后分别转置 values 和 indices。交换置换是自身的逆置换；改用一般置换时必须显式求逆。
- rank≤8 且排序轴后只有 singleton 维度时，`HasOnlyTrailingSingletons` 通过 reshape 将排序轴作为尾轴，避免物理转置。本例不保留其他架构的兼容分支。rank>8 的输入先折叠为三维再处理；折叠不改变物理转置是否必要的判断。

当前示例已跳过尾轴及轴后全为 singleton 的情况，但 N=1、I>1 仍进入原轴与末轴的置换路径。例如 `[2,1,3]` 沿中轴排序，置换不改变元素顺序却仍会提交转置。更高维的轴交换还可能重排其他非 singleton 维，不能只根据 N=1 就删除转置；须先统一折叠布局或核对实际置换。该路径尚未完全达到零冗余转置目标。

生成 Sort 类、TopK、KthValue 均沿用条件布局包装及 [统一输出映射](../../generic/output-mapping.md)：主体直接写长度 L 的尾轴输出，再按需恢复各字段。生成代码必须保留转置条件，不能以 `rank>0` 代替。按三维折叠处理时，输入和输出分别按 N、L 判断，L=1 的输出只需 reshape；若沿用 example 的高维置换，则按实际置换核对元素顺序，见 [布局判定与验收](../../generic/shape-preprocessing.md#物理转置的判定与验收)。最终输出与实际需要的输入/输出布局区分别预算。

转置后的轴内索引仍指向原排序轴位置，无需将索引数值转换为扁平地址。公开接口还承担连续化、类型处理、空张量及输出 view copy；API 的完整支持域与底层单个 schedule 的直接支持域应分别记录。性能比较累计该调用实际启动的全部 kernel，包括转入、排序、values/indices 还原和必要的输出拷贝，排除 Host 建图、编译与日志耗时。

构建需提供 `aclnn_kernels`、opdev、公共算子和 CANN SDK 等依赖，详见 [接入与验证](example/sort/README.md#接入与验证)。
