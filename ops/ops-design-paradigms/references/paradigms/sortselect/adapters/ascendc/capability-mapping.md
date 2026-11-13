# SortSelect AscendC 能力映射

## 公共能力与资源映射

核对项目见 [capability-contract](../../generic/capability-contract.md)。以下实现均受各自适用域约束，待验证不等于不存在。

能力核对与实现替换按 [实现复用与替换约束](design-spec.md#实现复用与替换约束) 执行。原语签名、scratch 或特殊值验证未完成时保留“未验证”；自有 VF 的声明和显式数组预算不能替代原语的适用性结论，也不能直接作为新实现已验证的证据。

| 标识 | 映射与适用范围 |
|---|---|
| `CAP_STABLE_COMPARE` | Sort 各路线的比较、稳定性与特殊值按 caller 核对；插入排序只移动严格不等的前项，两阶段依赖 Stage1 稳定排序 |
| `CAP_LOCAL_SORT` | Sort32/AscendC::Sort；dtype、长度与索引模式由 caller 限定 |
| `CAP_MERGE` | MrgSort/merge 路线；proposal 布局、counts 与 validBit 按 wrapper 解释 |
| `CAP_RADIX_SORT` | AscendC::Sort RADIX_SORT 路线；稳定性和特殊值按 caller 核对 |
| `CAP_BLOCK_SYNC` | 两阶段 `RankInverseScatter` 使用 `asc_syncthreads()`，先完成逆映射建表再供全块读取；搬运与 VF 的依赖另按实际事件核对 |
| `CAP_GLOBAL_SYNC` | SyncAll 与 GM lookback 发布协议；按算法的实际参与者与内存可见性协议核对 |
| `ALIGN_GRANULARITY` | block 对齐值由平台获取；TBuf/API 自身分配约束分别核对 |
| `LOCAL_MEMORY_BUDGET` | 平台 UB 容量减当前路线的运行保留；SIMT 保留按目标 SDK 和实际调用链核对，参考 Host 的 `SIMT_UB=32768` bytes；纯 Vector 路线不扣此项 |
| `INDEX_COUNT_DOMAIN` | 输出索引为 int32/int64；多核计数/前缀为 uint32/int64，保留两位状态后的数值上限分别为 0x3fffffff / 0x3fffffffffffffff；单核仍核对自身 uint32 范围 |

## API 与实现约束

具体调用方式见 [Sort 示例](example/sort/README.md)。

| 能力 | 实现与限制 |
|---|---|
| 短轴线程排序 | `SimtInsertionSortSegments` / `asc_vf_call`，SIMT 线程处理段；不要扩展为所有 dtype/N 最优 |
| 局部排序与归并 | `Sort32`、`MrgSort`、`GetSortOffset`；初始 run=32、fan-in=4；类型/布局受 caller 约束 |
| 基数排序原语 | `AscendC::Sort` 配 `SortConfig{RADIX_SORT,...}`，临时空间由 `GetSortMaxMinTmpSize` 查询；overload 与索引模式必须核对 |
| 两阶段恢复 | `RankInverseScatter` 或 uint16 Stage2 keys；inverse 上限 64 是当前策略，打包宽度是实际安全约束 |
| 布局适配 | 非末轴不注册专用模板；调用层转置/等价 reshape 后复用末轴排序主体，按统一名次区间直接写长度 L 的输出，再还原各字段 |
| UB 与资源 | 平台查询 UB/block/VReg 大小；仅实际 SIMT 路线计入其运行保留（参考值 32768 bytes）；算法 scratch、SIMT 保留和系统 workspace 分开预算 |
| 索引宽度 | 输出 int32/int64 与计数类型是不同维度；多核状态高两位保留，N≤2^30−1 采用 uint32，否则采用 int64（数值域最多 `0x3fffffffffffffff`） |
| 浮点特殊值 | 零符号恢复及 NaN canonicalize 的启用条件见对应 caller；Merge 的方向相关补齐值见 `CopyIn` |
| 跨核同步 | 保留的多核 radix 使用 `SyncAll`；lookback 还有 GM 数据/status 协议，不能删成单个 pipe barrier |
