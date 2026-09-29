# Reduction 范式 AscendC 能力映射

> **层级**：AscendC 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 AscendC 真值与 API 实体。

## 1 谓词真值表

| 谓词 | AscendC 真值 | API 实体 / 规则文档 |
|------|------|------|
| `REDUCE_OP` | `ReduceSum` / `ReduceMax` / `ReduceMin` / `ReduceProd` / `ReduceAny` / `ReduceAll` 高阶 API；Pattern 按 tail 类型选择（tail-R→`Pattern::Reduce::AR`，tail-A→`RA`）；srcShape 按 padded 值；`srcInnerPad=true` 恒成立；src/dst 与 sharedTmpBuffer 互不重叠 | [highlevel-api.md](highlevel-api.md) §1–§5 |
| `REDUCE_OP_PRECISION` | 多路并行二叉树折叠（单 chunk 内；与范式级二分缓存树构成两级树） | [highlevel-api.md](highlevel-api.md) §6 |
| `TREE_CACHE_CAPACITY` | **16 KB**（`cacheBuf`，二分缓存树专用，恒定值不得修改） | [overview.md](overview.md) §4.1.1、[templates/binary-base/tiling.md](templates/binary-base/tiling.md) §5.1 |
| `COMPUTE_PRECISION` | **FP32（4 字节）**；b16 输入先扩位再归约；`maxDtypeSize = max(sizeof(D_T), sizeof(float))` | [common/cast-rules.md](../../../common/cast-rules.md) §4 |
| `ALIGN_GRANULARITY` | **32 字节**（UB block / `ONE_BLK_SIZE`）；`bsElem = blockSize / sizeof(D_T)` | 平台接口 `GetUbBlockSize`（禁止写死 32） |
| `CACHELINE_SIZE` | **256 B**（Ascend950） | 平台接口 `GetCacheLineSize`（禁止写死 256/512） |
| `SINGLE_BUF_MAX` | **64 KB**（Empty 模板单 buf 封顶 `MAX_SINGLE_UB_BYTES`） | [templates/empty/dag-buffers.md](templates/empty/dag-buffers.md) §1.3 |
| `FUSION_CHAIN_MAX_LEN` | **7**（超过 7 条 Vector 操作必须拆为多次 `asc_vf_call`） | [vf-fusion-rules.md](vf-fusion-rules.md) §1.1 |
| `CONVERT_IN_CHAIN` | **true**（`Reg::Cast` 寄存器指令可串入 VF 链，不作为断点） | [common/cast-rules.md](../../../common/cast-rules.md) §4.3 |
| `REG_CHAIN_FUSION` | **true**（`asc_vf_call` + RegBase 寄存器链，中间值不落 UB，P < L 合法） | [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) |
| `BUFFER_MODEL` | **TBuf 显式单缓冲**（禁 TQue）；Double Buffer 当前**不开启** | [templates/binary-base/dag-buffers.md](templates/binary-base/dag-buffers.md) §1.4 |
| `SYNC_PRIMITIVE` | `SetFlag<HardEvent::XXX>` / `WaitFlag<HardEvent::XXX>` 成对使用；eventID 经 `GetTPipePtr()->FetchEventID()` 获取，不可硬编码；同类型事件可共用编号 | [common/sync-and-consistency.md](../../../common/sync-and-consistency.md) §2.2 |
| `EXEC_UNITS` | MTE2（搬入）/ V（计算）/ MTE3（搬出）；事件命名 `源_目标`（`MTE2_V` / `V_MTE3` / `V_MTE2` / `MTE3_V` / `MTE3_MTE2`） | [common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |
| `CROSS_CORE_FENCE` | `SyncAll()`（Group Phase 1→2 全核同步）；host 侧必须 `SetScheduleMode(1)`，否则挂死 | [templates/binary-group/tiling.md](templates/binary-group/tiling.md) §5.5 |
| `WORKSPACE_MODEL` | `ws[0] = GetLibApiWorkSpaceSize() + 用户部分`（Group：`usrWorkspaceBytes = rGroupCnt × aTotal × sizeof(fp32)`，`[rGroupCnt, aTotal]` fp32 dense 行优先） | [templates/binary-base/tiling.md](templates/binary-base/tiling.md) §5.2 |
| `MULTICORE_BALANCE` | 大小核式：前 `aBigCoreCnt` 核多担 1 份，最大负载差 ≤ 1；`usedCoreNum = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt` | [templates/binary-base/tiling.md](templates/binary-base/tiling.md) §2.2 |
| `SCHEDULE_2D` | **true**（Group：A×R 2D 分核，`ComputeGroupSplit()`，`aPerCore=1` 恒成立） | [templates/binary-group/tiling.md](templates/binary-group/tiling.md) §2 |
| `BLOCK_DISPATCH_MIN` | **1**（`SetBlockDim` 严禁 0；EMPTY_A 用核数为 0 时仍 `SetBlockDim(1)`，kernel 侧 `usedCoreNum=0` 短路） | [templates/empty/tiling.md](templates/empty/tiling.md) §1.2 |

## 2 通用术语 → AscendC 实体对照

| 通用层术语 | AscendC 实体 |
|------|------|
| 片上 buffer / 片上容量 | UB / `GetCoreMemSize(CoreMemType::UB)` |
| 树缓存 | `cacheBuf`（16KB，TBuf 分配） |
| 归约指令 | `ReduceXxx`（带 `sharedTmpBuffer` 重载，`preReduceResultTail` 兼作 sharedTmpBuffer，buffer 固定 3 份） |
| 搬入 / 搬出 | `DataCopyPad`（+ `LoopModeParams`，K 平台分级：K=2 全平台 / K∈[3,4] 仅 950 / K≥5 host for 拆分） |
| 寄存器链融合 | `asc_vf_call` + `LoadAlign` / `StoreAlign` |
| 精度转换 | `CastTrait` + `Reg::Cast`（b16↔fp32 配对 DIST 模式） |
| 单缓冲管理 | `TBuf<TPosition::VECCALC>` + `pipe_->AllocTBuf` |
| 编译期模板选择 | `ASCENDC_TPL_*` 宏（禁 `TILING_KEY_IS`） |
| dtype 注入 | `DTYPE_<INPUT_NAME>` 编译宏（不进 TilingKey） |
| kernel 入口分发 | `if constexpr (isEmptyTensor / isGroup)` 三路分发 |

## 3 关键语言级附加约束

1. 所有 API 可用性判定的唯一入口是 [API 白名单](../../../../knowledge/api/regbase_api_whitelist.md)；不得把"待验证"写成"已验证"
2. 平台参数（coreNum / ubSize / blockSize / cacheLineSize / vectorSize）严禁写死，必须通过接口获取并做零值守卫
3. `__global__` 模板参数仅含 TilingKey 的 bool（`isGroup` / `isEmptyTensor`），禁止加 `typename D_T`
4. 三模板物理分文件：`<op>_base.h` / `<op>_group.h` / `<op>_empty.h` + `<op>_apt.cpp` 入口（同 arch35/ 目录）
5. 归一化/合轴四步函数封装为独立函数（每函数有效行数 ≤ 50），可独立单元测试
