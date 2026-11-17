# DAV_3510 → DAV_9201 / DAV_9202：Cube 数据通路、MIX 与同步

**使用时机**：源码包含 Mmad/LoadData/Fixpipe、高阶 Matmul、AIC/AIV 协作或跨核同步时，Stage 1/2 必读；Stage 4/5 按命中路径复查。

本页覆盖本架构对的实施检查。3510 已有的数据通路不列作 9201 新增；各模板的核比例与 UB 布局只能按自身契约使用。

证据目录 `${EVIDENCE_DIR}` 由 Stage 0 为本次迁移单独建立，所有本页产物均写入该目录。

## 1. 先盘点完整计算链

将下列链路画入 `${EVIDENCE_DIR}/migration_plan.md`，记录张量布局、类型、偏移单位和同步点：

```text
GM → L1 → L0A/L0B → MMAD → L0C → GM / UB / L1
                             ↓
                     AIV epilogue / 归约
```

| 环节 | 必查内容 |
|---|---|
| 搬入 | ND/NZ、转置、C0、block/tile 边界、源和目标 stride |
| Cube | A/B/累加类型组合、bias、清零/累加、分 K 次序 |
| 搬出 | Fixpipe 输出方向、目标布局、量化、子块分发 |
| AIV | 每个 AIV 消费哪块结果、UB bank/stride、尾块和归约 |
| 同步 | 谁生产/消费、参与核集合、flag/buffer ID、生命周期 |

kernel 入口、tiling key、workspace 分配和 host 的核数计算须与上表一致。

## 2. MIX 与子块映射

先通过目标平台接口取得 AIC/AIV 可用核数，再核对 SDK 支持的任务类型与工程采用的协作方式。可在 FlatQuant 中检索 `KERNEL_TYPE_MIX_AIC_1_1` 及配套 host 核数检查，在 QuantBatchMatmul 中检索 `GetAivNumPerAic()`，追踪本次版本实际选择的子块映射。

案例中的映射属于相应算子/模板的实现契约，不构成所有 SKU 和所有 kernel 必须使用同一 MIX 比例的依据。

逐点审计：

1. `GetBlockIdx()`、`GetSubBlockIdx()`、blockDim 与任务类型的关系。
2. 原有 `/2`、`<<1`、固定双子块是否仍符合目标模板。
3. workspace 分片、每核 GM 基址、L0C/UB 输出区域是否重叠或遗漏。
4. 同步 Set/Wait 的参与集合是否对应，奇数核和无任务核是否退出。
5. 确定性归约的累加顺序和每核负责数据是否保持契约。

只改变任务类型宏而未修改依赖它的索引、内存和同步，不算完成改造。

## 3. Fixpipe：沿公开 API 核查

在目标版本官方文档和 SDK 中确认所需 L0C→L1/UB/GM 方向是否支持，沿 `Fixpipe` 入口查参数类型、布局、量化及子块约束。若实际签名仍使用 `FixpipeParamsArch3510<config.format>`，按该签名使用，并核对 `dualDstCtl` 等字段的约束。

分别沿所选目标的实际 SDK 分发核对公开 API、对应实现与参数类型；不能直接按 9201 的内部目录或类型名猜测 9202 的实现，也不凭架构号构造新参数类型。

- 不依据类型名字推断只适用于 3510，也不新增猜测的 `FixpipeParamsArch9201`。
- 直接调用 `copy_matrix_cc_to_*` 等 intrinsic 时，逐参数对比目标签名。
- 通过实际 API 文档/头文件确认 stride、ND 数、M/N 切分和 `subBlockId` 的单位与合法值。
- 变更输出路径后，重新核对 AIV 的消费布局、可见性和 buffer 生命周期。

即使目标版本保留双目的相关接口，也不意味着可以沿用原双子块路由配置；以实际 kernel 的生产消费关系验收。

## 4. UB 布局与 epilogue

以实际目标 API 的输出布局为依据。若工程采用 QuantBatchMatmul 的 `BlockEpiloguePertile4UbBank` 等模板，读取 `UbBank4Layout` 或等效布局定义，确认 bank 组、行、ND stride 与子块切分的关系。

迁移复用此模式时检查：

| 项目 | 应证明的条件 |
|---|---|
| 输入/累加区 | Fixpipe 写入与 Vector 读取的布局一致 |
| ping/pong | 地址区间独立，同步与实际最后使用点相符 |
| scale/bias/输出 | 类型和对齐后的物理占用完整计入预算 |
| 索引单位 | bank 字节偏移与元素偏移转换不会截断 |
| 峰值 | 按同时存活 buffer 计算，并满足目标实际可用资源 |

四 bank 代码是该实现的布局依据；其他实现按其目标 API 与平台信息核算。详细范围证明见 [tiling-and-memory-guide.md](tiling-and-memory-guide.md)。

## 5. StreamK、Blaze 和负载均衡

仅对相关 Matmul 模式评估：

- 检查实际模板的量化模式、batch、核数关系与 TensorAPI 运行时能力；可检索 `QBMMV3StreamKTiling960::IsCapable()` 作为入口。
- 沿实际 9201/9202 StreamK 分发确认是否进入 Blaze 及其依赖；不能仅改 tiling key 而不检查实际 kernel/API level。
- 负载均衡候选必须满足当前模板的 L0/L1/UB 布局，Mix 的 UB 公式不得套给纯 Cube、MX 或 StreamK 模板。
- 核对 MatMulV3 等 StreamK 路径的 workspace 是否包含额外 RPC 预留；复用时按实际版本保留所需预算。

性能候选与验收见 [feature-catalog.md](feature-catalog.md)。不能将源平台启发式或固定 tile 直接当目标性能最优值。

## 6. 验证与核查入口

验证电池覆盖 M/N/K 尾块、转置组合、layout、跨核分界、奇数参与核、空工作核、分 K 和声明支持的 dtype。只有执行了相关分支，精度/确定性结论才覆盖该路径。

按 [证据来源规则](platform-and-api-diff.md#6-核查入口与证据记录) 查目标 SDK 和实际工程，以下案例仅作检索线索：

- SDK：`Mmad`、`LoadData`、`Fixpipe`、任务类型及跨核同步的官方接口说明；`kernel_operator_fixpipe_impl.h` 中所选目标的分发与 `FixpipeL0cToUB` 等实现。
- FlatQuant：`flat_quant.cpp` / `flat_quant_tiling.cpp` 中的任务类型与 host 核数检查。
- QuantBatchMatmul：`quant_batch_matmul_v3_tiling_strategy.h` 中的 `GetAivNumPerAic`；`block_epilogue_pertile_4_ub_bank.h` 中的 `UbBank4Layout` 与 buffer 初始化。
- StreamK：`qbmm_streamk_tiling_960.cpp` 中的准入与 API level；`matmul_v3_basic_streamk_tiling.cpp` 中的 workspace 分配。
- Mix 预算：`base_block_load_balance_calculator.cpp` 中的 `IsBasicBlockUbOverflow` 及候选过滤；按实际公式核对，不依赖历史修复提交。
