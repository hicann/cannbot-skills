# DAV_3510 → DAV_9201 / DAV_9202：tiling、存储预算与整数位宽

**使用时机**：Stage 1/2 所有算子必读；Stage 4/5 出现错误结果、越界、AICore 超时或 tiling 变化时复查。

通过目标平台接口获取资源参数，按实际模板的 buffer 生命周期计算预算；不在迁移代码或本路线资料中写入固定容量、核数等硬件规格。

证据目录 `${EVIDENCE_DIR}` 由 Stage 0 为本次迁移单独建立，所有本页产物均写入该目录。

## 平台参数获取与传递

在 host tiling 中，通过 `context->GetPlatformInfo()` 构造 `platform_ascendc::PlatformAscendC`，按实际使用的资源查询：

| 参数 | 获取方式 |
|---|---|
| UB、L1、L0A/L0B/L0C、L2 容量 | `GetCoreMemSize(memType, size)`，`memType` 按资源选择 `CoreMemType` 的 `UB`、`L1`、`L0_A`、`L0_B`、`L0_C` 或 `L2` 枚举 |
| Cube 可用核数 | `GetCoreNumAic()` |
| Vector 可用核数 | `GetCoreNumAiv()` |
| 向量寄存器长度 | `GetVecRegLen()`，按目标 SDK 声明核对单位与适用范围 |
| API workspace | 按所调用 API 使用 `GetLibApiWorkSpaceSize()` 等对应查询接口 |

接口声明以实际编译用 devkit 的 `include/utils/tiling/platform/platform_ascendc.h` 为准。已有工程封装可继续使用，但须追溯到平台查询来源。将查询值、单位和必要的模板预留写入 `env_info.md`，据此计算工作预算；平台返回值已扣除的预留不得重复扣除。必要参数缺失或返回无效值时先核实，不以固定常量补齐。

检视 tiling/kernel 中的 `ubSize`、`buffer_size`、`block_num`、`core_num`、`UB_SIZE` 等字段，区分接口返回、从 context/tiling 传入和直接写死的数值。沿 host→tiling→kernel 核对传递来源；`GetBlockNum()` 表示本次启动的 block 数，不能替代可用核数查询。对硬编码项记录“文件:行号、变量、用途、风险、替代接口或传入字段”，并在改造中替换。

## 1. 同时满足资源和编码范围

tile 必须同时满足：实际可用 UB/L1/L0 容量、数据布局、对齐、索引位宽、计数字段、指令/API 参数范围。UB 容量增加后，旧代码的窄索引和计数边界更容易被触发。

即使资源预算允许，元素数量也可能超过窄计数字段的范围。以纯整数边界为例：

```cpp
#include <cstdint>
#include <limits>

uint64_t count = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
uint16_t narrowed = static_cast<uint16_t>(count); // 65536 截断为 0
```

**最大索引与数量分别证明**：65536 个元素的最大索引是 65535；索引本身可用 uint16 表示，不代表 uint16 计数字段可表示 65536。

## 2. 位宽链路表（Stage 1 必产出）

在 `${EVIDENCE_DIR}/migration_plan.md` 为命中的字段填写：

| 字段/表达式 | 单位与最大值推导 | host 类型 | tiling 字段 | kernel/API 类型 | 限制与处理 |
|---|---|---|---|---|---|
| 索引/行列数 | index、count 或最大下标分别注明 | 实际类型 | 实际字段 | 实际类型 | 扩宽、分块或准入 |
| offset/stride | 元素、字节、32B block 等 | 实际类型 | 实际字段 | 实际参数 | 对齐、乘法、编码范围 |
| repeat/循环数 | 循环次数或每次处理数量 | 实际类型 | 实际字段 | 实际参数 | 尾块及循环终止 |
| 资源剩余量 | byte，包含所有预留 | 实际类型 | 实际字段 | 分配参数 | 减法前检查 |

搜索 `uint16_t`、`int16_t`、`static_cast`、窄 tiling 字段和 API 参数只是入口；需要追踪真实范围，不能把数据类型为 uint16 的输入张量全部当索引问题。

### API 使用 uint16 索引时的分块判定

对使用 `uint16_t` 索引执行 UB 访存的接口，仅限制该接口相对基址实际寻址的数据块；其他 buffer 按各自约束独立预算。

1. **定位**：核对目标 SDK 的实际重载、dtype、索引类型、单位和基址，明确它访问哪一段 UB；区分数据块、索引存储区及寄存器内 lane 编号。
2. **限界**：在窄化前证明 `0 <= index <= min(UINT16_MAX, API允许最大索引)`，反推该数据块的 payload/分块上限，并按实际访问宽度、对齐和缓冲份数核算分配。不得据此统一缩小 `ubSize`，也不把上限套给未被该接口寻址的索引、输出或临时区。
3. **落实**：通过现有 tiling 字段限制对应 UB 数据块及块长，由原 kernel 的既有分块循环消费，保持原搬运顺序和计算流程。若现有字段/循环无法承载，单列实现缺口；不得自行新增 kernel 窗口逻辑或切换模板，并计作本项限界验证通过。

## 3. 必查的整数错误

| 风险 | 判定与处理 |
|---|---|
| 窄类型截断 | 转换前比较可表示范围；host 宽类型不能补救后续窄字段 |
| 窄循环回绕 | `uint16_t i` 的循环上界和最终递增均需合法，防回绕后不能退出 |
| 乘法先溢出后扩宽 | 先扩宽操作数再乘，并验证目标类型范围，赋值为 uint64 本身不足以保证正确 |
| uint64 减法下溢 | `available - reserved` 前验证 `reserved <= available`，并检查剩余空间是否容纳最小合法 tile |
| signed→unsigned | 先按接口规范归一化/校验负索引，再转换；负值转无符号可能生成巨大地址 |
| 移位与对齐计算 | 检查位数、加法进位、对齐后范围及乘积，禁止假设最终转换会修复中间结果 |

出现异常 uint64 大值时，沿上述链路检查中间表达式的范围、截断、回绕和下溢，并核对单位转换；资源预算与整数范围分别证明。

## 4. API 固定位宽的处置

1. 读取当前 SDK 对应字段的精确限制和单位。
2. 软件局部索引可按需要扩宽；API/ABI 固定字段不能通过改局部变量类型扩展。
3. 对单次 count、burst、offsetNum、stride 超限，降低 tile 或分批调用，并处理尾部。
4. 准入不满足且无法合法分块时，回退到已验证模板或阻塞该路径。
5. 每个字段独立约束；不同路径的 4095、65535 或其他上限不能合并成全局常数。

案例核查入口：GatherElements 的 `xLoadInUbNum_`、UnsortedSegmentSum 的 `ScatterDataCopy` offsetNum、IndexPutV2 DMA 的单次行数。读取实际版本的字段类型、单位和限界公式，再与目标 SDK 约束对应；不把案例中的数值直接作为本算子的上限。

## 5. 预算与 workspace

- 按生命周期列出 payload、索引、offset、mask、scale、bias、累加、输出、对齐、临时区与双缓冲；共享区必须有不同时存活的证据。
- 新 DMA 的 offset/mask 需与数据 tile 同步计入；不能只沿用旧 payload 公式。
- Cube/MIX 按实际每核任务分片核算；四 bank、双子块、单 AIV 等属于具体模板契约。
- 负载均衡搜索既检查初始 tile，也检查所有候选。Mix 公式不得套给其他布局模板。
- workspace 的形状乘积、每核偏移和 runtime/RPC 预留全部保留；核数变化后重新计算。

相关 Cube 模板依据见 [cube-migration-guide.md](cube-migration-guide.md)，量化存储单位见 [quantization-guide.md](quantization-guide.md)。

## 6. Stage 4 边界电池

| 边界 | 测试设计 |
|---|---|
| uint16 索引/计数 | 对真实字段设计 65535、65536、65537 及前后对齐值，区分 index 与 count |
| API uint16 寻址索引 | 按实际单位覆盖索引值上限前/等于/后；包含少量索引取大值、stride 乘加超限、基址重定位和尾部 mask |
| DMA 参数 | 针对实际模板的 count、burst、stride 上限前/等于/后分别覆盖 |
| UB 扩容 | 覆盖源平台不进入、目标平台进入的大 tile 路径，记录实际 tiling |
| 预算极限 | 最小合法 tile、刚好可容纳、加一项预留后超限、向下对齐为零 |
| 尾部与多核 | 满 tile、尾 tile、跨核分界、无任务核、非整组数据 |
| 表达式边界 | 大维度乘积、负索引处理、剩余空间不足、API 字段收窄 |

用例尺寸由触发条件决定；需要大 shape 才进入的路径不能用小用例替代。执行前估算内存预算，失败保留原始错误与 tiling 数据。

## 7. 核查入口

- 实际编译用 devkit 的 `include/utils/tiling/platform/platform_ascendc.h`：`PlatformAscendC` 资源查询接口与 `CoreMemType` 声明。
- 实际编译用 devkit 的 `include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h` 及目标分发实现：检索 `DataCopyGatherImpl` / `DataCopyScatterImpl`，核对所用 dtype 组合的索引类型与单位；确认为 uint16 索引时应用本页的分块判定。

其余案例按 [证据来源规则](platform-and-api-diff.md#6-核查入口与证据记录) 在实际工程或兼容的参考源码中检索，不要求固定路径、行号或提交：

- GatherElements：`gather_elements_tiling_arch35.cpp` 中的 `xLoadInUbNum_` 限界。
- UnsortedSegmentSum：`unsorted_segment_sum_simd_non_sort_tiling.cpp` 中的 offsetNum 与 offset 预算。
- IndexPutV2：`index_put_v2_simd_dma_tiling.cpp` 中的单次范围、offset/mask 与 buffer 预算。
- Mix 基本块：`base_block_load_balance_calculator.cpp` 中的 `IsBasicBlockUbOverflow` 与候选过滤。
