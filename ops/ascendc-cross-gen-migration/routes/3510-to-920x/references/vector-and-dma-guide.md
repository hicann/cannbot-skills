# DAV_3510 → DAV_9201 / DAV_9202：Vector 算术与索引 DMA

**使用时机**：Stage 1 命中掩码算术、Gather/Scatter、SIMT 索引路径时加载；Stage 2/6 按候选项实施。

本页以目标 SDK 的 API 契约、声明及实际分发为依据；算子案例仅用于定位实现模式。平台、dtype、shape、API 约束全部匹配后，才可把候选标为 `APPLICABLE`。

证据目录 `${EVIDENCE_DIR}` 由 Stage 0 为本次迁移单独建立，所有本页产物均写入该目录。

## 1. 算术 API：先确认语义，再选择实现

### 1.1 MERGING 是既有 API 语义

先在源/目标 SDK 中确认 `Add` 的 ZEROING/MERGING 语义，再沿 `AddImpl` 核对所选 dtype 的实现：是运算后通过 `CopyMergingImpl` 等逻辑合并，还是直接下传模式。若需判断指令数量，核查实际编译结果；不能从一个类型分支推广到 64 位、复数等全部类型。

用于 DAV_9201/DAV_9202 时分别沿目标分发核对，不能把一侧指令实现直接当作另一侧的实测结论。

对原代码的 Add + Select/Move 组合，逐项检查：

1. 掩码为假时保留的是原目标寄存器、源寄存器还是另一输入。
2. 目标与输入是否别名，原目标是否已经初始化。
3. 全掩码、空掩码、尾部掩码的结果是否一致。
4. SDK 的相同 API 已完成替换时，记录 `REUSED`，不要声称又产生一项代码优化。

### 1.2 FusedMulSubDst 按目标版本核查

在目标 SDK 检索 `FusedMulSubDst` 的公开声明与 `FusedMulSubDstImpl`，记录架构守卫、dtype、掩码和舍入约束，并检查实际后端。若只允许 ZEROING，就不能用于依赖保留掩码外 lane 的替换；不能把其他版本的类型列表或指令实现写成当前支持结论。

DAV_9201/DAV_9202 都可进入该 API 的目标支持核查，分别确认实际后端、dtype 和模式；API 声明覆盖两者不代表一次 9201 验证已覆盖 9202。

可在 SoftmaxGrad 实现中检索 Neg + MulAddDst 与 `FusedMulSubDst` 的使用方式。迁移前须对照实际 API 公式、目标寄存器输入语义、舍入及掩码行为；禁止仅凭名称做字符串替换。存在减法抵消、极大/极小数和尾部 lane 的用例应保留。

融合算术与 MERGING 两个候选应分别判断和验证，一项支持不代表另一项也支持。

## 2. Gather DMA

先按目标 SDK 的 `GatherDataCopy` 文档、声明和实现确认偏移类型/单位、同步与 padding 约束。可在 `EmbeddingBag` 中检索 Gather DMA 的索引转换及同步写法；仅在实际接口和数据通路一致时复用，不能将案例的形状条件推广到所有 Gather 算子。

| 必查项 | 实施要求 |
|---|---|
| 源地址与 offset | 区分元素索引和字节偏移，乘法先扩宽，检查负索引归一化与范围 |
| 单次搬运 | 查询 `offsetNum`、burst、stride 字段的范围和单位 |
| padding | 保持原算子越界/填充语义，不能用零填充掩盖非法输入 |
| 同步 | 证明索引 buffer 的 Vector 写入对 MTE2 可见；复用前证明消费结束 |
| 资源 | offset、mask、padding、双缓冲计入 UB；详见存储与位宽指南 |

高重复、随机、顺序索引及不同尾维宽度分别测量；不能从单个 shape 的结果宣称整体提速。

检查实际 host 模板的架构准入；若仅开放 `DAV_9201`，迁移 DAV_9202 时须另查 SDK 支持和合法兼容模板，未取得目标支持证据前不直接放宽条件。

## 3. Scatter DMA

先核对目标 SDK 的 `ScatterDataCopy` 参数、同步与原子操作约束，再检查实际模板的 `IsCapable()`、单次行数和 offset/mask 的 UB 预算。可在 `IndexPutV2` 的 SIMD DMA 模板中检索这些环节以及覆盖/累加分支的处理。

迁移时必须确认：

- 原接口是覆盖、累加还是归约；重复索引和写冲突语义是什么。
- 确定性要求是否存在；atomic 能否满足对应契约。
- 索引排序、去重、边界处理及输出初始化是否仍需要。
- DMA 参数、缓存 buffer 生命周期和事件同步是否全部覆盖尾 tile。
- 不满足 DMA 准入时保留已验证路径，不能无条件分发到新模板。

单次行数上限应从实际 API 与模板推导；即使某实现写有 `4095`，也不能直接套给其他模板或统一改成 `UINT16_MAX`。

DAV_9202 仍属于本路线。若实际模板采用 9201 专属准入，只说明该实现的选择；9202 是否可新增同类路径须另查实际 API 和参数支持，不能据此推断全部 Scatter DMA 硬件能力。

## 新增 DMA 模板的实施入口

判定 DMA 对本次 9201/9202 目标适用且原实现尚未覆盖时，MUST 按 [feature-implementation-guide.md](feature-implementation-guide.md) 实施新增或改造路径：kernel、host tiling、模板注册/tiling key、准入与分发、构建打包及合法回退。不能把“原 kernel 已能在目标设备运行”作为不实施的理由，也不能只替换一个 API 后宣称完整模板已接入。

以 `IndexPutV2` 为参考时，检索 `IndexPutV2SimdDmaTiling` / `IndexPutV2SimdDma`，追踪 host 的 `IsCapable()`、UB 预算、`GetTilingKey()`、模板注册与 kernel 的配合。具体继承关系、注册机制、选择顺序和 key 值按目标工程核对。新增路径必须覆盖命中/不命中、边界与尾块，并与同设备兼容基线做 A/B。

## 4. 128 位载体提升

在目标 SDK 确认搬运类型和对齐约束，再检查 Scatter 的 `PromoteDtype()` 或工程中的等效逻辑：各目标从何种位宽尝试、哪些维度须按提升因子整除、kernel 实例化是否匹配。

128 位 carrier 不能作为 DAV_9201/DAV_9202 的统一必启项，也不预设某目标只能使用 64 位。实际准入与支持范围由本次 SDK 和模板核验结果决定。

这里 `int4` 表示四个 int32 组成的 128 位载体；与 `int4b_t`、HiF4 等 4bit 数据类型分别核对。

搬运提升的验证应覆盖：

1. 原始 dtype 和提升后字节数、索引单位、地址对齐。
2. 整除与非整除 shape、头尾偏移、尾部 fallback。
3. host tiling key 与 kernel 实例化是否一致。
4. 原子、比较或计算是否仍按原数据类型执行；载体重解释不能改变数值语义。

## 5. SIMT 与候选验收

RegBase/SIMT 均有两代共享基础；仅在现有线程映射、合并因子、访存或寄存器使用形成瓶颈时调整。线程数、warp、寄存器宽度和缓存预留取实际平台与工具链数据。

每个候选按 [feature-catalog.md](feature-catalog.md) 记录适用性；实施后必须保留完整精度结果与所选目标的同一环境的 profiler A/B。正确性失败先回退该变体，再记录根因；未尝试的项记 `DEFERRED`。

## 6. 核查入口

按 [证据来源规则](platform-and-api-diff.md#6-核查入口与证据记录) 先查目标版本官方文档与实际 SDK，再按需查阅兼容的算子案例。以下是检索线索，路径和支持范围以实际版本为准：

- SDK 算术：`kernel_reg_compute_vec_binary_impl.h` 中的 `AddImpl`；`kernel_reg_compute_vec_fused_intf.h` / `kernel_reg_compute_vec_fused_impl.h` 中的 `FusedMulSubDst` / `FusedMulSubDstImpl`，沿目标分发读取。
- SDK 搬运：`GatherDataCopy`、`ScatterDataCopy` 的声明、参数结构及同步约束。
- SoftmaxGrad：`softmax_grad_ar_full_load.h` 中的掩码算术调用。
- EmbeddingBag：`embedding_bag_gather_dma.h` 与配套 host 准入中的偏移、同步和 padding。
- IndexPutV2：`index_put_v2_simd_dma_tiling.cpp` / `index_put_v2_simd_dma.h` 中的准入、预算与 kernel 实现。
- Scatter：`scatter_tiling_arch35.cpp` 中的 `PromoteDtype` 与实际 kernel 实例化。
