# DAV_3510 → DAV_9201 / DAV_9202：量化、打包与低精度计算

**使用时机**：Stage 1 命中 FP4/HiF4/FP8/MX、scale、低比特打包或混合低精度 MMAD 时加载；Stage 2/4/6 按适用项使用。

FP8/MXFP4 在 3510 已有基础。这里的增量是具体 API、类型组合、转换和寻址路径，不能将整个低精度编程模型列为 DAV_9201/DAV_9202 新增。

本流程覆盖 DAV_9201/DAV_9202，LoadData/MMAD 和 MX scale 寻址以各目标 SDK 的配套文档、声明与实际分发为依据。分别核对算子注册和 dtype 校验中的支持组合，不能从 `arch92/` 目录名推出支持相同的全部低精度类型。

证据目录 `${EVIDENCE_DIR}` 由 Stage 0 为本次迁移单独建立，所有本页产物均写入该目录。

## 1. 先固定原算子的数值与接口契约

在 `${EVIDENCE_DIR}/migration_plan.md` 写清：

| 项目 | 应记录内容 |
|---|---|
| 输入/输出 | 公开 dtype、逻辑 shape、存储 shape、format、可选参数 |
| 量化 | per-tensor/token/channel/group/block 模式、group 大小和广播规则 |
| scale | dtype、符号、指数偏置、编码、布局、计算与使用位置 |
| 计算 | 累加类型、转换点、round/saturate、eps 与反量化顺序 |
| 特殊值 | zero、signed zero、NaN/Inf、溢出/下溢的接口规定 |

迁移默认保留此契约。为已有 FP16 算子新增 HiF4 输入输出属于能力扩展，必须有相应用户需求；只有硬件/API 支持不构成扩接口的依据。

## 2. HiFloat4 与 scale 转换

若契约涉及 HiF4，先查目标 SDK 的 LoadData 类型集合是否包含 `hifloat4x2_t` 等所需类型，再查 MMAD 对相应 A/B/累加类型组合的支持；HiF4×HiF4、MXFP8×HiF4、FP16/BF16×HiF4/FP4 等组合逐项取证，不能预填为全部可用。

可在 SwiGluGroupQuant 中检索 `Cast<fp8_e6m2_t, bfloat16_t>`、`ReciprocalCast` 和打包输出的用法；只有目标 SDK 支持且量化模式一致时才作为实现参考。

选择候选前必须确认：

1. 本算子的量化模式和 scale 格式是否与参考实现一致。
2. 实际 SDK 是否声明该 API，以及目标 dtype、掩码、舍入是否受支持。
3. scale 为零、边界指数、饱和及非有限值的结果是否符合契约。
4. 目标输出需要的逻辑值和物理编码是否一致。

参考案例中存在某个类型转换，不代表它适用于所有 HiF4 或 MX 算子。

## 3. MX scale 的两类变化

### 3.1 Vector scale 转换

核对目标 SDK 是否支持所需的 BF16 运算与 `Cast<fp8_e8m0_t>` 组合，比较其与原指数位操作的舍入及特殊值行为。可在 `mx_quant_common.h` 检索相应分支，分别确认 FP8/FP4 输出和架构守卫，不把某一路径的实现选择当作全部类型的统一规则。

候选替换必须限定到命中的输出类型和算法分支，验证 OCP 相关舍入、指数边界、zero、NaN/Inf；不能把它推广成全文件的统一替换。

### 3.2 Cube scale 存储/寻址

优先使用公开 LoadData2D MX 与 MMAD API，由目标后端处理 scale 地址转换。核对源/目标实现中的 `MX_BUFFER_SCALE` 等定义及其单位，不将内部常量或 buffer 容量抄入 host 规则。

- 旧代码手写 scale offset、内部 helper 或 intrinsic 时，按实际目标版本重新证明寻址。
- host 的 scale shape 按公开 API 契约推导，不复制后端内部的地址换算表达式。
- 检查 scale 的转置、batch/group 偏移、L0A/L0B 容量与重用生命周期。

## 4. 混合低精度 Cube

若原契约涉及 FP16×INT4、BF16×INT4、INT8×INT4 等混合组合，分别核查目标 SDK 的 LoadData、Mmad、累加和 Fixpipe 完整支持链，不能只看一个 MMAD 分支就扩注册矩阵。

| 检查 | 验收要求 |
|---|---|
| 注册与 API | 平台注册、API 校验、图推导与 kernel 类型一致 |
| 转置与 layout | A/B 的 ND/NZ/转置组合均有对应的数据装载路径 |
| 累加与输出 | 保留约定累加精度、bias/scale 顺序、输出转换 |
| 资源 | 低比特 payload、scale 和对齐后容量分别计入 |
| fallback | 目标组合或 shape 不满足准入时走已验证实现 |

Cube 协作、Fixpipe 与同步另见 [cube-migration-guide.md](cube-migration-guide.md)。

## 5. Packed dtype 的验证

按实际公开接口的支持列表、注册与分发核对 packed dtype。可在 IndexFill 中检索 `CheckDtypeValid`、`DTYPE_960PR_SUPPORT_LIST` 等目标列表和 IndexFillD 回退；是否存在及其内容以实际版本为准。某算子的目标分支排除 FP4/HiF4，不代表整个平台缺少相同类型的硬件能力。

逻辑元素数、物理字节数和 API count 不能混用。按真实 ABI 处理奇数元素、尾部半字节、对齐和 padding；不默认所有接口允许奇数尾部。

必须覆盖：

- 打包/解包顺序与位序，首尾元素及跨字节边界。
- 最小 shape、非整组、尾 tile、转置与非连续输入的声明支持范围。
- scale group 边界及广播，scale 和 payload 是否同步分片。
- 各声明 dtype 的数值边界和特殊值。
- 参考实现的输入量化、中间转换和输出编码是否与算子契约一致。

这里的 4bit 类型与 Vector 搬运使用的 `int4` 128 位载体无关；载体说明见 [vector-and-dma-guide.md](vector-and-dma-guide.md)。

## 6. 核查入口

按 [证据来源规则](platform-and-api-diff.md#6-核查入口与证据记录) 查目标版本官方文档、SDK 声明和实现，仍须用实际工具链执行编译与精度验证：

- SDK：LoadData/MMAD、Cast/ReciprocalCast、Fixpipe 的类型与布局约束；沿源/目标分发检索 `kernel_operator_mm_impl.h`、`MX_BUFFER_SCALE`、`MmadCal`，核对 scale 地址单位。
- SwiGluGroupQuant 案例：`swiglu_hif4_quant_common.h` 中的转换、scale 和输出打包。
- MX 量化案例：`mx_quant_common.h` 中的输出类型、架构分支及 scale 转换。
- IndexFill 案例：`aclnn_index_fill_tensor.cpp` 中的 dtype 校验与回退；支持矩阵仅描述实际查到的算子版本。
