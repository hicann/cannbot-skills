# DAV_3510 → DAV_9201 / DAV_9202：平台接入与 API 差异

**使用时机**：Stage 0 确认目标与工具链；Stage 1 扫描完整调用链；Stage 2 实施命中项。

按实际 SDK 核对下列平台与 API 差异，确定命中项及迁移动作。以对应版本的官方接口与目标实现为准；使用其他知识 skill 前先核对其架构覆盖范围。

证据目录 `${EVIDENCE_DIR}` 由 Stage 0 为本次迁移单独建立，所有本页产物均写入该目录。

## 1. 证据来源与适用边界

以实际编译和运行使用的 CANN SDK、配套官方文档及平台接口为主要依据，不要求取得任何外部代码仓的指定提交。

| 核查对象 | 主要依据 | 用途 |
|---|---|---|
| 型号、架构与资源 | 目标 SDK 的平台接口、配置与配套文档 | 确认目标身份和可用资源，记录查询环境与返回值 |
| API 与编译支持 | 目标版本官方 API 文档、实际编译用头文件及目标分发实现 | 核对声明、dtype、参数、同步与编译结果 |
| 算子接入与模板 | 待迁移工程及其实际依赖 | 核对注册、准入、tiling、kernel、构建和回退 |
| 社区算子案例 | 与目标环境兼容的官方仓库实现，按需查阅 | 辅助理解实现模式，不作为 SDK 支持矩阵或硬件规格 |

- 本路线源架构为 `DAV_3510`，目标为 `DAV_9201`、`DAV_9202`；型号映射以目标 CANN SDK 和平台接口核验，方法与边界由 `npu-arch` 的 `references/npu-hardware-params.md` §0.1 维护。
- `ascend960dt:9201` 与 `ascend960pr:9202` 是两组不同的构建/设备身份；同一套迁移流程分别核对实际 API、模板、资源和分发，不能交叉使用产物或验收记录。
- 各专题的文件名、符号和算子名是检索线索，不是指定依赖或已验证支持结论。按实际 SDK 布局定位；文件不存在、宏未开放或实现不匹配时继续核查，不套用案例结论。
- 在 `${EVIDENCE_DIR}/env_info.md` 记录实际 SDK 发行版本、编译头文件来源、官方文档版本和平台查询结果；使用源码依赖时补记其实际版本。环境变化后重查对应声明与实现，不把记录版本变成迁移必须使用的版本。
- 外部 RegBase/SIMT 资料按其声明的架构与版本范围使用。用于 9201/9202 时，核对实际目标版本的官方 API 与实现，明确可复用路径与需要调整的部分。

### 9201/9202 按能力核对

| 能力或案例 | 每个目标分别取得的证据 | 实施规则 |
|---|---|---|
| RegBase / SIMT | 目标 SDK 的声明、分发和实现 | 复用经核验的可行路径，具体 API、dtype 和模式分别验证 |
| `FusedMulSubDst` | 声明的架构守卫、后端分发、dtype 和掩码模式 | 各自核对舍入与精度，不能由 9201 测试代替 9202 |
| Gather DMA / Scatter DMA | SDK 搬运接口与同步约束、实际 host 模板准入 | EmbeddingBag、IndexPutV2 可作检索案例；某模板未准入不能当作全部硬件/API 不支持的结论 |
| Scatter 128 位载体提升 | 目标 API 支持、`PromoteDtype` 或等效逻辑的架构守卫、对齐与尾块约束 | 不默认两个目标选择相同载体，不直接扩大单目标守卫 |
| Packed dtype 与分发 | 公开接口的 dtype 列表、注册和目标分发；可检索 IndexFill 的校验逻辑 | 保持原算子契约；不受支持时阻塞具体路径或采用有依据的合法替代 |
| Cube/MIX 模板 | 核数查询、任务类型、布局和运行时依赖 | FlatQuant、QuantBatchMatmul 可作检索案例，公共分支仍逐目标验证 |

上表是核查清单，不预填某个 SDK 版本的支持矩阵。9201/9202 未查明的能力均记 `NEEDS_EVIDENCE` 并继续查证，不能互相代替或笼统判为不适用。核查入口见本文末节及 Vector、量化、Cube 专题。

## 2. 目标接入检查

平台注册方式按工程规范执行，查阅 `ascendc-registry-invoke-template` 的 `references/arch-registration.md`。本路线记录迁移后的实际接入结果：

| 检查项 | 迁移证据 |
|---|---|
| 平台注册 | 注册结果覆盖所选目标，dtype/format/属性与源契约一致 |
| 构建 | compute unit、kernel 源和 binary 配置覆盖所选目标 |
| 分发 | API、图推导/融合、host tiling、kernel 分支均可到达 |
| 运行 | 实际加载的自定义库、kernel 与目标分支 |

`arch35/` 可以承载共享实现，`arch92/` 可以承载独立路径，按实际代码分发需求选择。

## 3. API 能力与底层实现分别核对

| 项目 | 目标 SDK 核查点 | 迁移动作 |
|---|---|---|
| `Add<..., MERGING>` | 源/目标 API 的掩码语义及 `AddImpl` 实现，是否已由后端完成合并 | 作为实现优化评估，检查掩码外 lane 和寄存器别名 |
| `FusedMulSubDst` | 接口是否声明、目标宏覆盖、dtype 与掩码限制及后端实现 | 按计算模式和数值契约选择，详见 Vector 指南 |
| DataCopyPad / ND2NZ | 实际重载、预读参数、stride 单位和同步条件 | 按准确签名适配，禁止统一追加 `true` |
| Fixpipe | 所需 L0C 输出方向、参数类型、布局与同步的支持范围 | 沿公开 API 使用真实类型，禁止猜造 `FixpipeParamsArch9201` |
| 低精度计算 | LoadData/MMAD 的类型组合、scale 格式与寻址 | 仅对命中量化路径查阅量化指南 |

AscendC API 由 SDK 提供；是否声明、支持哪些参数及生成何种指令取决于 SDK 版本和目标架构。“融合算术”本身不是 9201 独占能力。

## 4. 搬运 API 的核查步骤

1. 列出公共 API、内部 helper、编译器 intrinsic 的全部调用点。
2. 在实际 devkit 中读声明，再读 `dav_3510` 与所选 9201/9202 的实际目标分发和实现；不能依据某个目录名推断另一目标一定复用它。
3. 记录参数单位、范围、默认值、数据方向、允许的核类型和同步前置条件。
4. 优先保留公共 API；直接调用内部接口的代码按目标版本适配。
5. 对参数改变补输出数据对拍和边界用例，不能以编译成功证明搬运正确。

若目标实现出现 `enablePreReadIgnoreSync`、`RemoteDataCopyParams` 等名字，不构成默认使能理由。普通迁移保持既有语义；远程写等能力未命中需求时不扩展。

## 5. 按需路由与记录

| 扫描信号 | 后续参考 |
|---|---|
| 掩码算术、间接索引搬运 | [vector-and-dma-guide.md](vector-and-dma-guide.md) |
| Mmad/LoadData/Fixpipe、MIX、跨核同步 | [cube-migration-guide.md](cube-migration-guide.md) |
| HiF4/FP4/FP8、MX scale、打包 | [quantization-guide.md](quantization-guide.md) |
| UB 扩容、窄索引、tile/stride/workspace | [tiling-and-memory-guide.md](tiling-and-memory-guide.md) |
| 选择优化候选与记录状态 | [feature-catalog.md](feature-catalog.md) |

在 `${EVIDENCE_DIR}/migration_plan.md` 为每个命中项写“目标身份 → 调用位置 → 源行为 → 目标声明/实现 → 处理 → 验收用例”。不能将资料未覆盖外推为 9201/9202 不支持。

## 6. 核查入口与证据记录

- 型号映射、架构枚举与编程模型支持范围：按 `npu-arch` 的 `references/npu-hardware-params.md` §0.1 核对目标 CANN SDK、平台接口与配套文档，在迁移证据中记录实际版本和查询结果。
- 公开接口：从 [CANN 官方文档](https://www.hiascend.com/document) 选择与实际 SDK 匹配的版本，检索本页 API 的原型、支持产品、约束和示例；不以最新文档代替已安装版本。
- SDK 头文件与实现：在实际编译 include 路径检索 `kernel_operator_data_copy_impl.h`、`kernel_operator_fixpipe_impl.h`、`FixpipeL0cToUB` 等文件或符号，沿入口核对所选目标分发。安装包不含所需实现时，可查与该 SDK 匹配的 [asc-devkit 官方仓](https://gitcode.com/cann/asc-devkit)，记录版本对应关系；关系未查明时不能据其声明支持。
- 算子案例：按需在实际工程或兼容版本的 [ops-nn 官方仓](https://gitcode.com/cann/ops-nn) 检索算子名/符号。例如 `aclnn_index_fill_tensor.cpp` 中的 `CheckDtypeValid`、目标 dtype 列表及回退；结论限于查到的算子和版本，不外推为平台限制。

各专题末节列出同一规则下的检索线索。将实际命中的文档小节、SDK 文件/符号、目标守卫、限制和查询/测试结果写入 `${EVIDENCE_DIR}`；路径迁移后按符号追踪，不要求旧路径或固定行号存在。
