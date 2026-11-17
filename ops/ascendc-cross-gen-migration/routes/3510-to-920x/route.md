# 路线 DAV_3510 → DAV_9201 / DAV_9202

本包由根 `SKILL.md` 的 Step R 进入，覆盖已有 `DAV_3510` 算子向 `DAV_9201`、`DAV_9202` 的迁移。构建目标与架构的映射依赖 `npu-arch`，执行时使用以下参数：

| 目标架构 | 构建计算单元 `SOC` | `TARGET_ARCH` |
|---|---|---|
| `DAV_9201` | `ascend960dt` | `9201` |
| `DAV_9202` | `ascend960pr` | `9202` |

每次执行绑定一组目标。用户要求同时交付 9201/9202 时，分别生成能力矩阵、构建安装产物和精度/性能证据，不能用一侧结果代替另一侧。两者共用本流程，具体 API、dtype、资源和模板准入按实际目标核对；某个模板只覆盖 9201 不会把 9202 排除出迁移范围。

本路线以目标 CANN SDK、配套官方文档和平台接口为主要核验依据，尚无本路线的端到端真机验收记录。各专题中的算子名与符号用于检索参考实现，不代表任意工具链、dtype 和 shape 都已验证。

## 路线原则与实施方案

3510 与 9201/9202 已有 RegBase、SIMT 等共同编程基础。先保留源算子的接口、计算公式、数值契约和已有有效实现，再处理已证实的差异；增加量化输入输出等接口扩展需有用户需求。

根据实际调用链形成迁移方案，列出以下工作：

- 平台注册、构建与分发接入，以及可复用实现的实际代码位置。
- 必须处置的 API、资源、字段位宽、布局和同步差异，以及对应文件、修改依据和验证用例。
- 目标新特性的逐算子适用性、实现方式及验证计划。Stage 6 为必做：复用已有能力，并为尚未覆盖的适用项改造或新增实现。

## 阶段、产物与 Gate

在源工程中为本次迁移建立独立的 `${EVIDENCE_DIR}`，默认 `${OP_DIR}/migration/3510-to-920x/${SOC}/<run-id>/`，run-id 取实际时间或任务标识，已有目录不得覆盖。以下产物均位于该目录，不同目标的记录不得混用。

| 阶段 | 开始时必读文件 | 主要产物 | 完成标记 |
|---|---|---|---|
| 0 环境确认 | [stage_0_env.md](stages/stage_0_env.md) | `env_info.md` | `[GATE-0] ENV_CONFIRMED` |
| 1 评估 | [stage_1_assess.md](stages/stage_1_assess.md) | `migration_plan.md` | `[GATE-1] PLAN=READY` |
| 2 改造 | [stage_2_implement.md](stages/stage_2_implement.md) | `implementation.md`、实际差异 | `[GATE-2] FILES_MODIFIED=[...]` |
| 3 构建安装 | [stage_3_build.md](stages/stage_3_build.md) | `build_install.md`、日志与安装清单 | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 正确性 | [stage_4_precision.md](stages/stage_4_precision.md) | `precision_report.md`、逐例输出、调用链证据 | `[GATE-4] PRECISION=N/N_PASS` |
| 5 性能基线 | [stage_5_performance.md](stages/stage_5_performance.md) | `perf_report.md`、原始 profiler 数据 | `[GATE-5] PERF=COLLECTED` |
| 6 新特性实现与验收 | [stage_6_feature_enable.md](stages/stage_6_feature_enable.md) | `feature_trials.md`、新增/改造实现、A/B 数据与回退记录 | `[GATE-6] FEATURES=EVALUATED RETAINED=k/n DEFERRED=d` |

Gate 表中的值由实际证据填入。缺环境、执行失败或缺必需产物时输出对应 `BLOCKED / NOT_VERIFIED`，保留实际结果，不进入依赖该证据的后续执行。静态调研可交付分析，但不能替代这些 Gate。

Stage 6 必须完成“适用性判断 → 实现方式选择 → 复用验证或新增/改造 → 构建、精度与性能验证 → 保留或回退”。任务范围内适用且未被已有实现覆盖的项必须进入试验；需要新增 kernel/tiling 模板、调整分发或重构数据通路本身不是跳过理由。`n` 为本次应完成的试验项数，`k` 为通过验收并保留的项数，`d` 为暂缓项数。仅在任务范围内全部候选不适用或已验证复用时允许 `0/0`；暂缓项单列为未完成，不得以只完成评估冒充新特性适配完成。具体范围及 Gate 判据见 Stage 6。

## 参考资料按信号加载

| 资料 | 何时读取 | LOADED 标记 |
|---|---|---|
| [platform-and-api-diff.md](references/platform-and-api-diff.md) | Stage 0/1 必读；注册、API 或版本差异 | `[LOADED] 3510-to-920x/platform-and-api-diff` |
| [tiling-and-memory-guide.md](references/tiling-and-memory-guide.md) | Stage 1/2/4 必读；UB、位宽、计数、偏移检查 | `[LOADED] 3510-to-920x/tiling-and-memory-guide` |
| [feature-catalog.md](references/feature-catalog.md) | Stage 1 与 Stage 6 必读；逐算子能力选择 | `[LOADED] 3510-to-920x/feature-catalog` |
| [feature-implementation-guide.md](references/feature-implementation-guide.md) | Stage 1 制定实现方案；Stage 2/6 改造或新增目标实现 | `[LOADED] 3510-to-920x/feature-implementation-guide` |
| [vector-and-dma-guide.md](references/vector-and-dma-guide.md) | Vector 算术、间接索引、Gather/Scatter 搬运或 SIMT 路径 | `[LOADED] 3510-to-920x/vector-and-dma-guide` |
| [cube-migration-guide.md](references/cube-migration-guide.md) | `Mmad / LoadData / Fixpipe / MIX`、跨核协作 | `[LOADED] 3510-to-920x/cube-migration-guide` |
| [quantization-guide.md](references/quantization-guide.md) | FP8/FP4/HiF4/MX、scale、packed dtype | `[LOADED] 3510-to-920x/quantization-guide` |

LOADED 记录需包含实际读取的文件、版本和适用小节，落盘到对应阶段产物。外部知识按 skill 名称定位，依赖缺失时按根入口及本阶段说明处理；其他路线的专属清单不作为本路线的架构依据。

## 路线专属约束

1. **按算子判定并实施适用能力。** 在 `migration_plan.md` 中为候选项填写 `APPLICABLE / NOT_APPLICABLE / NEEDS_EVIDENCE`、实现方式与理由，实施状态单独记录。适用项按复用验证、原路径改造或新增专用模板落实；没有适用证据不能使能，也不能把缺知识写成不适用。
2. **验证目标接入。** 按工程已有规范完成平台注册，核对目标覆盖范围、dtype/format、构建及分发；注册方式查阅 `ascendc-registry-invoke-template` 的 `references/arch-registration.md`。
3. **API 语义与实现分开核实。** 对 `Add<MERGING>`、`FusedMulSubDst` 等候选，读取实际 SDK 的声明、目标分发和实现，分别确认 dtype、掩码、舍入和版本支持，不将整个融合算术类别归为 DAV_9201/DAV_9202 独占。
4. **资源参数通过接口获取，容量与字段位宽分别限界。** 按 [平台参数获取与传递](references/tiling-and-memory-guide.md#平台参数获取与传递) 查询 UB 等容量及可用核数，核对模板预留和 host→tiling→kernel 的传递，不写死硬件规格。逐条检查 host→tiling→kernel→API 中的索引、计数、字节偏移和有符号性；支持放进 UB 不代表 `uint16_t` 或硬件字段能表示。固定位宽 API 需分块或回退。
5. **Cube/MIX 按具体模板验证。** 分核、子块、bank 布局、Fixpipe 输出和同步以目标 API 与模板契约为准，不能把某个算子的 1:1 或四 bank 布局当作所有 DAV_9201/DAV_9202 SKU 的统一规格。
6. **构建按本路线模板。** Stage 3 使用本包的参数化模板，避免复用公共模板中固定的 transformer 包名、vendor 后缀和 host 架构。
7. **精度覆盖触发路径。** 保留公共标准与至少 30 个常规用例；另补位宽、DMA、资源和大 shape 触发用例。本路线的大 shape 专项允许超过 200K 元素，必须先验证预算并分批运行，不因旧上限漏掉风险路径。
8. **优化不改变验收口径。** 保留同目标设备上的兼容基线，逐能力做全量精度与性能 A/B；失败、未试验、回退分开报告。保持原算子接口与精度阈值，性能目标和试验预算按本次任务约定。

## 知识依据与适用边界

按 [证据来源与适用边界](references/platform-and-api-diff.md#1-证据来源与适用边界) 核对目标 SDK、官方文档、平台查询和实际工程依赖；源码案例仅作辅助检索，不绑定外部仓库提交。将本次实际使用的版本与证据记入运行记录。旧资料未列 9201 或 9202 不等于目标不支持；未查明的 SKU 资源、参数语义或同步支持保留 `NEEDS_EVIDENCE`，不能据此宣称迁移通过。
