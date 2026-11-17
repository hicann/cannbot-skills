# 阶段 1：评估契约、差异与能力适用性

前置：`[GATE-0] ENV_CONFIRMED`。本阶段产物为 `${EVIDENCE_DIR}/migration_plan.md`。

## LOADED 检查点

| 必读资料 | 记录 |
|---|---|
| [platform-and-api-diff.md](../references/platform-and-api-diff.md) | `[LOADED] 3510-to-920x/platform-and-api-diff` |
| [tiling-and-memory-guide.md](../references/tiling-and-memory-guide.md) | `[LOADED] 3510-to-920x/tiling-and-memory-guide` |
| [feature-catalog.md](../references/feature-catalog.md) | `[LOADED] 3510-to-920x/feature-catalog` |
| [feature-implementation-guide.md](../references/feature-implementation-guide.md) | `[LOADED] 3510-to-920x/feature-implementation-guide` |

命中 Vector/索引、Cube/MIX 或量化信号时，同时按 [route.md](../route.md) 的索引读取对应专题并记录 LOADED。

## Step 1.1：冻结源算子的可观察契约

沿接口→API 调度/图推导→host 注册与 tiling→kernel→调用者的链路分析。记录：

- 输入输出、属性、全部支持 dtype/layout、shape/索引有效域及别名/in-place 语义。
- 计算公式、中间精度、cast/round/饱和/NaN/Inf/零值行为，确定性要求适用哪些路径。
- 源实现和已有测试的依据；有参考实现则核对契约，无 golden 时参照公共 `source-code-reverse-analysis.md` 合成 CPU 参考，并用数学推导/既有测试/原实现交叉核对，避免同源错误互相印证。
- 实际被测 aclnn 与底层算子的对应关系，禁止仅按同名接口推断。

新增量化接口或改变数学/精度要求须已有用户需求；迁移方案不得自行扩大接口。

## Step 1.2：建立必须处置的差异清单

| 检查轴 | 证据与处置记录 |
|---|---|
| 平台接入 | 注册目标覆盖范围、build compute unit、binary 配置、kernel 源选择、API/图/tiling 守卫 |
| API | 实际 SDK 的声明与目标实现、dtype/mask/参数单位、是否依赖内部 intrinsic |
| 资源与位宽 | 容量与核数的接口来源、传递链及硬编码处置；UB 预留与逐 buffer 账本；host→tiling→kernel→API 各字段的最大值、类型、单位和限界 |
| Cube/MIX | 参与核、block/subblock、Fixpipe 输出布局、workspace 分片、Set/Wait 的生产消费关系 |
| 量化 | 数据/scale 的 shape、布局、打包、寻址、舍入、特殊值和支持组合 |

尤其检查容量变化让 tile 变大后，`uint16_t` 索引/计数、循环次数和 API 字段是否越界；检查窄乘法、有符号转换及 `uint64_t` 预算减法下溢。资源可容纳不代表计数、索引或 API 字段可表示。对每条风险给出保留、扩宽、分批、换合法路径或待查证的处理。

## Step 1.3：逐算子选择能力

依据候选能力目录，对每项填写以下内容；同一算子不同 dtype/shape 可有不同结论：

| 能力与目标 | 代码信号、dtype/shape 条件 | 源/目标 API 与版本证据 | 适用性 | 实现方式与文件 | 实施状态与理由 | 验证/回退 |
|---|---|---|---|---|---|---|
| 实际候选名；DAV_9201 或 DAV_9202 | 指向具体函数/分支 | 实际文件与版本 | `APPLICABLE / NOT_APPLICABLE / NEEDS_EVIDENCE` | 复用已有 / 改造原路径 / 新增专用模板；kernel、tiling、分发与构建位置 | `REUSED / PLANNED / DEFERRED` | 对应 case 和回退边界 |

同时交付 9201/9202 时，同一能力按两个目标分别填行；共同编程基础不等于每个 API、dtype 或模板均共用准入。9201 案例未覆盖 9202 时需继续核对目标证据，不能停止整个 9202 迁移，也不能直接扩展案例的架构守卫。

逐项评估是必做步骤；`APPLICABLE` 项已有实现覆盖时安排复用验证，否则必须制定改造或新增实现方案，并在 Stage 6 实际尝试；最终是否保留由验收决定。按瓶颈排序，不因需要新增模板或实现复杂而直接跳过。普通 elementwise 算子不因目标支持 DMA 就增加索引路径，未要求量化的接口不自动增加 HiF4。`NEEDS_EVIDENCE` 不能转写为不适用；暂缓项保留未完成状态。

新增模板按 [feature-implementation-guide.md](../references/feature-implementation-guide.md) 明确准入、kernel、host tiling、模板注册/tiling key、分发、构建、回退及新增分支用例；不能只写“使用 DMA API”。

## Step 1.4：形成改造清单与验证计划

将差异清单与能力矩阵落实为逐项实施方案，列出必改文件、修改依据、保留路径、依赖关系和验证用例。验证计划至少包含：

1. ≥30 个常规精度 case，并覆盖全部契约 dtype/layout 及边界。
2. 命中路径专项：uint16/其他实际字段临界点前后、满 tile/尾 tile、DMA offset/重复索引，以及 UB 扩容后才进入的分支；不适用项写理由。
3. 源路径回归，特别是共用 arch35 文件中的 3510 分支；明确是否具备源平台执行条件。
4. Stage 5 所选目标的兼容基线与 Stage 6 每项 A/B 的相同输入、采集口径、目标和停止条件；9201/9202 分开留存，不能交叉对比替代验收。

## Gate

仅当契约、差异处置、位宽/资源证明、完整能力矩阵、逐适用项实现方案和验证计划均落盘，且阻碍正确性实现的未知项已解决，输出 `[GATE-1] PLAN=READY`。新特性未知项可保留到后续补证但不得使能或计作完成；必要 API 或同步语义未明则输出 `[GATE-1] PLAN=NOT_VERIFIED BLOCKED`。
