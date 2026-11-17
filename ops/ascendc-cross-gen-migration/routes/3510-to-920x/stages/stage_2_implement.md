# 阶段 2：按差异清单实施改造

前置：实际的 `[GATE-1] PLAN=READY`。产物为本次代码差异及 `${EVIDENCE_DIR}/implementation.md`，按 `migration_plan.md` 的逐项方案实施。

## LOADED 检查点

读取 [platform-and-api-diff.md](../references/platform-and-api-diff.md)、[tiling-and-memory-guide.md](../references/tiling-and-memory-guide.md)，分别记录本路线 LOADED；再按 `migration_plan.md` 命中项读取 Vector/DMA、Cube、量化专题。源代码使用 RegBase/SIMT 时可查询相应知识 skill，但其未覆盖所选 9201/9202 目标的平台限制须向实际 SDK 核对。

## Step 2.1：接入并复用源实现

1. 按工程注册规范接入目标架构，核对注册结果及链接依赖，覆盖原算子契约中的 dtype/format。
2. 同步 build compute unit、kernel 源选择、二进制配置、host tiling 和 API/图路径。`arch35` 共用与 `arch92` 专属入口按真实分支需求选择，不全局重命名目录。
3. 检查 `__NPU_ARCH__ == 3510`、`__CCE_AICORE__ == 310` 等守卫；仅在已证明支持的分支增加目标，不能全仓机械替换。

## Step 2.2：完成正确性所需的兼容调整

| 类别 | 实施要求 |
|---|---|
| API | 以当前编译所用版本为准；优先公开接口，核实内部 intrinsic 参数变化；掩码与舍入保持契约 |
| 位宽 | 中间计算先按范围选足够宽类型再乘；转换前检查范围，减法前检查被减数；固定位宽字段需拆 tile/分批 |
| 资源 | 逐 buffer 更新对齐、生命周期、双缓冲、DMA offset/mask、scale/bias 和 cache 预留；不得把总 UB 直接作工作预算 |
| Cube/MIX | 同时调整分核、子块、输出布局、workspace 地址与同步配对；保留不受影响的源路径 |
| 量化 | 全链保持 dtype/format/packing/scale 合法性；扩展接口必须有需求依据 |

正确性适配之外的新 API、DMA、128 位搬运和量化能力在 Stage 6 实施，并与本阶段建立的兼容实现做 A/B；若属于实现正确性所必需的替代，则在此实施并说明原因，不能声称已有独立优化收益。迁移范围包括为原算子新增目标平台实现，不能因原实现已能运行就省略 Stage 6。

需要新增或重组实现时，MUST 读取 [feature-implementation-guide.md](../references/feature-implementation-guide.md)，记录 `[LOADED] 3510-to-920x/feature-implementation-guide`，完成准入、kernel/tiling、注册分发、构建和回退的完整改动；这些改动属于本阶段必要适配时一并接受 Stage 3/4 验收。

## Step 2.3：记录修改与待验证结论

`implementation.md` 按计划项列出实际文件/函数、改变原因、相关公式和范围证明、保留/回退路径、对应 case。使用 Git 差异核对本任务修改，保留其他用户改动；不通过 `reset --hard` 或整文件恢复来回退。

## Gate

必改项已实现，资源/位宽与接口检查有对应证据，测试计划已与实际分支同步后，输出 `[GATE-2] FILES_MODIFIED=[实际路径...]`。这只表示改造完成待构建，不代表运行正确；存在未解决必要项时输出 `IMPLEMENTATION=INCOMPLETE BLOCKED`。
