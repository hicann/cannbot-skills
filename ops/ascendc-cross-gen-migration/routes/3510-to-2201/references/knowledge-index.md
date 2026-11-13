# 知识地图与来源

> 本文件回答三个问题：**迁移要用到哪些事实、这些事实的真源在哪、置信度多少**。任何写进代码或报告的架构断言，都应能回溯到这里的某一行。

## 1. 关键术语

| 术语 | 含义 | 本技能中的用法 |
|---|---|---|
| A5 | Ascend 950（950PR / 950DT），`SocVersion=ASCEND950`，`NpuArch=DAV_3510`，`__NPU_ARCH__=3510`，目录 `arch35`，token `dav-c310` | **源平台** |
| A2 | Ascend 910B1~B4 / 910B2C，`SocVersion=ASCEND910B`，`NpuArch=DAV_2201`，目录 `arch22`，token `dav-c220` | **目标平台** |
| A3 | Ascend910_93，`NpuArch` 同为 `DAV_2201`（运行时 `SocVersion` 映射到 `ASCEND910B`） | 与 A2 同架构，差异只在 SKU 参数 |
| arch22 / arch35 | 算子仓按 DAV 编号前两位命名的架构目录 | 目录名即架构承诺，不可混装 |
| `__CCE_AICORE__` | 编译器宏：220 = arch22，310 = arch35 | 条件编译分界的现场验证点 |

> 芯片型号 ↔ 架构映射的唯一真源是 `npu-arch`（其表经白皮书与 CANN ini 验证）。本文件只做速记，冲突时以 `npu-arch` 为准。

## 2. 事实真源表

| 事实 | 真源 | 置信度 | 现场复核方式 |
|---|---|---|---|
| DAV_2201 参数：L1 512KB、L0C 128KB、UB 192KB、BT 1KB、支持 4:2 稀疏、SIMD-Membase | `npu-arch` 的 `references/npu-hardware-params.md` §2.3 | HIGH | `GetCoreMemSize` 实测；`platform_config/*.ini` |
| DAV_3510 参数：L1 512KB、L0C 256KB、UB 248KB/AIV、BT 4KB、不支持 4:2、RegBase+SIMT | `npu-arch` 同上 §2.5 | HIGH | 同上 |
| A5 相对 A2 的数据通路增删：新增 L0C→UB、UB→L1；删除 L1→GM、GM→L0A/L0B、`SetLoadDataBoundary`；分形 ZZ→NZ | `ascendc-api-best-practices` 的 `references/api-cross-gen-migration.md`、`ascendc-cross-gen-migration` 的 `references/api-diff-guide.md` §2.4 | MEDIUM | 查目标 CANN 头文件与官方迁移指导 |
| A5 裁剪 subnormal（FTZ），A2 原生支持 subnormal | `ascendc-api-best-practices` 的 `references/api-cross-gen-migration.md` §1 | HIGH | 精度用例实测（subnormal 输入/中间值） |
| A5 新增 FP8 / HiFloat8 / MXFP8 / MXFP4 Cube 数据类型 | `npu-arch` 的 SKILL.md「DAV_3510 关键变化」 | MEDIUM | 目标 CANN 的 `ge` 数据类型枚举与 kernel 头文件 |
| A5 独有同步：BufferID 取代 set/wait、同步模式 4、Mutex；A2 用 `SetFlag`/`WaitFlag` 事件 | `npu-arch` 的 `references/npu-arch-guide.md`；`ascendc-cross-gen-migration` 的 `references/cube-migration-guide.md` | MEDIUM | 查 A2 `sync_control` 头文件与官方文档 |
| A2 有高吞吐 `vnchwconv`（`TransDataTo5HD`）、`VSLDB` 为 A5 的一半、`BilinearInterpolation` 在 A2 为原生融合实现 | `ascendc-cross-gen-migration` 的 `references/api-diff-guide.md` §3 | MEDIUM | 实测对比（阶段 5） |
| A5 上 SIMT DCache 复用 UB，需预留 40KB；A2 无此预留。注意 `l1-guide.md` 的 216KB 是物理 256KB 起算口径，与本路线 248−40=208KB 的用户可用口径不同 | `ascendc-cross-gen-migration` 的 `references/l1-guide.md`「UB 预留规范」、`references/ub-budget-guide.md` §1.3 | MEDIUM | 目标平台 tiling 实测 |
| 精度容差（rtol/atol、matched_ratio 0.99、max_abs_error） | `ops-precision-standard`（唯一真源） | HIGH | 直接引用，不复制 |
| msprof 指标字段（`aiv_vec_ratio`、`aic_mte2_ratio` 等）与 arch22 可分析性 | `ops-profiling` 的 `references/csv_fields_reference.md`、SKILL.md「深度分析」 | HIGH | 采集产物字段核对 |
| arch22 工程接线：`arch22/`、`config/ascend910b/`、`AddConfig("ascend910b")`、`--soc=ascend910b` | `ascendc-cross-gen-migration` 的 `references/l1-guide.md`、`routes/2201-to-2002/references/m200-port-guide.md` | MEDIUM | 目标仓 `build.sh --help`、现有 arch22 算子先例 |

## 3. 按阶段加载

| 阶段 | MUST 读 | 按需读 |
|---|---|---|
| 0 | `references/input-contract.md`、`stages/stage_0_env.md` | `npu-arch`（型号映射） |
| 1 | `references/a5-only-inventory.md`、`stages/stage_1_assess.md` | `ascendc-regbase-best-practice`、`ascendc-simt-best-practices`（识别形态） |
| 2 | `references/api-migration-recipes.md`、`references/resource-budget.md`、`references/build-and-config.md` | `ascendc-api-best-practices`（具体 API 签名）、`ascendc-tiling-design` |
| 3 | `references/build-and-config.md`、`references/pitfalls.md` | 目标仓构建文档 |
| 4 | `references/precision-verification.md`、`ops-precision-standard` | `ascendc-precision-debug`（失败时） |
| 5 | `references/performance-tuning-a2.md`、`ops-profiling` | `ascendc-performance-best-practices`、`ops-theoretical-performance-estimation` |
| 6 | `assets/migration-report.md.template` | — |

## 4. 与其他技能的边界

| 技能 | 关系 |
|---|---|
| `ascendc-cross-gen-migration` | **反向**技能（A2→A5）。本技能不复用其路线包：其决策树、阶段语义、Stage 6「A5 特性使能」在迁移方向不成立。只把其路线包的差异清单当作"反向差异的交叉验证材料"，不照搬流程。 |
| `ascendc-regbase-best-practice` / `ascendc-simt-best-practices` | 只用于**识别要删除的代码形态**与理解其语义；本技能不教 RegBase/SIMT 开发。 |
| `ascendc-api-best-practices` | API 签名、参数、限制的查询入口。 |
| `ascendc-tiling-design` | A2 tiling 方法论；本技能只补充"预算变小后如何重算"。 |
| `ops-precision-standard` | 阈值与用例设计的真源。 |
| `ops-profiling` | 采集与解读的唯一入口。 |

## 5. 置信度纪律

- 报告与代码中出现的每个"A2 不支持 / 支持 X"结论，MUST 标注依据（文件路径 + 小节 / 头文件 grep 结果 / 实测命令）。
- `MEDIUM` 级事实在**进入代码前**必须现场复核一次；复核不了的，标记为 `UNVERIFIED` 并在阶段 1 报告中列为待确认项，不得据此直接删除功能。
- 不允许用"A5 是 X，所以 A2 一定不是 X"的推断代替查证：反向差异未必对称（例如 A5 删除的通路在 A2 存在，但 A2 的 `FixpipeParams` 命名与字段单位与 A5 不同）。
