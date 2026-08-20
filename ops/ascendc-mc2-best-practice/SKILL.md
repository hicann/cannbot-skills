---
name: ascendc-mc2-best-practice
description: Ascend C MC2 通算融合算子（多卡通信+计算融合）开发最佳实践。当用户需要设计、实现、调试、移植或优化通算融合算子，或提及"MC2"、"SHMEM"、"APACE"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"时使用。
---

# Ascend C MC2 通算融合算子开发最佳实践

MC2（Matrix Computation & Communication）= 多卡间集合通信 + 单卡内 Blaze Cube 计算 + 通算两层流水掩盖通信开销。提供 MC2 通算融合算子开发的技术参考：通信与计算的 API 选择、时序约束、验收标准。

本 skill 覆盖三种编码底座，每个底座支持特定的通信路径、算子类型与芯片组合（确切组合以 [`capability-declaration.md`](references/capability-declaration.md) 路径登记表为准）：

| 底座 | 支持的通信路径 | 支持的算子类型 | 支持芯片 | 知识目录 |
|------|---------|---------|------|------|
| **blaze-shmem** | AIV+URMA | collective-comm（AllToAll+Matmul、AllReduce+Matmul、TP/SP 融合） | dav-3510（Ascend 950） | [`references/foundations/blaze-shmem/`](references/foundations/blaze-shmem/) |
| **apace** | AIV+URMA | collective-comm（同上；moe 类 planned） | dav-3510（Ascend 950） | [`references/foundations/apace/`](references/foundations/apace/) |
| **ascendc-api** | AIV+UBMEM | moe（MoE Dispatch/Combine、专家并行 EP） | dav-3510（A5）+ dav-2201（A3）双平台 | [`references/foundations/ascendc-api/`](references/foundations/ascendc-api/) |

> 三种底座抽象层级不同（blaze-shmem 是库+模板手工组装，apace 是模板框架，ascendc-api 是直接使用ascendc API），但在"选什么写代码"这个决策点上是并列选项。apace 模板库虽基于 Ascend C 基础 API 构建，但与 ascendc-api 路线约束集、工程边界、可修改范围完全不同——详见 §1/§2/§3 各路线特有约束。

## 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的通算融合算子（AllToAll+Matmul、AllReduce+Matmul、MoE Dispatch/Combine、多卡 EP/TP/SP 融合 Kernel）
- 关键词："MC2"、"SHMEM"、"APACE"、"apace"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"
- 代码：现有工程同时出现通信 API（`shmem.h`/`aclshmem*` 或 `collective_comm_api.h`/`CollectiveComm`）与 `blaze/gemm/` 模板（blaze-shmem/apace 路线），或出现 `winContext`/`mc2Context`/`HcclOpParam` 与 MTE 通信窗口结构（MTE通信 → ascendc-api 路线）

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ `ascendc-blaze-best-practice`
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 通用 Ascend C API 用法查询 → `ascendc-api-best-practices`
- 非 3510 架构的 AIV+URMA 通算融合（MTE通信除外，支持 A3/A5）

> 确切的支持组合（chip × 算子类型 × 调用形态 × 通信路径 × 编程抽象）以 [`references/capability-declaration.md`](references/capability-declaration.md) 路线登记表为准；表内同时登记**明确不支持**的组合及原因，命中否定行时直接答复用户不可用 + 替代建议。

## 决策树：先判断算子类型

```
用户要做通算融合算子
│
├─ AllToAll+Matmul / AllReduce+Matmul / TP/SP 通算融合？
│   → AIV+URMA 路径（仅 dav-3510 / Ascend 950）
│      ├─ 提到"apace"/"CollectiveComm"或代码在 ops-transformer/apace/ 下？→ apace 路线（下方§2）
│      └─ 否则 → blaze-shmem 路线（下方§1）
│
├─ MoE Dispatch / MoE Combine / mega_moe / 专家并行 EP？
│   → MTE通信 → ascendc-api 路线（A3/A5 双平台，下方§3）
│      ├─ 先阅读已有实现？→ references/foundations/ascendc-api/moe-dispatch-combine/reading/
│      └─ 生成新算子或改造？→ references/foundations/ascendc-api/moe-dispatch-combine/samples/
│
└─ 不确定？
    → 看通信路径：SHMEM API = AIV+URMA 路径（仅 dav-3510）；winContext/mc2Context = MTE通信（A3/A5）
```

选定路线后：

1. 查 [`capability-declaration.md`](references/capability-declaration.md)，确认 chip × 算子类型 × 调用形态组合命中 `supported` 行（命中否定行或无行 → 答复用户不可用 + 原因）
2. 进入需求分析（下节），输出 REQUIREMENTS.md
3. 按 §1/§2/§3 进入对应路线的 references

## 需求分析（全路线共享）

> **MC2 特有需求维度**：除算子名/数学定义/dtype/shape 外，需求收集时还需明确通信路径（AIV+URMA 或 MTE通信）与编程抽象底座（blaze-shmem / apace / ascendc-api）。

| 文档 | 何时读 |
|------|--------|
| [`capability-declaration.md`](references/capability-declaration.md) | 确认路线可行性、多行命中时选编程抽象、定位参考实现 |
| [`requirement-analysis/grill-protocol.md`](references/requirement-analysis/grill-protocol.md) | 需求拷问 8 维度判据（生成 REQUIREMENTS.md 前逐项拷问，先拷问再生成） |
| [`requirement-analysis/template.md`](references/requirement-analysis/template.md) | 生成需求文档时（含 MC2 特有章节：组网规模、通信设计、开发就绪判断） |
| [`requirement-analysis/comm-path-decision.md`](references/requirement-analysis/comm-path-decision.md) | 用户询问通信路径选型时（HCCL 高阶 / AIV+URMA / MTE通信 / CCU） |
| [`requirement-analysis/classification.md`](references/requirement-analysis/classification.md) | 确定算子类型、查已有算子与可信源清单 |
| [`requirement-analysis/quality-checklist.md`](references/requirement-analysis/quality-checklist.md) | 需求文档生成后自检（14 项 + 开发就绪闸门） |
| [`operators/collective-comm.md`](references/operators/collective-comm.md) | 设计集合通信类算子时（通信语义、轴切分、通算流水的跨路线共性） |
| [`codebase-analysis.md`](references/codebase-analysis.md) | Brownfield 从代码推断路线坐标（Stub；Greenfield 不读） |

## 红线（全路线共性，与底座无关）

| # | 红线 | 一句话理由 | 详见 |
|---|------|-----------|------|
| ① | 架构白名单以路线登记表为准 | 未验证组合（如 dav-2201 × AIV+URMA）禁止使用 | [`capability-declaration.md`](references/capability-declaration.md) |
| ② | 性能采集必须刷 L2 cache | 前一轮热度会污染本轮指标 | [`shared/profiling_mc2.md`](references/shared/profiling_mc2.md) |

> 以下两条是 **AIV+URMA 路径下 blaze-shmem / apace 底座的选择性约束**，不是全路线共性——若未来在 ascendc-api 底座上构建集合通信类通算融合，HCCL 高阶 API 和 `AscendC::Matmul` 高阶 API 恰是可用路径：
> - 禁止 HCCL 高阶 API（`Hccl::*`）—— HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（详见 [`blaze-shmem/comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5，7 类 18 个 API 清单）
> - Matmul 走 Blaze 模板，禁止 `AscendC::Matmul` 高阶 API —— 同理（详见 [`blaze-shmem/matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) / [`apace/compute.md`](references/foundations/apace/compute.md)）

各底座特有约束与逐项审查清单见 §1/§2/§3。

---

## §1 blaze-shmem 路线

SHMEM 通信库（cann/shmem，与 HCOMM 无关）驱动 AIV+URMA 跨卡搬运，Blaze 计算模板负责 Cube 计算，二者手工组装为独立 CMake 工程（自带 `include/` 全套）。通信侧 host `aclshmem_*`、device `aclshmemx_udma_*` + `aclshmem_barrier_all`；计算侧 `Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`。

### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构白名单 | CMakeLists.txt 中 `npu-arch` = dav-3510 | 全路线共性红线 ① |
| R2 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到 | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5（7 类 18 个 API 清单） |
| R3 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景 | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmem_barrier_all` | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R6 | L2 flush 证据 | 代码中含 L2 flush kernel 调用或等效实现 | 全路线共性红线 ② |
| R7 | 流程门禁完整 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；环境检查通过 | — |

**逐项审查清单**：[`review-checklist.md`](references/foundations/blaze-shmem/review-checklist.md)（含常见 FAIL 原因与修复方向，违反红线项 = FAIL）。

### References

| 文档 | 何时读 |
|------|--------|
| [`references/foundations/blaze-shmem/workflow_integration.md`](references/foundations/blaze-shmem/workflow_integration.md) | 设计 MC2 算子前，看 MC2 场景的技术要点和门禁 |
| [`references/foundations/blaze-shmem/mc2_architecture.md`](references/foundations/blaze-shmem/mc2_architecture.md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/foundations/blaze-shmem/comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) | 写/改通信层，查 SHMEM API、UDMA 用法、禁止 HCCL 清单（§5）、扩展其他通信原语 |
| [`references/foundations/blaze-shmem/matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | 性能采集与调优时：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 精度调试阶段用 tileCnt=1 做串行基线；性能调优阶段扫描 tileCnt 找最优值 |
| [`references/foundations/blaze-shmem/codebase_map.md`](references/foundations/blaze-shmem/codebase_map.md) | 工程搭建时，定位"哪些文件改/不改"与改造食谱 |
| `references/foundations/blaze-shmem/all_to_all_matmul/` | 编译验证过的基底工程，blaze-shmem 路线所有 MC2 算子的起手模板 |

---

## §2 apace 路线

APACE（Ascend PArallel Communication-compute Engine）是通算融合算子的架构底座，提供可复用的 block 层接口、kernel 层参考实现和 tiling 算法。APACE 自建通信基础 API（基于 HCOMM 通信基础库构建，与 HCCL 集合通信库无关）。apace 路线在 `kernel/<op>/` 下新建算子，复用稳定共享层 `block/` `tiling/`。

> **apace 路线接口契约基准**：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer) master 最新主线 `mc2/common/op_kernel/apace/` 子树。关键文件：`block/aiv_comm/collective_comm_api.h`（四段式通信 API 契约）、`kernel/all_to_all_quant_matmul/` 与 `kernel/all_gather_quant_matmul/`（参考实现）、`tests/st/`（ST 测试工程 + `__global__` 入口）。样例代码经 scripts/fetch_apace.sh 从官网现取现读（默认跟踪 master 最新，实际 commit 记录于仓内 `.apace_fetch_manifest.json`；`--ref <commit>` 可锚定复现）；引用路径为仓内相对路径（apace/... 即 mc2/common/op_kernel/apace/...）。

### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 禁止 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]` | 会导致 AIC/AIV 串行调度→死锁（`aclError:507015`）；核配比唯一由 `KERNEL_TYPE_MIX_AIC_1_1` 保证为 1:1 | [`architecture.md`](references/foundations/apace/architecture.md) §10 ① |
| R2 | 有 `KERNEL_TYPE_MIX_AIC_1_1` | 每个入口函数都含核配比声明 | [`architecture.md`](references/foundations/apace/architecture.md) §10 ① |
| R3 | 入口变体与参考算子一致 | PUT=4 dtype 变体入口于 `kernel_launcher.h`；AG=单入口于 impl.h | [`operator-anatomy.md`](references/foundations/apace/operator-anatomy.md) |
| R4 | `block/` `tiling/` 未修改 | 与官网仓原始文件完全一致 | [`architecture.md`](references/foundations/apace/architecture.md) §10 ③ |
| R5 | CrossCore flag idx 配对 | AIV `WaitFlag` idx == AIC `SetFlag` idx | [`fusion.md`](references/foundations/apace/fusion.md) §3 |
| R6 | CommContext 与引擎匹配 | UDMA 模式有 `__gm__ CommContext*`；HCCL windows 无 | [`communication.md`](references/foundations/apace/communication.md) |
| R7 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景（与 blaze-shmem 共享此约束） | [`compute.md`](references/foundations/apace/compute.md) |
| R8 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（与 blaze-shmem 共享此约束） | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5 |

**逐项审查清单**：[`review-checklist.md`](references/foundations/apace/review-checklist.md)（含常见 FAIL 原因与修复方向，违反红线项 = FAIL）。

### References

**学习路径**（按序阅读建立完整认知）：

| 顺序 | 文档 | 读什么 |
|:---|:---|:---|
| 1 | [`references/foundations/apace/architecture.md`](references/foundations/apace/architecture.md) | **入门·心智模型**：三层架构、组合模式、GET/PUT 方向、四大约束（§10） |
| 2 | [`references/foundations/apace/compute.md`](references/foundations/apace/compute.md) | **计算**：MMAD 流水原理、Blaze 接口（BlockMmad/DispatchPolicy/BlockScheduler/Layout）、QuantMatmulMxKernel 骨架、MatmulMode、FragmentTensor |
| 3 | [`references/foundations/apace/communication.md`](references/foundations/apace/communication.md) | **通信**：URMA/Win 区原理、CollectiveComm 四段式契约、GET/PUT 钩子、TeamBarrier/CrossCore flag/SyncAll、host 建链机制 |
| 4 | [`references/foundations/apace/fusion.md`](references/foundations/apace/fusion.md) | **通算融合**：重叠原理、GET/PUT 选型、flag 编排模式、环形回压、localMatmul 0/1/2、winOffset 复用、扩展语义推导 |
| 5 | [`references/foundations/apace/operator-anatomy.md`](references/foundations/apace/operator-anatomy.md) | **算子解剖·kernel 侧**：已实现算子共性模式、tiling_data 结构、Impl 契约、入口函数规则 |
| 6 | [`references/foundations/apace/host-and-testing.md`](references/foundations/apace/host-and-testing.md) | **算子解剖·host 与测试**：host 初始化序列、launch、perf 模式、ST 工程与性能采集模板 |
| 7 | [`references/foundations/apace/development-guide.md`](references/foundations/apace/development-guide.md) | **开发新算子**：REUSE/MODIFY 地图、开发步骤、改造场景食谱、验收清单 |

**按需查阅**：

| 文档 | 何时读 |
|------|--------|
| [`references/foundations/apace/workflow_integration.md`](references/foundations/apace/workflow_integration.md) | 设计 apace 算子前，看 apace 场景的技术要点和门禁 |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 通算并行调优：tileCnt 两阶段策略（两路线共享） |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | 性能采集：msprof + L2 flush + 多卡后处理（两路线共享） |
| `scripts/fetch_apace.sh` | 从官网仓现取最新 apace 样例代码（kernel 头文件 + ST 测试工程 + 共享层），支持 `APACE_REPO` 环境变量或 sparse clone |

**失败排查导航**：

| 现象 | 去哪里查 |
|:---|:---|
| 死锁（aclError:507015） | [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) Step 5 修复路径表（注意 507015 有 schedmode/MTE 未排空双根因） |
| localMatmul=1 MTE 异常 | [`fusion.md`](references/foundations/apace/fusion.md) §5 PipeBarrier 修复方案 |
| 精度不达标 | [`fusion.md`](references/foundations/apace/fusion.md) §5 精度风险与回退路径 + [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) Step 6 |
| 编译失败 | [`development-guide.md`](references/foundations/apace/development-guide.md) 常见构建失败对照表 |
| Blaze matmul 排错 | [`compute.md`](references/foundations/apace/compute.md) §8 排错速查表 |
| 通信时序错误 | [`communication.md`](references/foundations/apace/communication.md) 常见陷阱 |

**PUT 模式补充阅读**：

| 主题 | 文档 | 章节 |
|:---|:---|:---|
| PUT 模式 Run 编排与验收 | [`fusion.md`](references/foundations/apace/fusion.md) | §4 PUT 算子级编排验收 |
| PUT 钩子差异 | [`communication.md`](references/foundations/apace/communication.md) | §3 GET/PUT 钩子职责 |
| flag 编排与回压 | [`fusion.md`](references/foundations/apace/fusion.md) | §3 Flag 编排模式 |
| winOffset 多对象复用 | [`fusion.md`](references/foundations/apace/fusion.md) | §6 多对象复用与扩展语义推导 |
| localMatmul 模式选择 | [`fusion.md`](references/foundations/apace/fusion.md) | §5 localMatmul 模式（0/1/2） |
| AtomicAdd 精度风险 | [`fusion.md`](references/foundations/apace/fusion.md) | §5 精度风险分析 |
| ReduceScatter 替代实现 | [`fusion.md`](references/foundations/apace/fusion.md) | §6 AllToAll PUT + AtomicAdd 模式推导 |
| AllGather PUT 模式 | [`operator-anatomy.md`](references/foundations/apace/operator-anatomy.md) | §6 AllGather 变体 |

---

## §3 ascendc-api 路线（MTE通信，MoE Dispatch/Combine）

裸 Ascend C API 全自建（不经任何通信库/模板库）。通信路径为 MTE通信 = AIV+UBMEM（AIV 触发 MTE 执行跨卡搬运），host 侧经 HCCL 分配 window 资源——此处 HCCL 仅做资源分配，非 HCCL 集合通信库高阶 API。支持 A3（dav-2201）+ A5（Ascend 950/dav-3510）双平台，compat 层抹平两平台 window 地址结构差异。注意区别于 apace 路线——apace 模板库虽基于 Ascend C API 构建（其通信基础 API 基于 HCOMM），但属独立编码底座。

> **跨路径设计原则**：阅读或改造 `dispatch`、`combine` 任一侧时，都要同步核对另一侧的接口和状态语义；`expandIdx`、`epRecvCounts`、`epSendCounts` 等中间量必须成对理解；host 侧只传总核数，kernel 侧负责决定每阶段实际使用多少核以及如何切分。

### 红线 / 约束

#### ① 跨卡搬运走 MTE通信，禁止 HCCL 高阶通信原语做数据搬运

MoE Dispatch/Combine 通过 host 侧 `HcclAllocComResourceByTiling` 创建通信 window 资源，device 侧用 MTE + `DataCopyPad` 访问 window 地址做跨卡数据搬运。禁止用 `Hccl::AlltoAll()`/`Hccl::AlltoAllV()`/`Hccl::AllReduce()` 等高阶通信原语替代——它们是对整块数据的黑盒集合通信，无法在 Kernel 内按 token 粒度插入路由计算和状态协议逻辑。

#### ② window 地址必须走 compat 层，禁止直接硬编码平台结构体偏移

`winContext`（`mc2Context`）按平台解释成 `HcclA3OpResParam` 或 `HcclA5OpResParam`，字段布局不同（A3 远端地址通过链表跳转，A5 状态区在前 1MB）。必须通过 compat 封装的 `GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` 等统一接口访问，不要在主流程中手写平台分支。**注意**：`HcclA3OpResParam`/`HcclA5OpResParam` 是样例内的**精简定义**，仅用于工程内自建 context，不适用于解析真实 HCCL 返回的 context（真实解析须用 SDK 完整 `HcclOpResParam`，见 `mte-address-access.md` 重要警告）。

#### ③ 共享 GM/状态区走 DataCopyPad，禁止 GetValue/SetValue 直接访问

共享 GM、`workspaceGM`、状态区、window 数据区的默认读写路径是 `DataCopyPad` 或等价 GM 路径。`SetValue()`/`GetValue()` 直接访问共享区域会导致跨核数据观察不稳定。`SyncAll` 只保证执行同步，**不保证** Data Cache 与 GM 一致性。极少数场景如必须使用 `SetValue`/`GetValue`，必须额外补齐 `DataCacheCleanAndInvalid` 并重新验证核间可见性。

#### ④ 状态协议：先数据后状态、每核只写自己槽位、消费后清理

- 发布顺序：先写 count/offset/附带信息，再发布 ready 状态
- 等待侧只轮询自己负责的状态段
- 每核只写自己的共享槽位，禁止多核直接累加同一地址
- 状态消费完成后才清理本核负责段，不要沿用单核整块 reset 习惯

**Reviewer 速查**：

| # | 检查项 | 方法 |
|---|--------|------|
| M1 | 无 HCCL 高阶通信原语 | `grep -rn "Hccl::AlltoAll\|Hccl::AlltoAllV\|Hccl::AllReduce\|Hccl::AllGather\|Hccl::ReduceScatter" operators/{op}/` 应为空 |
| M2 | window 地址走 compat 层 | kernel 中使用 compat helper（`GetBaseWindAddrByRankId` 等），不直接硬编码 `HcclA3OpResParam`/`HcclA5OpResParam` 字段偏移 |
| M3 | 共享 GM 走 DataCopyPad | `grep -rn "SetValue\|GetValue" operators/{op}/` 检查是否用于共享 GM/状态区，若有需确认有 `DataCacheCleanAndInvalid` |
| M4 | 状态协议正确 | 检查发布顺序（先数据后状态）、每核只写自己槽位、消费后清理 |

### References

**先判断任务类型**：当任务同时包含"先阅读现有实现"与"再进行改造"两部分时，路径顺序为：先完成入口 A 的阅读路径，再进入入口 B 的规格补齐。

#### 入口 A：阅读已有实现

适用场景：先建立现有 dispatch、combine 或 mega_moe / 融合实现的理解框架。

- 通用方法：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md)
- dispatch 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md)
- combine 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md)
- mega_moe / 融合实现：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md)

进入具体 guide 前，先判断是否属于融合实现（输入侧沿用 dispatch 风格、输出侧直接产出 combine 最终输出、主入口同时包含路由/发送/expert 计算/聚合/回写、通信协议不能拆成独立 dispatch+combine、关键内存布局同时承载多阶段共享状态）。命中任一特征时归入 mega-moe-reading-guide。

#### 入口 B：生成新算子或在样例工程基础上改造

适用场景：生成 moe dispatch、生成 moe combine，或在样例工程基础上修改 dispatch/combine。不再套用通用 `add.asc` 模板，默认拆成三部分推进：

1. **samples** / 规格补齐与样例工程：[`references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md) → `spec-template.md` → `dispatch-dataflow.md` 或 `combine-dataflow.md`；sample 内 helper 分层见 `sample-helper-map.md`
2. **api-rules** / MoE 特化 API 规则：[`references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md)（`DataCopyPad`、window 地址获取、同步与可见性、状态协议、dispatch/combine 接口契约）
3. **tiling-scheme** / MoE 特化分核方案：[`references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md) → `window-memory-layout.md` → `multi-core-formulas.md` → `split-core-design.md` → `double-buffer-protocol.md`

整体架构、阶段协作关系和设计动机背景见 [`references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md)，不替代三部分主路径中的规格、API 规则或分核设计文档。

#### 知识目录

| 目录 | 何时读 |
|------|--------|
| `references/foundations/ascendc-api/moe-dispatch-combine/reading/` | 阅读已有 dispatch/combine/mega_moe 实现时 |
| `references/foundations/ascendc-api/moe-dispatch-combine/samples/` | 规格补齐、工程组织参考、编译链路、接口语义和文件落点 |
| `references/foundations/ascendc-api/moe-dispatch-combine/api-rules/` | MoE 特有的 window 地址获取、DataCopyPad 规则、同步可见性、状态协议、接口契约 |
| `references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/` | window 物理布局、工作量公式、各阶段分核方案、双缓冲轮转协议 |
