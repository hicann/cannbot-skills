---
name: ascendc-mc2-best-practice
description: Ascend C 通算融合算子（多卡通信+计算融合 Kernel 直调）开发最佳实践。支持两种通信范式——SHMEM/UDMA（AllToAll+Matmul、AllReduce+Matmul、TP/SP 通算融合）和 MTE window（MoE Dispatch、MoE Combine、专家并行 EP、token 路由分发、mega_moe）。当用户提及 MC2、SHMEM、通算融合、多卡通信直调、UDMA、URMA、AllToAll+Matmul、MoE Dispatch、MoE Combine、专家并行、EP、mega_moe、MTE window、token 路由分发、多卡集合通信直调时必须使用。
---

# Ascend C 通算融合算子开发最佳实践

本 skill 覆盖多卡通信+计算融合的 Kernel 直调算子开发，支持两种正交通信范式：

- **SHMEM/UDMA 路径**：大块跨卡 Put + Blaze Cube 计算流水，适用于计算密集型通算融合（AllToAll+Matmul、AllReduce+Matmul、TP/SP 融合 Kernel）
- **MTE window 路径**：细粒度 window 地址寻址 + 状态位协议，适用于路由通信密集型场景（MoE Dispatch/Combine、专家并行 EP）

两种范式正交互补：SHMEM 适合 TP/SP 等计算密集型并行，MTE window 适合 EP 等路由通信密集型并行。

## 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的通算融合算子（AllToAll+Matmul、AllReduce+Matmul、MoE Dispatch/Combine、多卡 EP/TP/SP 融合 Kernel）
- 关键词："MC2"、"SHMEM"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE window"、"token 路由分发"
- 代码：现有工程同时出现 `shmem.h`/`aclshmem*` API 与 `blaze/gemm/` 模板（SHMEM 路径），或出现 `winContext`/`mc2Context`/`HcclOpParam` 与 MTE 通信窗口结构（MTE 路径）

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ `ascendc-blaze-best-practice`
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 通用 Ascend C API 用法查询 → `ascendc-api-best-practices`

## 决策树：先判断算子类型

```
用户要做通算融合算子
│
├─ AllToAll+Matmul / AllReduce+Matmul / TP/SP 通算融合？
│   → SHMEM/UDMA 路径（下方§1）
│
├─ MoE Dispatch / MoE Combine / mega_moe / 专家并行 EP？
│   → MTE window 路径（下方§2）
│      ├─ 先阅读已有实现？→ references/moe-dispatch-combine/reading/
│      └─ 生成新算子或改造？→ references/moe-dispatch-combine/samples/
│
└─ 不确定？
    → 看通信机制：SHMEM API = SHMEM 路径；winContext/mc2Context = MTE 路径
```

---

## §1 SHMEM/UDMA 路径

**架构约束**：仅支持 Ascend 950（dav-3510）。其他架构的 SHMEM/UDMA 行为未验证。

### 两大约束（红线）

#### ① 通信走 SHMEM，禁止 HCCL 高阶 API

通信侧统一用 SHMEM（host 侧 `aclshmem_*`，device 侧 `aclshmemx_udma_*` + `aclshmem_barrier_all`）。以下 HCCL API 出现在算子代码中即视为违反：

| 类别 | API |
|------|-----|
| 初始化/终结 | `Hccl::Init()` / `Hccl::InitV2()` / `Hccl::Finalize()` |
| 任务调度 | `Hccl::Commit()` / `Hccl::Wait()` / `Hccl::Query()` / `Hccl::Iterate()` |
| 集合通信原语 | `Hccl::AllReduce()` / `Hccl::AllGather()` / `Hccl::ReduceScatter()` / `Hccl::AlltoAll()` / `Hccl::AlltoAllV()` |
| 写操作 | `Hccl::BatchWrite()` / `Hccl::AlltoAllvWrite()` |
| Tiling | `Hccl::SetCcTiling()` / `Hccl::SetCcTilingV2()` |
| 跨组同步 | `Hccl::InterHcclGroupSync()` |
| Context | `GetHcclContext<>()` |

**理由**：asc-devkit 官方通算融合路径（HCCL 高阶 API + Matmul 高阶 API）仅支持单算子 API 调用，不支持 Kernel 直调（见 asc-devkit 官方文档《通算融合》章节）。在 Kernel 直调工程中，Hccl 高阶 API 依赖框架注入的 `GetHcclContext` 上下文，直调场景拿不到该上下文，因此无法使用。SHMEM/UDMA 能在同一 Kernel 内通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 精细同步，是直调场景下自建通信的可行路径。SHMEM 接口来源与用法见 `gitcode.com/cann/shmem`（v1.5.0，本 skill `comm_shmem.md` 已逐接口核对头文件位置），文档：<https://shmem-doc.pages.dev/>。

#### ② Matmul 走 Blaze 模板，禁止 asc-devkit matmul API

计算侧统一用 Blaze（`Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`）。asc-devkit 的 `AscendC::Matmul` 高阶 API 属于官方通算融合（单算子 API 调用）路径，**不支持 Kernel 直调场景**（与上一条 HCCL 同理，依赖框架注入上下文），故直调工程中不用；下述 R3 以 `AscendC::Matmul\b` 作为"误用官方高阶 API"的检查项。

### Reviewer 速查（SHMEM 路径）

| # | 检查项 | 方法（应为空/匹配） |
|---|--------|------|
| R1 | 架构=3510 | `grep "npu-arch" CMakeLists.txt` → `dav-3510` |
| R2 | 无 HCCL 高阶 API | `grep -rn "Hccl::" operators/{op}/` 应为空 |
| R3 | 无 asc-devkit matmul | `grep -rn "AscendC::Matmul\b" operators/{op}/` 应为空 |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmem_barrier_all` |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` |
| R6 | L2 flush 证据 | 性能验收时 src 中含 `heavy_add_kernel` 调用或等效 L2 flush 实现 |

### References（SHMEM 路径）

| 文档 | 何时读 |
|------|--------|
| [`references/workflow_integration.md`](references/workflow_integration.md) | 进入开发前，看 MC2 场景的具体动作和门禁 |
| [`references/mc2_architecture.md`](references/mc2_architecture.md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/comm_shmem.md`](references/comm_shmem.md) | 写/改通信层，查 SHMEM API、UDMA 用法、扩展其他通信原语 |
| [`references/matmul_blaze.md`](references/matmul_blaze.md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/profiling_mc2.md`](references/profiling_mc2.md) | 性能验收：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/pipeline_tuning.md`](references/pipeline_tuning.md) | 用 tileCnt=1 做串行基线；扫描 tileCnt 找通算并行最优值 |
| [`references/codebase_map.md`](references/codebase_map.md) | 复制基底工程后，定位"哪些文件改/不改"与改造食谱 |
| `references/all_to_all_matmul/` | 编译验证过的基底工程，所有 SHMEM 路径算子的起手模板 |

---

## §2 MTE window 路径（MoE Dispatch/Combine）

**架构约束**：支持 A3（dav-2201）+ A5（Ascend 950/dav-3510）双平台。两个平台的 window 地址结构不同，compat 层统一封装访问方式。

### 四大约束（红线）

#### ① 通信走 MTE window，禁止 HCCL 高阶通信原语做数据搬运

MoE Dispatch/Combine 通过 host 侧 `HcclAllocComResourceByTiling` 创建通信 window 资源，device 侧用 MTE + `DataCopyPad` 访问 window 地址做跨卡数据搬运。禁止用 `Hccl::AlltoAll()`/`Hccl::AlltoAllV()`/`Hccl::AllReduce()` 等高阶通信原语替代——它们是对整块数据的黑盒集合通信，无法在 Kernel 内按 token 粒度插入路由计算和状态协议逻辑（且与约束 ① 同理，官方通算融合不支持 Kernel 直调）。

#### ② window 地址必须走 compat 层，禁止直接硬编码平台结构体偏移

`winContext`（`mc2Context`）按平台解释成 `HcclA3OpResParam` 或 `HcclA5OpResParam`，字段布局不同（A3 远端地址通过链表跳转，A5 状态区在前 1MB）。必须通过 compat 封装的 `GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` 等统一接口访问，不要在主流程中手写平台分支。**注意**：`HcclA3OpResParam`/`HcclA5OpResParam` 是样例内的**精简定义**，仅用于工程内自建 context，不适用于解析真实 HCCL 返回的 context（真实解析须用 SDK 完整 `HcclOpResParam`，见 `mte-address-access.md` 重要警告）。

#### ③ 共享 GM/状态区走 DataCopyPad，禁止 GetValue/SetValue 直接访问

共享 GM、`workspaceGM`、状态区、window 数据区的默认读写路径是 `DataCopyPad` 或等价 GM 路径。`SetValue()`/`GetValue()` 直接访问共享区域会导致跨核数据观察不稳定。`SyncAll` 只保证执行同步，**不保证** Data Cache 与 GM 一致性。极少数场景如必须使用 `SetValue`/`GetValue`，必须额外补齐 `DataCacheCleanAndInvalid` 并重新验证核间可见性。

#### ④ 状态协议：先数据后状态、每核只写自己槽位、消费后清理

- 发布顺序：先写 count/offset/附带信息，再发布 ready 状态
- 等待侧只轮询自己负责的状态段
- 每核只写自己的共享槽位，禁止多核直接累加同一地址
- 状态消费完成后才清理本核负责段，不要沿用单核整块 reset 习惯

### 跨路径设计原则（MTE 路径）

- 阅读或改造 `dispatch`、`combine` 任一侧时，都要同步核对另一侧的接口和状态语义
- `expandIdx`、`epRecvCounts`、`epSendCounts` 等中间量必须成对理解，不要单独局部解释字段
- host 侧只传总核数；kernel 侧负责决定每个阶段实际使用多少核以及如何切分工作项

### 先判断任务类型

当任务同时包含"先阅读现有实现"与"再进行改造"两部分时，路径顺序为：先完成入口 A 的阅读路径，再进入入口 B 的规格补齐。

#### 入口 A：阅读已有实现

适用场景：先建立现有 dispatch、combine 或 mega_moe / 融合实现的理解框架。

- 通用方法：[`references/moe-dispatch-combine/reading/common-reading-method.md`](references/moe-dispatch-combine/reading/common-reading-method.md)
- dispatch 专用：[`references/moe-dispatch-combine/reading/dispatch-reading-guide.md`](references/moe-dispatch-combine/reading/dispatch-reading-guide.md)
- combine 专用：[`references/moe-dispatch-combine/reading/combine-reading-guide.md`](references/moe-dispatch-combine/reading/combine-reading-guide.md)
- mega_moe / 融合实现：[`references/moe-dispatch-combine/reading/mega-moe-reading-guide.md`](references/moe-dispatch-combine/reading/mega-moe-reading-guide.md)

进入具体 guide 前，先判断是否属于融合实现（输入侧沿用 dispatch 风格、输出侧直接产出 combine 最终输出、主入口同时包含路由/发送/expert 计算/聚合/回写、通信协议不能拆成独立 dispatch+combine、关键内存布局同时承载多阶段共享状态）。命中任一特征时归入 mega-moe-reading-guide。

#### 入口 B：生成新算子或在样例工程基础上改造

适用场景：生成 moe dispatch、生成 moe combine，或在样例工程基础上修改 dispatch/combine。

不再套用通用 `add.asc` 模板。算子生成默认拆成三部分推进：

1. **samples** / 规格补齐与样例工程：[`references/moe-dispatch-combine/samples/index.md`](references/moe-dispatch-combine/samples/index.md) → `spec-template.md` → `dispatch-dataflow.md` 或 `combine-dataflow.md`；sample 内 helper 分层见 `sample-helper-map.md`
2. **api-rules** / MoE 特化 API 规则：[`references/moe-dispatch-combine/api-rules/index.md`](references/moe-dispatch-combine/api-rules/index.md)（`DataCopyPad`、window 地址获取、同步与可见性、状态协议、dispatch/combine 接口契约）
3. **tiling-scheme** / MoE 特化分核方案：[`references/moe-dispatch-combine/tiling-scheme/index.md`](references/moe-dispatch-combine/tiling-scheme/index.md) → `window-memory-layout.md` → `multi-core-formulas.md` → `split-core-design.md` → `double-buffer-protocol.md`

整体架构、阶段协作关系和设计动机背景见 [`references/moe-dispatch-combine/reading/design-overview.md`](references/moe-dispatch-combine/reading/design-overview.md)，不替代三部分主路径中的规格、API 规则或分核设计文档。

### Reviewer 速查（MTE 路径）

| # | 检查项 | 方法 |
|---|--------|------|
| M1 | 无 HCCL 高阶通信原语 | `grep -rn "Hccl::AlltoAll\|Hccl::AlltoAllV\|Hccl::AllReduce\|Hccl::AllGather\|Hccl::ReduceScatter" operators/{op}/` 应为空 |
| M2 | window 地址走 compat 层 | kernel 中使用 compat helper（`GetBaseWindAddrByRankId` 等），不直接硬编码 `HcclA3OpResParam`/`HcclA5OpResParam` 字段偏移 |
| M3 | 共享 GM 走 DataCopyPad | `grep -rn "SetValue\|GetValue" operators/{op}/` 检查是否用于共享 GM/状态区，若有需确认有 `DataCacheCleanAndInvalid` |
| M4 | 状态协议正确 | 检查发布顺序（先数据后状态）、每核只写自己槽位、消费后清理 |

### References（MTE 路径）

| 目录 | 何时读 |
|------|--------|
| `references/moe-dispatch-combine/reading/` | 阅读已有 dispatch/combine/mega_moe 实现时 |
| `references/moe-dispatch-combine/samples/` | 规格补齐、工程组织参考、编译链路、接口语义和文件落点 |
| `references/moe-dispatch-combine/api-rules/` | MoE 特有的 window 地址获取、DataCopyPad 规则、同步可见性、状态协议、接口契约 |
| `references/moe-dispatch-combine/tiling-scheme/` | window 物理布局、工作量公式、各阶段分核方案、双缓冲轮转协议 |
