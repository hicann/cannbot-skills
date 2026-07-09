---
name: ascendc-mc2-best-practice
description: Ascend C MC2 通算融合算子（多卡通信+计算融合 Kernel 直调）开发最佳实践。仅支持 Ascend 950（npu-arch=dav-3510）。当用户需要开发 MC2 通算融合算子、多卡通信+计算融合的 Kernel 直调算子，或提及"MC2"、"SHMEM"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"多卡集合通信直调"时必须使用。强制约束：通信走 SHMEM（禁止 HCCL 高阶 API），Matmul 走 Blaze 模板（禁止 asc-devkit matmul API），开发必须按 CANNBot AGENTS.md Step 1→7 流程执行。
---

# Ascend C MC2 通算融合算子开发最佳实践

MC2（Multi-Card Compute-Communication Coupling）= 多卡间 UDMA/SHMEM 集合通信 + 单卡内 Blaze Cube 计算 + 通算两层流水掩盖通信开销。本 skill 是 CANNBot 工作流的技术参考，**不是独立开发入口**——识别到 MC2 需求后必须按父级 [`plugins-official/ops-direct-invoke/AGENTS.md`](../../plugins-official/ops-direct-invoke/AGENTS.md) Step 1→7 流程执行。

> **架构硬约束**：仅支持 **Ascend 950（dav-3510）**。其他架构（含 dav-2201）的 SHMEM/UDMA 行为未验证，禁止使用。

## 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的 Matmul/集合通信融合算子（AllToAll+Matmul、AllReduce+Matmul、多卡 EP/TP 融合 Kernel）
- 关键词："MC2"、"SHMEM"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"
- 代码：现有工程同时出现 `shmem.h`/`aclshmem*` API 与 `blaze/gemm/` 模板

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ `ascendc-blaze-best-practice`
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 非 3510 架构的通算融合

## 三大约束（红线）

Step 2 由 Architect 在 DESIGN.md 显式确认，Step 4 由 Reviewer 交叉检查；违反任意一条 = FAIL。

### ① 通信走 SHMEM，禁止 HCCL 高阶 API

通信侧统一用 SHMEM（host 侧 `aclshmem_*`，device 侧 `aclshmemx_udma_*` + `aclshmemx_barrier_all_vec`）。来自 `asc-devkit/adv_api/hccl/` 的以下 API **全部禁止**：

| 类别 | API |
|------|-----|
| 初始化/终结 | `Hccl::Init()` / `Hccl::InitV2()` / `Hccl::Finalize()` |
| 任务调度 | `Hccl::Commit()` / `Hccl::Wait()` / `Hccl::Query()` / `Hccl::Iterate()` |
| 集合通信原语 | `Hccl::AllReduce()` / `Hccl::AllGather()` / `Hccl::ReduceScatter()` / `Hccl::AlltoAll()` / `Hccl::AlltoAllV()` |
| 写操作 | `Hccl::BatchWrite()` / `Hccl::AlltoAllvWrite()` |
| Tiling | `Hccl::SetCcTiling()` / `Hccl::SetCcTilingV2()` |
| 跨组同步 | `Hccl::InterHcclGroupSync()` |
| Context | `GetHcclContext<>()` |

**理由**：HCCL 由服务端调度，Kernel 无法在通信进行中插入计算指令，通算只能串行。MC2 要求通信下发与计算下发在同一 Kernel 内通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 精细同步，只有 SHMEM/UDMA 能做到。SHMEM 文档：<https://shmem-doc.pages.dev/>。

### ② Matmul 走 Blaze 模板，禁止 asc-devkit matmul API

计算侧统一用 Blaze（`Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`），与 `ascendc-blaze-best-practice` 共享基底。`AscendC::Matmul` 等 asc-devkit 黑盒 API 无法接入通算流水，禁用。

### ③ 禁止跳过 CANNBot 7 步流程

```
Step 1 环境检查（含 NPU 架构校验=3510）→ environment.json all_passed=true
Step 2 设计（Architect）→ DESIGN.md + PLAN.md（含三大约束确认）
Step 2.5 设计串讲 → WALKTHROUGH.md
Step 3 开发（Developer）→ 复用 references/all_to_all_matmul/ 基底
Step 4 审查（Reviewer）→ REVIEW.md（用下方 R1~R7 速查）
Step 5 修复循环（最多 3 轮）— 仅 FAIL 时
Step 6 性能验收（含 L2 cache flush + msprof task-based 采集）
Step 7 完成汇报
```

**禁止**：跳过环境检查直接写代码；Step 2 未在 DESIGN.md 显式确认三大约束；Step 3 之前生成任何 kernel 代码；Step 6 性能采集时未刷 L2 cache（前一轮热度会污染本轮指标）。

## Reviewer 速查（Step 4 必查）

| # | 检查项 | 方法（应为空/匹配） |
|---|--------|------|
| R1 | 架构=3510 | `grep "npu-arch" CMakeLists.txt` → `dav-3510` |
| R2 | 无 HCCL 高阶 API | `grep -rn "Hccl::" operators/{op}/` 应为空 |
| R3 | 无 asc-devkit matmul | `grep -rn "AscendC::Matmul\b" operators/{op}/` 应为空 |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmemx_barrier_all_vec` |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` |
| R6 | 流程门禁完整 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；`environment.json` all_passed=true |
| R7 | L2 flush 证据 | src 中含 `heavy_add_kernel` 调用或等效 L2 flush 实现 |

## References（按需展开）

| 文档 | 何时读 |
|------|--------|
| [`references/workflow_integration.md`](references/workflow_integration.md) | 进入任意 Step 前，看 MC2 场景的具体动作和门禁 |
| [`references/mc2_architecture.md`](references/mc2_architecture.md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/comm_shmem.md`](references/comm_shmem.md) | 写/改通信层，查 SHMEM API、UDMA 用法、扩展其他通信原语 |
| [`references/matmul_blaze.md`](references/matmul_blaze.md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/profiling_mc2.md`](references/profiling_mc2.md) | Step 6 性能验收：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/pipeline_tuning.md`](references/pipeline_tuning.md) | Step 2-4 用 tileCnt=1 做串行基线；Step 6 扫描 tileCnt 找通算并行最优值 |
| [`references/codebase_map.md`](references/codebase_map.md) | Step 3 复制基底工程后，定位"哪些文件改/不改"与改造食谱 |
| `references/all_to_all_matmul/` | 编译验证过的基底工程，所有 MC2 算子的起手模板 |
