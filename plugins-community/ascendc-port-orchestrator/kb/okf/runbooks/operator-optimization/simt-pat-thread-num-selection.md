---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "按算子类型选择编译时常量 THREAD_NUM"
description: "AscendC THREAD_NUM 是编译时常量：查找/插入类取 512，批量初始化/清空取 1024，导出类取 2048，按算子复杂度与并行度权衡选定。"
confidence: single_run
original_id: SIMT_PATTERNS.md#1-函数签名规范-THREAD_NUM-选择规则
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, thread-num, parallelism]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
AscendC 使用编译时常量 `THREAD_NUM`（进入 `LAUNCH_BOUND(THREAD_NUM)`）声明 kernel 线程数。与 source 在运行时动态决定 block size 不同，AscendC 需在编译期按算子复杂度选定：

| 算子类型 | THREAD_NUM | 依据 |
|----------|------------|------|
| 查找类（find / find_and_update） | 512 | 平衡并行度与资源使用 |
| 插入更新类（insert / upsert） | 512 | 需要原子操作，moderate 并行度 |
| 批量初始化 / 清空（clear / init） | 1024 | 简单操作，高并行度 |
| 导出类（dump） | 2048 | 高并行度，配合协作组处理 |

决策要点：算子逻辑越简单、越无原子竞争，越可以拉高 THREAD_NUM 换取并行度；含原子操作的查找/插入类保持 512，避免资源与竞争开销。选定值需在头文件内以 `constexpr uint32_t THREAD_NUM = ...;` 固定。
