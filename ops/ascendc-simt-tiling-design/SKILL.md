---
name: ascendc-simt-tiling-design
description: AscendC SIMT 算子切分设计指南。提供 SIMT 算子独有的核数切分、线程数设置、DCache/UB空间分配方法。SIMT切分与SIMD完全不同（线程级并行 vs 向量级UB切分），本skill聚焦SIMT切分范式。触发：设计SIMT算子Tiling策略、设置SIMT线程数、规划SIMT核数切分时。
---

# SIMT 算子切分设计指南

SIMT 算子的切分设计与 SIMD 有本质差异。SIMD 以 UB 切分 + Buffer 规划为核心，SIMT 以核数切分 + 线程数设置为核心，不涉及 UB 切分和 Buffer 规划。

> **SIMD 算子 Tiling 设计**请参考 `ascendc-tiling-design` skill。

## SIMT vs SIMD 切分差异速查

| 要素 | SIMD | SIMT |
|------|------|------|
| 多核切分 | 按 UB 单次处理量切分 | 按元素总量切分（ceil(总量/单核最少元素数)） |
| 单核并行 | UB Buffer + 向量指令 | 线程数（constexpr 编译期常量） |
| 数据搬运 | 需显式 Load/Store | 支持直接读写 GM |
| UB 使用 | 全量使用 | 仅核内共享场景使用，DCache >= 32KB |
| Buffer 规划 | inQueue/outQueue/tmpBuf | TBuf（仅在需要共享内存时使用） |

详细设计指南：[references/guide.md](references/guide.md)

## 核心设计要素

### 1. 核数切分策略

- 总核数 = ceil(输出元素总数 / 单核最少处理元素数)
- 单核最少处理元素数建议 1024，需对 warp(32) 对齐
- 通过 tiling 侧 `SetBlockDim` 设置核数

### 2. 线程数设置策略

- 默认 1024，最大 2048，必须是 `constexpr` 编译期常量
- `LAUNCH_BOUND(N)` 与 `Simt::Dim3(N)` 必须使用同一个常量
- 禁止从 tiling 数据动态获取线程数

### 3. DCache 与 UB 空间分配

- SIMT 算子 DCache 必须 >= 32KB
- 可用 UB = 256KB - 8KB - 32KB = 216KB
- tiling 侧通过 `SetLocalMemorySize` 设置