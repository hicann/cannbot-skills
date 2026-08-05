# 硬件信息提取提示词 (V2)

> 用于指引 agent 从 Ascend 芯片手册 PDF 中完整提取算子开发所需的硬件信息。
> 覆盖范围从 V1 的"基本参数"扩展到"性能优化关键架构细节"。

## 源文件

PDF 位于 `/mnt/d/workspace/ai/a5/`。分卷说明:
- 分卷1: 芯片整体介绍和系统方案（芯片概览、Die 组成）
- **分卷2: REDACTED_INTERNAL_DOC 功能描述（AI Core 架构、SIMT/SIMD、MTE、UB、dcache — 最重要）**
- 分卷3: UnionDie 功能描述（HA、L2 Cache、互联 — 缓存和原子操作优化）
- 分卷4: MISC IO & Peri IP（外设，优先级低）
- 分卷6: BridgeDie（Die 间通信）

## 提取目标（按优先级）

### 第一优先级：SIMT 模式内存架构

这是最关键的信息，直接决定 SIMT kernel 的性能优化方向。

| 信息项 | 搜索关键词 | 用途 |
|--------|-----------|------|
| SIMT dcache 规格 | "dcache", "SIMT", "cacheline", "UB 预留" | 确定 SIMT 线程读 GM 的缓存行为 |
| dcache → L2 → HBM 数据通路 | "数据通路", "data path", "NOC", "HA" | 确定 SIMT 读延迟的分层 |
| L2 cache 大小和组织 | "L2", "bank", "set", "way", "cacheline" | 确定 L2 对重复 expert 读的缓存效果 |
| L2 alloc hint | "alloc hint", "victim hint", "cacheable" | 确定软件能否控制 L2 缓存策略 |
| HA coalescing 规则 | "Coalesce", "ReadOnce", "Reduce Atomic" | 确定同 cacheline 读/atomicAdd 并行能力 |
| dcache 大小配置方式 | "dcu", "sm_size", "SQE" | 确定 SIMT 如何分配 dcache vs shared memory |

### 第二优先级：SIMT/SIMD 混合模式

这是潜在的突破性优化方向。

| 信息项 | 搜索关键词 | 用途 |
|--------|-----------|------|
| VF 内 SIMT↔SIMD 切换 | "VF", "切换", "hybrid", "混合" | 确定能否在一个 kernel 内混用两种模式 |
| 切换时数据交换机制 | "UB 交换", "数据在 UB" | 确定 SIMD DataCopy 到 UB 的数据能否被 SIMT 线程访问 |
| 混合模式的约束 | "限制", "constraint", "不支持" | 确定混合模式的 overhead 和限制 |

### 第三优先级：VEC/MTE 管线细节

| 信息项 | 搜索关键词 | 用途 |
|--------|-----------|------|
| SIMT 模式下 VEC 的指令发射 | "in-order", "single-issue", "乱序", "多发射" | 确定 SIMT 指令级并行度 |
| MTE2/MTE3 管线宽度和延迟 | "MTE2", "MTE3", "Pipeline", "NDDMA" | 确定 DMA 搬运能力 |
| VEC 算力（SIMT vs SIMD） | "256B", "128B", "算力", "compute width" | 确定两种模式的计算吞吐比 |

### 第四优先级：原子操作和 shared memory

| 信息项 | 搜索关键词 | 用途 |
|--------|-----------|------|
| atomicAdd 在 L2 的行为 | "Reduce AtomicStore", "atomic", "L2 hit" | 确定 atomicAdd 是否受益于 L2 cache |
| shared memory 分配和大小 | "share memory", "sm_size", "SIMT" | 确定 SIMT shared memory 的实际可用量 |
| shared memory vs dcache 的关系 | "UB", "预留", "256KB" | 确定两者是否共享 UB 空间 |

### 第五优先级：补充参数

| 信息项 | 搜索关键词 | 用途 |
|--------|-----------|------|
| Die 间互联带宽 | "HCCS", "互联", "Die 间" | 跨 Die kernel 的通信开销 |
| HBM 控制器数量和交织 | "HBM", "Controller", "interleave" | 内存访问模式优化 |
| 寄存器文件详情 | "register file", "128KB", "寄存器" | 线程数 vs 寄存器数的 tradeoff |
| SIMT warp scheduler 细节 | "warp scheduler", "4 warp" | 延迟隐藏能力评估 |

## 输出要求

1. 每条信息标注**手册页码**（PDF 页码和文档内页码）
2. **原文引用**关键段落（不要意译，保留原文）
3. 对于图表，描述结构和关键数据点
4. 找不到的项标注"未在手册中找到，建议查看分卷X"
5. 发现手册与实测数据矛盾时，同时记录两者并标注差异

## 输出格式

写入 `src/skills/references/hardware/target/ascend950pr.md` 的对应章节。新增内容放在已有数据之后，标注"来源: 手册分卷X 页Y"。
