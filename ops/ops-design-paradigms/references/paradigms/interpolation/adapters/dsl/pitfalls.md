<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式陷阱与错题集（DSL）

> AscendC 侧错题集（[../ascendc/pitfalls.md](../ascendc/pitfalls.md)）的语义根因在 DSL 同样存在；本文件只列 DSL 特有形态 + DSL 侧新增的待验证风险。按"现象 → 根因 → 规避"组织。

## 1 正确性类（语义同源，形态不同）

### 1.1 FMA 收缩风险（形态未定）
- **现象**：AscendC 侧单表达式加权和收缩为 fma 链，恒定 1~2 ULP 偏差；DSL 侧生成路径未实测
- **根因**：代码生成器可能将 `w0*f0 + w1*f1 + w2*f2` 融合
- **规避**：分步中间绑定（[sampling-ops.md](sampling-ops.md) §4）；首例以恒定 ULP 用例实测后在 capability-mapping 登记结论

### 1.2 int64 索引读入未防御
- **现象**：int64 idx 负值/大值下越界
- **根因**：`idx[off]` 直读后未升宽未 clamp 即作下标
- **规避**：`dtypes.int64(...)` → 双向 clamp（[sampling-ops.md](sampling-ops.md) §2）

### 1.3 批次基址溢出
- **现象**：`b * ms * cs` 超 int32 写错地址
- **根因**：坐标变量用了 int32 中间量
- **规避**：坐标/偏移全程 `dtypes.int64`（AscendC 同款教训）

### 1.4 坐标映射次序错（同源）
- **现象**：边界列偏移 / lerp 末位不一致
- **根因**：`simt.fmax(·,0)` 施加在重基之后；lerp 用 clamp 后 f0
- **规避**：与 golden 逐句同序（[sampling-ops.md](sampling-ops.md) §3）；`simt.floor` 返回浮点值，转索引时再取整

## 2 DSL 特有

### 2.1 launch 形态混用
- **现象**：编译失败
- **根因**：`kernel[dim3, dim3]`（纯 SIMT）与 `kernel[block_count]` + `vf(thread=N)`（混合）混用
- **规避**：本范式默认纯 SIMT 形态；device 侧禁写死 `vf(mode="simt", thread=N)`

### 2.2 构图期读 GM
- **现象**：host 函数里试图读 idx/框坐标内容做分派，拿不到值
- **根因**：DSL host 是构图期执行，无运行期 GM 读取能力
- **规避**：值依赖走 AI CPU 或入口预解析（[kernel-entrance.md](kernel-entrance.md) §5）；kernel 内直读是正解

### 2.3 Constexpr 滥用导致变体爆炸
- **现象**：编译缓存膨胀、首编译时间失控
- **根因**：shape/extents 等运行期量设成 `Constexpr` 进缓存键
- **规避**：只有影响代码生成的量（对齐模式、路径开关）进 Constexpr； extents 走 params Tensor（对照 AscendC TilingKey 实例爆炸教训）

### 2.4 UB 预算误算
- **现象**：SIMT kernel 混用 SIMD 段时 UB 超限
- **根因**：`get_mem_size("ub")` 返回 248 KB 口径，开 SIMT 后实际 216 KB（Data Cache 划走 32 KB，不反映在接口）
- **规避**：SIMT 场景按 216 KB 手工留额（[capability-mapping.md](capability-mapping.md) §1）

## 3 性能类

### 3.1 整除/取模热点
- **现象**：大 totalOut 下坐标分解占比异常
- **根因**：`//`/`%` 生成路径的除法开销未登记（无 AscendC UintDiv 对等函数族）
- **规避**：热点算子实测；必要时 host 预计算 magic/shift 入 params Tensor 手工快除

### 3.2 线程档错配
- **现象**：1024 线程不升反降（同 AscendC 教训）
- **根因**：坐标映射 + 邻居索引寄存器压力；补偿路线 df32 再抬升
- **规避**：采样类 512 分档起评；补偿路线实测后定档

## 4 工程类

- **空输出**：`totalOut = 0` 时不 launch（`dim3(0)` 非法）——与 AscendC"下限 1 核短路"不同，DSL 直接不发起
- **散射原子 dtype 越界**：fp16/bf16 直调 `cb.simt.atomic_add` 不支持——升 fp32 或批量轨
- **缓存键漂移**：`lru_cache` 键漏掉影响代码生成的 Constexpr → 复用错实例；键设计对照 [kernel-entrance.md](kernel-entrance.md) §3
