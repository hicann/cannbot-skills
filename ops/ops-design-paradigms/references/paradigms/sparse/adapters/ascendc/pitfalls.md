<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式陷阱与错题集

> 蒸馏自 sparse 系算子的审查记录与复盘（sparse_bincount 复盘：14 项审查 FAIL；各 MDE 风险节）。按"现象 → 根因 → 规避"组织。

## 1 正确性类

### 1.1 零初始化竞争
- **现象**：散射累积输出读到非零随机初始值
- **根因**：假设框架保证输出清零；实际调用链无此契约（sparse_bincount 归档版即依赖 golden 侧 `np.zeros`，真实调用存在隐患）
- **规避**：默认 kernel 两阶段（零初始化 + `SyncAll()` + 散射）；host 清零路线必须验证契约后才可启用

### 1.2 SyncAll 不生效
- **现象**：加了 `SyncAll()` 仍然读到未初始化数据
- **根因**：tiling 侧漏配 `context->SetScheduleMode(1)`
- **规避**：SyncAll 与 SetScheduleMode(1) 成对出现，设计文档中作为一条检查项

### 1.3 原子 dtype 越界使用
- **现象**：double 直调 `asc_atomic_add` 编译失败或行为未定义
- **根因**：`CAP_SIMT_ATOMIC_ADD_DTYPE_SET` 不含 double
- **规避**：AtomicAddDispatch 分发；double 走 CAS 模拟（[atomic-ops.md](atomic-ops.md) §2）

### 1.4 half/bf16 原子返回值误用
- **现象**：用 `asc_atomic_add` 返回值做后续计算，结果随机错
- **根因**：half/bfloat16_t 的返回值不可靠（API 文档明确）
- **规避**：只用写语义；需要旧值逻辑改 CAS 或分桶

### 1.5 值依赖漏配
- **现象**：tiling 阶段 `GetInputTensor` 拿不到 shape/new_shape 数据
- **根因**：三处声明（op 定义 `ValueDepend` / infershape 注册 / tiling 注册）缺一
- **规避**：值依赖输入逐处核对清单（[../../../common/value-depend-process.md](../../../common/value-depend-process.md)）

### 1.6 int32 偏移溢出
- **现象**：大 shape 下索引越界或写错位置
- **根因**：`batch * sizeVal + val` 在 int32 域计算后溢出
- **规避**：索引中间量统一 `int64_t`（[index-mapping.md](index-mapping.md) §4）

## 2 设计一致性类

### 2.1 MDE–Code 设计偏差链
- **现象**：开发阶段擅自改 TilingKey 结构 / 拆分 VF 函数 / 改线程数，MDE 被动追认，触发多轮返工（sparse_bincount 复盘头号根因，浪费 ~1.6M tokens）
- **根因**：设计变更无审批闸门
- **规避**：TilingKey 结构、VF 函数划分、线程数分档属设计契约，变更须回写 MDE 并同步审查

### 2.2 唯一索引前提未核实
- **现象**：RowApply 按唯一索引设计直写，遇到重复索引静默丢数据
- **根因**：未核对竞品契约（TF SparseApply* 文档要求 indices 唯一）即当作前提
- **规避**：规格中记录唯一性的契约来源；无法证明时按重复索引处理

## 3 性能类

### 3.1 PER_CORE_MIN 取值错误
- **现象**：小 N 场景性能回退
- **根因**：PER_CORE_MIN 取 32（编组粒度）而非 1024，尾编组空转 + 全核调度开销（sparse_bincount 审查记录：R1 误取 32）
- **规避**：PER_CORE_MIN 对齐 32 且取 1024 量级；记入 tiling 审查项

### 3.2 线程数与负载错配
- **现象**：原子竞争场景吞吐低于预期 / 行级状态场景寄存器溢出
- **根因**：线程数未按负载分档（计算类 512 / 搬运类 2048）
- **规避**：按 [../../generic/splitting.md](../../generic/splitting.md) §4 分档；栈预算 64 B/线程 × 线程数核算

### 3.3 聚簇索引原子退化
- **现象**：重复索引聚簇分布时性能数倍劣于均匀分布
- **根因**：同地址原子串行化
- **规避**：性能规格声明索引分布；聚簇场景评估分桶预归约或 warp 内预归并

## 4 工程类

- **VF 参数超限**：入参 > 28 或传结构体/数组 → 编译失败；元数据改走 UB 共享内存
- **空输入核数**：N=0 时 `SetBlockDim(0)` 非法；下限 1 核（零初始化仍需执行）
- **TilingKey 编码 dtype**：用 TilingKey 位编码 dtype 组合导致实例爆炸；dtype 交 DTYPE_ 宏模板，TilingKey 只编码场景模式
