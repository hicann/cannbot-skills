<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式原子操作落地（DSL）

> CANNBotDSL 原子操作在散射累积场景的落地。冲突判定与决策树见 [../../generic/access-model.md](../../generic/access-model.md) §2/§3；dtype 真值见 [capability-mapping.md](capability-mapping.md) §1。

## 1 双轨原子：标量点加与批量随路累积

DSL 提供**两条互不等价**的原子轨，dtype 集合不同（与 AscendC 单轨差异最大的点）：

| 轨 | 接口 | dtype 集 | 适用 |
|----|------|---------|------|
| 标量点加 | `cb.simt.atomic_add(ptr, value)`（SIMT VF 内，`ptr = tensor.ptr(offset)`） | `{int32, uint32, float32, int64, uint64}`——**无 fp16/bf16/double** | 离散散射点加（scatter_elements `_scatter_direct_add_simt` 先例） |
| 批量随路 | `mem_copy(dst_gm, src_ub, atomic_add=True)` | `{int8, int16, int32, bf16, fp16, fp32}`，仅 UB/L0C → GM | 片上整块累积写回（部分和先聚 UB 再一次搬出） |

选型流程（接 access-model §3 决策树）：

```
目标 dtype ∈ 标量集？ ── 是 ── 离散写？ ── 是 ── cb.simt.atomic_add
   │                              └─ 否（整块）── mem_copy(atomic_add=True)
   └─ fp16/bf16 → 整块：mem_copy(atomic_add=True)；离散：fp32 累积域 + 出口 cast，或属主写去原子
       double  → CAS 模拟（待验证）或分桶/属主路线
```

## 2 分路径分发模式

与 AscendC `AtomicAddDispatch` 同构的隔离思想，DSL 落地为**host 侧路径选择**（dtype/语义决定路径 → 独立编译实例），kernel 主逻辑只写所选路径：

```python
# host（构图期）：按 dtype/唯一性/形状特征选路径（scatter_elements 先例）
if reduction == "add" and dtype == torch.float32:
    path = "direct_add"          # 标量原子
elif has_flat_axis_coordinates and unique:
    path = "compact_unique_add"  # RMW 直写（唯一索引，无原子）
...
```

- fp16/bf16 散射累积：`"fp32_accum_cast"` 型路径（散射进 fp32 中间张量 → 出口 `vcast`/SIMT cast），或属主写
- int64 点加原生支持（AscendC 需 `asc_atomic_add_int64` 专用接口，DSL 无此差异）

## 3 double 的 CAS 模拟（待验证路线）

`cb.simt.atomic_cas` 类目存在；**位宽集合未核对**（[capability-mapping.md](capability-mapping.md) 待验证项）。可用时的形态与 AscendC 同构：

```python
# SIMT 内：u64 位级 IEEE 加法 + CAS 循环（核上禁 double 算术，位操作承载）
old = cb.simt.atomic_cas(...)        # 假设 u64 支持——先验证后使用
```

就位前 double 路径首选：**分桶预归约**（部分和聚 UB → 单点直写）或**输出属主序扫描**（每目标唯一属主线程遍历源，scatter_elements `_scatter_axis0_owner_add_simt` 先例——完全免原子）。

## 4 幂等直写替代原子

输出语义为幂等值时**消除原子**（sparse_bincount 的 binary_output 同款语义）：

```python
with cb.vf(mode="simt"):
    ...
    if inside and hit:
        output[off] = one             # 直写，全核同值幂等（signal-model I3 同款论证）
```

## 5 使用要点

| 要点 | 说明 |
|------|------|
| 返回值 | `atomic_add` 声明返回旧值（`-> ScalarValue`），但可靠性未实测登记——**当前只用写语义**；需旧值逻辑改 CAS 或属主写 |
| 竞争代价 | 同地址高频竞争吞吐下降；聚簇索引优先分桶预归约/属主写 |
| 确定性 | 原子顺序非确定 → 浮点结果与竞品容差验证（同 AscendC 口径），整型精确 |
| 平台边界 | 标量原子仅 950PR/DT 支持（API 产品支持表）；A2/A3 等不支持——跨平台设计须声明 |
| 验证 | 重复索引用例必须覆盖（验证和守恒）；双轨混用路径单独验证可见性顺序（待验证项） |
