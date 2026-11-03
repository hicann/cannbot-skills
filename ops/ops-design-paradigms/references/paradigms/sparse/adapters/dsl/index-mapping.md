<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式索引数学实现（DSL）

> 线性化/反线性化、-1 维推断与 stride 表传递的 CANNBotDSL 落地。算法语义见 [../../generic/access-model.md](../../generic/access-model.md) §6；基准：`samples/scatter_elements/scatter_elements.py`、`samples/gather_v2/gather_v2.py`。

## 1 线性化 / 反线性化（SIMT 设备侧）

每条目独立变换、无冲突直写，与 AscendC 同构；差异在类型承载与分支纪律：

```python
with cb.vf(mode="simt"):
    tid = cb.simt.cast(cb.simt.thread_idx()[0], dtype=dtypes.int64)
    blk = cb.simt.cast(cb.simt.block_idx()[0], dtype=dtypes.int64)
    dim = cb.simt.cast(cb.simt.thread_dim()[0], dtype=dtypes.int64)
    grid = cb.simt.cast(cb.simt.grid_dim()[0], dtype=dtypes.int64)
    for flat in range(blk * dim + tid, total, grid * dim):   # grid-stride 必备
        # 线性化：linear = Σ coord[d] * stride_in[d]（int64 全程）
        # 反线性化：coord[d] = linear // stride_out[d]; linear %= stride_out[d]
```

- **索引中间量统一 int64**：`cb.simt.cast(..., dtype=dtypes.int64)` 后再做乘加（scatter_elements 全部 kernel 先例）；int32 在 `batch * size + val` 类偏移中会溢出
- **除法/取模**：SIMT 内直接写 `//` 与 `%`，编译器落硬件除法；热点内层可 host 预计算魔数+移位传入（`CAP_FAST_DIV` 类目见 [../../../random_sampling/adapters/dsl/prng-implementation.md](../../../random_sampling/adapters/dsl/prng-implementation.md) §4，同法适用）
- **VF 内分支纪律**：设备 `if` 合法（SIMT 每线程可走不同分支）；但**若该循环需降成 Hardware Loop，体内运行时 `if` 会静默阻断**——见 [pitfalls.md](pitfalls.md) §2.2

## 2 -1 维推断（值依赖——DSL 特有处理）

推断依赖 `shape` / `new_shape` 的**值**。DSL host 是**构图期**代码，不能读运行期 GM 值（与 AscendC tiling 的 `GetData` 能力关键差异）：

| 路线 | 适用 | 形态 |
|------|------|------|
| AI CPU 推断 | 输出 shape 依赖输入值（sparse_reshape 型） | `@aicpu_kernel` 运行期读 GM 算出 y_shape 与 stride 表 → 写 GM metadata；host 据此（或调用方直接）分配输出。分核计划同场产出（cannbot-dsl aicpu.md「依赖张量内容的分核计划 → AI CPU」） |
| 调用方预解析 | 公开 Python 入口可拿到真实 shape 时 | 入口侧（构图前）解析 -1 维、分配输出——设备值计算仍全在 kernel/AI CPU |
| 设备侧自算 | 推断仅影响映射、不影响输出分配 | kernel 内每条目自算（免 host 推断），-1 语义编码进 params |

值依赖三处声明（ValueDepend/注册块）是 AscendC 工程概念，DSL 无对应物；DSL 侧的等价义务是**声明"构图期不可读值"清单**并给出上述三路线之一。

## 3 stride / shape 表的传递（params Tensor ABI）

AscendC 的"UB 共享内存传 stride 表"在 DSL 的惯用对等物是 **params Tensor**（GM int64 数组 + 手工 ABI）：

```python
# host 侧（构图期纯 Python）：算好打包
MAX_RANK = 8
tables = data_strides + index_strides + update_strides + indices_shape + (rank, axis, ...)
params = torch.tensor(tables, dtype=torch.int64, device=data.device)

# kernel 侧：按下标常量读（编译期常量，两侧共享定义）
INDEX_STRIDES_BASE, AXIS_PARAM, ... = 2*MAX_RANK, 4*MAX_RANK+1, ...
axis = params[AXIS_PARAM]
```

- **ABI 契约两侧手工约定**：字段下标常量定义在一处、host 与 kernel 共享（scatter_elements `stride_parameters` 属性 + kernel 侧下标常量组先例；aicpu.md 同款纪律）
- 读取走标量流水，**只适合少量元数据**（几十个 int64 以内）；大表拆分或改设计
- 表体积超 MAX_RANK 口径时补零对齐，固定 ABI 宽度（`tables.extend([0] * (MAX_RANK - len(...)))`）

## 4 溢出与类型防护

| 防护 | DSL 写法 |
|------|---------|
| int64 承载 | 读出即 `cb.simt.cast`/`vreinterpret` 到 int64；比较与乘加全程 int64 |
| 越界检查先行 | `if 0 <= val < axis_size:` 防负数取模与越界写（scatter_elements 全部路径先例） |
| 运行期值界限 | `params[...]` 读出的是运行期值——只能用于 `range()` 边界、地址计算、设备分支；**不得用于 `range_constexpr` / `const_expr`**（aicpu.md 消费纪律同款） |
| 空输入 | `numel=0` 时不 launch（或 1 核空转幂等）；`dim3(0)` 非法 |

## 5 行级偏移（RowApply / ScatterAdd）

```
行基址   = idx * innerSize
输出偏移 = indices[j] * innerSize + k        # k ∈ [0, innerSize)
```

- 行内 `k` 循环由同一线程顺序执行（行内状态依赖）；行间完全独立
- 大 `innerSize` 行内循环是性能敏感点；行级局部变量体积受每线程寄存器/栈预算约束（[capability-mapping.md](capability-mapping.md) 待验证项）
- 唯一索引 RowApply 直写、重复索引走原子/属主写（owner-scan）——scatter_elements 的 `_scatter_unique_add_simt`（无原子 RMW）与 `_scatter_axis0_owner_add_simt`（输出属主序扫描）是两种去原子先例
