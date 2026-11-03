<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 Kernel 入口与编译组织（DSL）

> CANNBotDSL 的入口结构、params ABI、编译实例与 launch 要点。基准：`samples/scatter_elements/scatter_elements.py`（host 分派 + AOT 缓存完整先例）。

## 1 三层结构（对 AscendC 三段式的映射）

| AscendC（adapters/ascendc/kernel-entrance.md） | DSL 对等物 |
|------|------|
| `__global__` 入口 + `GET_TILING_DATA_WITH_STRUCT` | `@kernel` 函数/类：Tensor 形参 + `cb.Constexpr[int]` 编译期配置；tiling 载体无（见 §2） |
| TilingData（host 可算标量打包） | **params Tensor**（GM int64 数组，手工 ABI）——运行期元数据；`Constexpr`——编译期配置 |
| TilingKey 分派（场景模式编码） | host 侧 Python 路径选择 + **每路径独立编译实例**（scatter_elements：`"generic"/"generic_none"/"wide_reduce"/"direct_none"/"direct_add"/...` 路径表） |
| tiling 侧校验（`OP_CHECK_IF`） | 公开入口/构建函数侧纯 Python 校验（shape/rank/dtype/axis，scatter_elements `ScatterElementsTiling` 先例） |
| `asc_vf_call`（VF 实例化 + 线程常量） | `@kernel` 内 `with cb.vf(mode="simt")` 作用域；线程数由 launch（纯 SIMT）或 VF 参数（混合）给定 |

## 2 kernel 骨架（scatter_elements 实例化）

```python
@kernel
def _scatter_direct_add_simt(indices: Tensor, updates: Tensor,
                             output: Tensor, params: Tensor):
    with cb.vf(mode="simt"):                       # SIMT VF 作用域
        tid  = cb.simt.cast(cb.simt.thread_idx()[0], dtype=dtypes.int64)
        ...                                        # grid-stride 主循环
        cb.simt.atomic_add(output.ptr(off), updates[flat])

@host
def _launch_scatter_direct_add(data, indices, updates, output, params,
                               copy_cores: cb.Constexpr[int],
                               scatter_cores: cb.Constexpr[int]) -> None:
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_direct_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params)
```

- **方括号 launch 必须直接位于 `@host` 函数体**；`@kernel` 不能作为普通 Python 函数调用
- `@host` 由公开 Python 入口调用，输出 Tensor 经形参传入（返回 None）
- Kernel 类形态（`@kernel class` + `__init__` 绑定编译期几何 + `__call__`）适合静态 tile 计划（DMA copy kernel 先例）

## 3 编译实例管理（AOT 缓存）

```python
@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_scatter(path, data_numel, indices_numel, ..., axis_size=0, ...):
    descriptors = (
        cb.TensorSpec((data_numel,), value_dtype),        # 秩 1 裸视图口径
        ...
        cb.TensorSpec((DEVICE_PARAM_COUNT,), dtypes.int64),
    )
    return cb.compile(launch, *descriptors, copy_cores, ...)   # 按路径选 launch
```

- **缓存键 =（路径, shape, dtype, 核数, 全部 Constexpr）**——`Constexpr` 每加一项都放大变体数，只把"影响代码生成"的量设为 Constexpr
- 逻辑 shape 校验/stride 推导（`ScatterElementsTiling`）在构图期纯 Python 完成，产物打包进 params Tensor；`cb.TensorSpec` 描述真实 Tensor 元信息（不保存数据）
- 显式编译产出的程序由持有方管理（用毕 `close()`/上下文管理器）；`lru_cache` 缓存的是编译产物，不是程序句柄生命周期

## 4 两阶段骨架（零初始化 → 散射）的 DSL 惯用法

通用层 [sync-model](../../generic/sync-model.md) §1 的两阶段骨架有两条 DSL 落地路线：

| 路线 | 写法 | 适用 |
|------|------|------|
| **顺序双 launch（首选）** | 同一 `@host` 内先 launch 初始化 kernel、再 launch 散射 kernel——**同 stream 排序天然有序，零栅栏** | 两段可独立成 kernel（scatter_elements：`_copy_simt` → `_scatter_*` 实测形态） |
| 单 kernel + 栅栏 | kernel 内段间 `global_sync_all()`（代价高，禁内层循环；细粒度用 `vec_sync_block_arrive/wait` 模式 0） | 两段共享片上状态、拆 launch 不经济时 |

`CAP_SYNC_SCHEDULE_MODE` 在 DSL 语义下退化为"顺序 launch 即有序"（[capability-mapping.md](capability-mapping.md) §1）；通用层"SetScheduleMode 前提"检查项在 DSL 侧改查"两段是否同一 stream"。

## 5 值依赖输入（shape/new_shape 类）

DSL host 是**构图期**执行，不能读运行期张量内容（AscendC tiling `GetData` 的能力不存在）：

1. **AI CPU 路线**（正解）：`@aicpu_kernel` 运行期读 GM，产出 stride/-1 维推断/分核计划 → 写 GM metadata → AI Core kernel 按 [index-mapping.md](index-mapping.md) §2 消费。同 stream 保证 AI CPU 先于 AI Core
2. **入口预解析**：公开 Python 入口能拿到真实值时（调用方直接构图），入口侧解析、`Constexpr`/params 传递
3. 声明义务：设计文档登记"构图期不可读值"清单 + 所选路线；禁止假设"host 能读输入值"

## 6 launch 几何与核数

- 纯 SIMT：`kernel[dim3(block), dim3(thread)]`，thread ∈ [1,1024]；混合 SIMD+SIMT：`kernel[block_count]` + `vf(mode="simt", thread=N)`——**两形态不可互换，编译器校验**
- 核数经 `get_platform_info().vector_core_num` 查询（torch_npu `get_device_properties` 先例：`vector_core_num`），**严禁写死**；散射算子惯用"有用核数截断"：`useful = min(requested, ceil(N / THREAD_COUNT))`
- `N=0` 时不 launch（`dim3(0)` 非法）；多核数超过单 launch 覆盖（> 65535×1024 输出）时 host 多段 launch + 偏移累进（块协议同款，[../../generic/splitting.md](../../generic/splitting.md) §3）
