<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 Kernel 入口与编译组织（DSL）

> CANNBotDSL 的入口结构、params ABI、编译实例与 launch 要点。基准：`samples/scatter_elements/scatter_elements.py`（host 分派 + AOT 缓存完整先例）；同算子 DSL 草稿见 [example/three_interpolate_simt_draft/](example/three_interpolate_simt_draft/)。

## 1 三层结构（对 AscendC 三段式的映射）

| AscendC（[../ascendc/kernel-entrance.md](../ascendc/kernel-entrance.md)） | DSL 对等物 |
|------|------|
| `__global__` 入口 + `GET_TILING_DATA_WITH_STRUCT` | `@kernel` 函数：Tensor 形参 + `cb.Constexpr[int]` 编译期配置；tiling 载体无 |
| TilingData（host 可算标量打包） | **params Tensor**（GM int64 数组，手工 ABI）——运行期元数据（shape/extents）；`Constexpr`——编译期配置（对齐模式、线程相关常量） |
| TilingKey 场景分派 | host 侧 Python 路径选择 + **每路径独立编译实例**（RegularResize 的 half_pixel / align_corners、最近邻路径各一实例） |
| tiling 侧校验（`OP_CHECK_IF`） | 公开入口侧纯 Python 校验（shape/rank/dtype/extent 越界，scatter_elements `ScatterElementsTiling` 先例） |
| `asc_vf_call` + 线程常量 | `@kernel` 内 `with cb.vf(mode="simt")`；纯 SIMT 线程数由 `kernel[dim3(block), dim3(thread)]` launch 给定 |

## 2 kernel 骨架

```python
@kernel
def _three_interpolate_simt(features: Tensor, idx: Tensor, weights: Tensor,
                             output: Tensor, params: Tensor):
    with cb.vf(mode="simt"):
        tid = dtypes.int64(cb.simt.thread_idx()[0])
        blk = dtypes.int64(cb.simt.block_idx()[0])
        ...   # grid-stride 主循环：坐标分解 → 索引 clamp → 加权 gather
        output[i] = acc

@host
def three_interpolate(features, idx, weights, output, params,
                      cores: cb.Constexpr[int]) -> None:
    _three_interpolate_simt[cb.dim3(cores), cb.dim3(THREADS)](
        features, idx, weights, output, params)
```

- **方括号 launch 必须直接位于 `@host` 函数体**；`@kernel` 不能作为普通 Python 函数调用
- 同算子多路径（grad 正反向、对齐模式）各建 `@kernel` + host 分派函数，独立编译实例
- 值依赖张量（idx / weights / 框坐标）在 kernel 内运行期直读——host 构图期不可读 GM

## 3 编译实例管理（AOT 缓存）

```python
@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled(path, bs, ns, cs, ms, idx_dtype, w_dtype, out_dtype):
    descriptors = (
        cb.TensorSpec((bs*ms*cs,), feat_dtype),     # 秩 1 裸视图口径
        ...
        cb.TensorSpec((PARAM_COUNT,), dtypes.int64),
    )
    return cb.compile(launch, *descriptors, cores)
```

- **缓存键 =（路径, shape, dtype, 核数, 全部 Constexpr）**——只把"影响代码生成"的量设为 Constexpr；shape/extents 走 params Tensor 不进缓存键（同 shape 档位复用）
- shape 校验/stride 推导在构图期纯 Python 完成，产物打包进 params Tensor

## 4 grad 两段骨架（零初始化 → 散射）

| 路线 | 写法 | 适用 |
|------|------|------|
| **顺序双 launch（首选）** | 同一 `@host` 内先 launch 零初始化 kernel、再 launch 散射 kernel——同 stream 排序天然有序，零栅栏 | 标准形态（scatter_elements `_copy_simt` → `_scatter_*` 先例） |
| 单 kernel + 栅栏 | kernel 内段间 `global_sync_all()`（代价高，禁内层循环） | 两段共享片上状态时 |

- 散射项 `g × w` 乘积先独立舍入再 `cb.simt.atomic_add`（FMA 收缩未判定前分步绑定，[capability-mapping.md](capability-mapping.md) §1）
- fp16/bf16 grad_output 升 fp32 原子；贡献丢弃类越界在散射前过滤

## 5 值依赖输入（crop_size / 框坐标类）

DSL host 是**构图期**执行，不能读运行期张量内容（AscendC tiling `GetData` 的能力不存在）：

1. **AI CPU 路线**：`@aicpu_kernel` 运行期读 GM，产出框坐标计划/输出 shape → 写 GM metadata → AI Core kernel 消费（同 stream 保证先后）
2. **入口预解析**：公开 Python 入口能拿到真实值时（调用方直接构图），入口侧解析、`Constexpr`/params 传递
3. 声明义务：设计文档登记"构图期不可读值"清单 + 所选路线

## 6 launch 几何与核数

- 纯 SIMT：`kernel[dim3(block), dim3(thread)]`，thread ∈ [1,1024]；混合形态不可互换，编译器校验
- 核数经 `get_platform_info().vector_core_num` 查询，**严禁写死**；有用核数截断：`useful = min(requested, ceil(totalOut / THREAD_COUNT))`
- `totalOut = 0` 时不 launch（`dim3(0)` 非法）；正向无输出初始化需求（逐位直写），grad 才有零初始化段
- 输出超单 launch 覆盖（> 65535 × 1024）时 host 多段 launch + 偏移累进
