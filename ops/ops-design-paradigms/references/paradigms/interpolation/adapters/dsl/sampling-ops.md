<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式采样实现（DSL SIMT）

> CANNBotDSL 下坐标分解、邻居 gather 与累加精度的落地。以 scatter_elements（SIMT 数据面先例）+ three_interpolate 草稿为基准；算法模型见 [../../generic/coordinate-mapping.md](../../generic/coordinate-mapping.md) 与 [../../generic/weight-model.md](../../generic/weight-model.md)。

## 1 grid-stride 与坐标分解

```python
with cb.vf(mode="simt"):
    tid = dtypes.int64(cb.simt.thread_idx()[0])
    blk = dtypes.int64(cb.simt.block_idx()[0])
    tpb = dtypes.int64(cb.simt.thread_dim()[0])
    nblk = dtypes.int64(cb.simt.grid_dim()[0])
    start = blk * tpb + tid
    step = nblk * tpb
    for i in range(start, total_out, step):
        b = i // nsc            # 整除/取模直接表达
        r = i - b * nsc
        n = r // cs
        c = r - n * cs
        ...
```

- **整除/取模**：DSL 无 AscendC `UintDiv` 函数族——`//` 与 `%` 直接表达；生成路径的除法开销未登记（[capability-mapping.md](capability-mapping.md) §2），totalOut 大的热点算子实测；必要时 host 预计算 magic/shift 入 params Tensor 手工快除
- **索引全程 int64**：`dtypes.int64(...)` 承载坐标/偏移；批次基址乘加在 int64 域（`b * ms * cs` 溢出防护，同 AscendC 教训）
- 线程完全独立，尾块由步进覆盖

## 2 邻居 gather 与显式索引防御

```python
max_idx = ms - 1
raw0 = dtypes.int64(idx[idx_base + 0])      # 运行期直读（值驱动）
safe0 = raw0 if raw0 >= 0 else dtypes.int64(0)
safe0 = safe0 if safe0 <= max_idx else max_idx   # 双向 clamp 防御
f0 = features[feat_off + safe0 * cs + c]    # 离散直读 GM
w0 = weights[idx_base + 0]
```

- Tensor 下标读写直访 GM（`CAP_SIMT_GM_DIRECT`）；邻居地址无局部性，靠线程并行隐藏延迟
- int64 idx 输入的负值/大值先升 int64 再 clamp（契约保证类也保留防御）

## 3 坐标映射（RegularResize / RoiSample）

```python
g = dtypes.float32(out_idx) + dst_start      # fp32 各自独立舍入
if half_pixel:                               # 对齐模式经 Constexpr 或路径实例分派
    f = (g + 0.5) * scale - 0.5              # ⚠️ 收缩行为待验证（§4）
    f = simt.fmax(f, 0.0)                    # max 在全局坐标系（重基之前）
    f = f - src_start
else:
    f = g * scale - src_start
f0 = simt.floor(f)                           # 向负无穷，返回浮点值
lerp = simt.fmin(f, in_extent_m1) - f0       # lerp 用未 clamp 的 f0
i0 = clamp(f0, 0, in_extent_m1)              # 双向 clamp；i1 由 clamp 后 i0 推导
```

- `simt.floor` / `simt.fmax` / `simt.fmin` 均为已文档化 API（fp16/bf16/fp32；NaN/±inf 语义见表）
- scale / extent / srcStart 标量经 params Tensor 运行期读，或 host 已知时 `Constexpr`（影响代码生成的才进 Constexpr）

## 4 加权和与累加精度

```python
# 分步绑定（FMA 收缩行为未验证前的保守写法）：
p0 = dtypes.float32(w0) * dtypes.float32(f0)   # 乘积独立绑定
p1 = dtypes.float32(w1) * dtypes.float32(f1)
p2 = dtypes.float32(w2) * dtypes.float32(f2)
acc = (p0 + p1) + p2                            # 项序与 golden 一致
output[i] = acc
```

- **禁写单表达式** `w0*f0 + w1*f1 + w2*f2`——DSL 代码生成是否收缩未实测（`CAP_FMA_CONTROL` 待验证）；分步中间绑定是保守等价物，首例实测后再决定是否放宽
- half/bf16 输入升 fp32 累加，出口一次性 cast
- **补偿路线（fp64 golden 模拟）**：`th = h*q; tl = simt.fma(h, q, -th)`（TwoProdFMA 误差项，`simt.fma` 单次舍入语义）+ 顺序累加 + 2^-53 吸收截断 + `simt.isfinite` 非有限分支——算法与 [../ascendc/sampling-ops.md](../ascendc/sampling-ops.md) §5.2 一致，原语换 `simt.fma` / `simt.floor`；寄存器开销（df32 高低位字）未量化，首例实测（[capability-mapping.md](capability-mapping.md) §3）

## 5 grad 散射段

- 两段顺序 launch（零初始化 → 散射），同 stream 免栅栏（[kernel-entrance.md](kernel-entrance.md) §4）
- `cb.simt.atomic_add(output.ptr(off), g * w)`：dtype 域 `{int32, uint32, float32, int64, uint64}`；fp16/bf16 grad 升 fp32 或批量 `mem_copy(atomic_add=True)` 轨
- 散射项乘积先独立舍入（FMA 未判定）；贡献丢弃类越界在散射前过滤

## 6 设计输出清单

- [ ] 坐标分解路径（`//` 直除 vs params Tensor 手工快除）与性能依据
- [ ] 对齐模式分派形态（路径实例 vs Constexpr）
- [ ] 加权和写法（分步绑定）与 FMA 待验证声明
- [ ] 累加路线判定（fp32 / 补偿）与验证义务
- [ ] grad 散射的初始化/原子 dtype/越界过滤三段声明
