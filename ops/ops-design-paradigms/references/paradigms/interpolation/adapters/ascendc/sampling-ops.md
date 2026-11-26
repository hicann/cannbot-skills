<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式采样实现（SIMT）

> AscendC SIMT 下坐标分解、邻居 gather 与累加精度的落地。以 three_interpolate / sync_resize_bilinear_v2 实测结构为基准；算法模型见 [../../generic/coordinate-mapping.md](../../generic/coordinate-mapping.md) 与 [../../generic/weight-model.md](../../generic/weight-model.md)。

## 1 坐标分解（UintDiv 快除）

线性输出索引 → 多维坐标的标准落地（three_interpolate）：

```cpp
// host/Process 侧预计算快除参数，经 UB 广播
LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
LocalTensor<uint32_t> divParams = ubAlloc.Alloc<uint32_t>(8);
GetUintDivMagicAndShift<uint32_t>(magicCs, shiftCs, cs);
divParams.SetValue(0, magicCs);
divParams.SetValue(1, shiftCs);
DataSyncBarrier<MemDsbT::UB>();          // 写后同步，防读旧值

// kernel 侧：除法消为乘移
uint32_t nIdx = Simt::UintDiv<uint32_t>(ncRemainder, magicCs, shiftCs);
uint32_t cIdx = ncRemainder - nIdx * cs;
```

- 每维 2 参数（magic/shift）；参数表布局 `[magicW, shiftW, magicH, shiftH, ...]`，分配 32B 对齐
- **降维**：对平面基址等价的相邻轴可合并消一次除法（sync_resize R005：`(n×C+c) == index/(H×W)` 时 N/C 轴不分解，仅 H/W 两次快除）
- 快除的参数求法、适用位宽与限制见公共知识 [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md)

## 2 grid-stride 采样循环

```cpp
for (INDEX_SIZE_T index = blockIdx.x * blockDim.x + threadIdx.x;
     index < totalElements;
     index += blockDim.x * gridDim.x) {
    // 坐标分解（§1）→ 坐标映射（§3）→ 邻居 gather（§4）→ 加权求和（§5）
    y[index] = acc;
}
```

- 输出逐位独立，执行单元完全无协作；尾块由步进天然覆盖
- `INDEX_SIZE_T` 按 `totalElements` 是否超 INT32_MAX 由入口分派（[kernel-entrance.md](kernel-entrance.md) §3）
- **地址偏移一律 `uint64_t`**：`featuresBatchOffset = bIdx × ms × cs` 等批次基址可超 UINT32_MAX（实测修复）

## 3 坐标映射落地（对齐模式分派）

```cpp
// MapCoord：输出索引 → (i0, i1, lerp)，与 golden 严格同序（sync_resize_bilinear_v2）
float g = static_cast<float>(outIdx) + static_cast<float>(dstStart);  // 升全局坐标，fp32 相加独立舍入
float f;
if constexpr (HALF_PIXEL) {
    volatile float t = (g + 0.5f) * scale;   // 先舍入乘积（禁 FMA 融合）
    f = t - 0.5f;
    f = fmaxf(f, 0.0f);                     // max(·,0) 在全局坐标系（重基之前）
    f = f - static_cast<float>(srcStart);    // 重基到本地输入坐标
} else {                                      // align_corners
    volatile float p = g * scale;
    f = p - static_cast<float>(srcStart);
}
float f0 = floorf(f);                         // 向负无穷取整
i0 = clamp(f0, 0, inExtent - 1);              // 防御性双向 clamp
i1 = min(i0 + 1, inExtent - 1);               // i1 由 clamp 后 i0 推导
lerp = fminf(f, inExtent - 1) - f0;           // lerp 用未 clamp 的 f0
```

- `volatile` 是 [../../generic/coordinate-mapping.md](../../generic/coordinate-mapping.md) §5 浮点同序规则的落地手段（`CAP_FMA_CONTROL`）
- IndexedInterpolate 无此段：读入 idx 值升宽整型后 clamp 即为邻居（`rawIdx → uint64_t → clamp [0, ms-1]`）

## 4 邻居 gather

- 邻居地址 `base + i × stride + c` 离散读，直访 GM（`__gm__ const T*` 解引用）；不搬 UB 数据面
- 显式权重（IndexedInterpolate）与邻居同批读入（`weight[idxBase + k]`）
- k ≤ 4 邻居（bicubic 16 次）各自独立读；无预取，依赖线程级并行隐藏延迟

## 5 累加精度落地

### 5.1 常规路线（fp32 累加器）

```cpp
// volatile 阻断 FMA 收缩：乘积先舍入，加法序与 golden 一致（three_interpolate）
volatile float p0 = static_cast<float>(w0) * static_cast<float>(f0);
volatile float p1 = static_cast<float>(w1) * static_cast<float>(f1);
volatile float p2 = static_cast<float>(w2) * static_cast<float>(f2);
float acc = (p0 + p1) + p2;                  // 舍入顺序 = golden 的 ((p0+p1)+p2)
y[index] = static_cast<T>(acc);
```

- half/bf16 输入升 fp32 后累加，末位一次性 cast 回 T
- **禁写成单一表达式** `w0*f0 + w1*f1 + w2*f2`：ccec 收缩为 fma 链（如 `fma(w1,f1,w0*f0)`），消除乘积中间舍入，产生系统性 1~2 ULP 偏差（实测误差恒 2^-18）

### 5.2 补偿路线（df32 模拟 fp64 golden）

启用条件与判定树见 [../../generic/weight-model.md](../../generic/weight-model.md) §3。AscendC 原语组合（sync_resize_bilinear_v2 实测收敛：mere 4.4e-14 / mare 7.1e-8）：

```cpp
// 项分解：TwoProdFMA——th = h*q 单次舍入，tl = fmaf(h,q,-th) 精确误差项
float th = h * q;
float tl = fmaf(h, q, -th);

// 累加：Knuth TwoSum（分支自由，RN 下误差变换精确）
TwoSumF(sHi, tHi, hi, e);                    // s = RN(a+b)，e = 精确误差
float lo = (sLo + tLo) + e;
// 吸收截断：|lo| < 2^-53×|hi| → 0，复现 fp64 舍入吸收语义
float thr = fabsf(rHi) * 1.1102230246251565e-16f;
oLo = fabsf(rLo) < thr ? 0.0f : rLo;

// (1 - lerp) 精确残差：OneMinusResid（Knuth TwoSum 误差项恒等式）
```

- **非有限值分支**：任一邻居样本 `!isfinite` 时禁用补偿链，退回简单乘加——`fmaf(h, ±inf, -inf) = nan` 产生伪 NaN，且异号 inf 相消的 nan 是 golden 语义，不可屏蔽
- **溢出防护**：±FLT_MAX 值域对样本 2 的幂预缩（如 ×2^-4），末尾还原

## 6 grad 散射段

- 两段顺序 launch：零初始化段（幂等常量写）+ 散射段（`asc_atomic_add`），同 stream 排序免 SyncAll；无法保证时显式 `SyncAll()` + `SetScheduleMode(1)`（同 Sparse 范式骨架）
- 散射项 `g × w` 的乘积同样受 FMA 同序约束（先舍入再原子加）
- 贡献丢弃类越界在散射前过滤（`i0/i1` 出界的邻居项跳过）

## 7 设计输出清单

- [ ] 快除维度清单与参数表布局（含降维论证）
- [ ] 偏移承载类型（批次基址 uint64 核算）
- [ ] 坐标映射的 volatile 阻断点清单（乘积/max 时机/floor 位宽）
- [ ] 累加路线判定（§5.1/§5.2）与依据
- [ ] grad 散射的初始化/原子/越界过滤三段声明
