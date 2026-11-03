<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式索引数学实现

> 线性化/反线性化、-1 维推断与 stride 表传递的 AscendC 落地。算法语义见 [../../generic/access-model.md](../../generic/access-model.md) §6。

## 1 线性化 / 反线性化（sparse_reshape 基准）

```cpp
// host 侧预计算（tiling），kernel 不做乘积循环
//   strideIn[d] = product(shape[d+1:]),  strideIn[D_in-1] = 1
//   strideOut[d] = product(resolved[d+1:]), strideOut[D_out-1] = 1

// kernel 侧：每条目独立变换，无冲突直写
int64_t linear = 0;
for (int d = 0; d < D_in; ++d)
    linear += static_cast<int64_t>(indices[i * D_in + d]) * strideIn[d];   // 线性化
for (int d = 0; d < D_out; ++d) {
    yIndices[i * D_out + d] = linear / strideOut[d];                       // 反线性化
    linear %= strideOut[d];
}
```

## 2 -1 维推断（host 侧）

```
resolved = new_shape
若 resolved 恰有一个 -1：missing = total_elements / product(其余维)
  （除不尽 → 报错；多于一个 -1 → 报错）
y_shape = resolved
```

推断依赖 `shape` / `new_shape` 的**值**：静态 tiling 经 `GetConstIntData` 读、运行时 tiling 经 `GetData<int32_t>()` 读（值依赖三处声明见 [../../../common/value-depend-process.md](../../../common/value-depend-process.md)）。

## 3 stride / shape 表的传递

| 体积 | 传递方式 |
|------|---------|
| 少量标量（≤ `CAP_VF_PARAM_MAX`=28） | 直接 `asc_vf_call` 入参（sparse_apply_adadelta：11 参含 lr/rho/epsilon 标量） |
| 数组形态（如 8 维 stride 表 2×64B） | tiling 侧写入 UB 共享内存，kernel 以 `__ubuf__` 指针读取，全体线程复用 |

UB 共享内存经 TBuf 申请、`SetLocalMemorySize` 控制预算（混合场景上限 216 KB）；只有元数据走此路，主数据面保持 GM 直访。

## 4 溢出与类型防护

- 索引中间量统一 `int64_t`：`int64_t val = static_cast<int64_t>(values[idx])` 后再做比较与乘加——int32 索引在 `batch * sizeVal + val` 类偏移计算中会溢出
- 除法/取模无硬件指令时用预计算乘数 + 移位替代（反线性化内层循环的热点）
- 偏移计算先做值域检查（`val < 0 || val >= sizeVal → continue`），防负数取模与越界写

## 5 行级偏移（RowApply / ScatterAdd）

```
行基址   = idx * innerSize
输出偏移 = indices[j] * innerSize + k        // k ∈ [0, innerSize)
```

- 行内 `k` 循环由同一线程顺序执行（行内状态依赖）
- 行间完全独立；`innerSize` 大（如 19994）时行内循环上限是性能敏感点
