<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling TF 对标 SIMD 路线

> **层级**：AscendC 语言适配层。对标 TensorFlow 接口族（seed/seed2 属性 + shape 张量，或 stateless key/counter 输入）的向量实现路线。
>
> 基准算子：random_uniform_v2（最小样例）、stateless_random_uniform_v3、stateless_random_normal_v2/v3、stateless_drop_out_gen_mask。
> **适用域硬边界（R20）**：本路线适用于 TF 对标（A/C 族：seed/seed2 属性、stateless key/counter 张量）接口族；跨族引用即范式违规——按其他族的接口套用本路线形态、或以本路线参照算子为其他族实现背书，均违反路线选型机械匹配（patterns 场景判定·第二步）。


## 1 路线特征

| 维度 | 形态 |
|------|------|
| 执行体 | 向量核，TQue 流水（双缓冲） |
| 随机数生成 | `PhiloxRandom<10>`（向量化 Philox，一次产出整块 uint32）。**⚠️ 轮数必须显式写 10**——`PhiloxRandom` 模板默认轮数（7）与原生流（10）不一致，漏显式实参 → 流错且自身逐位复现无法发现（只能靠与流基准对拍暴露） |
| 数据通路 | GM ←(DataCopyPad)─ TQue(输出) ←(cast/位操作)─ TBuf(philox 结果) |
| buffer 预算 | `singleBufferSize = (可用 UB − sharedTmp − Dcache 扣减) / bufNum`，`bufNum = sizeof(uint32_t) + dtypeSize × 缓冲份数(2)`；复合型 K 份折算见 [common-template.md](common-template.md) §2 |

## 2 kernel 骨架（random_uniform_v2 实例）

```cpp
class Op : public RandomKernelBaseOp {          // VarsInit/Skip/GenRandomSIMD 由基类提供
    TBuf<VECCALC> philoxQueBuf_;                 // 流结果暂存（uint32）
    TQue<VECOUT, 2> outQueY_;                    // 输出双缓冲
};
Process():
  1. offset 张量读值 → Skip(offset)              // 流起点对齐（state-management §3）
  2. SyncAll()                                    // 所有核完成 offset 读取后才能回写
  3. Skip(blockOffSet / k)                        // 本核流区间对齐（k=4）
  4. 循环 ubRepeatimes 次：
       GenRandomSIMD(philoxRes, n)               // 向量 Philox 批量产出
       U32Conversion(y, philoxRes, n)            // uint32 → 目标 dtype（见 §3）
       CopyOut(...)                               // DataCopyPad 搬出
       Skip(n / k)                                // 流区间推进
  5. 唯一核回写 offset（见 §4）
```

## 3 随机数→目标域转换（U32Conversion）

IEEE 754 尾数位拼接技巧（寄存器级，无除法）：

| 目标 dtype | 构造 | 值域 |
|-----------|------|------|
| float | `(u & 0x7fffff) \| (127<<23)` 再 −1.0 | **[0,1)**（含 0） |
| half | u 截 16 位后 `(u & 0x3ff) \| (15<<10)` 再 −1.0 | **[0,1)**（含 0） |
| bfloat16 | `(u & 0x7f) \| (127<<7)` 再 −1.0 | **[0,1)**（含 0） |

- 值域为 **[0,1)**（位重解释含 0）；与乘法路径 (0,1) 是同流不同值的两条口径——路径一致性硬规则见 distribution-transform §4（R7）
- fp16/bf16 路径需 `SetCtrlSpr<60,60>(0/1)` 包裹（半精度旁路控制）

## 4 正态分布（Box-Muller 的向量实现）

两段寄存器级变换（stateless_random_normal 族）：

1. `BoxMullerFloatSIMD`：双发取数（DIST_DINTLV_B32 解交错）→ `Maxs(eps) → Ln → Muls → Sqrt`，产 `2π·u2` 与 `√(−2·ln u1)`
2. `BoxMullerMulSIMD`：`Cos/Sin` 向量库 + 交错写回（DIST_INTLV_B32）

成对消耗 c=2 的布局系数 g 在 tiling 侧按 `groupCnt = 偏移 / 4`（k=4 每组）推导。

## 5 状态回写

```cpp
if (blockIdx_ == 0) {                            // 唯一写者
    offsetGm_(0) += outputSize * K_Reserved_Per_Output;   // L·R 递进（R=256）
    DataCacheCleanAndInvalid<int64_t, SINGLE_CACHE_LINE, ...>(offsetGm_);  // 缓存可见性
}
```

- 回写前必须 `SyncAll()`（时序约束，state-management §3）
- 写者选择：0 号核或尾核（各算子实测有差异，保持算子内一致即可）

## 6 混合形态：SIMD 内嵌 SIMT 变换

变换段含逐元素分支/标量数学（如 Box-Muller 的 sincosf）时，在向量核内嵌 SIMT VF：

```cpp
PhiloxRandom<10>(yOutputTmp, key, counter, n);                    // SIMD 生成
asc_vf_call<SimtBoxMuller>(dim3{1024}, ubuf指针, n);              // SIMT 变换（__ubuf__ 直传）
Float32Conversion<T>(out, tmp, n);                                 // SIMD 收尾 cast
```

- **必须配置 `config.DcacheSize`**（tiling 侧扣减并 `SetLocalMemorySize`），漏配是常见错误（见 [pitfalls.md](pitfalls.md)）
- UB 传参：SIMT VF 收 `__ubuf__` 指针（片上地址直通）
- 基准：random_standard_normal_v2、random_uniform_int_v2、stateless_random_choice_with_mask
