<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式 DSL 陷阱与错题集

> 通用层错题（轮数默认值、count 截断、接口位拆分错位等语言中立项）见 [../../generic/](../../generic/index.md) 各文档与 [../ascendc/pitfalls.md](../ascendc/pitfalls.md) 同名条目；此处只列 DSL 特有。依据 cannbot-dsl 官方文档 danger/warning 块。

## 1 正确性类

### 1.1 VF 嵌套假设
- **现象**：想在 SIMT VF 内调 SIMD `v*` 批量接口，编译失败或语义错乱
- **根因**：一个 VF 只选一种模式；SIMT 内的"v*"是**线程内短向量**（`make_short_vector` 系），不是跨 lane SIMD
- **规避**：SIMD 段与 SIMT 段**串接**（各自 VF 作用域 + UB 交接），不嵌套

### 1.2 构图期读 seed/offset 张量
- **现象**：host 想读 offset 值算流起点，拿到构图期占位/垃圾
- **根因**：DSL host 构图期执行，不可读运行期 GM 值
- **规避**：offset→counter 换算设备侧自算（SIMT 内 `⌈offset/4⌉`）；回写走独立 launch 或唯一线程

### 1.3 未赋值掩码
- **现象**：SIMD 路线有效元素位置随机错
- **根因**：带 `mask` 参数的接口传入了未预先赋值的掩码寄存器（内容不确定）
- **规避**：掩码一律经 `full_mask`/`update_mask`/`mask_counter` 产出后传入；尾块用掩码不用分支

### 1.4 运行期值进 `range_constexpr`
- **现象**：编译报错或轮数/循环边界异常
- **根因**：把 params 读出的运行期标量用于编译期结构
- **规避**：运行期值只作 `range()` 边界/地址/设备分支；编译期常量走 `Constexpr`

### 1.5 半精度舍入口径未对拍
- **现象**：fp16/bf16 输出与流基准低位不一致
- **根因**：DSL 无 `SetCtrlSpr<60,60>` 类半精度旁路开关，`vcast` 的 RN 舍入与 AscendC 路径可能不同口径
- **规避**：fp32 中间域 + 出口单点 cast；舍入模式显式（`RoundingMode.RN`）；对拍锁死（待验证项）

## 2 性能类

### 2.1 Hardware Loop 静默退化
- **现象**：SIMD 路线主循环慢数倍且无警告
- **根因**：循环体内运行时 `if`（如变换分支）阻碍 Hardware Loop，连带阻断 VF 融合
- **规避**：分支段拆 SIMT VF（混合形态）；尾块改"做几次"；`const_expr` 消编译期分支

### 2.2 寄存器溢出（SIMD）
- **现象**：轮链 + 变换链中间值多，性能骤降
- **根因**：32 个数据寄存器超限，溢出到 UB（VF 溢出区仅 6 KB）
- **规避**：10 轮 Philox 直线展开与寄存器预算的平衡——每轮落 UB 的切分点首例实测；旋钮是 VF 长度与分段

### 2.3 线程数与寄存器错配（SIMT）
- **现象**：512→1024 线程性能反降
- **根因**：SIMT 寄存器档位——线程数翻倍、每线程寄存器减半；每线程 key/counter 状态 + 变换局部变量溢出 GM 栈
- **规避**：512 惯例起步；状态重的（变长消耗）降档实测

### 2.4 UB 双口径
- **现象**：SIMD 路线按 216 KB 保守、SIMT 路线按 248 KB 溢出
- **根因**：SIMT Data Cache ~32 KB 不在 `get_mem_size("ub")` 返回内；两口径适用场景不同
- **规避**：纯 SIMD 248 KB / 含 SIMT 216 KB / 混合取最紧；勿对 248 KB 再减顶部 8 KB（已扣）

## 3 工程类

- **launch 形态校验**：纯 SIMT `dim3(block), dim3(thread)` 与混合 `block_count` + `vf(thread=N)` 不可互换——整算子统一
- **`Constexpr` 泛滥**：缓存键乘积爆炸（每维度进键）；只编码影响代码生成的量（VL、block_loops、轮数）
- **流基准对拍缺失**：DSL 无仓内 golden——首例未做 [prng-implementation.md](prng-implementation.md) §6 五项对拍即合入，后续算子全部失去参照（对拍结果是本路线唯一的"实测基准"来源）
- **CANN 版本**：SIMT 需 ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）；混合 SIMD+SIMT 同受此限
- **`cbd.print` 覆盖内建**：`from cannbotdsl import *` 会覆盖 `print`——设备打印写 `cbd.print`，构图期打印用内建
