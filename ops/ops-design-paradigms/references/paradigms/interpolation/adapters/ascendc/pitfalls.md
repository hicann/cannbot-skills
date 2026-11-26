<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式陷阱与错题集

> 蒸馏自 three_interpolate / sync_resize_bilinear_v2 / crop_and_resize 的审查记录与上板修复（three_interpolate 上板两轮精度修复；sync_resize 五轮 TTK 收敛）。按"现象 → 根因 → 规避"组织。

## 1 正确性类

### 1.1 FMA 收缩产生系统性 ULP 偏差
- **现象**：黑盒误差恒定 2^-18（3.81e-06），同一用例所有元素偏差一致，FMA 链仿真与误差逐位吻合
- **根因**：`w0*f0 + w1*f1 + w2*f2` 被编译器收缩为 `fma(w1,f1,w0*f0)` 链，消除乘积中间舍入；golden 是分步 mul-then-add
- **规避**：加权和一律 volatile 中间积 + 显式加法序（[sampling-ops.md](sampling-ops.md) §5.1）；golden 的项序写进设计契约

### 1.2 坐标映射 max/floor 次序错
- **现象**：边界列/行整列偏移，或 lerp 末位不一致
- **根因**：`max(·,0)` 施加在重基之后；或 lerp 用 clamp 后的 f0；或 floor 在 fp64 域先取整再舍入
- **规避**：MapCoord 逐句与 golden 同序（max 时机全局坐标系、lerp 用未 clamp f0、floor 在 fp32 域，[sampling-ops.md](sampling-ops.md) §3）

### 1.3 int64 索引读入 OOB
- **现象**：int64 idx 输入下 profiling/执行崩溃
- **根因**：idx 含负值/超界值时直接 `static_cast<uint32_t>` 截断回绕，地址越界
- **规避**：读值先升 `uint64_t` 再 clamp 到 `[0, extent-1]`；契约保证合法的算子保留防御（每邻居两次比较，代价可忽略）

### 1.4 32 位路径批次基址溢出
- **现象**：totalElements 未超 INT32_MAX 仍写错地址
- **根因**：`bIdx × ms × cs` 批次基址超 UINT32_MAX，32 位下标 ≠ 32 位基址
- **规避**：GM 偏移统一 `uint64_t` 乘加；`totalElements` 与"各基址"分别核算上限

### 1.5 补偿链在 inf 下产生伪 NaN
- **现象**：±FLT_MAX 值域用例输出 NaN，golden 输出有限值
- **根因**：`fmaf(h, ±inf, -h×±inf) = inf + (-inf) = nan`——补偿链在非有限样本下产出伪 NaN，异号 inf 相消本应是 golden 语义
- **规避**：任一邻居非有限 → 分支退回简单乘加（[sampling-ops.md](sampling-ops.md) §5.2）；`0 × inf = nan` 与权重精确零条件逐点核对

## 2 精度类

### 2.1 golden fp64 吸收语义未复现
- **现象**：golden 相消得 0 的元素上相对误差爆炸（实测 mare 4.9e28）；补偿链"更精确"反而更差
- **根因**：golden 在 fp64 中顺序累加，每步舍入把 < 2^-53|累加器| 的小分量吸收丢弃；补偿链保留全部分量反而多出小项
- **规避**：df32 顺序累加 + 每步 2^-53 吸收截断（[sampling-ops.md](sampling-ops.md) §5.2）；验证指标用 mere/mare 双报

### 2.2 fp32 大维度累加精度不达标
- **现象**：大 C 维度 shape 精度 ~97%，容差 1e-4 边缘失败
- **根因**：half 输入直接在 half 域乘加；或乘积未独立舍入即累加
- **规避**：升 fp32 累加器 + volatile 分步舍入（three_interpolate 上板修复实测收敛）

### 2.3 `(1 - lerp)` 直减舍入
- **现象**：lerp < 0.5 时权重和偏差 ~2^-25，逐位对拍失败
- **根因**：fp32 中 `1 - lerp` 有舍入；golden 在 fp64 中精确
- **规避**：OneMinusResid 恒等式（Knuth TwoSum 误差项）补偿；常规容差场景可省

## 3 性能类

### 3.1 线程数与寄存器压力错配
- **现象**：1024 线程不升反降；补偿路线更低
- **根因**：坐标映射 + 快除 + 邻居索引 ~20+ 活跃寄存器值，1024 线程 32 regs 档有 spill；df32 高低位字再抬升
- **规避**：采样类固定 512 分档；补偿路线中低档；`__launch_bounds__` 与 dim3 共用常量

### 3.2 多余快除
- **现象**：3 次快除比 2 次慢
- **根因**：N/C 轴对平面基址等价时第 3 次除法冗余
- **规避**：按 `(n×C+c) == index/(H×W)` 类等价式降维（[sampling-ops.md](sampling-ops.md) §1）

### 3.3 TilingKey 实例爆炸
- **现象**：编译时间/产物大小失控（288 实例）
- **根因**：对齐模式×method×dtype×attr 全维度进 TilingKey
- **规避**：dtype 交 DTYPE_ 宏模板；TilingKey 只留主优化维度（crop_and_resize method-only 33 实例，实测影响可忽略）

## 4 工程类

- **空输出**：`totalElements == 0` kernel 短路 return，但 tiling 侧 `needCoreNum` 下限 1 核（`SetBlockDim(0)` 非法）
- **快除参数未同步**：UB 写 magic/shift 后漏 `DataSyncBarrier` → 读旧值
- **MDE–Code 偏差链**：TilingKey 结构、volatile 阻断点、线程分档属设计契约，变更须回写设计文档（sparse_bincount 同款教训，[../../../sparse/adapters/ascendc/pitfalls.md](../../../sparse/adapters/ascendc/pitfalls.md) §2.1）
- **grad 漏 SetScheduleMode(1)**：显式 SyncAll 路径下必配，成对检查
