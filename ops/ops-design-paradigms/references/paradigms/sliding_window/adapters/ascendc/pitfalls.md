<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式陷阱与错题集

> 蒸馏自 max_pool 系 / avg_pool 系算子的上库修复与 gen_simt_code 生成记录审查（max_pool_v3_grad design_peer 6 项 FAIL；avg_pool_update dev_peer 属性漏注册）。按"现象 → 根因 → 规避"组织。

## 1 正确性类

### 1.1 dilation 负起点 clamp 错位
- **现象**：dilation > 1 且 pad 使窗口起点为负时，窗口内混入窗格外元素（或漏格点）
- **根因**：负起点 `max(start, 0)` clamp 到 0——窗口格点是 `start + i·d`，clamp 后 0 不在格点序列上
- **规避**：`while (start < 0) start += dilation` 推进（[simt-path.md](simt-path.md) §1）；dilation=1 时两写法等价

### 1.2 max 聚合 NaN 漏传播
- **现象**：窗口含 NaN 输出却非 NaN
- **根因**：更新只写 `val > acc`——NaN 比较恒 false，winner 永不更新到 NaN
- **规避**：`(val > acc) || isnan(val)` 组合（CycleUpdate 形态）；winner 同步指向 NaN 位，反向路由随之正确

### 1.3 反窗口区间与正向窗口公式不同源
- **现象**：反向梯度漏路由 / 重路由（和 ≠ grad_output 和）
- **根因**：反窗口 pStart/pEnd 推导未与正向 start/end 公式互逆核对（span 用 k 而非 `(k−1)·d+1`、pad 项漏项）
- **规避**：两处推导同源核对列入设计检查项（[window-math](../../generic/window-math.md) §4）；梯度守恒用例必测

### 1.4 mask 索引口径漂移
- **现象**：反向 argmax 路由命中率为 0 或乱命中
- **根因**：正向 mask 记录的线性索引含批次/通道偏移（或不含），反向比较时口径不一致；或正反向 int32/int64 分档不齐
- **规避**：口径契约（批次内 `h·W + w`）与 dtype 分档写进两算子规格；跨口径比对必失败

### 1.5 平局 tie-break 与竞品不一致
- **现象**：argmax mask 与竞品逐位不一致（值相同但 winner 位不同），反向梯度位置偏移
- **根因**：更新用 `>=`（末遇）而竞品首遇；或反之
- **规避**：tie-break 规则（严格大于 = 首遇）与竞品逐字核对（[window-math](../../generic/window-math.md) §3）；mask 用例按平局构造数据验证

## 2 精度类

### 2.1 除子口径正反向不一致
- **现象**：avg 反向梯度整体偏倍数（恰差 countIncludePad / 有效面积之比）
- **根因**：反向独立实现除子逻辑，三分档（divisorOverride / countIncludePad / 有效面积）取值与正向不同步
- **规避**：除子取值唯一来源声明（正向定义、反向引用）；三分档用例各建一组正反向对拍

### 2.2 max 全空窗 −Inf 转 0
- **现象**：空窗输出 0 而竞品 −Inf（或反之）
- **根因**：空窗分支"顺手"写 0 / 输出 dtype 不含 −Inf 表示（uint 类）时未定义语义
- **规避**：空窗语义随竞品（−Inf 直写）；uint 输出域在规格中明确空窗值并与竞品核对

## 3 性能类

### 3.1 坐标分解直除热点
- **现象**：反向（inSize 大）耗时异常
- **根因**：元素级 `%` / `/` 直除在坐标分解密集场景开销高
- **规避**：按 `CAP_INT_DIV_COST` 双档结论选路线（反向用 UintDiv 8 参数表先例；正向 256 线程直除可接受）；热点算子分档实测

### 3.2 线程档与负载错配
- **现象**：正向 1024 不升反降 / 反向 256 偏慢
- **根因**：正向窗口循环 + 分支占寄存器（高档 spill）；反向无效扫描靠并行隐藏（低档不够）
- **规避**：正反向不同档是预期形态（256 / 1024，[capability-mapping.md](capability-mapping.md) §2 派生事实）；分档依据写设计文档

### 3.3 分支 B 无效扫描占比失控
- **现象**：mask 命中率低（大窗口少重叠）时反向耗时与窗口重叠度不匹配地涨
- **根因**：反窗口枚举大量 mask 不命中扫描
- **规避**：命中率低于阈值时评估 scatter-to-winner 形态（输出驱动，每窗口一次路由写）——两形态等价性见 [grad-routing](../../generic/grad-routing.md) §4，不得混用

## 4 工程类

- **分支 C 漏 SyncAll / SetScheduleMode(1)**：多段 Process 间原子散射跨核，缺栅栏即数据竞争（max_pool_v3_grad design_peer HIGH×2 实测）；成对检查
- **ATTR 漏注册**：def.cpp OpDef 属性集与 SE REG_OP 声明不齐（avg_pool_update FAIL 实测）；属性清单逐项核对
- **快除参数未同步**：UB 写参数后漏 `DataSyncBarrier` 读旧值
- **空输入核数**：`SetBlockDim(0)` 非法；count=0 仍 1 核短路（分支 C 零初始化不可省）
- **MDE–Code 偏差链**：线程分档、TilingKey 段结构属设计契约，变更回写设计文档（sparse 同款教训）
