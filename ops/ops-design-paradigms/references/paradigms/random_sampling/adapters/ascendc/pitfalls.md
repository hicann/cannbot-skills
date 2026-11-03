<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式陷阱与错题集

> 蒸馏自 random 全家桶实现与审查记录。按"现象 → 根因 → 规避"组织。

## 1 正确性类

### 1.1 流区间重叠（重复随机数）
- **现象**：不同输出位置拿到相同随机数；小样本看不出来，长序列统计检验才暴露
- **根因**：执行单元/块的流起点推导违反不变量 I1（如线程映射公式与布局不一致、块 offset 未累进）
- **规避**：流位置公式单点维护（模板内置 `ThreadMappingAndSkip`）；块 offset 只在 host 侧累进；验证跑"改切分方式逐位不变"专项用例

### 1.2 预留不足导致重样
- **现象**：截断正态在限制域小（拒绝率高）时输出重复样本
- **根因**：R 配额 < 最大重采样次数，配额耗尽回绕到已用区间
- **规避**：R 下界按拒绝概率论证（连续拒绝概率 < 2^-40 量级）；边界用例覆盖高拒绝率场景

### 1.3 offset 回写竞争/脏读
- **现象**：连续两次调用的输出序列重叠；或偶发流错位
- **根因**：① 多核同时回写（丢失更新）② 回写早于他核读 offset（缺 SyncAll）③ 缓存未清理，下次调用读到旧值（缺 DCCI）
- **规避**：唯一写者 + 回写前全核栅栏 + `DataCacheCleanAndInvalid`；三约束作为设计检查项（state-management §3）

### 1.4 seed=0 语义误判（两种方向）
- **现象 A**：C 族（TF 语义）seed=0 golden 全 FAIL 但功能正常——时间熵非确定，误用逐位验证
- **现象 B**：B/D 族 seed=0 双跑 bitwise 验证被错杀/公共实现误走时间熵——**seed=0 是合法裸 key**（torch 桥接实测）
- **根因**：把 C 族的"seed=0 → 时间熵"泛化为全局规则
- **规避**：按接口族分型判定（prng-model §1.1 / state-management §1）；双轨条件用"流确定模式判定"而非字面 seed≠0；B/D 族禁用 GetKeyAndCounter 的时间熵分支

### 1.5 key/counter 接口位拆分错位
- **现象**：输出"像随机数"但与竞品完全不一致
- **根因**：TF V2 接口的 `key_data[0]`→key[0..1]、`counter_data[0/1]`→counter 低/高 64 位映射写反
- **规避**：位拆分表逐位核对（simt-path §4）；错位不报错，必须用竞品逐位对齐用例覆盖

### 1.6 值依赖漏配
- **现象**：tiling/infershape 拿不到 shape 张量数据
- **根因**：`TilingInputsDataDependency({0})` / infershape `InputsDataDependency` / def `ValueDepend` 三处缺一
- **规避**：[../../../common/value-depend-process.md](../../../common/value-depend-process.md) 清单逐处核对

### 1.7 数值域开闭不一致
- **现象**：与竞品逐位比对个别位差异
- **根因**：uint32→float 的 (0,1) vs [0,1) 开闭、eps 抬升策略与竞品不同
- **规避**：变换目录逐项登记开闭域与 eps（distribution-transform §1/§4）；竞品逐位对齐专项

### 1.8 漏显式 Philox 轮数（默认 7 ≠ 流基准 10）
- **现象**：流自身可复现但与原生流/竞品不匹配，逐位验收永远红
- **根因**：`PhiloxRandom` 模板默认 7 轮，流基准 10 轮；依赖默认实参
- **规避**：显式 `PhiloxRandom<10>` 写进模板实例；capability-mapping 已登记（R1）

### 1.9 offset 写错 counter 字段（subsequence vs position）
- **现象**：流整体错位，不报错
- **根因**：原生 seed/offset 族 offset 应入 counter[2:3]（subsequence），误入 counter[0:1]（position）
- **规避**：按接口族三型布局表逐位核对（prng-model §1.1，R2/R3）

### 1.10 u 构造路径混用（同流不同值）
- **现象**：与流基准/golden 对拍必红，且流本身复现正常——难以定位
- **根因**：位重解释（[0,1)，SIMD 族/rrelu 基准）与乘法+半步（(0,1)，SIMT 模板族）两条口径混用——同 (key,counter) 下 u 值不同
- **规避**：u 构造路径与流基准同路径（R7 路径一致性，distribution-transform §4）；值域开闭随路径登记

### 1.11 短路口径双源歧义（设计/UT 死锁）
- **现象**：implement-tiling 反复失败——设计文档"ubFactor 置 0"与 UT 预言机"无条件公式填充"冲突，实现按规则拒绝吸收
- **根因**：短路口径 TilingData 填充语义未模板化（rrelu_with_noise 三连败实录）
- **规避**：范式裁决——计数字段清零/置 1，平台派生量无条件公式填充（splitting §4 / R18）；偏离须双侧同步声明

### 1.12 host 侧擅自推进 offset
- **现象**：固定 (seed, offset) 复现链断裂，跨调用流错位
- **根因**：host/adapter 层对 offset 做了"顺手"推进或变换；空 tensor 短路路径推进了流
- **规避**：host 侧 int64 原样透传 + 不下发 kernel 路径流零消耗（state-management §3.1，R15）

### 1.13 uint16 count 静默截断
- **现象**：大块产出量时输出尾部缺失/分布异常
- **根因**：流生成接口 count 形参 uint16，超 65535 静默截断
- **规避**：ubFactor 上界核算（splitting §5，R10）

### 1.14 RNG 暂存 K 份漏算
- **现象**：UB 溢出（运行时踩内存，编译期不报）
- **根因**：复合型 buffer 预算分母漏加 K 份（fp32 暂存折算）
- **规避**：`数据面份数×BUFFER_NUM + K` 公式（common-template §2，R9）

## 2 工程类

### 2.1 混合形态漏配 DcacheSize
- **现象**：SIMD kernel 内 `asc_vf_call` 运行时崩溃或结果错乱
- **根因**：`config.DcacheSize = 0`（默认），未向 SIMT VF 分配片上空间（SetLocalMemorySize 未调用）
- **规避**：kernel 含 simtvf 必须配 DcacheSize（simd-path §6）；capability-mapping 已登记

### 2.2 isNeedSyncAll 与 SetScheduleMode 失配
- **现象**：SyncAll 行为不生效或死锁
- **根因**：tiling 写死/漏写 `SetScheduleMode`（模板经 `isNeedSyncAll` 配置下发）
- **规避**：经 OpTilingConfig.isNeedSyncAll 配置，不手工绕过模板

### 2.3 大张量静默截断
- **现象**：输出规模 > 2^31 时越界/错乱
- **根因**：32bit 索引超限未走块协议，或块数超上限 M=8 未报错
- **规避**：`enableSplitBlocks=true` + 超限显式报错（不截断）；大 shape 边界用例必测

### 2.4 空输出核数
- **现象**：L=0 时 `SetBlockDim(0)` 非法
- **根因**：均分结果 usedUnits=0
- **规避**：模板已内置下限 1（`std::max(1L, ...)`）；自研切分须同款防护

### 2.5 fp16/bf16 转换漏 SetCtrlSpr
- **现象**：半精度输出偶发异常值
- **根因**：尾数拼接路径的半精度旁路控制未包裹
- **规避**：沿用 U32Conversion 公共实现，不自写转换
