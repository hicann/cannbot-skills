<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式 DSL 陷阱与错题集

> 依据 cannbot-dsl 官方文档 danger/warning 块与 scatter_elements 实测结构蒸馏。按"现象 → 根因 → 规避"组织。通用层错题（零初始化竞争、唯一索引前提等语言中立项）见 generic 层与 [../ascendc/pitfalls.md](../ascendc/pitfalls.md) 同名条目，此处只列 DSL 特有。

## 1 正确性类

### 1.1 UB 预算双口径算错
- **现象**：开 SIMT 后片上分配溢出/踩内存，或 SIMD 侧白白少用 32 KB
- **根因**：SIMT Data Cache 从 UB 划走 ~32 KB 但**不反映在 `get_mem_size("ub")` 里**；或把 248 KB 又减了一次顶部 8 KB 预留
- **规避**：SIMD 按 `get_mem_size("ub")`（253952）直接当总量；开 SIMT 按 **216 KB** 口径自己留；混用两模式按最紧侧算

### 1.2 构图期读运行期值
- **现象**：host 侧想"先看看 indices 再定分核"，拿不到数据或拿到的是构图期垃圾
- **根因**：DSL host 在**构图期**执行，看不到运行期张量内容（与 AscendC tiling `GetData` 关键差异）
- **规避**：值依赖分核/推断走 `@aicpu_kernel`（运行期 AI CPU 可读 GM）；纯映射逻辑设备侧自算

### 1.3 `dim3` 形态混用
- **现象**：编译失败——"纯 SIMT 不能 `vf(mode="simt", thread=N)` 后再传 `dim3(thread)`"类校验报错
- **根因**：纯 SIMT（`kernel[dim3(block), dim3(thread)]`）与混合（`kernel[block_count]` + VF `thread=N`）两套启动形态不可互换
- **规避**：按 [compute.md「Host 启动规则」] 二选一并全 kernel 统一；scatter_elements 全部纯 SIMT（`cb.dim3(core), cb.dim3(THREAD_COUNT)`）

### 1.4 运行期值进编译期结构
- **现象**：行为错乱或编译报错——把 `params[...]` 读出的运行期标量用于 `range_constexpr` / `const_expr`
- **根因**：运行期值只能作 `range()` 边界、地址计算、设备分支；`const_expr` 需编译期可判
- **规避**：ABI 读出的字段按用途分账；编译期配置走 `Constexpr` 形参

### 1.5 ABI 契约漂移
- **现象**：host 打包的 params 布局与 kernel 下标常量对不上，读出莫名字段
- **根因**：字段下标常量两侧手工约定，改一处漏一处
- **规避**：下标常量定义在一处共享导入（scatter_elements / flash_attn_metadata 同款纪律）；ABI 变更属设计契约，回写设计文档

### 1.6 DCache 跨核脏读
- **现象**：核 A 写的元数据/累加值，核 B 读到旧值
- **根因**：Scalar 侧 Tensor 下标读写经 DCache（多核独立缓存），框架不自动插 dcci
- **规避**：跨核标量交接显式 Cache 操作 + 核间同步；SIMT VF 的 GM 写由框架保证一致性；跨核数据量大改 `Channel(kind=ChannelKind.CrossCore)`

## 2 性能类

### 2.1 Hardware Loop 静默退化
- **现象**：主循环慢数倍且无警告
- **根因**：循环体内运行时 `if`/三元表达式阻碍 Hardware Loop 生成，连带阻断 VF 融合（**静默**，不报错）
- **规避**：能 `const_expr` 消掉的分支优先消；消不掉的提到循环外；"做不做"改"做几次"（`for _ in cbd.range(has_tail)`）；SIMD VF 内禁 `switch/while/do-while`

### 2.2 VF 作用域纪律违例
- **现象**：寄存器溢出 / VF 融合被阻断
- **根因**：VF 区域内调 `Channel.produce()/consume()`、或访问 `self._xxx` 成员变量
- **规避**：进 VF 前选好槽位取成局部变量；搬运/交接在 VF 外完成

### 2.3 线程数与寄存器错配
- **现象**：SIMT kernel 性能反而低于小线程数版本
- **根因**：SIMT 寄存器档位——线程数翻倍、每线程寄存器减半；计算密集在大线程数下溢出到 GM 栈
- **规避**：旋钮是**线程数**（SIMD 侧是 VF 长度）；分档实测（512 惯例起步，行级状态类降档）

### 2.4 栅栏进内层循环
- **现象**：`global_sync_all()` 后性能崩塌
- **根因**：一次调用 = 模式2→0→0→2 四次同步，代价高
- **规避**：两阶段优先拆**顺序双 launch**（stream 排序免栅栏）；必须单 kernel 时栅栏放循环外、按轮批量同步

## 3 工程类

- **`Constexpr` 泛滥**：每个 Constexpr 进缓存键，变体数 = 各维度乘积——只编码"影响代码生成"的量
- **空输入 launch**：`dim3(0)` 非法；`N=0` 时 host 直接不 launch 或短路路径（对照 generic 短路口径）
- **核数写死**：经 `get_platform_info()` / `torch.npu.get_device_properties` 查询；分档出货（32/28 核等）硬编码必错
- **CANN 版本**：SIMT（含混合）需 CANN ≥ 9.2.0；目标环境低于此版本时仅 SIMD 路线可用，设计须声明
- **AOT 登记漏文件**：Native 打包时 metadata/AI CPU 侧 `.py` 须与主算子一并列入 `operator_groups.toml`，漏了 wheel 缺模块
