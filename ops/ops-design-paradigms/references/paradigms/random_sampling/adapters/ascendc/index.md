<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层与 Sparse（纯 SIMT）不同：**SIMD 与 SIMT 双路线并列**，且有活代码公共底座 `ops-math/random/random_common/`——接入优先于自研。

## 1 定位

本目录承载 RandomSampling 范式中强耦合 AscendC 的知识：公共模板接入契约、双路线实现、能力真值与陷阱。

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ 设计与代码产出
```

## 2 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 模板接入（首选） | [common-template.md](common-template.md) | OpTilingConfig 四回调、TransformFunc/Launcher 契约、新算子 checklist ★ |
| TF 对标 SIMD 路线 | [simd-path.md](simd-path.md) | RandomKernelBaseOp 继承、U32Conversion 尾数技巧、Box-Muller 向量实现、offset 回写、混合形态（内嵌 asc_vf_call） |
| torch 对标 SIMT 路线 | [simt-path.md](simt-path.md) | 双模板选型、魔数快除、变长消耗（rejection）、块协议数据流、离散采样/排列特殊族 |
| 能力真值 | [capability-mapping.md](capability-mapping.md) | 谓词真值表、分档常数池、待验证项 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | 流重叠/回写竞争/预留不足/位拆分错位等错题集 |

## 3 路线选型表

| 条件 | 路线 | 理由 | 禁止项（R20） |
|------|------|------|--------------|
| TF 对标接口（seed/seed2 属性或 stateless key/counter 张量；A/C 族） | SIMD（TQue 流水） | 向量批量产出吞吐高；接口族先例齐 | 禁引 torch/SIMT 系参照（sim_thread_exponential 等）作实现形态参照 |
| torch 对标接口（seed/offset 张量、self 填充；B/D 族） | **SIMT（GM 直写）** | 逐元素变换灵活；线程映射模板成熟 | **禁引 TF/SIMD 系参照（stateless_random_uniform_v2 等）作实现形态参照**；其流协议事实（轮数/布局）仍可参照 |
| 随机 + 逐元素复合（路由 b） | **按接口族选**（同上两行），与复合结构无关 | 复合只改接入方式（自管编排），不改执行体形态 | 禁按"复合结构像 elementwise"而套 elementwise/SIMD 流水形态 |
| 变换含分支/标量数学（sincosf、rejection） | 主路线 + SIMD 内嵌 SIMT（混合） | 变换段交 SIMT VF；注意 DcacheSize | 混合是叠加项，不得以混合为由改主路线 |
| 变长消耗（拒绝式） | SIMT + 预留律 | 每线程本地配额，R=256 | — |
| 输出超 32bit 索引 | 任一路线 + enableSplitBlocks | 块协议（≤8 块，offset 累进） | — |

## 4 基准算子表（22 个，三形态）

| 形态 | 算子 |
|------|------|
| 纯 SIMD（7） | random_uniform_v2 ★最小样例、stateless_drop_out_gen_mask、stateless_random_normal_v2/v3、stateless_random_uniform_v2/v3 |
| 纯 SIMT（11） | sim_thread_exponential ★最小样例（73 行）、drop_out_v3、stateless_bernoulli、stateless_exponential、stateless_normal、stateless_random、stateless_randperm、stateless_sample_multinomial、stateless_truncated_normal_v2（变长消耗）、stateless_uniform、truncated_normal_v2 |
| 混合（3） | random_standard_normal_v2（SIMD Philox + SIMT Box-Muller）、random_uniform_int_v2、stateless_random_choice_with_mask |
| 随机+逐元素复合（路由 b，首例） | rrelu_with_noise（训练态：noise 逐元素作用 + 写回复用；来自 ops-registry-invoke 工作流实录，其缺口分析文档见该 work_dir docs/） |

★ = 新算子接入前必读对照。代码路径按算子名在开源算子仓 [cann/ops-math](https://gitcode.com/cann/ops-math)（`random/<op>/`）检索。

**内嵌样例包**：random_uniform_v2（路由 a 最小样例）已整包镜像至 [example/random_uniform_v2_package/](example/random_uniform_v2_package/)（op_graph/op_host/op_kernel/tests 完整结构，依赖的 `random_common` 公共底座在源仓；provenance 见 [example/README.md](example/README.md)）。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；"待验证"项作为假设输出并附验证义务
4. **按路线判定接入**（common-template §0 判定表）：路由 a → 继承公共底座；路由 b/c → 自管编排 + 复用原子件，对照基准算子核实
5. 公共底座代码在 ops-math/random/random_common/（开源仓真身），本适配层引用其契约与实测值，不内嵌副本
