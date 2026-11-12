<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 与 AscendC 适配层的根本差异：**cannbot-dsl 仓无随机算子样例、无公共底座**（AscendC 侧有 `ops-math/random/random_common/`）。本适配层提供 PRNG 的 DSL 自研实现知识（API 已核对齐备），首例接入须履行流基准对拍义务。权威依据：cannbot-dsl 仓 docs/（开源：https://gitcode.com/cann/cannbot-dsl ；DSL 0.8.0，Ascend 950PR/950DT，SIMT 需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1））。

## 1 定位

本目录承载 RandomSampling 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：PRNG 轮函数的 DSL 表达、双路线（SIMD VF / 纯 SIMT）实现、编译组织、能力真值与陷阱。

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

**无底座声明**：DSL 侧不存在 `random_common` 对应物——`CAP_PRNG_ALG` 的实现载体为**自研**（算法语义仍锁定通用层 Philox4x32-10 流基准）。结构参考见 [example/random_uniform_v2_simd_draft/](example/random_uniform_v2_simd_draft/)（**DRAFT 未验证**，与 AscendC 侧同算子对齐：轮函数/u 构造/offset 递进/host 启动的 DSL 组织骨架 + 待验证清单，语义契约对照 [../ascendc/example/random_uniform_v2_package/](../ascendc/example/random_uniform_v2_package/)）。一切设计产出必须：

1. 与流基准（torch/TF 对标接口的原生序列）**逐位对拍**，不用"自实现正确"当验证
2. 复用本适配层登记的 API 清单与口径，禁止假设与 AscendC 相同的接口行为

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（向量化、线程上限、状态回写等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| PRNG 落地（首选） | [prng-implementation.md](prng-implementation.md) | Philox4x32-10 轮函数的 DSL 表达（SIMD 寄存器算子清单 / SIMT 标量数学）、u 构造口径、常数池、快除 ★ |
| TF 对标 SIMD 路线 | [simd-path.md](simd-path.md) | `vf(mode="simd")` 流水、Buffer/掩码、Hardware Loop 纪律、混合形态（串接 SIMT VF） |
| torch 对标 SIMT 路线 | [simt-path.md](simt-path.md) | 纯 SIMT launch、grid-stride 映射、seed/offset 读取与回写、变长消耗、多段 launch 块协议 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | VF 不可嵌套、Hardware Loop 退化、寄存器/线程权衡、UB 双口径等错题集 |

★ = 新算子接入前必读。

## 4 路线选型表（DSL 版，R20 机械匹配）

接口族判定见 [../../generic/scene.md](../../generic/scene.md)（A/B/C/D 四族）与 [../../patterns.md](../../patterns.md) 场景判定；本表给出 DSL 执行体形态，禁止项与 ascendc 版同源：

| 条件 | DSL 路线 | 理由 | 禁止项（R20） |
|------|---------|------|--------------|
| TF 对标接口（seed/seed2 属性或 stateless key/counter 张量；A/C 族） | **SIMD VF**（`vf(mode="simd")` + Buffer 流水） | 寄存器级批量流推进吞吐高；vmull/vbitwise/vshift 算子齐备 | 禁按 torch 族接口套本路线形态 |
| torch 对标接口（seed/offset 张量、self 填充；B/D 族） | **纯 SIMT**（`vf(mode="simt")` + `dim3` launch） | 每线程独立 key/counter 状态、GM 直写灵活 | 禁按 TF 族接口套本路线形态；流协议事实（轮数/布局）仍可参照 |
| 随机 + 逐元素复合（路由 b） | 按接口族选（同上两行），复合只改编排不改执行体 | 同 ascendc 版 R20 | 禁按复合结构像 elementwise 而套 SIMD 流水 |
| 变换含分支/标量数学（Box-Muller、rejection） | 主路线 + 串接 SIMT VF（混合） | 同一 `@kernel` 内 SIMD VF 与 SIMT VF 串接，UB 交接 | 混合是叠加项，不得以混合为由改主路线 |
| 变长消耗（拒绝式） | SIMT + 预留律 R=256 | 每线程本地配额（通用层 splitting §5） | — |
| 输出超单 launch 覆盖 | 多段 launch + offset 累进（host 编排） | `dim3` block ≤ 65535、thread ≤ 1024 | — |

**设计产出必须在首个设计文档登记两步路由结果**（路线 + 本表依据行），与 ascendc 版义务同款。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md)（DSL：seed/offset 张量属运行期值，构图期不可读——见 simt-path §2） |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；"待验证"项作为假设输出并附验证义务
4. **首例验证义务**：本适配层无 DSL 仓内基准算子——PRNG 实现与 u 构造口径必须与流基准对拍（torch 同接口 `torch.manual_seed` 系 / TF stateless 系）并登记对拍结果，才可作为后续算子的参照
5. 通用层分档常数（R=256、k=4 等）真值以 [capability-mapping.md](capability-mapping.md) §2 派生事实为准
