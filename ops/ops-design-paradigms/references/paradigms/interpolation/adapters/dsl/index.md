<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层基于 **CANNBotDSL SIMT 编程模型**（`vf(mode="simt")`，浮点坐标驱动离散采样场景的标准选择），调度密集场景叠加 **AI CPU metadata 模式**（`@aicpu_kernel` 分核计划外置）。权威依据：cannbot-dsl 仓（开源：https://gitcode.com/cann/cannbot-dsl ，docs/zh/guide/ 与 samples/，DSL 0.8.0，Ascend 950PR/950DT）。

## 1 定位

本目录承载 Interpolation 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：入口与编译组织、坐标分解与采样落地、能力真值与陷阱。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

DSL 与 AscendC 适配层的范式级差异（详见各叶子文档）：

| 差异 | DSL 形态 | 影响 |
|------|---------|------|
| 构图期 host **不能读运行期 GM 值** | 框坐标 / crop_size 类值依赖无法在 host 读取；改走 AI CPU（运行期可读 GM）或入口预解析；kernel 内直读是正解 | [kernel-entrance.md](kernel-entrance.md) §5 |
| 同 stream 顺序 launch 天然有序 | grad 零初始化 + 散射两阶段拆两次 launch（stream 排序免 SyncAll）；单 kernel 内跨段依赖才用 `global_sync_all()` | [kernel-entrance.md](kernel-entrance.md) §4 |
| 无 TilingData/TilingKey 载体 | 元数据经 **params Tensor（GM int64 数组）+ 手工 ABI** 传递；对齐模式等场景分派 = host 侧路径选择 + 独立编译实例（或 `Constexpr` 编译期分支） | [kernel-entrance.md](kernel-entrance.md) §1/§3 |
| 无 `UintDiv` 快除函数族 | 坐标分解 `//` / `%` 直接表达；生成开销未登记，热点场景实测或 params Tensor 手工 magic/shift | [sampling-ops.md](sampling-ops.md) §1 |
| 无 `volatile` 阻断手段 | FMA 收缩行为待实测；加权和分步中间绑定是保守写法 | [sampling-ops.md](sampling-ops.md) §4 |

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（fp64 可用性、FMA 阻断、floor 原语、线程上限等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| Kernel 入口与编译 | [kernel-entrance.md](kernel-entrance.md) | `@kernel`/`@host` 结构、params Tensor ABI、`cb.compile`+`lru_cache`、launch 几何、grad 两段骨架、值依赖处理 |
| 坐标分解与采样 | [sampling-ops.md](sampling-ops.md) | grid-stride、整除分解、对齐模式分派（MapCoord）、加权和分步绑定、补偿路线（simt.fma 误差项）、grad 散射段 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | FMA 收缩、索引防御、launch 形态、构图期读 GM、Constexp 爆炸、UB 预算等错题集 |

## 4 基准样例（cannbot-dsl 仓实测）

本适配层结论蒸馏自以下 cannbot-dsl 样例（路径相对仓根 `samples/`），编码前可对照源码组织方式：

| 样例 | 关键特征 |
|------|---------|
| scatter_elements ★ | 纯 SIMT 数据面、params Tensor ABI、多路径 host 分派、atomic_add、DMA 初始化 + SIMT 散射两段 launch——本范式 grad 散射与 launch/编译组织的直接先例 |
| gather_v2 | 索引驱动离散读、host 分档——IndexedInterpolate 的 gather 段先例 |

★ = 新算子接入前必读对照（SIMT 数据面完整先例）。SIMT 数学 API（`simt.floor` / `simt.fma` / `simt.fmin` / `simt.fmax` / `simt.isfinite`）见 cannbot-dsl 仓 `docs/zh/api/simt/math/`。

**内嵌样例**：与 AscendC 侧同算子对齐的 DSL 结构草稿见 [example/three_interpolate_simt_draft/](example/three_interpolate_simt_draft/)（**DRAFT 未验证**——IndexedInterpolate 算子族骨架：grid-stride 坐标分解、索引 clamp、分步加权和；语义契约对照 [../ascendc/example/three_interpolate_package/](../ascendc/example/three_interpolate_package/)，实现载体按语言）。cannbot-dsl 仓当前没有 Interpolation 类算子样例，草稿填补"从文档到首例"之间的结构空白。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md)（DSL 差异见 kernel-entrance §5） |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md)（DSL 差异见 kernel-entrance §4） |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；标注"待验证"的项（FMA 收缩、除法开销、寄存器分档）作为假设输出并附验证义务
4. 编码前对照 §4 基准样例核实 API 用法；样例的具体参数（线程数、核数）须按当前算子规格重新核定
5. DSL 权威资料：cannbot-dsl 仓（开源：https://gitcode.com/cann/cannbot-dsl ）docs/zh/guide/（编程模型）与 docs/zh/api/（API 详情）；引用接口时以目标版本 API 文档为准
