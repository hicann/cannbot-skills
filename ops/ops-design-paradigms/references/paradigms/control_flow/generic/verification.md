<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式验证方法

> **层级**：通用知识层（generic/），语言中立。

## 1 验证原则

数据面算子验"输出值"；本范式验**信号契约**：幂等性、副作用可观测性、降级兼容性、失败路径。逐位比对只在极小角色（固定零值输出）。

## 2 用例矩阵

| 用例 | 验证点 | 对应算子类 |
|------|--------|-----------|
| **闭环 clear→get** | 清后读为零（副作用真实生效，I4） | 状态读/清 |
| **幂等并发** | 全执行单元写同值，结果正确（I3） | 状态快照 |
| **读不改状态** | 连续两次 get 同结果 | 状态读 |
| **alloc 初始化** | 输出全零、缓存可见（下次读一致） | 状态分配 |
| **降级契约** | 能力未启用平台：调用不报错、输出规范零值（I5） | 状态读 |
| **条件真短路** | assert 条件真：标志 == 通过码（短路零开销） | 断言 |
| **条件假标志** | 标志 == 失败码，消费方读取后中止调用链（I4） | 断言 |
| **摘要边界** | summarize < 数据量时宿主侧摘要截断；输入含字符串类型的特例路径 | 断言 |
| **case 全分支** | 条件求值的每个离散编号可达、边界值正确 | 条件求值 |
| **时序区间** | 实际延迟 ≥ 请求周期数（只做下界断言，不做等值） | 时序 |
| **图模式** | 框架映射（torch_npu / TF 调用链）与多轮循环（alloc→…→get→clear）序列正确 | 全部 |

## 3 golden 策略

- 固定零值输出：直接常量断言（无需 golden 脚本）
- 条件求值：真值表驱动的参考实现（输入组合 → 期望编号）
- 断言失败路径：以**标志取值**为期望（张量断言），宿主侧摘要内容另行断言
- 状态闭环：序列化用例（多算子编排），单算子 golden 无法覆盖副作用

## 4 平台迁移验证义务

| 项 | 义务 |
|----|------|
| 能力补齐切换 | 真实 intrinsic 启用后：空实现 → 真实现的切换点回归（输出从恒 0 变为真实状态） |
| 副作用通道 | 依赖工程性通道（借道计算单元活动）的算子：每平台实测闭环，不信任代码复用 |
| 时钟粒度 | 时序算子的精度下界按平台时钟重标 |

## 5 验证输出清单

- [ ] 用例矩阵按命中算子类全覆盖
- [ ] 闭环/幂等/降级三不变量各有专项用例
- [ ] 失败路径（断言假）标志取值与消费闭环断言
- [ ] 图模式多轮序列用例
- [ ] 平台迁移验证义务已列入交付
