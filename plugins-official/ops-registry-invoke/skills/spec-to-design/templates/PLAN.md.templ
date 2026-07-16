---
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
iteration_count: 3
iterations:
  - id: 1
    goal: "骨架搭建"
    wave1:
      a1_main:
        scope: "单 TilingKey 骨架"
      a1_p:
        - task_name: "probe_tiling1"
          tiling_key: "TilingKey_1"
          dtype: "float16"
          memory_strategy: "single_buffer"
      b:
        scope: "L0 基础用例"
    wave2:
      a2:
        scope: "核心路径 UT"
    acceptance:
      ut: "核心路径 UT 通过"
      st: "L0 用例覆盖完整"

  - id: 2
    goal: "策略整合"
    wave1:
      a1_main:
        scope: "整合迭代一穿刺结果 → 多 TilingKey 实现"
      a1_p:
        - task_name: "probe_full_dtype"
          tiling_key: "TilingKey_2"
          dtype: "float32"
          memory_strategy: "double_buffer"
      b:
        scope: "C++ 多 shape 用例"
    wave2:
      a2:
        scope: "Tiling 分支 UT 覆盖"
    acceptance:
      ut: "Tiling 分支 UT 覆盖达标"
      st: "多 shape 用例通过"

  - id: 3
    goal: "全量覆盖"
    wave1:
      a1_main:
        scope: "全功能实现"
      a1_p: []
      b:
        scope: "C++ 全量用例"
    wave2:
      a2:
        scope: "全覆盖 UT"
    acceptance:
      ut: "UT 全覆盖且无回归"
      st: "全部必需用例已执行"
---

# {{ op.name }} 迭代执行计划

本模板用于生成 `operators/{{ op.name }}/docs/PLAN.md`。最终内容必须使用简体中文，并与 `DESIGN.md` 的模板划分、TilingKey 和 spec.yaml 字段保持一致。

**YAML frontmatter 规范**：
- `iteration_count`：迭代总数，取值 ∈ {1, 2, 3}
- `iterations`：迭代列表，长度必须等于 iteration_count
- 每个迭代包含 `id`、`goal`、`wave1`、`wave2`、`acceptance` 字段
- `wave1.a1_p`：穿刺任务列表，空列表表示无穿刺
- `wave1.a1_main.scope`、`wave1.b.scope`、`wave2.a2.scope`：任务范围描述
- `acceptance.ut`、`acceptance.st`：验收标准

**Markdown 正文规范**：
- 正文章节根据 iteration_count 动态生成
- iteration_count = 1：只生成"全覆盖目标"和"穿刺结果判定"
- iteration_count = 2：生成"迭代一穿刺列表"、"迭代二整合与全覆盖目标"和"穿刺结果判定"
- iteration_count = 3：生成全部章节（保持向后兼容）

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 基于 spec.yaml 生成初始迭代计划 | YYYY-MM-DD | [gitId] |

<!-- 以下章节根据 iteration_count 动态生成 -->

<!-- BEGIN iteration_count >= 2 -->
## 迭代一穿刺列表

目标：完成单 dtype、核心 shape、核心 TilingKey 的骨架搭建，并并行穿刺其他关键 TilingKey。

| 任务类型 | TilingKey | Dtype | Shape/场景 | Memory Strategy | 验收点 |
| --- | --- | --- | --- | --- | --- |
| 主线 | TilingKey_0 | 从 spec 选择主要 dtype | 核心正常 shape | 明确 UB/workspace 策略 | 编译通过 + L0 核心用例通过 |
| 穿刺1 | TilingKey_1 | 同主线 dtype | 关键边界或广播场景 | 明确 UB/workspace 策略 | 穿刺结果可复用或形成风险结论 |
<!-- END iteration_count >= 2 -->

<!-- BEGIN iteration_count = 3 -->
## 迭代二整合目标

说明迭代一成功穿刺项如何整合到主线，实现多 TilingKey 和单 dtype 多 shape 覆盖。

## 迭代二穿刺列表

目标：提前验证迭代三的难点，例如全 dtype、复杂边界、广播、动态 shape 或性能路径。

| 任务类型 | 验证目标 | 输入规格 | dtype | 预期结论 |
| --- | --- | --- | --- | --- |
| 穿刺1 | 全 dtype 或关键边界 | 来自 spec.yaml | 来自 supported_combinations | 成功/风险/需修订 |
<!-- END iteration_count = 3 -->

<!-- BEGIN iteration_count >= 1 -->
<!-- iteration_count = 2 时生成"迭代二整合与全覆盖目标"；iteration_count = 3 时生成"迭代三全覆盖目标" -->
## 迭代{N}全覆盖目标

| 覆盖维度 | 内容 | 来源 | 说明 |
| --- | --- | --- | --- |
| 全 dtype | 来自 `dtype_policy.supported_combinations` | spec.yaml | 覆盖所有输出 dtype |
| shape/broadcast | 来自 `outputs[].shape_rule` 和 `broadcast` | spec.yaml | 覆盖正常、边界和广播 case |
| boundary/extreme | 来自 `boundary_conditions` 和 `extreme_inputs` | spec.yaml | 覆盖有效边界和数值极端输入 |
| precision | 来自 `numerical_tolerance.per_dtype` | spec.yaml | 验收标准与 TEST.md 一致 |
<!-- END iteration_count >= 1 -->

## 穿刺结果判定

| 状态 | 判定标准 | 后续处理 |
| --- | --- | --- |
| 成功 | 编译通过，核心用例通过，设计假设成立 | 复用到主线 |
| 部分成功 | 部分 case 通过，存在明确限制 | 记录限制，调整后续迭代范围 |
| 失败 | 编译失败、精度失败或资源预算不可行 | 归档问题，下一迭代重做或回到方案设计修订 |
