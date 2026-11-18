---
type: CATLASS DSL Programming Concept
title: 显式声明支持域，禁止静默回退
description: 用前置断言声明算子支持域；不得在 host 侧静默回退到 torch/vendor 实现。
tags: [catlass-dsl, host, assertion, fallback, anti-hack]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T12:00:00Z'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T12:00:00Z'}
sources:
  - id: workflow
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: host 侧允许的 torch 调用
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

当 kernel 只支持部分形状/精度时，有两种处理：静默回退到 torch，或显式拒绝。
静默回退会让"融合实现"在未覆盖的形状上退化为 torch 计算，绕过 anti-hack 与性能口径，
也使正确性结论失真。

# 用法

**优先用动态 extent 支持尾块**，而不是用断言把形状挡在外面：

```python
nr = tla.as_numeric(bm)
if m - mi * bm < bm:
    nr = m - mi * bm          # 行尾块
nc = tla.as_numeric(bn)
if v - vi * bn < bn:
    nc = v - vi * bn          # 列尾块
# K 方向尾块按 16 对齐后送入 mmad，并让 fixpipe 使用同样的 tile 语义
```

动态 extent 让算子天然覆盖边界形状；只有在确实无法支持时才前置断言：

```python
assert hidden.dtype == torch.bfloat16 and ids.dtype in (torch.int32, torch.int64)
```

**无论哪种方式，都不得在库内部静默换成 torch/vendor 实现。**

# 约束

- 尾块 extent 必须同时用于搬运、mmad 与 fixpipe，三者对"实际 tile 大小"的认知要保持一致。
- 若确需回退，必须在设计中显式登记为偏差并让 anti-hack 的声明与测量路径一致，
  不得让回退路径悄悄参与性能测量。

接口行为以固定提交的实现为准。[^workflow]

[^workflow]: 固定提交的源码与文档中对应实现。

# 代码模式

见上文 `assert` 片段：入口先校验 shape/dtype，不支持直接报错。

# 失败表现

- 未覆盖形状被静默交给 torch，anti-hack 判 `torch_computation`；
- 同一算子在不同 shape 上走不同实现，性能口径不可比。

# 验证方法

用不支持形状调用，确认得到明确报错而非结果；审计 host 源码中 torch 调用
是否都在允许清单内。
