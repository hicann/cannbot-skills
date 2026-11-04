<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 断言/条件求值信号算子的设备核实现（DSL）

> **层级**：CANNBotDSL 语言适配层。信号模型见 [../../generic/signal-model.md](../../generic/signal-model.md)；形态选型见 [../../generic/kernel-placement.md](../../generic/kernel-placement.md) §2。
>
> 覆盖信号类型：**断言判定**（assert 型）、**条件求值**（case_condition 型）。实现形态为**设备核（AI Core）SIMT VF**——读微量输入 → 离散输出/状态标志（与 [status-kernel.md](status-kernel.md) 的退化形态同族）。DSL 形态按 cannbot-dsl 编程模型（compute.md）落地；**DSL 仓无同类样例**，本文件是 API 级设计指引，首例须实测收敛。
>
> **负知识（设备核边界）**：DSL kernel 无宿主级日志/错误中止对应物。断言失败路径必须**信号化**——kernel 写失败标志输出，上报与中止由消费方（宿主/下游）读取标志后执行，摘要打印在宿主侧 Python 完成。

## 1 assert 型：断言判定 → 状态标志

语义形态：控制面信号生产者——条件真写通过标志，条件假写失败标志；幂等常量写（I3）。

```python
@kernel
def _assert_signal(cond: Tensor, output: Tensor):
    with cb.vf(mode="simt"):
        if cbd.simt.thread_idx()[0] == 0:   # 单线程：单点读 + 单点写
            if cond[0] != 0:
                output[0] = 0.0             # ① 条件真：通过标志（短路零开销）
            else:
                output[0] = 1.0             # ② 条件假：失败标志，宿主消费后中止

@host
def run_assert(cond: Tensor, output: Tensor) -> None:
    _assert_signal[cb.dim3(1), cb.dim3(1)](cond, output)   # 最小 launch（1×1）
```

要点：

- **条件真短路**：成功路径零额外开销
- **失败路径 = 标志 + 消费闭环（I4）**：验证双断言——条件假 `output[0] == 失败码`、条件真 `output[0] == 0`；消费方（读标志中止调用链）行为向调用方声明
- **摘要打印移宿主侧**：失败详情由宿主 Python 读取输入张量生成（`summarize` 截断约束在宿主实现），kernel 不承担
- **无 tiling 概念**：常量几何 launch（`dim3(1)×dim3(1)`），交付契约三点制见 [status-kernel.md](status-kernel.md) §5

## 2 case_condition 型：条件求值 → 离散编号

```python
@kernel
def _case_condition(ijk: Tensor, output: Tensor):
    with cb.vf(mode="simt"):
        if cbd.simt.thread_idx()[0] == 0:   # 单线程：三元素读 + 分支 + 单点写
            i, j, k = ijk[0], ijk[1], ijk[2]
            if i >= k and j < k:
                output[0] = 0
            elif i == k and j == k:
                output[0] = 1
            # ...                            # 离散编号枚举（互斥分支链）
            else:
                output[0] = CASE_DEFAULT     # else 兜底
```

- **分支链互斥完备**（else 兜底）：任何输入落进唯一编号——真值表驱动验证（[../../generic/verification.md](../../generic/verification.md) §2）
- **运行期真值**：构图期不可读的输入由 kernel 运行期读 GM——设备核路线的本职场景
- Host 参考取值（构图期可读的常量条件）不必进 kernel——那是普通常量折叠，不属本范式信号算子

## 3 与设备核状态类的形态对齐

| 要素 | assert 型 | case_condition 型 | FloatStatus get |
|------|-----------|--------------------|-----------------|
| 执行体 | SIMT VF 单线程 | SIMT VF 单线程 | SIMT VF 幂等写 |
| launch | `dim3(1)×dim3(1)` 常量 | 同 | 常量（1×32） |
| 输出 | `[1]` 通过/失败标志 | `[1]` int32 编号 | `[8]` 规范零值 |
| 写语义 | 幂等常量（I3） | 幂等（输入确定 → 编号确定） | 幂等常量 |
| 失败上报 | 标志 + 宿主消费闭环（kernel 无宿主上报对应物） | 无失败语义 | 无（写规范值，不失败） |
| 编译 | `cb.compile` 单实例（缓存键仅输出契约形状） | 同 | 同 |
