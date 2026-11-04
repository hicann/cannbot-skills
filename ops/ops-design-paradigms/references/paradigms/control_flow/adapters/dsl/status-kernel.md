<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 状态信号算子的设备核实现（DSL）

> **层级**：CANNBotDSL 语言适配层。信号模型见 [../../generic/signal-model.md](../../generic/signal-model.md)；落核判据见 [../../generic/kernel-placement.md](../../generic/kernel-placement.md)。
>
> 基准：AscendC 侧 FloatStatus 家族 / sleep（ops-nn/control/）的语义契约；DSL 形态按 cannbot-dsl 编程模型（compute.md / hardware-model.md）落地。**DSL 仓无 ControlFlow 类样例**——本文件是 API 级设计指引，首例须实测收敛。

## 1 alloc：分配清零

```python
@kernel
def _alloc_zero(output: Tensor):
    with cb.vf(mode="simt"):
        tid = cbd.simt.thread_idx()[0]
        if tid < 8:                       # [8]×fp32 固定契约（CAP_STATUS_LAYOUT）
            output[tid] = 0.0             # SIMT VF 写 GM，框架保证一致性

@host
def run_alloc(output: Tensor) -> None:
    _alloc_zero[cb.dim3(1), cb.dim3(32)](output)
```

- 最小 launch（1 block × 32 thread）；幂等常量写（signal-model I3）
- 也可整块 `mem_copy` 清零（DMA 路线）；两形态等价，取实现简者

## 2 get：状态读取（空实现契约）

**降级态是唯一形态**：DSL API 面未暴露 FP 异常状态读取（[capability-mapping.md](capability-mapping.md) §1）——与 AscendC 950 侧"未启用"同型：

```python
@kernel
def _get_zero(data: Tensor, output: Tensor):
    with cb.vf(mode="simt"):
        for i in range(cbd.simt.thread_idx()[0], 8, cbd.simt.thread_dim()[0]):
            output[i] = 0.0               # 恒写规范零值（幂等，I3）
```

- 设计契约同 AscendC 版：输出**不是**硬件状态的承诺；能力补齐（DSL 未来暴露状态 intrinsic）后切换真实读取，接入点 = kernel 内取值传标量进 VF
- 验证断言只做"全零 + 形状/契约"，不做状态语义

## 3 clear：副作用工程

真实语义是复位硬件溢出标志；无复位接口时借道**向量单元活动**（V1 行为兼容，参数无接口保证——脆弱性声明同 AscendC 版）：

```python
@kernel
def _clear_side_effect(data: Tensor, output: Tensor, dup_size: cb.Constexpr[int]):
    buf = Buffer(MemLoc.UB, (dup_size,), dtypes.float16)
    with cb.vf(mode="simd"):              # ① 副作用段：向量活动（值无意义）
        m = full_mask()
        for i in cbd.range_constexpr(6):
            vstore(buf, 0, vdups(...), m) # 触发向量单元活动（形态参数按 V1 对齐，逐平台重论证）
    with cb.vf(mode="simt"):              # ② 数据面段：写 [8] 零值
        for i in range(cbd.simt.thread_idx()[0], 8, cbd.simt.thread_dim()[0]):
            output[i] = 0.0
```

- 两 VF 串接（同核顺序执行）；混合形态启动：`kernel[block_count](...)`，SIMT VF 带 `thread=N`
- 副作用段参数（触发次数/填充值/缓冲大小）来自 V1 行为对齐，**代码注释依据**（同 AscendC 义务）；DSL 的指令序列与 AscendC `Duplicate` 不同——**不可平移实测结论**，逐平台闭环（clear→get==0）
- 防御性检查：元素数 ≤ 0 直接返回

## 4 sleep：时序控制

```python
@kernel
def _sleep_spin(cycles: Tensor):
    with cb.vf(mode="simt"):
        if cbd.simt.block_idx()[0] == 0 and cbd.simt.thread_idx()[0] == 0:   # 单线程
            start = get_system_cycle()                    # 位置约束待验证（capability-mapping）
            while get_system_cycle() - start < cycles[0]: # 自旋（SIMT 内 while 合法）
                pass
```

- **单线程承担延迟**（1 block × 1 thread launch）
- `cycles` 从张量读值（int64，运行期）；时钟粒度决定精度下界——验证只做下界断言
- VF 内 `while` 在 SIMT 模式合法（SIMD VF 禁 while/do-while——路线差异注意）

## 5 退化形态的 DSL 交付契约

对照 AscendC 版"退化 tiling 交付契约"（[../ascendc/status-kernel.md](../ascendc/status-kernel.md) §6）——DSL **无 tiling 概念**，契约退化为三点：

| 交付项 | 契约 |
|--------|------|
| launch 几何 | 常量（`dim3(1)`×最小有效线程或单 block 多核幂等写）；核数只影响冗余度不影响正确性（I3）——**不需要**按负载反推 |
| 编译实例 | 单实例（无场景/dtype 分档自由度）；`cb.compile` + `lru_cache` 缓存键仅含输出契约形状 |
| 形态注册 | 标准 JIT（默认）；发布走 AOT `@export` + `aot.build`（对应 AscendC binary.json 单实例形态——见 capability-mapping `CAP_PREBUILT_BINARY`） |

**验证义务**：tiling-exception-coverage 类清单 N/A（无数据面自由度，判定同 AscendC 版——非"未覆盖"）。

## 6 共同形态归纳

| 要素 | 取值 | 依据 |
|------|------|------|
| 执行体 | SIMT VF 幂等写 / SIMD VF 副作用段 | signal-model I3/I4 |
| 线程数 | 最小有效（1/32） | 输出量微小 |
| 写语义 | 幂等常量（全核同值） | I3 |
| 输出缓存 | SIMT VF 写由框架保证 | hardware-model.md |
| 失败路径 | 无（写规范值，不失败）——断言类失败语义走标志化路线 | [assert-case-kernel.md](assert-case-kernel.md) §3 |
