<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式 AscendC 能力映射

> 通用层 [capability-contract.md](../../generic/capability-contract.md) 谓词在 AscendC（ascend950 / arch35，ops-nn/control 实测口径）下的真值。其他 SoC 须重新核实，禁止外推。

## 1 真值表

| 谓词 | 真值（ascend950 / arch35） | 依据 |
|------|---------------------------|------|
| `CAP_FP_STATUS_READ` | ⚠️ **未启用**——`get_overflow_status()` intrinsic 存在但空实现未调用，get 恒写 0；能力补齐接入点：Process 内取值传标量给 VF | npu_get_float_status（950 空实现注释） |
| `CAP_FP_STATUS_CLEAR_API` | ❌ **不支持** `set_overflow_status` 接口 | npu_clear_float_status（V1 行为注释） |
| `CAP_VECTOR_SIDE_EFFECT_CLEAR` | ✅ 向量单元活动（Duplicate×6，half[38400]）可触发溢出标志复位；**无接口保证，逐平台实测** | npu_clear_float_status 实测 |
| `CAP_SCALAR_SCOPE_INTRINSIC` | ✅ 状态 intrinsic 仅标量作用域可用，`__simt_vf__` 内禁止调用 | npu_get_float_status 设计约束 |
| `CAP_DEVICE_CLOCK` | ✅ `clock()`（返回 uint64 周期），SIMT VF 内可用 | sleep |
| `CAP_ASSERT_SIGNAL_LAYOUT` | 固定 `[1]`（或对齐 `[8]` 布局）× 接口 dtype；通过=0、失败=1（消费方约定，随接口契约登记） | signal-model §6 信号契约（设计约定） |
| `CAP_PREBUILT_BINARY` | ✅ `op_host/config/<soc>/<op>_binary.json`：shape `[-2]` + FormatAgnostic 单实例 | npu_get_float_status_binary.json |
| `CAP_STATUS_LAYOUT` | `[8]` × fp32（元素数 8、float32） | FloatStatus 三算子 |

## 2 派生事实

| 项 | 真值 | 出处 |
|----|------|------|
| get 线程数 | 32（8 元素最小有效档） | npu_get_float_status_simt.h |
| clear 线程数 | 128；副作用段 Duplicate 6 次、half[38400]、值 3~8 | npu_clear_float_status_simt.h |
| sleep 线程数 | 1（LAUNCH_BOUND(1) 自旋） | sleep.cpp |
| 输出缓存 | SIMT VF 写 GM 后框架保证一致性（无需 DCCI）；GlobalTensor 标量写需显式 DCCI | clear / alloc |
| assert summarize | 默认 3（宿主侧摘要条数约束）；DT_STRING 特例随目标接口契约重新声明 | 接口语义 |

## 3 待验证项

| 项 | 验证义务 |
|----|---------|
| `get_overflow_status()` 补齐后的真实行为与传参模式 | 平台升级时以官方 intrinsic 文档 + 单测复核，切换空实现 |
| 副作用通道（向量活动清标志）在其他 SoC 的有效性 | 逐平台闭环实测（clear→get==0），不信任代码复用 |
| `clock()` 时钟粒度与自旋精度 | 迁移平台时重标时序用例的下界断言 |
| 二进制与源码双形态并存时的一致性 | binary.json 更新与源码变更同步核对 |
