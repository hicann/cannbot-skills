<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式 DSL 陷阱与错题集

> 通用层负知识（空实现契约、副作用无接口保证、断言失败信号化等语言中立项）见 generic 层与 [../ascendc/pitfalls.md](../ascendc/pitfalls.md) 同名条目，此处只列 DSL 特有。依据 cannbot-dsl 官方文档 warning 块。

## 1 正确性类

### 1.1 `lru_cache` 套在 DSL 装饰器上
- **现象**：`@kernel` / `@host` 行为异常或缓存失效
- **根因**：`lru_cache` 只可包**普通 Python 函数**（如编译获取函数），不可套 DSL 装饰产物
- **规避**：`_get_compiled()` 普通函数 + `@lru_cache(maxsize=1)`，装饰器产物在其内产出

### 1.2 状态能力假设平移
- **现象**：把 AscendC 侧"向量活动可清标志"的实测结论直接用于 DSL
- **根因**：副作用通道无接口保证，且 DSL 生成的指令序列与 AscendC `Duplicate` 不同
- **规避**：DSL 侧逐平台闭环实测（clear→get==0），不信任跨语言平移

### 1.3 `get_system_cycle` 位置越界
- **现象**：编译失败或读数异常
- **根因**：周期读取 API 的 VF 作用域约束未核对（待验证项）
- **规避**：sleep 型设计前 API 文档 + 编译实测；不可用时降级为循环空转口径并声明

## 2 设计一致性类

### 2.1 空 tiling 惯性套用
- **现象**：给 DSL 算子设计"空 TilingData / TilingFunc 四步"——无从落地
- **根因**：DSL 无 tiling 概念，把 AscendC 退化交付契约当语言契约
- **规避**：DSL 交付契约三点制（[status-kernel.md](status-kernel.md) §5：常量 launch 几何 / 单编译实例 / JIT 或 AOT 形态）

### 2.2 断言失败语义想当然
- **现象**：assert 型算子设计/验证里出现"kernel 抛异常中止"或"错误码断言"
- **根因**：DSL 设备核路线无错误码/异常中止语义，失败载体是**状态标志输出**
- **规避**：失败路径信号化——标志 + 宿主消费闭环（[assert-case-kernel.md](assert-case-kernel.md) §1）；验证按标志取值断言

## 3 性能类

### 3.1 SIMT Data Cache 侵蚀退化 kernel 预算
- **现象**：副作用段 + SIMT 段的 UB 分配超预期
- **根因**：开 SIMT 后 UB 按 216 KB 口径（Data Cache ~32 KB 不在 `get_mem_size` 返回内）
- **规避**：混合 kernel 按 216 KB 核算；副作用缓冲（dup_size）按 V1 对齐参数核定

## 4 工程类

- **`cbd.print` 覆盖内建**：设备打印写 `cbd.print`（仅调试用，生产禁用）；构图期打印用内建
- **CANN 版本**：SIMT VF（clear 数据面段）需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
