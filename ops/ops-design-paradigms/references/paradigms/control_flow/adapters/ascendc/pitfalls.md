<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式陷阱与错题集

> 蒸馏自 FloatStatus 家族与 assert / case_condition 的实现记录。按"现象 → 根因 → 规避"组织。

## 1 正确性类

### 1.1 intrinsic 作用域违规
- **现象**：在 `__simt_vf__` 内调用 `get_overflow_status()` 编译失败或行为未定义
- **根因**：状态类 intrinsic 是标量作用域接口，VF 线程作用域不可达
- **规避**：Process（标量作用域）取值 → 标量参数传入 VF（status-kernel §2 的传参模式）

### 1.2 把 clear 当数据面算子验证
- **现象**：单算子 golden 全过（输出 `[8]` 全零），但硬件标志实际未清——下轮训练依旧误判溢出
- **根因**：clear 的真实效果是副作用，输出只是载体（signal-model I4）
- **规避**：闭环用例 `clear → get == 0`；副作用通道逐平台实测

### 1.3 副作用通道静默失效
- **现象**：clear 代码原样迁移到新平台后不再清标志
- **根因**：借道向量单元活动的通道是工程性行为（V1 兼容），无接口保证
- **规避**：capability-mapping 登记为"逐平台实测"项；迁移必跑闭环用例；代码注释保留依据与脆弱性声明

### 1.4 幂等性破坏
- **现象**：不同核写出不同状态值，输出随机
- **根因**：某核写了非约定值（如把输入数据当输出写）
- **规避**：全执行单元同值写是设计约束（signal-model I3）；review 检查写值来源是否为常量

### 1.5 元素数负值越界
- **现象**：SIMT 循环越界写
- **根因**：`totalElements <= 0` 经 `static_cast<uint32_t>` 变巨大无符号数
- **规避**：防御性早退（clear 实测先例）

## 2 兼容与降级类

### 2.1 空实现被当 bug "修复"
- **现象**：审查者把 get 恒写 0 当缺陷改成读真实状态，平台不支持导致崩溃
- **根因**：空实现是降级兼容契约（I5），有明确注释与接入点
- **规避**：设计文档声明降级态与补齐接入点；capability-mapping 的 ⚠️ 登记联动

### 2.2 消费方未适配降级
- **现象**：训练循环在 950 上永不触发溢出跳步（get 恒 0），掩盖真实溢出
- **根因**：消费方假设状态语义恒可用
- **规避**：降级期间消费方行为（恒不触发）须向调用链声明（signal-model §5）

## 3 工程类

### 3.1 断言失败语义误植设备核
- **现象**：断言型设计出现"kernel 内打印 / 返回错误码中止"，或验证断言写成错误码/日志
- **根因**：设备核无宿主日志/中止通道（kernel-placement §1 负知识），把宿主侧语义误植进 kernel
- **规避**：失败路径信号化——标志输出 + 消费方闭环（[assert-case-kernel.md](assert-case-kernel.md) §1）；摘要打印在宿主侧；验证按标志取值断言

### 3.2 二进制漂移
- **现象**：binary.json 注册的实例与源码行为不一致（改了源码忘更新二进制）
- **根因**：双形态并存无同步机制
- **规避**：变更源码时核对 binary.json 是否需要重出；capability-mapping 列为待验证项

### 3.3 时序等值断言
- **现象**：sleep 用例偶发失败
- **根因**：时钟粒度导致实际延迟略大于请求，等值断言过严
- **规避**：只做下界断言（实际 ≥ 请求，signal-model §5）

### 3.4 case 分支不互斥
- **现象**：case_condition 输出编号与预期不符
- **根因**：分支链条件重叠（同时命中多分支）或未兜底
- **规避**：真值表全覆盖验证（verification §2）；else 兜底强制

### 3.5 标量循环形态触发评审 INCONCLUSIVE（非门控故障）
- **现象**：alloc 形态的 `GlobalTensor::SetValue` 标量循环使 `scalar_mainchain_check` 报 INCONCLUSIVE（转人工核），bringup 期易误判为门控故障
- **根因**：信号算子的固定微小输出走标量循环是合法形态（kernel-placement §3 退化形态），评审脚本按常规算子阈值判定
- **规避**：人工核放行即可；已在本范式登记，评审侧不视为缺陷
