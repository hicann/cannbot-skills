<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 状态信号算子的设备核实现（AscendC）

> **层级**：AscendC 语言适配层。语言中立模型见 [../../generic/index.md](../../generic/index.md)。
>
> 基准算子：npu_alloc_float_status（SIMD 写零 + DCCI）、npu_get_float_status（SIMT 空实现 + 预编译二进制）、npu_clear_float_status（副作用工程 + SIMT 写零）、sleep（时钟自旋）。均在 ops-nn/control/。

## 1 alloc：分配清零

```cpp
// 输出固定 [8]，GlobalTensor 标量写 + 缓存清理（下次跨核读一致）
GlobalTensor<T> dataGmTensor;
dataGmTensor.SetGlobalBuffer(dataGm, 8);
for (int32_t i = 0; i < 8; i++) { dataGmTensor.SetValue(i, (T)0); }
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dataGmTensor);
```

## 2 get：状态读取（含空实现契约）

**作用域铁律**：状态类 intrinsic（如 `get_overflow_status()`）是**标量作用域**接口，禁止在 `__simt_vf__` 内调用——在 Process（标量作用域）取值，作为标量参数传入 VF：

```cpp
template <typename T>
__aicore__ inline void Process(GM_ADDR addr, GM_ADDR data)
{
    // 能力启用后：int32_t status = get_overflow_status();  ← 只能在这里调用
    __gm__ T* dataGm = (__gm__ T*)data;
    asc_vf_call<NPUGetFloatStatusSimt<T>>(dim3(THREAD_NUM), dataGm /*, status */);
}

__simt_vf__ ... void NPUGetFloatStatusSimt(__gm__ T* dataGm /*, int32_t status*/)
{
    for (uint32_t tid = threadIdx.x; tid < 8; tid += blockDim.x) {
        dataGm[tid] = (T)0.0;      // 空实现：恒写规范零值（幂等）
    }
}
```

- **线程数取最小有效**（8 元素用 32 线程）：多余线程循环条件即退出，全核执行同样逻辑（幂等合法，signal-model I3）
- **预编译二进制形态**：`op_host/config/<soc>/<op>_binary.json` 注册单实例，`shape: [-2]`（任意 shape）+ `FormatAgnostic`——逻辑极简且与 shape 无关时选此形态，免去多 shape 编译

## 3 clear：副作用工程

真实语义是复位硬件溢出标志；平台无 `set_overflow_status` 接口时，**借道向量计算单元的活动**触发复位（V1 行为兼容）：

```cpp
__aicore__ inline void Process(GM_ADDR addr, GM_ADDR data, const TilingData* tilingData)
{
    // ① 副作用段：向量单元活动触发标志复位（值本身无意义，3~8 连续填 6 次）
    TPipe pipe;  TBuf<QuePosition::VECCALC> buf;
    pipe.InitBuffer(buf, VEC_DUP_SIZE * sizeof(half));
    LocalTensor<half> t = buf.Get<half>();
    for (int32_t i = 0; i < 6; ++i) { Duplicate(t, (half)(3 + i), VEC_DUP_SIZE); }

    // ② 数据面段：SIMT 写 [8] 零值输出（副作用载体）
    asc_vf_call<OpNPUClearFloatStatusSimt<T>>(dim3(THREAD_NUM), OUTPUT_SIZE, outputGm);
}
```

- 副作用段参数（触发次数/填充值/缓冲大小）来自 V1 行为对齐——**无接口保证**，代码须注释依据（脆弱性声明，signal-model §3）
- SIMT VF 写 GM 后框架保证 cache 一致性，无需显式 DCCI（与 alloc 的宿主侧写不同）
- 防御性检查：元素数 ≤0 直接返回（负数转无符号会变巨大值导致越界）

## 4 sleep：时序控制

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void SleepSpin(int64_t cycles)
{
    uint64_t startClock = clock();
    uint64_t clockOffset = 0;
    while (clockOffset < (uint64_t)cycles) { clockOffset = clock() - startClock; }
}
```

- **单线程**（LAUNCH_BOUND(1)）：延迟由一个执行体自旋承担
- 输入 `cycles` 从张量读值（int64）；时钟粒度决定精度下界——验证只做下界断言

## 5 共同形态归纳

| 要素 | 取值 | 依据 |
|------|------|------|
| tiling | 常量级（needCoreNum 等极简字段），无切分设计 | 退化 kernel（kernel-placement §3） |
| 线程数 | 最小有效（1/32/128） | 输出量微小 |
| 写语义 | 幂等常量（全核同值） | signal-model I3 |
| 输出缓存 | alloc（宿主侧写）需 DCCI；SIMT VF 写由框架保证 | 两条路径实测差异 |

## 6 退化 tiling 交付契约（设备核路线）

本路线算子无数据面自由度，tiling 退化为固定四步（工作流 `write-tiling-ut` 红基线与 `implement-tiling` constexpr 转录以本契约为**预言机**）：

| 交付项 | 契约 | 依据 |
|--------|------|------|
| TilingData | **空结构体（0 字节）**，仅作 `GET_TILING_DATA_WITH_STRUCT` 载体 | 无切分/缓冲字段需求（kernel-placement §3 退化形态） |
| TilingFunc 四步 | ① `GetCoreNumAiv` → ② **`SetBlockDim(coreNum)`** → ③ memset 空载体 → ④ **`SetTilingKey(DEFAULT)`** | npu_get_float_status 基准 tiling 实证 |
| blockDim = coreNum | 全核执行同值写（幂等 I3），核数只影响冗余度不影响正确性——**不需要**按负载反推 usedCore | signal-model I3 |
| tilingKey | 单值集 `{DEFAULT}`（无场景/dtype 分档） | 无分支自由度 |
| tiling-exception-coverage 清单 | 切分/对齐/缓冲类条目 **N/A**（判定基准：无数据面自由度，spec 证据 = 固定输出 shape + 无数据面输入），非"未覆盖" | 本契约 |

**注册形态二选一（D5 裁决注记）**：
- **标准 TPL 单 key 注册（范式默认，工作流机械链原生支持）**——update-opdef / implement-tiling / kernel 子图任务均有对应口径
- `config/<soc>/<op>_binary.json` 源侧预编译（基准形态：shape `[-2]` + FormatAgnostic 单实例，免多 shape 编译）——**工作流链无先例**，采用时须补齐任务契约映射；设计默认登记为「优化待选、本次不采用」
