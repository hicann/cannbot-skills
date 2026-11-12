<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling torch 对标 SIMT 路线（DSL）

> **层级**：CANNBotDSL 语言适配层。对标 PyTorch 接口族（seed/offset 张量、self 填充语义）的纯 SIMT 实现路线；也是变长消耗（拒绝式）算子的标准路线。
>
> **适用域硬边界（R20）**：仅适用 B/D 族（torch 对标）接口族；跨族引用即范式违规。形态先例：`samples/scatter_elements/`（纯 SIMT 数据面完整样例，非随机算子但 launch/映射/原子/回写骨架同构）；PRNG 本体见 [prng-implementation.md](prng-implementation.md) §2。

## 1 路线特征

| 维度 | 形态 |
|------|------|
| 执行体 | 纯 SIMT：`@kernel` 内 `vf(mode="simt")`，host `kernel[dim3(block), dim3(thread)]` 启动 |
| 随机数生成 | 每线程本地 key/counter 状态 + `cb.simt` 标量数学轮链（自研） |
| 数据通路 | GM 直写（Tensor 下标赋值），变换即写；无 RingBuffer 流水 |
| 线程几何 | thread ≤ 1024/block、block ≤ 65535；512 线程/核惯例起步（寄存器档位权衡） |
| UB 口径 | 216 KB（SIMT Data Cache 扣减后） |

## 2 线程映射与流位置

```python
with cb.vf(mode="simt"):
    tid  = cbd.simt.thread_idx()[0]
    blk  = cbd.simt.block_idx()[0]
    bdim = cbd.simt.thread_dim()[0]
    gdim = cbd.simt.grid_dim()[0]
    g = cbd.simt.cast(blk * bdim + tid, dtype=dtypes.int64)     # 全局线程号
    total = cbd.simt.cast(gdim * bdim, dtype=dtypes.int64)
    for out_idx in range(g, output_size, total):                # grid-stride
        stream_pos = map_to_stream(out_idx, magic, shift)       # 连续/离散双布局公式
        ...                                                       # 本线程流区间生成 + 变换 + 直写
```

- 映射公式（连续/离散双布局）按 [../../generic/splitting.md](../../generic/splitting.md) §2；`i / totalThreads` 用 host 预计算 magic/shift 快除（[prng-implementation.md](prng-implementation.md) §4）
- seed/offset 张量是**运行期值**——kernel 内从 Tensor 读（构图期不可读）；offset→counter 起点换算 `⌈offset/4⌉` 设备侧自算
- **接口位拆分逐位核对**：key/counter 的 64 位拆装（`counter_data[0]→counter[0..1]` 低 64 位等，prng-model §1.1）——错位不报错但序列全错，对拍锁死

## 3 self 填充语义（B/D 族）

输出张量可能预填 self 值：直写覆盖（`out[i] = transform(...)`）语义不变；`masked_fill` 型（drop_out_v3）按 `u < p` 写掩码/填充值——无原子、无回读。

## 4 状态回写（offset 递进协议）

[../../generic/state-management.md](../../generic/state-management.md) §3 时序约束的 DSL 落地：

```python
# 方案 A（首选）：回写拆独立顺序 launch——stream 排序保证读-改-写不在生成前发生
@host
def run(seed: Tensor, offset: Tensor, out: Tensor, ...) -> None:
    _read_offset_and_generate[dim3(cores), dim3(512)](...)      # ① 各核读 offset
    _writeback_offset[dim3(1), dim3(1)](...)                    # ② 唯一写者递进（顺序保证）

# 方案 B：单 kernel 内 global_sync_all() 后唯一线程写 + cbd.simt.threadfence
```

- 写者唯一性：`block_idx==0 and thread_idx==0`（或尾核，算子内一致）
- 递进量按消耗协议（`L × R` 或调用契约量，rrelu_with_noise 型 12/调用同源）；**不下发 kernel 的路径不得推进流状态**（R19）

## 5 变长消耗（拒绝式）与预留律

- 预留律 R=256：每线程本地配额，流位置 = `counter₀ + groupIndex × R`；重试上限 R/c，溢出即重样 bug
- 拒绝分支（`if |z| < 2.0`）在 SIMT 天然合法（每线程独立控制流）——与 SIMD 路线的掩码化不同，无需改写
- 离散采样/排列特殊族（multinomial/randperm）：SIMT + 前缀和/排序手段借用相邻范式（同 [../ascendc/simt-path.md](../ascendc/simt-path.md) §6 边界条款），随机流管理仍按本范式

## 6 块协议（超单 launch 覆盖）

输出规模 > block×thread 覆盖效率区间（或 32bit 快路径缺失）时，host 多段 launch：

- host（构图期或经 params 运行期）划分 `splitBlocks[≤M]`：每段 GM 偏移 + 流起点（kernelOffset 累进，prng-model §5）
- 各段独立 launch、**块间无同步**——GM 偏移与流区间双重隔离；段内 grid-stride 收尾
- M 上限自定（capability-mapping `CAP_SPLIT_BLOCK_MAX` 待验证项），首例登记

## 7 编译与 launch 检查单

1. `@host` 函数体内直接 `kernel[dim3(block), dim3(thread)](...)`——**纯 SIMT 形态**，禁与混合形态（`kernel[block_count]` + `vf(thread=N)`）混用（编译器校验）
2. `cb.compile` + `lru_cache` 缓存键含（shape、dtype、core 数、thread 数、全部 Constexpr）——thread 数进键（AscendC 侧线程随 TilingKey 分档的等价物）
3. `N=0` 不 launch；核数经平台 API 查询（`vector_core_num`），有用核数截断同 Sparse 范式 DSL 适配层 [kernel-entrance.md](../../../sparse/adapters/dsl/kernel-entrance.md) §6
4. CANN ≥ 9.2.0（SIMT 前提，cannbot-dsl 仓配套 9.2.1）；目标环境低于此版本时 B/D 族无可用路线，设计须声明
