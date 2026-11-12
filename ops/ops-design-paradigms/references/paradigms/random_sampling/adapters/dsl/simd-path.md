<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling TF 对标 SIMD 路线（DSL）

> **层级**：CANNBotDSL 语言适配层。对标 TensorFlow 接口族（seed/seed2 属性 + shape 张量，或 stateless key/counter 输入）的 `vf(mode="simd")` 向量实现路线。
>
> **适用域硬边界（R20）**：仅适用 A/C 族（TF 对标）接口族；跨族引用即范式违规（同 [../ascendc/simd-path.md](../ascendc/simd-path.md) 边界条款）。DSL 无仓内基准——本路线为 API 级设计指引，首例须履行 [prng-implementation.md](prng-implementation.md) §6 对拍义务。

## 1 路线特征

| 维度 | 形态 |
|------|------|
| 执行体 | `@kernel` 内 `vf(mode="simd")` 作用域，AIV 寄存器级计算 |
| 随机数生成 | 自研 SIMD Philox（[prng-implementation.md](prng-implementation.md) §1；`vmull/vbitwise_xor/vshl/vshr/vaddc` 轮链） |
| 数据通路 | GM ─mem_copy→ `Buffer(UB)` ─vload→ 寄存器 ─轮链+变换→ 寄存器 ─vstore→ UB ─mem_copy→ GM |
| buffer 预算 | 片上存活节点 P = 流结果暂存（uint32）+ 输出 × 缓冲份数；单块 ≤ UB 预算（248 KB 口径，`get_mem_size("ub")`）÷ P，对齐粒度按实测登记 |
| 线程几何 | 混合形态（`kernel[block_count]` + `vf(mode="simt", thread=N)`）；纯 SIMD 无 thread 维 |

## 2 kernel 骨架

```python
@kernel
def _uniform_simd(seed_key: Tensor, out: Tensor, params: Tensor,
                  block_loops: cb.Constexpr[int], vl: cb.Constexpr[int]):
    buf = Buffer(MemLoc.UB, (PER_BLOCK,), dtypes.uint32)      # 流结果暂存
    out_buf = Buffer(MemLoc.UB, (PER_BLOCK,), dtypes.float32)
    # key/counter 经 params 读出（运行期）或 Constexpr（构图期已知）
    with vf(mode="simd"):
        full = full_mask()
        for blk in range(block_loops):                        # Hardware Loop 纪律（§4）
            philox_round_chain(...)                           # vmull 轮链 → uint32 批量
            vstore(buf, 0, ..., full)
            u = 位拼接/vreinterpret 口径链                     # prng-implementation §3
            vstore(out_buf, 0, ..., mask)
    mem_copy(tile_slice(out, ...), out_buf)                   # 搬出在 VF 外
```

- **VF 内禁 `RingBuffer.next_write()/next_read()` 与 `self._xxx` 成员访问**——槽位与局部变量在进 VF 前备好（compute.md VF 作用域两条铁律）
- 掩码处理非整除尾块：`mask_counter` + `update_mask` 迭代推进（设备循环不支持 `break`，尾块用掩码不用分支）

## 3 状态与 seed 处理

| 输入形态 | DSL 处理 |
|------|---------|
| seed/seed2 **属性**（A/C 族） | 构图期常量——公开入口/构建函数解析属性值，经 `Constexpr` 或 params 传入；**C 族时间熵分支在构图前取值**（host 构图期无运行期时间熵进流） |
| stateless key/counter **张量**（C 族） | 运行期值——kernel 内从 Tensor 读出（SIMD 侧经 UB 中转 vload），构图期不可读 |
| shape 张量（输出规模） | 值依赖输出规模：A 族属性形态在构图期解析；张量形态走 AI CPU 或调用方预解析（同 [../../generic/state-management.md](../../generic/state-management.md) §2 判定） |

offset 回写（A 族递进语义）：本路线 SIMT 段承担或独立顺序 launch（[simt-path.md](simt-path.md) §4 同款）。

## 4 Hardware Loop 纪律（本路线性能命门）

主循环须满足规范否则**静默退化**软件循环 + 阻断 VF 融合：

- 迭代从 0 开始、步长 1；体内无运行时跳转（`if`/三元）；最多四层嵌套
- **"做不做"改"做几次"**：`for _ in cbd.range(has_tail)` 替代尾块分支
- 相邻可融合矢量操作放**同一 VF 作用域**（轮链 + 变换链不落 UB）；需独立同步的拆开
- 指令过多触发 ICache Miss 阻断双发——10 轮 Philox 展开与双发预算的平衡首例实测登记

## 5 正态分布（Box-Muller 的 SIMD 表达）

两段变换（[../../generic/distribution-transform.md](../../generic/distribution-transform.md) §2 同构）：

1. 成对取数：`vdeinterleave`（或下采样 load）拆 u1/u2 → `vmaxs(eps)` → `vlog` → `vmuls` → `vsqrt` 产 `√(−2·ln u1)`
2. `vexp`/三角向量接口（`vexp` 类目存在；sin/cos 表达以 API 面核对为准——**待验证**）→ `vinterleave` 交错写回

成对消耗 c=2 的布局系数 g 按通用层 splitting §2 推导；变换段若需逐元素分支，转**混合形态**（§6）。

## 6 混合形态：SIMD 生成 + 串接 SIMT 变换

同一 `@kernel` 内先 SIMD VF 生成、后 SIMT VF 变换（DSL 混合编程原生形态，对照 AscendC 的 `asc_vf_call` 内嵌）：

```python
@kernel
def _mixed(seed_key: Tensor, out: Tensor, params: Tensor):
    buf = Buffer(MemLoc.UB, (N,), dtypes.uint32)
    with vf(mode="simd"):                     # ① 生成段
        ... vstore(buf, ...)
    with vf(mode="simt", thread=512):         # ② 变换段（逐元素分支/标量数学）
        i = cbd.simt.thread_idx()[0]
        ... out[i] = transform(buf[i])
```

- 数据交接经 **UB Buffer**（SIMT 从 UB 读）；UB 预算按 **216 KB** 口径（SIMT Data Cache 扣减）
- Host 启动：`kernel[block_count](...)`——thread 维在 SIMT VF 参数，**不传 `dim3(thread)`**（launch 形态校验）
- DcacheSize 类扣减：AscendC 的 `config.DcacheSize` 显式配置在 DSL 无对应开关——216 KB 口径即扣减后预算（[capability-mapping.md](capability-mapping.md) §1）
