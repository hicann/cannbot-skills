<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 CANNBotDSL 能力映射

> **层级**：CANNBotDSL（下文简称 DSL）语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 DSL 真值与 API 实体。
>
> **真值状态声明**：已实测证实的能力谓词填 API 实体并标注来源（`cb.simt` / `ops.*`）；尚未实测的标**待验证**。DSL 有 SIMT VF（`cb.vf(mode="simt")`），其线程/block/grid 模型与 AscendC SIMT 高度对应，见 §1 各行。待验证谓词的影响与收敛规则见 §3。

## 1 谓词真值表

| 谓词 | DSL 真值 | 依据 / 验证义务 |
|------|------|------|
| `WARP_SIZE` | 32 | DSL SIMT `cb.simt.warp_size()` 实测返回 32（Ascend950PR，`cannbotdsl-0.8.0-cp313`）；warp 级原语 `cb.simt.lane_idx()/shfl/ballot/reduce_add` 均存在。本范式骨架不依赖 warp 级原语，lane 划分暂不落地 |
| `THREAD_COUNT_PER_BLOCK` | ✔（`cb.vf(mode="simt", thread=N)` + `cb.simt.thread_dim()`） | 每 block（核）的 SIMT 线程数由 `cb.vf(mode="simt", thread=N)` 启动时给定、`cb.simt.thread_dim()` 读取（默认 `thread=256`，按性能测试结果调整）。对应通用层「核内 blockDim 跨步」 |
| `GRID_DIM` | ✔（`get_block_idx`/`get_block_num` 已实测） | 核数/核索引由 `ops.arch.get_block_idx/get_block_num` 提供（已实测）。subblock（`get_subblock_id/dim`）是 **cube 相关**的核内子块概念，本范式（纯 Vector SIMT 计数）不依赖；核内并行单位是 SIMT thread（见 §4 规则 6 修订） |
| `LOCAL_HIST_ATOMIC` | ✔（`cb.simt.atomic_add(ptr, v)`，UB 地址） | 片上 Buffer 原子加：`cb.simt.atomic_add(self.local_hist.ptr(off), v)`（0.7.0 前端确认 `MemLoc in (GM, UB)`，已实测 int32/float32）。对应 ascendc 的 `asc_atomic_add(__ubuf__)` |
| `LOCAL_HIST_CLEAR` | ✔（SIMT 线程跨步清 0） | `cb.vf(mode="simt")` 内 `for i in range(thread, bins, THREAD_COUNT_PER_BLOCK): local_hist[i] = 0`（已实测；随后须 `` `cb.simt.syncthreads()` ``） |
| `LOCAL_HIST_FLUSH` | ✔（`mem_copy(dst, src, atomic_add=True)`） | UB→GM 原子归并：`mem_copy(hist[start:start+bins,], local_hist[0:bins,], atomic_add=True)`（已实测）。注意 1-D 切片须用单元素元组 `[a:b,]`（shape tree 约束，见 §4 规则 9） |
| `GLOBAL_MERGE_ATOMIC` | ✔（`scalar.atomic_add(ptr, v)`，GM 地址） | GM 逐元素原子加：`cannbotdsl.ops.scalar.atomic_add(hist.ptr(idx), v)`（已实测 int32/float32） |
| `THREAD_BARRIER` | ✔（`cb.simt.syncthreads()`） | SIMT VF 内线程栅栏：`cb.simt.syncthreads()`（清 0 与累加之间逐批复用 Buffer，已实测） |
| `MEMORY_FENCE` | ✔（`cb.simt.threadfence()` / `cb.simt.threadfence_block()`） | SIMT 内存/线程栅栏已实测编译+运行（Ascend950PR）。片上写对后续搬运/原子操作的可见性由 DSL 编译器自动同步（`` `VECTOR_MTE3_SYNC` ``）或显式 fence 保证 |
| `VECTOR_MTE3_SYNC` | ✔（SIMT VF 边界 + 编译器自动同步） | DSL 无显式 VECTOR→MTE3 跨引擎同步原语：SIMT VF 内 `cb.simt.syncthreads()` 保证核内累加完成，退出 SIMT VF 后 `mem_copy` 由 DSL 编译器自动插入跨引擎同步（已实测无竞态，无需显式 SetFlag/WaitFlag） |
| `VECTOR_LOAD_WIDTH` | 256 字节（`cb.get_vf_len()`，dav-3510 向量寄存器宽度） | 单次向量化宽度 = 256 字节 = 2048 bits（实测 `get_vf_len()`=256；）。单次元素数 = 256 / sizeof(T)（int8→256、fp16→128、int32/fp32→64、int64/fp64→32）。SIMT 侧经 `cb.simt.ShortVector`/`l2cache_vload` 向量加载。通用层 [../../generic/thread-mapping.md](../../generic/thread-mapping.md) §5 向量化映射按此落地 |
| `ONCHIP_HIST_CAPACITY` | **可用预算 221184 字节**（dav-3510） | `cb.get_mem_size("ub")` 返回的是 UB **总容量** 253952 字节（248 KB；L1 = 524288 = 512 KB），但 SIMT kernel 运行时上下文固定预留 **32768 字节**，局部直方图可用预算 = `253952 − 32768 = 221184 字节（= 55296 × int32）`。**推导 `MAX_BATCH_BINS` 必须用可用预算而非总容量，且不可省略 depth 槽位因子**：`(MAX_BATCH_BINS + 对齐填充) × sizeof(HistT) × depth ≤ 221184`（`depth = useDoubleBuffer ? 2 : 1`，与 [../../generic/bin-model.md](../../generic/bin-model.md) §4 一致），否则单批 local_hist Buffer 溢出、Bisheng launch 静默失败（ret 107000）。启用多槽位 RingBuffer（depth 2 双缓冲，见 §4 规则 3）时片上占用翻倍、`MAX_BATCH_BINS` 相应减半，漏乘 depth 会复现上述溢出——实测踩坑：误用总容量 253952 得 `MAX_BATCH_BINS=61440`（`(61440+8)×4=245792>221184`）导致分批用例 `TD-u04-batch` 失败。约束局部直方图**单批** bins 数量（分批加载突破总 nbins 限制）；另注意 DSL Buffer **仅静态 shape**（见 §4 规则 5），分批 Buffer 尺寸须编译期确定 |
| `ALIGN_GRANULARITY` | 32 字节（UB block） | 片上数据对齐粒度 = 32 字节（与 AscendC `GetUbBlockSize()` = 32 字节一致，dav-3510 UB 最小搬运/对齐块，独立于向量寄存器宽度 256 字节）；局部直方图行宽填充元素数 = 32 / sizeof(HistT)（int32/float→8、fp16→16、int64→4、int8→32） |
| `PREFIX_MINMAX_REDUCE` | ✔（`cb.simt.atomic_min/atomic_max`，GM 地址） | 「range 缺省时对输入做前置 min/max 归约」：每线程局部 min/max + `` `cb.simt.atomic_min/atomic_max(gm.ptr(0), v)` `` 归约到 GM 暂存（已实测 float32 跨核原子收敛）。range 缺省分支据此落地 |
| `FLAT_INDEX_CALC` | ✔（纯标量整数乘加） | 展平 index 是纯标量整数乘加（Horner 递推），DSL kernel 内普通表达式即可，不涉及原子/向量/搬运原语 |

## 2 通用术语 → DSL 实体对照

| 通用层术语 | DSL 实体 |
|------|------|
| 片上局部直方图（每核一份、分批 bins） | `Buffer(MemLoc.UB, (MAX_BATCH_BINS,), dtype)`（编译期静态 shape，阶段内反复读写） |
| 片上原子累加（`` `LOCAL_HIST_ATOMIC` ``） | `cb.simt.atomic_add(ptr, v)`（SIMT VF 内，UB 指针） |
| DMA 原子归并（`` `LOCAL_HIST_FLUSH` ``） | `mem_copy(dst, src, atomic_add=True)`（UB → GM 累加） |
| GM 直接原子（`` `GLOBAL_MERGE_ATOMIC` ``） | `scalar.atomic_add(ptr, v)`（GM 逐元素原子加） |
| 计数段 → 归并段跨引擎同步 | SIMT VF 边界 + DSL 编译器自动同步（`` `cb.simt.syncthreads()` `` 保证核内累加完成，`mem_copy` 由编译器插入跨引擎同步） |
| 线程栅栏 | `cb.simt.syncthreads()`（SIMT VF 内） |
| 元素 → 线程映射（每核分块 + 核内跨步） | core（`get_block_idx/get_block_num`）+ SIMT 线程（`cb.simt.thread_idx()/thread_dim()/grid_dim()`）两级并行；核内跨步即 SIMT 线程跨步 |
| 前置 min/max 归约 | `cb.simt.atomic_min/atomic_max(gm.ptr(0), v)`（`PREFIX_MINMAX_REDUCE` 已实测） |
| 展平 index | 纯标量整数乘加（Horner 递推） |
| 设备侧计算 | `@kernel`（设备函数）/ `@host`（宿主 launcher）装饰器 |
| Host 侧调度 | 公开入口（torch 张量参数）+ `@host` launcher |

## 3 待验证谓词的使用与收敛规则

1. **产出声明**：设计/代码产出引用任一待验证谓词支撑的通用层公式时，必须显式标注「该结论依赖待验证假设」并写出验证义务
2. **禁止虚假确定**：不得把「职责类目存在」写成「能力已验证」——API 分类只描述代码职责，不证明硬件支持、算法适用性或运行时行为（API 分类的证据边界）
3. **真值收敛**：CANNBotDSL 权威材料（wheel API 提取索引、官方文档、实测结论）就位后，逐项替换待验证真值并删除对应验证义务；某谓词长期填不出真值时，按 [../../generic/capability-contract.md](../../generic/capability-contract.md) §1 规则 3 修订通用层抽象而非绕过

## 4 关键语言级附加约束

1. **设备计算归属**：spec / DESIGN 规定在设备上完成的 bin index 计算、原子累加、分批归并、越界丢弃、最终输出合成，必须落在 CANNBotDSL kernel 内；禁止在公开入口、Host launcher 或其可达辅助函数中用 `torch` / `torch.nn.functional` 小算子补全实现。Host 仅限：读取 shape/dtype/device 元数据、分配输出/workspace、建立零拷贝视图、发起 DSL kernel
2. **两模板与原子/归并能力绑定**：局部直方图模板依赖 `` `LOCAL_HIST_ATOMIC` `` + `` `LOCAL_HIST_FLUSH` ``，无局部直方图模板依赖 `` `GLOBAL_MERGE_ATOMIC` ``。这些谓词真值待验证时，先按「原子累加可用」假设落地并声明验证义务；验证为 false 则回退（局部直方图模板改「每核独立输出 + 跨核归约」，无局部直方图模板改「排序/分桶归约」），不在设计中当作已验证事实
3. **RingBuffer/Buffer 选择**：局部直方图需「计数 → 归并」跨阶段多槽位流水交接才用 RingBuffer；仅为当前阶段可反复读写的临时直方图用 Buffer（单槽、无 ring-buffer 语义）。RingBuffer 用 `head_next()`（写游标）/`tail_next()`（读游标）独立推进读写游标（据仓内源码证据，待上板实测确认），不做 FIFO 满/空检查与配对约束，计数/归并间同步由 Tensor 数据访问推导
4. **平台参数获取**：核数、片上容量等经架构/平台信息类目 API 获取，严禁写死；分批阈值（`` `ONCHIP_HIST_CAPACITY` ``）按运行时容量计算。**注意区分「总容量」与「可用预算」**：`cb.get_mem_size("ub")` 返回的是总容量，`ONCHIP_HIST_CAPACITY` 必须取扣除 SIMT kernel 上下文固定预留（32768 字节）后的可用预算参与 `MAX_BATCH_BINS` 推导（见 §1 `ONCHIP_HIST_CAPACITY` 行）
5. **Buffer 静态 shape 约束**：DSL Buffer 仅支持静态（编译期）shape，不能按运行时 `batchTotalBins` 动态变长。局部直方图分批 Buffer 须建成不变量 `` `batchTotalBins = min(totalBins, MAX_BATCH_BINS) ≤ MAX_BATCH_BINS` ``，Buffer 物理 shape 统一为编译期 `` `MAX_BATCH_BINS` ``（Host 侧 clamp），运行时只用前 `` `batchTotalBins` `` 个槽位
6. **SIMT 并行与原子性**：DSL SIMT VF（`` `cb.vf(mode="simt")` ``）以 SIMT thread 为核内并行单位（`` `cb.simt.thread_idx()` ``），多个线程可能写同一 bin。「片上普通 `` `+=` `` 无需原子」只在单线程顺序执行时成立；存在多线程写同 bin 时**必须用 `` `cb.simt.atomic_add` ``**（本范式局部直方图计数步的标准落法）。subblock（`` `get_subblock_id/dim` ``）是 cube 相关概念，本纯 Vector SIMT 算子不依赖
7. **UB tensor 点赋值限制**：DSL 的 UB `Buffer` 元素点赋值/自增（`` `hist[i] = ...` `` / `` `hist[i] += ...` ``）**只允许在 `` `cb.vf(mode="simt")` `` 内**，普通 `` `cb.range`/`if` `` 控制流里对 UB 点赋值会报 `NotImplementedError`；GM tensor 点赋值不受此限。片上累加统一用 `` `cb.simt.atomic_add` ``
8. **SIMT VF 内整数拓宽用 `` `cb.simt.cast` ``**：SIMT VF 内不能走标量 arith 路径，`` `dtypes.int64(x)` `` 需改为 `` `cb.simt.cast(x, dtype=dtypes.int64)` ``（int64 输入用 `` `const_expr` `` 跳过）
9. **`` `mem_copy` 1-D 切片用单元素元组**：`` `mem_copy` `` 的 shape tree 会拒绝 `` `hist[start:start+bins]` `` 这类裸切片，须写成 `` `hist[start:start+bins,]` ``（单元素元组）、`` `local_hist[0:bins,]` ``，否则翻译报 shape 错误
10. **1024 字节双缓冲阈值是 AscendC 启发式**：通用层 [../../generic/bin-model.md](../../generic/bin-model.md) §4 的 `` `useDoubleBuffer` ``（`totalBins×sizeof > 1024`）源自 Ascend C 侧经验，DSL 侧是否同样适用以实测为准（待验证），设计引用时标注来源
11. **边界条件与模板分派**：`bins/inputsize < GM_ATOMIC_ADD_FACTOR` 的模板选择、越界元素丢弃、min==max 特判、边界钳制均在 kernel 内按通用层 [../../generic/bin-index.md](../../generic/bin-index.md) / [../../generic/accumulation.md](../../generic/accumulation.md) 落地，不因 DSL 而改变语义
