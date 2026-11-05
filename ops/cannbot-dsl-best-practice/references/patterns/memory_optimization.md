# 搬运与内存访问编码优化

在既定方案内合并小搬运、批量处理有效数据、消除冗余写入。保持数学语义、Tiling、分核、外部布局、Buffer 容量与深度、阶段划分和必要同步。

示例是已有 Kernel 中的局部替换片段。`gm_*` 为 GM Tensor，`ub`、`*_ub`、`*_slot` 为原有受管理片上资源；保留实际生产消费依赖，视图不分配存储或改变物理排列。正反例不连续执行。

共用导入放在算子文件顶部：

```python
import cannbotdsl as cb
from cannbotdsl import reg as rr
```

布局与物理范围见 [布局与尾块](layout-tail.md)，选槽见 [RingBuffer 与 Buffer](channel-buffer.md)，容量见 [Workspace](workspace.md)，完成等待见 [同步](synchronization.md)，块内向量接口见 [VF](vf.md)。

## MV01．合并同一阶段的规则小搬运

**内容。** 同一 tile 中的数据已全部就绪时，把逐行、逐小段 `mem_copy` 合为多行搬运；保留原来的行跨度和数据顺序。

**原理。** 用一份规则视图表达整个区间，减少标量循环与重复搬运发射，让搬运引擎按二维范围执行。

例中 `gm_x` 与 `ub` 都有 4 行、每行至少 128 个 FP32；这四行原本就在同一阶段装入，行间可以有既定 padding。

**反例：同一块输入分成四次调用。**

```python
for row in range(4):
    src = cb.tile_slice(gm_x, (1, 128), (row, 0))
    dst = cb.tile_slice(ub, (1, 128), (row, 0))
    cb.mem_copy(dst, src)
```

**正例：用保留 stride 的多行视图一次表达。**

```python
src = cb.tile_slice(gm_x, (4, 128), (0, 0))
dst = cb.tile_slice(ub, (4, 128), (0, 0))
cb.mem_copy(dst, src)
```

同根、等间隔、形状相容的 GM 源片段也可通过 tuple/list 表达沿第 0 维拼接。不合并不同生产阶段或原本与计算重叠的流水，不用 `view` 伪造连续布局，不借此更换搬运引擎或扩大 tile。

## MV02．批量写回已经就绪的有效输出

**内容。** 一个既定阶段的输出全部完成后，用有效源视图批量写回，避免逐行发射，也不把 UB 槽的预留容量带入输出。

**原理。** 合并重复的写回控制与发射，同时由源视图明确实际 payload。源和目标各自保留原有物理行跨度。

例中 `rows`、`cols` 是既定正整数，输出从 GM 第 0 行开始，UB 可能有更多预留行。

**反例：已完成的输出仍逐行写回。**

```python
for row in range(rows):
    src = cb.tile_slice(y_ub, (1, cols), (row, 0))
    dst = cb.tile_slice(gm_y, (1, cols), (row, 0))
    cb.mem_copy(dst, src)
```

**正例：限定有效源域后批量写回。**

```python
src = cb.tile_slice(y_ub, (rows, cols), (0, 0))
dst = cb.tile_slice(gm_y, (rows, cols), (0, 0))
cb.mem_copy(dst, src)
```

源视图限定普通逻辑搬运范围，不能仅缩小目标、仍传入整个源槽。不要合并跨消费者、原子累加阶段或尚未完成的输出；已有逐行计算/写回流水保持原协议。

## MV03．以批量搬运和 VF 计算替代逐元素 Scalar 处理

**内容。** 规则数组沿用既有 GM↔UB 路径，在 UB 中用 VF 批量计算，去掉逐元素 Tensor 下标读写。

**原理。** 让数据沿搬运和 Vector 流水批量流动，避免每个元素进入 Scalar 流水执行装载、算术和存储。

例中 GM 与两个既有 UB Tensor 均为 64 个 FP32，计算为乘以 1.25。

**反例：块已在 UB，仍逐元素计算。**

```python
cb.mem_copy(x_ub, gm_x)
for i in range(64):
    y_ub[i] = x_ub[i] * 1.25
cb.mem_copy(gm_y, y_ub)
```

**正例：搬运路径保持不变，块内改用 VF。**

```python
cb.mem_copy(x_ub, gm_x)
with cb.vf(mode="simd"):
    mask = rr.create_mask(pattern="all", elem_bits=32)
    values = rr.vload(x_ub, 0)
    rr.vstore(y_ub, 0, rr.vmuls(values, 1.25, mask=mask), mask)
cb.mem_copy(gm_y, y_ub)
```

用于控制的少量长度、地址或索引仍可由 Scalar 处理。此项不新增 UB 中转资源，也不把不支持的 GM→GM 直接复制当作批量搬运接口。

## MV05．用有效视图统一主块与尾块

**内容。** 用 `tile_slice` 描述当前有效输入，写回时同步收窄 UB 源域；避免尾部逐元素补搬以及越界写出槽中旧数据。

**原理。** 视图将逻辑有效范围交给批量搬运，主块与尾块共用同一处理方式。有效域只决定哪些元素属于任务，不会自动缩小 VF 的物理访问宽度。

例中总共 10 行、每行 128 个 FP32，每槽 4 行；最后一块 `tile_id=2` 仅有 2 行，`y_slot` 的前 2 行已经算好。

**反例：正确但逐行写回最后两行。**

```python
for row in range(2):
    src = cb.tile_slice(y_slot, (1, 128), (row, 0))
    dst = cb.tile_slice(gm_y, (1, 128), (8 + row, 0))
    cb.mem_copy(dst, src)
```

**正例：直接表达整个有效尾块。**

```python
dst = cb.tile_slice(gm_y, (4, 128), (2, 0))
src = cb.tile_slice(y_slot, (2, 128), (0, 0))
cb.mem_copy(dst, src)
```

实际代码由当前块的有效行数构造 UB 源域，不硬编码最后一块的行数；输入侧同样使用有效 GM 源视图。**错误写法是把上例改成 `cb.mem_copy(dst, y_slot)`，企图让目标的 2 行自动裁掉源槽的 4 行。**

列尾还要独立证明普通 `vload` 的完整访问合法，Mask 只控制后续对应操作。GM→UB 的末块补齐不能当作整个槽已初始化；后续读取的完整物理范围仍须有合法分配与所需初值。保留必要的 padding 初始化，并避免零长 tile 进入搬运。

## MV06．删除完整覆盖前的冗余初始化和重复复制

**内容。** 若某段初始化值在首次被读取前已被完全覆盖，删除该初始化；相同数据的重复复制也按同样的生产消费关系判断。

**原理。** 去掉没有消费者的写入、对应计算及无用访存，保留真正产生最终数据的操作。

例中 `gm_tile` 与 `ub_tile` 都恰好为连续 512 个 FP32，无额外 padding；后续消费者只读取这 512 个元素，且没有任何消费者观察预清零结果。

**反例：先全量清零，随后又全量覆盖。**

```python
with cb.vf(mode="simd"):
    mask = rr.create_mask(pattern="all", elem_bits=32)
    zero = rr.vdups(0.0, cb.dtypes.float32, mask=mask)
    for i in range(8):
        rr.vstore(ub_tile, i * 64, zero, mask)
cb.mem_copy(ub_tile, gm_tile)
```

**正例：直接装载最终数据。**

```python
cb.mem_copy(ub_tile, gm_tile)
```

必须覆盖后续消费者会读取的全部字节，而不只是逻辑有效元素。只搬入 32 个 FP32、随后对 64 个元素归约且后半必须为零时，清零仍有语义，不能删除。原子累加初值、跨阶段可见写入和必要同步同样保留。

同一 Buffer 连续两次 MTE2 纯写，若要求后一次覆盖前一次，仍需 `vec_sync_pipe(cb.PIPE.MTE2)` 等待前一次完成；该接口从 `cannbotdsl.ops.sync` 导入。`RingBuffer.next_write()/next_read()` 只是选槽并推进游标，不能代替完成等待。删除冗余操作后，应继续保持真实读写依赖及其余消费者的同步。
