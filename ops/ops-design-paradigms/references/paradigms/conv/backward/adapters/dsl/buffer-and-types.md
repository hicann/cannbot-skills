# DSL 类型、布局与存储槽

本页共用于 dX/dW。矩阵轴义从对应 generic 分支取得，不能把 dX 的空间 M 直接套到 dW 的输出通道 M。

类型组合由 [input scene](../../generic/input/scene.md) 和 [filter scene](../../generic/filter/scene.md) 定义。下面将这些类型映射到 DSL dtype、存储空间和字节单位，供分配与指令构造使用。

## 1 类型与单位

| 对象 | DSL 类型 / 空间 | 约定 |
|---|---|---|
| 输入特征、权重、dedy | `dtypes.float16` 或 `dtypes.bfloat16`，GM | 每个分支要求两路输入同类型 |
| L1、L0A、L0B | 输入类型；`MemLoc.L1/L0A/L0B` | B16 输入分形粒度为 16；具体轴序按分支映射 |
| 部分和 | `dtypes.float32`，`MemLoc.L0C` | NZ，16×16 分形 |
| dedx / Transpose 输出 | 与数值输入同类型，GM | 最终 Fixpipe 执行类型转换 |
| dedw | `dtypes.float32`，GM | FP32 写回不附加类型转换；地址偏移按每元素 4B 计算 |
| Transpose bias | FP32，GM→L1→`MemLoc.BIAS` | 只有带 bias 的实例分配 |

`make_layout(shape, stride=...)` 的 shape 和 stride 都以**当前 dtype 的元素**计。`from_tensor(..., offset=...)` 以**字节**计；运行时 offset 使用 i64。DMA 的 32B block、分形步长由实际布局转换，不能把 Ascend C 结构体中的数值原封不动填成 DSL stride。

## 2 视图不搬数据

```python
import cannbotdsl as cbd
from cannbotdsl import MemLoc, dtypes, make_layout

# 位于 @kernel / @jit；src 是 GM Tensor，地址已经过边界检查。
# rows、cols、row_stride 是元素数，byte_offset 是非负字节偏移。
gm_tile = cbd.from_tensor(
    src, layout=make_layout((rows, cols), stride=(row_stride, 1)),
    offset=byte_offset, data_format="nd",
)
```

`from_tensor` 只在同一地址上建立解释，既不转置也不补零。`permute` 同样只改变逻辑视图；后续 `mem_copy` 才按源和目的布局选择硬件转换。NCDHW、NDHWC、DHWCN 是外部轴序，DSL 的 `data_format="nd"/"nz"/"zn"` 是存储格式，两者分开描述。

静态容量用 `Buffer` 或 `RingBuffer` 分配，实际尾长通过切片或 `from_tensor` 表达。对于 NZ/ZN 父块，尾视图必须保留父物理 pitch。用更小的逻辑 shape 重新推导紧凑物理布局，会改变下一分形的地址；需要重新绑定布局时显式提供 `physical_layout`。`from_tensor` 不检查容量，调用方核对最大访问末地址。

GM→L1 的批量矩阵可把批轴写入 layout，让硬件使用矩阵数和矩阵步长。若既有流程能用一条 ND2NZ/DN2NZ 搬入多行，就不要在 Python/设备代码中按 H 逐行发 copy；检查生成指令中的矩阵数量和间隔是否与计划一致。

### NZ 的逻辑轴、物理轴与 stride

这套轴义适用于 L1、L0 等存储空间中的 NZ 视图，也适用于声明为 NZ 的 GM 视图；它不是 L0C 专用规则。对于 DSL 的 ND→NZ 映射：

```text
逻辑轴：(..., M, N)
物理轴：(..., N1, M1, M0, N0)
```

最后两根逻辑轴是矩阵 M/N，前面的轴是批轴。因此逻辑三维 `(B,M,N)` 对应五维物理 `(B,N1,M1,M0,N0)`；逻辑二维则只有四根物理轴。B 表示当前视图的矩阵批轴，可以承载 H、tap 或 co，不一定是网络输入的 batch。普通 ND 五维 Tensor 不遵循此分形解释；ZN 也须按其自身映射核对。

`physical_layout.stride` 与上述**物理轴逐项对应**，不是与逻辑 shape 中同名变量对应。记分形内尺寸为 `m0_size/n0_size`，五维 NZ 视图的元素地址为：

```text
offset(b,m,n) = b*sB
              + (n//n0_size)*sN1 + (m//m0_size)*sM1
              + (m%m0_size)*sM0 + (n%n0_size)*sN0
```

各 stride 均为元素数，字节地址还需乘 dtype 字节数并加视图基址。分形内尺寸由 dtype、存储空间及布局规则确定：本适配的 B16 L1 NZ 和 FP32 L0C NZ 使用 16×16，不能据此把所有存储空间的 FP32 NZ 都当成 16×16。显式填写物理 shape/stride 只改变这些轴的容量和地址步进，不改变逻辑 M/N 的拆轴规则，也不实际搬动数据。

例如将 dW 的 L0C 写回视图设为 `(co,tap,ci)`，则 `B=co、M=tap、N=ci`。记元素坐标为 `(co_idx,tap_idx,ci_idx)`；在 `0<=tap_idx<16、0<=ci_idx<16` 的范围，tap 坐标由 M0 承载，维度长度可以等于 16。若原数据地址为 `co_idx*16 + tap_idx*256 + ci_idx`，源视图须表达 `sB=16、sM0=256、sN0=1`。把物理 shape 写成 `(co,tap,1,16,16)`，同时保留 `sM0=16`，不会让逻辑 tap 自动走 N1 的 256 步幅。先用上式核对首项、相邻项和分形边界，再核对该布局能否生成目标搬运指令；地址语义正确不等于任意 stride 都被硬件支持。

## 3 Buffer 与 RingBuffer

L1/L0A/L0B/L0C 的 `Buffer` 和 `RingBuffer` 默认按 NZ 解释，NZ 的逻辑 shape 至少二维。分配矩阵时直接保留 `(BM, BK)` 等矩阵轴；不能把容量展平成 `(BM * BK,)` 后省略格式，否则会在构造时触发 `nd2nz requires logical rank >= 2`。若确实需要原始存储，再自行绑定各视图，分配时显式指定 `data_format="nd"`，或使用下节的空间根；搬运和计算的视图仍须分别声明正确的 NZ/ZN 布局。

```python
# 接口片段：BM/BN/BK 为已选模板的编译期容量。
a0 = cbd.RingBuffer(MemLoc.L0A, dtypes.float16,
                    make_layout((BM, BK)), depth=2)
c0 = cbd.Buffer(MemLoc.L0C, dtypes.float32,
                make_layout((BM, BN)))

# l1_window 已具有目标加载所需布局；这里只演示选择槽。
aw = a0.next_write()
cbd.mem_copy(aw, l1_window)
ar = a0.next_read()
# 后续计算使用 ar；不要为了重复读取同一槽再次调用 next_read()。
```

`Buffer` 返回一块存储的 Tensor。`RingBuffer` 给多块存储加独立读写游标，每次 `next_write/next_read` 选择当前槽并立即推进游标。选择槽不会等待、检查队列是否满或初始化数据；编译器从随后对 Tensor 的访问生成依赖。需要驻留并多次读取同一父段时，保留已选 Tensor，直到最后一个消费者结束。

`depth=2` 仅提供两槽，计算与搬运能否重叠还取决于指令发射顺序。预取下一父段应早于对当前计算结果的等待，且不能覆盖仍在使用的槽。驻留侧的生命周期覆盖整个复用域，不能随着流式侧每步推进。

对同一存储空间内需同时驻留的独立分配，先核对 `Σ(depth_i × slot_bytes_i) <= capacity_bytes`；同一底层存储的空间根和派生视图不重复计数。单槽 Buffer 按 depth=1 计，`slot_bytes_i` 是包含分形补齐的单槽物理跨度。各路可以使用不同槽大小。这只是容量必要条件，实际分区还要核对对齐、间隔和最大末地址，见下一节。

## 4 按既有分区绑定地址

当模板要求 L1 ping/pong 跨半区放置时，用完整空间根和静态视图表达。以下片段中的容量、基址和形状来自 Host 的资源预算；同一 L1 的 A/B/bias 都从同一个根派生。

```python
# 位于 @kernel；这些常量属于已选编译配置。
l1_root = cbd.L1.view(L1_BYTES)
a_l1 = cbd.make_ring_buffer((
    cbd.from_tensor(l1_root, dtype=Tin,
                    layout=make_layout((A_ROWS, A_COLS)),
                    offset=A_PING_BYTES, data_format="nz"),
    cbd.from_tensor(l1_root, dtype=Tin,
                    layout=make_layout((A_ROWS, A_COLS)),
                    offset=A_PONG_BYTES, data_format="nz"),
))
```

同时核对：每槽物理跨度、首地址对齐、最大末地址、A/B/bias 是否重叠。只算数据量之和不能证明地址分区合法。L0A/L0B/L0C 可以用相应空间根按同样方式绑定所选分区；不要再在相同地址隐式分配另一份独立 Buffer。

950 的 L1 为 512 KiB，常用两组起点为 0 和 256 KiB；实际 A/B 子区仍按模板预算。L0C 槽数沿用已选模板，并核对 950 的 bank 关系；分成两块地址不等于读写可以并行。本文不改变 generic 的选块和 buffer 份数策略。

## 5 分配核对表

| 检查项 | 要保留的事实 |
|---|---|
| 静态容量与动态有效尺寸 | 所有运行时窗口都落在分配容量内；换 shape 后父 pitch 和槽间距不变 |
| 源目的矩阵朝向 | Load2D/Load3D、L0B 布局和 MMAD 的 K/N 解释一致 |
| 尾部与零洞 | 硬件 padding 覆盖哪些位置，哪些位置必须显式初始化 |
| 复用 | 同一物理槽最后一次异步读取完成后才能覆盖 |
| 输出 | 只写有效坐标，不把分形补齐元素当作真实输出 |
