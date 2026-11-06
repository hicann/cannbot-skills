# VF：向量计算与局部编码优化

面向 Ascend 950 系列 SIMD VF 的编码示例；指令、向量宽度、mask 粒度与 dtype 支持按实际 DSL 和设备核对。局部改写保持数学语义、舍入与饱和规则、特殊值处理、归约顺序，以及既定 Tiling、分核、外部布局、资源配置、阶段划分和必要同步。

设备循环优先显式 `cb.range`；只有需要编译期索引或明确静态展开时使用 `cb.range_constexpr`。`unroll` 是优化请求，不能把循环次数固定当作必须编译期展开的理由。

## 示例约定

- Python 代码块为已有 `@cb.kernel` 或 `@cb.jit` 中的局部片段；正反例相互替换，不连续执行。`text` 块为结构伪码。
- 除显式标出作用域的片段外，VF 示例位于同一个 `with cb.vf(mode="simd"):` 内，`m = rr.create_mask(pattern="all", elem_bits=32)` 已在该 VF 创建；不使用 `@vf`，寄存器值不跨 VF。
- 默认 x、y 为已有 FP32 ND UB Tensor，输入输出不重叠，完整块长度为 64 的倍数。普通 vload/vstore 起址按 32B 对齐，完整物理访问范围合法。示例 FP32 每向量 64 个元素，紧凑 16 位数据为 128 个元素，展开到 32 位槽后为 64 个逻辑元素。
- 反例通常表示冗余写法；违反生成约定或正确性前提的情况按相应条目判断。资源选槽在 VF 外完成。

共用导入放在算子文件顶部：

```python
import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ops.reg import PackMode, UnpackMode, RegLayout, RoundingMode
```

按问题阅读：

- [控制流与循环](#控制流与循环)：VF03、VF04、VF05。
- [寄存器数据流](#寄存器数据流)：VF01、VF06、VF07、VF08 与访存屏障。
- [地址与有效 lane](#地址与有效-lane)：VF02、VF09、VF11。
- [类型转换与 lane 重排](#类型转换与-lane-重排)：VF10、VF12—VF15。
- [归约与结果复用](#归约与结果复用)：跨块累计与 VF16。

## 控制流与循环

### 条件执行约定

本 skill 要求 VF 区域及其展开的辅助逻辑不使用 `if/elif/else` 或条件表达式；这是生成约定，不宣称接口禁止所有分支。整个计算片段的条件放在 VF 外，逐 lane 条件使用受支持的 mask/选择操作。

必须在 VF 内按标量条件执行一次片段时，使用已规范化的整数 `execute_count`（假为 0，真为 1）：

```text
with cb.vf(mode="simd"):
    for _ in range(0, execute_count):
        完成一次条件向量计算
    执行后续公共计算
```

不把 lane mask 当循环次数，不在体内修改上界或依赖 break 退出。被跳过的代码不能是输出或后续输入的唯一初始化位置。核对展开的 helper，以及真假两条路径的初始化和后续计算。

### VF03．把循环不变分支移出热循环

**内容。** 调用期间不变的运行时条件在循环外选择路径；编译期已知的 dtype、模式使用 `const_expr` 特化。

**原理。** 将每次迭代的条件判断改为一次路径选择，让热循环仅保留当前模式的计算。编译期特化进一步避免生成未选中的路径。

例中 `enabled` 是不随迭代变化的运行时标量，未开启时要求原样复制。

**反例：每次处理一个向量都判断模式。**

```python
for i in range(n_vec):
    v = rr.vload(x, i * 64)
    if enabled != 0:
        v = rr.vmuls(v, 1.25, mask=m)
    rr.vstore(y, i * 64, v, m)
```

**正例：在 VF 外先选路径，再执行完整循环。**

```python
if enabled != 0:
    with cb.vf(mode="simd"):
        m = rr.create_mask(pattern="all", elem_bits=32)
        for i in range(n_vec):
            v = rr.vmuls(rr.vload(x, i * 64), 1.25, mask=m)
            rr.vstore(y, i * 64, v, m)
else:
    with cb.vf(mode="simd"):
        m = rr.create_mask(pattern="all", elem_bits=32)
        for i in range(n_vec):
            rr.vstore(y, i * 64, rr.vload(x, i * 64), m)
```

编译期路径也在 VF 外选择；仅当条件在构图期确定时才写 `if const_expr(condition):`；动态 shape 或设备标量不能伪装成静态值。不得通过同时执行两条分支消除判断，以免引入越界或副作用。

### VF04．有限展开循环，增加独立计算机会

**内容。** 对独立向量块的设备循环采用小倍数 `unroll`，保持每块内部运算顺序。

**原理。** 展开暴露相邻块的独立计算，减少循环控制开销，为装载与算术调度提供空间；同时限制代码体积和同时存活的寄存器值。

例中 `total` 是 128 的倍数，各块独立。

**反例：每个向量块单独迭代。**

```python
for off in cb.range(0, total, 64, unroll=1):
    v = rr.vload(x, off)
    v = rr.vmuls(v, 0.5, mask=m)
    v = rr.vadds(v, 0.125, mask=m)
    rr.vstore(y, off, v, m)
```

**正例：有限展开为两路，计算次序不变。**

```python
for off in cb.range(0, total, 64, unroll=2):
    v = rr.vload(x, off)
    v = rr.vmuls(v, 0.5, mask=m)
    v = rr.vadds(v, 0.125, mask=m)
    rr.vstore(y, off, v, m)
```

展开倍数与现有独立流数量共同控制。不要把较大的展开倍数固定套到所有循环，也不要将大循环整体改成构图期展开；归约循环必须保持原有累加顺序。

### VF05．保持规整循环，并约束硬件循环计数

**内容。** 将已有向量块数直接作为循环边界，保留固定步长、简单地址和无内部跳转的循环体；长循环分段执行。

**原理。** 简单的控制流有利于生成硬件循环，减少标量分支和循环控制。循环计数还必须处于目标硬件路径的可表示范围内。

例中 `n_vec` 为运行时有效块数，且不大于 `capacity_vec`。

**反例：按槽容量循环，每块再判断是否有效。**

```python
for i in range(capacity_vec):
    if i < n_vec:
        v = rr.vmuls(rr.vload(x, i * 64), 1.25, mask=m)
        rr.vstore(y, i * 64, v, m)
```

**正例：直接遍历有效块，移除循环内判断。**

```python
for i in range(n_vec):
    v = rr.vmuls(rr.vload(x, i * 64), 1.25, mask=m)
    rr.vstore(y, i * 64, v, m)
```

步长为 64 的源码循环也可以被归一化，无需机械改为步长 1。

在 DSL 0.8.0 + CANN 9.2.1 中，SIMD VF 的 `range(65536)` 重复加法已出现编译成功、设备结果错误；保持顺序拆成两段 32768 次可正确执行。遇到同类长循环时分段并校验结果，不把该现象推广为所有 DSL 循环的通用上限。

```python
# VF 内；values 已加载，m 对应有效 lane
for chunk in range(2):
    for step in range(32768):
        values = rr.vadds(values, 0.001, mask=m)
```

## 寄存器数据流

### VF01．外提并复用不变量、常量和广播值

**内容。** 同一行内不变的缩放系数只广播一次；整个 VF 不变的寄存器常量在循环外构造，重复用于后续计算。

**原理。** 将重复的装载、广播和常量构造移出热循环，减少每个向量块的准备工作。值在什么范围内不变，就在那个范围内复用，避免无必要地延长活跃区间。

例中 `cols` 是 64 的倍数，`scales[row]` 在列循环中保持不变。

**反例：每个向量块重复读取同一系数。**

```python
for j in range(cols // 64):
    scale = rr.vload_broadcast(scales, row)
    bias = rr.vdups(0.25, cb.dtypes.float32, mask=m)
    off = row * cols + j * 64
    value = rr.vmul(rr.vload(x, off), scale, mask=m)
    rr.vstore(y, off, rr.vadd(value, bias, mask=m), m)
```

**正例：一次准备，多块复用。**

```python
scale = rr.vload_broadcast(scales, row)
bias = rr.vdups(0.25, cb.dtypes.float32, mask=m)
for j in range(cols // 64):
    off = row * cols + j * 64
    value = rr.vmul(rr.vload(x, off), scale, mask=m)
    rr.vstore(y, off, rr.vadd(value, bias, mask=m), m)
```

只外提实际值和访存依赖均不变的操作；源可能被覆盖时不能跨越该写入。真正静态的 Python 算术与设备侧广播、运行时 Scalar 运算要分别判断。

### VF06．按块消费数据，缩短寄存器活跃区间

**内容。** 避免先装载大量向量，再依次计算；按小组装载、计算和写回，让中间值尽早结束使用。

**原理。** 减少同时存活的数据寄存器和 Mask，降低分配压力，为循环展开及独立指令交错留下资源。

例中两个向量块互不依赖。较多块时沿用同样的小组组织方式。

**反例：提前装载后续数据，延长其等待时间。**

```python
a = rr.vload(x, 0)
b = rr.vload(x, 64)
a = rr.vmuls(a, 1.25, mask=m)
rr.vstore(y, 0, a, m)
b = rr.vmuls(b, 1.25, mask=m)
rr.vstore(y, 64, b, m)
```

**正例：一块完成后再装载下一块。**

```python
a = rr.vload(x, 0)
a = rr.vmuls(a, 1.25, mask=m)
rr.vstore(y, 0, a, m)
b = rr.vload(x, 64)
b = rr.vmuls(b, 1.25, mask=m)
rr.vstore(y, 64, b, m)
```

与 VF08 配合时采用有限小组：组内交错，组间及时释放。优化对象是实际活跃区间；单纯把静态 `self.*` 改名为局部变量，不属于本条的核心动作。

### VF07．直接用寄存器串接计算链

**内容。** 中间值只供当前 VF 的紧邻计算使用时，直接传递寄存器结果，去掉写 UB 再读回的中转。

**原理。** 消除中间 Store、Load 及只服务于该回读的等待，让生产者结果直接进入消费者。

例中 `tmp` 没有其他消费者，计算为先乘后加。

**反例：把中间乘积落到 UB 再取回。**

```python
p = rr.vmuls(rr.vload(x, off), 0.5, mask=m)
rr.vstore(tmp, off, p, m)
rr.vmem_bar("vst_vld")
p = rr.vload(tmp, off)
r = rr.vadd(p, rr.vload(delta, off), mask=m)
rr.vstore(y, off, r, m)
```

**正例：后续运算直接消费乘积。**

```python
p = rr.vmuls(rr.vload(x, off), 0.5, mask=m)
r = rr.vadd(p, rr.vload(delta, off), mask=m)
rr.vstore(y, off, r, m)
```

只删除完全消失的中转依赖；屏障若还保护其他访问，必须保留。不跨阶段或跨 VF 延续寄存器，不擅自把乘加替换为改变舍入行为的融合运算。

### VF08．交错独立计算链

**内容。** 对同一 VF 内的少量独立向量链，交错安排装载、乘法和加法，保留每条链内部次序。

**原理。** 用另一条链的就绪指令填补依赖等待，增加调度空间。交错的链数保持有限，避免同时活跃值过多。

**反例：第一条链完成后才开始第二条。**

```python
a = rr.vload(x, off)
a = rr.vmuls(a, 0.5, mask=m)
a = rr.vadds(a, 0.125, mask=m)
rr.vstore(y, off, a, m)
b = rr.vload(x, off + 64)
b = rr.vmuls(b, 0.5, mask=m)
b = rr.vadds(b, 0.125, mask=m)
rr.vstore(y, off + 64, b, m)
```

**正例：两条链逐层交错。**

```python
a = rr.vload(x, off)
b = rr.vload(x, off + 64)
a = rr.vmuls(a, 0.5, mask=m)
b = rr.vmuls(b, 0.5, mask=m)
a = rr.vadds(a, 0.125, mask=m)
b = rr.vadds(b, 0.125, mask=m)
rr.vstore(y, off, a, m)
rr.vstore(y, off + 64, b, m)
```

不重排存在别名的读写，不移动跨链归约或必要同步。VF06 管理活跃值数量，本条管理小组内部顺序，两者需要一起考虑。

### 内存屏障：只保护实际访问依赖

这里所说的 `vmembar` 对应包接口 `rr.vmem_bar(mode)`，可用模式为 `"vst_vld"`、`"vld_vst"` 和 `"vst_vst"`。它用于 VF 内存访问顺序，不替代 MTE 与向量流水线之间的事件，也不替代跨核同步；作用域选择见 [同步](synchronization.md)。

- 先确定哪些存储地址重叠，以及是写后读、读后覆盖还是写后覆盖，再选择对应模式。在必要的依赖边界插入，不在每条寄存器算术、每个循环迭代或 VF 首尾机械添加。
- 后续仍可直接使用已有寄存器值时，保留该值，避免无必要的 store → barrier → load。寄存器之间的值依赖不需要靠内存屏障表达。若地址互不重叠，或已有机制确实覆盖同一访问顺序，不重复添加。
- 不为减少数量而跨越必需的读写依赖合并屏障；不能仅凭源码顺序或一次结果正确就认定屏障冗余。

**反例：纯寄存器计算之间反复插入屏障。**

```text
scaled = rr.vmuls(values, scale, mask=mask)
rr.vmem_bar("vst_vld")
shifted = rr.vadds(scaled, bias, mask=mask)
rr.vmem_bar("vst_vld")
```

`vld_vst` 用于需要保护的读后覆盖边界，`vst_vst` 用于写后覆盖边界；不要叠加模式来掩盖未分析的依赖。检查或调整屏障时，核对生成代码中的相关访存与屏障，再用覆盖该依赖的最小用例验证；普通命名、注释修改无需重复设备测试。

## 地址与有效 lane

### VF02．简化地址表达，用固定步长推进连续访问

**内容。** 行基址在行循环内计算一次；连续流使用普通 `vload/vstore` 的 `post_update`，以固定步长替代每次重建地址。

**原理。** 将重复索引运算收敛为起址与步长，使多输入、多输出的连续访问保持简单。保留真实物理行跨度，不用逻辑列数替代带 padding 的 stride。

例中连续处理 `n_vec * 64` 个 FP32 元素。

**反例：每次显式计算各路偏移。**

```python
for i in range(n_vec):
    a = rr.vload(x, i * 64)
    b = rr.vload(z, i * 64)
    rr.vstore(y, i * 64, rr.vadd(a, b, mask=m), m)
```

**正例：为各路建立独立游标，访问后推进。**

```python
xc = x.reinterpret()
zc = z.reinterpret()
yc = y.reinterpret()
with cb.vf(mode="simd"):
    m = rr.create_mask(pattern="all", elem_bits=32)
    for i in range(n_vec):
        a = rr.vload(xc, 64, post_update=True)
        b = rr.vload(zc, 64, post_update=True)
        rr.vstore(yc, 64, rr.vadd(a, b, mask=m), m, post_update=True)
```

`64` 是访问后的推进量，首次访问仍从当前起点开始；不能再叠加 `i * 64`。此写法针对普通 Load/Store，不套用于具有不同推进契约的 Unpack/Pack 接口。索引和地址类型保持原有值域，不为简化表达随意收窄。

RingBuffer 的选槽与视图准备位于 VF 外；`vload_unpack` 使用显式偏移，不依赖 post-update 推进。

### VF09．复用 Mask，以剩余计数推进尾部

**内容。** 每行初始化一次有效元素计数，每个向量更新一次 Mask，并由同一范围的算术和 Store 共享。

**原理。** 避免每个操作重复构造 Mask，也避免每个向量反复计算“总长度减当前位置”。显式接收更新后的计数，使主块与尾块采用同一条循环。

例中 `cols > 0`，当前行 UB 物理容量至少为 `ceil(cols / 64) * 64`，读取 padding 合法。

**反例：逐块重算有效长度，并重复创建同一 Mask。**

```python
for j in range((cols + 63) // 64):
    mask1, _ = rr.update_mask(cols - j * 64, elem_bits=32)
    v = rr.vmuls(rr.vload(x, base + j * 64), 1.5, mask=mask1)
    mask2, _ = rr.update_mask(cols - j * 64, elem_bits=32)
    rr.vstore(y, base + j * 64, v, mask2)
```

**正例：推进剩余计数，复用当前 Mask。**

```python
remaining = rr.mask_counter(cols)
for j in range((cols + 63) // 64):
    mask, remaining = rr.update_mask(remaining, elem_bits=32)
    v = rr.vmuls(rr.vload(x, base + j * 64), 1.5, mask=mask)
    rr.vstore(y, base + j * 64, v, mask)
```

每行重新初始化计数。普通 `vload` 仍完整读取一个向量，后续 Mask 不保护它的越界访问；归约也要按原语义排除无效 lane。不同槽宽、dtype 或有效范围不能直接共用 Mask。

`rr.vload` 不接受 mask 参数；满 mask 使用 `rr.create_mask(pattern="all", elem_bits=32)`。

### VF11．按访问语义选择 Load/Store 入口

**内容。** 单值广播直接使用 `vload_broadcast`；仅输出一个有效结果时使用 `vstore_first`；规则 DataBlock 跨距访问使用 strided 接口。

**原理。** 用匹配数据分布的访存指令表达需求，减少“整向量访问后再提取、复制或重排”的工作。

例中 `coeff` 至少覆盖从 `off` 开始的 64 个 FP32 元素，但计算仅使用其中第一个系数。

**反例：完整装载后，再复制 lane 0。**

```python
all_coeff = rr.vload(coeff, off)
scale = rr.vdup(all_coeff, mask=m)
v = rr.vmul(rr.vload(x, off), scale, mask=m)
rr.vstore(y, off, v, m)
```

**正例：直接广播所需系数。**

```python
scale = rr.vload_broadcast(coeff, off)
v = rr.vmul(rr.vload(x, off), scale, mask=m)
rr.vstore(y, off, v, m)
```

`vstore_first` 只写一个元素且不接 Mask；只用于原本就要求单值输出的地方。`vload_strided/vstore_strided` 的 offset 为元素数，`block_stride/repeat_stride` 为 32B 块数；块内任一 lane 有效时，读取仍可能覆盖整个 DataBlock。局部入口修改要保持原有 VF 阶段和同步位置。

## 类型转换与 lane 重排

### VF10．复用必要的类型转换结果

**内容。** 相同数据、目标类型、槽布局、Mask 和舍入条件下，转换一次供多个消费者使用；不为同一计算块单独增加完整的 UB 转型遍历。

**原理。** 将重复转换收敛为一个公共结果，同时减少为转换而发生的装载和中间存储。

例中 `slots` 来自 FP16/BF16 的 `vload_unpack`，两个消费者都要求 FP32。

**反例：同一个输入分别转换。**

```python
a = rr.vcast(slots, cb.dtypes.float32, mask=m)
square = rr.vmul(a, a, mask=m)
b = rr.vcast(slots, cb.dtypes.float32, mask=m)
scaled = rr.vmuls(b, 0.5, mask=m)
rr.vstore(y, off, rr.vadd(square, scaled, mask=m), m)
```

**正例：一次转换，多个消费者复用。**

```python
v = rr.vcast(slots, cb.dtypes.float32, mask=m)
square = rr.vmul(v, v, mask=m)
scaled = rr.vmuls(v, 0.5, mask=m)
rr.vstore(y, off, rr.vadd(square, scaled, mask=m), m)
```

FP32→FP16→FP32 包含中间舍入，不是可任意删除的恒等操作。不得把原算法要求的中间窄化移到最后，也不得用位解释替代数值转换。

### VF12．将槽展开和压缩融合到 Load/Store

**内容。** 窄输入、宽计算、窄输出采用 `vload_unpack → vcast → 计算 → vcast → vstore_pack`，减少独立的槽展开和压缩操作。

**原理。** 将数据槽布局调整并入访存端口，寄存器算术只处理实际计算需要的布局。Unpack/Pack 处理位与槽位置，Cast 处理数值，二者职责不同。

例中输入输出为 FP16，当前处理前 64 个元素；为保证反例完整 `vload` 合法，输入 backing storage 至少有 128 个 FP16 元素。输出采用 RN 舍入。

**反例：显式展开、压缩后再普通读写。**

```python
raw = rr.vreinterpret(rr.vload(x16, 0), cb.dtypes.uint16)
u = rr.vunpack(raw, cb.dtypes.uint32, part="lower")
slots = rr.vreinterpret_lanes(u, cb.dtypes.float16)
v = rr.vcast(slots, cb.dtypes.float32, mask=m)
v = rr.vmuls(v, 1.25, mask=m)
z = rr.vcast(v, cb.dtypes.float16, mask=m,
             rounding=RoundingMode.RN, reg_layout=RegLayout.ZERO)
bits = rr.vreinterpret_lanes(z, cb.dtypes.uint32)
packed = rr.vreinterpret(rr.vpack(bits, cb.dtypes.uint16), cb.dtypes.float16)
rr.vstore(y16, 0, packed, rr.create_mask("vl64", 16))
```

**正例：使用端口展开与压缩。**

```python
slots = rr.vload_unpack(x16, 0, unpack_mode=UnpackMode.B16_TO_B32)
v = rr.vcast(slots, cb.dtypes.float32, mask=m)
v = rr.vmuls(v, 1.25, mask=m)
z = rr.vcast(v, cb.dtypes.float16, mask=m,
             rounding=RoundingMode.RN, reg_layout=RegLayout.ZERO)
rr.vstore_pack(y16, 0, z, m, pack_mode=PackMode.B32_TO_B16)
```

两次必要的数值 Cast 均保留。`B32_TO_B16` 使用 32 位槽对应的 Mask，不能因为目标为 16 位就改用 16 位 Mask；尾部处理还需满足对应 Cast 的掩码约束。

### VF13．融合加载与 lane 重排，在寄存器内完成交织

**内容。** 对既定的交织数据，使用 `vload_deinterleave` 直接得到偶数、奇数两路；计算后用 `vinterleave` 恢复原输出顺序。

**原理。** 合并装载与解交织，消除独立重排步骤；后续 lane 组合留在寄存器内，不经 UB 中转。

例中 `x` 从 `off` 起有 128 个连续 FP32，分别缩放偶数和奇数位置，输出布局保持不变。

**反例：两次装载，再单独解交织。**

```python
lo = rr.vload(x, off)
hi = rr.vload(x, off + 64)
even, odd = rr.vdeinterleave(lo, hi)
even = rr.vmuls(even, 0.5, mask=m)
odd = rr.vmuls(odd, 1.25, mask=m)
lo, hi = rr.vinterleave(even, odd)
rr.vstore(y, off, lo, m)
rr.vstore(y, off + 64, hi, m)
```

**正例：装载时直接得到两路。**

```python
even, odd = rr.vload_deinterleave(x, off)
even = rr.vmuls(even, 0.5, mask=m)
odd = rr.vmuls(odd, 1.25, mask=m)
lo, hi = rr.vinterleave(even, odd)
rr.vstore(y, off, lo, m)
rr.vstore(y, off + 64, hi, m)
```

输入范围必须覆盖两向量；逐 lane 保持偶奇映射。不要只为减少 Store 数量额外插入复杂拼接，也不把局部 lane 重排扩大为外部布局变更。

### VF14．直接从已有寄存器系数做 Gather 广播

**内容。** 小维系数已经由当前 VF 产生时，用 `vgather_reg` 按固定索引复制到所需 lane，省去为 Gather 再落 UB、回读的路径。

**原理。** 让寄存器生产者直接连接广播消费者，避免中间内存访问。索引在映射不变的范围内构造一次并复用。

例中 `produced` 是已有的 64-lane FP32 结果，前 16 个值为列系数；对一个 `(4,16)` 输入块按列缩放。两版共用以下索引：

```python
lane = rr.vreinterpret(rr.varange(0, cb.dtypes.int32), cb.dtypes.uint32)
index = rr.vbitwise_and(lane, rr.vdups(15, cb.dtypes.uint32, mask=m), mask=m)
```

**反例：为读取系数，先写 UB 再 Gather。**

```python
rr.vstore(weights_ub, 0, produced, m)
rr.vmem_bar("vst_vld")
repeated = rr.vgather(weights_ub, index, mask=m)
v = rr.vmul(rr.vload(x, 0), repeated, mask=m)
rr.vstore(y, 0, v, m)
```

**正例：直接 Gather 已有寄存器。**

```python
repeated = rr.vgather_reg(produced, index)
v = rr.vmul(rr.vload(x, 0), repeated, mask=m)
rr.vstore(y, 0, v, m)
```

索引明确限定为 0～15，不依靠越界回绕。若系数还需输出给其他消费者，保留对应 Store，仅消除当前消费者的回读；不要为改用寄存器 Gather 又重复装载同一组系数。

### VF15．用完整的非对齐写入协议替代逐字节处理

**内容。** 固定非对齐尾部或 padding 的批量写入，采用寄存器构造、压紧写入和完整收尾，替代逐元素 Scalar 写。

**原理。** 将多次标量存储合为一次向量数据生成和批量输出，并通过 finalize 写完尚未落地的尾部。

例中 `out_u8` 为既有 uint8 UB，`1 <= count <= 256`，只把 `[offset, offset + count)` 清零；整个协议涉及的物理范围均合法。两段代码放在原 Kernel 的相同阶段。

**反例：逐字节写零。**

```python
for i in range(count):
    out_u8[offset + i] = 0
```

**正例：批量写，并完成同一组状态的收尾。**

```python
with cb.vf(mode="simd"):
    mask, _ = rr.update_mask(count, elem_bits=8)
    zero = rr.vdups(0, cb.dtypes.uint8, mask=mask)
    packed = rr.vsqueeze_and_storeunalign_init(zero, mask=mask)
    cursor = rr.vstore_unalign_begin(out_u8)
    rr.vsqueeze_and_storeunalign(out_u8, offset, packed, cursor)
    rr.vsqueeze_and_storeunalign_finalize(out_u8, offset, cursor)
```

begin、写入、finalize 必须使用匹配的 Tensor、偏移和状态，中途不混用另一输出流；零长度在调用方跳过。涉及重叠物理块时保留必要同步，不能仅按逻辑字节区间判断独立性。

**检查：** 窄化转换的舍入、溢出和特殊值；不同类型的 lane 数与 mask 粒度，不能直接复制固定步长。

## 归约与结果复用

### 一行归约，跨向量块累计

前提：二维输入列连续、行跨度明确，输出为每行一个 FP32 和。`VL` 表示当前计算类型的 lane 数。

```text
对每个 row:
    row_sum = FP32(0)                         # 每行重新初始化
    对 col = 0, VL, 2*VL, ... 且 col < cols:
        valid_lanes = min(VL, cols - col)
        mask = lane_id < valid_lanes
        values = 安全读取 input[row * row_stride + col] 的有效 lane
        values = 转为 FP32；无效 lane 填 0
        chunk_sum = 对有效 lane 做求和归约
        row_sum = row_sum + chunk_sum
    output[row * output_stride] = row_sum     # 每行只写最终结果
```

“安全读取”须落到当前指令支持的 masked load，或已分配且可访问的 padding。只对后续计算加 mask，不能保护前面的完整向量读。归约顺序与累计类型按误差要求选择，不保证与串行求和逐位一致。

**反例：** `row_sum` 放在所有行外面；每个向量块覆盖同一个输出；最后一个块沿用满 mask；FP16 输入未经设计就在 FP16 中累计；用逻辑行宽跨越带 padding 的行。


### VF16．在寄存器内广播归约结果

**内容。** 统计量已经位于寄存器 lane 0，且继续参与当前 VF 计算时，直接用 `vdup` 广播，不为广播而绕经 UB 或 Scalar。

**原理。** 省去统计量的回读及相应等待，延续 Vector 数据流。需要对外输出的统计量仍可同时保存。

例中 `safe_max` 是按既定归约顺序产生并加好下界的最大绝对值；保留 `stats` 的原有输出。

**反例：保存统计量后，从 UB 再读一次。**

```python
rr.vstore(stats, off, safe_max, m)
rr.vmem_bar("vst_vld")
denominator = rr.vload_broadcast(stats, off)
result = rr.vdiv(values, denominator, mask=m)
rr.vstore(y, off, result, m)
```

**正例：保留必要输出，计算直接复用寄存器。**

```python
rr.vstore(stats, off, safe_max, m)
denominator = rr.vdup(safe_max, mask=m)
result = rr.vdiv(values, denominator, mask=m)
rr.vstore(y, off, result, m)
```

只有源统计量确实位于 lane 0 才采用该广播。来自其他 VF 或阶段的统计量继续通过原有存储和同步读取，不改变归约树、特殊值规则或除法精度。

**检查：** 行宽为 VL−1、VL、VL+1，带行 padding，两行取不同常数；核对累计状态重置、大幅值与抵消输入的误差。
