# input 指令映射

本页使用 `import cannbotdsl as cbd`，类型从 `cannbotdsl` 导入。`make_layout` 的 stride 按元素计，`from_tensor(..., offset=...)` 按字节计。矩阵朝向、反向边界和原张量偏移仍按 [通用计算流程](../../../generic/input/compute-flow.md) 推导。

## 1 一条计算链

| 阶段 | DSL 表达 | 应核对的生成行为 |
|---|---|---|
| GM dY → L1 fmap | 带 batch/行距的视图 + `mem_copy`，目的为 `data_format="nz"` | ND2NZ 或 DN2NZ；前放大通过目的步幅插入空位。 |
| GM weight → L1 | 按变换分支构造逻辑轴、`permute` 和物理布局后 `mem_copy` | 对应 ND2NZ/DN2NZ 父布局；可批量的矩阵不退化成逐行 DMA。 |
| L1 fmap → L0A | `load3d` | Load3D，配套 FMatrix、padding、repeat 配置。 |
| L1 weight → L0B | `mem_copy`，显式 NZ/ZN 和反向物理步幅 | Load2D 的转置与逆序参数。 |
| L0A × L0B → L0C | `matmul(acc, a, b, init=...)` | B16 MMAD、FP32 累加。 |
| L0C → GM dX | `mem_copy(dst, acc.convert(dtype))` | NZ2ND/NZ2DN Fixpipe，并直接转换 FP16/BF16。 |

`mem_copy` 按视图选指令，不通过名字指定“Load2D”。视图元数据必须描述已有物理数据；`permute` 或重新声明 layout 本身不执行重排。

## 2 dY 按 H 批量搬入与前放大

L1 的有效 fmap 是 `[Co1,Hexpand,Wexpand,Co0]`，B16 的 Co0=16。`strideH/W>1` 时，先清零本轮将被 Load3D 读取的插零区域，再把紧凑 dY 写到间隔位置。Load3D 自身的 stride 随后取 `(1,1)`，不再重复应用正向 stride。

下段展示**已对齐 Co、单 batch/单 depth 的搬入片段**。`src` 已包含调用者的原始 pitch：NCDHW 可从 `[Co,H,W]` 做 `permute(1,2,0)`，NDHWC 可直接建立 `[H,W,Co]`。实际 H/W/Co 可为运行时值；tile 原点由外层加入。

```python
# H、W、Co：本次实际搬入范围；Sh、Sw：正向 stride。
# a1 已按对齐后的物理跨度分配，C0=16，B16 每元素 2B。
Eh = (H - 1) * Sh + 1
Ew = (W - 1) * Sw + 1
row_batch = cbd.from_tensor(
    a1, layout=cbd.make_layout((H, W, Co)), data_format="nz",
    physical_layout=cbd.make_layout(
        (H, Co // 16, (W + 15) // 16, 16, 16),
        stride=(Sh * Ew * 16, Eh * Ew * 16,
                16 * Sw * 16, Sw * 16, 1)))
cbd.mem_copy(row_batch, src)
```

外轴 H 表示一次 ND2NZ/DN2NZ 的矩阵数（`ndNum/dnNum`）；目的矩阵间距承载 Sh，目的空间步幅承载 Sw。生成后检查矩阵数与两个步幅，不能默认任意三维视图都会得到同一指令。如果参数超出硬件单指令范围，才按该范围拆段。

插零清理可用 `cannbotdsl.ops.memcpy.cube_fill_l1_zero`，`blk_num` 的单位是 32B；在清零与随后 MTE2 写入间保留 `cbd.cube_sync_pipe(cbd.PIPE.MTE2)` 的同流水顺序。对齐通道尾部、H/W 插零孔洞和 Load3D 越界 padding 是三件事：padding 不会替你初始化 L1 内部的孔洞。完整无插零且所有读取元素已由 DMA 定义时，不额外全量清零。

## 3 weight 的三种准备方式

| `weight_mode` | GM→L1 父布局 | L1→L0B |
|---|---|---|
| transpose_only | 核体积为 1 的 Co×Ci 父矩阵，ND2NZ | 转置通道分形，无 tap 逆序。 |
| transpose_and_reverse | 保留 Co/tap/Ci 的父跨度；NCDHW 用 DN2NZ，NDHWC 用 ND2NZ | 在逆序 tap 起点取子视图，NZ→ZN 触发通道转置；tap 分段沿用原流程。 |
| reverse_only | 两种 weight format 都使用 DN2NZ，按各自源 stride 形成相同父轴序 | 按逆向 tap 的物理步幅取窗口，Load2D 不再转置分形。 |

下面给出可直接用于 `from_tensor` 的父布局。`O/N` 是当前父块 Co/Ci 向 16 对齐后的长度，`o/n` 是有效通道数，`T=hk*wk`，`V=dk*T`，`CinG` 是原 GM 的组内输入通道数。所有 source 都从已经包含 group、Co/Ci 起点和当前 kd 的字节基址开始；下表的 stride 均以 B16 元素计。

| 方式 / weight format | GM 逻辑 shape | GM stride | L1 父布局 |
|---|---|---|---|
| transpose_only / 两种格式 | `(o,n)` | `(CinG,1)` | P0 |
| transpose_and_reverse / NCDHW | `(o,T,n)` | `(CinG*V,1,V)` | P1 |
| transpose_and_reverse / NDHWC | `(o,T,n)` | `(CinG*V,CinG,1)` | P1 |
| reverse_only / NCDHW | `(n,T,o)` | `(V,1,CinG*V)` | P2 |
| reverse_only / NDHWC | `(T,n,o)` | `(CinG,1,CinG*V)` | P3 |

所有 L1 父视图均用 `data_format="nz"`。创建容量视图后，只对相应有效 `o/n` 子区执行 copy，保留表中父 pitch：

| 父视图 | 逻辑 shape | `physical_layout` 的 shape | `physical_layout` 的 stride |
|---|---|---|---|
| P0 | `(O,N)` | `(N//16,O//16,16,16)` | `(O*16,256,16,1)` |
| P1 | `(O,T,N)` | `(O,N//16,(T+15)//16,16,16)` | `(16,T*O*16,16*O*16,O*16,1)` |
| P2 | `(N,T,O)` | `(N,O//16,(T+15)//16,16,16)` | `(16,T*N*16,16*N*16,N*16,1)` |
| P3 | `(T,N,O)` | `(T,O//16,N//16,16,16)` | `(N*16,T*N*16,256,16,1)` |

P1 的物理轴序是 `[Ci1,tap,Co1,Co0,Ci0]`；P2/P3 都形成 `[Co1,tap,Ci1,Ci0,Co0]`。例如 P1 的有效目的切片是 `parent[:o, :, :n]`。容量按实际物理访问的最大末地址检查，不能只计算有效 `o*T*n`。

weight 的父块也有未搬入区域：transpose_only/transpose_and_reverse 在 `o<O` 时先初始化父块，Ci 尾部由格式转换补齐；reverse_only 在 `n<N` 时先初始化父块，Co 尾部由格式转换补齐。其余完整覆盖的父块无需额外清零。清理接口及 MTE2 顺序与上一节一致。

不能对三者统一使用紧凑 `[N,K]` 的默认 L1 layout。尤其 `permute` 后的逻辑形状相同，也不代表父矩阵的物理跨度相同。批量 weight 搬入时，用最高逻辑轴承载矩阵数，把 Co/tap 的真实 pitch 写入物理布局。

下面是 **reverse_only、N 为 16 的倍数、一个完整 Co0 窗口**的读片段。父 L1 已排列为 `[Co1,tap,Ci1,Ci0,Co0]`，每个 Co1 共 T 个 tap；这里只示范逆向 Load2D 地址，不承担前一步排布：

```python
rev = cbd.from_tensor(
    weight_l1, layout=cbd.make_layout((N, T * 16)), data_format="nz",
    physical_layout=cbd.make_layout(
        (T, N // 16, 16, 16), stride=(-N * 16, 256, 16, 1)),
    offset=((co1 + 1) * T - 1) * N * 16 * 2)
cbd.mem_copy(weight_l0, rev)
```

transpose_only 与 transpose_and_reverse 可以共用下面的 Load2D 片段，前者取 `T=1`。`b_k_start` 为相对 L1B 父段的 K 起点，`j` 是当前 L0B 窗口内的 K 分形编号；`k_use` 是 16 的倍数。父 L1 已按 P0/P1 建好，`b0` 是足够大的 L0B 存储：

```python
for j in range(k_use // 16):
    r = b_k_start // 16 + j
    src = cbd.from_tensor(
        weight_l1, layout=cbd.make_layout((16, N)), data_format="nz",
        physical_layout=cbd.make_layout(
            (N // 16, 1, 16, 16), stride=(T * O * 16, 256, 16, 1)),
        offset=((r // T) * 256 + (T - 1 - r % T) * O * 16) * 2)
    dst = cbd.from_tensor(
        b0, layout=cbd.make_layout((16, N)), data_format="zn",
        physical_layout=cbd.make_layout(
            (1, N // 16, 16, 16), stride=(256, 256, 16, 1)),
        offset=j * N * 16 * 2)
    cbd.mem_copy(dst, src)

# 对同一 L0B 字节建立 MMAD 的 [N,K] 视图。
rhs = cbd.from_tensor(
    b0, layout=cbd.make_layout((N, k_use)), data_format="nz",
    physical_layout=cbd.make_layout(
        (k_use // 16, N // 16, 16, 16), stride=(N * 16, 256, 16, 1)))
```

该循环与原模板每 K 分形一次转置 Load2D 的粒度对应。源逆序起点处理 tap，NZ→ZN 处理分形内部的 Co/Ci 交换；最后的 `rhs` 只是计算视图，不再搬运。Ci 尾块通过保留父 pitch 的 `rhs[:n,:]` 及对应 L0C/输出有效范围消费。

逆序起点落在该窗口最后一个 tap；负步幅不允许越过父 buffer。transpose_and_reverse 则给 copy 的目的子视图显式设置 `data_format="zn"` 与物理布局，生成转置 Load2D；MMAD 使用的 RHS 视图仍按 `[N,K]` 解释同一 L0B 数据。不要靠修改 MMAD 的 K 尾长触发隐式布局推断。

## 4 Load3D：几何与目的步幅一起绑定

源 NC1HWC0 的字节布局与多 C1 参数规则见 [Load3D 使用说明](../load3d.md)。本节目的为 L0A，使用下述 dX 的 M/K 映射，不照搬 dW L0B 的目的步幅。

以下是 **一个 K 子窗口的接口片段**，A1 是当前父段，A0 已分配足够空间。`k_local` 是相对 A1 父段的起点，不能使用完整 K 域起点。`m_local` 也须按当前 L1 驻留行带的原点与虚拟补边推导，见 [父行带公式](../../../generic/input/compute-flow.md#521-从-dedx-块反推-dedy-行带)。不要把全局 `mStart` 直接传入，再靠持续增大的 `pad_top` 抵消全局行偏移；950 的 FMatrix padding 字段只有 8 位，长 H 用例会超出可表示范围。至少用一个包含全局行号 `h >= 256`（从 0 计）的输出块，验证局部窗口与搬入原点是否一致。

```python
cbd.load3d(
    a0, a1,
    feature=cbd.Load3DFeatureMap(
        height=parent_h, width=expanded_w, channels=parent_co,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right),
    filter=(hk, wk), stride=(1, 1), dilation=(dilation_h, dilation_w),
    window=cbd.Load3DWindow(
        m_extent=m_use, k_extent=k_use,
        m_start=m_local, k_start=k_local),
    destination=cbd.Load3DDestination(k_stride_fractals=m_compute // 16))
```

H/W、channels、m/k_start 和 extent 都按元素/窗口数量计；`k_stride_fractals` 按 512B 分形计。`m_compute` 是此次 L0A **实际物理 M 跨度**，一般为 `alignUp(m_use,16)`。静态大 buffer 中复用尾块时，也要建立与该步幅一致的目的及 MMAD 视图，不能只把 extent 改小而保留另一套物理解释。

本路径使用默认 `transpose=False`、`config_bank="a"`、一次 repeat 和零 padding。配置对象在每次调用中完整描述状态，由编译器生成辅助设置并在允许时消减重复设置；调用者不依赖入口寄存器恰好为零。`pad_bottom=255` 是原父行带通路可用的边界配置，是否使用由通用窗口公式决定，不能把它当所有输入的数学 pad。

D 轴的有效 kd/dY 配对在外层循环完成。Load3D 只展开当前 H/W 面，`channelSize` 表示该父段覆盖的通道范围；它与本次 DMA 的未对齐有效通道数可以不同。

## 5 MMAD 与写回

A 为 `[M,K]`，B 为 `[N,K]`，L0C 为 `[M,N]`。`init=True` 绑定到本输出块的**第一次实际乘加**；后续 K、Co 和 kd 均使用 `init=False`。始终使用同一物理 L0C。返回 view 的元数据只有在与实际布局一致时才供后续调用，不能因 K 变短而改变已经积累的 C 布局。

GM 输出 view 的形状统一为 `[m_use,n_use]`，format 决定其 stride：

| 输出格式 | 逻辑 stride（元素） | 目标指令 |
|---|---|---|
| NDHWC | `(Cin,1)` | NZ2ND |
| NCDHW | `(1,Di*Hi*Wi)` | NZ2DN |

```python
# dst 已包含 batch/group/depth/空间/通道的字节基址。
# acc_valid 通过保留父 pitch 的切片得到，未重新紧凑化。
cbd.mem_copy(dst, acc_valid.convert(output_dtype))
```

N 尾块只写 `n_use`，NDHWC 的空间 stride 使用总 Cin，不使用 CinG。写回直接做 FP32→B16；无需另起 cast kernel。带 bias 的首次 MMAD 见 [BT 通路](bias-bt-path.md)。
