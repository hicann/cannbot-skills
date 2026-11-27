# dW：搬运、窗口、乘加与写回

轴义和地址公式见 [compute-flow](../../../generic/filter/compute-flow.md)。本页使用 `import cannbotdsl as cbd`，以及从 `cannbotdsl` 导入的 `MemLoc、dtypes、make_layout`。代码片段位于 `@cbd.kernel` 或其调用的 `@cbd.jit` 内；变量均来自已校验的 tiling 和当前子块。

## 1 指令对应关系

| 阶段 | DSL 接口 | 生成结果要核对什么 |
|---|---|---|
| dedy / x：GM→L1 | `from_tensor` + `mem_copy`；需要交换逻辑轴时用 `permute` | ND2NZ/DN2NZ、矩阵数、源总通道步幅、目的父段 pitch |
| dedy：L1→L0A | `mem_copy(a0, a1_view.permute(1, 0))` | 源为 `(K,M)` 时转为 `(M,K)`；Load2D 转置及父 K stride |
| x：L1→L0B | `load3d` + 几何描述对象 | 正向窗口、L0B 转置、当前有效 K/N、目的分形跨度 |
| L0A/B→L0C | `matmul(c, a, b, init=first)` | MMAD 的实际 m/n/k、当前段首次贡献、FP32 L0C |
| L0C→dedw GM | `mem_copy(dst, src, copy_options(atomic="add"))` | FIX 的 ND/DN 朝向、批矩阵数、源 pitch、仅有效 co/ci、FP32 原子写回 |

`permute` 和 `from_tensor` 只描述视图，不搬数据。GM→L1 的格式打包由实际 `mem_copy` 完成；`load3d` 从已排好的 NC1HWC0 展开窗口，不替调用方转换任意 CHW/HWC 源。修改 `data_format` 不能转换已经存在的字节。API 版本与能力检查见 [公共能力](../capability-mapping.md)。

## 2 两路 GM→L1

### 2.1 先确定 L1 父布局

与 generic 数据流对应的一种绑定如下：

| 操作数 | L1 逻辑视图 | 物理解释 | 下一步 |
|---|---|---|---|
| dedy | `(Kparent, Mparent)`，NZ | `(M1,K1,K0,M0)` | Load2D 转为 L0A `(M,K)` |
| x | `(Hcopy*Wcopy, Cparent)`，NZ/5HD | `(C1,Hcopy,Wcopy,C0)` | Load3D 提取正向窗口 |

`C0=16`。x 的 C1 步幅应是 `Hcopy*Wcopy*16` 个元素，不能无条件换成 `alignUp(Hcopy*Wcopy,16)*16`。后者可能与 Load3D 的寻址不一致。物理容量可以向上预留，实际物理 stride 仍显式声明。

dedy 也可在 GM 搬入时直接得到 `(M,K)` 的计算排布，但这会改变 ND2NZ/DN2NZ 与后续 Load2D 的组合；本适配按上表与既有模板对齐，避免只改一次搬运而继续沿用旧父 stride。

### 2.2 NCDHW 与 NDHWC 的轴序

取固定 batch、depth、group 及通道起点，`rows×cols` 是本次真实有效行带。x 的源视图可以这样构造：

```python
# NCDHW：先按实际存储解释，再把 H 作为批矩阵轴。
src = cbd.from_tensor(x,
    layout=make_layout((ci_use, rows, cols),
                       stride=(di * hi * wi, wi, 1)),
    offset=x_element_offset * 2).permute(1, 2, 0)  # (H,W,C)

# NDHWC 的等价源视图：
# src = cbd.from_tensor(x,
#     layout=make_layout((rows, cols, ci_use),
#                        stride=(wi * cin, cin, 1)),
#     offset=x_element_offset * 2)
```

对普通窗口、`cols` 为 16 的倍数，目标可按 H 批矩阵表达为下面的 5HD 排布。`c_parent` 为对齐到 16 的父通道容量，`ci_use` 为有效搬入数：

```python
dst = cbd.from_tensor(x_l1_slot,
    layout=make_layout((rows, cols, c_parent)), data_format="nz",
    physical_layout=make_layout(
        (rows, c_parent // 16, cols // 16, 16, 16),
        stride=(cols * 16, rows * cols * 16, 256, 16, 1)))
cbd.mem_copy(dst[:, :, :ci_use], src)
```

这是“以 H 为批轴，一条指令搬 C×W”的接口片段；`x_l1_slot` 须按完整跨度预留。非对齐宽片按实际父布局重新核算，不直接沿用这里的 `cols//16`。是否合成一条 ND2NZ/DN2NZ，必须查看此次编译产物；不能用 Python/设备端逐 H 循环掩盖本可批量表达的搬运。

dedy 同理：NCDHW 源按 `(co_use, rows, cols)`、stride=`(dout*ho*wo, wo, 1)` 解释后变为 `(rows,cols,co_use)`；NDHWC 直接用 `(rows,cols,co_use)`、stride=`(wo*cout,cout,1)`。L1A 的批矩阵目标和后续展平视图保留同一父段 pitch。

### 2.3 裁边、切宽与拆核

每个宽片重新计算 x 有效交集、dedy 起点、父段真实宽高和局部 pad。负坐标不直接送入 DMA；深度没有交集时按无贡献段处理。

拆核窗口先固定原 `kh/kw`，将原 stride 应用到 GM 采样地址，再用单位窗口 Load3D。若一行采样点不连续，以 W 作为批矩阵轴、每矩阵一行通道描述源 stride；跨 H 不满足同一硬件矩阵步进时保留既有逐行分支。不要把稀疏采样误写成一次连续的 `rows*cols` 搬入，也不要在硬件能用一个批轴时按每个点逐条发 DMA。

片上通道补齐和几何 pad 分开处理：Load3D 的零填充处理窗口越界，不初始化任意 L1 尾部。只对后续指令确实会读取而搬运没有定义的范围补零；必要时使用 `cannbotdsl.ops.memcpy.cube_fill_l1_zero`，按该接口的 32B block 数传参。

## 3 L1→L0 与 MMAD

多 C1 的源打包、窗口轴与独立探针见 [Load3D 使用说明](../load3d.md)。不要把单 C1 下恰好有效的连续 HWC 搬入直接扩成多通道。

L1A 源 `(Kparent,Mparent)` 的物理布局不随当前 K 尾长改变。先在父视图中选中本次 K/M 窗口，再转置加载到 L0A；不要重建一个紧凑父布局而丢掉原来的 stride。L0B 采用下面的轴映射：

| 量 | 含义 |
|---|---|
| `Load3DWindow.m_start / m_extent` | 当前父段中的输出空间起点 / 本轮 `kUse` |
| `Load3DWindow.k_start / k_extent` | 当前 `(ci1,kh,kw,ci0)` 起点 / 本轮 `nUse` |
| `Load3DFeatureMap.channels` | 当前 L1 源基址对应的父 fmap 通道数，按 C0 对齐；不随 `k_start/k_extent` 缩小 |
| `feature.height/width/pad_*` | 当前 L1 父段坐标系，不是未裁剪的全图 |
| `filter/stride/dilation` | 普通路径取原 H/W 属性；拆核路径全为 `(1,1)` |
| `config_bank` | 选 Load3D 配置寄存器组，通常 B 路用 `"b"`；与 L1 存储 bank 无关 |
| `destination.k_stride_fractals` | 目的分形步幅，单位 512B；必须与实际 L0B 排布一致 |

下面的 b0 逻辑视图按 `(N,K)` 绑定，物理存储已按 Load3D→MMAD 的 L0B 排布准备；示例沿用本轮紧凑 N 分形跨度：

```python
cbd.load3d(b0, x_l1,
    feature=cbd.Load3DFeatureMap(
        height=l1_h, width=l1_w, channels=channel_span,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right),
    window=cbd.Load3DWindow(
        m_start=spatial_start, m_extent=k_use,
        k_start=window_start, k_extent=n_use),
    filter=(hk, wk), stride=(stride_h, stride_w),
    dilation=(dilation_h, dilation_w), config_bank="b",
    destination=cbd.Load3DDestination(
        k_stride_fractals=(n_use + 15) // 16))

# DSL 的 rhs 视图为 (N,K)，两路最后一轴都是本轮有效归约长度。
# acc 的物理 pitch 已固定，换 k_use 不重建 acc。
cbd.matmul(acc, a0[:m_compute, :k_use], b0[:n_use, :k_use], init=first)
```

`load3d` 写 L0B 时自身执行所需转置，不再补一个矩阵转置。默认 `repeat` 是一次；本范围 B16 不需要移植 FP32 分形的逐 tap 双次加载。`channel_span` 必须匹配当前 L1 源基址对应的父通道范围；`nUse/kStart` 只选择该父段展开后的窗口，不能简单把 `nUse` 当通道数。若同时改变源基址或父布局，再重算这些参数。若该槽采用容量 pitch 而非紧凑 pitch，`destination` 要随其实际跨度调整。

`first` 表示当前归约段、当前 L0C 子区间的**首次实际 MMAD**。跳过无效深度后仍要保持它为真。跨 batch/dout、换父段、换宽片不能重新清零；新归约段才重置。K=70、baseK=32 时实际三次 k 为 32、32、6。`matmul` 返回目标视图，已显式绑定布局时也可像上面一样原地使用 acc；它不会每轮分配新 L0C。

### 单个有效 co 的计算 M

保留既有 ComputeInit 的扩展：`m_valid` 为真实 co 数；当它为 1 时，`m_compute=16`，其余情况 `m_compute=m_valid`。上面 MMAD 片段使用的是计算 M，写回使用有效 M。

| 阶段 | `m_valid=1` 时的绑定 |
|---|---|
| GM→L1A | 只读取一个真实 co，ND2NZ/DN2NZ 将该通道轴补到一个 16 元素分形；核对补齐位置为零，不能从 GM 多读 15 个通道 |
| L1A→L0A | 按计算 M=16 做同一转置加载；父 K pitch 和槽容量保留原计划 |
| MMAD | A 的视图为 `(16,k_use)`，L0C 使用对应的物理 M 跨度 |
| Fixpipe | 恢复 `co_use=m_valid=1`，只写真实的一行 co |

普通窗口和拆核窗口都保留这项规则。默认 `disable_gemv=True` 不代替上述补齐和写回绑定，不能只把 m=1 直接传给 MMAD 就认为与原模板等价。

## 4 L0C→三种 dedw 布局

FP32 L0C 逻辑列顺序为 `(ci1,kh,kw,ci0)`。Fixpipe 的 `mSize/nSize` 不一定等于 GEMM M/N：通过三维逻辑视图和显式物理 stride，核位置或 co 可成为矩阵批轴。

| dedw 格式 | 整窗写回的逻辑视图 | 目的元素 stride | 期待指令 |
|---|---|---|---|
| NCDHW | `(co, tap, ci)` | `(CinG*dk*hk*wk, 1, dk*hk*wk)` | co 为批轴的 NZ2DN |
| NDHWC | `(tap, co, ci)` | `(CinG, CinG*dk*hk*wk, 1)` | tap 为批轴的 NZ2ND |
| DHWCN | `(tap, co, ci)` | `(CinG*Cout, 1, Cout)` | tap 为批轴的 NZ2DN |

本表是固定 g/kd/ci 起点下的局部视图，基址须先含任务起点。DHWCN 的 Cout 是总输出通道数。`tap` 表示本次连续的 Hk×Wk 核位置区间。

若改成逐 tap 写 `(co, ci)`，去掉批轴也不能丢掉通道步幅。记 `Kvol=dk*hk*wk`：NCDHW 的目的 stride 为 `(CinG*Kvol, Kvol)`，NDHWC 为 `(CinG*Kvol, 1)`，DHWCN 为 `(1, Cout)`。不能统一使用 `(row_stride, 1)`；例如 NCDHW 的 `Kvol=9` 时，相邻 ci 应相隔 9 个元素，写到相邻元素会覆盖其他 tap。`ci_use=1` 的测试不能发现这一错误。

上述 stride 是地址语义，不保证任意二维视图都可直接生成 Fixpipe。dense 写回要求指令矩阵的 M/N 至少一轴为单位步幅，且当前编译器能证明它为 1。NCDHW 的逐 tap 二维视图通常不满足，因此优先按表中的批轴组织；不能为了通过编译把真实步幅改成 1。重组源视图时逐轴核对实际 L0C 地址，`from_tensor` 不会把原有数据重新排列成新声明的布局。

五维 NZ 源的轴序及地址公式见 [公共布局说明](../buffer-and-types.md#nz-的逻辑轴物理轴与-stride)。NCDHW 的 `(co,tap,ci)` 将 tap 放在矩阵 M，须核对源 M0 行步幅；NZ2DN 目的 M 方向连续，`dst_stride` 对应目的 N 方向步进，即 `dk*hk*wk`，不能因 tap 的目的步幅为 1 就要求 `dst_stride=1`。源分形行步幅则由 NZ2DN 的 channel 参数表达，核对时与目的 stride 分开。

以一个 ci 分形、`T` 个连续 tap、L0C 物理 M 跨度为 `MP` 为例，NDHWC 写回可直接表达多矩阵 Fixpipe：

```python
src = cbd.from_tensor(acc,
    layout=make_layout((T, co_use, ci_use)), data_format="nz",
    physical_layout=make_layout((T, 1, MP // 16, 16, 16),
                                stride=(MP * 16, T * MP * 16, 256, 16, 1)))
dst = cbd.from_tensor(dedw,
    layout=make_layout((T, co_use, ci_use),
                       stride=(cin_g, cin_g * dk * hk * wk, 1)),
    offset=output_element_offset * 4)
cbd.mem_copy(dst, src, cbd.copy_options(atomic="add"))
```

该片段要求 `1<=ci_use<=16`，`MP` 为真实计算时使用的 16 对齐 M 跨度。覆盖多个 ci 分形时，源 C1 stride 应跨完整 `hk*wk`，不能继续使用只有 T 个局部 tap 的紧凑布局。NCDHW/DHWCN 按表声明目的 stride，并构造与真实 L0C 相符的源视图；不能只改变目的 shape 就推定 NZ2DN 正确。

非整窗 baseN 沿用 generic 的 `head/tail/nLoopHead/lastNIdx` 状态，每次只输出当前连续核区间，跨到下一个 ci 分形后重新计算地址。同一 N 子块切换 M 时重放其起始状态；新归约段或新任务重置。输出只包含实际 co/ci，不能原子写出补齐通道。核体积为 1 时保留既有单矩阵简化分支。

普通 FP32 写回无需 `.convert()`。原子选项只作用于这次 copy，仍需验证编译产物的原子设置及恢复。初始化和段间顺序见 [sync-implementation.md](sync-implementation.md)。
