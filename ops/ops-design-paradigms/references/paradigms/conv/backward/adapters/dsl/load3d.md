# Load3D：先排好 L1，再描述窗口

适用本 adapter 的 950/3510、B16 常规 C0=16 路径。FP32、量化和 C04/C08 的分形尺寸另按对应指令核对，不套用这里的 16。dW/dX 的轴义分别见 [dW 映射](filter/instruction-mapping.md#3-l1l0-与-mmad) 和 [dX 映射](input/instruction-mapping.md#4-load3d几何与目的步幅一起绑定)。

## 1 源布局是字节契约

`load3d` 从 L1 读取 **NC1HWC0**，不替调用方把 CHW/HWC 转成 NC1HWC0。固定一个 batch、depth 后，B16 源的元素地址应为：

```text
addr(c,h,w) = (c//16)*H*W*16 + h*W*16 + w*16 + c%16
```

`from_tensor`、`permute` 和修改 shape 只改变描述，不重排已有字节。以下两种方式都不成立：

- 按 CHW 连续拷到 L1，再用 `(H,W,C)` 视图调用 load3d。
- 按 HWC 连续拷到 L1，再把通道参数从 16 改为 64。C=16 时 HWC 恰好等于单 C1 的 HWC0；C>16 后不再等价。

L1 源可以是足够大的 ND 原始 buffer；关键是**搬入时真的形成上述地址**，不是传给 load3d 的 view 必须有几根轴。构造一个四维 shape 也不等于完成 5HD 排布。

### 连续 H/W 面：一次 DN2NZ 打包

下面位于 kernel 内，`x` 是某个固定 batch/depth 下的连续 `[C,H,W]` 面，C 是 16 的倍数。`raw_l1` 按所需跨度分配；`from_tensor.offset` 若使用，单位是字节。实际 NCDHW 大张量的 C 步幅应使用原图 `D*H_full*W_full`，不能照抄本例的 `H*W`。

```python
# import cannbotdsl as cbd
# from cannbotdsl import make_layout
src = cbd.from_tensor(x, layout=make_layout((C, H * W))).permute(1, 0)
packed = cbd.from_tensor(
    raw_l1, layout=make_layout((H * W, C)), data_format="nz",
    physical_layout=make_layout(
        (C // 16, (H * W + 15) // 16, 16, 16),
        stride=(H * W * 16, 256, 16, 1)))
cbd.mem_copy(packed, src)
```

这里 NZ 的逻辑 M=H×W、N=C，N1 就是 C1。**C1 stride 为实际 H×W×16**，不是 `alignUp(H*W,16)*16`；尾部容量可以多预留，不能因此更改有效数据的 pitch。是否只发一条搬运指令需核对编译产物。

对带源行距的裁剪窗口，不可把多行强行展平成连续面。使用 [H 为批轴的搬入示例](filter/instruction-mapping.md#22-ncdhw-与-ndhwc-的轴序)，同时让目标仍满足上述地址公式。通道尾部和插零孔洞需保证被读取区域已定义；load3d 的几何 padding 不负责初始化 L1 内部空洞。

## 2 父通道数与窗口展开长度不同

load3d 的窗口轴先按 img2col 解释，再映射到 dW/dX：

```text
img2col M = 输出空间位置，按 (oh,ow) 展平
img2col K = (c1,kh,kw,c0)，c0 最快
q = (((c1*Kh + kh)*Kw + kw)*16 + c0)
```

| 参数 | 应填写什么 |
|---|---|
| `feature.height/width/channels` | 当前 L1 源基址对应的父 fmap 几何；channels 是父通道数，不是展开后的列数 |
| `window.m_start/m_extent` | img2col 空间轴的起点/长度；dW 中对应归约 K，dX 中对应计算 M |
| `window.k_start/k_extent` | img2col `(c1,kh,kw,c0)` 轴的起点/长度；dW 中对应计算 N，dX 中对应归约 K |
| `filter/stride/dilation/pad_*` | 同一 L1 父坐标系下的窗口属性 |

例如 C=64、Kh×Kw=1×3：完整展开宽度是 192，填 `channels=64, k_extent=192`。从同一个父 fmap 只提取第二个 C1 的三个 tap，填 `channels=64, k_start=48, k_extent=48`，不需要先把 channels 改成 16。

若改用偏移后的 L1 子 fmap 基址，才按新父段重算 channels 和 k_start。不能一边移动源到第二个 C1，一边又保留相对原父段的 k_start，造成重复偏移。B16 常规 C0=16 路径的 k_start 按 16 对齐，源基址按 32B 对齐；完整范围及尾部规则以 3510 指令文档为准。

## 3 dW 的 L0B：窗口轴不等于 MMAD 轴

以下片段接上面的打包：C=64、H=2、W=18、filter=(1,3)，取前 16 个空间位置。`b0` 是足够大的 L0B buffer，按紧凑逻辑 `(192,16)` 绑定；其后 MMAD 的 RHS 也使用这套 `(N,K)` 解释。

```python
cbd.load3d(
    b0, raw_l1,
    feature=cbd.Load3DFeatureMap(height=2, width=18, channels=64),
    window=cbd.Load3DWindow(
        m_start=0, m_extent=16, k_start=0, k_extent=192),
    filter=(1, 3), config_bank="b",
    destination=cbd.Load3DDestination(k_stride_fractals=192 // 16))
```

L0B 搬入会自动转置 img2col；无需额外 `transpose=True` 或再做一次 Load2D 转置。`config_bank="b"` 选配置寄存器组，不决定数据转置，也不是 L1 的物理 bank 选择。

`k_stride_fractals` 按 **512B 分形**计，用于目的硬件排布，不是源 C1 stride。上例为紧凑 dW L0B 的 N 分形跨度 12；有容量 pitch、偏移或分多次写入时，要把实际落点与后续 MMAD 视图一起推导。L0B 起点按 512B 对齐，并计算所有落点的最大地址。分配更大的 buffer 不会自动修正错误的步幅；改变窗口 extent 也不应无意改变父布局。

这里 K 是 dW 的归约轴（load3d 的空间 M），N 是展开通道/窗口轴（load3d 的 K）。把这套轴义代入已有的 [NZ 地址公式](buffer-and-types.md#nz-的逻辑轴物理轴与-stride)，核对 destination 落点与后续计算视图。

## 4 如何隔离 load3d 错误

先用小尺寸、多个 C1、至少两个空间行的输入验证地址。用通道/位置可区分且可精确表示的数值，避免全零、全一掩盖错位。

1. **GM→L1**：按实际 DMA 描述核对 C1/H/W/C0 地址；必要时独立读回。不能仅凭目标 view 的名字认定已经转成 5HD。
2. **load3d→L0B**：用单位矩阵做 MMAD，将提取结果完整写为 ND，并与按 `(c1,kh,kw,c0)` 构造的 CPU img2col 对照。先隔离这个链路，再接复杂 dW 写回和 atomic。
3. **读回链路**：沿用 [公共布局规则](buffer-and-types.md#nz-的逻辑轴物理轴与-stride)；可先完整读回再在 Host 取第一行，避免探针自己的视图错误。
4. **扩大验证**：覆盖 channels=16/48/64、非对齐 H×W、非零 m_start、非零 k_start，再接动态参数、父容量 pitch 和真实卷积。一次端到端错误或 507015 不能单独证明通道数不支持。

归因框架前，保留最小 DSL、生成的 GM→L1/load3d/MMAD/Fixpipe 参数与预期地址，区分源打包、目的布局、容量、同步及读回问题。某一探针未通过不构成新的 dtype/通道数准入限制。

## 接口依据

- DSL 0.8.2 的 `cannbotdsl/ops/load3d.py`：`Load3DFeatureMap` 为 NC1HWC0 几何，L0B 自动转置；显式 destination 要求调用方提供匹配的存储。
- asc-devkit `docs/zh/api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_3d_arch_3510.md`：3510 源格式、通道约束、源/目的对齐与容量限制。
- 上述连续面打包及 H 批轴表达已在 DSL 0.8.2 / 3510 独立小 kernel 中验证：FP16/BF16，C=16/48/64，1×3/3×3，非对齐 H×W，非零 m/k 起点，空间块长 16/32/64；以单位矩阵 MMAD 完整读回对照 CPU img2col，9 组精确一致。生成代码为一次 fmap DN2NZ 与一次 load3d，无逐 C1 加载循环。H 批轴例使用连续 CHW、W=32，编译器将其合并为整面 DN2NZ；带源行距的裁剪窗口未在这组探针中验证。这是静态配置小 kernel 的证据，不代表动态参数、尾通道或完整卷积组合均已覆盖。
