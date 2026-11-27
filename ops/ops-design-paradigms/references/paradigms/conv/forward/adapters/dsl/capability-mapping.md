# Convolution CANNBotDSL 能力与接口映射

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。通用层算法与模型见 [../../generic/index.md](../../generic/index.md)；适配层入口见 [index.md](index.md)。
>
> 本文把通用层的能力依赖（缓存容量、指令限制、数据通路、同步）映射到 CANNBotDSL 实体；平台数值一律运行时获取，文档不固化设备/版本快照。

## 1 平台参数获取

平台相关参数一律在运行时获取，文档不固化具体设备/版本数值：

- 片上容量：`cannbotdsl.get_mem_size("l1" | "l0a" | "l0b" | "l0c" | "bt")`；
- Cube core 数：`cannbotdsl.get_platform_info(stream=...)` 的 `cube_core_num`；
- 指令字段上限：以实际安装包的接口定义为准（§4 列出本适配层会预检的字段）；
- 版本差异：以实际安装的 CANNBotDSL 版本为准；跨版本行为变化按 §5 的待验证项处理。

参考实现的容量核算与选块全部由这些运行时值驱动（见 [implementation-guide.md](implementation-guide.md) §3）。

## 2 类型、布局与物理资源

### 2.1 数值类型

| 通用层概念 | DSL 绑定 |
|------------|---------|
| Tin（输入/权重存储） | `dtypes.float16` |
| Tacc（MMAD 累加） | `dtypes.float32`（L0C Buffer） |
| Tout（输出存储） | `dtypes.float16`（L0C→GM 时 `tensor.convert`） |
| BF16 输入/权重 | `dtypes.bfloat16`（本适配层未实现，见 §5 G2） |
| FP32 strict / HF32 | 本适配层未提供通路（见 §5 G2） |

### 2.2 物理资源与 Buffer 形态

| 片上资源 | DSL 实体 | 本示例的形态 | 说明 |
|----------|---------|-------------|------|
| L1（特征图行带） | `RingBuffer(MemLoc.L1, dtype, make_layout((band, CT)), depth∈{1,2}, data_format="nz", physical_layout=...)` | 1/2 缓冲，NZ | `band = (PH + (mrows−1)·Sh) · W`，物理布局 `(CT/C0, ceil16(band)/16, 16, C0)`；存放输入**整行**，一个任务可覆盖 `mrows` 个输出行 |
| L1（权重，流式） | `RingBuffer(MemLoc.L1, dtype, make_layout((weight_elements,)), depth=1, data_format="nd")` | 单缓冲 | 每个 K 分块重新装载 |
| L1（权重，驻留） | `Buffer(MemLoc.L1, dtype, make_layout((steps × weight_elements,)), data_format="nd")` | 单块 | `steps = Kd × ci_tiles`；整组权重一次装载，按步取视图复用 |
| L1（bias） | `Buffer(MemLoc.L1, dtype, make_layout((NT,)), data_format="nd")` | 单块 | 尾块需先清零 |
| BIAS | `Buffer(MemLoc.BIAS, dtypes.float32, make_layout((NT,)), data_format="nd")` | 单块 | 仅 32 位元素，容量 4096 B |
| L0A（im2col 行带） | `RingBuffer(MemLoc.L0A, dtype, make_layout((MT, K)), depth∈{1,2})` | 1/2 缓冲，NZ | `K = CT × Kh × Kw` |
| L0B（权重矩阵） | `RingBuffer(MemLoc.L0B, dtype, make_layout((NT, K)), depth∈{1,2})` | 1/2 缓冲，NZ | 与 L1 打包视图同轴序；每个归约步只读一次 |
| L0C（累加） | `Buffer(MemLoc.L0C, dtypes.float32, make_layout((MT, NT)))` × `nsubs` | 每个 M 子块一份 | 写回时 `acc[sub][0:ext, 0:n_actual].convert(fp16)`；容量 `4·MT·NT·nsubs` |

**关键约定**：

- 特征图 L1 是「整行带」：逻辑 `(band_rows·W, CT)`，`band_rows = PH + (mrows−1)·Sh`。GM→L1 是**一条连续多行 DMA**（`feature[a·W:b·W, 0:count] ← x[ci, a·W:b·W].permute(1,0)`）；左右 pad 由 `Load3DFeatureMap(pad_left, pad_right)` 硬件零填充，上下/深度越界或 Ci 尾块（`count < CT`）时对行带整体清零。
- **物理打包口径**：`physical_layout = (CT/C0, ceil16(band)/16, 16, C0)`、`stride = (band·C0, 16·C0, C0, 1)`——通道块 `c1` 的 stride 用**未补齐**的 `band`，只有最后一组的空间维补齐到 `ceil16(band)`。因此容量按 `((CT/C0 − 1)·band + ceil16(band))·C0` 计，而不是标准 NZ 的 `(CT/C0)·ceil16(band)·C0`；`band` 非 16 对齐时以该打包口径为准（末组 padding 与下一通道块起始地址重叠，但 load3d 不读取 `band` 之外的数据，故语义上无冲突）。
- 权重 L1 是「打包 ND 容器 + NZ 物理视图」：容器元素数 `NT·K + (ceil16(Kh·Kw) − Kh·Kw)·NT·C0`，`from_tensor` 的 `layout=(NT, Kh·Kw, CT)` + 手写 `physical_layout` 决定分形。
- `Buffer`/`RingBuffer` 只接受 `data_format="nd"/"nz"`；ZN 视图可由 `permute(1,0)` 或手写 `physical_layout` 构造。
- 对齐：`C0 = 32 / sizeof(dtype) = 16`（FP16）；`MT/NT/CT` 必须是 16 的倍数；`Kh·Kw` 在权重分形里向上取 16 的倍数。
- 流水：特征行带与 L0A/L0B 的 `depth` 由容量决定（`2·MT·K ≤ L0A`、`2·NT·K ≤ L0B`），不手写预取——让编译器按 Buffer 依赖排布重叠；驻留权重用无轮转 `Buffer`。

## 3 指令 ↔ DSL API 对照

本适配层采用**底层接口**，与 AscendC 的指令级对照一一对应：

| 硬件语义 | AscendC 形态（对照用） | DSL 实体 | 本示例位置 |
|----------|----------------------|----------|-----------|
| GM → L1 特征图整行带 | `DataCopy`/`DataCopyPad` | 需要时 `cube_fill_l1_zero`，然后**一条** `mem_copy(feature[row0·W:row1·W, 0:count], x[ci, row0·W:row1·W].permute(1,0))` | `_load_feature` |
| GM → L1 权重（OIHW → NZ 分形） | `DataCopy` + 手动分形排布 | `from_tensor(packed, layout=(NT,Kh·Kw,CT), data_format="nz", physical_layout=...)` + `mem_copy`；3D 按 `kd·Kh·Kw:(kd+1)·Kh·Kw` 取 slice | `_load_weight` |
| L1 → L0A 滑窗展开（LOAD3D） | `Load3D` + `FMatrix` 位域 | `c.load3d(dst, feature, feature=Load3DFeatureMap(height=band_rows, width=W, channels=CT, pad_left, pad_right), window=Load3DWindow(m_start=<输出坐标偏移>, m_extent=rows, k_extent=K), filter=..., stride=..., dilation=..., destination=Load3DDestination(k_stride_fractals=MT//16))` | `_load_a` |
| L1 → L0B 矩阵装载 | `LoadData` | `mem_copy(b_write, from_tensor(packed, layout=(NT,K), data_format="nz"))` | kernel |
| L0A×L0B → L0C（MMAD，K 归约） | `Mmad` + `unitFlag` | `c.matmul(accs[sub], a, b, init=(step==0), bias=...)`；每个 M 子块独立累加器 | kernel |
| L1 → BT（bias table） | `DataCopy` | `mem_copy(bias_table, bias_l1)` | `_load_bias` |
| L0C → GM（NZ2DN + 类型转换） | `Fixpipe` | `mem_copy(dst, result[0:rows, 0:cols].convert(fp16))` | kernel 末尾 |

**窗口语义（接口行为）**：`m_start` 以**输出坐标**计（非负；硬件按 `stride` 自行换算到输入列）。设 `start_in = m_start·Sw − pad_left`，则窗口最后一个采样点对应的输入列为 `end_in = start_in + (m_extent−1)·Sw + (Kw−1)·Dw`。当 `mrows>1` 时 `m_start = out_row·Ow + col`（展平输出偏移），窗口跨输出行时硬件按 `width` 自动换行，因此该模式要求 `Ow == W`。左右越界由 `pad_left/pad_right` 硬件零填充承担；上下/深度越界仍由行带零填承担。

**`load3d.m_extent`（接口行为）**：允许非 16 对齐（如 `m_extent=7`），但窗口必须落在 fmap 的（含 pad）有效列内，即 `end_in ≤ (W − 1) + pad_right`（等价地 `(m_start + m_extent − 1)·Sw − pad_left + (Kw−1)·Dw ≤ W − 1 + pad_right`；起点下界由 `m_start ≥ 0` 保证）。越界会读穿 L1 并触发 AICore error。整行带布局下 `m_extent=ceil16(row_count)` 会在尾块越过该界，因此尾块必须传 `m_extent=row_count`。

**M 超块与子块（接口行为）**：`matmul` 的目标必须是整块 L0C 分配；对 `acc[sub·MT:(sub+1)·MT, :]` 这类切片会报 `MatMulOp` 创建失败，`from_tensor(offset=...)` 同样不被接受。正确做法是每个子块声明独立 `Buffer`（`nsubs ≤ 8`），尾块通过“重叠重算 + 幂等写回”处理。

**RingBuffer 游标配对（接口行为）**：`depth=2` 时 `next_write()/next_read()` 必须成对；把同一个 B 分块在子块循环里 `next_read()` 多次会让写/读游标错位，数值错误（表现为整块偏差，非尾块）。

### 3.1 专用接口（`cannbotdsl.ops.conv`）不采用

DSL 0.8.2 提供一组 Conv2D 专用接口：`make_conv2d_spec` / `Conv2dSpec.tile` / `conv2d_load_fmap` / `conv2d_load_filter` / `conv2d_load_im2col` / `conv2d_store_output` / `make_conv2d_fmap_l1`（官方文档：CANNBotDSL 仓 `docs/zh/api/data-movement/convolution/` 与 `docs/zh/api/data-structures/convolution/`）。它们把几何、尾块和分形封装起来，写法更短，但**在 N 方向尾块上存在数值缺陷**：

| 规格 | 结果 |
|------|------|
| `Co=32, BN=32`（`BN == n_actual`） | 正确，maxdiff≈`9.7e-4` |
| `Co=16, BN=32`（`n_actual=16 < BN`） | **NaN** |
| `Co=32, BN=64`（`n_actual=32 < BN`） | **maxdiff≈4.65** |
| `Co=20, BN=16` / `Co=16, BN=64`（本示例路径） | 正确 |

因此本适配层的参考实现统一走 §3 的底层接口，不依赖该组专用接口；专用接口的修复状态与验证义务见 §5 G1。

## 4 约束真值：由谁校验

| 约束 | 值 | 校验位置 |
|------|-----|---------|
| stride | 1~63 | `_normalize`（拒绝）+ `_instruction_rejections` |
| dilation | 1~255 | 同上 |
| 单侧 pad | ≤255 | `_pad_pairs` |
| 每侧 pad | **< kernel**（含 3D 的 D 轴；`Kd=1` 时 D 轴 pad 必须为 0） | `_pad_pairs` |
| filter 单边 | ≤255（`MAX_FILTER`） | `_normalize` |
| `load3d.feature.width` | `W ≤ 32767` | `_instruction_rejections` |
| `Kh·Kw` / `band` | ≤65535（DMA n_value / dst_nz_c0_stride）；`band = (PH+(mrows−1)·Sh)·W` | `_instruction_rejections` |
| `m_extent` | `1..65535`，允许非 16 对齐；须满足 `(m_start + m_extent − 1)·Sw − pad_left + (Kw−1)·Dw ≤ W − 1 + pad_right` | `_instruction_rejections` + `_load_a` |
| `m_start` | `0..32767`（输出坐标；`mrows>1` 时为展平输出偏移，且 `Ow == W`） | `_load_a` + 编译器校验 |
| `k_extent` | `K ≤ 65535` | `_instruction_rejections` |
| MMAD `m/n/k` | ≤4095 | `_instruction_rejections` |
| DMA `n_value/d_value/dst_nz_*` | uint16 / uint32 字段 | `_instruction_rejections` |
| Fixpipe `dst_stride` | `Od·Oh·Ow ≤ 2^32−1` | `_instruction_rejections` |
| D 轴（3D） | 无 LOAD3D 字段；按平面逐 kd 装载，越界平面零填 | `_load_feature` |
| MT/NT 搜索上界 | 224（设计选择，非硬件上限；受 L0/L1 容量再收紧） | `MAX_TILE`：`MT/16` 即 `k_stride_fractals`，224 对应 14 |
| CT 搜索 | `{16,32,48,64,96,112,128}`，受 `K ≤ 4095` 与 L0A/L0B 容量 | `CT_STEPS` + `_resources` |
| M 超块子块数 | 1..8（受 L0C 容量与 `MAX_SUBS`） | `_search` |
| `mrows` | 1..4，且 `mrows ≤ Oh`、`Ow == W` | `_search` |
| groups | 正整数；本适配层固定 1 | `_normalize` |
| 空输出 | 返回完整 shape 空 tensor，零 launch | `_conv` |
| 片上容量 | L1/L0A/L0B/L0C/BIAS 字节预算 | `_resources` + `get_mem_size` |

`_instruction_rejections` 的字段名刻意与 AscendC 适配层 [../ascendc/instruction-mapping.md](../ascendc/instruction-mapping.md) 的约束表对齐（`load3d.*` / `mmad.*` / `dma.*` / `fixpipe.*`），便于跨语言核对。

## 5 待验证项、缺口与补充验证

| # | 项目 | 现状 | 验证义务 |
|---|------|------|---------|
| G1 | **`ops.conv` 专用接口的 N 方向尾块** | 已知缺陷（§3.1 记录）；本适配层已改用底层接口规避 | 向 DSL 维护方确认 `l1b_view/l0b_view` 在父 NZ 缓冲上的尾块视图语义；修复后可用专用接口重写并回归全部基例 |
| G2 | BF16 / FP32 strict / HF32 | 未跑 | 按 `generic/verification.md` 的数值合同逐 dtype 建 golden 与容差，HF32 不得默认开启 |
| G3 | groups>1 / depthwise | 未跑（底层权重视图天然支持逐组切片，未组合验证） | 显式 group 索引 + 每组权重切片，禁止零扩展 dense 路径 |
| G4 | channels-last（NHWC 等） | 未跑 | 需要布局归一或 kernel 内物理 stride 支持，二者择一并记录代价 |
| G5 | Conv1D / Conv3D | **已支持**：1D 升维（D=H=1）、2D 升维（D=1）、3D 真 kd 归约；矩阵 31 例（1D×2 + 2D×22 + 3D×7）通过 | 大 Kd（如 >7）使 kernel 展开变大、编译时间上升，尚未做性能/编译时间验证 |
| G6 | 双缓冲 / 多态驻留（A 驻留、ALL） | **已实现**：L1 行带 + L0A/L0B `depth=2`（由容量自动选择）+ B 驻留/流式；整行 A 驻留（full_a）未实现 | `depth=2` 已完成设备矩阵回归；full_a 需要 `m_start` 在整平面展平，可作为下一步（大空间窄 N 用例的经验上限约 1.2×） |
| G7 | AOT / Native 导出 | 未跑 | 用 `cannbotdsl.aot` collector 导出并核对 Native 命中，编译成功不等于数值通过 |
| G8 | 性能模型标定 | 已完成基于 msprof 的 tiling/结构校准（1D/2D/3D 形状集），系数仍是最小集拟合；AscendC 侧只有 2D 对比。方法与结论见 [tuning.md](tuning.md)，原始数据库外 | 扩大形状集合并用 min/p25 继续标定；补 3D 的 AscendC 对比；跨平台/版本重取 |
| G9 | 并发（多 stream / 多 Host 线程） | 未测 | 编译缓存、`_PLANS` 与 Buffer 生命周期在并发下的隔离验证 |
| G10 | Ci 尾块 + 缓冲复用 | 特征行带在 `count < CT` 时整体清零；流式权重在尾块写入前整体清零（驻留缓冲每步分块独立，启动时清零一次） | 矩阵已含 `stream_ci_tail`（流式 + `ci % ct ≠ 0`）；再补充多组 `ci % ct ≠ 0` 与冷启动残留注入（片上 L1 残留依赖分配历史，需多次冷启动复现 NaN/Inf 场景） |

## 6 数值合同

| 项 | 值 |
|------|-----|
| golden | FP64 `torch.nn.functional.conv2d`（先按 `(left,right,top,bottom)` 顺序 `F.pad`） |
| 判定式 | `abs(actual − ref) <= atol + rtol·abs(ref)`，逐元素 |
| atol / rtol | `1e-3 / 1e-3`（FP16 输入、FP32 累加、FP16 输出） |
| 特殊值 | 未验证 NaN/Inf/subnormal/负零的传播语义，不声称 IEEE 全覆盖 |
