# Convolution CANNBotDSL 实现指引

> **层级**：CANNBotDSL 语言适配层叶子文档。通用层算法见 [../../generic/index.md](../../generic/index.md)，能力真值与待验证项见 [capability-mapping.md](capability-mapping.md)，可运行实现见 [example/conv/](example/conv/README.md)。
>
> 本文只写「通用层的卷积算法如何落到 DSL 的三层结构与具体实体」，不重复推导切分算法（见 [../../generic/tiling.md](../../generic/tiling.md)）与数据流语义（见 [../../generic/compute-flow.md](../../generic/compute-flow.md)）。

## 1 三层职责

| 层 | 本适配层的划分 | 边界 |
|------|---------------|------|
| 公开入口 | `conv1d` / `conv2d` / `conv3d`（统一走 `_conv(rank, ...)`） | 校验连续性/设备/dtype、升维归一化、分配输出；**不把设备计算藏在入口** |
| Host | `Problem` / `_normalize` / `_resources` / `_instruction_rejections` / `_cost` / `_search` / `_compile` | 指令预检、容量核算、代价模型选块、任务数、编译缓存；可用 `torch.empty`，不做数值变换 |
| Kernel | `ConvKernel.kernel`（`@c.kernel`）+ `run/run_without_bias`（`@c.host`） | 真实 DSL 计算：GM→L1→L0A/L0B→L0C→GM，无 torch/NPU 算子替代，无 fallback |

切分结果直接作为 kernel 对象的 Python 属性，在 `@c.kernel` 追踪时成为编译期常量；运行期只通过 `core_num` 标量与 Tensor 传入。tiling 决策按 `(Problem, cores, 容量快照)` 缓存（`_PLANS`），避免每次调用重跑搜索。

## 2 数据流骨架 → DSL 落地

通用层的四侧数据流（[../../generic/compute-flow.md](../../generic/compute-flow.md) §2）在底层接口上的落地：

| 通用层阶段 | DSL 落地 | 备注 |
|-----------|----------|------|
| fmap：GM → L1（整行带） | 仅在上下/深度越界时 `cube_fill_l1_zero(band)` + `cube_sync_pipe(MTE2)`；随后一条 `mem_copy(band[row0·W : row1·W, 0:count], x[ci, plane, row0·W:row1·W].permute(1,0))` | 行带按**输入整行宽 W** 存放，连续多行一次搬完（`row1-row0 = band_rows`）；左右越界由 load3d 的 `pad_left/pad_right` 处理，不再逐行搬运 |
| weight：GM → L1（OIHW → NZ） | 启动时整块零填一次 → 每个 `(kd, ci)` 步 `mem_copy(NZ 视图, weight[co, ci, kd·rs:(kd+1)·rs].permute(0,2,1))` | 分形尾部的零只在启动时写一次；驻留模式下每个 `(batch,n)` 分组搬一次整组权重 |
| fmap：L1 → L0A（滑窗展开） | `load3d(dst, band, Load3DFeatureMap(height=band_rows, width=W, channels=CT, pad_left, pad_right), Load3DWindow(m_start=<展平输出偏移>, m_extent=rows, k_extent=K), destination=Load3DDestination(k_stride_fractals=MT//16))` | `m_start` 以**输出坐标**计（步长由硬件乘）；左右 pad 由硬件零填充；上下/深度越界由行带零填承担 |
| weight：L1 → L0B | `mem_copy(b_write, from_tensor(packed, layout=(NT,K), data_format="nz"))` | 每个归约步只读一次（`next_read()` 与写入配对），供全部 M 子块复用 |
| MMAD | `matmul(acc[sub], a, b, init=(step==0), bias=...)` | 每个 M 子块一个独立 L0C 累加器；`init` 为 i1 SSA，运行期选择初始化/累加 |
| output：L0C → GM | `mem_copy(output[ch, row_base+sub_begin : ...].permute(1,0), acc[sub][0:ext, 0:n_actual].convert(fp16))` | 每个子块单独写回；输出按 `(Co, Od·Oh·Ow)` 展平，天然支持跨输出行 |

**M/N/K 三元组**：M 为输出展平（`Od·Oh·Ow`，一个任务可覆盖 `mrows` 个输出行），N = `Co`，K = `Kd × Ci × Kh × Kw`。K 切成 `Kd × ci_tiles` 步、每步 `CT × Kh × Kw`；M 用「超块 = `nsubs × MT`」覆盖连续输出位置；A 侧整行带一次搬入。

> 为什么不用 `ops.conv` 专用接口：其 N 方向尾块存在数值缺陷，见 [capability-mapping.md](capability-mapping.md) §3.1 与 §5 G1。

## 3 Host 侧 tiling 决策

### 3.1 搜索空间与硬约束

```text
MT, NT ∈ {16,32,48,64,80,96,112,128,160,192,224}   # MAX_TILE=224
CT     ∈ {16,32,48,64,96,112,128}                  # 大 Ci 时减少归约步数
resident ∈ {True, False}     # B 侧整组驻留 / 逐 k 分块流式
depth    ∈ {2, 1}            # L1 行带 + L0A/L0B 同步双缓冲
nsubs    ∈ 1..8              # M 超块子块数（受 L0C 与 MAX_SUBS 限制）
mrows    ∈ {1,2,3,4}         # 每任务输出行数（仅 Ow==W 且 mrows≤Oh 时开放）
```

每个候选依次通过：

1. **指令字段** `_instruction_rejections`（`load3d.*` / `mmad.*` / `dma.*` / `fixpipe.*`）；
2. **片上容量** `_resources`（L1/L0A/L0B/L0C/BIAS vs `get_mem_size`）；
3. **代价模型** `_cost` 排序。

容量公式（与 Buffer 形态一一对应）：

```text
PH        = (Kh-1)·Dh + 1
band_rows = PH + (mrows-1)·Sh
band      = band_rows · W
feature   = ((CT/C0 - 1)·band + ceil16(band)) · C0
K         = CT · Kh · Kw
weight    = NT·K + (ceil16(Kh·Kw) - Kh·Kw)·NT·C0

L1  = 2·( feature·depth + weight·(steps if resident else depth) + (NT if bias) )
L0A = 2·MT·K·depth      L0B = 2·NT·K·depth
L0C = 4·MT·NT·nsubs    BIAS = 4·NT
```

`ceil16(band)` 是 NZ 分形对行数向上取整的填充；`ceil16(Kh·Kw)` 是权重 NZ 分形对 K 方向的填充。两者都必须计入容量。

### 3.2 代价模型

```text
band_used = min(band_rows, H)
A_bytes   = tasks · steps · band_used · W · CT · 2
W_bytes   = resident ? max(groups, min(cores, tasks)) · steps · NT · K · 2
                     : tasks · steps · NT · K · 2
O_bytes   = batch · Co · Od · Oh · Ow · 2
cost = A + W + O
     + TASK_OVERHEAD·tasks + KLOOP_OVERHEAD·tasks·steps + SUB_OVERHEAD·tasks·steps·nsubs
cost *= (channel_tiles·NT/Co)²                # N 尾块浪费
cost *= width_tiles·nsubs·MT/(mrows·Ow)       # M 尾块浪费
cost *= 1 + 0.5·(waves·cores/tasks - 1)       # 负载不均（半权）
cost *= sqrt(cores/tasks)  if tasks < cores   # 小输出：空转核额外惩罚
cost *= 1 - 0.25·(steps-1)/steps  if depth=2  # 双缓冲随归约步数生效
```

- `TASK_OVERHEAD_BYTES=4096`、`KLOOP_OVERHEAD_BYTES=1024`、`SUB_OVERHEAD_BYTES=4096` 是按 22 例 msprof 抽样校准的量级假设，只用于候选间排序，不预测绝对耗时；
- 双缓冲收益按「可重叠窗口 ∝ steps-1」折算：只有 1 个归约步时不给折扣（避免点卷积误选 depth=2）；
- 并列时优先 `resident` → 更深流水 → 更大的 `mrows`/`nsubs` → 更小 L1；
- 系数的校准顺序、结构上限判断与无效尝试见 [tuning.md](tuning.md)；原始测量数据与脚本留在库外验证报告。

### 3.3 多核与任务映射

```text
width_tiles   = 1                              # M 超块覆盖整行或整行组
oh_groups     = ceil(Oh/mrows)
spatial_tasks = Od · oh_groups
groups        = batch · channel_tiles          channel_tiles = ceil(Co/NT)
tasks         = groups · spatial_tasks
```

- 扁平任务序为 **n 分组在外、m 在内**；每核取一段连续区间：
  `base = tasks // block_num`、`extra = tasks % block_num`、
  `begin = block·base + min(block, extra)`、`end = begin + base + (1 if block < extra else 0)`；
- 连续区间保证同一核连续命中同一 `(batch,n)` 分组，B 侧权重可复用；`tasks < block_num` 时多余 block 区间为空；
- 发射核数 `launch_cores = min(device_cores, tasks)`，核内按实际 `get_block_num()` 重算区间。

> 与 ascendc 的「完整 team」不同：DSL 侧用连续区间而不做 team 反解，因为**权重复用要求分组连续**；空区间天然安全，不需要额外的空尾核判界。

### 3.4 空输出与缓存

- 任一空间轴输出为 0：返回完整 shape 空 tensor，在 tiling 之前返回，不编译、不 launch；
- 编译缓存键 = `(Problem, tile, resident, device, cannbotdsl.__version__)`；
- tiling 缓存键 = `(Problem, cores, 容量快照)`；`close()` 同时清空两者。

## 4 Kernel 骨架

```python
@c.kernel
def kernel(self, x, weight, bias, output):
    features = RingBuffer(L1, fp16, (band, CT), depth=depth, "nz", physical_layout=...)
    weights  = Buffer(L1, fp16, (steps*weight_elements,), "nd") if resident else \
               RingBuffer(L1, fp16, (weight_elements,), depth=1, "nd")
    left     = RingBuffer(L0A, fp16, (MT, K), depth=depth)
    right    = RingBuffer(L0B, fp16, (NT, K), depth=depth)
    accs     = [Buffer(L0C, fp32, (MT, NT)) for _ in range(nsubs)]   # 每个子块一份
    if has_bias: bias_l1 = Buffer(L1, fp16, (NT,), "nd"); bias_table = Buffer(BIAS, fp32, (NT,), "nd")

    # 权重缓冲只在启动时整块清零一次，之后只覆写有效 rs 行/通道
    if resident: _fill_zero(weights, steps*weight_elements//C0)
    else:
        for _ in range(depth): _fill_zero(weights.next_write(), weight_elements//C0)
    c.cube_sync_pipe(PIPE.MTE2)

    # 本核连续任务区间
    base, extra = tasks // block_num, tasks % block_num
    begin = block*base + (block if block < extra else extra)
    end = begin + base + (1 if block < extra else 0)
    last_group = c.dtypes.int64(-1)
    for task in range(begin, end):
        group, spatial = task // spatial_tasks, task % spatial_tasks
        batch, n_tile  = group // channel_tiles, group % channel_tiles
        row_index = spatial // width_tiles
        out_depth, out_row = row_index // oh_groups, (row_index % oh_groups) * mrows
        out_col = (spatial % width_tiles) * ms           # ms = nsubs·MT（width_tiles=1 时为 0）
        row_count = min(mrows, Oh - out_row)
        total_m   = row_count * Ow
        band_rows = PH + (row_count - 1)·Sh
        input_row = out_row·Sh - pad_top
        input_depth = out_depth·Sd - pad_front
        row_base  = out_depth·Oh·Ow + out_row·Ow
        last_begin = (max(out_col, out_col+total_m-MT) if mrows > 1
                      else max(0, Ow-MT))
        if resident:
            if group != last_group:                 # 动态 if，循环携带 last_group
                for kd in range(Kd):                 # 编译期展开
                    for chunk in range(ci_tiles):
                        _load_weight(weights, weight, n_tile, kd, chunk,
                                     (kd*ci_tiles+chunk)*weight_elements*2)
                if has_bias: _load_bias(...)
                last_group = group
        elif has_bias:
            _load_bias(...)
        step = 0
        for kd in range(Kd):                         # 编译期展开
            for chunk in range(ci_tiles):            # 编译期展开
                feature = features.next_write()
                _load_feature(feature, x, batch, input_depth + kd·Dd,
                              input_row, band_rows, chunk)
                if resident: packed = from_tensor(weights, (weight_elements,), offset=...)
                else:        packed = weights.next_write(); _load_weight(...)
                mem_copy(right.next_write(), from_tensor(packed, (NT,K), "nz"))
                b = right.next_read()                # 每个 step 只读一次，供所有子块复用
                for sub in range(nsubs):             # 编译期展开
                    sub_begin = min(out_col + sub·MT, last_begin)
                    ext = min(MT, out_col + total_m - sub_begin)
                    _load_a(left.next_write(), feature, sub_begin, ext)
                    matmul(accs[sub], left.next_read(), b, init=(step == 0), bias=...)
                step += 1
        for sub in range(nsubs):                     # 编译期展开
            sub_begin = min(out_col + sub·MT, last_begin)
            ext = min(MT, out_col + total_m - sub_begin)
            mem_copy(output[ch, row_base+sub_begin : ...].permute(1,0),
                     accs[sub][0:ext, 0:n_actual].convert(fp16))
```

要点：

1. **循环变量是运行期 i64**：`task/spatial/out_row/out_col/row_count/band_rows` 都在 kernel 内推导，切块不随形状重编译；
2. **驻留分组切换用动态 `if`**：`last_group` 必须在循环前初始化（`c.dtypes.int64(-1)`），否则跨分支状态不合法；
3. **`m_extent` 允许非 16 对齐**，但不得越过行带末端：`m_start` 以输出坐标计、由硬件乘步长，`m_start + (m_extent-1)·Sw + Kw ≤ W + pad_left + pad_right`；`mrows>1` 时窗口跨行，要求 `Ow == W`；
4. **每个 M 子块必须有独立 L0C Buffer**：`acc[sub]` 用切片/`from_tensor` 偏移作为 `matmul` 目标会被拒（`MatMulOp` 只接受整块累加器）；
5. **B 的读写游标必须配对**：`right.next_write()` 每步一次、`next_read()` 也只每步一次；子块循环里重复 `next_read()` 会让 depth=2 的游标错位（实测数值错误）；
6. **`init` 用 i1 SSA**：`init=(step == 0)` 由编译器包成 `scf.if`，不要外层 Python 分支复制整段 MMAD；
7. **bias 只在首块参与**：`matmul(..., bias=...)`，不额外后加。

## 5 Buffer 规划与同步

- **L1 行带**：`RingBuffer(depth=2)` 时第 i+1 步的搬入可覆盖第 i 步的消费之前的 WAR 依赖；`RingBuffer(depth=1)` 为纯串行。
- **L0A/L0B**：与行带同深度；`depth=2` 需要 `2·MT·K ≤ L0A`、`2·NT·K ≤ L0B`，否则搜索自动退回 1。
- **同步**：Tensor 访问依赖由编译器推导；显式屏障只剩「整块零填 → 后续覆盖」之间的 `cube_sync_pipe(PIPE.MTE2)`（启动时的权重、上下/深度越界时的行带、bias）。
- **零填策略**：左右 pad 由 load3d 硬件零填充，行带只在上下/深度越界时清零；权重分形尾部只在启动时清零。实测这一步在 22 例上带来 1.0~1.6×（G 7%、B 13%、S 17%）。
- **行带布局与 load3d**：`(band, CT)` 逻辑视图 + `(CT/C0, ceil16(band)/16, 16, C0)` 物理布局；`Load3DFeatureMap.height` 取 `band_rows`（`mrows>1` 时允许窗口跨行），`pad_left/pad_right` 直接传真实 pad。

## 6 尾块与边界

| 场景 | 处理方式 |
|------|---------|
| M 尾块 | 子块起点 `sub_begin = min(out_col+sub·MT, last_begin)`，尾块与前一子块重叠重算并重复写回（值相同，幂等）；写回按 `ext` 截断 |
| N 尾块 | 只搬有效输出通道；L0B 视图取 `0:n_actual`；BIAS 段先清零再覆盖；写回截断到 `n_actual` |
| Ci 尾块 | `count = min(CT, Ci-ci_start)`，K 方向其余位置保持零填 |
| D 轴越界（3D） | `input_depth ∉ [0,D)` 时该平面整块保持零填，仅参与 K 归约 |
| 输入上下越界 | 只搬 `[max(0,input_row), min(H,input_row+band_rows))`，其余零填，不越界读 GM |
| 输入左右越界 | 由 `Load3DFeatureMap(pad_left, pad_right)` 硬件零填充；行带本身只存 `[0,W)` |
| 全 padding 窗口 | `valid_end<=valid_begin` 时整块为零（行带零填），输出为 bias 或 0 |
| 任务区间 | 连续区间均分；空区间不执行，无「空尾核反解」问题 |
| 驻留分组切换 | 每核在分组边界重载整组权重；同组内 m 任务复用 |

## 7 与 AscendC 适配层的差异

| 主题 | AscendC 适配层 | 本适配层 |
|------|---------------|---------|
| 切分结果载体 | `TilingData` POD 结构体 | kernel 对象属性（编译期常量）+ 运行期标量 |
| 多路径分发 | `TilingKey` + 模板枚举 | 无 TilingKey；`(Problem, tile, resident, depth, mrows)` 即不同的已编译程序 |
| 切分决策 | 因子搜索 + 代价模型 + fullLoad 四态 | 有界搜索（MT/NT≤224、CT≤128、L1/L0 深度、M 超块、每任务行数）+ 解析代价模型 + B 驻留/流式两态 |
| A 侧装载 | 行带 + 最坏情形行推理 | 整行带宽 `W`、连续多行一条 DMA；左右 pad 交给 load3d |
| B 侧 | fullLoad 四态（A/B/ALL/NONE） | 流式 或 B 驻留（按 L1 是否容纳整组权重）；一个 B 分块被多个 M 子块复用 |
| FMatrix / LOAD3D | 手填全部位域 | 手填 `Load3DFeatureMap/Window/Destination`，`m_start` 用输出坐标，pad 由硬件承担 |
| 流水 | `TQue` 自动事件 / 手写 flag | `RingBuffer(depth=2)` 让编译器排布重叠；无手写预取 |
| unitFlag | Mmad 末步 `unitFlag=3` | `matmul(init=(step==0))` |
| 1D/3D | 1D 升维、3D od/kd 外层 | 已实现：1D 用 D=H=1，2D 用 D=1，3D 走 `kd` 归约循环；`mrows>1` 只作用于 H 轴且需 `mrows≤Oh`；1D/2D/3D 均有设备用例（见 `example/conv/tests/test_device.py`） |

**专用接口的取舍**：`cannbotdsl.ops.conv` 的 `Conv2dSpec`/`conv2d_load_*` 能显著缩短代码，但 N 方向尾块实测错误（[capability-mapping.md](capability-mapping.md) §3.1），因此参考实现选择底层接口。

## 8 验证义务

通用验证方法（用例设计、输入/golden 生成、混合容差判定、失败定位）以 [../../generic/verification.md](../../generic/verification.md) 为准，本文不重复；适配层示例的设备验证落地见 [example/conv/tests/test_device.py](example/conv/tests/test_device.py)。

本适配层只补充 DSL 特有的义务：

- 能力/指令字段相关结论按 [capability-mapping.md](capability-mapping.md) §5 声明假设与验证义务；
- 数值判定按 §6 的逐元素混合容差（`|a−r| ≤ atol + rtol·|r|`），不得用全局最大容差代替，也不得在失败后放宽；
- 编译通过、IR 校验通过都不等于数值正确，必须用独立 golden 的设备用例；
- 设备测试需要 NPU，不在仓库普通 CI 执行（CI 只做资料/路由检查）；
- 性能口径、调优方法与经验按 [tuning.md](tuning.md) 记录，原始测量数据不入库。
