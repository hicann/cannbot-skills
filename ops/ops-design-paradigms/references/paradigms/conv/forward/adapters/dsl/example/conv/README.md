# conv — Conv1D/2D/3D 推理算子 CANNBotDSL 参考实现（Host tiling + 多核流式 Cube kernel）

> 定位：Convolution 范式的 DSL 参考实现：**1D 升维（D=H=1）、2D 升维（D=1）、3D 真 kd 归约**共用同一计算主体；Host 侧做能力谓词与 tiling 决策，kernel 侧做多核流式迭代。
>
> 与 [../../../../adapters/ascendc/example/conv/](../../../../adapters/ascendc/example/conv2d/README.md) 同构：同一份卷积算法，两种语言的落地形态。AscendC 版用 `TilingKey/TilingData` + `FMatrix`/NZ 打包；本版把同样的语义落到 `load3d` + `from_tensor(physical_layout=...)` + `mem_copy` + `matmul`，Host 与 kernel 都是 Python。

> **为什么不用 `cannbotdsl.ops.conv` 专用接口**：`Conv2dSpec` / `conv2d_load_fmap|filter|im2col` / `conv2d_store_output` 在 N 方向尾块（`BN > n_actual`）上实测数值错误（`Co=16,BN=32` → NaN；`Co=32,BN=64` → maxdiff≈4.65），且只覆盖 2D。本实现改用底层接口。缺口证据见 [../../capability-mapping.md](../../capability-mapping.md) §5 G1。

## 支持的规格范围

| 维度 | 范围 | 实现位置 |
|------|------|---------|
| 空间维 | **1D / 2D / 3D**（1D/2D 升维到 D/H 轴；3D 真 `kd` 归约） | `_normalize` + kernel 的 `kd` 循环 |
| dtype | **FP16**（输入/权重/bias 同 dtype，FP32 累加） | `_normalize` |
| 布局 | NCW/NCHW/NCDHW 输入、OIW/OIHW/OIDHW 权重、同 rank 输出 | `_normalize` + `Problem.shape/weight_shape` |
| stride / dilation | 1~63 / 1~255（逐轴） | `_normalize` + `_instruction_rejections` |
| pad | 非对称、逐侧 < kernel（标量 / 每轴对称 / 逐侧成对） | `_pad_pairs` |
| groups | =1 | `_normalize` |
| bias | 可选一维 `(Co,)` | kernel 的 `has_bias` 编译期分支 |
| K 归约 | `(kd, ci)` 两重分块；仅首个 `(kd,ci)` 对 `init=True` | kernel 的 `kd`/`ci` 循环 |
| 尾块 | M/N/CI 三个方向均可小于切块；D 轴越界平面走零填 | 见 §5.4 |
| B 驻留 | `kd·ci_tiles × 权重块 + 行带 + bias ≤ L1` 时按 n 分组驻留复用，否则流式 | `_search` 的 `resident` |
| 空输出 | 返回完整 shape 空 tensor，**不编译、不 launch** | `Problem.empty` |

> 3D 且 `Kd=1` 时，D 轴不能用标量 padding（`pad < kernel` 会拒绝），请用逐轴 padding 如 `padding=(0, 1, 1)`。

## 文件结构

```
conv/
├── conv_dsl.py           单文件：接口契约 → 指令预检/容量核算 → tiling 搜索 → 多核 kernel → 缓存/launch
├── tests/
│   └── test_device.py    设备验收：快速集（1D/2D/3D + 缓存复用 + 空输出）+ 31 规格泛化矩阵
└── README.md             本文件
```

> 性能采集/调参脚本与原始数据不在本目录：按适配层约定留在库外验证报告；可复用的调参经验见 [../../tuning.md](../../tuning.md)。

| 本文件 | 对应 ascendc 文件 | 职责 |
|--------|------------------|------|
| `Problem` / `_normalize` / `_pad_pairs` | `conv2d_tiling_data.h` 的 `Conv2DSpec` + host 校验 | 归一化规格（含升维）、输出尺寸、空输出判定 |
| `_resources` / `_instruction_rejections` / `_cost` / `_search` | `conv2d_tiling.h` 的 `CalcConv2DTiling` | 指令字段预检、容量核算、代价模型、选块 |
| `ConvKernel` | `conv2d_kernel.cpp` | 多核任务区间 + `(od,oh,ow)` 行 + n 分组 + `(kd,ci)` 归约 + 搬运/计算/写回 |

## Tiling 决策

### 5.1 搜索空间与硬约束

```
MT, NT ∈ {16,32,48,64,80,96,112,128,160,192,224}   # MAX_TILE=224（MT/16≤14）
CT     ∈ {16,32,48,64,96,112,128}                   # 大 Ci 时减少归约步数
resident ∈ {True, False}   # B 侧整组驻留 / 逐 (kd,ci) 分块流式
depth    ∈ {2, 1}          # L1 行带 + L0A/L0B 同步双缓冲
nsubs    ∈ 1..8            # M 超块子块数（受 L0C 与 MAX_SUBS 限制）
mrows    ∈ {1,2,3,4}       # 每任务输出行数（仅 Ow==W 且 mrows≤Oh）
```

候选必须同时通过：

1. **指令字段**：`load3d.*`、`mmad.*`、`dma.*`、`fixpipe.dst_stride = Od·Oh·Ow`；
2. **片上容量**（`K = CT·Kh·Kw` 为单平面 K，D 轴按 `kd` 分块，不进入 LOAD3D）：
   ```text
   PH        = (Kh-1)·Dh + 1
   band_rows = PH + (mrows-1)·Sh
   band      = band_rows · W
   feature   = ((CT/C0 - 1)·band + ceil16(band)) · C0
   weight    = NT·K + (ceil16(Kh·Kw) - Kh·Kw)·NT·C0
   steps     = Kd · ci_tiles
   L1  = 2·(feature·depth + weight·(steps if resident else depth) + (NT if bias))
   L0A = 2·MT·K·depth   L0B = 2·NT·K·depth
   L0C = 4·MT·NT·nsubs  BIAS = 4·NT
   ```
   其中 feature 采用**打包 NZ 口径**：通道块 stride 为未补齐的 `band`，仅末组空间维补齐到 `ceil16(band)`（与 `physical_layout=(CT/C0, ceil16(band)/16, 16, C0)`、`stride=(band·C0, 16·C0, C0, 1)` 对应），不是标准 NZ 的 `(CT/C0)·ceil16(band)·C0`。

### 5.2 代价模型（字节当量，按 22 例校准）

```text
A_bytes = tasks · steps · min(band_rows,H) · W · CT · 2
W_bytes = resident ? max(groups, min(cores,tasks)) · steps · NT · K · 2
                   : tasks · steps · NT · K · 2
cost = A + W + batch·Co·Od·Oh·Ow·2
     + 4096·tasks + 1024·tasks·steps + 4096·tasks·steps·nsubs
cost *= (channel_tiles·NT/Co)²                  # N 尾块浪费
cost *= width_tiles·nsubs·MT/(mrows·Ow)         # M 尾块浪费
cost *= 1 + 0.5·(ceil(tasks/cores)·cores/tasks - 1)   # 负载不均（半权）
cost *= sqrt(cores/tasks)  if tasks < cores     # 小输出：空转核
cost *= 1 - 0.25·(steps-1)/steps  if depth=2    # 双缓冲随归约步数生效
```

系数只用于**同一套候选间排序**，不预测绝对耗时；并列时优先 resident → 更深流水 → 更大 `mrows/nsubs` → 更小 L1。`_PLANS` 按 `(Problem, cores, 容量快照)` 缓存决策（单次 `_search` 约 2 ms）。

### 5.3 任务映射与多核

```text
width_tiles = 1                    # M 超块覆盖整行或整行组
oh_groups   = ceil(Oh/mrows)       channel_tiles = ceil(Co/NT)
spatial_tasks = Od · oh_groups     groups = batch · channel_tiles
tasks = groups · spatial_tasks
```

- 扁平任务序：**n 分组在外、输出行组在内**；每核取一段连续区间（`base/extra` 均分），同一核连续命中同一 `(batch,n)` 分组，B 侧权重可复用；
- `launch_cores = min(device_cores, tasks)`；核内按 `get_block_num()` 重算区间，`tasks` 少于核数时多余 block 区间为空。

### 5.4 尾块、多行展平与 D 轴

- 一个任务覆盖 `mrows` 个输出行：`total_m = row_count·Ow`，`M` 以展平输出坐标寻址（`m_start = out_row·Ow + col`），因此 `mrows>1` 要求 `Ow == W`；
- `load3d.m_extent` 允许非 16 对齐，但窗口不得越过 `W + pad_left + pad_right`；越界会触发 AICore error；
- M 尾块：`sub_begin = min(out_col + sub·MT, last_begin)`，尾块与前一子块**重叠重算**并重复写回（值相同，幂等）；
- `n_actual = min(NT, Co - channel_start)`、`ci 有效通道 = min(CT, Ci - ci_start)`；
- 3D 的 `input_depth = od·Sd − pad_front + kd·Dd`；越界平面整块零填，只参与 K 归约；
- 上下越界：只搬 `[max(0,input_row), min(H, input_row+band_rows))`，其余零填；左右越界交给 `Load3DFeatureMap(pad_left,pad_right)`。

## Kernel 迭代骨架

```python
for task in 本核连续区间:                       # n 分组在外、输出行组在内
    group, spatial = task // spatial_tasks, task % spatial_tasks
    batch, n_tile  = group // channel_tiles, group % channel_tiles
    row_index = spatial // width_tiles
    out_depth  = row_index // oh_groups
    out_row    = (row_index % oh_groups) * mrows
    ...
    if resident and group 变化:                  # 动态 if + 循环携带 last_group
        逐 (kd,ci) 装载整组权重；装载 bias
    step = 0
    for kd in range(Kd):                         # 编译期展开
        for ci in range(ci_tiles):               # 编译期展开
            _load_feature(...)                   # 整行带一条多行 DMA（必要时零填）
            if not resident: 装载当前 (kd,ci) 权重块
            mem_copy(L0B, L1 权重 NZ 视图)
            b = right.next_read()                # 每步只读一次，供所有子块复用
            for sub in range(nsubs):             # 编译期展开
                _load_a(..., sub_begin, ext)
                matmul(accs[sub], a, b, init=(step == 0), bias=bias_table)
            step += 1
    for sub in range(nsubs):                     # 编译期展开
        mem_copy(输出视图[...], accs[sub][0:ext, 0:n_actual].convert(fp16))
```

## 运行与验收

环境：可用 NPU、`cannbotdsl >= 0.8.2`、`torch_npu`；平台参数（容量/核数）运行时读取，见 [../../capability-mapping.md](../../capability-mapping.md) §1。

```bash
source <CANN>/set_env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
python tests/test_device.py -v      # 设备验收（需 NPU；不在仓库普通 CI 执行）
```

**NPU 测试不在仓库普通 CI 中执行**；仓库的 `tests/test_language_route.py` 只检查资料入口与语言路由，不代表算子编译或上板验收。

验收范围：

- 快速集（`tests/test_device.py::ConvDeviceTests`）：1D/2D/3D 各两例 + 程序缓存复用 + 空输出零编译；
- 泛化矩阵（`tests/test_device.py::ConvMatrixTests`）：31 个数值规格（1D×2 + 2D×22 + 3D×7）+ 空输出，含 Co/Ci 尾块、1x1/大核、stride/dilation、非对称 pad、驻留/流式回退、3D；
- 判定：逐元素 `|a−r| ≤ atol + rtol·|r|`（`1e-3 / 1e-3`）；方法与通过判定以 [../../../../generic/verification.md](../../../../generic/verification.md) 为准。

## 性能与调优

- 口径：msprof `--task-time=l1 --ai-core=on` 的 **kernel 时间**（不含 Host 分发/编译），采样与归属校验方式见 [../../tuning.md](../../tuning.md) §1；
- 结论：与 native 同量级（具体数据与对比留在库外验证报告）；
- 有效结构手段、无效/有害尝试、结构上限判断与代价模型校准方法见 [../../tuning.md](../../tuning.md)；原始数据、脚本与环境快照不入库。

## 关键正确性约束（实现/移植时勿回退）

| 约束 | 位置 | 说明 |
|------|------|------|
| 空输出零 launch | `_conv` | 返回前不编译、不发射 kernel |
| `load3d.m_extent` | `_load_a` | 允许非 16 对齐；须满足 `(m_start+m_extent−1)·Sw − pad_left + (Kw−1)·Dw ≤ W−1+pad_right` |
| `m_start` 语义 | `_load_a` | 以输出坐标计、非负；`mrows>1` 时用展平输出偏移，且要求 `Ow == W` |
| M 子块累加器 | kernel | 每个子块独立 L0C `Buffer`；切片/`from_tensor` 偏移作为 matmul 目标会被拒 |
| B 侧游标配对 | kernel | 每归约步 `next_write()` 一次、`next_read()` 一次；子块复用同一个 `b` |
| D 轴平面越界 | `_load_feature` | `0 ≤ input_depth < D` 才搬，否则整块保持零填 |
| 行带钳制 | `_load_feature` | 只搬有效行 `[max(0,input_row), min(H,input_row+band_rows))`；左右列由 load3d pad 承担 |
| 零填后必须屏障 | `_load_feature`/kernel | 行带/权重清零与随后的 MTE2 拷贝之间 `cube_sync_pipe(PIPE.MTE2)` |
| 权重只清零一次 | kernel | 驻留：启动时整块清零（分形尾部保持零）；流式：缓冲被复用，Ci 尾块写入前需重新清零 |
| Ci 尾块确定性 | `_load_feature`/`_load_weight` | `count < CT` 时行带整体清零（避免 0×NaN）；流式权重同样在写入前清零，避免 K 尾块残留上一步权重 |
| 仅首个归约步初始化 | kernel | `init=True` 只在 `step==0`（`kd=0, ci=0`），其余 `init=False` |
| 驻留分组切换 | kernel | `last_group` 循环前初始化（`dtypes.int64(-1)`），按分组变化重载 |
| `mrows` 限制 | `_search` | 仅 `Ow == W` 且 `mrows ≤ Oh`；1D 的 `Oh=1` 自动退化为 1 |
| 任务区间边界 | kernel | `base/extra` 均分；`tasks < block_num` 时多余 block 区间为空 |
| bias 尾块清零 | kernel | `n_actual < NT` 时 BIAS 段多余通道必须为零 |
| 3D 通道步进 | `_load_weight` | 权重按 `(Co,Ci,Kd·Kh·Kw)` 取 `kd·rs:(kd+1)·rs`；跨通道步进是 `Kd·Kh·Kw` 而非 `Kh·Kw` |
| 编译缓存键 | `_conv` | `(Problem, tile, resident, device, DSL 版本)`；同规格复用、异规格重编译 |

## 验证边界

- 本示例交付 Host tiling + kernel 主体、1D/2D/3D 快速验收、泛化矩阵与 tiling 抽样计时；
- 不在本示例范围：aclnn/GE 注册与 `<<<>>>` 直调（上层工作流）、AOT/Native 发布、多 stream/多 Host 线程并发、BF16/FP32/HF32、groups>1、channels-last；
- 性能结论只在本机、本轮抽样下成立；共享 NPU 上以 min/p25 为稳定信号，不构成对其它平台/版本的性能承诺。
