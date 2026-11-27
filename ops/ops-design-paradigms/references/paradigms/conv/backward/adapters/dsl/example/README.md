# 可运行的 dW / dX DSL 样例（950）

这些样例演示 adapter 中的布局、搬运、计算与同步如何组成完整计算链，支持范围见下表。范围外的输入会拒绝，不调用 Torch 卷积兜底。Transpose 仅提供 Host 入口，复用 dX 的同一设备程序。

依赖：`cannbot-dsl 0.8.2`、与本机 CANN 配套的 `torch/torch-npu`，目标 `dav-3510`。先按本机方式加载 CANN 环境。设备测试结果见本文末尾；性能采集时使用空闲设备，避免其他任务干扰。

## 文件与阅读顺序

| 文件 | 阅读重点 |
|---|---|
| [common.py](common.py) | 当前流核配额、L0 槽预算、任务波次利用率、程序生命周期 |
| [dw.py](dw.py) | 多 C1 打包、Load3D→MMAD、父段预取、批量 NZ2DN，尤其是 `BM/16` 对写回步幅的影响 |
| [dw_splitk.py](dw_splitk.py) | 跨核切 K、独占 FP32 partial、固定顺序 UB 合并；复用 dW 计算链 |
| [dx.py](dx.py) | H 批轴插零搬入、P1 权重布局与逆序转置 Load2D、跨 M 权重驻留、动态 K 尾块、一次 Fixpipe 转换输出 |
| [verify_splitk.py](verify_splitk.py) | 切 K 自动/手动分片、完整两阶段计时、不切分对照与工作区检漏 |
| [verify.py](verify.py) | 从实际量化输入生成 CPU double Golden、复用旧 program 处理新 shape、逐位复跑、设备任务计时及 ASC 导出 |

## 当前支持范围

| 项目 | dw.py（不切 K） | dx.py / 无 bias Transpose |
|---|---|---|
| dtype / format | FP16、BF16 输入；NCDHW；FP32 输出 | FP16、BF16；NCDHW；输出同输入类型 |
| 卷积属性 | groups=1、pad=0、dilation=1、stride=1 | groups=1、pad=0、dilation=1；D stride=1，H/W 可前放大 |
| kernel | 正尺寸，Kh×Kw≤9；D 轴运行时循环 | 正尺寸，Kh×Kw≤9；D 轴运行时循环 |
| 通道与尾块 | Ci/Co 可不对齐；M/Ci/K 尾块 | Co 是16倍数；Ci、空间 M 可不对齐；K 尾块 |
| 资源范围 | 完整宽度的 x 行带≤192 KiB、dY 父段≤64 KiB；双份父段 | 扩展 dY 面≤128 KiB；当前 Ci 块完整权重≤128 KiB |
| 累加/所有权 | 完整归约≤65535，单任务唯一拥有输出 | 单任务拥有输出，归约所有有效 Co/tap/kd |
| 未演示部分 | 长 K 分段原子写回、拆核窗口、切宽、其他 format | 局部行带裁剪、Co 分段父窗口、reverse_only、bias、其他 format |

跨核切 K 的范围见下文[扩展示例](#跨核切-k-示例)：它沿用 dw.py 的卷积属性，可用多片处理总 K>65535。

容量检查考虑对齐和槽数。`prepare()` 只做几何、tiling、分配和编译绑定，不读取张量值、不重排权重。动态 shape、kernel 尺寸、stride、任务 chunk、核数都从运行参数获得；dtype 和片上容量为编译配置。输入的 `requires_grad` 与样例执行无关。

dW 对固定 kd 的完整矩阵轴为 `M=Co`、`N=Ci*Kh*Kw`、归约 `K=Nbatch*Do*Ho*Wo`，其中 Nbatch 是输入 batch。基本块 `tile=(BM, CI_TILE, BK)` 的第二项不是 L0 N：`Nblock=CI_TILE*Kh*Kw`，列顺序是 `[ci1,kh,kw,ci0]`。例如 3×3、CI_TILE=32 时 N=288，双份 B16 L0B 配 BK=64 超过64 KiB，因此样例的布局验证使用 BK=32。**这是该份数/块组合不合法，不是多 Ci 无法实现。**

dX 使用 `tile=(BM,BN,BK)`，M 是单个 D 面内展平的 `Hi*Wi`，N 是 Ci，K 沿 Co 与 Kh×Kw 归约。空间尾块使用 `mc=align16(mv)`：Load3D 请求实际 `mv` 个点，而 L0A、MMAD、L0C 和 Fixpipe 统一使用 mc 的物理 pitch。特别是1×1/pad0不能在最后3个点仍请求整块32点。

完整 weight 驻留的字节数是 `Kd*Kh*Kw*Co*BN*2`，不能只预算一个 kd。NCDHW 的小核 reverse_only 准入不属于本例，入口会拒绝对应场景。

## 运行与复用

在本目录执行；产物默认放 `/tmp`，不会写进 skill。两个验证脚本都接受 `--dtype fp16/bf16/all`，也兼容 `float16/bfloat16` 拼写：

```bash
python verify.py --op all --dtype all --blocks 32 --profile --output /tmp/conv-example-32
python verify.py --op all --dtype all --blocks 36 --output /tmp/conv-example-36
```

`--blocks` 是请求的 launch 上限，不能超过当前流 AIC 配额；实际启动数还受任务数限制。测试某个上限不等于所有 case 都用了这么多核，`results.json` 保存实际 blocks。Kernel 的任务步进使用 `get_block_num()`，没有写死32/36。计划分别给出已启动核的 `launch_wave_util` 和以请求配额为分母的 `quota_wave_util`；例如8个任务、32核配额，前者可能为1，后者仅为0.25。

**核数扫描须保持任务覆盖完整。** 每个档位通过 `prepare(..., blocks=...)` 重新生成计划并绑定启动参数。若其他实现使用 meta 中的 `used_core_num` 分配任务，该值必须与实际 launch 核数一致，不能只修改 launch 参数。这里的样例按 `get_block_num()` 轮转：32个任务启动1核时，该核完成32个任务；启动32核时，每核完成1个任务。比较前应验证每个档位的完整输出，不能把只完成部分任务的耗时当成单核算子耗时。详细约定见 [Launch 与任务分配](../kernel-entrance.md#31-launch-与任务分配使用同一核数)。

入口均接收连续的五维 NPU Tensor，权重的轴与 fmap 不同：

| 入口 | 输入与输出 |
|---|---|
| `dw.prepare(x, dy, kernel_size)` / `dw_splitk.prepare(...)` | x=`[N,Ci,Di,Hi,Wi]`，dy=`[N,Co,Do,Ho,Wo]`，返回 FP32 `[Co,Ci,Kd,Kh,Kw]` |
| `dx.prepare(dy, weight, input_shape)` | dy=`[N,Co,Do,Ho,Wo]`，weight=`[Co,Ci,Kd,Kh,Kw]`，返回与 dy 同 dtype 的 `[N,Ci,Di,Hi,Wi]` |
| `dx.prepare_transpose(x, weight, ...)` | x 使用上行 dy 的轴定义，weight 不变；Host 推导输出 shape 后复用 dX |

2D 用例对 fmap/dy 和 weight 使用 `unsqueeze(2)` 插入大小为1的 D/Kd 轴，H/W 参数补成三元组；这些入口不直接接收四维 Tensor，结果可按需 `squeeze(2)`。`prepare()` 要求当前 NPU device 与输入 Tensor 的 device 一致，否则明确报错。计划绑定 prepare 时的设备、当前流配额；改变设备、流或配额后重新 prepare。

Invocation 绑定本次输入、输出、workspace 和 program；每次 `call()` 会覆盖并返回同一个输出 Tensor，要保留历史结果需自行 `clone()`。可以更新原输入的内容；shape、stride 或执行配置变化时重新 prepare。验证脚本另行展示兼容容量下的底层 program 跨 shape 复用。

公开调用形态：

```python
import torch
import torch_npu
import dx

# dy=[N,Co,Do,Ho,Wo]，weight=[Co,Ci,Kd,Kh,Kw]；先设定输入所在设备。
torch.npu.set_device(dy.device)
call = dx.prepare(dy, weight, input_shape, stride=(1, 2, 2))
try:
    result = call()   # 异步启动；call 持有输出和 program
    # 使用 result，或先 synchronize 再读取数值/复用外部资源。
finally:
    call.close()     # 等待完成后释放 program
```

`dx.prepare_transpose(x,weight,stride=...,output_padding=...)` 推导输出形状后直接调用 `dx.prepare`，没有另一个 Transpose kernel；不支持 bias/pad/groups 参数时不要默默忽略它们。

## 如何把样例用作性能实验

默认 tile 刻意让 M>16、Ci>16、K 尾块和跨 N 块出现，便于抓到布局错误，**不代表推荐的性能最优块**。`--auto-tile` 用资源过滤后的候选排序，输出依据为每波补齐 MAC、再比较 GM 字节和 MMAD 条数；它只是可复算的起点，不是校准过的周期模型。

补齐 MAC 除以3510 B16单核的 `16×16×16=4096 MAC/cycle`，得到不含指令头开销的计算项估计；原理见 [3510 Cube 建模](../../../../../../../../ascendc-performance-best-practices/references/matmul/arch35/performance_modeling.md)。计划还输出每波 GM 字节和 MMAD 条数估计。dW 按完整带、短尾带分别统计 K 对齐；dX 的 chunk 与深度边界按上界估计。这些值用于候选解释，不把 MAC、字节和指令数直接相加冒充真实周期；MTE/Fixpipe、同步和重叠还需实测校准。

```bash
# 同一用例改变一个主要配置，比较计划和设备耗时，不比较 Python 墙钟。
python verify.py --op dw --dtype fp16 --tile 32 16 32 --blocks 32 --profile --output /tmp/dw-ci16
python verify.py --op dw --dtype fp16 --tile 32 32 32 --blocks 32 --profile --output /tmp/dw-ci32
python verify.py --op dx --dtype fp16 --auto-tile --blocks 32 --profile --output /tmp/dx-tiling
```

`--profile` 先预热，再记录5次设备 kernel 时间，并保存原始 trace；它只测样例，不自动生成 ASC 加速比。普通 dW/dX 每次调用一个 kernel，切 K 为计算与合并两个 kernel；脚本据设备 `Task Type` 提取事件并校验条数，保存各阶段耗时、设备任务合计和首尾跨度。跨度可能包含 Host 发射间隙，不能与任务耗时合计混称。

编译时默认导出生成 ASC 到输出目录的 `dump/asc/.../kernel.asc`。核对以下动态路径，不把全文件静态指令条数直接当作每核执行次数：

| 源码位置 | 预期核对的指令/行为 | 应观察的性能权衡 |
|---|---|---|
| dW `load_plane` | 每行带 x 一次 DN2NZ、dY 一次 ND2NZ；完整通道时没有 C1/H 拷贝循环 | 合并通道减少发射，但增大 N 也可能压短 BK、减少输出任务 |
| dW `step` 循环 | 下一行带发射在当前行带计算之前，L1 槽位于两个半区 | RingBuffer 两槽只是前提，还需查看真实依赖是否允许重叠 |
| dW 输出 | 每个 co0/ci0 组合一个批矩阵 Fixpipe，无逐 tap/逐 co 标量循环 | `sM0=BM*16` 与 N1 偏移一致；BM=16 的拼写不能照搬到 BM>16 |
| dX `load_dy` | H 作为 DN2NZ 批轴，strideH/W 由目的 stride 表达 | 只在前放大时清理内部孔洞；外部 padding 不等于内部清零 |
| dX `load_weights` | 每个 kd 一次父布局打包，在一个任务的多个 M 块间复用 | 扩大 chunk 减少权重重载，同时减少独立任务 |
| dX `load_b0` | 每 K 分形的转置 Load2D，逆序 tap 起点正确 | 此循环对应参考变换粒度，不能与多余 GM 行循环混为一谈 |
| 两例 MMAD→输出 | K 子块不触发 GM 写回；完整归约后一次输出阶段 | BK 是单次 MMAD 的 K，不是 L0C 连续累加段长 |

dW 每个输出只写一次，因而本例无需原子累加和输出清零。若扩展成长 K 分段或跨核合并，不能直接添加 `atomic="add"`：须同时恢复初始化、所有权及对应同步协议，见 [初始化](../initialization.md)。dX 直接 Fixpipe 转成 B16，不另起 cast kernel。其单份驻留 weight 位于 L1 前半区：A1 的 ping/pong 分开只保证两代 fmap 的地址隔离，不代表 weight 读取与下一轮搬入也没有 bank 竞争；优化时仍须检查实际流水。

## 数值验证

dW 测试覆盖2/2/1行带的短尾、Kh halo和通道尾组合。宽图用例的输入为 `(1,64,1,32,1026)`，输出通道为128，卷积核为1×1×3，用于检查行带循环和多 Ci 处理。

验证输入先量化为实际 B16，再用 CPU float64 计算 Golden；不能拿原始 FP32 输入与量化后结果比较，再把差异归为 Load3D 错误。写回前预填 NaN 检查覆盖，并要求两次运行逐位一致。脚本里的容差是该教学测试的 smoke 标准，不替代完整 ops-test-kit 准入。

同一 dtype/容量下，验证脚本让前一个 program 处理下一组不同 shape；Transpose 的 stride 与 output_padding 也通过动态参数消费。`results.json` 中的 `reused_program` 记录实际是否进行了这一项验证。输出目录含设备 trace、生成 ASC 和结果，可据此区分编译成功、精度通过和性能结论。

## 设备验证记录

2026-11-24，在950、CANN 9.3.0、上述 DSL 接口环境上：18组主覆盖分别在32/36核上限通过，包含FP16/BF16、独立 Golden、重复运行和同 program 换 shape；其中 dX 的120任务用例实际分别启动32/36个核。较小任务按实际任务数启动，未把空闲配额当成满核利用。另在32核上限补测了 dW 的1×1×1、Ci/Co=17、FP16/BF16，均通过。

以下为样例自己的 tile 对照，单位 μs；各取5次设备任务的中位数，不含 Host，也不是 ASC 加速比：

| FP16 场景 | 固定布局验证配置 | 自动候选 | 结果分析 |
|---|---|---|---|
| dW：x=(1,64,1,32,1026)，Co=128，1×1×3 | (M,Ci,K)=(32,32,32)，8任务，69.87 | (16,16,128)，32任务，22.38 | 同时改善核利用和 MMAD 发射次数；没有跨核切 K、原子争用或归约 workspace |
| dX：dY=(5,48,3,6,9)，Ci=47，2×3×3 | (M,N,K)=(32,32,64)，chunk=1，17.83 | (32,16,128)，chunk=3，18.73 | 计算项估计降低但总耗时未降，当前模型尚不足以给出最优配置 |

表中数据来自独立采集，复现实验使用 `--case dw_fp16_2` 或 `--case dx_fp16_1`，分别搭配对应 `--tile` 和 `--auto-tile`。比较配置时应在一致环境下重新采集。自动候选根据几何与资源预算选择，不包含这些 shape 的专用分支。


## 跨核切 K 示例

[dw_splitk.py](dw_splitk.py) 是独立的确定性 split-K 扩展示例，**不是当前 generic `MN_StreamK` 无 workspace 模板的逐项复现**。它复用 `dw.py` 的 Load3D、预取和批量写回，仅扩大任务空间和输出第一轴；读取时先看 `prepare()`，再看共享 `kernel()`，最后看 `merge()`。

- 将归约轴按完整行带划分。总带数 `B=N*Do*ceil(Ho/rows)`；第 s 片处理 `[s*B//S, (s+1)*B//S)`，包括各深度面自己的短尾带。分片首个 MMAD 初始化 L0C，片内所有 BK/行带连续累加。
- 每个 `(输出块, split)` 独占一份 FP32 部分和。工作区逻辑为 `[S, Co, Ci, Kd, Kh, Kw]`，首两轴合并存放；计算阶段仍批量 NZ2DN 写入最终权重顺序。没有多个核竞争同一地址，也不需要生产路径预清零。
- 后续纯 AIV kernel 对每个连续256元素输出段，用一次二维 DMA 将 S 份部分和搬到 UB，再按编号 `0→S-1` 固定顺序相加，最后一次 DMA 写最终输出。纯 AIV 的 block 数直接代表向量核数，独立查询 AIV 配额，不能沿用混合 kernel 的核对数量再除2。两 kernel 在同一 stream 顺序启动，**不需要全核 barrier，也不需要自写跨核事件**。工作区和输出由同一个 Invocation 持有。
- VF 在外部准备循环次数，循环从0开始、每次+1；四个寄存器分别累加连续256个 FP32 的四段，读取地址显式为 `(s+1)*256 + {0,64,128,192}`，没有 VF 内条件分支或逐元素循环。本机反汇编确认归约生成 `VLOOPV2_V310`，循环体为4条 VLD 与4条 VADD，地址使用 A0 推进；不能仅凭 DSL 中写了 for 就认定已有硬件循环。固定 split/tiling 下要求逐位一致，不要求不同 split 数结果逐位相同。

默认 tiler 联合搜索合法基本块和 split 数，按实际 round-robin 分核累加各分片的补齐 MAC、MMAD 条数，并考虑部分和流量。**每片至少4096个有效 K、相对模型选中的不切 K 候选至少减少25%的计算工作量**是保守的启发式，不是硬件性能公式，也不保证选到最快方案；`--splits` 用于对照验证。小 K 可以保留空核，S=1 时不分配 workspace、不启动 merge。例如本机 tails 用例有18个输出块：32核上限选择3片形成54任务，36核上限选择2片形成36任务；split 数由当前核配额与工作量推导，没有按 shape 写死。当前每片有效 K 保守限制为65535，最多128片、工作区最多64 MiB；超出范围明确拒绝。长 K 可以通过多片支持，但仍须按正式精度规程验收，不能将“确定性”理解为“自动满足精度”。

与原 dW 一样，本例只覆盖 B16、NCDHW、groups=1、valid stride1/dilation1，以及能装入 L1 的完整宽度行带。未加入原子分段、宽度切分或新的卷积算法。合并使用普通 FP32 定序加法；输出仍为 FP32，若调用者要求 B16，应将最终转换计入完整设备路径。

```bash
# 自动选择：含小 K 不切、通道/行带尾块、少输出块和总 K>65535。
python verify_splitk.py --blocks 32 --profile --compare --output /tmp/dw-splitk-32
python verify_splitk.py --blocks 36 --output /tmp/dw-splitk-36
# 固定切5片，检查不等长分片、跨 batch/depth 和短尾带。
python verify_splitk.py --case tails --splits 5 --tile 32 32 32 --output /tmp/dw-splitk-tail5
# 同一小 K 测不切/强制切，观察合并固定开销。
python verify_splitk.py --case small --splits 1 --profile --output /tmp/dw-splitk-small1
python verify_splitk.py --case small --splits 2 --profile --output /tmp/dw-splitk-small2
```

[verify_splitk.py](verify_splitk.py) 从实际 B16 输入生成 CPU double Golden，检查 NaN 预填工作区的有效位置全部写入、重复执行逐位一致、更换输入不会读到旧 partial，并在模板容量一致时复用旧 program 处理新 shape。`--compare` 对总 K≤65535 的切 K 用例额外测模型选中的不切 K 候选；比值为同一样例的不切 K 设备耗时除以切 K 两阶段设备耗时之和。默认容差沿用教学 smoke 测试。

### 切 K 设备计时

2026-11-24，950，32 AIC 上限；下表取5次设备任务耗时的中位数，不含 Python/Host。两阶段合计按每次调用先求和再取中位数，因此可能与分别取中位数的阶段值之和略有不同。

| 场景（参数见 verify_splitk.py） | dtype | 自动 split | 不切 K，μs | 切 K 任务合计，μs | 任务合计加速比 | 切 K 首尾跨度，μs |
|---|---|---:|---:|---:|---:|---:|
| tails：通道尾、短尾行带 | FP16 | 3 | 26.08 | 21.31 | 1.22× | 23.18 |
| tails | BF16 | 3 | 26.13 | 21.15 | 1.24× | 21.27 |
| wide：只有1个输出块，K=32768 | FP16 | 8 | 21.63 | 5.72 | 3.78× | 23.91 |
| wide | BF16 | 8 | 21.65 | 5.76 | 3.76× | 18.59 |
| long：K=81920，超过单 owner 样例范围 | FP16 / BF16 | 20 | 不支持 | 6.45 / 6.49 | — | 30.81 / 18.38 |

这里 **3.78× 是不切 K 与切 K 两阶段设备任务合计的比值**。两 kernel 之间可能有明显的 Host 发射空隙，表中 wide/FP16 的首尾跨度并未优于不切 K；评估端到端延迟时应另测设备图回放或实际调用环境。小 K=30 默认选择 S=1；强制切2片的配对测试中，FP16 从1.74 μs 变为2.63 μs，BF16 从1.75 μs 变为2.72 μs，说明增加并行任务的收益可能被合并的固定开销抵消。

切 K 验证记录包含28次通过的样例校验：32/36核上限各8组自动选择、两档各2组强制5分片、2组强制2片小K、2组强制1片，以及4组原dW样例回归。覆盖FP16/BF16、FP32输出、工作区完整写入、重复运行、变更输入、同program跨shape复用；适用范围与smoke精度口径见上文。

重复运行检查比较输出的原始字节，可区分正零和负零。dW、dX、Transpose 和5分片 dW 的6组 FP16/BF16 代表用例已通过字节级检查；上述计时记录采集时使用的是逐元素相等检查。
