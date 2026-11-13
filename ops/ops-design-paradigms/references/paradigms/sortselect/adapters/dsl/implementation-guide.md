# SortSelect DSL 实现指南

本页维护所有 DSL 路线共用的布局、输出适配和执行合同；每条算法的切分、Buffer 和 Kernel 细节在[模板导航](index.md#按模板读取)下独立维护。Markdown 中给出完整计算步骤与参数意义，具体 API/编译绑定再用 [Sort example](example/sort/README.md) 核对。公开 Python 入口分配输出和 workspace，`@host` 分派，`@kernel` 排序；独立交付时所需模板与公共原语随算子组织，不在运行期依赖 example。

## 算法模板与支持范围

| 算法 | DSL 模板文件与表达 | 实现约束 |
|---|---|---|
| small-axis insertion | [插入 Kernel](templates/insertion/kernel-template.md)：连续路径 DMA 加载 → 每线程一行稳定插入 → 连续合作写回；低层跨步路径协作加载 | 原值/保序键/原轴索引三份 UB，批次上限与线程数配套 |
| radix one-core | [单核 Kernel](templates/radix-one-core/kernel-template.md)：32/64 位键/索引 LSD 后回读 GM；1/2 字节局部排列后恢复 | [字节 permutation](local-radix.md)由 SIMD 组合，未调用原生高阶 Sort |
| two-stage rank-inverse | [两阶段 Kernel](templates/two-stage/kernel-template.md#2-逆映射恢复)：批次 LSD → inverse → 行内秩 | 按原输入恢复 value/index，无第二次排序；门槛与[两阶段 Tiling](templates/two-stage/tiling.md)配套 |
| two-stage stable row grouping | [同一模板的行号分组分支](templates/two-stage/kernel-template.md#3-稳定行号分组)：批次 LSD → 行号键 → 稳定分组 → 恢复 | 第二次始终升序；键的表示范围与载体 S 分开 |
| radix more-core | [多核 Kernel](templates/radix-more-core/kernel-template.md)：Histogram → 局部排列/lookback/Scatter → 末轮恢复 | 单次 launch 的全核阶段屏障，按活跃行容量复用 GM scratch |
| MergeSort（内部 Sort32/MrgSort 原语） | 尚未适配 | 无归并模板；指定归并路线时报告缺口，不静默替换算法 |

Host 选路顺序、完整 dtype 档位、恢复分支和批次均衡集中见 [选路与容量](tiling.md)。算法的位宽/容量硬约束与本例实现策略分开使用。

## 模板组织与读取

通用算法见 [compute-flow](../../generic/compute-flow.md)；本层按四类算法分别维护 `templates/<算法>/tiling.md`、`dag-buffers.md`、`kernel-template.md`。先按文档计算批次、块长、索引与资源，再在实现中核对绑定。`entry.py` 对应公开接口与 Tiling，`sort_host.py` 对应 Host 分派，`sort_common.py`/`radix.py` 对应[保序键与局部 Radix](local-radix.md)。文件和函数定位见[实现导航](../../source-map.md#cannbotdsl)。

32/64 位单核 Radix 与 two-stage 复用 `sort_local_keys`；1/2 字节单核 `NarrowRadixOneCore` 直接调用一/两次 `radix_permutation` 并组合排列，不分配该 helper 的键/索引双缓冲。two-stage 的逆映射与稳定行号分组在同一模板中编译期选型；多核 Radix 在同一文件中组织批首清零、Histogram，以及融合局部排列/lookback 的 Scatter；没有独立 Prefix 调用。

## example 的参数与执行机制对应

下表对应公开 Sort 配置 L=N。width/groups 的含义由路线决定；groups 不是统一的核分组参数。生成工程可调整命名和文件组织，处理域、数据通道与生产消费关系沿模板实例化。

| 路由 | width / groups | 核 c 的工作 | 当前原值和排序通道 |
|---|---|---|---|
| 1 插入 | Ncap / B | batch=c+k×blocks | 连续路径 DMA 原值、每线程生成键/索引并排序，再连续 SIMT 写回；跨步低层保留协作路径 |
| 2 单核 Radix | align256(N) / 1 | row=c+k×blocks | D=1：SIMT digit→一次排列→整数反变换；D=2：原始 UB 值→两次排列组合；D=4/8：连续 32 位 DMA+SIMD / 其余 SIMT 构键→逐字节键/索引双缓冲→SIMT 按最终索引回读 GM |
| 4/5 两阶段 | align256(BN) / B | batch=c+k×blocks | 连续 32 位 DMA+SIMD / 其余 SIMT 构键→共用局部 SIMD LSD；4 以 inverse 计秩，5 以行号键做单字节 LSD；两者都用 SIMT 回读原输入并恢复输出 |
| 3 多核 Radix | T / P_act，0 表示全 M 行直调 | 每批 task=c+k×blocks，task=row×K+tile | 清零全部 digit 状态并汇合；每轮 Histogram 后汇合，局部排列/lookback/Scatter 融合；非末轮与批次复用前汇合 |

1/4/5 的 blocks=min(C,ceil(M/B))，2 为 min(C,M)，3 为 min(C,MK)，每批仅处理有效行。公开入口一律传 inner=1；低层 strided 调用按各模板的原布局地址公式处理。公开输出索引为 int64 原轴坐标，不能加入行基址。

这四类模板均没有调用原生高阶 Sort。AscendC 的原生 Sort、Queue/DMA 原值侧或 Gather/Cast 不能仅凭算法名直接映射成本例机制；具体原语、缓冲和同步分别沿上述模板核对。

## 路线命名与非末轴布局

所有输出配置共用四类排序主体，不提供独立 Sort32 或按长轴/短轴划分的非末轴模板。布局按 [统一判定](../../generic/shape-preprocessing.md#物理转置的判定与验收) 处理：连续尾轴直接使用，等价布局必须 reshape，仅 N>1 且 I>1 时通过 [SortTranspose](example/sort/sort_transpose.py) 将输入 `[O,N,I]` 物化为 `[O,I,N]`，主体按 I=1 处理。输出 `[O,I,L]` 仅 L>1 且 I>1 时物理转回 `[O,L,I]`，否则 reshape，再按接口处理 keepdim。索引值仍为原轴坐标。

Sort example 的 L=N，`PreparedSort.needs_transpose` 控制输入及两份输出；生成其他输出区间时分别判断 N 与 L，不能把 example 的同一布尔标志直接用于全部输出。设计、编码与验收核对实际布局转置次数，尾轴和等价布局必须为零，不能只检查结果精度。

转置使用独立 DSL kernel，优先用 `mem_copy(..., copy_options(engine="nddma"))` 将 GM 的 permute 视图物化到 UB，再 DMA 写回。`permute` 本身只改视图，默认 `mem_copy` 不自动选择 NDDMA；不是把一次逻辑 view 当成已完成转置。输出短行完整落在 tile 内时用紧凑 UB 布局和连续写回，避免固定 padding 造成大量短 burst。64 位以两个 uint32 字一起移动；小容量 64 位短行保留 1024 线程 SIMT。具体形状门槛、64 位绑定和 Native ABI 见 [转置映射](toolkit-transpose-port.md)。Host 仍核对 `max(T_tr×(T_tr+1),128×65)×8≤U`；容量不足时报错。该策略是本例的实测选型，不是任意形状的最优保证。

转置数量与统一布局合同保持一致。启动在同一流依次提交，输入转置区保留到最后一次原值读取，输出转置区保留到各自转回完成。`run()` 重用缓冲时重新转入本次输入；性能包含全部布局搬运。容量按 [统一预算](tiling.md#容量预算) 计算。

## 统一输出适配

[公共输出映射](../../generic/output-mapping.md) 将 Sort、Argsort、TopK、KthValue 转为方向、`first_rank`、`output_length` 和输出字段。以下是生成实现的代码片段，不是 example 已有编译签名或 DSL 内置 API。排序模板保留完整 N 项，在最终恢复/写回中应用区间。

以当前 UB 批次已有的最终原轴索引 `final_ids` 为例，一次 VF 处理全部有效行的输出。`row_start` 为当前张量视图中的批次首行，`valid_rows` 为本批有效行数，`index_pitch` 为 UB 索引行距；x、values、indices 均为该视图规范化后的扁平连续张量。Host 保证 `output_length>0`、区间落在完整 N 项内、`index_pitch≥N`，且 `valid_rows×output_length` 落在 int32 域：

```python
with cb.vf(mode="simt", thread=256):
    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
    for p in range(tid, cb.dtypes.int32(valid_rows * output_length), 256):
        b = cb.dtypes.int64(p // cb.dtypes.int32(output_length))
        t = cb.dtypes.int64(p % cb.dtypes.int32(output_length))
        axis = cb.dtypes.int64(final_ids[b * index_pitch + first_rank + t])
        row = row_start + b
        dst = row * output_length + t
        # 按接口保留所需字段；values 按索引读取可保留原始位模式。
        values[dst] = x[row * length + axis]
        indices[dst] = axis
```

L=1 时，上述线程在行间分工；B=1 也使用同一表达。对原本合批的路线，不在外层添加 row 循环逐行执行这段 VF，也不为迁就单行写回而把 Host 批次改为 1；必要分块及例外依据见 [通用批处理约束](../../generic/output-mapping.md#保持批处理粒度)。保留原模板的 VF/UB 消费者完成依赖，按批次或容量块复用缓冲，不改成逐行等待。

该片段适用于已有最终 UB 索引的路线。插入可直接写出对应原值；two-stage 先恢复行归属和行内 rank，逆映射建表与必要比较仍覆盖完整 N，不能将其循环上限换成 L。已有恢复与 GM 写回融合的模板保留融合，在确定行内 rank 后应用区间，不为套用该片段额外落盘完整结果。32/64 位单核 Radix 按轮次奇偶选择最终索引 buffer；1/2 字节使用局部排列，最终原位置分别为 A[p] 或 A[B[p]]；归并尚未适配。多核最后 Scatter 使用相同映射，其他轮不裁剪：

```python
if final_pass:
    rank = destination - local_row * length
    if first_rank <= rank and rank < first_rank + output_length:
        dst = global_row * output_length + rank - first_rank
        values[dst] = raw
        indices[dst] = cb.dtypes.int64(source_axis)
else:
    output_values[destination] = raw
    output_indices[destination] = source_axis
```

## 当前活跃批次的工作区复用

当前 example 在单次 launch 内遍历活跃行批次，scratch/bins 按 P_act 分配。Host 以目标工作集和核间任务均衡选择容量，具体公式和软件目标见 [Tiling](tiling.md#多核活跃行容量)。groups=0 表示全部 M 行，正 groups 表示容量；32 位中间区仍只由 N<2^30 决定。

workspace 按批内行号 r 寻址，输入/输出按 `row_start+r` 寻址；每批完成全部 digit。最后 Scatter 仍读取 hist/states，因此下一批复用前全部核必须到达屏障。Histogram、Scatter 只遍历本批有效行，尾批不读取前批遗留状态。不能只缩小分配或拆成多次 Host launch。字段及输出区间不改变完整 N 域和原有依赖，输出/布局缓冲另计。

1/2 字节单核以局部排列表示稳定 LSD 的原位置映射，逻辑依据在 generic，物理缓冲和 DMA/SIMT 细节在单核模板。两字节原值保存在 UB，最终不回读 GM；一字节整数仅在可逆键条件下反变换恢复。

## 数据表示

全部 UB 公式、对齐和位宽上限见 [容量预算](tiling.md#容量预算)，行号分组与恢复分支的约束见同页。

四类模板和转置均用显式 Layout 分配 UB：`cb.Buffer(cb.MemLoc.UB, dtype, cb.make_layout((elements,)))`。构造参数顺序为内存位置、dtype、Layout；shape 元组须先经 `make_layout`，不能作为 dtype 参数或直接替代 Layout。Tensor 的逻辑长度、实际 stride 与存储 dtype 一并核对，分配表达变化不改变原有容量公式和同步。

键保留原 dtype 的排序语义，LSD 只处理有效字节宽度。浮点正负零等键、NaN 同键；32/64 位单核和 two-stage 按索引回读原输入，多核逐轮搬移原值并直接输出，16 位单核从原始 UB 值恢复，均保留 payload；全局地址使用 int64，前缀按单行 N 选择 int32/int64。本例的 uint64 字节提取使用 32 位分字实现，SIMT 路径见 `radix_digit`，多核连续 SIMD 路径见 `load_digit_word`。

VF 内无符号键参与比较时，零、无穷位模式和符号位常量均显式转换为对应无符号 dtype。64 位键的高字用无符号除法提取，不依赖有符号除法的历史舍入行为；具体表达见 [保序键与有效字节](local-radix.md#1-保序键和有效字节) 和 [多字节 LSD](local-radix.md#3-多字节-lsd)。

## 多核 Radix 的局部排序与散射

统计阶段只用 SIMD histogram 计算频数与局部桶起点，并将桶起点原子累加为全局桶起点。Scatter 执行局部排列，排列仅在 UB 存活。lookback 命中前驱完整前缀时直接使用，未发布时累加已完成统计的 hist，不轮询前驱核。

全部连续 dtype 以 DMA 搬到 UB，SIMD 抽取 digit；64 位使用字节视图。Scatter 从 raw_tile 和 index_tile 收集原值与索引，低层 strided 使用 SIMT L2 读取。浮点只归一比较键，原值逐轮搬移并在末轮直接输出。当前两种计数宽度都使用 SIMT 原子累加，不宣称已提供 AscendC 的 DMA 原子或原生 Sort 高阶 API。具体执行和数据生存期见 [多核 Kernel](templates/radix-more-core/kernel-template.md)。

全局地址和最终索引为 int64，中间 ids/hist/states/UB scatter bases 按单行 N 统一选择 int32/int64。states 为各 digit 独立分配 K 份前缀和一份全局桶起点；0 为未发布，负编码 `−inclusive−1` 为已发布。六个工作区的容量见 [Buffer](templates/radix-more-core/dag-buffers.md)。

## 同步与编译

SIMD 压缩写入后 finalize，再以 VST/VLD 屏障保护后续索引读取；SIMT 逆映射建表与行内读取之间保留全块 barrier。多核每批先清零全部 digit 的 states 并全核汇合；每轮 Histogram 后汇合，非末轮 Scatter 后汇合，批间复用前另行汇合。所有启动核参加，包括无有效任务的核；block 数不超过当前流有效 AIV 核数。GM 元数据和散射经 L2 load/store 及生产 VF 的 threadfence 发布，DMA/UB 的依赖同时由编译流水管理，不能仅凭 destination 唯一删除交接。`PreparedSort` 重复运行会重新初始化状态并覆盖输出。无等待回退读的是统计屏障前的完整 hist，不等待尚未执行的 tile。

two-stage 的连续 32 位构键使用 DMA+SIMD，其余构键及恢复使用 1024 线程；共用局部 Radix 使用 SIMD。逆映射行内计秩使用 8 个独立 int32 累加器，每次处理 8 项，再处理不足 8 项的尾部，比较范围仍为完整 N。two-stage 的两种恢复由 Host 选择 constexpr `rank_inverse` 变体；该组织避免在 Kernel 中引入恢复路线的运行时分支。四个 2-bit pass 在同一个 SIMD VF 内执行，每轮使用独立的连续 cursor，末尾统一 finalize 并建立 VST/VLD 依赖。局部批次复用仍需保留现有 VF 依赖、SIMD finalize/VST-VLD 屏障和 SIMT 建表后的全块 barrier；不能因输出区间较短而删掉中间同步。读取结果或关闭程序前按运行时协议等待设备完成。
