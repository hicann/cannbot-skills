# SortSelect DSL 实现指南

本页维护所有 DSL 路线共用的布局、输出适配和执行合同；每条算法的切分、Buffer 和 Kernel 细节在[模板导航](index.md#按模板读取)下独立维护。Markdown 中给出完整计算步骤与参数意义，具体 API/编译绑定再用 [Sort example](example/sort/README.md) 核对。公开 Python 入口分配输出和 workspace，`@host` 分派，`@kernel` 排序；独立交付时所需模板与公共原语随算子组织，不在运行期依赖 example。

## 算法模板与支持范围

| 算法 | DSL 模板文件与表达 | 实现约束 |
|---|---|---|
| small-axis insertion | [插入 Kernel](templates/insertion/kernel-template.md)：合作加载 → 每线程一行稳定插入 → 合作写回 | 原值/保序键/原轴索引三份 UB，批次上限与线程数配套 |
| radix one-core | [单核 Kernel](templates/radix-one-core/kernel-template.md)：构键 → 局部 LSD → 按索引回读原值 | [字节 permutation](local-radix.md)由 SIMD 组合，未调用原生高阶 Sort |
| two-stage rank-inverse | [两阶段 Kernel](templates/two-stage/kernel-template.md#2-逆映射恢复)：批次 LSD → inverse → 行内秩 | 按原输入恢复 value/index，无第二次排序；门槛与[两阶段 Tiling](templates/two-stage/tiling.md)配套 |
| two-stage packed key | [同一模板的紧凑键分支](templates/two-stage/kernel-template.md#3-紧凑键恢复)：批次 LSD → 紧凑键 → 两字节 LSD → 恢复 | 第二次始终升序；键的表示范围与载体 S 分开 |
| radix more-core | [多核 Kernel](templates/radix-more-core/kernel-template.md)：Histogram → Prefix → Scatter → 末轮恢复 | 单次 launch 的全核阶段屏障，当前按全 M 行保留 GM scratch |
| MergeSort（内部 Sort32/MrgSort 原语） | 尚未适配 | 无归并模板；指定归并路线时报告缺口，不静默替换算法 |

Host 选路顺序、完整 dtype 档位、恢复分支和批次均衡集中见 [选路与容量](tiling.md)。算法的位宽/容量硬约束与本例实现策略分开使用。

## 模板组织与读取

通用算法见 [compute-flow](../../generic/compute-flow.md)；本层按四类算法分别维护 `templates/<算法>/tiling.md`、`dag-buffers.md`、`kernel-template.md`。先按文档计算批次、块长、索引与资源，再在实现中核对绑定。`entry.py` 对应公开接口与 Tiling，`sort_host.py` 对应 Host 分派，`sort_common.py`/`radix.py` 对应[保序键与局部 Radix](local-radix.md)。文件和函数定位见[实现导航](../../source-map.md#cannbotdsl)。

单核 Radix 和 two-stage 分别维护自己的 Buffer 与调用流程，复用同一个 `sort_local_keys`。two-stage 的逆映射与紧凑键恢复在同一模板中编译期选型；多核 Radix 的三个依赖阶段放在同一文件，便于一起检查桶计数、前缀与散射契约。

## 路线命名与非末轴布局

所有输出配置共用四类排序主体，不提供独立 Sort32 或按长轴/短轴划分的非末轴模板。布局按 [统一判定](../../generic/shape-preprocessing.md#物理转置的判定与验收) 处理：连续尾轴直接使用，等价布局必须 reshape，仅 N>1 且 I>1 时通过 [SortTranspose](example/sort/sort_transpose.py) 将输入 `[O,N,I]` 物化为 `[O,I,N]`，主体按 I=1 处理。输出 `[O,I,L]` 仅 L>1 且 I>1 时物理转回 `[O,L,I]`，否则 reshape，再按接口处理 keepdim。索引值仍为原轴坐标。

Sort example 的 L=N，`PreparedSort.needs_transpose` 控制输入及两份输出；生成其他输出区间时分别判断 N 与 L，不能把 example 的同一布尔标志直接用于全部输出。设计、编码与验收核对实际布局转置次数，尾轴和等价布局必须为零，不能只检查结果精度。

转置使用独立 DSL kernel；启动在同一流依次提交，输入转置区保留到最后一次原值读取，输出转置区保留到各自转回完成。`run()` 重用缓冲时重新转入本次输入；性能包含全部布局搬运。容量按 [统一预算](tiling.md#容量预算) 计算。

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

该片段适用于已有最终 UB 索引的路线。插入可直接写出对应原值；two-stage 先恢复行归属和行内 rank，逆映射建表与必要比较仍覆盖完整 N，不能将其循环上限换成 L。已有恢复与 GM 写回融合的模板保留融合，在确定行内 rank 后应用区间，不为套用该片段额外落盘完整结果。单核 Radix 按轮次奇偶选择最终索引 buffer；归并尚未适配。多核最后 Scatter 使用相同映射，其他轮不裁剪：

```python
if self.final_pass:
    rank = destination - local_row * length
    if first_rank <= rank and rank < first_rank + output_length:
        dst = global_row * output_length + rank - first_rank
        values[dst] = x[global_row * length + cb.dtypes.int64(source_axis)]
        indices[dst] = cb.dtypes.int64(source_axis)
else:
    output_keys[destination] = key
    output_indices[destination] = source_axis
```

## 当前活跃批次的工作区复用

当前 example 的多核路径在一次 launch 中处理全部 M 行，`scratch` 和 `bins` 按全 M 行分配；它尚未实现有界 P_act 复用。全 launch 屏障要求该 digit 的全部 Histogram 完成后才能进入 Prefix/Scatter；不能只缩小分配而保留原寻址；拆成多次 Host 调用也会改变现有提交粒度和同步组织，需要另行验证。

生成有界批次实现时，键、索引和桶状态按批次内行号寻址，输入/输出按全局行号寻址；输入 M×N 与 scratch P_act×N 使用独立 TensorSpec 维度。跨批次先完成最后读写及全部参与者同步，再复用 scratch。全局计数/前缀与中间原轴索引沿用 int64，局部 uint16/uint32 排列和计数按原模板检查。

应优先在单次 launch 内设计行/核组和批次复用，并与原实现比较实际成本；简单增加批次循环仍可能增加屏障和并行度损耗，精度或内存通过不能代替性能验证。单次排序 launch 的默认路径在替代实现通过前保留；当前能力状态与实际预算见 [统一工作区](tiling.md#统一批次工作区)。不在不同流并发复用同一 scratch；公共内存原则见 [统一批次内存](../../generic/liveness-fusion.md#统一批次内存)。

## 数据表示

全部 UB 公式、对齐和位宽上限见 [容量预算](tiling.md#容量预算)，紧凑键与恢复分支的约束见同页。

键保留原 dtype 的排序语义，LSD 只处理有效字节宽度。浮点正负零等键、NaN 同键，原值由原索引恢复，保留 payload；全局地址和前缀使用 int64。本例的 uint64 字节提取使用 32 位分字实现，见 `radix_digit`。

## 多核 Radix 的局部排序与散射

Histogram 阶段复用 `radix_permutation`，将块内原位置和桶内秩编码为 `原位置×T+桶内秩`，按已排序位置存入 uint32 workspace；Scatter 按此顺序读取，恢复原位置并计算全局目标。T≤8192 时编码最大为 T²−1，落在 uint32 范围内。不能仅保留“计数、前缀、散射”的名称，就用逐组 ballot 循环替代局部排序而假定成本不变。

T 的候选范围、容量与并行度条件见 [批次与块长选择](tiling.md#批次与块长选择)。Prefix 使用 warp shuffle 扫描与跨 warp 汇总。首轮 Histogram 同时构造键和原索引，末轮 Scatter 直接恢复原值与输出索引。Histogram、Prefix、Scatter 均为 `@cb.jit` 阶段，全部行的全部 digit 只启动一个 RadixMoreCore kernel，不增加 Encode/Recover kernel。当前使用全 launch 屏障而非 lookback 协议，需计入 GM workspace 往返和屏障成本。

局部容量允许的循环计数和除法使用 int32，行基址提前计算；全局地址、前缀和输出索引保持 int64。

## 同步与编译

SIMD 压缩写入后 finalize，再以 VST/VLD 屏障保护后续索引读取；频数、累计直方图和 cursor 语义见 [能力映射](capability-mapping.md)。SIMT 逆映射建表与行内读取之间使用全块 barrier。多核 Radix 在单 kernel 内使用 `cb.global_sync_all()` 建立 Histogram→Prefix、Prefix→Scatter 及 Scatter→下一 digit 的依赖。所有核都必须到达，包括没有 tile 或 Prefix 行的核；启动 block 数不得超过当前流有效 AIV 核数。跨阶段 GM load/store 显式使用 `cb.simt.l2cache_load/store`，各生产 VF 在结束前执行 `cb.simt.threadfence()`；只加屏障而沿用普通 GM 访问不能保证多轮重用数据的可见性。`PreparedSort` 复用 workspace，下一次 `run()` 覆盖前次输出。

two-stage 的两种恢复由 Host 选择 constexpr `rank_inverse` 变体；该组织避免在 Kernel 中引入恢复路线的运行时分支。每个 2-bit pass 使用独立 VF 和连续 cursor，末尾统一 finalize。局部批次复用仍需保留现有 VF 依赖、SIMD finalize/VST-VLD 屏障和 SIMT 建表后的全块 barrier；不能因输出区间较短而删掉中间同步。读取结果或关闭程序前按运行时协议等待设备完成。
