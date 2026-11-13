# AscendC 多核 Radix Buffer 与依赖

## 1 核内 UB

T 为 tile 容量，D 为值字节数，S 为 CounterT 字节数，K 为每行 tile 数，P_act 为活跃行数，A 为按 UB block 对齐。

| 物理区 | 逻辑字节数 | 用途 |
|---|---|---|
| 输入值 | TD | 当前 tile 的值 |
| 输入中间索引 | TS，保守分配 | 后续轮读取原轴索引 |
| uint8 排序输入/输出 | T+T | 当前 digit 与排序后的 digit |
| 排序排列 | 4T | uint32 的块内原位置 |
| 全局计数搬入 | 256S | lookback 前驱状态读取 |
| 局部桶计数、桶起点 | `2×256×2` | uint16 的 tile 内统计 |
| 两份计数/状态辅助区 | `2×256S` | 原子累加、状态发布、前缀和散射 |
| tmp | tmpAllocated，见 Tiling §3 | 批首清零源 → uint8 Sort → scatter 偏移 |
| SIMT 保留 | R_simt | 按[运行保留预算](../../output-mapping.md#scratch-与-ub-由-host-规划)核对；参考值 32768 |

合计为 `T(D+S+6)+256(3S+4)+tmpAllocated+R_simt`，每份实际物理分配的对齐另核对。tmpAllocated 包含参考 Host 的剩余 UB 扩容；Sort 查询需求只是其中一部分。S 加宽时要同步改变计数区及索引搬运的实际视图；uint16 局部计数与 uint32 排列保持各自范围。

## 2 六段 GM workspace

清零参数 p0..p5 的定义见 [Tiling](tiling.md#5-清零分工与下发字段)。按参考 Host 顺序分配：

| 段 | 字节数 | 索引坐标 |
|---|---|---|
| exclusiveBins | `A(p1×p4×S)` | slot / digit / bucket，容量含清零余量 |
| globalHist/status | `A(p3×p2×p0×S)` | slot / digit / tile / bucket，容量含清零余量 |
| 中间原轴索引 | `A(P_act×N×S)` | slot / axis |
| tile histogram 与 tile bucket offsets | `2×P_act×K×256×2` | 两份 uint16 数组 |
| 预提取字节 | `A(P_act×K×T)` | slot / tile / local |
| 中间原值 | `A(P_act×N×D)` | slot / axis |

六段求和后再加框架 16 MiB 预留；框架预留不是算法需求或可用 UB。前两段必须按真实清零循环的覆盖长度分配，不能只用逻辑桶数量替代。Host 的累计字节基址与 Kernel 的 uint32 载体视图要逐段对应。

## 3 存活与复用

```text
统计：原值 → digit / tile histogram / tile bucket offsets / 全局桶起点
排列：digit → 稳定 perm
回溯：tile histogram → 状态与完整前缀
散射：perm + 桶起点 + 前缀 + 当前值/原轴索引 → 另一侧值/索引
digit 结束 SyncAll → 下一 digit
批次末 SyncAll → 清零并覆盖下一批 workspace
```

清零区按 digit 保留各轮状态，tile histogram、tile bucket offsets 和预提取字节按轮次复用。原生 Sort 的 tmp 在 Sort 完成后暂存 24 字节 scatter 偏移，scatter 最后读取后才可被下一 tile 的 Sort 覆盖。

## 4 最终输出与受限区间

参考 Sort 使用最终 M×N 输出作为 ping/pong 一侧，另一侧为上述活跃行 scratch。索引临时实际宽度 I=min(S,J)，Host 可以按 S 保守预留；最终输出 J 另算。输出侧的批次基址按 J 推进，中间轮的批内行距按 I 推进；完整字节公式和逐轮视图见 [Kernel §5](kernel-template.md#5-末轮输出与-workspace-基址)。

派生 L<N 时，中间完整 N 项与最终 M×L 输出必须分离。可以按 P_act 为两侧完整排序状态规划独立区域，但要保留原有每批处理和单次 Kernel 的同步协议，并重新计入真实峰值；不能把“应有界复用”直接当作参考包装已支持的接口。

固定 N/C/T/G 后，P_act 受核数限制，算法 scratch 可跨 M 的行批次复用；最终输出和实际转置区仍随 M 增长。用 M>P_act、尾批及同地址更新重跑检查旧计数是否清零、基址是否仍用 slot。

分配依据：[Host 六段预算](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `ComputeRadixSortWorkspace` 与 [Kernel Init](../../example/sort/op_kernel/arch35/sort_radix_more_core.h)。

## 5 贯穿算例的 workspace 账本

沿用 M=37、N=8192、T=4096、int32/int64 的选核结果，进一步给定 a=32、tmpAllocated=4096 作为预算演算输入；这不是 SDK 查询及扩容的实测结果。S=4、Z=1024，清零参数为 p0=p1=37、p2=p4=1024、p3=2、p5=2048。代入前面的六段公式：

| 段 | 字节数 |
|---|---:|
| exclusiveBins | 151552 |
| globalHist/status | 303104 |
| 中间原轴索引 | 1212416 |
| tile hist + tile bucket offsets | 75776 |
| 预提取 digit | 303104 |
| 中间原值 | 1212416 |

算法 workspace 合计 3258368 字节，框架预留、最终输出和布局区另列。段基址按各段长度顺序累计，并分别建立 CounterT、uint16 或字节视图；本例数字只用于复算，不作为通用常数。

## 6 逐 digit 的存活与复用

完整 Sort 逐 digit 的值/索引来源、目的及 IndexT/J 视图切换统一见 [Kernel §5](kernel-template.md#5-末轮输出与-workspace-基址)。workspace 侧不加 batch 偏移；输出侧批次基址按各自最终类型推进。

hist、局部桶起点和 digit 区在当前轮散射结束后复用；globalHist/exclusiveBins 为各 digit 保留独立切片，并在新行批次开始前统一清零。

核内 tmp 的角色依次为“排序临时区 → Scalar 写入 scatter 参数 → scatter 读取”，对应 V_S、S_V 和 scatter 后的 Vector 完成边界；不存在两个角色同时存活而仍安全覆盖的阶段。lookback 的前驱快照在累加 VF 最后读完后释放，避免队列下一次搬入提前覆盖正在累加的数据。
