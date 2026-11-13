# DSL 多核 Radix Tiling

## 1 输入与计数域

小轴及单核局部 Radix 不可行时进入本模板。M 为行数，N 为轴长，C 为当前流有效 AIV 核数，U 为设备 UB 扣除 SIMT 32768 字节后的容量。D 是输入元素字节数。

N<2^30 时中间 ids、hist、states、lookback 与 scatter bases 统一使用 int32，否则使用 int64。该范围依据单行 N，不依据 M×N；地址及最终原轴索引保持 int64。局部计数/桶起点使用 uint16，T≤8192；局部排列使用 uint32。

## 2 块长与核数

F=6+D+H，H 为当前 CounterT/中间索引字节数。静态 Buffer 和固定桶区都计入 [UB 预算](dag-buffers.md#1-每核-ub)。

```text
T=1024
for candidate in [1024,2048,4096,8192]:
  若 candidate×F+4096≤U 且 M×ceil(N/candidate)≥2C:
    T=candidate
若 T×F+4096>U：在编译和 workspace 分配前报告容量不足
K=ceil(N/T)
blocks=min(C,M×K)
```

采用两波任务偏好，候选升序覆盖；它不是正确性硬条件或最优性证明。无满足并发偏好的候选时使用默认 T，但仍核对容量。所有连续 dtype 均使用 DMA/SIMD 提取 digit，64 位按字节视图搬运；低层 strided 使用 SIMT L2 构键；该差异不改变路由或 LSD 轮数。

## 3 活跃行与任务分配

令每行 workspace 字节数 `R=N(2D+2H)+256H×[K+D(K+1)]`，q=blocks/gcd(blocks,K)。按 [统一活跃行策略](../../tiling.md#多核活跃行容量)取 `P=max(q,floor((64MiB/R)/q)×q)`；仅 M≥2P 时启用容量 P，否则 P_act=M。64 MiB 是软件目标，最小均衡批次可能超过它。

一次 launch 内按 P_act 行分批，尾批 v≤P_act。Histogram 和 Scatter 均按 `task=core+k×blocks` 遍历 v×K；workspace 使用批内行号，输入/输出用 row_start+row。lookback 与局部排列、Scatter 在同一 tile 内执行，无独立逐行 Prefix 任务。所有核参加批首、跨 digit 及批间屏障。

route 3 的正 groups 表示活跃行容量，groups=0 表示全部 M 行。states 按 D×P_act×(K+1)×256 个 CounterT 分配。

## 4 范围与算例

行数、tile 数、地址与字节乘积按足够宽整数检查。状态使用负前缀编码，不占用计数字的状态高位；末轮才转换最终输出索引和应用名次区间。输出 L 不改变完整 N 的排序域及状态容量。

M=2、N=16385、C=8、U=65536、float32 时，T=1024/2048 满足至少两波任务，T=4096/8192 不满足；选 T=2048，K=9、blocks=8，局部预算 32768 字节，尾 tile 仅 1 项。task17 对应第二行最后一个 tile：统计覆盖其全部 256 桶，仅一个有效元素参与排列与散射；无任务核仍参加屏障。

来源：[PreparedSort](../../example/sort/entry.py)，执行及同步见 [Kernel](kernel-template.md)，完整空间公式见 [Buffer](dag-buffers.md)。
