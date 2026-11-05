# DSL 多核 Radix Tiling

## 1 输入与选路

M 为独立行数，N 为轴长，C 为当前流有效向量核数，U 为扣除 SIMT 32768 字节后的 UB。S 是保序键存储宽度 4/8；全局计数、偏移和原轴索引固定 int64。

小轴及单核局部 Radix 不可行时选择本路线。全局计数和前缀固定使用 int64，阶段之间通过全核屏障同步。

## 2 块长与任务数

每 tile 的线性 UB 需求为 7T 字节，固定区及余量按 1024 字节预算。当前选择算法：

```text
T = 1024
for candidate in [1024,2048,4096,8192]：
  Q = M×ceil(N/candidate)
  若 7×candidate+1024≤U 且 Q≥2C：T=candidate
若 7T+1024>U：在编译/完整 workspace 分配前报告容量不足
K = ceil(N/T)
tasks = M×K
blocks = min(C,tasks)
```

候选按升序覆盖，最终留下满足“两轮任务”性能偏好的最大块；若无候选满足该偏好，保留默认 1024，但仍检查其容量。两轮任务不是合法性的硬条件，也不是全局周期最优证明。

## 3 三个阶段的任务分配

| 阶段 | 工作域 | 核 c 的遍历 |
|---|---|---|
| Histogram | M×K 个行/tile | task=c+k×blocks，row=task//K、tile=task%K |
| Prefix | M 行 | row=c+k×blocks，每行 256 桶线程协作 |
| Scatter | M×K 个行/tile | 与 Histogram 相同 |

所有行在一次 launch 内完成。当前实现按 M 行分配 GM scratch，`P_act=M`；尚未实现有界活跃行工作区复用。不能只缩小分配量而保留全行索引。

## 4 计数与编码范围

每 tile 的局部桶计数为 uint16，T≤8192 满足其范围。ranks 使用 uint32 编码 `originalLocal×T+withinBucket`，最大 T²−1；T=8192 时小于 2^26。

行数、块数、全局前缀、地址与 workspace 字节乘积均按 int64/足够宽整数检查。输出长度 L 不改变中间 M×N 键/索引和 M×K×256 计数域。

## 5 演算

取 M=2、N=16385、C=8、U=65536。T=1024/2048 的任务数分别为 34/18，均满足 Q≥16；T=4096/8192 的任务数为 10/6，不满足该性能偏好，因此选 T=2048，K=9、blocks=8，尾 tile 只有 1 项。其局部预算为 15360 字节。

来源：[PreparedSort](../../example/sort/entry.py) 的多核分支。阶段计算与同步见 [Kernel](kernel-template.md)，完整 workspace 公式见 [Buffer](dag-buffers.md)。

本例的任务与输出追踪见 [Kernel §6](kernel-template.md#6-贯穿两个长行与单元素尾-tile)，工作区与存活阶段见 [Buffer §5](dag-buffers.md#5-贯穿算例的规模)。
