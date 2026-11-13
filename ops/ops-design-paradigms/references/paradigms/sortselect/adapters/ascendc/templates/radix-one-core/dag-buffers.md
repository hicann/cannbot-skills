# AscendC 单核 Radix Buffer 与依赖

## 1 物理区

记 X=A(ND)、I32=A(4N)、Y=I32×(J/4)，b 为 1/2 队列份数。

| 物理区 | 字节数 | 生命周期 |
|---|---|---|
| 输入 value Queue | bX | 当前行搬入至原生 Sort 最后读取 |
| 输出 value Queue | bX | 原生 Sort 输出至符号恢复及 MTE3 完成 |
| 输出 index Queue | bY | uint32 索引产生至宽化和 MTE3 完成 |
| sourceOrder | I32，仅浮点 | 零符号记录至按排序索引恢复完成 |
| tmp | 实际查询值 | 原生 Sort 调用期间 |

## 2 峰值与别名

总 UB 为 `b×(2X+Y)+sourceOrder+tmp+adaptBytes≤U`。本模板从排序、恢复到区间写回保持 Vector/Queue/MTE3，`R_run=0`；`adaptBytes` 按[完整调用链预算](../../output-mapping.md#scratch-与-ub-由-host-规划)求值，并参与 [Tiling 的单双缓冲选择](tiling.md#2-容量推导与双缓冲)。只有三个 Queue 随 b 倍增，sourceOrder 和 tmp 不倍增。

int64 输出 Queue 的 uint32 后半区与最终 int64 视图共用存储，宽化依据参考调用的有效区和重叠语义完成。不要另造一个不计预算的 int64 临时数组，也不要将后半区偏移当字节数使用。

## 3 依赖图

```text
输入 Queue → 浮点预处理 → Sort → 输出 value Queue → 恢复符号 → 搬出
                  sourceOrder ────────────────────────┘
                              输出 index Queue → 宽化 → 搬出
                              tmp 只在 Sort 调用内活跃
```

输入在 Sort 最后读完后可释放，不能因 values 最终不输出就提前覆盖 Sort 的输入。tmp 无后续消费者时可以复用，但引入其他别名仍需说明新数据的最后消费者和下一次 Sort 的覆盖边界。

## 4 GM 与派生输出

本模板没有算法 GM ping/pong；每核只保留当前行 UB。最终输出按 M×L、转置区按实际需要分别计算，框架 workspace 另列。L 只缩减最终写回有效域，不缩减 X、Y、sourceOrder 或 Sort scratch。

来源：[Kernel Init 与 ParseTilingData](../../example/sort/op_kernel/arch35/sort_radix_one_core.h)，字节计算及例子见 [Tiling](tiling.md#5-容量演算)。

## 5 阶段存活

| 阶段 | 当前读写区 | 复用边界 |
|---|---|---|
| CopyIn | 当前输入 Queue 槽被 MTE2 写入 | EnQue/DeQue 后供排序读取 |
| Sort / 恢复 | 输入、输出值/索引、tmp 和可选 sourceOrder | 排序及零符号恢复后释放输入 |
| 索引宽化 | 输出索引的 uint32 后半区与 int64 目标视图 | Cast 完成后交给输出 Queue |
| CopyOut | MTE3 分别读取两个输出槽 | Queue 管理槽位重用 |

uint32 后半区属于输出索引 Queue，不额外分配；改变重叠 Cast 的偏移、计数或类型时需核对真实 API 的支持条件。tmp/sourceOrder 只有一份，Queue 双缓冲不允许两个排序实例并发覆盖它们。
