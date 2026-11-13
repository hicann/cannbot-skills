# AscendC 归并排序 Buffer 与依赖

## 1 分配表

H 为对齐轴长，B 为每批行数，b 为队列份数，D/J 为值/输出索引字节数。令 `p1=HBD`、`p2=HBJ`；H 与字节对齐后的搬运长度分别核对。

| 区域 | 字节数 | 存活范围 |
|---|---|---|
| 输入 value Queue | `b×p1` | MTE2 搬入至当前行/批次排序读完 |
| 输出 value Queue | `b×p1` | Extract/恢复至 MTE3 读完 |
| 输出 index Queue | `b×p2` | Extract/索引转换至 MTE3 读完 |
| 原始索引序列 | `4K` | 初始化后供 Sort32 使用 |
| proposal ping | `8K` | 初始 run 与交替归并层 |
| proposal pong | `8K` | 交替归并层 |
| BF16→FP32 转换区 | `4HB`，仅 BF16 | 输入转换至排序和转回完成 |

普通模式 K=H；float16/int64 的合批 proposal 模式 K=HB。proposal 两侧同一归并层分别作为读端和写端，不允许别名。

## 2 峰值

```text
UB_data = b×(2p1+p2) + K×(4+8+8) + BF16_cast
BF16_cast = 4HB 或 0
UB_peak = UB_data + adaptBytes + R_run ≤ U
```

`adaptBytes` 按[完整调用链预算](../../output-mapping.md#scratch-与-ub-由-host-规划)求值；本模板保留 Vector/Queue/MTE3，`R_run=0`。这是真实分配的求和，不把多个阶段分别求最大后误删仍存活的 Queue。普通模式的 proposal 不随 B 倍增，合批模式才随 B 倍增；切换模式时 Host 与 Kernel 必须使用同一条件。

当前参考 Host 的基础行批次计划按固定规划尺度产生，合批分支另作 requiredUb 检查。生成设计仍须将基础计划的真实峰值与 U 对照，不能把经验轴长门槛或规划尺度当作所有 dtype/输出类型的完整容量证明。

## 3 数据依赖

```text
input Queue → 类型/方向转换 → Sort32 → proposal ping
                                           ↕ 分层交替
                                      proposal pong
最终 proposal → Extract → 输出 Queue → MTE3
```

BF16 转换输入在当前行最后一次被排序读取之后，可以被当前行 Extract 结果覆盖；不能在 Sort32/MrgSort 尚未消费完时覆盖。合批中的不同文档阶段不自动意味着物理区已释放，要以最后读者为准。

## 4 输出适配

L<N 时在最后提取或 Vector 压紧中生成 B×L 的输出；proposal 仍按 H 或 HB。L=1 时保留行间并行和合批搬运，输出 Queue 仍由 MTE3 消费。压紧区采用存活期安全的别名或计入 adaptBytes 的独立分配，纯 Vector 路线不占 SIMT 保留区。

仅在目标合同需要时保留原始位模式、类别 mask/计数或区间选择临时区，并把真实对齐字节与最后读者计入峰值。优先保留排序携带的位模式；确需 UB 原值区时，它不能被构键、BF16 转换或 Extract 提前覆盖，恢复在该区按索引做 Vector Gather。类别信息必须在原值被转换/覆盖前生成，且不能从已经碰撞的 score 反推 NaN、+Inf 或 padding。B=1 也检查轴内分类/选择的临时区与同步，具体处理域见 [区间定位与原值恢复](kernel-template.md#区间定位与原值恢复)。

算法不需要 GM proposal scratch；框架 workspace、最终 M×L 输出和实际转置区另计。来源：[MergeSort::Init](../../example/sort/op_kernel/arch35/sort_merge_sort.h)，完整批次推导见 [Tiling](tiling.md)。

## 5 存活与槽位交接

| 阶段 | 同时存活的数据 | 可复用边界 |
|---|---|---|
| Sort32 | 当前输入、baseIndex、产生初始 run 的 proposal | 当前输入最后读取后释放输入 Queue |
| 分层归并 | source proposal 读端、destination proposal 写端 | 本层结束后交换角色，两块分配仍保留 |
| Extract / Cast | 最终 proposal、值输出、索引临时与最终视图 | Extract/Cast 消费完成后交给输出 Queue |
| CopyOut | 值和索引输出 | MTE3 完成后 Queue 槽位可重用 |

普通逐行模式在每行 Extract 后复用同一对 proposal；合批模式逐层覆盖本批所有行。尾批切回逐行计算不缩减静态分配。Queue 双缓冲也不自动提供两套可并发使用的 proposal。
