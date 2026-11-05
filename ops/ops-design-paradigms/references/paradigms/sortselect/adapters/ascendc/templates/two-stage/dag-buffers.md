# AscendC 两阶段排序 Buffer 与依赖

## 1 分配表

E=B×N 是容量，D/J 为值/最终索引字节数，A 为 UB block 对齐。尾批 e≤E 不改变静态容量。

| 物理区 | 字节数 | 使用过程 |
|---|---|---|
| input / finalValues | `A(ED)` | 原值加载 → Stage1 读取结束 → 最终 values |
| stage1Values | `A(ED)` | Stage1 产生 → 逆映射读取或 Stage2 后 Gather |
| stage1Order | `A(4E)` | 全批原位置 → 紧凑键分支改为原轴索引 |
| finalIndices | `A(JE)` | 暂存负零标记或 uint16 key2 → 最终索引 |
| inverse / stage2KeysOut / gatherOffsets | `A(4E)` | 逆映射表，或 Stage2 键输出 → Gather 字节偏移 |
| stage2Order | `A(4E)`，仅紧凑键分支 | Stage2 排列 → Gather；可在后续阶段暂存收集的 int32 索引 |
| tmp | `max(A(tmp1),A(tmp2))` | Stage1 / Stage2 排序 scratch 顺序复用；逆映射没有 tmp2 |
| SIMT 保留 | 32768 | 单独扣除 |

## 2 预算推导

逆映射分支有两份 uint32 索引区，紧凑键分支有三份。总量：

```text
2×A(ED) + q×A(4E) + A(JE) + tmp + 32768 ≤ U
q = 2（inverse）或 3（packed）
```

finalIndices 的临时视图不再额外加一份 key2 或负零区；只有在其存活期与最终索引不重叠时才成立。J=4/8 必须与实际输出索引类型和 `InitSortBuffers` 一致。

## 3 复用次序

```text
input ──Stage1 最后读取──> 可成为 finalValues
finalIndices ──负零恢复或 Stage2 键最后读取──> 可成为最终索引
stage2KeysOut ──Stage2 完成──> 可成为 Gather 字节偏移
stage2Order ──value Gather 完成──> 可复用为收集的 int32 索引
最终 values/indices ──MTE3 结束──> 下一批可覆盖
```

关键不是“buffer 名称相同”，而是旧视图的最后消费者确已结束。生成不同 L 时，原始输入还可能被按索引回读，必须重新核对 input→finalValues 别名条件。

## 4 输出适配的峰值

最终 GM 输出是 M×L，与本批排序容量 E 分开。L=1 不能把 Stage1/inverse/key2 降到 B 项。若压紧输出复用既有最终区，记录压紧读写顺序及最后搬出依赖；新加临时区须加入上述预算并重新选 B。

没有算法必需的 GM 中间排序结果；框架 workspace、布局转换区与最终输出另列。算法及别名来源：[SmallAxisTwoStageBase](../../example/sort/op_kernel/arch35/common/small_axis_two_stage_base.h)，具体算例见 [Tiling](tiling.md#6-可复算例子)。

## 5 贯穿算例的物理角色变化

以下按 packed 分支追踪上一表的物理区；字节数和容量算例统一见 [Tiling](tiling.md#6-可复算例子)。

| 阶段 | 当前有效内容 | 最后消费者与随后的别名 |
|---|---|---|
| 搬入 / Stage1 | input=原值；stage1Values/order=排序输出 | Stage1 读完后 input 改作 finalValues |
| BuildStage2Keys | order 从批内 f 改为行内 j；finalIndices 暂存 uint16 key2 | key2 读者是 Stage2 |
| Stage2 | rankInverse 底层区暂存 uint16 排序键；stage2Order=perm2 | Stage2 完成后排序键不再使用 |
| 值 Gather | rankInverse 改作 uint32 `perm2×D`；finalValues 收集排序值 | 值 Gather 完成后 offset 和 perm2 才可改写 |
| 索引 Gather / Cast | offset 改为 `perm2×4`；stage2Order 暂存 int32 j | Cast 最后读取 j，finalIndices 改作 int64 输出 |
| CopyOut | finalValues / finalIndices 为 DMA 读端 | MTE3_MTE2/MTE3_S 后才能加载下一批 |

tmp 在两次 Sort 之间顺序复用，所以使用 max(tmp1,tmp2)；值/索引/排列区是不同物理区，不能据此也取 max。uint16 key2 的有效字节较少，承载它的 finalIndices 仍按最终索引宽度保留容量。

逆映射分支没有 stage2Order。其第一段循环先按 order 读取原位置的负零标记，恢复 stage1Values 并写 inverse；所有线程经过全块 barrier 后，第二段循环才扫描 inverse、写 finalIndices。因此标记最后读取先于同一区最终索引的首次写入；若融合或重排这两段，仍需保留整个线程组的该边界。
