# AscendC 两阶段排序 Kernel

## 1 整体计算流

```text
B 行紧凑加载 → Stage1 跨行稳定排序
             ├─ 逆映射 → 统计行内秩 → 生成 values / 原轴 indices
             └─ uint16 紧凑键 → Stage2 升序排序 → Gather values / 原轴 indices
             → 整批写回 → 下一批复用
```

两种恢复分支共用一个模板。核 c 按 `batchId=c+k×C_act` 处理，`rowStart=batchId×B`，有效行数 v，`e=v×N`。Stage1 只排序 e 个有效元素；容量仍按 B×N 分配。

## 2 Stage1 的索引语义

输入展平位置 `f=b×N+j`。Stage1 按目标方向排序原 dtype，输出 `stage1Values[r]` 和 `order[r]=f`，r 是全批排序名次。排序必须满足目标稳定性契约；后续分行不能修复 Stage1 的不稳定。

本实现采用隐式源索引的 `AscendC::Sort` Radix 重载。浮点逆映射分支在 Stage1 前记录负零标记并形成比较键，恢复时按原位置保存的标记还原值位模式。

## 3 逆映射分支

```text
一次 SIMT 调用覆盖本批 e=v*N 项，线程 t 执行：
  for r=t; r<e; r+=THREADS:
    f=order[r]
    浮点有负零标记时，按原位置 f 恢复 stage1Values[r] 的零符号
    inverse[f]=r
  全块 barrier，所有线程参加，包括本轮没有有效 r 的线程
  for f=t; f<e; f+=THREADS:
    b=f//N; j=f%N; rank=inverse[f]
    acc[0..7]=0; k=0
    while k+7<N:
      并列执行 acc[u] += (inverse[b*N+k+u]<rank)，u∈[0,8)
      k+=8
    pos=sum(acc)
    while k<N: pos+=(inverse[b*N+k]<rank); k+=1
    finalValues[b*N+pos]=stage1Values[rank]
    finalIndices[b*N+pos]=j
```

inverse 中的秩唯一，即使值相等也能得到唯一 pos。每行扫描 N 次，共 O(vN²) 次比较；参考 VF 用多路独立累加器展开扫描以降低依赖链。barrier 必须由全块线程参与。

八路累加器表达的是独立计数链，不是八次串行更新同一个 pos。SIMT 按 e 个元素分工，Stage1 仍用全批 Radix 排序；这两级并行共同构成逆映射路线。

## 4 紧凑键分支

```text
并行处理 Stage1 的每个名次 r：
  f = order[r]; b = f // N; j = f % N
  key2[r] = b×e+r
  order[r] = j
Stage2 对 key2 做升序排序，得到 perm2
finalValues[t] = stage1Values[perm2[t]]
finalIndices[t] = order[perm2[t]]
```

行号在高位、Stage1 秩在低位，因此 Stage2 先按行归组，再保留每行的目标方向。Stage2 始终升序，包括 Stage1 为降序的情况。`key2` 使用 uint16，Host 检查 `B²N≤65536`；尾批按 e 形成键。

Gather 的 offset 是字节偏移：值使用 `perm2×D`，索引使用 `perm2×4`。int64 输出索引先收集 uint32 原轴索引再 Cast；不能用值的字节偏移读取索引。

恢复阶段的 Vector 执行顺序为：

```text
Stage2Sort(key2, 升序, e) → sortedKeys、perm2
offset = VectorMultiply(perm2,D)
finalValues = Gather(stage1Values,offset,e)
offset = VectorMultiply(perm2,4)          # 值 Gather 最后读完后复用 offset
gatheredIndex = Gather(stage1Order,offset,e)
finalIndices = J==8 ? VectorCastToInt64(gatheredIndex,ceilAlign(e,4)) : gatheredIndex
```

sortedKeys 的存储在键已无消费者后用作 uint32 offset；perm2 在最后一次形成 offset 后可用于暂存 gatheredIndex。复用依据是最后消费者，分配容量仍覆盖两个角色中较大者。Stage2 排列和两次 Gather 都处理 e 项；参考 `BuildOutputs` 的 int64 Cast 实参为 ceilAlign(e,4)，额外 lane 不属于有效结果。这里记录实际 caller 的长度，不将4对齐当作通用 API 要求；所用 SDK 的完整 Reg 尾读、初始化与别名约束另按实现核对。输出区间只改变最后 v×L 项的压紧/搬出。

## 5 整批写回与同步

完整 Sort 的两个结果 buffer 均是 v×N 的紧凑数组，各用一次连续搬运写出。输入区在 Stage1 读完后别名为 finalValues；写回前 V→MTE3 保护生成结果，批末 MTE3→MTE2 保护下一次加载，MTE3→Scalar 保护后续控制流。

派生输出在行内 pos/rank 已确定后，只写 `first_rank≤pos<first_rank+L`，目标偏移为 `(rowStart+b)×L+(pos−first_rank)`。仍按 v×L 并行处理，不在外层逐行启动恢复 VF；inverse 建表、Stage1 和 Stage2 仍覆盖完整 e。

## 6 手算验证

输入两行 `[3,1,5]`、`[4,2,6]`，Stage1 升序 `order=[1,4,0,3,2,5]`。inverse 为 `[2,0,4,3,1,5]`。紧凑键为 `[0,7,2,9,4,11]`，Stage2 排列为 `[0,2,4,1,3,5]`。两分支均恢复 values=`[1,3,5,2,4,6]`，indices=`[1,0,2,1,0,2]`。

实现来源：[两阶段公共主体](../../example/sort/op_kernel/arch35/common/small_axis_two_stage_base.h) 与 [Sort 包装](../../example/sort/op_kernel/arch35/sort_small_axis_two_stage.h)。物理别名见 [Buffer](dag-buffers.md)。

## 7 从 Init 到最终 DMA 的执行骨架

本模板使用 TBuf；`inputValues` 与 `finalValues` 是同一物理区的两个阶段角色。下面的阶段名与公共主体对应，省略模板签名而保留调用顺序：

```text
Init:
  E = B*N
  分配 input、stage1Values、stage1Order、finalIndices、rankInverse、tmp
  packed 时另分配 stage2Order
  建立 finalValues=input、key2=finalIndices 等别名视图
Process(core c):
  for batchId=c; batchId<Q; batchId+=C_act:
    rowStart=batchId*B; v=min(B,M-rowStart); e=v*N
    DataCopyPad(inputValues, x[rowStart*N], {1,e*D,0,0,0})
    SetFlag/WaitFlag<MTE2_V>
    Stage1Sort(e)
    if inverse:
      RankInverseScatter(e,N): 建 inverse → syncthreads → 行内计秩与恢复
    else:
      BuildStage2Keys(e,N) → Stage2Sort(e) → BuildOutputs(e)
    SetFlag/WaitFlag<V_MTE3>
    DataCopyPad(values[rowStart*N], finalValues, {1,e*D,0,0,0})
    DataCopyPad(indices[rowStart*N], finalIndices, {1,e*J,0,0,0})
    SetFlag/WaitFlag<MTE3_MTE2>
    SetFlag/WaitFlag<MTE3_S>
```

`DataCopyExtParams` 中 blockLen 为字节，GM `operator[]` 的 rowStart×N 则为对应字段的元素数。当前批次在 UB 中紧凑排列，整批复制一次即可，不使用逐行 pitch。

## 8 跨流水和别名的交接

| 生产/消费 | 被保护数据 | 当前实现的交接 |
|---|---|---|
| MTE2 → Stage1 | inputValues | `MTE2_V` 成对 SetFlag/WaitFlag |
| inverse 建表 → 同块计秩 | rankInverse | RankInverseScatter 内全部线程执行 `asc_syncthreads()` |
| Stage2 → Gather | perm2、排序键与原索引 | 保持 Sort、Muls、Gather 的既有调用依赖；值 Gather 完成后才改写索引 offset |
| 最终恢复 → MTE3 | finalValues、finalIndices | `V_MTE3` 成对事件 |
| MTE3 → 下一次加载 | input/finalValues 同一区 | `MTE3_MTE2`，避免写回尚在读时重载 |
| MTE3 → 下一批控制 | 本批 DMA | `MTE3_S` 后才推进批次 |

packed 分支的 `stage2KeysOut` 在第二次排序结束后转为 uint32 字节 offset；先生成 `perm2×D` 并收集值，再生成 `perm2×4` 收集原轴索引。int64 结果使用 stage2Order 暂存收集的 int32 索引，再 Cast 到 finalIndices；此时 perm2 已无消费者。SIMT VF 的调用完成与原生 Vector 原语的依赖遵循现有调用机制，不额外臆造每两个函数之间的通用硬事件。

## 9 贯穿：B=16 的最后一批

沿用 [Tiling 算例](tiling.md#6-可复算例子)。核 16 的 batchId 为 16/33/50/67；最后一批 rowStart=1072、v=16、e=4096。整批 DMA 的起点按 rowStart×N 求，值/索引长度分别为 eD/eJ 字节。

批内原位置 f=3847 表示 b=15、原轴 j=7，对应全局行 1087。若恢复后行内名次为 2，则完整 Sort 写到元素 `1087×256+2`；L=5、first_rank=0 时改写到元素 `1087×5+2=5437`，输出索引仍为 7。

本批短输出为 16×5 项，Stage1 和 Stage2 仍处理 4096 项。压紧在最终 B×L 域并行完成，若需要新临时区则加入容量预算；输出地址变化不改变原值、原轴索引和行内名次三者的含义。

### 真正不足 B 行的尾批

另设 int32 输入、int64 索引、M=1091、N=255、B=16、C_act=17，作为给定批次参数的边界演算，不宣称是 Host 搜索结果。Q=69，最后 batchId=68 由核0处理：rowStart=1088、v=3、e=765，分配容量仍为 E=4080。

| 对象/操作 | 本例长度 | 含义 |
|---|---:|---|
| 容量与别名 | E=4080 | 各物理 buffer 按 Buffer 文档的 A(ED)、A(4E)、A(FE) 等分配 |
| Stage1 / Stage2 / 两次 Gather | e=765 项 | 仅处理本批有效域 |
| uint16 紧凑键 | `b×765+r` | 使用实际 e；最后一行 b=2、r=764 时 key=2294 |
| int64 Cast | ceilAlign(765,4)=768 项 | 源/目的容量覆盖操作长度；SDK更宽的尾读另核对 |
| 完整输出 DMA | 值3060 / 索引6120字节 | int32/int64下分别为 eD/eJ，不搬Cast的3项尾部 |

输入 DMA 同样只搬 eD 字节，排序 count 和输出长度均不随物理容量或 Cast 对齐扩大。短输出 L 时仍只压紧/搬出 v×L 项。
