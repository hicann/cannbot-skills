# AscendC 归并排序 Kernel

输入契约：先实例化 [公共布局包装模板](../../layout-wrapper.md)，本模板接收连续 `[M,N]`、输出连续 `[M,L]`；输入 GM 按 `row*N+j` 寻址，原始轴与 stride 由调用层处理。白盒输入遵循同一交接。

## 1 计算流

```text
搬入一批行并补齐 → 比较类型/方向转换 → Sort32 生成 32 项有序 run
→ 四路 MrgSort，proposal ping/pong 交替 → Extract → 值/索引恢复 → 合批写回
```

本模板每行完整归并，H 为对齐轴长。原生 proposal 记录的偏移由 `GetSortOffset<T>` 换算，不能把普通 tensor 的元素或字节偏移直接用于 proposal。

## 2 行批次及搬入

`rowStart=(round×C_act+blockIdx)×B`；先判断 `rowStart≥M` 时跳过本批，否则 `v=min(B,M−rowStart)`。输入 GM 的第 b 行从 `(rowStart+b)×N` 读取 N 项，UB 第 b 行从 `b×H` 开始，补齐区不属于有效输出。

行号使用 int64，GM 元素偏移使用 uint64。Tiling 的核数、批行数和轴长虽为 uint32，其乘积必须在乘法前提升；把 32 位乘积赋给 64 位变量无法避免回绕。参考实现预计算并递增：

```text
coreRowStart = int64(blockIdx) * B
rowStride = int64(C_act) * B
coreGmOffset = uint64(coreRowStart) * N
gmStride = uint64(rowStride) * N
rowStart = coreRowStart; gmOffset = uint64(0)
每轮：
  若rowStart<M：
    v=min(B,M-rowStart)
    从输入GM的gmOffset+coreGmOffset搬入v行，排序后向同一元素偏移写回values/indices
  rowStart += rowStride
  gmOffset += gmStride
```

`GetRowCount` 直接消费 int64 行号，搬入和搬出复用同一组 uint64 偏移，不再各自用轮号重算 32 位连乘。values/indices 的偏移以各自类型的元素数计。生成长度 L 的输出时，输入仍使用 N，输出核基址和轮间步长改按 L 计算，并保留 64 位运算。

`CopyIn` 的升序 paddingValue 为 `NAN`，降序为 `-INFINITY`。MTE2 只完成字节对齐范围内的补齐；剩余 Sort32 尾部由 Vector 用同一 paddingValue 补齐。不能假设一次 DataCopyPad 自动完成任意 H−N 项 padding。

## 3 run 构造与四路归并

```text
baseIndex[j]=j，j∈[0,排序域长度)              # 用Reg按向量分工生成
整批v*H项统一完成比较类型转换及升序符号翻转
对一个普通行或一组可合批行：
  Sort32：每32项生成一个降序proposal run
  runLength=32; runCount=H/32; source=ping; destination=pong
  while runCount>1:
    for proposal域中的每一行 b:             # 普通模式只有当前行，b=0
      fullGroups=runCount//4; tailRuns=runCount%4
      if tailRuns==0 and H!=runCount*runLength:
        fullGroups-=1; tailRuns=4            # 最后一个run不足，整组留给尾部
      if fullGroups>0:
        input[i]=source[ProposalOffset(b*H+i*runLength)]，i=0..3
        output=destination[ProposalOffset(b*H)]
        一次MrgSort(counts=[runLength]*4, validBit=0xf, repeatTimes=fullGroups)
      tailStart=fullGroups*4*runLength
      if tailRuns==1:
        复制source中从ProposalOffset(b*H+tailStart)开始的剩余proposal
      else if tailRuns>=2:
        counts[i]=min(runLength,H-tailStart-i*runLength)，i<tailRuns；其余为0
        input[i]=source[ProposalOffset(b*H+tailStart+i*runLength)]，i<tailRuns
        其余input指向尾组起点；output=destination[ProposalOffset(b*H+tailStart)]
        一次MrgSort(counts, validBit=(1<<tailRuns)-1, repeatTimes=1)
    等本层proposal写入可供下一层消费；交换source/destination
    runLength*=4; runCount=ceil(runCount/4)
  Extract 从最终source提取值和uint32索引
  合批时Vector将第b行索引减去b*H，恢复原轴位置
整批v*H项统一恢复升序符号；BF16按参考路径整批转回
值/索引按H行距读取目标名次区间，再做必要的索引宽化和批量搬出
```

完整四路分组由一次带 repeatTimes 的 MrgSort 覆盖，尾组再单独提交；不是逐 group 启动归并。尾组为 1–4 个 run，counts/validBit 反映真实 proposal 列表，不用原输入 N 截短已生成的 run。四路归并的层数和最终所在侧一起维护，不能固定从 ping 抽取。

## 4 方向、类型与合批

参考 Sort32/MrgSort 包装产生降序 proposal：升序先翻转浮点符号位，将 −x 降序排列，再恢复符号；降序直接排序原值。这不是 Radix 的无符号单调键变换。NaN、相等值次序和零符号按实际包装及接口合同核对。

BF16 先批量转 FP32，在 FP32 区完成每行排序，再转回。转换区可在当前行形成 proposal 且不再读取输入后承接提取结果。

合批模式先对 v×H 项一次 Sort32，每一归并层再逐行处理各行 run，最后一次 Extract。生成的批内索引减去各行的 `b×H` 得到原轴索引；行间不能互相归并。普通模式复用同一对 proposal 依次完成各行。

两种模式的处理域不同，执行骨架为：

```text
一次多block DMA搬入v行并补到H，输入Queue交给Vector
if BF16:
  一次Vector Cast将v*H项扩为FP32
  升序时一次翻转v*H项符号位
  逐行执行完整proposal归并，复用同一对单行proposal
  每行Extract后本行转换输入已无读者，可存放提取值
  升序时一次恢复v*H项符号位
  一次Vector Cast将v*H项转回BF16
else if batchMerge==1 and v>=4:             # Host已按类型、E及完整调用链预算选定
  升序时一次翻转v*H项符号位；一次Sort32覆盖v*H项
  每个归并层处理完全部v行再换层；一次Extract覆盖v*H项
  按行向量化修正索引，整批恢复值
else:
  升序时一次翻转v*H项符号位
  逐行完成Sort32、全部归并层和Extract
  proposal按行复用，输入/结果仍属于本批v行
  升序时一次恢复v*H项符号位
整批转换所需索引宽度，values/indices分别入输出Queue
合批写回v行；L<N时先沿v*L域压紧再写回
```

普通与合批模式都按 v×H 批量翻转/恢复符号；BF16 Cast 同样保留整批域。合批进一步减少 Sort32、Extract 的提交次数，跨行归并边界保持独立。逐行 proposal 复用仍使用 Vector 原语。具体可合批条件由 [Tiling](tiling.md) 下发，不在 Kernel 中另造启发式选路。

## 5 写回与同步

完整 Sort 合批搬出 v 行，每行只写 N 项。生成区间输出时，在最终提取/压紧阶段取 `[first_rank,first_rank+L)`，输出 GM 行距改为 L；Sort32、归并和必要的索引恢复仍处理完整排序域。

### 区间定位与原值恢复

归并已产生有序 proposal，Extract 后的索引就是名次到原轴位置的映射。其比较键、有效性及特殊值顺序已经满足目标合同时，收尾只处理 v×L 项：

```text
p 按 Reg lane 分配到 [0,v*L)
b=p//L; t=p%L; rank=first_rank+t
j=final_ids[b*indexPitch+rank]            # 合批偏移已恢复；0≤j<N
indices[b*L+t]=widen(j)                  # 只宽化实际输出项
若原值已随排序恢复：values[b*L+t]=sorted_values[b*valuePitch+rank]
否则：values[b*L+t]=original_values[b*originalPitch+j]  # 安全保留的未排序原值区
紧凑输出UB交给输出Queue，由MTE3合批搬出v*L项
```

rank 索引有序值区与有序索引区，j 索引未排序原值区；不能用 j 再读取已排序的 values。两种值来源按实际恢复链选定，所需原值区计入存活期与容量。上述选择/压紧按整个 v×L 域生成 Gather 偏移并批量执行；归并层内允许的逐行 proposal 处理不延伸为收尾逐行提交 `count=L` 的 Muls/Gather。

这是截取，不需要再遍历全部有序索引来寻找第 k 项。L=1、v=1 时，少量有效 lane 完成这一次定位是合理的；若恢复中仍需遍历 N 或 H 项，该完整轴工作应另以连续轴位置分配 lane/线程，不能沿用“一行一个 lane”执行 H 次串行迭代。批内多行并行不能代替 B=1 时的轴内并行。

本模板从构键、类型转换、排序、恢复到最终写回均保留 Vector/Reg 与 Queue/MTE3。原值恢复优先保留排序携带的位模式；构键覆盖输入 UB 时，在覆盖前保留目标合同确需的原值/类别区，按最终索引做 Vector Gather。输出选择与压紧按 [Vector 区间写回](../../output-mapping.md#vector-路线ub-选择压紧与-dma)执行。新增保留区及最后读者见 [Buffer 输出适配](dag-buffers.md#4-输出适配)。

### 特殊值键碰撞

直接按名次读取的前提是排序结果已表达目标顺序。例：升序用 −x 形成降序 score，又将 NaN 和 padding 都映射为 −Inf，会与合法 +Inf 碰撞；原语对相等 score 稳定也不能把混排的 +Inf/NaN 自动分成两个类别。不能丢弃 NaN、忽略合法无穷或直接取错误的 proposal 名次。

对这个具体键布局，一种可推导的适配是：构键时沿轴并行分类，得到 `F=非NaN且非+Inf项数`、`P=+Inf项数`，保留原轴类别及原始索引。目标 rank<F 时直接取有序索引；`F≤rank<F+P` 时取原轴序中第 `rank−F` 个 +Inf；其余有效 rank 取第 `rank−F−P` 个 NaN。类别内按原轴顺序做并行 mask/前缀定位，padding 不参与计数。这样普通名次无需再过滤整行，特殊桶也不靠逐项串行计秩。这里是算法候选，不是 SDK 已有接口或已验证的完整 wrapper；实际 prefix/选择实现、稳定性、位模式及资源仍需按目标 SDK 核对。其他方向或 NaN 合同应重新推导类别顺序。

完整轴的必要分类/计数尽量并入已有构键遍历；若仍需独立遍历，展开轴内分工和依赖，不能只写一个“恢复 VF”掩盖 O(H) 的串行依赖链。保留既有 Sort32/MrgSort 原语并不代表这类收尾机制与参考等价。

输入 Queue 的 EnQue/DeQue 保护搬入→计算，输出 Queue 保护计算→MTE3；每层 proposal 写入可见后才能被下一层读取，旧输入最后读完前不能交换别名。保留批量构键、转换、索引初始化和搬出，详见[辅助处理并行度](../../kernel-helpers.md#辅助处理并行度)。

## 6 推演与来源

N=65、H=96 时，初始 runCount=3，归并一层后得到一条 run；输出仍只有 65 项。H=160 时先处理 4+1 个 run，再合并两条 run。分别检查最终 ping/pong 选择和尾组掩码。

源码：[MergeSort](../../example/sort/op_kernel/arch35/sort_merge_sort.h) 的 `SortOneRow/SortBatchedRows/CopyIn/CopyOut`，以及 [proposal 公共归并](../../example/sort/op_kernel/arch35/common/ping_pong_merge_sort.h)。容量见 [Buffer](dag-buffers.md)。

64 位行号与 GM 偏移递增方式同步自 `ops-math` 提交 `bcbaab4745aa08ef96f6e0439f849080b89a76e9` 的 `math/sort/op_kernel/arch35/sort_merge_sort.h`。

## 7 搬运行距与 Queue 交接

令 `nD=A(ND)/D`、`nJ=A(NJ)/J`，a 为 UB block 字节数：

| 操作 | blockCount / blockLen | UB block 间隙 |
|---|---|---|
| CopyIn | v / ND 字节 | dstStride=`(H−nD)D/a` |
| CopyOut values | v / ND 字节 | srcStride=`(H−nD)D/a` |
| CopyOut indices | v / NJ 字节 | srcStride=`(H−nJ)J/a` |

GM stride 均为 0。以已有 N=65、H=96 的 float16/int64 例子说明：nD=80、nJ=68，值/索引 CopyOut 的 srcStride 分别为 1/7 个 block。CopyIn 先由 DMA 补到 nD，再由 Vector 补到 H；两段写入互不重叠。输出改成 L 时，按压紧后的物理行距重算 stride。

输入 Queue 交接后才排序，最后读完输入后释放；值与索引输出分别经 Queue 交给 MTE3。int64 索引的 uint32 临时视图从 HB 元素处开始，尾批也不改成 vH。proposal/baseIndex 是共享工作区，不随 Queue 份数复制；v<4 时合批计算可回到逐行归并，静态容量不变。
