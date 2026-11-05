# AscendC 归并排序 Kernel

## 1 计算流

```text
搬入一批行并补齐 → 比较类型/方向转换 → Sort32 生成 32 项有序 run
→ 四路 MrgSort，proposal ping/pong 交替 → Extract → 值/索引恢复 → 合批写回
```

本模板每行完整归并，H 为对齐轴长。原生 proposal 记录的偏移由 `GetSortOffset<T>` 换算，不能把普通 tensor 的元素或字节偏移直接用于 proposal。

## 2 行批次及搬入

`rowStart=(round×C_act+blockIdx)×B`，`v=min(B,M−rowStart)`。输入 GM 的第 b 行从 `(rowStart+b)×N` 读取 N 项，UB 第 b 行从 `b×H` 开始，补齐区不属于有效输出。

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
else if Host已选择合批且本批符合合批条件:
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

输入 Queue 的 EnQue/DeQue 保护搬入→计算，输出 Queue 保护计算→MTE3；每层 proposal 写入可见后才能被下一层读取，旧输入最后读完前不能交换别名。保留批量构键、转换、索引初始化和搬出，详见[辅助处理并行度](../../kernel-helpers.md#辅助处理并行度)。

## 6 推演与来源

N=65、H=96 时，初始 runCount=3，归并一层后得到一条 run；输出仍只有 65 项。H=160 时先处理 4+1 个 run，再合并两条 run。分别检查最终 ping/pong 选择和尾组掩码。

源码：[MergeSort](../../example/sort/op_kernel/arch35/sort_merge_sort.h) 的 `SortOneRow/SortBatchedRows/CopyIn/CopyOut`，以及 [proposal 公共归并](../../example/sort/op_kernel/arch35/common/ping_pong_merge_sort.h)。容量见 [Buffer](dag-buffers.md)。

## 7 搬运行距与 Queue 交接

令 `nD=A(ND)/D`、`nJ=A(NJ)/J`，a 为 UB block 字节数：

| 操作 | blockCount / blockLen | UB block 间隙 |
|---|---|---|
| CopyIn | v / ND 字节 | dstStride=`(H−nD)D/a` |
| CopyOut values | v / ND 字节 | srcStride=`(H−nD)D/a` |
| CopyOut indices | v / NJ 字节 | srcStride=`(H−nJ)J/a` |

GM stride 均为 0。以已有 N=65、H=96 的 float16/int64 例子说明：nD=80、nJ=68，值/索引 CopyOut 的 srcStride 分别为 1/7 个 block。CopyIn 先由 DMA 补到 nD，再由 Vector 补到 H；两段写入互不重叠。输出改成 L 时，按压紧后的物理行距重算 stride。

输入 Queue 交接后才排序，最后读完输入后释放；值与索引输出分别经 Queue 交给 MTE3。int64 索引的 uint32 临时视图从 HB 元素处开始，尾批也不改成 vH。proposal/baseIndex 是共享工作区，不随 Queue 份数复制；v<4 时合批计算可回到逐行归并，静态容量不变。
