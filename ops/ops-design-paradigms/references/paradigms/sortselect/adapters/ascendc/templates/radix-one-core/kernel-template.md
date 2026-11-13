# AscendC 单核 Radix Kernel

输入契约：先实例化 [公共布局包装模板](../../layout-wrapper.md)，本模板接收连续 `[M,N]`、输出连续 `[M,L]`；输入 GM 按 `row*N+j` 寻址，原始轴与 stride 由调用层处理。白盒输入遵循同一交接。

## 1 执行骨架

```text
Init:
  输入、值输出、索引输出 Queue 各按 Host 下发份数 b 分配
  sortScratch 和 sourceOrder 独立分配，容量不随 b 重复计算
for round in [0,ceil(M/C_act)):
  row=round*C_act+blockIdx
  若 row>=M：跳过数据处理
  x=输入 Queue 取空槽；DMA 加载完整 N 项；入队再出队等待输入就绪
  浮点时：Vector 检测本行负零
    若存在：按原位置保存 sourceOrder，再批量生成比较输入
  values、indices=输出 Queue 取空槽
  ids32=索引 Queue 的 uint32 结果区，位置由 J 和对齐容量决定
  RADIX_SORT(values,ids32,x,sortScratch,N,方向)   # 隐式源索引重载
  若本行生成过零键：按 ids32 用 Vector 恢复 values 的原零符号
  输入最后读完后归还输入 Queue 槽
  J==8 时，对 N 项原轴索引做 Vector Cast；J==4 时使用 ids32
  用Vector对 [first_rank,first_rank+L) 批量选择/压紧已恢复的 values 与 indices
  两输出各自入队、出队交给 MTE3，写 row*L 起的 L 项并归还槽
  scratch/sourceOrder 下次使用前，其本行最后消费者必须完成
```

使用 `AscendC::Sort` 的隐式源索引重载。方向由 SortConfig 的 isDescend 配置；第二个模板参数不应猜成 stable 开关。稳定性与特殊值仍以真实调用合同及目标接口为准。

## 2 输入与索引地址

值的 DMA 长度为 ND 字节，索引为 NJ 字节，均不搬出对齐容量尾部。输入 GM 基址为 `row×N`。原生 Sort 产生的是本行 `0..N−1` 的索引，不能将 row×N 加到接口输出索引上。

int32 输出时直接写 uint32 索引区。int64 输出时，索引 Queue 的字节容量扩大两倍：先把 uint32 结果放在 `halfIndex=A(4N)/4` 之后的半区，再 Cast 到同一物理区的 int64 视图。halfIndex 始终是 uint32 元素数，不能乘错字节宽度。参考索引 GM 使用 uint32 载体，因此最终 GM 载体起点为 `row×N×(J/4)`，而值仍按 T1 元素寻址。

## 3 浮点原值恢复

参考 [signed_zero_sort_utils.h](../../example/sort/op_kernel/arch35/common/signed_zero_sort_utils.h) 的三个 helper 按 N 项 Vector/Reg 分工；没有负零的行直接进入原生 Sort。检测为真时的映射如下：

```text
HasNegativeZeroVec / HasNegativeZeroVecVF：按位检测-0，归约出是否需要处理
PrepareSignedZeroKeysVec / PrepareSignedZeroKeysVecVF：对原位置j
  isNegZero = bits(x[j]) == signBit
  sourceOrder[j] = uint32(j) | (isNegZero ? 0x80000000 : 0)
  x[j] = isNegZero ? +0位模式 : x[j]
Sort(values,ids32,x,tmp,N)：隐式原轴索引，sourceOrder不作为源索引传入
RestoreSignedZeroValuesByIndexVec / RestoreSignedZeroValuesByIndexVecVF：
  flags = Gather(sourceOrder,ids32) & 0x80000000
  对flags非零的输出lane写回-0位模式，其余保留排序值
```

signBit 为对应16/32位浮点的最高位，局部排序域须小于2^31以容纳标记。准备/恢复按 B32 索引的 Reg 容量分块；16位值的负零谓词经 `UnPack<LOWEST>` 扩为 B32 后选择 sourceOrder，恢复时 B32 标记谓词经 `Pack<LOWEST>` 转回 B16 后选择值。值和索引的 UpdateMask 使用独立计数，沿用[共享 mask 约束](../../kernel-helpers.md)，不改变 SIMD 分工或增设恢复 buffer。sourceOrder 仍为按原位置保存的恢复数据，ids32 才是 Sort 输出的原轴位置。

## 4 输出区间

完整 Sort 连续写 N 项；派生接口从排序后的 UB 取 `rank=first_rank+t`，输出到 `row×L+t`，indices 仍取该 rank 对应的原轴索引。中间值/索引、scratch 和原生 Sort 参数保持 N。

区间选择、索引宽化和压紧使用 Vector/Reg，消费已恢复的 UB 值/索引；输出 Queue 将紧凑结果交给 MTE3，不新增 SIMT 调用。源/目的别名和新增压紧区按 [Vector 区间写回](../../output-mapping.md#vector-路线ub-选择压紧与-dma)处理，本路线 `R_run=0`。

不同方向分别配置比较关系；不能将稳定升序输出倒序生成稳定降序。若仅输出 indices，可省最终 values 写回，但原生排序和必要的恢复数据按实际消费者核对，不能仅按接口字段名删区。

## 5 Queue 与复用

输入 EnQue/DeQue 保护 MTE2→Vector，输出 EnQue/DeQue 保护 Vector→MTE3。sourceOrder 与 tmp 是独立于 Queue 份数的共享物理区，下一行使用前应完成前一行相关读取。不能把增加双缓冲理解成所有临时区都自动独立复制。

检查跨核行覆盖、最后一轮不足 C_act 行、513 等非对齐轴长、两个索引宽度、重复值及正负零。来源：[SortRadixOneCore](../../example/sort/op_kernel/arch35/sort_radix_one_core.h)。资源计划见 [Tiling](tiling.md)。
