# AscendC 插入排序 Kernel

输入契约：先实例化 [公共布局包装模板](../../layout-wrapper.md)，本模板接收连续 `[M,N]`、输出连续 `[M,L]`；输入 GM 按 `row*N+j` 寻址，原始轴与 stride 由调用层处理。白盒输入遵循同一交接。

## 1 数据流与参数

```text
连续 B 行搬入 → 必要的比较类型转换 → 各行稳定插入 → 必要的值类型恢复 → 合批搬出
```

Kernel 消费 `[M,N]`、B、Q、C_act。每核令 `w=ceil(Q/C_act)`，遍历 `batchId∈[blockIdx×w,min(Q,(blockIdx+1)×w))`，`rowStart=batchId×B`，`validRows=min(B,M−rowStart)`。每行原轴索引从 0 初始化，与值一起移动。

## 2 搬入与 UB 地址

第 b 行第 j 项的输入 GM 元素偏移为 `(rowStart+b)×N+j`。普通值的 UB 元素偏移为 `b×valuePitch+j`，索引为 `b×indexPitch+j`，两个 pitch 独立对齐。

BF16 先搬入 BF16 区，再批量转换为 FP32；转换区和比较区使用一致的行元素跨度。线程只访问 `b<validRows、j<N`，不把 pitch 补齐区视为有效元素。

## 3 稳定插入计算

以下是算法伪码，比较关系包含目标接口约定的 NaN、正负零和方向处理；具体类型 helper 见[辅助约束](../../kernel-helpers.md)。

```text
每核按分配的 batchId 区间循环：
  rowStart=batchId*B; v=min(B,M-rowStart)
  一次多 block DMA 搬入 v 行，每 block 仅 N 项，UB 行距分别计算
  等待 MTE2_V；BF16 时对本批 v*valuePitch 项做一次 Vector Cast
  一次 SIMT 调用处理本批，线程 t 执行：
    for b=t; b<v; b+=THREADS:
      rowValue = values + b*valuePitch
      rowIndex = indices + b*indexPitch
      for j in [0,N): rowIndex[j]=j
      for j in [1,N):
        savedValue, savedId = rowValue[j], rowIndex[j]
        pos=j
        while pos>0 and 前项按目标方向严格排在 savedValue 后面:
          rowValue[pos], rowIndex[pos] = rowValue[pos-1], rowIndex[pos-1]
          pos-=1
        rowValue[pos], rowIndex[pos] = savedValue, savedId
  排序生产完成后，BF16 时整批 Vector Cast 回存储类型
  对 v*L 项做一次批量输出适配，再合批提交 values / indices
  等待当前输出的最后读者完成，复用本批 UB
```

SIMT 线程 t 处理行 `b=t+k×1024<validRows`；核间批次区间和核内行分工分别计算。只移动“严格排在后面”的元素，等值项保持原先顺序；稳定降序使用方向化比较，不能将稳定升序结果倒序。

这里的并行单位是行，行内插入循环有顺序依赖；适用于短 N，不把整批 v 行都交给一个线程。THREADS 是 VF 的线程块参数，与 C_act 个核不同。L<N 时只缩小最后的输出域，排序仍覆盖每行完整 N 项。若按原索引回读原值，线程沿 v×L 项分工，值/索引的压紧或写回在同一批内完成。

## 4 写回与输出区间

完整 Sort 按 validRows 个 block 搬出，每个 block 只包含 N 项。blockLen 为 ND/NJ 字节，UB srcStride 分别为 `floor((valuePitch−N)D/a)`、`floor((indexPitch−N)J/a)` 个 block，GM dstStride 为 0；a 是 UB block 字节数。非对齐长度在 UB 侧占据对齐跨度，不能混用有效长度和物理行距。BF16 在搬出前批量转回；两个字段的字节数分别为 `N×D`、`N×J`。

派生接口在最终写回取 `rank=first_rank+t，0≤t<L`，输出偏移为 `(rowStart+b)×L+t`。索引仍为原轴坐标，不能加 rowStart 或 batchId。保留原有合批搬运；L<N 时按 B×L 工作域压紧或并行写出，见[批次写回粒度](../../output-mapping.md#保持批次写回粒度)。

## 5 同步与批次复用

- MTE2 搬入后，计算才能读取 UB；类型转换完成后，SIMT 比较才能消费转换区。
- 所有行排序完成后，合作写回才能读结果；V→MTE3 事件保护值、索引及转换输出。
- 当前 MTE3 读者结束后，下一批才能覆盖同一 UB；参考包装在批末等待 MTE3→Scalar 完成。

## 6 推演样例与实现核对

一行 `[3,1,3,2]` 升序后为 `[1,2,3,3]`，原轴索引为 `[1,3,0,2]`；降序为 `[3,3,2,1]`，索引 `[0,2,3,1]`。两行合批时每行索引各自从 0 开始。

源码：[插入 Kernel](../../example/sort/op_kernel/arch35/sort_small_axis_insertion.h) 的 `ProcessBatch/StoreBatch`，以及[公共插入主体](../../example/sort/op_kernel/arch35/common/small_axis_insertion_base.h)。容量推导见 [Tiling](tiling.md)。
