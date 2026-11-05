# AscendC 插入排序 Buffer 与依赖

## 1 物理缓冲

记 B 为容量规划的批次行数，N 为完整轴长，D/J 为值/索引字节数，A 为按 UB block 向上对齐。UB 分配按 B，尾批计算和搬运按 validRows。

| 缓冲 | 字节数 | 生产者 → 最后消费者 |
|---|---|---|
| 普通 values | `B×A(ND)` | MTE2 → 插入计算 → MTE3 |
| indices | `B×A(NJ)` | 索引初始化 → 与值同步移动 → MTE3 |
| BF16 搬运/转换区 | `B×A(2N)` | 搬入 → 转 FP32；排序后用于转回并搬出 |
| BF16 的 FP32 比较区 | `B×2×A(2N)` | 批量转换 → 插入 → 转回 BF16 |
| SIMT 保留 | 32768 | 运行时保留，不作为算法 buffer |

BF16 分支使用后两项代替普通 values；索引仍单列。输入和输出转换区可使用同一物理缓冲，因为原始输入完成转换后不再读取，最终结果转换在排序后发生。

## 2 峰值与布局

普通类型峰值为 `B×(A(ND)+A(NJ))+32768`。BF16 峰值为 `B×(3×A(2N)+A(NJ))+32768`。每个值/索引 buffer 都独立对齐，不能先把未对齐字节求和后只对齐一次。

本模板没有算法 GM scratch；参考工程仍保留框架 workspace 约定。最终输出 M×L 和实际发生的布局转换区单独计算，不能把它们当成 UB 预算的一部分。

## 3 复用边界

```text
Load BF16 ──转换──> FP32 values ──插入──> 转回 BF16 ──MTE3──> GM
indices  ──初始化───────────────> 随 value 移动 ──MTE3──> GM
                         批末等待全部搬出，再覆盖下一批
```

转换区重新作为输出区使用前，确认原始输入的最后一次转换读取完成。下一批 MTE2 不得覆盖上一批 MTE3 正在读取的区域。

## 4 设计核对

- 从[切分公式](tiling.md#2-ub-切分)独立复算 B；改变输出索引宽度要同步重算 indexPitch 和容量。
- 生成 L<N 时，values/indices 的有效输出变小，中间插入区仍是 B×N。
- 只有新增的临时结果与原数据没有重叠存活期时才允许别名；不能凭名称相同复用。

分配来源：[公共插入主体](../../example/sort/op_kernel/arch35/common/small_axis_insertion_base.h) 的 `InitInsertionBuffers`，写回来源：[StoreBatch](../../example/sort/op_kernel/arch35/sort_small_axis_insertion.h)。

## 5 阶段存活

| 阶段 | value / cast | index | 复用边界 |
|---|---|---|---|
| CopyIn / 转换 | MTE2 写原值，Cast 读原值并写比较值 | 旧内容已无消费者 | 上批 MTE3_S 后才加载 |
| 插入 | SIMT 读写比较值 | 初始化原轴坐标并随值移动 | 排序后仍有输出消费者 |
| CopyOut | 必要时转回存储类型，MTE3 读最终值 | MTE3 读最终坐标 | MTE3_S 后进入下一批 |

尾批仅减少有效访问，静态 B 行预留不变。若输出压紧需要新增 UB，排序结果的读端与压紧目标会同时存活，应将新增目标区加入预算。
