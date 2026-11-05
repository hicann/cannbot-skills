# DSL 多核 Radix Buffer

## 1 每核 UB

T 为所选 tile 长度。

| 区域 | 字节数 | 用途 |
|---|---|---|
| digits | T | 当前 digit |
| permutation | 4T | 排序位置到原 tile 位置 |
| scratch | T+VECTOR_BYTES | 字节稳定分组临时区 |
| byte_order | T | 四轮 2-bit 分组的 lane 排列 |
| local offsets | 512 | uint16 的 256 桶起点 |
| warp totals | 64 | 8 个 int64 warp 总量 |

线性项为 7T，Host 使用 `7T+1024` 覆盖固定区、向量额外区和对齐余量；SIMT 保留 32768 字节另行扣除。Histogram/Scatter 使用局部排列区，Prefix 使用 warp totals，实际所有静态 Buffer 的分配仍须计入预算。

## 2 GM workspace

M 为总行数，N 为轴长，K=ceil(N/T)，S 为键字节数。

| 区域 | 字节数 | 存活范围 |
|---|---|---|
| ping/pong keys | `2MN S` | 跨字节轮的键重排 |
| ping/pong ids | `2MN×8` | 贯穿全部轮次的原轴坐标 |
| ranks | `MN×4` | Histogram 产生至本轮 Scatter 完成 |
| hist | `MK×256×8` | Histogram 产生至 Prefix 完成 |
| offsets | `MK×256×8` | Prefix 产生至 Scatter 完成 |

当前公开包装两侧完整分配时，总算法临时字节数为：

```text
MN(2S+20) + 16MK×256
```

其中 20=两侧 int64 ids 的 16 字节+uint32 ranks 的 4 字节；hist 和 offsets 各 8 字节。此值不含最终输出、布局转换、缓存池或运行时内部占用。特定一字节配置若省一侧，必须同步调整实际签名与消费者，不能仅修改预算表达式。

## 3 各轮复用

非末 digit 的 hist/ranks/offsets 在该轮 Scatter 及全核屏障结束后重用；末 digit 完成最终输出后不再复用这些状态。ping/pong 只有完成全部散射后才交换读写角色。它们不能彼此别名。

本实现按全部 M 行分配，即 P_act=M，固定 N/C 后增大 M，scratch 也随 M 增长。当前实现尚未将工作区限制为固定活跃行数。最终 values 从原输入回读，原输入必须保留至最后一轮完成。

## 4 输出区间与布局

最终字段按 M×L 分配，临时键与状态仍按完整 N。公开非末轴包装单独分配实际转入/转出缓冲，不把这些布局区计入上面的算法 scratch。

若实现有界 P_act，必须同时改 Host 分配、编译签名、三个阶段任务域与 workspace 行坐标；阶段屏障仍由全部核参与。修改前后用相同输入/输出与完整提交粒度比较，不能只用空间公式证明实现已完成。

来源：[PreparedSort.workspace](../../example/sort/entry.py) 与 [RadixMoreCore.__init__](../../example/sort/sort_radix_more_core.py)。任务选择见 [Tiling](tiling.md)。

## 5 贯穿算例的规模

沿用 M=2、N=16385、T=2048 的 float32 例子，K=9、S=4。代入 workspace 公式得到算法临时量 991288 字节，各 Tensor 独立分配，其内地址不能使用跨 Tensor 的累计偏移。

每核局部声明为 `7T+VECTOR_BYTES+512+64`；当前 VECTOR_BYTES=256 时为 15168 字节，Host 预算为 15360。声明字节和、预算余量与编译后的实际对齐分别核对。

## 6 阶段存活与最终轮的差异

| 阶段 | GM 保留内容 | 本核 UB 角色 |
|---|---|---|
| Histogram | 当前 keys/ids；产生 hist/ranks；首轮还初始化当前 keys/ids | digits → permutation → ranks 编码；local offsets 供桶内秩使用 |
| Prefix | hist/ranks 仍存活，产生全局 offsets | 8 项 totals 供跨 warp scan；局部排序区已无本阶段消费者 |
| Scatter | 当前 keys/ids、ranks、offsets；非末轮产生另一侧 | SIMT 直接读取 GM，不将整行搬入 UB |
| 非末轮屏障后 | 新侧 keys/ids 进入下一轮 | hist/ranks/offsets 可被下一轮重新生成 |
| 末轮完成 | 输出字段；原输入的最后一次回读结束 | 无后继 digit，无第三次全核屏障 |

hist 在 Prefix 读完后逻辑上已无消费者，但当前独立 Tensor 仍分配到整个 Kernel 结束；参考包装不将其物理区别名给 offsets。类似地，两侧 GM keys/ids 和一套每核 UB 是不同层次，不能把每核只处理一块等同于 GM 只保留一块。

task17 的单元素尾 tile 只初始化 ranks[32769]，hist 的全部 256 桶仍被覆盖。新 digit 不依赖 Host 清零 hist，因为每个有效 task 在 Histogram 内完整写该桶向量；更改任务域时应重新检查这一覆盖条件。
