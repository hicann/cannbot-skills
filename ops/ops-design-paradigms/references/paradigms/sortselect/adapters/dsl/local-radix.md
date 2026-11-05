# DSL 保序键与局部 Radix 原语

本页维护两阶段、单核 Radix 和多核块内排列共用的算法。这里的 permutation 始终表示“排序位置 → 原位置”；它与“原位置 → 排序名次”的 inverse 相反。

## 1 保序键和有效字节

令输入有效位宽为 w，符号位 `s=1<<(w−1)`，`mask=2^w−1`，u 为输入位模式：

| 类型/值 | 升序键 |
|---|---|
| 无符号整数 | u |
| 有符号整数 | u XOR s |
| 正浮点非零 | u XOR s |
| 负浮点非零 | mask−u |
| 浮点 ±0 | 同一键 s，依靠稳定性保留输入顺序 |
| 浮点 NaN | 同一最大键 mask；只改比较键，不改原始 value |

降序键统一为 `mask−升序键`，随后仍做稳定的键升序。载体 S 为 4/8 字节，但 LSD 轮数取原 dtype 的有效字节数 D；int8 使用 uint32 载体也只有一轮。原值按最终原轴索引回读，以保留零符号和 NaN payload。

## 2 一字节的稳定 permutation

输入 length 个有效 digit，物理存储可向上补齐，局部桶为 256 个。算法分三步：

```text
1. 跨所有向量块累计 frequency[d]
   bucketStart[d] = sum(frequency[x], x<d)
2. 每个向量内依次对 bit[1:0]、[3:2]、[5:4]、[7:6] 稳定分组
   同组按原 lane 次序压紧，四轮后按完整 digit 排列
3. 按 digit 计算该向量的桶内排名，加此前向量同桶频数和 bucketStart
   permutation[目标排序位置] = 原位置
```

当前 SIMD 表达用 `reg.vhistogram_frequency` 统计本块频数，显式累加到总频数；`reg.vhistogram_accumulate` 接续累计直方图，减去 frequency 得到 exclusive 起点。bin=0/1 分别覆盖 0..127 / 128..255。

四轮 2-bit 分组由位移、比较、`vgather_reg` 和 squeeze store 组合。每轮用一个连续 cursor 写完整序列，开始一次、最后统一 finalize，不在每个小块重置 cursor。尾部可填最大 digit，但所有计数和最终散射仅包括真实 length。

## 3 多字节 LSD

```text
初始 keys=保序键，ids=原位置
for byte in 0..D−1:
  从当前 keys 提取 digit
  perm = stable_byte_permutation(digit)
  next_keys[p] = keys[perm[p]]
  next_ids[p] = ids[perm[p]]
  交换 ping/pong
```

偶数轮读 ping 写 pong，奇数轮反向；D 偶数最终在 ping，D 奇数在 pong。第二阶段紧凑键明确使用 D=2，不继承 Stage1 的原 dtype 字节数。

64 位键按两个 32 位字提取。DSL 定宽 int64 除法向零截断：先取 `low=uint32(key)`、`signed=int64(key)`，计算 `high=signed/2^32`；若 signed<0 且 low!=0，再减 1 后转 uint32，以恢复正确的无符号高字。byte≥4 时从 high 提取，避免有符号除法破坏高位 digit。

## 4 发布与存储依赖

SIMD 排列中对同一 UB 数据先写后读时保留 `reg.vmem_bar("vst_vld")`；SIMT 同块线程依赖使用 `cb.simt.syncthreads()`，两者作用域不同。跨 VF 的生产消费依赖由实际 DSL 编译和调用顺序核对，不能把删除原依赖当成纯文档重组。

局部计数为 uint16，物理宽度按 256 对齐，因此最大合法 W=65280；不得把 65535 直接当作对齐宽度。多核的块内 permutation 也受其 tile 和局部计数域约束。

## 5 可复算例子与来源

digit=`[2,1,2,0]`，桶起点为 `{0:0,1:1,2:2}`，permutation=`[3,1,0,2]`；两个 digit=2 的元素按原位置 0、2 保持顺序。按该 permutation 同时 gather keys/ids，不能只重排键。

实现依据：[radix.py](example/sort/radix.py) 的 `radix_permutation/sort_local_keys` 和 [sort_common.py](example/sort/sort_common.py) 的 `ordered_key/radix_digit`。以上是计算过程说明，API 的具体签名仍以实际 DSL 版本为准。
