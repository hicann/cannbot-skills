# DSL 保序键与局部 Radix 原语

本页维护两阶段、单核 Radix 和多核块内排列共用的算法。逻辑 permutation 表示“排序位置 → 原位置”，与“原位置 → 排序名次”的 inverse 相反；多核物理存储另在高 16 位携带桶号，须先解码。

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

降序键统一为 `mask−升序键`，随后仍做稳定的键升序。SIMT 键和单核/two-stage 中间键载体 S 为 4/8 字节，多核 SIMD 使用原值对应的寄存器宽度且不将比较键写入 GM。LSD 轮数取原 dtype 的有效字节数 D；int8 使用 uint32 载体也只有一轮。原值的物理保存方式由 caller 决定：32/64 位单核和 two-stage 最终回读 GM；多核逐轮成对搬移原值/索引，末轮直接输出；两字节单核保存原始 UB 值；一字节整数单核对双射键反变换。不能把浮点归一化键当作完整原值。

SIMT 中的无符号键与比较常量必须具有一致的有符号性。浮点构键使用 `magnitude == result_type(0)`、`magnitude > result_type(infinity)`、`bits >= result_type(sign)`；不能依赖 Python 整数字面量在 VF 内自动推导为无符号类型。键的位宽和比较语义保持一致，地址与输出索引仍使用 int64。

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

### 与源码 VF 和缓冲的对应

`local_sort` 的 `pack_digits` 默认 False，返回纯原位置；多核显式开启时返回 `originalLocalPosition + (digit<<16)`，利用 W≤65280 的边界携带桶号，其他 caller 不受影响。解码仅针对局部位置，不缩窄全局原轴索引。

`radix_permutation` 本身不读写 GM，也不搬运 caller 的 keys/ids。字节排列依次为一个 SIMD 统计作用域、一个含四轮 2-bit 分组的 SIMD 作用域、一个 SIMD 全块排列作用域；共用 LSD 和多核连续路径则将统计融合进字节提取的 SIMD VF；256 字节寄存器块与整个 length 项 tile 是不同层次。

| 阶段 | 读取 / 生成 | 关键约束 |
|---|---|---|
| 全块桶起点 | 从原 digit 累计频数和包含式直方图，生成 256 个 uint16 offsets | 只统计 length 项；offsets 在后续排列和 caller 中仍表示初始桶起点 |
| 每个 2-bit pass | 每个寄存器块 gather 原 digit，按当前两位对原 lane 索引 squeeze，原位写回 byte_order | 初次索引为 0..255，后续 pass 读取前轮 byte_order；不搬动原 digit |
| 全块 permutation | 按 byte_order 得到有序 digit/lane，计算目的名次，vscatter 写 permutation | 原位置为 chunkBase+lane，最终读写只覆盖 length 项 |

对某寄存器块的有序位置 p 和桶 d，目的名次为 `bucketStart[d]+earlierChunkFrequency[d]+p−localBucketStart[d]`。桶游标在寄存器中累加以覆盖下一块，不把 offsets UB 改成最终累加值。uint16 游标拆成高/低字节按桶 gather 后重新组合，不能把全块游标直接截成 uint8；向量内有效元素所属桶的 localBucketStart≤255，只有这个局部减项可以取低字节。

每个 2-bit pass 对整 tile 使用一个连续 cursor，按 chunk 顺序写各块的四个分组，末尾统一 finalize 并建立 VST/VLD 依赖，再开始下一轮。每块包含补齐后完整的 256 个 lane，四组总长度仍为 256，因此原位写回不会覆盖下一块尚未读取的索引。尾 digit=255 经过稳定分组留在真实最大桶元素之后，最后一个寄存器块才能安全按真实有效 lane 计数和散射。

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

上述 `sort_local_keys` 每轮先用 SIMD `extract_key_digit` 同时提取 digit、累计直方图并生成桶起点，再调用 `eight_bits_sort/local_sort` 生成字节 permutation，最后用 SIMD `update_keys_and_ids` 对 length 项执行 UB Gather，重排 keys/ids。字节提取按 256 项块读取已分配的 UB，尾块无效 lane 统一置 255；只需写到 align256(length)，不读取更后面的容量余量。32 位键通过移位及两层 DeInterleave 压成字节，64 位键先从 uint32 视图选高/低字。重排按 64 项一组，64 位键分别 gather 两个字，再 Interleave 写回；所有 gather 和 store 均受真实 length 的 mask 约束。偶数轮读 ping 写 pong，奇数轮反向；D 偶数最终在 ping，D 奇数在 pong。第二阶段以行号为键，ping_ids 直接携带原展平索引；公开入口 B≤256，只处理 D=1，结果在 pong_ids。低层 B>256 处理 D=2，结果在 ping_ids，不继承 Stage1 的原 dtype 字节数或起始侧。two-stage 的 ids_only=True 仅在末轮跳过键重排，保留索引重排。此键/索引双缓冲骨架适用于 32/64 位单核和 two-stage；窄类型单核直接组合排列，多核末轮直接写最终输出，不能统一套用这个最终 ping/pong 位置规则。

SIMT 调用方的 `radix_digit` 对 64 位键按两个 32 位字提取，始终在无符号域内计算：`low=uint32(key)`，`high=uint32(key // uint64(2^32))`。byte<4 时选择 low，否则选择 high；所选字先除以 `uint32(2^(8×(byte%4)))`，再对 `uint32(256)` 取模。不能将键转为 int64 后套用额外的负数商修正；`//` 已按向下取整实现，重复修正会破坏低字非零时的高字。逐字节检查需覆盖跨 32/63 位边界，以及高位为 1 且低字为零/非零的键。

## 4 发布与存储依赖

SIMD 排列中对同一 UB 数据先写后读时保留 `reg.vmem_bar("vst_vld")`；SIMT 同块线程依赖使用 `cb.simt.syncthreads()`，两者作用域不同。跨 VF 的生产消费依赖由实际 DSL 编译和调用顺序核对，不能把删除原依赖当成纯文档重组。

局部计数为 uint16，物理宽度按 256 对齐，因此最大合法 W=65280；不得把 65535 直接当作对齐宽度。多核的块内 permutation 也受其 tile 和局部计数域约束。

## 5 可复算例子与来源

digit=`[2,1,2,0]`，桶起点为 `{0:0,1:1,2:2}`，permutation=`[3,1,0,2]`；两个 digit=2 的元素按原位置 0、2 保持顺序。按该 permutation 同时 gather keys/ids，不能只重排键。多核开启桶号打包时对应 `[3,65537,131072,131074]`；对 65536 取模仍得到上述原位置，整除则得到 `[0,1,2,2]`。

实现依据：[radix.py](example/sort/radix.py) 的 `radix_permutation/sort_local_keys` 和 [sort_common.py](example/sort/sort_common.py) 的 `ordered_key/radix_digit`。以上是计算过程说明，API 的具体签名仍以实际 DSL 版本为准。

与 toolkit 高阶 Sort 的阶段映射、8-bit/2-bit 粒度及 SIMD 移植边界见 [toolkit Radix 映射](toolkit-radix-port.md)。

连续 float32/int32/uint32 单核与 two-stage 输入使用 `load_keys32`：DMA 写入 ping 的原 dtype 视图，再由 SIMD 原位构建保序键、生成原位置。它复用既有键缓冲；其他类型或跨步输入继续使用 caller 的 SIMT 构键。
