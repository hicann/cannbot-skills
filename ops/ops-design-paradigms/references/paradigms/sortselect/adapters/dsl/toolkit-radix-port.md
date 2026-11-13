# 从 toolkit Radix Sort 映射到 DSL

## 源码依据与当前状态

toolkit 的 `asc/impl/adv_api/detail/sort/sort/sort_impl.h` 中，带输出索引的
`SortImpl` 在 `SortType::RADIX_SORT` 分支调用 `SortU8ElementsWithRound`，再按
局部排列更新值和索引。该文件是内部实现，只用于阅读算法，不作为生成工程的直接 include。
入口为 `asc/include/adv_api/sort/sort.h`。

[radix.py](example/sort/radix.py) 按下面的阶段组织 DSL 组合原语；它不是 DSL
提供的原生高阶 Sort API。共用 `sort_local_keys` 的字节提取、字节排列及 key/index
重排均使用 SIMD；连续 32 位输入的构键采用 DMA+SIMD，其余构键、输出恢复和窄类型特化仍含 SIMT。阶段对应不代表指令、缓冲布局或性能已与 toolkit 等价。

| toolkit 源码阶段 | DSL 计算体 | 映射边界 |
|---|---|---|
| 连续输入搬入 / `TwiddleInData` | `load_keys32` | 连续 float32/int32/uint32 DMA 到 ping 视图，SIMD 保序键和原索引；其他类型与跨步输入仍走 SIMT |
| `ConvertRegToWithShift` | `extract_key_digit` / 多核 `load_digit_word` | SIMD 选高/低字、移位和 DeInterleave 压缩字节；多核从原值即时生成比较键 |
| `GetExclusiveSum` | `extract_key_digit` / 多核 `extract_value_digit` / `get_exclusive_sum` | 连续路径在 digit 提取的同一 SIMD VF 内统计桶起点；已有 digit 的 caller 使用独立统计函数 |
| `EightBitsSort` | `eight_bits_sort` | 一个 SIMD VF 内四轮稳定 2-bit 分组，原位更新 lane 排列；各轮 finalize 后建立 VST/VLD 依赖 |
| `LocalSort` | `local_sort` | SIMD 合成整 tile 名次并 scatter permutation |
| `SortU8ElementsWithRound` | `radix_permutation` | 组合上述三个排列阶段，输入已提取的 digit |
| `UpdateValueAndIndexByLocalIndex` | `update_keys_and_ids` | SIMD UB Gather，64 位值按双字 gather/interleave 处理 |
| 末轮只消费索引的特化 | `update_ids` | `ids_only=True` 时仅重排索引；中间轮仍重排键，不可用于需要最终有序键的 caller |
| `SortImpl` 的逐字节循环 | `sort_local_keys` | 按原 dtype 有效字节数执行稳定 LSD，key/id 同步 ping-pong |

## 各模板实际复用到哪一层

上表 `load_keys32` 的类型范围只描述该 helper。它供 32/64 位单核与 two-stage 的连续 32 位构键使用，不能据此推断多核的其他连续 dtype 仍从 GM 逐元素读取。

| 调用方 | 输入准备 | 共享局部排序 | 排列后的处理 |
|---|---|---|---|
| 32/64 位单核 `RadixOneCore` | 连续 32 位用 `load_keys32`；64 位及跨步构键用 SIMT | `sort_local_keys`：提取并统计 → `eight_bits_sort` → `local_sort` → key/id Gather | 按最终 ids 从原 GM 回读值 |
| 1/2 字节单核 `NarrowRadixOneCore` | SIMT 构造 digit；连续两字节原值先 DMA 到 UB | 一次或两次 `radix_permutation` | 一字节整数反变换；两字节组合排列后从原始 UB 值恢复 |
| `SmallAxisTwoStage` | 连续 32 位用 `load_keys32`，其余用 SIMT | Stage1 调 `sort_local_keys(ids_only=True)`；稳定行号分组时 Stage2 再调一次 | inverse 计秩或按稳定行号分组结果回读原 GM；不消费最终有序键 |
| `RadixMoreCore.histogram` | 每轮连续原值 DMA 到 UB；`prepare_digits` 统一准备 | 本模板 `extract_value_digit` 融合 256 项原值 digit 提取与桶统计；跨步路径用共用 `get_exclusive_sum` | 发布频数、原子累加桶起点 |
| `RadixMoreCore.scatter` | 同样调用 `prepare_digits`；复用 raw UB，后续轮另 DMA ids | 共用 `eight_bits_sort/local_sort`；`pack_digits=True` 在排列项高 16 位携带桶号 | lookback 确定全行位置，所有轮次搬移原值/id，末轮直接输出 |

因此多核已复用优化后的三个 SIMD 排列阶段，但不调用完整 `sort_local_keys`：多核每字节之后需要全行散射和跨核同步，不能在单个 tile 内先排完所有字节。多核现在逐轮搬移原值，使用 `load_digit_word/extract_value_digit` 每次从原值生成 256 项 digit 并融合直方图；归一化只发生在寄存器，不写入中间原值。单核/two-stage 仍用 `extract_key_digit` 读取已生成的键。连续 DMA 使用字节视图，低层跨步调用仍走 SIMT L2。具体地址、缓冲和同步见 [多核 Kernel](templates/radix-more-core/kernel-template.md) 与 [Buffer](templates/radix-more-core/dag-buffers.md)。

## 8-bit 与 2-bit 的关系

外层每轮取一个 **8-bit digit**，全 tile 有 256 个桶。
在每个 256-lane 字节向量内，依次按 `[1:0]`、`[3:2]`、`[5:4]`、`[7:6]`
做四次稳定分组，每次只有 4 组。随后按整 tile 桶起点、前面向量的同桶频数和
本向量桶内排名计算目标位置。四次分组不是四次全 tile 的 256 桶排序。

普通键的外层轮数为原 dtype 的 1/2/4/8 字节，而不是 uint32/uint64 载体的
固定字节数。two-stage 第一阶段对合批值排序；行号恢复分支第二阶段利用稳定性仅按行号分组，公开 B≤256 时处理一个字节；低层更宽批次处理两个字节。逆排名恢复分支不做第二次 Radix。

## 移植时保留的语义

- `permutation[排序位置] = 原位置`，重排 key 时必须一起重排 index。two-stage 末轮仅消费 index 时使用 ids_only 省去无消费者的最终 key 写回。
- `offsets` 保留初始独占桶起点，跨向量游标仅在寄存器中递增。
- 当前 DSL 不物化 toolkit 的有序 digit 副本和 `tmpIndexU16`：通过原 digit
  与 lane permutation gather，最后在寄存器中扩展索引。四轮分组原位更新 byte_order。
- DSL `vhistogram_frequency` 返回单块频数，显式相加；不可直接照搬 C++ 的
  输出参数调用方式。累计直方图与频数相减才能得到独占前缀。
- DSL 当前将物理宽度补齐到 256，尾 digit 为 255；直方图和最终散射只处理
  有效 length。调用方须满足缓冲容量及 uint16 计数限制。
- 浮点 ±0、NaN 的等键归一化与原值恢复沿用本范式契约，不能直接用原始
  位模式排序替换。原值不从归一化键还原。

## SIMD 接口与验证边界

DSL `reg.vgather` 的索引单位是元素；64 位键用 uint32 UB 视图，访问索引
`2*order` 和 `2*order+1`。移位始终在 uint32 无符号域执行，尾块读入物理
补齐区后用有效 mask 选择 digit=255，不将补齐区的内容计入直方图。

`load_keys32` 在输入 DMA 完成后原位构键，不增加 UB 缓冲；零与 NaN 只归一比较键，恢复仍读取原输入，保留原始位模式。

保持真实 length 的尾部访问边界与 ping-pong 奇偶合同。其他构键、输出恢复、
窄类型单核的原值 gather、以及多核的全局调度不属于此次共用局部 LSD 的
SIMD 替换；不能据此声称整个 Sort kernel 不含 SIMT。

验收应通过编译产物确认 `extract_key_digit` 和 `update_keys_and_ids` 为
SIMD VF，并验证重复键、尾块、跨向量同桶、64 位高低字边界、浮点特殊值及
两条 two-stage 恢复路线。算法对应表不替代设备正确性及性能测试。
