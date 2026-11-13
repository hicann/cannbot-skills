# SortSelect DSL 能力映射

依据 DSL 公开接口与 [Sort example](example/sort/README.md) 说明能力及实现约束。公共核对项目见 [capability-contract](../../generic/capability-contract.md)。

| 标识 | 映射与边界 |
|---|---|
| `CAP_STABLE_COMPARE` | 整数保留全宽语义；浮点单调键，零等键、NaN 同键，稳定重排原索引 |
| `CAP_LOCAL_SORT` | SIMT 插入、SIMD 字节 Radix；组合 two-stage 批处理，受 UB 与计数位宽限制 |
| `CAP_MERGE` | MergeSort 的 Sort32/MrgSort 原语封装及归并路线尚未适配，无对应 DSL 模板；见 [模板对照](implementation-guide.md#算法模板与支持范围) |
| `CAP_RADIX_SORT` | 组合实现稳定 LSD；局部及多核块内均使用 SIMD permutation，多核在同一个 kernel 内完成分阶段前缀/散射 |
| `CAP_BLOCK_SYNC` | `cb.simt.syncthreads()`；SIMD VF 内重叠写读使用 `reg.vmem_bar("vst_vld")`，作用域不同 |
| `CAP_GLOBAL_SYNC` | `cb.global_sync_all()` 建立同一 launch 的全核阶段边界；block 数不超过当前流有效 AIV 核数，所有核以相同次数到达。跨阶段 GM 数据使用 SIMT L2 load/store，生产 VF 结束前 `cb.simt.threadfence()` 发布；lookback 使用已发布完整前缀或已完成 hist 的无等待回退；各 digit 独立状态，批首清零后汇合 |
| `ALIGN_GRANULARITY` | 使用 `cb.Buffer(mem_loc, dtype, cb.make_layout(shape))` 声明存储，物理字节对齐按编译布局核对；局部宽度补齐到 256 元素是模板容量规则，见 [容量预算](tiling.md#容量预算) |
| `LOCAL_MEMORY_BUDGET` | `cb.get_mem_size("ub")` 减SIMT 32768-byte 保留区，另计全部存活 buffer 与对齐 |
| `INDEX_COUNT_DOMAIN` | 全局地址和输出索引 int64；多核中间原轴索引/计数/前缀在 N<2^30 时用 int32，否则 int64；局部循环 int32、局部排列索引 uint32、局部桶计数 uint16，Host 证明范围。TensorSpec、分配与写入 dtype 一致 |

## SIMD Radix 映射

[radix.py](example/sort/radix.py) 实现字节 permutation 和局部 LSD 原语，单核、two-stage 与多核模板分别完成构键及原值/索引恢复，文件对应见 [实现指南](implementation-guide.md)。

- `reg.vhistogram_frequency` 每次返回本块频数，多块需显式相加；不能假定目标寄存器自动累加。
- `reg.vhistogram_accumulate` 计算 inclusive 累计直方图，并接回前一块状态；减去频数得到 exclusive 桶起点。u8 的 `bin=0/1` 对应桶 0..127/128..255。
- 四轮 2-bit 稳定分组使用位移、比较、`vgather_reg` 与 `vstore_unalign_begin → vsqueeze_and_storeunalign_init/store → finalize`。四轮位于同一个 SIMD VF，每轮一个 cursor 连续写全部块，末尾统一 finalize 并建立 VST/VLD 依赖。每块原位重排 byte_order。
- 最后按桶起点、块内排序位置和本块桶前缀计算全局 permutation，`vscatter` 写出后再 gather 键和原索引。填充字节置 255，但直方图与最终散射只计真实元素；稳定排序保证同值填充留在有效尾部之后。

共用 `sort_local_keys` 的字节提取与 key/index 重排也使用 SIMD：`extract_key_digit` 通过 uint32 视图、移位和 DeInterleave 提取字节，`update_keys_and_ids` 用元素索引 UB Gather，64 位键按双字 gather/interleave。连续 32 位单核/two-stage 的构键采用 DMA+SIMD；其他构键、最终恢复及窄类型单核特化仍含 SIMT。具体源码映射见 [toolkit Radix 移植](toolkit-radix-port.md)。

本例通过下层接口组合排序能力，未使用原生 Sort/Sort32/MrgSort 高阶封装；`reg.vmerge` 是 mask 选择，不能代替有序序列归并。

## 多核前缀与数据通道

统计以 SIMD 频数/累计直方图生成局部桶起点，通过 SIMT 原子求和得到全局桶起点。Scatter 在 UB 内稳定排列后，用 L2 接口读前驱缓存前缀；未发布时回退到已经完成的 hist，不等待前驱执行。两种计数宽度都通过 SIMT 原子，当前没有将 32 位分支改为 DMA 原子。完整状态协议见 [多核 Kernel](templates/radix-more-core/kernel-template.md#3-无等待-lookback)。

全部连续 dtype 的原值/键及后续轮索引通过 `cb.mem_copy` 搬入 UB，寄存器抽取 digit，SIMT 从 UB 收集本 tile 的 key/id；64 位存储通过字节视图 DMA。低层 strided 保留 SIMT L2 加载。这里记录本例的实际数据通道，不由该选择推导其他 dtype 的 DMA 能力或性能保证。
