# SortSelect DSL 能力映射

依据 DSL 公开接口与 [Sort example](example/sort/README.md) 说明能力及实现约束。公共核对项目见 [capability-contract](../../generic/capability-contract.md)。

| 标识 | 映射与边界 |
|---|---|
| `CAP_STABLE_COMPARE` | 整数保留全宽语义；浮点单调键，零等键、NaN 同键，稳定重排原索引 |
| `CAP_LOCAL_SORT` | SIMT 插入、SIMD 字节 Radix；组合 two-stage 批处理，受 UB 与计数位宽限制 |
| `CAP_MERGE` | MergeSort 的 Sort32/MrgSort 原语封装及归并路线尚未适配，无对应 DSL 模板；见 [模板对照](implementation-guide.md#算法模板与支持范围) |
| `CAP_RADIX_SORT` | 组合实现稳定 LSD；局部及多核块内均使用 SIMD permutation，多核在同一个 kernel 内完成分阶段前缀/散射 |
| `CAP_BLOCK_SYNC` | `cb.simt.syncthreads()`；SIMD VF 内重叠写读使用 `reg.vmem_bar("vst_vld")`，作用域不同 |
| `CAP_GLOBAL_SYNC` | `cb.global_sync_all()` 建立同一 launch 的全核阶段边界；block 数不超过当前流有效 AIV 核数，所有核以相同次数到达。跨阶段 GM 数据使用 SIMT L2 load/store，生产 VF 结束前 `cb.simt.threadfence()` 发布；当前实现不使用 lookback 协议 |
| `ALIGN_GRANULARITY` | 物理字节对齐按 `cb.Buffer` 编译布局核对；局部宽度补齐到 256 元素是模板容量规则，见 [容量预算](tiling.md#容量预算) |
| `LOCAL_MEMORY_BUDGET` | `cb.get_mem_size("ub")` 减SIMT 32768-byte 保留区，另计全部存活 buffer 与对齐 |
| `INDEX_COUNT_DOMAIN` | 全局地址/计数/前缀/输出索引及多核中间原轴索引 int64；局部循环 int32、局部索引和位置/秩编码 uint32、局部桶计数 uint16，Host 证明范围。TensorSpec、分配与写入 dtype 一致 |

## SIMD Radix 映射

[radix.py](example/sort/radix.py) 实现字节 permutation 和局部 LSD 原语，单核、two-stage 与多核模板分别完成构键及原值/索引恢复，文件对应见 [实现指南](implementation-guide.md)。

- `reg.vhistogram_frequency` 每次返回本块频数，多块需显式相加；不能假定目标寄存器自动累加。
- `reg.vhistogram_accumulate` 计算 inclusive 累计直方图，并接回前一块状态；减去频数得到 exclusive 桶起点。当前包内注释不准确。u8 的 `bin=0/1` 对应桶 0..127/128..255。
- 四轮 2-bit 稳定分组使用位移、比较、`vgather_reg` 与 `vstore_unalign_begin → vsqueeze_and_storeunalign_init/store → finalize`。每轮独立 VF，一个 cursor 连续写全部块，末尾统一 finalize；复制索引前后建立 VST/VLD 依赖。
- 最后按桶起点、块内排序位置和本块桶前缀计算全局 permutation，`vscatter` 写出后再 gather 键和原索引。填充字节置 255，但直方图与最终散射只计真实元素；稳定排序保证同值填充留在有效尾部之后。

本例通过下层接口组合排序能力，未使用原生 Sort/Sort32/MrgSort 高阶封装；`reg.vmerge` 是 mask 选择，不能代替有序序列归并。

## 多核前缀的组内交换

`sort_radix_more_core.py::prefix` 使用 32-lane 的 `cb.simt.shfl_up` 扫描，再以 `cb.simt.syncthreads()` 发布各 warp 总数并汇总；所有参与 lane 都须执行 shuffle。该约束属于当前 DSL 前缀实现，不是全部排序路线的组宽要求；SIMD 字节向量宽度由 `radix.py` 的 `VECTOR_BYTES` 单独确定。
