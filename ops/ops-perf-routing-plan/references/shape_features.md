# shape 特征与模板命名

## 常用特征

| 特征 | 表达式示例 | 含义 |
|---|---|---|
| 总元素数 | `numel` / `total_elems` | 规模 |
| 输入字节 | `input_bytes` | 搬运量 |
| 末维对齐 | `(shape[-1] * elem_bytes) % 32 == 0` | 是否 32B 对齐 |
| 单行字节 | `shape[-1] * elem_bytes` | 行宽 |
| UB 容纳 | `numel * elem_bytes * 2 <= ub_size` | 能否全载 |
| 工作集 | `gm_read_bytes + gm_write_bytes` | L2/HBM 分界 |
| 行/列比例 | `shape[0] > shape[1] * 4` | 高瘦 |
| 归约维 | `shape[-1]` | 归约规模 |
| dtype | `elem_bytes` / `is_fp16` / `is_fp32` | 精度路径 |
| 核占用 | `ceil(total_elems / 1024)` | 单核/多核 |
| 尾块 | `numel % (aiv_core_num * tile)` | 负载均衡 |

## 命名示例

- `large_n_ub_full`
- `small_m_single_core`
- `tail_unaligned_stream`
- `tall_narrow_row_batch`
- `wide_short_multi_core`
- `k_large_split_reduce`
- `l2_resident_pipeline`
- `hbm_stream_tile`
- `fp32_mixed_precision`
- `core_balanced_tail`

## 建议

1. 只用 host 侧可静态计算的量。
2. 不要依赖运行时数据值。
3. 优先使用 2~4 个强特征，不要堆砌弱条件。
4. 保证谓词至少命中当前最差 shape。
5. 允许单 shape 模板，但报告标记为“未形成有效分组”。
