# Base Code: 固定核数简单划分

来源：lingxi-code (adaptive_avg_pool3d)

```cpp
// 固定 16 核简单划分
const uint32_t BLOCK_DIM = 16;
uint32_t elems_per_core = (total_output_elems + BLOCK_DIM - 1) / BLOCK_DIM;
context->SetBlockDim(BLOCK_DIM);

// Kernel 端简单处理
uint32_t start_elem = block_idx * elems_per_core;
uint32_t end_elem = (block_idx + 1) * elems_per_core;
if (end_elem > total_output_elems) {
    end_elem = total_output_elems;
}
```

**问题**：
1. 固定使用 16 核，无法根据实际数据量动态调整
2. 简单的向上取整划分，最后一个核心可能负载很轻或空闲
3. 当 `total_output_elems < 16` 时，部分核心完全空闲，浪费资源
4. 当 `total_output_elems % 16 != 0` 时，负载不均衡明显
