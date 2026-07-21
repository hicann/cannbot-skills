# Base Code: 运行时计算每个输出点的索引

来源：lingxi-code (adaptive_avg_pool3d)

```cpp
class KernelAdaptiveAvgPool3d {
    __aicore__ inline void Process()
    {
        uint32_t start_elem = GetBlockIdx() * elems_per_core;
        uint32_t end_elem = start_elem + elems_per_core;
        if (end_elem > total_output_elems) {
            end_elem = total_output_elems;
        }

        // 问题：每个输出点都要重新计算索引
        for (uint32_t out_idx = start_elem; out_idx < end_elem; out_idx++) {
            // 运行时计算输出坐标
            uint32_t d_out = out_idx / (H_out * W_out);
            uint32_t tmp = out_idx % (H_out * W_out);
            uint32_t h_out = tmp / W_out;
            uint32_t w_out = tmp % W_out;

            // 运行时计算输入索引范围（Adaptive Pooling 的核心逻辑）
            uint32_t d_start = d_out * D_in / D_out;
            uint32_t d_end = (d_out + 1) * D_in / D_out;
            if ((d_out + 1) * D_in % D_out != 0) {
                d_end += 1;
            }

            uint32_t h_start = h_out * H_in / H_out;
            uint32_t h_end = (h_out + 1) * H_in / H_out;
            if ((h_out + 1) * H_in % H_out != 0) {
                h_end += 1;
            }

            uint32_t w_start = w_out * W_in / W_out;
            uint32_t w_end = (w_out + 1) * W_in / W_out;
            if ((w_out + 1) * W_in % W_out != 0) {
                w_end += 1;
            }

            // 计算当前窗口大小（用于平均）
            uint32_t window_size = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
            float avg_scale = 1.0f / static_cast<float>(window_size);

            // 使用计算出的索引进行 Pooling
            ComputePooling(d_start, d_end, h_start, h_end, w_start, w_end, avg_scale);
        }
    }

    __aicore__ inline void ComputePooling(
        uint32_t d_start, uint32_t d_end,
        uint32_t h_start, uint32_t h_end,
        uint32_t w_start, uint32_t w_end,
        float avg_scale)
    {
        // 累加窗口内所有输入值
        for (uint32_t d = d_start; d < d_end; d++) {
            for (uint32_t h = h_start; h < h_end; h++) {
                for (uint32_t w = w_start; w < w_end; w++) {
                    // 读取输入数据并累加
                    // ...
                }
            }
        }

        // 计算平均值
        // ...
    }
};
```

**问题**：

1. **重复计算开销大**
   - 每个输出点都要计算 6 个索引（d_start/end, h_start/end, w_start/end）
   - 涉及除法、取模、条件判断等昂贵操作
   - 对于大输出 shape（如 [1, 512, 16, 16, 16]），计算开销显著

2. **Scalar Unit 压力大**
   - 索引计算主要在 Scalar Unit 执行
   - Scalar Unit 无法向量化，成为瓶颈
   - 阻碍 Vector Unit 和 Memory Unit 的流水线并行

3. **编译器优化受限**
   - 循环内的复杂控制流难以优化
   - 分支预测失败率高（条件判断 `% D_out != 0`）
   - 无法充分利用指令级并行

4. **缓存局部性差**
   - 索引计算和数据访问交织
   - 频繁在计算和访存之间切换
   - Cache Miss 率高

**典型问题场景**：
- AdaptiveAvgPool / AdaptiveMaxPool 算子
- 输出 shape 较大（如 [1, 512, 16, 16, 16]）
- 输入输出 shape 不是整数倍关系
- 需要精确计算每个窗口的起止位置
