# avg_pool3_d 算子设计文档

## 算子概述

`avg_pool3_d` 是 3D 平均池化算子，支持 4D/5D 输入张量 (NCDHW/NDHWC)。

## 实现变体

本算子提供 5 种 AscendC kernel 实现，根据输入特征自动选择：

| 实现 | 触发条件 | 说明 |
|------|---------|------|
| `generic` | 默认 | 通用实现，逐点计算池化窗口平均值 |
| `reduce_d` | kH=1, kW=1, sH=1, sW=1 | 仅对 D 维度做 reduce，优化一维情况 |
| `split_c` | tilelang_mode=split_c | 沿 C 维度切分，利用向量化并行 |
| `split_w` | tilelang_mode=split_w | 沿 W 维度切分，优化宽方向大 kernel |
| `multi_w` | tilelang_mode=multi_w | 多窗口并行，提升 W 方向吞吐 |

## 目录结构

```
csrc/ops/avg_pool3_d/
├── CMakeLists.txt
├── design.md
├── op_host/
│   └── avg_pool3_d.cpp          # Host 侧封装：参数处理、tiling、kernel 分发
└── op_kernel/
    ├── avg_pool3_d_generic.cpp
    ├── avg_pool3_d_generic_kernel.h
    ├── avg_pool3_d_reduce_d.cpp
    ├── avg_pool3_d_reduce_d_kernel.h
    ├── avg_pool3_d_split_c.cpp
    ├── avg_pool3_d_split_c_kernel.h
    ├── avg_pool3_d_split_w.cpp
    ├── avg_pool3_d_split_w_kernel.h
    ├── avg_pool3_d_multi_w.cpp
    ├── avg_pool3_d_multi_w_kernel.h
    ├── avg_pool3_d_tiling.h
    └── kernel_common.h
```

## Host 侧逻辑

1. **输出尺寸计算**：参考 PyTorch `avg_pool3d` 公式，支持 `ceil_mode`
2. **数据格式转换**：NCDHW -> NDHWC flat (n*d*h*w, c)
3. **Tiling 构建**：填充 `AvgPool3DKernelTiling` 结构，包含所有几何参数和调度参数
4. **Kernel 分发**：根据 `kH/kW/sH/sW/pH/pW` 选择最优实现变体
5. **结果回写**：NDHWC -> NCDHW，4D 输入去除 channel 维度

## 注册信息

- `ops.h`：`at::Tensor avg_pool3d(...)`
- `register.cpp`：`torch.ops.npu.avg_pool3d`
- `csrc/CMakeLists.txt`：已添加 host/kernel 源文件，移除旧 ACLNN 封装
