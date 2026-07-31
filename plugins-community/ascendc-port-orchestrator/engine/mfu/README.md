# MFU 求解器 (算子理论最大 MFU 模型) — v0

数学口径见 [`../docs/design/OPERATOR_MFU_DEFINITION.md`](../docs/design/OPERATOR_MFU_DEFINITION.md)、硬件常数见 [`../docs/design/L0_HARDWARE_CONSTANTS.md`](../docs/design/L0_HARDWARE_CONSTANTS.md)。

## 这是什么
给定算子（FLOPs/访存/dtype/可选通信）+ 硬件目标，算出**理论最大 MFU**（roofline 上界 ∩ 完全通算掩盖）+ 瓶颈归因（compute/mem/comm-bound）。

```
理论最大 MFU = T_compute / max(T_compute, T_mem, T_comm) × η_util
  T_compute = useful_FLOPs / cube_peak(dtype)
  T_mem     = HBM_bytes / HBM_BW
  T_comm    = comm_bytes / interconnect_BW
  η_util    默认 1.0(纯理论天花板)，待 msprof 真机标定下调
```

## 文件
- `mfu_model.py` — L0 硬件模型(910C/950PR/950DT) + L1 原语(matmul fwd/bwd) + L3 roofline 求解器
- `demo.py` — matmul 在三目标上的 MFU_max 示例

## 跑
```bash
cd mfu && python3 demo.py
```

## v0 结果验证
- 大方阵 matmul(4096³): MFU_max=1.0 compute-bound ✓
- GEMV(M=1): MFU_max≈0.003 memory-bound ✓
- **cube peak 交叉验证**: 950PR FP8 推导 973 TFLOPS ≈ 官方 1 PFLOPS ✓（"1拍 2·M·K·N FLOPs"口径成立）

## 进度
- [x] 通算掩盖：分布式 DP 梯度 AllReduce 模型 + SuperPod384 互联(`with_data_parallel`)
- [x] FlashAttention(cube+vector 混合) 原语
- [x] L2 调度演算：多级 roofline，tiling 用 **matmul I/O 下界(Hong-Kung)** = 最优 tiling 搬运量
  - 这是"理论最优调度"对应的搬运量(理论最大 MFU), 而非某具体次优 tiling
  - 修正: 早期用 L0C-tile 计数的 2 级模型误判"910C 大矩阵 L2-bound 0.66" —— 那是次优 tiling 的人为产物;
    I/O 下界正确给出大矩阵 **compute-bound MFU=1.0**(理论最优), 实测与理论的差距归入 η_util(待 msprof)
  - 验证: 大/中矩阵 compute-bound=1.0; 瘦长(256×256×8192)/小(128³) 正确 memory-bound
- [x] 非 matmul 原语: softmax/rmsnorm/elementwise(向量引擎, 深度 memory-bound, MFU 0.007~0.18)
  - 洞察: vec_peak 越高 MFU 越低(同搬运量) → 必须算子融合减 HBM 往返; 950DT(4TB/s) 比 950PR 优 2.5×
- [x] 单算子 → 训练 e2e MFU 聚合(`transformer_layer_e2e`, OPERATOR_MFU_DEFINITION §4 契约)
  - Σuseful_FLOPs/(cube_peak×Σ各算子最优时间); 向量算子 FLOPs 小但耗时拖低 e2e; 给时间占比找瓶颈
- [ ] η_util msprof 真机标定 + 预测 vs 实测回归(需 NPU, 已申请)
- [ ] 验证闭环：FA + a5_ops 优化 agent，验证理论能否指导优化（若无用则止损）
