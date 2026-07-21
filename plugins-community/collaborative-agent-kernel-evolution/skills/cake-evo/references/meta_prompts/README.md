# Meta Prompts for Operator Generation

本目录包含用于指导CAKE2算子生成的优化策略metaprompts。所有策略均从生产级算子的专家实现中提取，覆盖ops-nn和ops-transformer中的算子。

## 目录结构

```
meta_prompts/
├── strategy_index.md          ← 主索引表（注入为 {meta_prompt}）
├── strategies/
│   ├── dtype_01_mixed_precision.md
│   ├── dtype_02_template_kernel.md
│   ├── dtype_03_tilingkey_dispatch.md
│   ├── dtype_04_fp8_int4_conversion.md
│   ├── dtype_05_bf16_special_handling.md
│   ├── perf_01_double_buffer.md
│   ├── perf_02_adaptive_tiling.md
│   ├── perf_03_small_d_optimization.md
│   ├── perf_04_load_balance.md
│   ├── perf_05_pipeline_sync.md
│   ├── perf_06_multi_algo_selection.md
│   ├── perf_07_data_alignment.md
│   ├── perf_08_ub_memory_mgmt.md
│   ├── perf_09_deterministic_output.md
│   ├── perf_10_vectorized_copy.md
│   ├── perf_11_tail_block_handling.md
│   ├── perf_12_broadcast_mask.md
│   ├── perf_13_special_algorithms.md
│   ├── acc_01_fp32_intermediate.md
│   ├── acc_02_welford_algorithm.md
│   ├── acc_03_rounding_mode.md
│   ├── acc_04_pipeline_barrier.md
│   ├── acc_05_softmax_stability.md
│   ├── acc_06_high_precision_rsqrt.md
│   ├── acc_07_index_boundary.md
│   └── acc_08_quant_precision.md
└── README.md
```

## 使用方法

### Agent工作流（推荐）

在dsl_lowering步骤中：

1. 读取 `references/meta_prompts/strategy_index.md`
2. 选择所有适用于当前算子的策略ID
3. 读取每个引用的 `strategies/XXX.md` 详情文件
4. 在 `tiling_pass`、`init_pass`、`process_pass`、`process_nonaligned_pass` 中应用选定的模式

### 按算子类型快速参考

| 算子类型 | 推荐策略 |
|---------|---------|
| LayerNorm / RMSNorm | D1, D2, P1, P2, P3, A1, A2, A6 |
| 量化算子 | D1, D4, D3, P2, P4, A3, A8 |
| 逐元素操作 | D1, D2, P1, P2, A1, A4 |
| Softmax / 注意力 | D1, P1, P5, P16, A3, A4, A5 |
| Pooling / Gather | D1, P1, P2, P11, P14, A7 |
| 优化器算子 | D1, P1, P2, P9, A1 |
| 广播/掩码算子 | D1, P1, P12, A5 |
| 索引/散射算子 | D1, P1, P7, A7 |
| 特殊/复杂算子 | P13, D2, D3 |

## 策略概览

### 多数据类型支持 (D1-D5)

| ID | 策略 | 适用场景 |
|----|------|---------|
| D1 | 混合精度架构 | 任何FP16/BF16输入的算子 |
| D2 | 模板化内核类型分发 | 编译时多类型内核 |
| D3 | TilingKey驱动类型分发 | Host端多类型选择 |
| D4 | FP8/INT4量化转换 | 量化输出算子 |
| D5 | BF16/多平台特殊处理 | BF16输出或多硬件平台 |

### 性能优化 (P1-P16)

| ID | 策略 | 适用场景 |
|----|------|---------|
| P1 | 双缓冲机制 | 任何内存密集型内核 |
| P2 | 自适应分块策略 | 可变行/列维度 |
| P3 | 小D多行合并优化 | Hidden size ≤ 640 |
| P4 | 多核负载均衡 | 数据分布不均匀 |
| P5 | 流水线同步控制 | 所有双缓冲内核 |
| P6 | 多算法自适应选择 | 规范化、归约算子 |
| P7 | 32B对齐与DataCopyPad | 非对齐输入 |
| P8 | UB内存分区管理 | 多UB张量内核 |
| P9 | 确定性输出 | 训练算子 |
| P10 | 向量化数据拷贝 | CopyIn/CopyOut优化 |
| P11 | 尾块处理 | Pooling/Gather |
| P12 | 广播与掩码操作 | 带Mask输入、广播维度 |
| P13 | 特殊算法与高阶API | 复杂控制流、不规则访问 |

### 精度优化 (A1-A8)

| ID | 策略 | 适用场景 |
|----|------|---------|
| A1 | FP32中间计算 | BF16/FP16精度要求 |
| A2 | Welford数值稳定算法 | LayerNorm, BatchNorm |
| A3 | 舍入模式控制 | 任何Cast到低精度 |
| A4 | SetFlag/WaitFlag事件同步 | 跨管道数据依赖 |
| A5 | 数值安全与特殊值处理 | Softmax, NaN/Inf风险 |
| A6 | 高精度rsqrt (Newton-Raphson) | 需要rsqrt的规范化算子 |
| A7 | 索引与边界安全处理 | Gather/Scatter, 用户控制索引 |
| A8 | 量化专用精度处理 | 量化输出、自定义浮点格式 |

## 更新维护

使用 `build_strategy_index.py` 从IdeaPool重新生成：

```bash
python3 build_strategy_index.py
```

该脚本会：
1. 读取所有 `IdeaPool/*/expert_ideas.json`
2. 按策略主题分组（关键字匹配，跨类别）
3. 在每个组内检测变体（Jaccard相似度）
4. 生成26个策略详情文件和主索引表

## 质量保证

- **来源**: ops-nn 中生产级 Vector 算子的专家实现
- **覆盖率**: Vector 路径策略全覆盖
- **未匹配**: 0条策略（全部分类成功）

---

**版本**: v5.1 (Vector only)
**最后更新**: 2026-03-31
**来源**: ops-nn和ops-transformer中生产级算子的专家实现
