---
id: H05
title: einsum 转置语义错误（矩阵索引方向写反）
symptom: precision_bias
when: always
root_cause: index_semantics
evidence: code
escalate_to: null
source: ascendc-debug.md#案例5
---

## triggers
- 偏差大小随输入值线性增长（系统性倍数关系）
- matmul / weighted-sum 类操作结果与 golden 有规律性偏差
- 小 shape (N=2) 手工验证时发现结果是转置版本

## read_target
- `kernel/{op_name}.cpp` → 查 matmul / einsum 实现中的索引表达式
  - grep: `combVals\|\[j \* N\]\|\[h \* N\]\|einsum`
- `op_host/{op_name}_custom.cpp` 或调用方 → 查 golden 的 einsum 字符串
- 对照：einsum 中哪个下标是求和轴（j），哪个是输出轴（h）

## code_pattern
```cpp
// golden Python: torch.einsum('bsjh,bsjd->bshd', comb_weight, residual)
// 含义：j 是求和轴，h 映射到输出
// comb_weight 内存布局：[b,s,j,h] 行主序 → 索引为 combVals[j*N + h]

// ❌ 错误：j 和 h 的索引写反，等价于 C * R 而非 C^T * R
for (uint32_t h = 0; h < N; ++h) {
    for (uint32_t j = 0; j < N; ++j) {
        float cv = combVals[h * N + j];   // ← 应为 j*N+h，写成了 h*N+j
        Axpy(x2[h * dimTile_], res[j * dimTile_], cv, tD);
    }
}
```

## fix_template
```cpp
// ✅ 正确：j 是求和轴（外层），h 是输出轴（内层）
// comb_weight[j, h] 在行主序下 = combVals[j * N + h]
for (uint32_t h = 0; h < N; ++h) {
    for (uint32_t j = 0; j < N; ++j) {
        float cv = combVals[j * N + h];   // ← 正确：j*N+h
        Axpy(x2[h * dimTile_], res[j * dimTile_], cv, tD);
    }
}
// 建议：在注释中写明完整索引映射
// comb_weight[b,s,j,h] → combVals[j*N+h]（j=求和轴, h=输出轴）
```

## verify_cmd
- 用 N=2 的极小 shape 手工验证：手算矩阵乘法结果，与 kernel 输出逐元素对比
- 打印 `combVals[0..N*N-1]` 和 golden 的 `comb_weight[0,0,:,:]`，确认行列一致

## notes
- einsum 下标顺序定义了数学含义，内存布局（行主序）决定了 C++ 索引
- 规律：偏差随输入线性增长 ≈ 矩阵乘法方向错误的典型特征
- 调试技巧：先用 scalar 逐元素实现验证逻辑，再换成向量化
