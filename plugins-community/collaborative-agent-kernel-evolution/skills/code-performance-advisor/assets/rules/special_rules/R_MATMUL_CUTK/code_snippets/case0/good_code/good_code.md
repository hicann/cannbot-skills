# Good Code

```cpp
// 切K模板：同时切分 M、N、K 轴
tileNum = Div(m, baseM) * Div(n, baseN) * Div(k, baseK); // tileNum 大幅增加

if ASCEND_IS_AIC {
    // AI Core: 负责计算
    for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum) {
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            CopyInA1();
            CopyInB1();
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                CopyInA2();
                CopyInB2();
                Mmad();
            }
            // 单核内 K 轴累加后整体搬移到 workspace（按核数开辟）
            CopyOut(); // 搬移到 workspace，而非直接 atomic 写回 GM
        }
    }
}

if ASCEND_IS_AIV {
    // AI Vector: 负责累加
    // workspace 按核数开辟，保证确定性
    DataCopyPad<>(); // 第一份数据
    for (uint64_t i = 1; i < Div(k, baseK); ++i) {
        Add(); // 在 UB 上按序累加，保证确定性
    }
    DataCopyPad<>(); // 最终结果写回 GM
}
```