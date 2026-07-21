---
id: H08
title: 空闲核未安全退出（tokens < coreNum）
symptom: crash
when: large_shape_only
root_cause: idle_core_no_exit
evidence: code
escalate_to: null
source: ascendc-debug.md#原则3
---

## triggers
- 小 batch（如 B*S=1 或 2）时崩溃，大 batch 正常
- 总 token 数 < 使用的核数时出现异常
- 错误可能是越界访问或非法内存操作

## read_target
- `kernel/{op_name}.cpp` → 查 `Init` 函数开头是否有空闲核检查
  - grep: `blockIdx.*usedCoreNum\|tokenStart.*tokenEnd\|return.*Init`
- `op_host/{op_name}_custom.cpp` → 查 `usedCoreNum` 的计算
  - grep: `usedCoreNum\|SetBlockDim\|min.*totalTokens`

## code_pattern
```cpp
// ❌ 未检查 blockIdx 是否超出有效核范围
__aicore__ inline void Init(...) {
    uint32_t tokenStart = blockIdx * rowsPerCore;
    uint32_t tokenEnd   = tokenStart + rowsPerCore;
    // 当 totalTokens=2, coreNum=8 时：
    // blockIdx=2 → tokenStart=2 → 越界访问 GM
}
```

## fix_template
```cpp
// ✅ Init 开头检查，超出范围直接标记为无任务
__aicore__ inline void Init(..., uint32_t usedCoreNum, ...) {
    if (GetBlockIdx() >= usedCoreNum) {
        tokenStart_ = 0;
        tokenEnd_   = 0;
        return;   // 提前退出，不分配任何资源
    }
    // ... 正常初始化 ...
}

// ✅ Process 函数同样保护
__aicore__ inline void Process() {
    if (tokenStart_ >= tokenEnd_) return;
    // ... 正常处理 ...
}

// ✅ Host 端：usedCoreNum = min(totalTokens, availableCores)
uint32_t usedCoreNum = std::min((uint32_t)totalTokens, GetCoreNum());
context->SetBlockDim(usedCoreNum);
```

## verify_cmd
- 测试 B*S=1（单 token）、B*S=核数-1（少一核）、B*S=核数（恰好整除）三个边界
- 确认各 blockIdx 的 tokenStart/tokenEnd 值正确

## notes
- 与 H09（余数分配）配合使用：H08 处理 tokens < cores，H09 处理 tokens % cores ≠ 0
