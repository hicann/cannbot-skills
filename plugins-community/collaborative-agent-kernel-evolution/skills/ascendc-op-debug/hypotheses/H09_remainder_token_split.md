---
id: H09
title: 核间 token 余数分配逻辑错误
symptom: precision_bias
when: large_batch_only
root_cause: remainder_split_error
evidence: code
escalate_to: null
source: ascendc-debug.md#原则2
---

## triggers
- totalTokens 不整除 coreNum 时（如 9 tokens / 8 cores）部分 token 被重复处理或漏处理
- BS=8 正确，BS=9 或 BS=15 错误
- 部分核的输出区域重叠（写冲突）

## read_target
- `kernel/{op_name}.cpp` → 查 blockIdx 到 tokenStart/tokenEnd 的映射逻辑
  - grep: `tokenStart\|tokenEnd\|rowsPerCore\|blockPivot\|remainTokens`
- 检查：是否有 `blockPivot` 分界点区分"多处理 1 个 token 的核"和"正常核"

## code_pattern
```cpp
// ❌ 简单整除，余数丢失（9 tokens, 8 cores → 每核 1 个，最后 1 个 token 未处理）
uint32_t rowsPerCore = totalTokens / usedCoreNum;  // 9/8 = 1
tokenStart_ = blockIdx * rowsPerCore;               // 最后 1 个 token 没人处理
tokenEnd_   = tokenStart_ + rowsPerCore;

// ❌ 简单向上取整，最后几核越界（9 tokens, 8 cores → ceil=2 → 总处理 16 个，越界）
uint32_t rowsPerCore = (totalTokens + usedCoreNum - 1) / usedCoreNum;  // 2
// blockIdx=7 → tokenStart=14 → 越界
```

## fix_template
```cpp
// ✅ 余数分配：前 remainTokens 个核多处理 1 个 token
// Host 端计算 tiling：
uint32_t rowsPerCore   = totalTokens / usedCoreNum;
uint32_t remainTokens  = totalTokens % usedCoreNum;
uint32_t rowsPerCoreSp = rowsPerCore + 1;   // special cores（多 1 个）
uint32_t blockPivot    = remainTokens;       // 前 pivot 个核是 special

// Kernel 端映射：
if (blockIdx < blockPivot_) {
    // special core：多处理 1 个 token
    tokenStart_ = blockIdx * rowsPerCoreSp_;
    tokenEnd_   = tokenStart_ + rowsPerCoreSp_;
} else {
    // normal core
    tokenStart_ = blockPivot_ * rowsPerCoreSp_
                + (blockIdx - blockPivot_) * rowsPerCore_;
    tokenEnd_   = tokenStart_ + rowsPerCore_;
}
```

## verify_cmd
- 覆盖测试：BS ∈ {1, 核数-1, 核数, 核数+1, 2×核数-1, 2×核数}
- 打印每个核的 tokenStart/tokenEnd，验证无重叠、无漏洞、总和等于 totalTokens

## notes
- 与 H08 配合：H08 处理 tokens < cores，本 H 处理 tokens > cores 但不整除
- `blockPivot = remainTokens`：当 remainTokens=0 时所有核都是 normal core
