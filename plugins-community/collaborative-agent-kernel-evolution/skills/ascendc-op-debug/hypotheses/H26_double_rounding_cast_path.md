---
id: H26
title: 多阶段 Cast 路径双重舍入（Double-Rounding via FP16 Intermediate）
symptom: precision_bias
when: fp16_intermediate_cast_only
root_cause: double_rounding_cast_path
evidence: code
escalate_to: null
source: DynamicQuant 算子开发复盘（2026-04-10）
---

## triggers
- 精度误差 ±1，占比约 0.01%—0.1%，系统性而非随机
- 误差集中在边界附近（如 `.4999...` 被错误进位）
- 转换路径经过两步 Cast：**FP32 → FP16(CAST_ROUND) → INT8(CAST_ROUND)**
- 单步直接 FP32→INT8 路径不会触发此问题

## read_target
- `kernel/{op_name}.cpp` → 搜索连续两行 `Cast` 调用
  - grep: `Cast.*CAST_ROUND`
- 检查是否存在 FP16 中间 buffer：`quantFp16`、`tmpFp16`、`xHalf` 等
- 检查中间 Cast 的 RoundMode 是否为 `CAST_ROUND`

## code_pattern
```cpp
// ❌ 双重舍入：FP32 → FP16(CAST_ROUND) → INT8(CAST_ROUND)
// 示例：FP32 38.4999...
//   → FP16 CAST_ROUND → 38.5 （第一次舍入，向上偏移）
//   → INT8 CAST_ROUND → 39   （第二次舍入，错误，正确应为 38）
AscendC::Cast(quantFp16, quantFp32, AscendC::RoundMode::CAST_ROUND, tileLength);
AscendC::Cast(outInt8,   quantFp16, AscendC::RoundMode::CAST_ROUND, tileLength);
```

## fix_template
```cpp
// ✅ 单次舍入：先在 FP32 空间显式 Round，再用 CAST_NONE 精确转换
AscendC::Round(quantFp32, quantFp32, tileLength);                                // FP32 空间取整（单次舍入）
AscendC::Cast(quantFp16, quantFp32, AscendC::RoundMode::CAST_NONE, tileLength); // 精确，无额外舍入
AscendC::Cast(outInt8,   quantFp16, AscendC::RoundMode::CAST_NONE, tileLength); // 精确，无额外舍入
```

## verify_cmd
```bash
# 修复后误差应降至 0（或极少量 ±1，由 Div vs / 的末位差引起，atol=1 覆盖）
# 对比修复前后误差元素数：
python3 -c "
import torch
# 载入修复前后的输出，对比 mismatch 元素数
"
```

## note
- `AscendC::Round` 是设备端向量取整指令（等价于 `std::round`），见 `api-restrictions.md §1.1`
- 本假设专指 FP16 作为**中间步骤**的两段 Cast；若 FP16 是最终输出则不适用
- 910B Magic Number 取整技巧（`Adds(2^23)`）在此场景失效，不要使用（见 `api-precision.md`）
