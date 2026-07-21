---
id: H13
title: 内存对齐违规（向量化操作数据不对齐）
symptom: precision_bias
when: large_shape_only
root_cause: alignment_violation
evidence: code
escalate_to: mssanitizer
source: ascendc-debug.md#常见向量化问题
---

## triggers
- 部分元素正确，部分元素（尾部）错误或为零
- 不整除 tile 大小的边界（如 dataSize 不是 16 的倍数时 half 计算出错）
- mask 相关的向量 API 结果在尾部异常

## read_target
- `kernel/{op_name}.cpp` → 查向量化 API 的 dataSize 参数
  - grep: `dataSize\|tileLength\|ALIGN\|align`
- 计算：`dataSize × sizeof(dtype)` 是否满足 32B 对齐
  - float: 需要 8 元素倍数（8 × 4B = 32B）
  - half:  需要 16 元素倍数（16 × 2B = 32B）
- 检查：lastTile 的大小是否也满足对齐要求（向上 pad 或特殊处理）

## code_pattern
```cpp
// ❌ 直接用原始数据长度，不保证对齐
uint32_t dataSize = totalLen % tileSize;  // lastTile 可能不对齐
Mul(outputLocal, inputLocal, weightLocal, dataSize);  // 未对齐的向量操作

// ❌ Cast 后未重新检查对齐（half→float 后元素数减半，float 需要 8 对齐）
uint32_t halfLen = 32;  // 满足 half 对齐（16 的倍数）
Cast(floatBuf, halfBuf, CAST_NONE, halfLen);
// floatBuf 中现在有 32 个 float，但 float 要求 8 对齐 → 32 满足
// 若 halfLen=24（满足 half），floatBuf 有 24 float → 满足 float 的 8 对齐 ✓
// 若 halfLen=20（满足 half），floatBuf 有 20 float → 不满足 float 的 8 对齐 ❌
```

## fix_template
```cpp
// ✅ 计算实际处理长度时向上 pad 到对齐要求
constexpr uint32_t ALIGN_FLOAT = 8;
constexpr uint32_t ALIGN_HALF  = 16;

uint32_t rawLen    = lastTileLen;
uint32_t alignedLen = (rawLen + ALIGN_FLOAT - 1) / ALIGN_FLOAT * ALIGN_FLOAT;
// 用 alignedLen 做向量计算，多出的元素无害（buffer 已预留足够空间）
Mul(outputLocal, inputLocal, weightLocal, alignedLen);

// ✅ 写出时只写实际有效数据（用 DataCopyPad 的 tail padding 机制）
DataCopyPad(outputGm[offset], outputLocal,
            {1, (uint32_t)(rawLen * sizeof(float)), 0, 0});
```

## verify_cmd
- 测试 dataSize 不对齐的 shape（如 D=2560+4，导致 lastTile 不对齐）
- 用 scalar 替代版本（见调试策略）验证尾部元素正确性

## notes
- 对齐要求：half=16元素(32B), float=8元素(32B), bf16=16元素(32B)
- DataCopyPad 的第 2 个结构体参数可指定实际字节数，避免写入多余元素到 GM
- 升级路径：代码审查不确定时，用 msSanitizer memcheck 检测对齐违规
