---
id: H18
title: 向量 API 最小操作数违规（Cast/DataCopy/Duplicate 元素数低于硬件下限）
symptom: hang
when: small_shape_only
root_cause: vector_api_min_count
evidence: code
escalate_to: null
source: FlashAttentionV2 FA_V2_REPORT2.md
---

## triggers
- kernel 在小 shape 下（d=16, d=32）hang，大 shape 正常
- kernel hang 无报错，直接超时
- 或：精度偏差仅在 d < 64 时出现（Cast 返回未定义值）

## read_target
- `op_kernel/{op_name}.cpp` → 检查 Cast/DataCopy/Duplicate 的 count 参数
  - grep: `Cast\|DataCopy\|Duplicate`
- 计算实际 count：是否可能 < 64（Cast）或 < 16 fp16 / 8 fp32（DataCopy）

## code_pattern
```cpp
// ❌ 危险：d=16 时 Cast count = 16 < 64（910B 硬件下限）
uint32_t d = 16;
Cast(dstF, srcH, RoundMode::CAST_NONE, d);   // count=16 → hang 或结果错误

// ❌ 危险：DataCopy count = 8 fp16 = 16 bytes < 32 bytes
DataCopy(dst, src, 8);    // 16 bytes < 32 bytes → 非对齐访问崩溃

// ❌ 危险：Duplicate count 不等于 buffer 元素总数
pipe.InitBuffer(buf, 256 * sizeof(half));    // 128 个 fp16
Duplicate(buf.Get<half>(), val, 64);         // count=64 ≠ 128 → hang
```

## fix_template
```cpp
// ✅ 强制满足硬件下限约束

// Cast 最少 64 元素（half ↔ float）：
constexpr uint32_t MIN_CAST_COUNT = 64;
uint32_t dCast = std::max(dAligned, MIN_CAST_COUNT);
// 注意：buffer 也要按 dCast 分配，保证空间足够

// DataCopy 最少 32 字节（= 16 fp16 = 8 fp32）：
constexpr uint32_t MIN_DATACOPY_FP16 = 16;
constexpr uint32_t MIN_DATACOPY_FP32 = 8;
uint32_t copyCount = std::max(actualCount, MIN_DATACOPY_FP16);

// Duplicate：count 必须 == buffer 总元素数（不能部分填充）：
uint32_t bufElems = bufSizeBytes / sizeof(half);
Duplicate(buf.Get<half>(), val, bufElems);   // 必须等于 bufElems
```

## 硬件约束速查表（Ascend 910B）

| API | 类型 | 最小单位 |
|-----|------|---------|
| Cast | half ↔ float | 64 元素 |
| DataCopy | fp16 | 16 元素（32 字节） |
| DataCopy | fp32 | 8 元素（32 字节） |
| Duplicate | any | == buffer 总元素数（不能部分填充） |
| ReduceMax/ReduceSum | float workspace | 必须 >= src buffer 大小 |

## verify_cmd
```bash
# 静态检查：找出所有 Cast 调用，提取 count 参数
grep -n "Cast(" op_kernel/*.cpp
# 人工检查：count 表达式是否可能在小 shape 下 < 64

# 运行时验证：用 d=16, d=32, d=64 分别测试
# d=16 hang → d=64 正常 → 确认 MIN_CAST_COUNT 违规
```

## notes
- 违规不报任何错误，直接 hang（硬件超时）或静默返回错误值
- 即使实际计算只需要 d 个元素，buffer 和操作数也必须按 ≥ 下限分配
- 超出 d 的多余元素会被计算但结果被忽略（只使用前 d 个元素的结果）
- 此约束是 910B 硬件 SIMD 宽度决定的，不同芯片代际可能不同
