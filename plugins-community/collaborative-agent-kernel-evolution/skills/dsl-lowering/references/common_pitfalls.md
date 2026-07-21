# Common AscendC Pitfalls and Solutions

## 🔴 Critical Pitfalls (Will Cause Incorrect Results)

### 1. Not Accumulating Across Tiles

**Problem:**
```cpp
// WRONG: Overwrites previous tile results
for (uint32_t tile = 0; tile < num_tiles; tile++) {
    float dotProduct = compute_dot_product();
    resultLocal.SetValue(idx, dotProduct);  // ❌ Loses previous tiles!
}
```

**Solution:**
```cpp
// CORRECT: Accumulate across tiles
for (uint32_t tile = 0; tile < num_tiles; tile++) {
    float dotProduct = compute_dot_product();
    float currentVal = resultLocal.GetValue(idx);
    resultLocal.SetValue(idx, currentVal + dotProduct);  // ✅ Accumulates
}
```

**Impact:** Results will only reflect the last tile, causing massive accuracy errors.

---

### 2. Using Only One Input Tensor

**Problem:**
```cpp
// WRONG: Only loads input, ignores weight
AscendC::LocalTensor<float> inputLocal = inputQueue.AllocTensor<float>();
AscendC::DataCopy(inputLocal, inputGm, size);

// Computes something with only input
for (uint32_t i = 0; i < size; i++) {
    output.SetValue(i, inputLocal.GetValue(i));  // ❌ Where's the weight?
}
```

**Solution:**
```cpp
// CORRECT: Load and use ALL input tensors
AscendC::LocalTensor<float> inputLocal = inputQueue.AllocTensor<float>();
AscendC::LocalTensor<float> weightLocal = weightQueue.AllocTensor<float>();

AscendC::DataCopy(inputLocal, inputGm, size);
AscendC::DataCopy(weightLocal, weightGm, size);

// Use both tensors in computation
AscendC::Mul(output, inputLocal, weightLocal, size);  // ✅ Uses both
```

**Impact:** Operator will produce wrong results that don't depend on all inputs.

---

### 3. Placeholder/Simplified Logic

**Problem:**
```cpp
// WRONG: Just initializing with zeros
AscendC::Duplicate(output, 0.0f, size);  // ❌ Not implementing actual logic

// WRONG: Just copying input
AscendC::DataCopy(output, input, size);  // ❌ Not doing the computation

// WRONG: Simplified approximation
float result = (x > 10.0f) ? x : 0.5f * x;  // ❌ Not the real formula
```

**Solution:**
```cpp
// CORRECT: Implement the actual algorithm
// For softplus: log(1 + exp(x))
AscendC::Exp(expBuf, input, size);
AscendC::Adds(expBuf, expBuf, 1.0f, size);
AscendC::Log(output, expBuf, size);  // ✅ Correct implementation
```

**Impact:** Operator will fail accuracy tests completely.

---

## 🟡 Common Pitfalls (Will Cause Compilation Errors)

### 4. ReduceSum Without Workspace

**Problem:**
```cpp
// WRONG: Missing workspace parameter
AscendC::ReduceSum(dst, src, count);  // ❌ Compilation error
```

**Solution:**
```cpp
// CORRECT: Provide workspace buffer
AscendC::LocalTensor<float> workspace = workBuf.Get<float>();
AscendC::ReduceSum(dst, src, workspace, count);  // ✅ Correct
```

**Error Message:**
```
error: no matching function for call to 'ReduceSum'
```

---

### 5. DataCopy Alignment Issues

**Problem:**
```cpp
// WRONG: Non-32-byte aligned transfer
AscendC::DataCopy(dst, src, 7);  // ❌ 7 * 4 = 28 bytes (not aligned)
```

**Solution:**
```cpp
// CORRECT: Use DataCopyPad for non-aligned transfers
AscendC::DataCopyPad(dst, src, 
                     {1, static_cast<uint16_t>(7 * sizeof(float)), 0, 0},
                     {false, 0, 0, 0});  // ✅ Handles non-aligned
```

**Error Message:**
```
error: data size must be multiple of 32 bytes
```

---

### 6. Buffer Size Mismatch

**Problem:**
```cpp
// WRONG: Allocating too small buffer
pipe.InitBuffer(tempBuf, 100 * sizeof(float));

// Later trying to use more
AscendC::LocalTensor<float> temp = tempBuf.Get<float>(200);  // ❌ Overflow!
```

**Solution:**
```cpp
// CORRECT: Allocate sufficient buffer
uint32_t maxSize = block_batch * block_experts;
pipe.InitBuffer(tempBuf, maxSize * sizeof(float));

AscendC::LocalTensor<float> temp = tempBuf.Get<float>(maxSize);  // ✅ Fits
```

**Impact:** Memory corruption, undefined behavior.

---

## 🟢 Performance Pitfalls (Will Cause Slowness)

### 7. Excessive GetValue/SetValue Calls

**Problem:**
```cpp
// WRONG: Scalar operations in loop
for (uint32_t i = 0; i < size; i++) {
    float val = input.GetValue(i);
    val = val * 2.0f;
    output.SetValue(i, val);  // ❌ Very slow!
}
```

**Solution:**
```cpp
// CORRECT: Use vector operations
AscendC::Muls(output, input, 2.0f, size);  // ✅ Much faster
```

**Impact:** 10-100x slower performance.

---

### 8. Not Reusing Buffers

**Problem:**
```cpp
// WRONG: Allocating new buffers repeatedly
for (uint32_t i = 0; i < iterations; i++) {
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;  // ❌ Wasteful
    pipe.InitBuffer(tempBuf, size);
}
```

**Solution:**
```cpp
// CORRECT: Allocate once, reuse
AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
pipe.InitBuffer(tempBuf, size);

for (uint32_t i = 0; i < iterations; i++) {
    AscendC::LocalTensor<float> temp = tempBuf.Get<float>();  // ✅ Reuse
}
```

**Impact:** Memory allocation overhead, slower execution.

---

---

## 🔴 Critical Pitfalls (Will Cause Incorrect Results)

### 9. Wrong Cast Mode for f32→bf16 Output

**Problem:**
```cpp
// WRONG: CAST_NONE for f32→bf16 silently produces garbage on dav_c220
AscendC::Cast(yBf16Local, yF32Local, AscendC::RoundMode::CAST_NONE, cnt);  // ❌
```

On dav_c220 (Ascend910b), `Cast(bfloat16_t_dst, float_src, CAST_NONE)` contains only
`ASCENDC_ASSERT(false)` with no `vconv` instruction. The destination buffer is never
written and retains uninitialized UB data. The kernel compiles and runs without errors,
but precision tests will all fail.

**Solution:**
```cpp
// CORRECT: use CAST_RINT for f32→bf16
AscendC::Cast(yBf16Local, yF32Local, AscendC::RoundMode::CAST_RINT, cnt);  // ✅
```

**Cast mode reference table for dav_c220:**

| Direction    | Correct Mode  | Notes                                              |
|--------------|---------------|----------------------------------------------------|
| `bf16 → f32` | `CAST_NONE`   | `vconv_bf162f32` ✓                                |
| `f32 → bf16` | **`CAST_RINT`**| `CAST_NONE` → `ASCENDC_ASSERT(false)` ❌          |
| `f16 → f32`  | `CAST_NONE`   | `vconv_f162f32` ✓                                 |
| `f32 → f16`  | `CAST_NONE`   | `vconv_f322f16` ✓                                 |

**Impact:** bfloat16_t output will be garbage values, failing all precision tests. The
kernel appears to succeed (no runtime error) but produces wrong results.

---

## 📋 Quick Reference: Common Fixes

| Error/Issue | Quick Fix |
|-------------|-----------|
| Results all zeros | Check if you're initializing but not computing |
| Results don't change with input | Check if all inputs are loaded and used |
| Compilation error: ReduceSum | Add workspace parameter |
| Compilation error: DataCopy alignment | Use DataCopyPad instead |
| Accuracy error: only last tile | Add accumulation: `current + new` |
| Very slow performance | Replace GetValue/SetValue loops with vector ops |
| Memory corruption | Check buffer sizes match usage |
| Wrong activation results | Use proper AscendC APIs, not approximations |
| bfloat16_t output garbage | Use `CAST_RINT` (not `CAST_NONE`) for f32→bf16 Cast |

---

## 🔍 Debugging Checklist

When results are incorrect:

1. **Check all inputs are used**
   - [ ] All input tensors are loaded
   - [ ] All input tensors appear in computation
   - [ ] No input is ignored

2. **Check accumulation logic**
   - [ ] Results are accumulated across tiles (not overwritten)
   - [ ] Accumulation uses `current + new` pattern
   - [ ] Initial value is correct (usually 0)

3. **Check tile loops**
   - [ ] All tiles are processed (loop over all tile indices)
   - [ ] Tile offsets are calculated correctly
   - [ ] Tile boundaries are respected

4. **Check activations**
   - [ ] Using proper AscendC APIs (Exp, Log, etc.)
   - [ ] Not using approximations or simplified formulas
   - [ ] Activation order is correct

5. **Check buffer management**
   - [ ] Buffers are large enough
   - [ ] Buffers are initialized before use
   - [ ] Tensors are freed after use

---

## 💡 Best Practices

### DO:
✅ Use vector operations (Mul, Add, etc.) instead of scalar loops  
✅ Accumulate results across tiles  
✅ Use ReduceSum for dot products  
✅ Use proper AscendC APIs for activations  
✅ Allocate buffers once and reuse  
✅ Check buffer sizes carefully  
✅ Free tensors after use  

### DON'T:
❌ Generate placeholder code  
❌ Use only one input when multiple are needed  
❌ Overwrite accumulated results  
❌ Use approximations instead of exact formulas  
❌ Forget workspace for ReduceSum  
❌ Use DataCopy for non-aligned transfers  
❌ Use GetValue/SetValue in tight loops  

---

## 📚 Related Documentation

- [Manual Matmul Example](./matmul/manual_matmul_example.md)
- [AscendC API Reference](../ascend_api/tl_asc_routing.md)
- [Error Correction Examples](../error_correction/error_correction_examples.md)
- [Multi-Dtype Guide](../multi_dtype/multi_dtype_guide.md)
- [Multi-Dtype Reference Kernel](../multi_dtype/add_custom_reference.cpp)
