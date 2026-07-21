# AscendC API 验证参考手册

**用途**: 为 code-performance-advisor skill 提供 AscendC 语法验证依据，确保生成的优化建议不混淆 AscendC 和标准 C++ 语法。

**版本**: 1.0
**日期**: 2026-02-24
**适用范围**: AscendC 算子内核代码（op_kernel/*.cpp）

---

## 1. AscendC vs C++ 关键区别

### 1.1 命名空间

```cpp
// ✅ CORRECT: AscendC APIs
AscendC::LocalTensor<float>
AscendC::SetVectorMask<T, Mode>(n)
AscendC::DataCopy(dst, src, count)

// ❌ WRONG: Incorrect namespace
ascendc::LocalTensor<float>  // 小写错误
Ascend::LocalTensor<float>   // 缺 'C'
::LocalTensor<float>         // 无命名空间
```

### 1.2 函数修饰符

```cpp
// ✅ CORRECT: AscendC kernel qualifiers
__aicore__ inline void Process() { }
__aicore__ inline void Compute() { }

// ❌ WRONG: Standard C++ keywords (不适用于 kernel)
__global__ void Process() { }  // CUDA语法，不是AscendC
__device__ void Compute() { }  // CUDA语法，不是AscendC
inline void Process() { }      // 缺少 __aicore__
```

### 1.3 数据类型

```cpp
// ✅ CORRECT: AscendC types
AscendC::LocalTensor<float>          // Local buffer
AscendC::GlobalTensor<half>          // Global memory
AscendC::TBuf<AscendC::TPosition::VECCALC>
AscendC::TQue<AscendC::TPosition::VECIN, 2>

// ❌ WRONG: Standard C++ containers (kernel中禁止)
std::vector<float>
std::array<int, 10>
float* data = new float[100];  // kernel中无 new/delete
```

### 1.4 内存管理

```cpp
// ✅ CORRECT: AscendC buffer management
AscendC::LocalTensor<float> tensor = queue.AllocTensor<float>();
queue.EnQue(tensor);
AscendC::LocalTensor<float> tensor = queue.DeQue<float>();
queue.FreeTensor(tensor);

// ❌ WRONG: Standard C++ memory (kernel中禁止)
float* ptr = malloc(1024);     // 禁止
float* ptr = new float[256];   // 禁止
delete[] ptr;                  // 禁止
```

---

## 2. 已验证的 AscendC API 列表

### 2.1 Mask 操作

| API | 用途 | 示例 |
|-----|------|------|
| `SetMaskCount()` | 进入 COUNTER 模式 | `AscendC::SetMaskCount();` |
| `SetVectorMask<T, Mode>(n)` | 设置向量掩码 | `AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(1024);` |
| `ResetMask()` | 还原掩码状态 | `AscendC::ResetMask();` |

**Mask Modes**:
- `AscendC::MaskMode::COUNTER` - 硬件计数器模式
- `AscendC::MaskMode::NORMAL` - 普通模式

### 2.2 Tensor 操作

| API | 用途 | 示例 |
|-----|------|------|
| `LocalTensor<T>` | Local buffer 类型 | `AscendC::LocalTensor<float> buf;` |
| `GlobalTensor<T>` | Global memory 类型 | `AscendC::GlobalTensor<half> gm;` |
| `SetGlobalBuffer(ptr, size)` | 绑定 Global Memory | `gm.SetGlobalBuffer((__gm__ float*)ptr, size);` |
| `Get<T>()` | 从 TBuf 获取 Tensor | `auto tensor = buffer.Get<float>();` |

### 2.3 Buffer & Queue

| API | 用途 | 示例 |
|-----|------|------|
| `TPipe` | 流水线管理 | `AscendC::TPipe pipe;` |
| `TQue<Pos, N>` | Queue 类型 | `AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;` |
| `TBuf<Pos>` | Buffer 类型 | `AscendC::TBuf<AscendC::TPosition::VECCALC> buf;` |
| `InitBuffer(que, slots, size)` | 初始化 Queue | `pipe.InitBuffer(inQue, 2, 4096);` |
| `InitBuffer(buf, size)` | 初始化 Buffer | `pipe.InitBuffer(calcBuf, 4096);` |
| `AllocTensor<T>()` | 分配 Tensor | `auto t = queue.AllocTensor<float>();` |
| `EnQue(tensor)` | 入队 | `queue.EnQue(tensor);` |
| `DeQue<T>()` | 出队 | `auto t = queue.DeQue<float>();` |
| `FreeTensor(tensor)` | 释放 Tensor | `queue.FreeTensor(tensor);` |

**TPosition Enums**:
- `AscendC::TPosition::VECIN` - Vector 输入
- `AscendC::TPosition::VECOUT` - Vector 输出
- `AscendC::TPosition::VECCALC` - Vector 计算缓冲
- `AscendC::TPosition::GM` - Global Memory

### 2.4 Vector 计算指令

| API | 用途 | 示例 |
|-----|------|------|
| `Abs(dst, src, n)` | 绝对值 | `AscendC::Abs(dst, src, 1024);` |
| `Add(dst, s1, s2, n)` | 向量加法 | `AscendC::Add(dst, src1, src2, 1024);` |
| `Sub(dst, s1, s2, n)` | 向量减法 | `AscendC::Sub(dst, src1, src2, 1024);` |
| `Mul(dst, s1, s2, n)` | 向量乘法 | `AscendC::Mul(dst, src1, src2, 1024);` |
| `Div(dst, s1, s2, n)` | 向量除法 | `AscendC::Div(dst, src1, src2, 1024);` |
| `Adds(dst, src, scalar, n)` | 向量加标量 | `AscendC::Adds(dst, src, 1.0f, 1024);` |
| `Muls(dst, src, scalar, n)` | 向量乘标量 | `AscendC::Muls(dst, src, 2.0f, 1024);` |
| `Exp(dst, src, n)` | 指数运算 | `AscendC::Exp(dst, src, 1024);` |
| `Log(dst, src, n)` | 对数运算 | `AscendC::Log(dst, src, 1024);` |
| `Sqrt(dst, src, n)` | 平方根 | `AscendC::Sqrt(dst, src, 1024);` |
| `Reciprocal(dst, src, n)` | 倒数 | `AscendC::Reciprocal(dst, src, 1024);` |

### 2.5 数据搬移

| API | 用途 | 示例 |
|-----|------|------|
| `DataCopy(dst, src, n)` | 数据拷贝 | `AscendC::DataCopy(local, global, 1024);` |
| `DataCopyPad(dst, src, ...)` | 带 Padding 的拷贝 | `AscendC::DataCopyPad(dst, src, params);` |

### 2.6 矩阵运算（Cube）

| API | 用途 | 示例 |
|-----|------|------|
| `Mmad(c, a, b, ...)` | 矩阵乘加 | `AscendC::Mmad(matC, matA, matB, M, K, N);` |

---

## 3. 常见错误模式

### 错误 #1: 缺少命名空间前缀

❌ **错误代码**:
```cpp
LocalTensor<float> buf;  // 缺少 AscendC::
SetVectorMask<float, COUNTER>(1024);  // 缺少 AscendC:: 和 MaskMode::
```

✅ **正确代码**:
```cpp
AscendC::LocalTensor<float> buf;
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(1024);
```

---

### 错误 #2: 混淆 AscendC 和 CUDA 语法

❌ **错误代码**:
```cpp
__global__ void MyKernel() {  // CUDA语法
    __shared__ float buffer[1024];  // CUDA语法
    threadIdx.x  // CUDA语法
}
```

✅ **正确代码**:
```cpp
__aicore__ inline void MyKernel() {  // AscendC语法
    AscendC::LocalTensor<float> buffer = ...;  // AscendC语法
    uint32_t blockIdx = AscendC::GetBlockIdx();  // AscendC语法
}
```

---

### 错误 #3: 在 kernel 中使用标准 C++ 库

❌ **错误代码**:
```cpp
__aicore__ inline void Process() {
    std::vector<float> data;  // ❌ kernel中禁止STL
    float* ptr = new float[100];  // ❌ kernel中禁止new
    printf("Debug\n");  // ❌ kernel中禁止printf
}
```

✅ **正确代码**:
```cpp
__aicore__ inline void Process() {
    AscendC::LocalTensor<float> data = queue.AllocTensor<float>();  // ✅
    // 调试：使用 scalar print APIs 或 host-side logging
}
```

---

### 错误 #4: Enum 值缺少作用域

❌ **错误代码**:
```cpp
AscendC::SetVectorMask<float, COUNTER>(1024);  // COUNTER 未限定
AscendC::TQue<VECIN, 2> queue;  // VECIN 未限定
```

✅ **正确代码**:
```cpp
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(1024);
AscendC::TQue<AscendC::TPosition::VECIN, 2> queue;
```

---

### 错误 #5: 模板参数顺序错误

❌ **错误代码**:
```cpp
AscendC::SetVectorMask<COUNTER, float>(1024);  // 顺序错误
```

✅ **正确代码**:
```cpp
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(1024);
// 顺序: <数据类型, 模式>
```

---

## 4. 语法检查清单（用于 suggest.md 验证）

在生成优化建议时，**必须**逐项检查：

### 4.1 Namespace 检查

- [ ] 所有 AscendC API 都有 `AscendC::` 前缀
- [ ] 没有使用 `ascendc::` (小写错误)
- [ ] 没有使用 `std::` 或其他 C++ 标准库（kernel代码）

### 4.2 Qualifier 检查

- [ ] Kernel 函数使用 `__aicore__` 修饰符
- [ ] 没有使用 CUDA 关键字（`__global__`, `__device__`, `__shared__`）

### 4.3 Type 检查

- [ ] 使用 `AscendC::LocalTensor<T>` 而非 `std::vector<T>`
- [ ] 使用 `AscendC::GlobalTensor<T>` 访问 Global Memory
- [ ] 没有使用 `new/delete/malloc/free`

### 4.4 API 存在性检查

- [ ] 使用的 API 在"已验证 API 列表"中
- [ ] 如果使用未列出的 API，标注 `[VERIFICATION NEEDED]`

### 4.5 Enum 检查

- [ ] `MaskMode::COUNTER` 而非 `COUNTER`
- [ ] `TPosition::VECIN` 而非 `VECIN`

### 4.6 模板语法检查

- [ ] `<T, Mode>` 顺序正确（先类型，后模式）
- [ ] 尖括号闭合正确

---

## 5. 验证示例

### 示例 #1: 向量循环优化

**待验证的代码**:
```cpp
// 建议修改
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(totalElements);
AscendC::ResetMask();
```

**验证过程**:
- ✅ Namespace: 所有 API 有 `AscendC::`
- ✅ API 存在性: `SetMaskCount`, `SetVectorMask`, `ResetMask` 在列表中
- ✅ Enum: `AscendC::MaskMode::COUNTER` 完整限定
- ✅ 模板参数: `<float, MaskMode>` 顺序正确

**结论**: ✅ 通过验证

---

### 示例 #2: 错误的优化建议（应被拒绝）

**待验证的代码**:
```cpp
// ❌ 错误建议
SetVectorMask<COUNTER, float>(n);  // 缺少 namespace + 参数顺序错误
std::vector<float> buffer(1024);  // kernel中不允许STL
```

**验证过程**:
- ❌ Namespace: 缺少 `AscendC::`
- ❌ Enum: `COUNTER` 未限定
- ❌ 模板参数: 顺序错误（应为 `<float, COUNTER>`）
- ❌ Type: 使用 `std::vector`（kernel禁止）

**结论**: ❌ 拒绝此建议，要求重新生成

---

## 6. 使用指南

### 在 suggest.md 中集成

```markdown
## Code Generation Requirements

### Step 3: Syntax Validation

Before outputting code, verify against AscendC API reference:

1. Load validation checklist from:
   `references/standards/ascendc_api_validation_reference.md`

2. For each API call in generated code:
   - Check: API in validated list?
   - Check: Namespace correct (`AscendC::`)?
   - Check: Enum values fully qualified?

3. If validation fails:
   - DO NOT output the code
   - Report error: "Generated code failed syntax validation"
   - List specific violations

4. If validation passes:
   - Proceed to output
   - Mark code as `[VALIDATED]`
```

---

## 7. 维护说明

### 如何更新此参考

1. **新增 API**:
   - 验证 API 在官方 AscendC 文档中存在
   - 添加到相应类别（Mask/Tensor/Vector等）
   - 提供示例用法

2. **发现新错误模式**:
   - 在"常见错误模式"章节增加条目
   - 提供错误示例和正确示例

3. **版本更新**:
   - 更新文档头部的版本号和日期
   - 在 CHANGELOG 中记录变更

---

## 8. 参考资料

- **官方文档**: AscendC 开发指南
- **API 头文件**: `kernel_operator.h`
- **规则库**: `assets/rules/special_rules/*/code_snippets/`

---

**最后更新**: 2026-02-24
**维护者**: code-performance-advisor skill
**审查周期**: 每月或新规则入库时
