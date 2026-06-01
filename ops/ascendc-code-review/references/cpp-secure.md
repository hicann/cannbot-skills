# CANN C++ 安全编码规范

<适用>
语言: C++
侧别: All, Tiling
领域: false
默认启用: true
</适用>

> **适用场景**：Tiling 侧（Host 侧）和 Kernel 侧（Device 侧）
>
> **说明**：安全编码红线规范，所有代码必须 100% 遵守。条款标注适用范围：`[适用: All]` / `[适用: Tiling]`

## 快速索引

### 两者都适用 `[适用: All]`（17 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 1.1 | 保证静态类型安全 | 总体原则 | 高 |
| 1.2 | 保证内存安全 | 总体原则 | 高 |
| 1.3 | 禁止使用未定义行为 | 总体原则 | 高 |
| 2.1 | 有符号整数运算不溢出 | 数值安全 | 高 |
| 2.2 | 无符号整数运算不回绕 | 数值安全 | 高 |
| 2.3 | 除法/余数运算除零保护 | 数值安全 | 高 |
| 3.1 | 禁止使用未初始化的变量 | 内存安全 | 高 |
| 3.3 | 数组索引校验 | 内存安全 | 高 |
| 3.4 | 禁止 sizeof 指针 | 内存安全 | 中 |
| 3.5 | 指针使用前判空 | 内存安全 | 高 |
| 4.1 | 外部输入合法性校验 | 输入验证 | 高 |
| 4.2 | 内存操作长度校验 | 输入验证 | 高 |
| 9.1 | 禁止逐位操作非 trivially copyable 对象 | 类与对象 | 中 |
| 10.3 | 敏感信息使用后清零 | 标准库 | 高 |
| 10.4 | 结构体字段末尾添加 | 标准库 | 中 |
| 10.5 | 接口变更考虑兼容性 | 标准库 | 中 |

### 仅 Tiling 适用 `[适用: Tiling]`（15 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 3.2 | 资源释放后指针置新值 | 内存安全 | 中 |
| 3.6 | 字符串存储有足够空间 | 内存安全 | 高 |
| 5.1 | 资源申请后判断是否成功 | 资源管理 | 高 |
| 5.2 | 资源泄露防护 | 资源管理 | 高 |
| 5.3 | new/delete 配对使用 | 资源管理 | 高 |
| 5.4 | new 操作符错误处理 | 资源管理 | 高 |
| 8.1 | 使用安全函数替代危险函数 | 安全函数 | 高 |
| 8.2 | 正确设置安全函数 destMax 参数 | 安全函数 | 高 |
| 8.3 | 检查安全函数返回值 | 安全函数 | 高 |
| 10.1 | 禁止从空指针创建 std::string | 标准库 | 高 |
| 10.2 | 不要保存 c_str/data 指针 | 标准库 | 中 |
| 11.1 | LOG API 禁止传入空指针 | LOG API 安全 | 高 |
| 11.2 | LOG API 参数数量与顺序必须与格式化占位符逐位一致 | LOG API 安全 | 高 |
| 11.3 | LOG API 参数类型必须与格式化说明符逐位匹配 | LOG API 安全 | 高 |
| 11.4 | LOG API 禁止传入已释放内存的指针 | LOG API 安全 | 高 |

---

### 1. 总体原则

#### 1.1 保证静态类型安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 模板类需注意类型转换（如 half ↔ float）和范围错误（FP16 溢出）。

C++应该是静态类型安全的，这样可以减少运行时的错误，提升代码的健壮性。但是由于C++存在下面的特性，会破坏C++静态类型安全，针对这部分特性要仔细处理：

- 联合体
- 类型转换
- 缩窄转换
- 类型退化
- 范围错误
- void* 类型指针

可以通过约束这些特性的使用，或者使用C++的新特性，例如std::variant（C++17）、std::span（C++20）等来解决这些问题，提升C++代码的健壮性。

#### 1.2 保证内存安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 使用 UB（Unified Buffer）和 GM（Global Memory），需要通过 `DataCopy` API 安全访问，避免越界和未初始化访问。

C++语言的内存完全由程序员自己控制，所以在操作内存的时候必须保证内存安全，防止出现内存错误：

- 内存越界访问
- 释放以后继续访问内存
- 解引用空指针
- 内存没有初始化
- 把指向局部变量的引用或者指针传递到了函数外部或者其他线程中
- 申请的内存或者资源没有及时释放

建议使用更加安全的C++的特性，比如RAII，引用，智能指针等，来提升代码的健壮性。

#### 1.3 禁止使用编译器"未定义行为" `[适用: All]`

遵循ISO C++标准，标准中未定义的行为禁止使用。对于编译器实现的特性或者GCC等编译器提供的扩展特性也需要谨慎使用，这些特性会降低代码的可移植性。

---

### 2. 数值运算安全

#### 2.1 确保有符号整数运算不溢出 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 `uint32_t` 等固定宽度类型进行循环索引和 Buffer 偏移计算，需防止溢出。

**【描述】**
有符号整数溢出是未定义的行为。出于安全考虑，对外部数据中的有符号整数值在如下场景中使用时，需要确保运算不会导致溢出：

- 指针运算的整数操作数(指针偏移值)
- 数组索引
- 变长数组的长度(及长度运算表达式)
- 内存拷贝的长度
- 内存分配函数的参数
- 循环判断条件

在精度低于int的整数类型上进行运算时，需要考虑整数提升。程序员还需要掌握整数转换规则，包括隐式转换规则，以便设计安全的算术运算。

**乘法示例（int32_t 乘法溢出）：**

```cpp
// 错误写法 — 两个 int32_t 相乘，结果可能超出 int32_t 范围
int32_t calcHeightAlign = GetAlignedSize(...);  // 对齐后高度，可达 65536
int32_t calcWidth = GetWidth(...);              // 宽度，可达 65536
int32_t size = calcHeightAlign * calcWidth;     // 65536 × 65536 = 4,294,967,296 溢出！

// 正确写法 — 提升为 int64_t 计算
int64_t size = static_cast<int64_t>(calcHeightAlign) * calcWidth;
```

**取反示例（INT64_MIN 取反溢出，红线问题）：**

```cpp
// 错误写法 — delta 取 INT64_MIN 时，-delta 溢出
int64_t delta = input2 - input1;       // 可能为 INT64_MIN = -9223372036854775808
int64_t absDelta = -delta;             // -(-9223372036854775808) = 9223372036854775808 > INT64_MAX!
// 有符号整数溢出是未定义行为（C++ 红线）

// 正确写法 — 转换为无符号类型后再求绝对值
uint64_t absDelta = (delta < 0) ? static_cast<uint64_t>(-delta) : static_cast<uint64_t>(delta);
```

**多维连乘示例（多维 shape 连续累乘溢出）：**

```cpp
// 错误写法 — 多维 shape 用 int32_t 连乘，极易溢出
int32_t totalSize = dim0 * dim1 * dim2 * dim3 * dim4;
// dim0=1024, dim1=1024, dim2=128, dim3=64 时积 ≈ 8.6 × 10^9 > INT32_MAX

// 正确写法 — 使用 int64_t 并提前提升
int64_t totalSize = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3 * dim4;
```

**【检视策略 — 工具驱动】**

核心流程：运行 check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个有符号算术表达式。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "X ≤ Y" 但找不到对应代码行）
- 用类型标签代替边界（"int64_t 所以够大不会溢出"——int64_t 的值可以是 1）

```bash
python3 {skill_base}/scripts/check_bounds.py \
  --expr "{表达式}" \
  --vars "a=int32_t:0:47" "b=int32_t:3:3" "c=int64_t:100:1000000" \
  --check overflow
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

#### 2.2 确保无符号整数运算不回绕 `[适用: All]`

> **Kernel 侧说明**：Kernel 中大量使用 `uint32_t` 进行 tileLength、blockLength 计算，需防止回绕。

**【描述】**
涉及无符号操作数的计算永远不会溢出，因为超出无符号整数类型表示范围的计算结果会按照（结果类型可表示的最大值 + 1）的数值取模。这种行为更多时候被非正式地称为无符号整数回绕。

**乘法示例（uint32_t 乘法回绕后再 cast uint64_t——值已经错了）：**

```cpp
// 错误写法 — 乘法在 uint32_t 完成，回绕发生后才 cast 到 uint64_t，无法恢复
uint32_t blockSize = 65536;    // 来自 TilingData
uint32_t strideKV = 65536;     // 来自 TilingData
uint64_t result = blockSize * strideKV;
// blockSize * strideKV 在 uint32_t 空间计算：65536 × 65536 = 4,294,967,296 > UINT32_MAX
// 实际结果: (65536 × 65536) mod 2^32 = 0 → 回绕后的 0 再 cast 到 uint64_t = 0

// 正确写法 — 乘法前至少一个操作数提升为 uint64_t
uint64_t result = static_cast<uint64_t>(blockSize) * strideKV;
```

**减法示例（uint32_t 减法回绕——结果用作数组索引）：**

```cpp
// 错误写法 — aivIdx * singleCoreSize 可能大于 totalOutputSize，减法回绕
uint32_t tailSize = totalOutputSize - aivIdx * singleCoreSize;
// totalOutputSize=100, aivIdx=47, singleCoreSize=3:
//   47 × 3 = 141, 100 - 141 按 uint32_t 计算 = 4294967255（回绕）
//   tailSize 被误认为合法大小，后续 DataCopy 搬运 4GB 数据 → 越界崩溃

// 正确写法 — 先判断大小关系，或使用 int64_t 中间结果
int64_t tailSizeSigned = static_cast<int64_t>(totalOutputSize) - 
                         static_cast<int64_t>(aivIdx) * singleCoreSize;
uint32_t tailSize = (tailSizeSigned > 0) ? static_cast<uint32_t>(tailSizeSigned) : 0;
```

**类型混合示例（size_t 与 int64_t 混合运算——负数回绕成极大值）：**

```cpp
// 错误写法 — N_ALIGN 是 size_t 常量（无符号），numIters 是 int64_t
// 按 C++ 整型提升规则 int64_t → size_t，负数变成极大正数
constexpr size_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
// 若 numIters 为 0 或负值，提升为 size_t 后回绕成 2^64-127 级别的极大值
// 再经 SetDim 传出，得到非预期的 shape，后续所有计算均错

// 正确写法 — 统一为有符号类型
constexpr int64_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
```

**【检视策略 — 工具驱动】**

核心流程：运行 check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个无符号算术表达式（减法、乘法、混合运算）。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "a ≥ b 恒成立" 但找不到对应代码行）
- 用类型标签代替边界（"uint64_t 所以够大不会回绕"——uint64_t 的值可以是 0）

```bash
python3 {skill_base}/scripts/check_bounds.py \
  --expr "{表达式}" \
  --vars "a=uint32_t:0:47" "b=uint32_t:3:3" "c=int64_t:100:1000000" \
  --check wraparound
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

#### 2.3 确保除法和余数运算不会导致除以零的错误 `[适用: All]`

> **Kernel 侧说明**：对每个除法/取余运算，按 SEC-2.1 的 Step 2 方法收集除数边界，按 Step 4 判定表做判定。不采用变量名模式匹配。

**【检视策略】**

与 SEC-2.1/2.2 相同的边界推演方法，将「除数」作为操作数，重点确认除数在所有极端组合下非零。除数的边界收集（Step 2）按以下优先级：

1. **代码级守卫** — 紧邻除法的 `if (divisor != 0)` 或 `if (divisor == 0) return`
2. **编译期常量** — `constexpr` 声明，值固定非零（如 `BLOCK_SIZE=32`）
3. **硬件固定值** — chip arch 参数，查 `/npu-arch`（如 `aivNum=48`）
4. **TilingData 值域推演** — Read Tiling 代码中除数的赋值语句，追溯其来源和值域
   - 来自常量计算且可证明非零 → PASS
   - 来自 shape / 运行时变量 → 必须考虑零值可能性
   - 无法确定 → 标记为「边界未知」

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| actS1Size / actS2Size | `GetActualSeqLen()` 运行时获取 | `if (actS1Size == 0) { return; }` |
| usedCoreNum 可能为零 | 空任务场景 | `if (usedCoreNum == 0) { return; }` |
| curActualSeqLen 动态值 | TND 布局累积差值 | `if (curActualSeqLen == 0) { return; }` |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验静态参数非零
OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
           OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
```

**【Kernel 侧校验示例】**

```cpp
// Kernel 阶段校验动态值零值分支
GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
if ((actS1Size == 0) || (actS2Size == 0)) {
    curActSeqLenIsZero = true;
    return;  // 早期退出，避免后续除法
}
// 后续计算：loopTimes = actS1Size / mBaseSize（actS1Size 已确保非零）
```

**【描述】**
整数的除法和取余运算的第二个操作数值为0会导致程序产生未定义的行为，因此使用时要确保整数的除法和余数运算不会导致除零错误。

---

### 3. 内存与指针安全

#### 3.1 禁止使用未初始化的变量 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类的成员变量必须在 `Init()` 函数中初始化，UB Buffer 通过 `AllocTensor` 获取后才能使用。

这里的变量，指的是局部动态变量，并且还包括内存堆上申请的内存块。因为他们的初始值都是不可预料的，所以禁止未经有效初始化就直接读取其值。

```cpp
void foo(...)
{
    int data;
    bar(data); // 错误：未初始化就使用
    ...
}
```

#### 3.2 指向资源句柄或描述符的变量，在资源释放后立即赋予新值 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源管理，Buffer 由 `InitBuffer` 静态分配，无需释放后置空。

**【描述】**
指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。

以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。

**【正确代码示例】**

```cpp
int foo(void)
{
    SomeStruct *msg = NULL;
    ... // 初始化msg->type，分配 msg->body 的内存空间

    if (msg->type == MESSAGE_A) {
        ...
        free(msg->body);
        msg->body = NULL;
    }

    ...
EXIT:
    ...
    free(msg->body);
    return ret;
}
```

#### 3.3 外部数据作为数组索引时必须确保在数组大小范围内 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 blockIdx、tileLength 等变量访问 GM/UB，需确保索引不越界。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 索引来自 TilingData | `constInfo.*`, `baseInfo.*` | Tiling 阶段已校验范围（如 Shape 维度校验） |
| 循环边界内索引 | `for (i = 0; i < bound; i++)` 内的 `arr[i]` | 循环条件保证索引在范围内 |
| GM/UB Buffer 内偏移 | `gmTensor[offset]`，offset 来自 Tiling | Tiling 阶段计算偏移范围 |

**判定方法**：
- 识别索引变量名匹配 `constInfo.*|baseInfo.*` 时，直接判定为 PASS
- 识别索引在循环边界内使用时，直接判定为 PASS

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需校验：

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| aiCoreIdx 核索引 | `GetBlockIdx()` 运行时获取 | `if (aiCoreIdx >= usedCoreNum) { return; }` |
| bIdx batch 累积差值边界 | TND 布局 `actualSeqLen[bIdx] - actualSeqLen[bIdx-1]` | `if (bIdx > 0) { ... } else { return actualSeqLen[0]; }` |
| 动态计算的偏移 | 运行时计算值 | 边界判断逻辑 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape 维度范围
OP_CHECK_IF(shape->GetDimNum() != expectedDim, 
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDim(i) > MAX_SIZE,
           OP_LOGE(context_, "dim %d exceeds limit", i), return ge::GRAPH_FAILED);
```

**【Kernel 侧校验示例】**

```cpp
// Kernel 核索引范围校验
if (aiCoreIdx >= tilingData->baseParams.usedCoreNum) {
    if ASCEND_IS_AIV {
        SyncAll();  // superkernel 同步
    }
    return;  // 超范围核退出
}

// Kernel TND 布局累积差值边界处理
if (bIdx > 0) {
    return actualSeqLen[bIdx] - actualSeqLen[bIdx - 1];  // 累积差值
} else {
    return actualSeqLen[0];  // 首元素，避免访问 bIdx-1
}
```

**【描述】**
外部数据作为数组索引对内存进行访问时，必须对数据的大小进行严格的校验，确保数组索引在有效范围内，否则会导致严重的错误。

**【正确代码示例】**

```cpp
#define DEV_NUM 10
static Dev devs[DEV_NUM];

int set_dev_id(size_t index, int id)
{
    if (index >= DEV_NUM) {
        ... // 错误处理
    }
    devs[index].id = id;
    return 0;
}
```

#### 3.4 禁止通过对指针变量进行sizeof操作来获取数组大小 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `LocalTensor<T>` 通过 API（如 `GetSize()`）获取大小，不能用 sizeof。

**【描述】**
将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。

**【错误代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

**【正确代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // 使用申请的缓冲区大小
```

#### 3.5 指针操作，使用前必须要判空 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
解引用空指针会导致程序产生未定义行为，通常会造成程序异常终止。

- 指针变量在使用前，一定要做好初始化的赋值，严禁对空指针进行访问
- 对于指针所代表的地址空间的任何操作，一定要保证空间的有效性
- 指针指向的内存释放后，需要调用者将指针显式置为NULL，防止"野指针"

#### 3.6 确保字符串存储有足够的空间容纳字符数据和null结束符 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 C 风格字符串处理。但 GM 数据搬运时需确保目标 Buffer 有足够空间。

**【描述】**
将数据复制到不足以容纳数据的缓冲区，会导致缓冲区溢出。

---

### 4. 输入验证

#### 4.1 外部输入数据需要做合法性校验 `[适用: All]`

> **Kernel 侧说明**：Kernel 中的 `TilingData` 参数（如 `constInfo.*`、`baseInfo.*`）已在 Tiling 阶段校验，无需重复校验。校验职责归属 Tiling 层。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 参数来自 TilingData | `constInfo.*`, `baseInfo.*`, `tilingData->*` | Tiling 阶段已校验（Shape、Dtype、范围、存在性） |
| __aicore__ 函数入参 | 模板类 Init/Process 参数 | 架构约定：尽量减少校验，有效性由调用者保证 |
| GM 指针可选输入 | `actualSeqLengths` 可能为 nullptr | 通过标志位 fallback 处理 |

**判定方法**：
- 识别参数变量名匹配 `constInfo.*|baseInfo.*|tilingData->*` 时，直接判定为 PASS
- 识别参数赋值来源为 `tilingData->xxx` 时，直接判定为 PASS
- 识别参数在 `__aicore__` 函数签名中时，不报告"输入验证缺失"

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需处理（非"校验"，而是"分支处理"）：

| 处理条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| actualSeqLengths 可选输入 | GM 指针可能为 nullptr | `if (ptr != nullptr) { SetGlobalBuffer(ptr); }` |
| isActualLenDimsNull 标志位 | Tiling 传递 | `if (flag == 1) { return staticSize; } else { return gm[bIdx]; }` |
| 空 Tensor 专用 Kernel | ShapeSize == 0 | 专用模板 `FiaKernelEmptyTensor`，InitOutput 为 0 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape、Dtype、范围
OP_CHECK_IF(context_->GetInputDesc(QUERY) == nullptr,
           OP_LOGE(context_, "query desc is null"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDimNum() != expectedDim,
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(headDim == 0,
           OP_LOGE(context_, "headDim is 0"), return ge::GRAPH_FAILED);

// Tiling 阶段校验参数组合存在性
ge::graphStatus FiaTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
        OP_LOGE(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

**【Kernel 侧处理示例】**

```cpp
// Kernel 可选 GM 指针条件处理（非"校验"，而是"分支处理"）
if (actualSeqLengthsQ != nullptr) {
    actualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
}

// Kernel 标志位 fallback（Tiling 已传递 isActualLenDimsNull）
if (constInfo.isActualLenDimsNull == 1) {
    return constInfo.s1Size;  // 静态值 fallback
} else {
    return actualSeqQlenAddr[bIdx];  // 动态值
}
```

**【描述】**

- 外部输入数据需要做合法性校验且确保校验范围正确
- 边界接口需要对传入的地址做合法性校验避免任意地址读写
- 需要对入参进行合法性校验避免数组越界
- 需要对地址偏移校验避免任意地址读写
- 外部传入指针需要判空后使用
- 外部入参参与循环、递归条件的运算，必须严格校验边界和终止条件
- 文件路径来自外部数据时，必须对其做合法性校验

#### 4.2 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `DataCopy` 的搬运长度需校验，确保不超过 UB 容量和 GM 数据范围。

**【描述】**
将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量足够大以容纳要复制的数据。

---

### 5. 资源管理

#### 5.1 资源申请后必须判断是否成功 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源申请（malloc/new），Buffer 由 `InitBuffer` 静态分配，编译期确定。

**【描述】**
内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。

**【正确代码示例】**

```cpp
struct tm *make_tm(int year, int mon, int day, int hour, int min, int sec)
{
    struct tm *tmb = (struct tm *)malloc(sizeof(*tmb));
    if (tmb == NULL) {
        ... // 错误处理
    }
    tmb->year = year;
    ...
    return tmb;
}
```

#### 5.2 资源泄露（内存、句柄、锁等） `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态内存、无锁、无句柄，Buffer 静态分配无需释放。

**【描述】**

- 资源申请和释放必须匹配，包括：内存类的malloc/free/alloc_page/free_page, 锁lock/unlock、文件open/close等
- 释放结构体/类/数组/各类数据容器指针前，必须先释放成员指针
- 对外接口处理涉及资源申请但未释放，引起资源泄露，导致拒绝服务
- C++捕获异常时确保恢复程序的一致性; 建议使用RAII模式，确保资源在异常发生时自动释放

#### 5.3 new和delete配对使用，new[]和delete[]配对使用 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new/delete。

#### 5.4 使用恰当的方式处理new操作符的内存分配错误 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new。

---

### 8. 安全函数使用

#### 8.1 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 memcpy_s/memset_s，使用 Ascend C API（如 `Duplicate`、`DataCopyPad`）。

| 函数类别 | 危险函数 | 安全替代函数 |
|---------|---------|------------|
| 内存拷贝 | memcpy或bcopy | memcpy_s |
| 内存拷贝 | memmove | memmove_s |
| 字符串拷贝 | strcpy | strcpy_s |
| 字符串串接 | strcat | strcat_s |
| 格式化输出 | sprintf | sprintf_s |
| 格式化输出 | snprintf | snprintf_s |
| 格式化输入 | scanf | scanf_s |
| 内存初始化 | memset | memset_s |

#### 8.2 正确设置安全函数中的destMax参数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

#### 8.3 必须检查安全函数返回值，并进行正确的处理 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

原则上，如果使用了安全函数，需要进行返回值检查。如果返回值!=EOK, 那么本函数一般情况下应该立即返回，不能继续执行。

```cpp
{
    ...
    err = memcpy_s(destBuff, destMax, src, srcLen);
    if (err != EOK) {
        MS_LOG("memcpy_s failed, err = %d\n", err);
        return FALSE;
    }
    ...
}
```

---

### 9. 类与对象安全

#### 9.1 禁止逐位操作非trivially copyable对象 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类都是 POD 类型，可以使用 `Duplicate` 进行内存操作。

---

### 10. 标准库安全

#### 10.1 禁止从空指针创建std::string `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 10.2 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 10.3 内存中的敏感信息使用完毕后立即清0 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 UB 数据可通过 `Duplicate` 清零，GM 数据需在 Host 侧处理。

口令、密钥等敏感信息使用完毕后立即清零，避免被攻击者获取。

#### 10.4 对外结构体接口新增字段时必须在结构体最后添加 `[适用: All]`

> **Kernel 侧说明**：`TilingData` 结构体新增字段需在末尾添加，保持 ABI 兼容性。

为了最大程度上在ABI层面的兼容，对外结构体接口添加新字段时必须在结构体最后添加。

#### 10.5 外部接口或数据结构变更必须考虑兼容性 `[适用: All]`

> **Kernel 侧说明**：Kernel 接口（如 TilingData 结构体）变更需考虑版本兼容性。

外部接口、接口参数、返回值、数据结构、消息字段等变更会引起版本兼容性问题，非必要不建议变更。

---

### 11. LOG API 安全使用

> **适用范围**：仅 Tiling 侧（Host 侧）。Kernel 侧使用 `AscendC::PRINTF`，无下列风险。

Tiling 侧使用 `OP_LOGE` / `OP_LOGD` / `OP_LOGW` 等格式化 LOG 宏时，若参数使用不当，轻则输出乱码，重则引发段错误（SIGSEGV）。以下 4 条为强制要求。

LOG 宏签名（业务代码标准调用形式）：

```cpp
OP_LOGE(context->GetNodeName(), "format string %s %ld", arg1, arg2);
OP_LOGD(context->GetNodeName(), "format string %lu", arg1);
```

---

#### 11.1 LOG API 禁止传入空指针作为字符串参数 `[适用: Tiling]`

**【问题说明】**

`%s` 会解引用传入指针，若指针为 `nullptr`，将访问地址 0（受 OS 保护），导致段错误。Tiling 侧常见场景：从 `context` 获取 Desc/Attr 后未判空直接传入 LOG。

**错误示例**

```cpp
// 来自 quant_grouped_matmul_dequant_tiling.cpp 同类风险
auto inputDesc = context->GetInputDesc(0);
// 若 inputDesc 为 nullptr，GetDataType() 返回的字符串描述也可能为空
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
// 风险：inputDesc 未判空就调用成员函数
```

**正确示例**

```cpp
auto inputDesc = context->GetInputDesc(0);
if (inputDesc == nullptr) {
    OP_LOGE(context->GetNodeName(), "GetInputDesc(0) returned nullptr, skip dtype log.");
    return ge::GRAPH_FAILED;
}
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
```

---

#### 11.2 LOG API 参数数量与顺序必须与格式化占位符逐位一致 `[适用: Tiling]`

**【问题说明】**

参数数量少于占位符时，LOG 宏会从栈上读取垃圾值填充缺失参数。若垃圾值被解释为非法指针（`%s`/`%p`），将触发非法内存访问。

**参数顺序错位是同等严重的问题**：即使参数数量正确，若参数传入顺序与格式符位置不对应，会导致：
- `%u`(位置1) 收到 `const char*` → 未定义行为（指针值被当作整数打印）
- `%s`(位置3) 收到整数 → **段错误(SIGSEGV)**（整数被当作地址去读字符串）

> **注意**：计数验证（参数数 == 格式符数）只是必要条件。**数量正确但顺序错误时，本条同样判定为 FAIL**。检视时必须对每个 LOG 调用执行**逐位顺序一致性验证**（见检视方法），并**与 SEC-11.3 联合执行**——本条验证数量+顺序，SEC-11.3 验证逐位类型匹配。两条联合才能完整覆盖所有 LOG 格式符安全问题。

**错误示例 1 — 参数数量不一致**

```cpp
// 2 个占位符，但只传了 1 个参数
OP_LOGD(context->GetNodeName(),
        "gmmSwigluBaseParams.M: %ld, K: %ld", m);   // 缺少 k，栈数据被错误读取
```

**错误示例 2 — 参数数量一致但顺序错位**

```cpp
// 格式符: %u %u %s %u %u（位置1→uint, 位置3→string）
// 参数:   inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size
//         ↑ const char*       ↑ uint        ↑ uint        ↑ uint  ↑ uint
//         ❌ 位置1: %u 收到 const char* → 指针值被当作整数打印
//         ❌ 位置3: %s 收到 uint → 整数被当作地址去读字符串 → 段错误
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
        "the last dim (D0) of kvCache(%u) should be %u; "
        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

**正确示例 1**

```cpp
OP_LOGD(context->GetNodeName(),
        "gmmSwigluBaseParams.M: %ld, K: %ld", m, k);
```

**正确示例 2 — 参数顺序与格式符逐位对应**

```cpp
// 格式符: %u %u %s %u %u
// 参数:   tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size
//         ↑ uint        ↑ uint        ↑ const char*     ↑ uint  ↑ uint
//         ✅ 逐位对应正确
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
        "the last dim (D0) of kvCache(%u) should be %u; "
        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
        tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

**【检视方法 — SEC-11.2 专属】**

对每个 LOG 调用，必须执行以下验证：

```
Step 1 — 提取格式符位置序列
  从格式字符串中按出现顺序提取所有格式说明符，记录 (位置序号, 格式符)
  示例: "%u %u %s %u %u" → [(1, %u), (2, %u), (3, %s), (4, %u), (5, %u)]

Step 2 — 提取参数位置序列
  从参数列表按传入顺序提取所有参数表达式，记录 (位置序号, 参数表达式)
  示例: inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size
        → [(1, inputName.c_str()), (2, tempD0/NUM8), (3, d0Size/NUM8), (4, tempD0), (5, d0Size)]

Step 3 — 数量验证
  格式符数量 == 参数数量?
  不等 → FAIL（参数缺失或多余）

Step 4 — 顺序一致性验证（本条新增核心步骤）
  对每个位置 i:
    推断 param[i] 的类型 → 对照 format[i] 期望的类型族:
      %s  → 期望: string-like（const char*, .c_str(), 字符串字面量）
      %u  → 期望: unsigned-integer-like（uint32_t, uint8_t, unsigned int...）
      %d  → 期望: signed-integer-like（int32_t, int, bool...）
      %ld → 期望: signed-64bit-like（int64_t, long...）
      %lu → 期望: unsigned-64bit-like（uint64_t, unsigned long...）
      %lld→ 期望: signed-64bit-like（int64_t, long long）
      %llu→ 期望: unsigned-64bit-like（uint64_t, unsigned long long）
      %zu → 期望: size_t-like
      %p  → 期望: pointer-like

    param[i] 类型不在 format[i] 期望类型族内 → FAIL，记录:
      位置 i: format[i]=%X 期望 <类型族>，实际 param[i]=<表达式> 类型 <T>

  特判 — 顺序错位的危险模式:
    若参数列表中存在 string-like 参数（.c_str()、字符串字面量），
    但它出现在 %u/%d/%ld/%lu 位置 → 高风险 FAIL（%s 收到整数 → 段错误）
    若参数列表中存在 integer 参数，
    但它出现在 %s 位置 → 高风险 FAIL（整数被当作地址 → 段错误）
```

---

#### 11.3 LOG API 参数类型必须与格式化说明符逐位匹配 `[适用: Tiling]`

**【问题说明】**

类型大小不匹配时，LOG 宏按说明符宽度截断或读取超量字节，导致后续参数全部错位。Tiling 侧最常见：`uint64_t` shape 维度误用 `%d`（4字节），实际类型为 8 字节，造成参数错位。

**参数顺序错位同样导致类型不匹配**：即使每个参数类型在参数列表中都能找到对应格式符，但若参数与格式符的位置不对应（如 `%s` 在位置3，但 `const char*` 参数在位置1），则该位置的逐位类型仍然不匹配。此类问题属于 SEC-11.2（顺序）+ SEC-11.3（类型）的交叉违规，两条**必须联合检出**。

> **联合检视要求**：SEC-11.3 的逐位比对（Step 4）同时覆盖了 SEC-11.2 的顺序验证——当位置 i 的参数类型与格式符期望类型不匹配时，可能是"类型本身错误"也可能是"顺序错位导致类型错配"，两条均判定 FAIL。**禁止将 SEC-11.2 和 SEC-11.3 分开独立执行后简单合并结果**——必须通过同一次逐位比对同时产出两条的判定。

**错误示例 1 — 类型大小不匹配（传统场景）**

```cpp
// 参考 quant_grouped_matmul_dequant_tiling.cpp：_Params.originM 为 uint64_t
OP_LOGE(context->GetNodeName(),
        "No valid row found for n = %d, ubSize = %d\n", n, ubSize);
// 错误：n/ubSize 均为 uint64_t，%d 只读 4 字节，后续参数全部错位
```

**错误示例 2 — 顺序错位导致逐位类型不匹配（高风险场景）**

```cpp
// 格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
// 参数:   inputName.c_str()(1)  tempD0/NUM8(2)  d0Size/NUM8(3)  tempD0(4)  d0Size(5)
// 逐位比对:
//   位置1: %u 期望 unsigned-integer，实际 const char* → ❌ FAIL (SEC-11.2 顺序错位 + SEC-11.3 类型不匹配)
//   位置2: %u 期望 unsigned-integer，实际 uint → ✅ PASS
//   位置3: %s 期望 string-like，实际 uint → ❌ FAIL (段错误风险)
//   位置4: %u 期望 unsigned-integer，实际 uint → ✅ PASS
//   位置5: %u 期望 unsigned-integer，实际 uint → ✅ PASS
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
        "the last dim (D0) of kvCache(%u) should be %u; "
        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
// 注意：参数数量=5 == 格式符数量=5 → 单独看 SEC-11.2 计数验证会 PASS
//       但逐位比对发现位置1和位置3不匹配 → 两条同时 FAIL
```

**正确示例**

```cpp
// 业务代码正确写法（grouped_matmul_swiglu_quant_tiling.cpp 第 75 行）
OP_LOGE(context->GetNodeName(),
        "GMM_SWIGLU_QUANT TILING: No valid row found for n = %lu, ubSize = %lu\n", n, ubSize);
```

**Tiling 侧常见类型与说明符对照**

| 类型 | 推荐说明符 (通用) | 常见错误 | 说明 |
| :--- | :--- | :--- | :--- |
| `int64_t` | `%lld` | `%d`, `%ld` | `%ld` 在 Windows/32位系统上会截断数据。`%lld` 是标准且通用的写法。 |
| `uint64_t` | `%llu` | `%u`, `%lu` | 同上，`%lu` 在 Windows 上仅读取 32 位。 |
| `uint32_t` | `%u` | `%d` | `%d` 会导致大于 2^31 的数值显示为负数。 |
| `int32_t` | `%d` | `%u` | 标准整型，直接对应。 |
| `bool` | `%d` | `%s` | 除非手动转字符串，否则 `%d` (0/1) 最安全且无需额外逻辑。 |
| `size_t` | `%zu` | `%d`, `%u` | `size_t` 在 64 位系统上是 64 位，用 `%u` 会截断。 |
| `void*` | `%p` | `%x` | 永远用 `%p` 打印指针地址。 |

```cpp
// bool 的正确记录方式（业务代码第 248 行）
OP_LOGD(context->GetNodeName(),
        "isSplitWorkSpace: %s", isSplitWorkSpace ? "true" : "false");
```

**参数顺序错位示例（SEC-11.2 + SEC-11.3 交叉违规）**

```cpp
// 格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
// 参数:   inputName.c_str()(1)  tempD0/NUM8(2)  d0Size/NUM8(3)  tempD0(4)  d0Size(5)
// 逐位比对结果:
//   位置1: %u 期望 unsigned-integer → 实际 const char* → ❌ SEC-11.2 顺序错位 + SEC-11.3 类型不匹配
//   位置3: %s 期望 string-like     → 实际 uint        → ❌ SEC-11.3 类型不匹配（段错误风险）
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(fiaInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
        "the last dim (D0) of kvCache(%u) should be %u; "
        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
// 后果：%s(位置3) 收到整数 → 将整数当地址读字符串 → 段错误(SIGSEGV)
//       %u(位置1) 收到字符串指针 → 未定义行为
```

**【联合检视方法 — SEC-11.2 + SEC-11.3 必须同时执行】**

> **强制要求**：SEC-11.2 和 SEC-11.3 **禁止分开独立执行**。必须通过同一次逐位比对流程同时产出两条的判定结果。分开执行会导致：SEC-11.2 计数 PASS + SEC-11.3 独立类型匹配 PASS → 顺序错位漏检。

对每个 LOG/printf 类调用，必须执行**逐位类型交叉验证**：

```
Step 0 — 发现所有 LOG 调用
  0.1 grep `OP_LOGE\|OP_LOGD\|OP_LOGW\|OP_LOGI` 定位宏调用行号（不依赖格式符匹配，避免跨行漏检）
  0.2 对每个命中，向后 Read 至 `);` 获取完整调用（可能跨 3-15 行）
  0.3 若格式字符串为 C 字面量多行拼接（`"a" "b"`），先合并再解析

Step 1 — 提取格式符位置序列
  从**已合并**的格式字符串中提取所有格式说明符，按出现顺序记录 (位置序号, 格式符)
  示例: "%u %u %s %u %u" → [(1, %u), (2, %u), (3, %s), (4, %u), (5, %u)]
  支持的格式符: %u %d %ld %lld %lu %llu %s %f %lf %p %zu %x %X %c

Step 2 — 提取参数位置序列
  从 LOG 调用的参数部分提取 N 个参数表达式，按传入顺序记录 (位置序号, 参数表达式)
  示例: inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size
        → [(1, inputName.c_str()), (2, tempD0/NUM8), (3, d0Size/NUM8), (4, tempD0), (5, d0Size)]

Step 3 — 推断每个参数的类型
  对每个参数表达式:
    - 变量名 → Grep 声明位置获取类型
    - .c_str() → const char*
    - 表达式(如 a/b) → 推断结果类型（整数除法→整数类型）
    - 字符串字面量 → const char*
    - 三目运算符 ?: → 推断两侧公共类型
    - 解引用指针 *ptr → 推断 ptr 的指向类型（如 `const int64_t*` 解引用 → `int64_t`）

Step 4 — 逐位比对（同时产出 SEC-11.2 和 SEC-11.3 判定）

  4.1 数量验证（SEC-11.2 基础检查）
    格式符数量 == 参数数量?
    不等 → SEC-11.2 FAIL（参数缺失或多余）

  4.2 逐位类型交叉验证（SEC-11.2 顺序 + SEC-11.3 类型 联合核心步骤）
    对每个位置 i (0 to N-1):
      param[i] 的推断类型 是否兼容 format[i] 的格式符？

      兼容规则:
        %s   → const char*, char*, std::string(.c_str()), 字符串字面量
        %u   → uint32_t, uint16_t, uint8_t, unsigned int, unsigned short
        %d   → int32_t, int16_t, int8_t, int, bool
        %ld  → int64_t, long, long long (有符号)
        %lld → int64_t, long long
        %lu  → uint64_t, unsigned long
        %llu → uint64_t, unsigned long long
        %zu  → size_t
        %p   → 任意指针类型, void*
        %f   → float, double
        %lf  → double

      不兼容 → 判定:
        若 param[i] 的类型在参数列表中能找到某个其他格式符与其兼容
        → SEC-11.2 FAIL(顺序错位) + SEC-11.3 FAIL(类型不匹配) — 交叉违规
        若 param[i] 的类型在整个格式符序列中找不到任何兼容的格式符
        → SEC-11.3 FAIL(类型本身不匹配) — 纯类型违规

  4.3 高风险特判
    以下位置不匹配模式为**高风险 FAIL**，必须在报告中标注风险等级:
      - %s 位置收到 integer → 段错误风险（整数被当作地址读字符串）
      - %u/%d/%ld/%lu 位置收到 string-like（.c_str()、字符串字面量）→ 未定义行为
      - int64_t 参数配 %d（8字节配4字节）→ 参数截断+后续错位
      - uint32_t 参数配 %d（无符号配有符号格式符）→ 大值显示为负数

Step 5 — 输出判定表
  对每个 LOG 调用，输出逐位比对判定表:

  | 位置 | 格式符 | 期望类型族 | 实际参数 | 推断类型 | SEC-11.2 | SEC-11.3 |
  |------|--------|-----------|---------|---------|----------|----------|
  | 1    | %u     | unsigned-int | inputName.c_str() | const char* | FAIL(顺序) | FAIL(类型) |
  | 2    | %u     | unsigned-int | tempD0/NUM8      | uint      | PASS      | PASS      |
  | 3    | %s     | string-like  | d0Size/NUM8      | uint      | FAIL(顺序) | FAIL(类型) |
  | ...

  全部 PASS → SEC-11.2 PASS + SEC-11.3 PASS
  任一位置 FAIL → 标注对应条例的 FAIL 及具体违规类型
```

---

#### 11.4 LOG API 禁止传入已释放内存的指针 `[适用: Tiling]`

**【问题说明】**

Tiling 侧手动管理的堆内存（`new` / `malloc`）释放后若仍传入 `%s`，行为未定义，大概率触发段错误。典型场景：在函数末尾统一释放资源，但 LOG 语句写在释放之后。

**错误示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
delete[] errMsg;
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 野指针，已释放
```

**正确示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 先记录
delete[] errMsg;
errMsg = nullptr;
```