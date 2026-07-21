# ascendc-review

**Tagline**: AscendC 结构/安全红线自动检查引擎

**Triggers**:
- 当 `cake-code-review` 主 skill 进入 Phase 2（Structural Review）时自动调用
- User 明确说"只检查不修复"时独立使用

---

## 1. Introduction

`ascendc-review` 是结构/安全审查引擎，负责扫描 AscendC 算子代码，识别违反《编码红线.md》规范的结构性、红线和安全性问题。

**边界说明**:
- 本阶段负责 API 之外的结构性、红线、安全性、代码模式问题
- 本阶段不负责算法链路是否完整、数学不变量是否成立、迭代更新是否遗漏
- 算法完整性和不变量检查统一转到 `ascendc-algorithm-check.md`

**检查范围**：
- ✅ 7 大编码红线规范（1.1 - 1.7）
- ✅ 6 条 TopN 问题（2.1 - 2.6）
- ✅ 覆盖 `op_kernel/*.h` 和 `op_host/*_tiling.cpp`
- ❌ 不负责算法级完整性审查（见 `ascendc-algorithm-check.md`）

**输出格式**：
- JSON（供 `ascendc-fix` 消费）
- Markdown（供人类阅读）

---

## 2. Usage

### 完整检查

```
使用 ascendc-review 检查 ai_infra_sinkhorn_grad 算子
```

### 仅检查特定文件

```
使用 ascendc-review 检查 op_kernel/ai_infra_sinkhorn_grad_generalized.h 文件
```

### 批量检查

```
使用 ascendc-review 检查 training/ascendc/src/ops-transformer/mhc/ 下所有算子
```

---

## 3. How It Works

### 检查流程

```
1. 定位算子目录
   └─> 查找 op_kernel/ 和 op_host/ 子目录

2. 读取源代码文件
   └─> op_kernel/*.h, op_kernel/*.cpp
   └─> op_host/*_tiling.cpp, op_host/*_tiling.h

3. 应用结构/安全检查规则（见下文"检查清单"）
   └─> 逐行扫描代码
   └─> 匹配违规模式
   └─> 标记问题（ID、严重性、位置）

4. 生成报告
   └─> JSON: 机器可读，供 ascendc-fix 使用
   └─> Markdown: 人类可读，供开发者审查
```

### 检查清单

#### 红线 1.1 - 除零保护

**规则**：所有除法和求余操作前必须检查除数非零

**检测模式**：
```cpp
// ❌ 违规示例
result = a / b;
remainder = a % b;
value = CeilDiv(num, denom);  // 如果 denom=0 会崩溃

// ✓ 正确示例
if (b == 0) return ERROR;
result = a / b;
```

**扫描逻辑**：
1. 查找所有 `/` 和 `%` 操作符
2. 检查前面 3 行内是否有对除数的非零检查
3. 查找自定义除法函数（如 `CeilDiv`, `CeilAlign`）并检查其实现

**问题标记**：
```json
{
  "id": "XXX-01",
  "rule": "红线1.1 - 除零保护",
  "severity": "P0",
  "location": "file.cpp:123",
  "code_snippet": "cubeDealnDPeCore_ = nD_ / GetBlockNum();",
  "description": "GetBlockNum() 返回值未检查是否为 0"
}
```

#### 红线 1.2 - 数组越界保护

**规则**：所有数组访问必须验证索引合法性

**检测模式**：
```cpp
// ❌ 违规示例
for (int i = 0; i < count; i++) {
    buffer[index++] = value;  // index 可能超出 buffer 大小
}

// ✓ 正确示例
for (int i = 0; i < count; i++) {
    if (index >= bufferSize) return;
    buffer[index++] = value;
}
```

**扫描逻辑**：
1. 查找所有数组访问（`arr[index]`, `SetValue(index, ...)`, `GetValue(index)`）
2. 检查是否有边界检查（`if (index >= size)` 或 `if (index < 0 || index >= size)`）
3. 特别关注循环内的累加索引（`index++`）

**问题标记**：
```json
{
  "id": "XXX-02",
  "rule": "红线1.2 - 数组越界",
  "severity": "P0",
  "location": "file.cpp:387-407",
  "code_snippet": "gatherOffsetBuf_.SetValue(offset1++, ...)",
  "description": "循环内 offset1 累加但未检查是否超过 hFusionBufLen_"
}
```

#### 红线 1.3 - 溢出保护

**规则**：加法、乘法、减法操作必须有溢出和下溢保护（int64/uint64 除外）

**检测模式**：
```cpp
// ❌ 违规示例（uint32_t 可能溢出）
uint32_t globalUbOffset_;
globalUbOffset_ = hFusionBufLen_ * sizeof(uint32_t);      // 溢出风险
globalUbOffset_ += hFusionBufLen_ * sizeof(P);             // 继续累加

// ✓ 正确示例
uint64_t globalUbOffset_;  // 使用 64 位类型
if (offset > UINT64_MAX - increment) return ERROR;  // 溢出检查
offset += increment;
```

**扫描逻辑**：
1. 查找涉及内存偏移或大小的变量（命名包含 `offset`, `size`, `length`）
2. 检查其类型是否为 `uint32_t` 或 `int32_t`（应改为 `int64_t`/`uint64_t`）
3. 查找累加操作（`+=`, `+`, `*`）并检查是否有溢出保护

**问题标记**：
```json
{
  "id": "XXX-03",
  "rule": "红线1.3 + TopN2.3 - 溢出保护",
  "severity": "P1",
  "location": "file.cpp:304,379-383",
  "code_snippet": "uint32_t globalUbOffset_;",
  "description": "GM 内存偏移应使用 int64_t，且需添加溢出检查"
}
```

#### 红线 1.4 - 变量初始化

**规则**：变量使用前必须进行有效初始化

**检测模式**：
```cpp
// ❌ 违规示例
bool alphaBufInitialized_;  // 声明时未初始化
float scaleMean_;

// ✓ 正确示例
bool alphaBufInitialized_ = false;
float scaleMean_ = 0.0f;
```

**扫描逻辑**：
1. 查找类成员变量声明（不在构造函数或初始化列表中）
2. 检查是否有初始值（`= value`）
3. 特别关注 `bool` 类型（未初始化时值不确定）

**问题标记**：
```json
{
  "id": "XXX-04",
  "rule": "红线1.4 - 变量初始化",
  "severity": "P1",
  "location": "file.h:307-309",
  "code_snippet": "bool alphaBufInitialized_;",
  "description": "成员变量声明时未初始化"
}
```

#### 红线 1.5 - 空指针保护

**规则**：指针解引用前必须检查是否为 null

**检测模式**：
```cpp
// ❌ 违规示例
__aicore__ inline void Init(..., const TilingData *tilingData, ...) {
    tilingData_ = tilingData;
    totalLength_ = tilingData->totalLength;  // 直接解引用
}

// ✓ 正确示例
__aicore__ inline void Init(..., const TilingData *tilingData, ...) {
    if (tilingData == nullptr) return;
    tilingData_ = tilingData;
    totalLength_ = tilingData->totalLength;
}
```

**扫描逻辑**：
1. 查找所有指针参数（特别是 `Init` 函数）
2. 查找指针解引用操作（`ptr->member`, `*ptr`）
3. 检查前面是否有 `if (ptr == nullptr)` 检查

**问题标记**：
```json
{
  "id": "XXX-05",
  "rule": "红线1.5 - 空指针保护",
  "severity": "P0",
  "location": "file.h:63-73",
  "code_snippet": "totalLength_ = tilingData->totalLength;",
  "description": "tilingData 未进行 null 检查即解引用"
}
```

#### TopN 2.1 - 特殊值处理

**规则**：算子设计时必须考虑 nan/inf/-inf/±0 等特殊值

**检测模式**：
```cpp
// ❌ 风险示例
Div(gradLocal[offset], subResultTensor[offset], sumLocal[i], tAlign_);
// 如果 sumLocal[i] 包含 0 或 NaN，会产生 Inf 或 NaN
```

**扫描逻辑**：
1. 查找浮点除法操作（特别是向量指令 `Div`, `Muls`, `Adds`）
2. 仅检查表层防护是否存在，例如文档输入约束（如"sum_out 必须为正数"）或局部运行时 guard
3. 不在本阶段证明分母不变量是否由前序归约、迭代或归一化链正确建立；该项转交 `ascendc-algorithm-check.md`

**问题标记**：
```json
{
  "id": "XXX-06",
  "rule": "TopN2.1 - 特殊值处理",
  "severity": "P1",
  "location": "file.h:350,390",
  "code_snippet": "Div(gradLocal, subResult, sumLocal, nt);",
  "description": "除法操作未处理 sumLocal 可能为 0 或 NaN 的情况",
  "suggestion": "在文档中明确输入约束，或添加运行时检查"
}
```

#### TopN 2.2 - 输入校验

**规则**：Infershape/Tiling 的外部输入必须校验合法性

**检测模式**：
```cpp
// ❌ 违规示例（Tiling 中未校验 nSize_）
nSize_ = gradOutputShape_->GetDim(DIM_2);
// 后续直接使用 nSize_ 进行计算，未检查范围

// ✓ 正确示例
nSize_ = gradOutputShape_->GetDim(DIM_2);
if (nSize_ < MIN_N_SIZE || nSize_ > MAX_N_SIZE) {
    OP_LOGE(...);
    return ge::GRAPH_FAILED;
}
```

**扫描逻辑**：
1. 查找所有从 `context->GetInputShape()` 获取的维度值
2. 检查是否有合法性校验（范围检查、非零检查）
3. 特别关注后续用于除法或内存计算的参数

**问题标记**：
```json
{
  "id": "XXX-07",
  "rule": "TopN2.2 - 输入校验",
  "severity": "P2",
  "location": "file_tiling.cpp:285-306",
  "code_snippet": "nSize_ = gradOutputShape_->GetDim(DIM_2);",
  "description": "nSize_ 从外部输入获取后未校验合法性"
}
```

#### TopN 2.3 - int64 偏移

**规则**：GM 内存偏移和大小必须使用 int64 表示

**检测模式**：
```cpp
// ❌ 违规示例
uint32_t globalUbOffset_;  // 应为 uint64_t
int colOffset = i * n_ * tAlign_;  // 应为 int64_t

// ✓ 正确示例
uint64_t globalUbOffset_;
int64_t colOffset = static_cast<int64_t>(i) * n_ * tAlign_;
```

**扫描逻辑**：
1. 查找所有命名包含 `offset`, `size`, `length` 的变量
2. 检查其类型是否为 `int64_t` 或 `uint64_t`
3. 检查是否涉及 GM 内存地址计算

**问题标记**：
```json
{
  "id": "XXX-08",
  "rule": "TopN2.3 - int64偏移",
  "severity": "P2",
  "location": "file.h:332",
  "code_snippet": "int colOffset = i * n_ * tAlign_;",
  "description": "GM 内存偏移应使用 int64_t 类型"
}
```

#### TopN 2.4 — stdlib 数学函数在 NPU kernel 中不可用 (WF-007)

**规则**: AscendC kernel 编译为 NPU ELF，不链接 C 标准库，严禁调用 `expf`、`logf`、`sqrtf`、`__builtin_expf`、`std::exp` 等 stdlib 函数。

**检测模式**:
```cpp
// ❌ 违规示例 — 链接报错: undefined symbol: expf
float corr = __builtin_expf(rowMax - m_new);
float val  = std::expf(x);

// ✓ 正确示例 — 用 AscendC 向量指令替代
expLocal.SetValue(0, rowMax - m_new);
expLocal.SetValue(1, tileMax - m_new);
AscendC::Exp(expLocal, expLocal, 8);   // 至少 8 个 float (32B 对齐)
AscendC::PipeBarrier<PIPE_V>();
float corr_old  = expLocal.GetValue(0);
float corr_tile = expLocal.GetValue(1);
```

**扫描逻辑**:
1. 扫描 `op_kernel/` 下所有 `.cpp`/`.h` 文件
2. 查找 `__builtin_expf`, `__builtin_logf`, `__builtin_sqrtf`, `std::exp`, `std::log`, `std::sqrt`, `expf(`, `logf(`, `sqrtf(`, `sinf(`, `cosf(` 等
3. 标记为 P0（会导致链接失败，kernel 无法编译）

**问题标记**:
```json
{
  "id": "XXX-09",
  "rule": "TopN2.4 - stdlib数学函数禁用",
  "severity": "P0",
  "location": "op_kernel/xxx.cpp:42",
  "code_snippet": "float corr = __builtin_expf(diff);",
  "description": "NPU kernel 不链接 C stdlib，调用 expf 将导致 undefined symbol 链接错误",
  "fix_suggestion": "用 AscendC::Exp(localTensor, localTensor, n) 向量指令替代，将标量值写入 SetValue(0,...) 再 GetValue(0) 取回"
}
```

---

#### TopN 2.5 — PipeBarrier 命名空间错误 (WF-008)

**规则**: `PIPE_V`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_S` 等 pipe 枚举是全局枚举，**不**在 `AscendC::` 命名空间内。

**检测模式**:
```cpp
// ❌ 违规示例 — 编译报错: no member named 'PIPE_V' in namespace 'AscendC'
AscendC::PipeBarrier<AscendC::PIPE_V>();
AscendC::PipeBarrier<AscendC::PIPE_MTE2>();

// ✓ 正确示例
AscendC::PipeBarrier<PIPE_V>();
AscendC::PipeBarrier<PIPE_MTE2>();
```

**扫描逻辑**:
1. 查找所有 `PipeBarrier<AscendC::` 模式
2. 标记为 P0（编译直接失败）

**问题标记**:
```json
{
  "id": "XXX-10",
  "rule": "TopN2.5 - PipeBarrier命名空间",
  "severity": "P0",
  "location": "op_kernel/xxx.cpp:88",
  "code_snippet": "AscendC::PipeBarrier<AscendC::PIPE_V>();",
  "description": "PIPE_V 是全局枚举，不属于 AscendC:: 命名空间",
  "fix_suggestion": "改为 AscendC::PipeBarrier<PIPE_V>()"
}
```

---

#### TopN 2.6 — Tiling header 字段名与 kernel 不一致 (WF-009)

**规则**: `op_host/*_tiling.h` 中 `TILING_DATA_FIELD_DEF` 的字段名必须与 `op_kernel/*.cpp` 中 `tilingData.fieldName` 引用完全一致（大小写敏感）。

**检测模式**:
```cpp
// ❌ 违规示例 — tiling.h 定义 rowsPerCoreHead，kernel 引用 rowsPerHeadCore
// tiling.h:
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreHead);   // ← 定义
// op_kernel/xxx.cpp:
tilingData.rowsPerHeadCore;  // ← 引用（字段名不同，编译错误）

// ✓ 正确示例 — 字段名完全一致
TILING_DATA_FIELD_DEF(uint32_t, rowsPerHeadCore);   // tiling.h
tilingData.rowsPerHeadCore;  // op_kernel (一致)
```

**扫描逻辑**:
1. 从 `op_host/*_tiling.h` 提取所有 `TILING_DATA_FIELD_DEF(type, fieldName)` 中的 `fieldName`
2. 从 `op_kernel/*.cpp`/`.h` 中提取所有 `tilingData\.(\w+)` 引用
3. 对比两个集合：存在于引用集合但不在定义集合中的字段名 → 标记为 P0

**问题标记**:
```json
{
  "id": "XXX-11",
  "rule": "TopN2.6 - Tiling字段名不一致",
  "severity": "P0",
  "location": "op_kernel/xxx.cpp:67 ↔ op_host/xxx_tiling.h:12",
  "code_snippet": "tilingData.rowsPerHeadCore  (tiling.h defines: rowsPerCoreHead)",
  "description": "kernel 引用的 tiling 字段名与 tiling.h 定义不匹配",
  "fix_suggestion": "统一字段名：将 tiling.h 中 rowsPerCoreHead 改为 rowsPerHeadCore，或反向修改 kernel 引用"
}
```

---

## 4. Examples

### 示例输出 - JSON 格式

```json
{
  "operator_name": "ai_infra_sinkhorn_grad",
  "scan_time": "2026-02-11T15:30:00",
  "files_scanned": [
    "op_kernel/ai_infra_sinkhorn_grad_generalized.h",
    "op_host/ai_infra_sinkhorn_grad_tiling.cpp"
  ],
  "issues": [
    {
      "id": "SINK-01",
      "rule": "红线1.5 - 空指针保护",
      "severity": "P0",
      "location": "op_kernel/ai_infra_sinkhorn_grad_generalized.h:63-73",
      "function": "Init",
      "code_snippet": "pipe_ = tPipe;\ntilingData_ = tilingData;\ntotalLength_ = tilingData->totalLength;",
      "description": "Init 函数未检查 tPipe 和 tilingData 是否为 null",
      "fix_suggestion": "在解引用前添加:\nif (tPipe == nullptr) return;\nif (tilingData == nullptr) return;"
    },
    {
      "id": "SINK-02",
      "rule": "TopN2.1 - 特殊值处理",
      "severity": "P1",
      "location": "op_kernel/ai_infra_sinkhorn_grad_generalized.h:350,390",
      "function": "colNormGrad, rowNormGrad",
      "code_snippet": "Div(gradLocal[i * nt], subResultTensor[i * nt], sumLocal, nt);",
      "description": "除法操作未处理 sumLocal 可能为 0 或 NaN",
      "fix_suggestion": "在文档 docs/npu_sinkhorn_grad.md 中明确输入约束:\n- sum_out 必须为正数，不能包含 0/NaN/Inf"
    },
    {
      "id": "SINK-03",
      "rule": "TopN2.2 - 输入校验",
      "severity": "P2",
      "location": "op_host/ai_infra_sinkhorn_grad_tiling.cpp:285-306",
      "function": "CheckInputShape",
      "code_snippet": "nSize_ = gradOutputShape_->GetDim(DIM_1);",
      "description": "nSize_ 从外部输入获取后未校验合法性",
      "fix_suggestion": "添加范围检查:\nconstexpr int64_t MAX_N_SIZE = 163;\nconstexpr int64_t MIN_N_SIZE = 1;\nif (nSize_ < MIN_N_SIZE || nSize_ > MAX_N_SIZE) {\n    OP_LOGE(...);\n    return ge::GRAPH_FAILED;\n}"
    },
    {
      "id": "SINK-04",
      "rule": "代码规范 - 边界注释",
      "severity": "P3",
      "location": "op_kernel/ai_infra_sinkhorn_grad_generalized.h:346-351",
      "function": "colNormGrad",
      "code_snippet": "for (int i = 0; i < n_; ++i) {\n    Sub(subResultTensor[i * nt], ...);\n}",
      "description": "数组访问缺少边界安全注释",
      "fix_suggestion": "添加注释说明:\n// Buffer大小验证:\n// - gradLocal: nnt = n_ * nt 元素\n// - 循环访问: i ∈ [0, n_), 最大索引 = (n_-1)*nt + nt-1 = nnt - 1 ✓"
    }
  ],
  "summary": {
    "total_issues": 4,
    "by_severity": {
      "P0": 1,
      "P1": 1,
      "P2": 1,
      "P3": 1
    },
    "by_rule": {
      "红线1.5": 1,
      "TopN2.1": 1,
      "TopN2.2": 1,
      "代码规范": 1
    }
  }
}
```

### 示例输出 - Markdown 格式

```markdown
# ai_infra_sinkhorn_grad - 编码红线检查报告

**检查时间**: 2026-02-11 15:30
**扫描文件**: 2 个
**发现问题**: 4 个 (P0×1, P1×1, P2×1, P3×1)

---

## 问题清单

### 🔴 P0 - 极严重 (1个)

#### SINK-01: Init函数缺少空指针保护

**位置**: `op_kernel/ai_infra_sinkhorn_grad_generalized.h:63-73`
**违反规范**: 红线1.5 - 空指针保护

**问题描述**:
Init 函数接收 `tPipe` 和 `tilingData` 指针参数，但未进行 null 检查即直接解引用：

```cpp
__aicore__ inline void Init(...,
                           const AiInfraSinkhornGradTilingData *__restrict__ tilingData,
                           TPipe *tPipe)
{
    pipe_ = tPipe;         // ❌ 未检查 tPipe 是否为 null
    tilingData_ = tilingData;  // ❌ 未检查 tilingData 是否为 null
    totalLength_ = tilingData->totalLength;  // ❌ 直接解引用
}
```

**修复建议**:
```cpp
__aicore__ inline void Init(...) {
    // ✓ 添加空指针保护
    if (tPipe == nullptr) {
        return;
    }
    if (tilingData == nullptr) {
        return;
    }

    pipe_ = tPipe;
    tilingData_ = tilingData;
    totalLength_ = tilingData->totalLength;  // ✓ 安全
}
```

---

### 🟠 P1 - 严重 (1个)

#### SINK-02: colNormGrad/rowNormGrad 缺少特殊值处理

**位置**: `op_kernel/ai_infra_sinkhorn_grad_generalized.h:350,390`
**违反规范**: TopN2.1 - 特殊值处理

**问题描述**:
`colNormGrad` 和 `rowNormGrad` 函数中有除法操作，但未处理除数可能为 0 或 NaN 的情况：

```cpp
for (int i = 0; i < n_; ++i) {
    Div(gradLocal[i * nt], subResultTensor[i * nt], sumLocal, nt);
    // ❌ 如果 sumLocal 包含 0 值，会产生 Inf
    // ❌ 如果输入有 NaN，会传播到输出
}
```

**修复建议**:
在文档 `docs/npu_sinkhorn_grad.md` 中明确输入约束：

```markdown
## 输入约束

### 数据有效性要求
1. **sum_out**: 必须为正数，不能包含 0、NaN 或 Inf
2. **norm_out**: 必须为有限正数
3. **grad_output**: 建议不包含 NaN，否则会传播到输出
```

---

（后续 P2、P3 问题省略...）

---

## 检查总结

| 严重性 | 数量 | 规范分布 |
|--------|------|----------|
| **P0 - 极严重** | 1 | 红线1.5 ×1 |
| **P1 - 严重** | 1 | TopN2.1 ×1 |
| **P2 - 中等** | 1 | TopN2.2 ×1 |
| **P3 - 低** | 1 | 代码规范 ×1 |
| **总计** | **4** | - |

## 建议修复顺序

1. **立即修复 P0 问题** (SINK-01) - 可能导致崩溃
2. **24小时内修复 P1 问题** (SINK-02) - 可能导致数据错误
3. **1周内修复 P2 问题** (SINK-03) - 影响稳定性
4. **2周内修复 P3 问题** (SINK-04) - 代码可维护性
```

---

## 5. Dependencies

- **必需文件**:
  - 算子源代码（`op_kernel/`, `op_host/`）
  - 编码红线规范文档（`编码红线.md`）

- **可选文件**:
  - 算子文档（`docs/npu_*.md`）- 用于检查特殊值约束说明

---

## 6. Configuration

可在 `.ascendc-review.json` 中配置检查规则：

```json
{
  "rules": {
    "division_by_zero": {
      "enabled": true,
      "severity": "P0"
    },
    "array_bounds": {
      "enabled": true,
      "severity": "P0"
    },
    "overflow_protection": {
      "enabled": true,
      "severity": "P1"
    },
    "variable_initialization": {
      "enabled": true,
      "severity": "P1"
    },
    "null_pointer": {
      "enabled": true,
      "severity": "P0"
    },
    "special_values": {
      "enabled": true,
      "severity": "P1"
    },
    "input_validation": {
      "enabled": true,
      "severity": "P2"
    },
    "int64_offsets": {
      "enabled": true,
      "severity": "P2"
    }
  },
  "output": {
    "json": true,
    "markdown": true
  }
}
```

---

## 7. Known Limitations

1. **False Positives**:
   - 某些除零场景可能是开发者已验证安全的
   - 需要人工判断是否为真实问题

2. **漏报风险**:
    - 复杂的指针别名可能导致漏检
    - 宏定义中的代码无法完全分析

3. **责任边界**:
   - 不判断算法归一化链是否完整
   - 不证明分母不变量是否由前序步骤正确建立
   - TopN2.1 只检查表层特殊值风险和缺少输入约束/guard 的现象
   - 以上问题交由 `ascendc-algorithm-check.md`

4. **性能**:
    - 大型算子（>2000 行）检查时间较长
    - 建议拆分文件或使用增量检查

---

**Version**: 1.0.0
**Last Updated**: 2026-02-11
