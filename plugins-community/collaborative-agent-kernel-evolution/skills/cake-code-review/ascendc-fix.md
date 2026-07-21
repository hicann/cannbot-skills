# ascendc-fix

**Tagline**: 编码红线问题标准修复引擎

**Triggers**:
- 当 `cake-code-review` 主 skill 进入 Phase 4（Fix）时自动调用
- User 明确说"修复问题"时独立使用

---

## 1. Introduction

`ascendc-fix` 负责根据 `ascendc-review` 或 `ascendc-algorithm-check` 生成的问题清单，应用标准修复模式自动修复代码。

当问题来源于 `ascendc-algorithm-check.md` 时，本阶段同样负责按通用算法修复模式补齐缺失步骤、恢复不变量和统一归约结果消费方式。

**修复范围**：
- ✅ 7 大编码红线问题的标准修复
- ✅ 6 条 TopN 问题的规范化处理
- ✅ 自动生成详细整改报告

**输出格式**：
- 修复后的源代码文件
- Markdown 格式整改报告（修复前后对比）

---

## 2. Usage

### 根据问题清单修复

```
使用 ascendc-fix 修复 ai_infra_sinkhorn_grad 算子的问题（从 review/issues.json 读取）
```

### 修复特定问题

```
使用 ascendc-fix 仅修复 ai_infra_sinkhorn_grad 的 SINK-01 和 SINK-03 问题
```

### 批量修复

```
使用 ascendc-fix 修复 review/ 目录下所有算子的问题
```

---

## 3. How It Works

### 修复流程

```
1. 读取问题清单
   └─> 从 JSON 文件、`ascendc-review` 输出或 `ascendc-algorithm-check` 输出中获取

2. 按优先级排序
   └─> P0 → P1 → P2 → P3

3. 对每个问题应用修复模式
   └─> 定位到具体代码行
   └─> 匹配修复模式（见下文"标准修复模式"）
   └─> 生成修复代码
   └─> 添加安全注释

4. 更新相关文档
   └─> docs/*.md（添加输入约束说明）

5. 生成整改报告
   └─> Markdown 格式，包含修复前后对比
```

### 标准修复模式

#### 模式 1: 除零保护

**适用规则**: 红线1.1

**检测**:
```cpp
result = a / b;
remainder = a % b;
```

**修复**:
```cpp
// ✓ 添加除零保护
if (b == 0) {
    // 错误处理：返回错误码或默认值
    return ERROR_CODE;  // 或设置 result = 0;
}
result = a / b;
```

**特殊情况 - GetBlockNum()**:
```cpp
// 修复前
cubeDealnDPeCore_ = nD_ / GetBlockNum();

// 修复后
uint32_t blockNum = GetBlockNum();
if (blockNum == 0) {
    v1UsedCubeCoreNum_ = 0;
    cubeDealnDPeCore_ = nD_;
    return;
}
cubeDealnDPeCore_ = nD_ / blockNum;  // ✓ 安全
```

**特殊情况 - CeilDiv 函数**:
```cpp
// 修复前
inline int64_t CeilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;  // ❌ b=0 崩溃
}

// 修复后
inline int64_t CeilDiv(int64_t a, int64_t b) {
    if (b == 0) {  // ✓ 添加保护
        return 0;
    }
    return (a + b - 1) / b;
}
```

#### 模式 2: 数组越界保护

**适用规则**: 红线1.2

**检测**:
```cpp
for (uint32_t i = 0; i < count; i++) {
    gatherOffsetBuf_.SetValue(offset1++, value);
}
```

**修复**:
```cpp
for (uint32_t i = 0; i < count; i++) {
    // ✓ 添加边界检查
    if (offset1 >= hFusionBufLen_) {
        // 错误处理：记录日志或返回
        return;
    }
    gatherOffsetBuf_.SetValue(offset1++, value);
}
```

**添加边界安全注释**:
```cpp
// 【编码红线1.2修复】Buffer大小验证注释
// gatherOffsetBuf_ 大小: hFusionBufLen_ = VEC_DEAL_CHUNK × (N_ + N_ + N_×N_)
// 预期写入次数: VEC_DEAL_CHUNK × (N_ + N_ + N_×N_) = hFusionBufLen_
// 循环索引验证: offset1 ∈ [0, hFusionBufLen_)
// 最大索引 = hFusionBufLen_ - 1 ✓

uint32_t offset1 = 0;
for (uint32_t i = 0; i < VEC_DEAL_CHUNK; i++) {
    for (uint32_t j = 0; j < N_; j++) {
        if (offset1 >= hFusionBufLen_) return;
        gatherOffsetBuf_.SetValue(offset1++, ...);
    }
}
```

#### 模式 3: 溢出保护

**适用规则**: 红线1.3 + TopN2.3

**检测**:
```cpp
uint32_t globalUbOffset_;  // ❌ 应为 uint64_t
globalUbOffset_ = hFusionBufLen_ * sizeof(uint32_t);
globalUbOffset_ += hFusionBufLen_ * sizeof(P);
```

**修复**:
```cpp
// ✓ 第一步：改为 uint64_t
uint64_t globalUbOffset_;

// ✓ 第二步：添加溢出检查宏
#define SAFE_ADD_OFFSET(offset, increment) \
    do { \
        if (offset > UINT64_MAX - (increment)) { \
            return;  /* 溢出，返回错误 */ \
        } \
        offset += (increment); \
    } while(0)

// ✓ 第三步：使用宏保护累加操作
globalUbOffset_ = hFusionBufLen_ * sizeof(uint32_t);
SAFE_ADD_OFFSET(globalUbOffset_, hFusionBufLen_ * sizeof(P));
SAFE_ADD_OFFSET(globalUbOffset_, fusionSize_ * sizeof(P));

// ✓ 第四步：总量检查
if (globalUbOffset_ > FP32_BUF_SIZE) {
    // UB空间不足，报错
    return;
}
```

**特殊情况 - GM 内存偏移**:
```cpp
// 修复前
int colOffset = i * n_ * tAlign_;  // ❌ int 可能溢出

// 修复后
int64_t colOffset = static_cast<int64_t>(i) * n_ * tAlign_;  // ✓ int64_t
```

#### 模式 4: 变量初始化

**适用规则**: 红线1.4

**检测**:
```cpp
bool alphaBufInitialized_;  // ❌ 声明时未初始化
float scaleMean_;
```

**修复**:
```cpp
// ✓ 声明时显式初始化
bool alphaBufInitialized_ = false;
float scaleMean_ = 0.0f;
```

**删除未使用的变量**:
```cpp
// 修复前
float alphaPre = alphaGm_.GetValue(0);  // ❌ 读取后从未使用

// 修复后
// 删除此行
```

#### 模式 5: 空指针保护

**适用规则**: 红线1.5

**检测**:
```cpp
__aicore__ inline void Init(...,
                           const TilingData *tilingData,
                           TPipe *tPipe)
{
    pipe_ = tPipe;
    tilingData_ = tilingData;
    totalLength_ = tilingData->totalLength;  // ❌ 直接解引用
}
```

**修复**:
```cpp
__aicore__ inline void Init(...,
                           const TilingData *tilingData,
                           TPipe *tPipe)
{
    // ✓ 添加空指针保护
    if (tPipe == nullptr) {
        return;  // 或通过错误码通知调用方
    }
    if (tilingData == nullptr) {
        return;
    }

    pipe_ = tPipe;
    tilingData_ = tilingData;
    totalLength_ = tilingData->totalLength;  // ✓ 安全
}
```

#### 模式 6: 特殊值处理（文档）

**适用规则**: TopN2.1

**检测**:
```cpp
Div(gradLocal[offset], subResultTensor[offset], sumLocal[i], tAlign_);
// 除法操作但未说明输入约束
```

**修复**:
在算子文档（`docs/npu_*.md`）中添加"输入约束"章节：

```markdown
## 输入约束

### 数据有效性要求

1. **sum_out**: 必须为正数，不能包含 0、NaN 或 Inf
   - 如果前向 Sinkhorn 迭代收敛性不佳，可能导致 sum_out 接近 0
   - 建议在调用前对 sum_out 进行数值稳定性检查

2. **norm_out**: 必须为有限正数

3. **grad_output**: 建议不包含 NaN，否则会传播到输出

### 特殊值处理说明

- 本算子**不支持** sum_out 中有 0 值的输入
- 如果输入包含 NaN，会通过计算链传播到输出
- 除法操作依赖前向保证除数 > 0
```

#### 模式 7: 输入校验

**适用规则**: TopN2.2

**检测**:
```cpp
// Tiling 中从外部获取参数但未校验
nSize_ = gradOutputShape_->GetDim(DIM_2);
D_ = hInGradTensor->GetStorageShape().GetDim(INDEX_D);
```

**修复**:
```cpp
// ✓ 获取参数
nSize_ = gradOutputShape_->GetDim(DIM_2);

// ✓ 添加合法性校验
constexpr int64_t MAX_N_SIZE = 163;  // 基于 UB 内存限制计算
constexpr int64_t MIN_N_SIZE = 1;

if (nSize_ < MIN_N_SIZE || nSize_ > MAX_N_SIZE) {
    OP_LOGE(context_,
        "Invalid nSize=%ld. Expected range [%ld, %ld] based on UB size limit.",
        nSize_, MIN_N_SIZE, MAX_N_SIZE);
    return ge::GRAPH_FAILED;
}

// ✓ D_ 参数校验
if (D_ == 0) {
    OP_LOGE(context_->GetNodeName(),
        "Invalid D=%lu. D must be positive.", D_);
    return ge::GRAPH_FAILED;
}

// ✓ 合理范围检查（根据实际业务需求调整）
constexpr uint64_t MAX_D = 8192;  // 示例值
if (D_ > MAX_D) {
    OP_LOGE(context_->GetNodeName(),
        "D=%lu exceeds maximum allowed value %lu.", D_, MAX_D);
    return ge::GRAPH_FAILED;
}
```

**注意**: 如果后续 UT 测试发现校验参数过严，需要根据实际测试用例调整范围。

#### 模式 8: 硬编码常数消除

**适用规则**: TopN2.3 + 代码规范

**检测**:
```cpp
uint32_t v2Elements = 1024 * 128 * 24 * 2 +  // ❌ 硬编码魔术数字
                      1024 * 128 * 24 * 2 +
                      2 * 1024 * 1024;
```

**修复**:
```cpp
// ✓ 使用常量替换硬编码
constexpr uint64_t SINGLE_M = 1024;
constexpr uint64_t ND_BLOCK_SIZE = 128;
constexpr uint64_t NUM_BUFFERS = 2;
constexpr uint64_t NUM_CORES = 24;

uint64_t xRsGradMmSize = SINGLE_M * ND_BLOCK_SIZE * NUM_BUFFERS * NUM_CORES;
uint64_t xRsSize = SINGLE_M * ND_BLOCK_SIZE * NUM_BUFFERS * NUM_CORES;
uint64_t reserveSize = 2 * 1024 * 1024;

uint64_t v2Elements = xRsGradMmSize + xRsSize + reserveSize;
```

#### 模式 9: 算法级修复模式

**适用来源**: `ascendc-algorithm-check.md`

**原则**:
- 优先恢复算法步骤完整性，而不是用 operator-specific hack 掩盖症状
- 优先补齐不变量建立/消费链，而不是把问题下推给 `ascendc-verify`
- 不要求数值基准测试或参考实现比对；只要求逻辑闭环、编译通过、UT/ST 回归通过

**常见修复类别**:

1. **归一化链补齐**
   - 若存在 `max -> exp -> sum -> divide` 或类似归一化链，缺任一步时补齐完整链路
   - Softmax 示例: 已有 `ReduceMax` 和 `Exp`，但缺少对 `ReduceSum` 结果的消费时，应补上 divide/scale 步骤，而不是只加注释

2. **分母安全与不变量恢复**
   - 将分母对应的不变量在生成点或消费点显式化，例如 `sum > 0`、`norm finite`、`count >= 1`
   - 若算法依赖前序阶段保证分母非零，代码或文档中要明确该约束，必要时增加守卫或早退

3. **迭代更新补齐**
   - 检查循环体内是否遗漏核心状态更新、收敛计数、前后缓冲切换、残差刷新
   - SinkhornKnopp 示例: 若只更新 row scaling 而未消费列归一化结果，修复应补齐缺失更新链，而不是局部改常量

4. **归约结果消费一致化**
   - `ReduceSum`、`ReduceMax`、`WholeReduce*` 等结果必须在后续逻辑中被一致消费
   - 若结果仅写入临时 buffer 但后续未使用，修复应补齐读取和应用，而不是删除归约来消除表面问题

**整改报告补充格式**:

```markdown
#### ALG-XX: [算法问题标题]

**严重性**: P0/P1/P2
**违反规范**: Algorithm Correctness Check - [类别]
**代码位置**: `[文件路径]:[行号]`

##### 问题分析
- 缺失的算法步骤: [描述]
- 被破坏的不变量: [描述]

##### 修复方案
- 补齐步骤: [描述]
- 保持的边界条件: [描述]

##### 验证范围
- 编译: 通过
- UT/ST 回归: 通过或待执行
```

---

## 4. Examples

### 示例 1: 修复除零问题（MHC-01）

**输入问题清单**:
```json
{
  "id": "MHC-01",
  "rule": "红线1.1 - 除零保护",
  "severity": "P0",
  "location": "op_kernel/*.h:440-446",
  "code_snippet": "cubeDealnDPeCore_ = nD_ / GetBlockNum();"
}
```

**修复过程**:
1. 定位到文件和行号
2. 读取上下文代码
3. 应用修复模式 1（除零保护）
4. 生成修复代码

**修复前代码**:
```cpp
__aicore__ inline void InitStage2()
{
    cubeDealnDPeCore_ = nD_ / GetBlockNum();  // ❌ 未检查返回值

    if (cubeDealnDPeCore_ == 0) {
        cubeDealnDPeCore_ = nD_;
        v1UsedCubeCoreNum_ = 1;
    }
    // ...
}
```

**修复后代码**:
```cpp
__aicore__ inline void InitStage2()
{
    // 【编码红线1.1修复】除零保护
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0) {
        // 错误处理：设置默认值
        v1UsedCubeCoreNum_ = 0;  // 标记无可用核心
        cubeDealnDPeCore_ = nD_;
        return;  // 提前退出
    }

    cubeDealnDPeCore_ = nD_ / blockNum;  // ✓ 安全

    if (cubeDealnDPeCore_ == 0) {
        cubeDealnDPeCore_ = nD_;
        v1UsedCubeCoreNum_ = 1;
    } else {
        cubeDealnDPeCore_ = CeilAlign(cubeDealnDPeCore_, uint32_t(128));

        // 【编码红线1.1修复】再次检查对齐后是否为 0
        if (cubeDealnDPeCore_ == 0) {
            v1UsedCubeCoreNum_ = 0;
            return;
        }

        v1UsedCubeCoreNum_ = CeilDiv(nD_, cubeDealnDPeCore_);  // ✓ 安全
    }
}
```

**整改报告片段**:
```markdown
### MHC-01: GetBlockNum()除零保护缺失

**严重性**: P0 - 极严重
**违反规范**: 红线1.1 - 除零保护
**代码位置**: `op_kernel/ai_infra_manifold_constrained_hyper_connection_pre_grad.h:440-446`

#### 问题分析

在 `InitStage2()` 函数中，直接使用 `GetBlockNum()` 的返回值作为除数，未进行非零检查：

```cpp
cubeDealnDPeCore_ = nD_ / GetBlockNum();  // ❌ 崩溃风险
```

**风险说明**:
- GetBlockNum()在异常情况下可能返回0
- 1C2V模式下，如果核数配置错误，会导致崩溃

#### 修复方案

**修复前**:
```cpp
cubeDealnDPeCore_ = nD_ / GetBlockNum();
```

**修复后**:
```cpp
uint32_t blockNum = GetBlockNum();
if (blockNum == 0) {
    v1UsedCubeCoreNum_ = 0;
    cubeDealnDPeCore_ = nD_;
    return;
}
cubeDealnDPeCore_ = nD_ / blockNum;  // ✓ 安全
```

#### 修改影响

- **功能影响**: 无，纯防御性编程
- **性能影响**: 可忽略不计（仅增加1次条件判断）
- **配套修改**: Process()函数增加v1UsedCubeCoreNum_检查
```

### 示例 2: 修复输入校验问题（SINK-03）

**输入问题清单**:
```json
{
  "id": "SINK-03",
  "rule": "TopN2.2 - 输入校验",
  "severity": "P2",
  "location": "op_host/ai_infra_sinkhorn_grad_tiling.cpp:285-306",
  "code_snippet": "nSize_ = gradOutputShape_->GetDim(DIM_2);"
}
```

**修复过程**:
1. 定位到 Tiling 文件的 CheckInputShape 函数
2. 应用修复模式 7（输入校验）
3. 根据 UB 内存限制计算合法范围
4. 添加校验逻辑

**修复前代码**:
```cpp
ge::graphStatus CheckInputShape()
{
    if (isTShape_) {
        tSize_ = gradOutputShape_->GetDim(DIM_0);
        nSize_ = gradOutputShape_->GetDim(DIM_1);  // ❌ 未校验
        totalLength_ = tSize_;
    } else {
        bSize_ = gradOutputShape_->GetDim(DIM_0);
        sSize_ = gradOutputShape_->GetDim(DIM_1);
        nSize_ = gradOutputShape_->GetDim(DIM_2);  // ❌ 未校验
        totalLength_ = bSize_ * sSize_;
    }

    // 后续直接使用 nSize_
    numIters_ = normOutDim0 / 2;
    // ...
    return ge::GRAPH_SUCCESS;
}
```

**修复后代码**:
```cpp
ge::graphStatus CheckInputShape()
{
    if (isTShape_) {
        tSize_ = gradOutputShape_->GetDim(DIM_0);
        nSize_ = gradOutputShape_->GetDim(DIM_1);
        totalLength_ = tSize_;
    } else {
        bSize_ = gradOutputShape_->GetDim(DIM_0);
        sSize_ = gradOutputShape_->GetDim(DIM_1);
        nSize_ = gradOutputShape_->GetDim(DIM_2);
        totalLength_ = bSize_ * sSize_;
    }

    // 【TopN2.2修复】nSize合法性校验
    // 根据UB内存限制,n不能过大
    // 公式: (7*n*n + 3*n) * 4B < 190KB
    // 解得: n_max ≈ 163
    constexpr int64_t MAX_N_SIZE = 163;
    constexpr int64_t MIN_N_SIZE = 1;  // 允许n=1的边界场景

    if (nSize_ < MIN_N_SIZE || nSize_ > MAX_N_SIZE) {
        OP_LOGE(context_,
            "Invalid nSize=%ld. Expected range [%ld, %ld] based on UB size limit.",
            nSize_, MIN_N_SIZE, MAX_N_SIZE);
        return ge::GRAPH_FAILED;
    }

    // ... 后续代码
    return ge::GRAPH_SUCCESS;
}
```

**整改报告片段**:
```markdown
### SINK-03: nSize范围校验缺失

**严重性**: P2 - 中等
**违反规范**: TopN2.2 - 输入校验
**代码位置**: `op_host/ai_infra_sinkhorn_grad_tiling.cpp:285-306`

#### 问题分析

在 `CheckInputShape()` 函数中，从外部输入获取 `nSize_` 后未进行合法性校验：

```cpp
nSize_ = gradOutputShape_->GetDim(DIM_2);  // ❌ 直接使用
```

**风险说明**:
- Sinkhorn算法对n×n矩阵有特定要求
- 过大的n会导致UB内存不足
- 过小的n可能违反算法前提

#### 修复方案

根据 UB 内存限制计算合法范围：

**公式推导**:
- UB 使用量: (7*n*n + 3*n) * 4B < 190KB
- 解得: n_max ≈ 163

**修复代码**:
```cpp
constexpr int64_t MAX_N_SIZE = 163;
constexpr int64_t MIN_N_SIZE = 1;

if (nSize_ < MIN_N_SIZE || nSize_ > MAX_N_SIZE) {
    OP_LOGE(context_,
        "Invalid nSize=%ld. Expected range [%ld, %ld].",
        nSize_, MIN_N_SIZE, MAX_N_SIZE);
    return ge::GRAPH_FAILED;
}
```

#### 验证结果

- **UT 测试**: 34/34 通过（包括 n=1 边界场景）
- **ST 测试**: 2/2 通过
```

---

## 5. Output Format

每次修复后生成详细整改报告（Markdown 格式），包含以下章节：

### 报告模板

```markdown
# [算子名] - 编码红线整改报告

**整改日期**: 2026-02-11
**整改人**: Claude Code
**整改范围**: [算子路径]

---

## 一、整改概览

| 问题总数 | P0 | P1 | P2 | P3 |
|---------|----|----|----|----|
| **X个** | X | X | X | X |

**涉及规范**:
- 红线1.1 - 除零保护 (X个)
- 红线1.2 - 数组越界 (X个)
- 红线1.3 - 溢出保护 (X个)
- TopN2.2 - 输入校验 (X个)
- ...

---

## 二、问题修复详情

### 🔴 P0 - 极严重问题

#### [问题ID]: [问题标题]

**严重性**: P0 - 极严重
**违反规范**: [规范编号]
**代码位置**: `[文件路径]:[行号]`

##### 问题分析

[详细描述问题]

**风险说明**:
- [风险1]
- [风险2]

##### 修复方案

**修复前**:
```cpp
[原代码]
```

**修复后**:
```cpp
[修复后代码，带注释]
```

##### 修改影响

- **功能影响**: [说明]
- **性能影响**: [说明]
- **配套修改**: [如有]

---

（后续P1、P2、P3问题...）

---

## 三、文件修改清单

| 文件路径 | 修改行数 | 新增注释 | 主要修改 |
|---------|---------|---------|----------|
| `op_kernel/*.h` | X行 | X行 | 除零保护、边界检查 |
| `op_host/*_tiling.cpp` | X行 | X行 | 输入校验、溢出保护 |
| `docs/npu_*.md` | X行 | - | 输入约束说明 |

---

## 四、整改成果

### 代码质量提升

#### 修复前（存在严重风险）:
```cpp
// ❌ 除零崩溃风险
cubeDealnDPeCore_ = nD_ / GetBlockNum();

// ❌ 数组越界风险
gatherOffsetBuf_.SetValue(offset1++, ...);
```

#### 修复后（安全可靠）:
```cpp
// ✓ 除零保护
uint32_t blockNum = GetBlockNum();
if (blockNum == 0) return;
cubeDealnDPeCore_ = nD_ / blockNum;

// ✓ 边界检查
if (offset1 >= hFusionBufLen_) return;
gatherOffsetBuf_.SetValue(offset1++, ...);
```

### 统计数据

```
修改文件数: X个
修改代码行数: X行
新增安全注释: X行
删除不安全代码: X行
```

---

## 五、后续建议

1. **编译验证**: 运行 `bash build.sh -c "ascend910b"` 确认编译通过
2. **UT 验证**: 运行单元测试确认功能正确性
3. **ST 验证**: 运行集成测试验证端到端流程
4. **Code Review**: 提交前进行同行评审

---

**报告生成时间**: 2026-02-11 15:30
**报告版本**: v1.0
```

---

## 6. Dependencies

- **必需输入**:
  - 问题清单 (JSON 或来自 ascendc-review)
  - 算子源代码

- **可选输入**:
  - 配置文件 (`.ascendc-review.json`)

---

## 7. Configuration

可在 `.ascendc-review.json` 中配置修复行为：

```json
{
  "fix": {
    "auto_fix_p0": true,         // 自动修复 P0 问题
    "auto_fix_p1": true,         // 自动修复 P1 问题
    "auto_fix_p2": false,        // P2 问题需用户确认
    "auto_fix_p3": false,        // P3 问题需用户确认
    "add_comments": true,        // 自动添加安全注释
    "update_docs": true,         // 自动更新文档
    "backup_original": true      // 修复前备份原文件
  }
}
```

---

## 8. Known Limitations

1. **复杂逻辑**:
   - 某些复杂的指针操作可能需要人工修复
   - 自动修复可能不适用所有场景

2. **上下文理解**:
   - 修复时依赖局部代码上下文
   - 可能需要人工调整以适应全局逻辑

3. **参数调优**:
   - 校验参数（如 MIN_N_SIZE、MAX_D）可能需要根据 UT 结果调整
   - 建议修复后立即运行验证

---

**Version**: 1.0.0
**Last Updated**: 2026-02-11
