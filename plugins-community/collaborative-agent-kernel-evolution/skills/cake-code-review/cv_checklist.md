# CV 算子专项检查清单

> **触发条件**：源代码含 `ASCEND_IS_AIC`、`ASCEND_IS_AIV` 或 `BlockMmad` 关键字时，自动追加本检查
> **执行方式**：在常规 7 大红线检查完成后，追加运行本清单，结果合并到整改报告

---

## CV-P0：必须修复（数据竞争 / 死锁）

### CV-P0-1　ASCEND_IS_AIV 守卫
**检查**：AIV-only kernel 入口是否有 `if ASCEND_IS_AIV` 守卫
```bash
grep -n "ASCEND_IS_AIV\|ASCEND_IS_AIC" op_kernel/*.cpp
```
**判定**：若 kernel 为 AIV-only（无 `BlockMmad`），但入口没有 `if ASCEND_IS_AIV` → **违规**
**后果**：AIC 子核也执行写操作，输出大量相同行（→ H20）
**修复**：
```cpp
extern "C" __global__ __aicore__ void my_kernel(...) {
    GET_TILING_DATA(t, tiling);
    if ASCEND_IS_AIV {   // 注意：无括号
        MyOp op; op.Init(...); op.Process();
    }
}
```

---

### CV-P0-2　AIV 侧禁止 PipeBarrier\<PIPE_ALL\>
**检查**：AIV 分支内是否出现 `PipeBarrier<PIPE_ALL>` 或 `PipeBarrier<PIPE_FIX>`
```bash
grep -n "PIPE_ALL\|PIPE_FIX" op_kernel/*.cpp
```
**判定**：在 `if ASCEND_IS_AIV` 块内出现上述调用 → **违规**
**后果**：PIPE_ALL 包含 PIPE_FIX，AIV 侧等待 AIC BlockMmad，永久死锁（→ H21）
**修复**：AIV 侧只用 `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`，不用 `PIPE_ALL`/`PIPE_FIX`

---

### CV-P0-3　CrossCoreSetFlag\<PIPE_FIX\> 必须有 BlockMmad 前置
**检查**：每个 `CrossCoreSetFlag<..., PIPE_FIX>` 调用之前是否有 `matmul.Iterate()` 或 `blockMmad(...)` 等 BlockMmad 操作
```bash
grep -n "CrossCoreSetFlag\|Iterate\|blockMmad" op_kernel/*.cpp
```
**判定**：存在 AIC 路径（含条件分支）在 SetFlag<PIPE_FIX> 之前无 BlockMmad → **违规**
**后果**：PIPE_FIX 管道为空，AIV 侧 WaitFlag 永久阻塞（→ H22）
**修复**：每条到达 SetFlag<PIPE_FIX> 的路径都必须有 BlockMmad 前置；或改用 `PIPE_MTE3`

---

### CV-P0-4　KERNEL_TASK_TYPE_DEFAULT 宏缺失
**检查**：CV 算子 OpDef 是否包含 `KERNEL_TASK_TYPE_DEFAULT` 宏注册
```bash
grep -rn "KERNEL_TASK_TYPE_DEFAULT\|kernel_task_type" op_host/*.cpp
```
**判定**：CV 算子（含 `ASCEND_IS_AIC` 分支）但 OpDef 中无 `KERNEL_TASK_TYPE_DEFAULT` 声明 → **违规**
**后果**：框架不分配 AIC 核，Cube 侧整体不执行；manifest 中 aicCoreNum=0，等效于 fixpipe 退化或全零输出（3/7 算子曾中招）
**修复**：
```cpp
// op_host/op_custom.cpp 中 OpDef 注册时添加：
REG_OP(OpCustom)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .KERNEL_TASK_TYPE_DEFAULT(AscendC::KERNEL_TYPE_AIR_CPU_DEFAULT)  // 必须添加
    .OP_END_FACTORY_REG(OpCustom);
```

---

### CV-P0-5　Resource 对象多 Phase 生命周期冲突
**检查**：KernelCube 类中 `Arch::Resource<ArchTag>` 是否在每次 `Process()` 调用中局部创建（而非成员变量跨 phase 复用）
```bash
grep -n "Resource\|resource" op_kernel/*.cpp
```
**判定**：`resource` 声明为类成员变量，或在循环外创建后在多个 `blockMmad(...)` 调用间复用 → **违规**
**后果**：Resource 管理 L0B 等内部缓冲区生命周期；跨 phase 复用导致读写冲突 → AIC 死锁或精度错误（3/7 算子曾中招）
**修复**：
```cpp
// WRONG: resource as class member, reused across phases
class KernelCube {
    Arch::Resource<ArchTag> resource;  // 不要这样
    ...
};

// CORRECT: create locally in each Process() or each phase block
void Process() {
    for (auto [m, n] : tileIterator) {
        Arch::Resource<ArchTag> resource;    // 每次迭代局部创建
        BlockMmad blockMmad(resource);
        blockMmad(gmA[offsetA], layoutA, gmB[offsetB], layoutB, gmC[offsetC], layoutC, tileShape);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(matmulReady);
    }
}
```

---

## CV-P1：严重（静默精度错误 / 大模型越界）

### CV-P1-1　bf16 BlockMmad enableUnitFlag
**检查**：若 `ElementA` 或 `ElementB` 为 `bfloat16_t`，确认 `enableUnitFlag = true`
```bash
grep -n "ElementA\|ElementB\|enableUnitFlag\|bfloat16" op_kernel/*.cpp
```
**判定**：dtype 为 bf16 但 `enableUnitFlag = false` 或未设置 → **违规**
**后果**：AtlasA2 上 bf16 矩阵乘静默精度错误（→ H23）
**修复**：
```cpp
constexpr bool enableUnitFlag = true;   // bf16 必须为 true
using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
```

---

### CV-P1-2　大模型专家权重偏移使用 uint64_t
**检查**：含 `expertId * K * N` 或 `expertId * hidden` 类乘法时，确认使用 `uint64_t` 转型
```bash
grep -n "expertId\|expert_id\|wBase\|weightOffset" op_kernel/*.cpp
```
**判定**：偏移计算中乘积可能超过 4GB（K\*N > 4B），但使用 `uint32_t` 或隐式 `int` → **违规**
**后果**：整数溢出，越界访问（K=7168, N=4096 时单专家 ≈ 58MB bf16，8 专家总量超 int32 范围）
**修复**：
```cpp
uint64_t wBase = (uint64_t)expertId * (uint64_t)K * (uint64_t)N;
```

---

### CV-P1-3　GM→UB 数据加载使用 VECIN TQue
**检查**：Vector 分支中 GM→UB 数据加载（含 fp16 辅助数据 gamma/cos/sin）是否通过 `TQue<VECIN>` + `EnQue/DeQue`

> API 选择规则详见：`skills/cake-code-review/api-best-references/api-buffer.md`

**判定**：直接 DataCopy 到 TBuf 后无对应 TQue 管理 → **可疑**（需结合 PipeBarrier 使用情况判断）
**后果**：数据搬运与计算无正确同步，读到未完成数据（→ H10 变体）

---

### CV-P1-4　ReduceMax/ReduceSum tmpBuffer 类型匹配
**检查**：ReduceMax/ReduceSum 第三个参数（tmpBuffer）类型是否与 src 相同

> 完整约束见：`skills/cake-code-review/api-best-references/api-reduce.md`

```bash
grep -n "ReduceMax\|ReduceSum\|ReduceMin" op_kernel/*.cpp
```
**判定**：tmpBuffer 为 `uint8_t` 或其他与 src 不同的类型 → **违规**
**后果**：静默精度错误或编译失败（→ H19）

---

### CV-P1-5　BlockMmad N 维度 ≤ 256 约束
**检查**：`L1Shape` 的 N 维度（`L1Shape<M, N, K>` 第二参数）是否满足 ≤ 256
```bash
grep -n "L1Shape\|GemmShape" op_kernel/*.cpp
```
**判定**：`L1Shape<M, N, K>` 中 N > 256 → **违规**
**后果**：Atlas A2 硬件约束，BlockMmad N 维度超限 → kernel 挂起或编译期异常（症状类似 hang）
**修复**：将 `L1_N` 限制在 ≤ 256；如 N > 256 则通过外层循环分块：
```cpp
// WRONG:
using L1Shape = GemmShape<128, 512, 256>;  // N=512 超限

// CORRECT:
using L1Shape = GemmShape<128, 256, 256>;  // N≤256
// 外层 N 循环 step = 256 分块
```

---

## CV-P2：中等（可维护性 / 性能）

### CV-P2-1　group_list 不在 kernel 内 GM scalar read
**检查**：kernel 内是否出现 `gmGroupList.GetValue(expertId)` 或类似 GM scalar 读取模式
```bash
grep -n "GetValue\|groupList\|group_list" op_kernel/*.cpp
```
**判定**：在 kernel 循环内对 GM tensor 逐元素 GetValue → **建议优化**
**后果**：DMA pipeline stall，性能严重退化
**修复**：在 host TilingFunc 中将 `groupCounts[E]`、`groupOffsets[E]` 写入 tiling struct，kernel 直接读常量

---

### CV-P2-2　UB 预算检查（Atlas A2 ≤ 192KB/sub-block）
**检查**：所有 `pipe.InitBuffer(...)` 的 size 参数之和 / VEC_NUM 是否 ≤ 192KB

> 内存管理规则见：`skills/cake-code-review/api-best-references/api-buffer.md`

```bash
grep -n "InitBuffer" op_kernel/*.cpp
```
**判定**：sum(buffer sizes) / VEC_NUM > 196608 字节 → **违规**
**后果**：UB 溢出，kernel 编译时越界（→ H06 变体）
**修复**：缩减 `L1_N`、`computeLength` 或 tile size，直到满足约束

---

## 输出格式

CV 专项检查结果追加到整改报告末尾：

```
## CV 算子专项检查结果

| 检查项 | 状态 | 位置 | 说明 |
|--------|------|------|------|
| CV-P0-1 ASCEND_IS_AIV 守卫 | ✅ PASS / ❌ FAIL | 行号 | ... |
| CV-P0-2 AIV PIPE_ALL 禁用 | ✅ PASS / ❌ FAIL | 行号 | ... |
| CV-P0-3 PIPE_FIX BlockMmad 前置 | ✅ PASS / ❌ FAIL | 行号 | ... |
| CV-P0-4 KERNEL_TASK_TYPE_DEFAULT | ✅ PASS / ❌ FAIL | - | OpDef 注册检查 |
| CV-P0-5 Resource 多 Phase 生命周期 | ✅ PASS / ❌ FAIL | 行号 | ... |
| CV-P1-1 bf16 enableUnitFlag | ✅ PASS / N/A | - | 非 bf16 算子跳过 |
| CV-P1-2 uint64_t 专家偏移 | ✅ PASS / N/A | - | 无 expertId 跳过 |
| CV-P1-3 VECIN TQue 数据加载 | ✅ PASS / ⚠️ WARN | - | ... |
| CV-P1-4 Reduce tmpBuffer 类型 | ✅ PASS / ❌ FAIL | 行号 | ... |
| CV-P1-5 BlockMmad N ≤ 256 | ✅ PASS / ❌ FAIL | - | 实测 L1_N=XXX |
| CV-P2-1 group_list host 预处理 | ✅ PASS / N/A | - | ... |
| CV-P2-2 UB 预算 ≤ 192KB | ✅ PASS / ❌ FAIL | - | 实测 XXX KB |
```
