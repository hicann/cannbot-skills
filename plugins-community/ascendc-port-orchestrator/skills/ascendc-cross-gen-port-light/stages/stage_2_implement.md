# 阶段 2：代码改造（含 API 差异适配）

**前置条件**：`[GATE-1] LEVEL=L1/L2/L3` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/impl/api-diff-guide.md` | `[LOADED] api-diff-guide` |
| 2 | `references/impl/cube-migration-guide.md`（**cube 类算子** MUST，见 Step 2.1） | `[LOADED] cube-migration-guide` |

**★ 本文件 MUST 在代码改造前加载。** 阶段 1 扫描出的 API 差异风险项，在此阶段逐项适配。

## 路由：根据层级加载实现指南

```
GATE-1 LEVEL 值
  ├─ L1 → Step 2.1（L1 改造清单）+ Step 2.4（API 差异适配）
  ├─ L2 → Step 2.2（L2 改造清单）+ Step 2.4（API 差异适配）
  └─ L3 → Step 2.3（L3 改造清单）+ Step 2.4（API 差异适配）
```

---

## Step 2.1：L1 改造清单

**MUST READ**：`references/impl/l1-guide.md`
读取后输出：`[LOADED] l1-guide`

### 改造步骤

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `op_host/<op>_def.cpp` | 添加 `AddConfig("ascend950", aicore_config)`（复用 910b 配置） |
| 2 | `op_host/config/ascend950/` | 创建目录，复制 `binary.json` + `simplified_key.ini` |

**L1 不需要做的事**：
- 不创建 `_apt.cpp`
- 不创建 `arch35/` 目录
- 不修改 Tiling
- 不修改 kernel 代码
- 不修改 CMakeLists.txt

**L1 适用条件**：纯 INT32 / 无浮点 / 无 DataCopyPad / UB 使用量小 / kernel 无 `__CCE_AICORE__ == 220` 条件分支。
如果 kernel 中有 V220 guards 或有浮点计算，升级为标准 L1（见 `references/impl/l1-guide.md`）。

**cube 类算子**（kernel 含 Mmad/LoadData/Fixpipe 等 cube API）：追加 MUST READ `references/impl/cube-migration-guide.md`，输出 `[LOADED] cube-migration-guide`——L1 对 cube 类算子的改造范围仅 host 侧（_def.cpp / config / 入口分发，见 cube-guide 改动 1/8），kernel 侧 A5 差异不在 L1 范围；若评估后需改 kernel，按 stage_1 定级升级到 L2。

### 标准 L1（有 V220 guards 时）

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `op_host/<op>_def.cpp` | 添加独立 `OpAICoreConfig` + `opFile.value = "<op>_apt"` |
| 2 | `op_kernel/<op>_apt.cpp` | 创建 A5 入口文件，include 指向 `arch35/` |
| 3 | `op_kernel/arch35/` | 复制 kernel 头文件，移除 V220 guards |
| 4 | `op_host/config/ascend950/` | 创建目录，复制配置文件 |

**arch35/ 微调项**：

| 调整项 | 910b 版本 | 950 arch35/ 版本 |
|--------|-----------|-----------------|
| BF16 条件编译 | `#if __CCE_AICORE__ == 220` | 直接支持，移除条件编译 |
| BF16 架构保护 | `#if __NPU_ARCH__==3003 OR __NPU_ARCH__==3113` | 移除保护 |
| V220 includes | `#include "impl/dav_c220/..."` | 移除 |

---

## Step 2.2：L2 改造清单

### LOADED 检查点（MUST 全部完成后才能写代码）

**必加载（所有 L2 算子）**：

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/impl/l2-guide.md` | `[LOADED] l2-guide` |
| 2 | `references/impl/api-mapping.md` | `[LOADED] api-mapping` |
| 3 | `cannbot-skills/ops/ascendc-regbase-best-practice/references/regbase_development_guide.md` | `[LOADED] regbase_development_guide` |

**cube 类算子追加必读**（kernel 含 Mmad/LoadData/Fixpipe/DataCopyCO12DstParams/CrossCoreSetFlag 等 cube API 时）：

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 4 | `references/impl/cube-migration-guide.md` | `[LOADED] cube-migration-guide` |

**按需加载**：

| 文件 | LOADED Token | 何时需要 |
|------|-------------|---------|
| `references/impl/ub-budget-guide.md` | `[LOADED] ub-budget-guide` | 算子有归约操作或 UB 预算紧张时（cube 类算子 AIV 侧同样适用） |
| `references/impl/multi-stage-guide.md` | `[LOADED] multi-stage-guide` | 算子有 2 个以上算法阶段（如 Sort + Softmax + Sampling） |
| `references/impl/cube-debug-lessons.md` | `[LOADED] cube-debug-lessons` | cube 类算子测试/异常排查（阶段 4） |

**所有要求的 LOADED Token 输出后，方可进入代码改造。**

**cube 类算子的 L2 改造范围**：按 `cube-migration-guide.md`「性能关键 cube 算子的 L2 语义」三部分执行——① AIC 侧低阶直跑评估（改动 2/3/4，禁止默认沿用高阶 Matmul API，保留高阶时先核对双主模式约束）② 同步协议重写（改动 6）③ AIV 侧 Vector 路径评估/重写。第 ③ 部分使用本阶段的 l2-guide / api-mapping / regbase_development_guide（这些指南对 cube 类算子的适用边界见各自头部声明）；第 ①② 部分不使用本阶段 l2-guide 的改动内容。

### L2 改造步骤

1. **先完成 L1 全部步骤**（`_def.cpp` + `_apt.cpp` + `arch35/` + `config/`）
2. 在 `arch35/` kernel 头文件中，将核心计算路径从 Memory-based 改为 Register-based：

| 改动 | Memory-based (A2/A3) | Register-based (A5) |
|------|---------------------|---------------------|
| 数据载体 | `LocalTensor<T>` | `RegTensor<T>` |
| 有效元素 | `count` 隐式 mask | `MaskReg` 显式 mask |
| 类型转换 | `RoundMode` | `CastTrait` 四要素 |
| 数据搬运 | `DataCopy` (GM-UB) | `LoadAlign`/`StoreAlign` (UB-Reg) |
| 同步 | `SetFlag`/`WaitFlag` | `LocalMemBar<MemType::UB>` |
| 计算包裹 | 直接调用 | `__VEC_SCOPE__` 或 `__simd_vf__` 函数 |

3. **保持 outer shell 不变**（Init/Process/CopyIn/CopyOut），只改 Compute 内层
4. 排序类硬件指令（Sort/MrgSort/Extract/GatherMask）保持不变

### 全量仓 API 查阅（按需）

改造过程中如需查阅具体 API 的签名、参数、约束或示例，按 `references/search-rules.md` 路由到全量仓：

- **API 文档**：`$DEVKIT_PATH/docs/zh/api/SIMD-API/` 下按 API 类型查找对应子目录
- **头文件声明**：`$DEVKIT_PATH/include/` 目录下 grep 函数名确认原型
- **实现源码**：`$DEVKIT_PATH/impl/` 目录下 grep 函数名查看实现细节
- **算子实践参考**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/` 查找 DoubleBuffer、Tiling 策略等优化技巧

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

### L2 改造检查点

- [ ] L1 步骤全部完成
- [ ] 核心计算路径已用 `__VEC_SCOPE__` + MicroAPI 重写
- [ ] CastTrait 的 SatMode 正确（量化用 SAT，反量化用 NO_SAT）
- [ ] FP32→INT8 使用三步量化（FP32→INT16→FP16→INT8）
- [ ] 溢出模式控制已保存/恢复（`SetCtrlSpr` 配对）
- [ ] `SetFlag/WaitFlag` 已改为 `LocalMemBar<MemType::UB>`（仅核内流水间事件）
- [ ] 跨核同步点已按 Step 1.5 处置决策执行：沿用（原样保留）或禁用（移除该路径），无遗漏
- [ ] 排序类硬件指令保持不变

---

## Step 2.3：L3 改造清单

**MUST READ**：`references/impl/l3-guide.md`
读取后输出：`[LOADED] l3-guide`

### L3 改造步骤

1. 先完成 L1 全部步骤
2. 在 `arch35/` 中新增 SIMT kernel 文件
3. SIMT 函数标记 `__simt_vf__` + `LAUNCH_BOUND(2048)`
4. SIMT 函数内直接访问 `__gm__` 指针，用 `Simt::GetThreadIdx/GetThreadNum`
5. 在 `_apt.cpp` 中用 `__NPU_ARCH__ == 3510` 条件编译切换

### L3 禁止事项

- SIMT 函数内禁止使用 `LocalTensor`/`RegTensor`
- 不要忘记 `_apt.cpp` 中 `__NPU_ARCH__ == 3510` 切换
- 多线程写同一地址必须用 `Simt::AtomicAdd`

---

### 同步协议处置原则（适用于所有层级）

- **跨核同步机制（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）按阶段 1 Step 1.5 盘点结果执行**：判定"沿用"的同步点原样保留；判定"禁用"的同步点**移除该路径，改用不依赖该同步的标准流程**。
- **无法验证跨核机制在目标平台模式下成立时，禁用该路径走标准流程，禁止原样移植。**
- 需新增 C/V 配对握手时，先核对目标平台模式语义与型号支持范围（见 `cube-migration-guide.md` 改动 6），再按参与集合配对写。
- 跨核同步机制不属于 RegBase 改造范围——它与 Memory/Register 编程模型正交，L2 的 `SetFlag/WaitFlag → LocalMemBar<MemType::UB>` 只改核内事件同步（MTE2/MTE1/V 等流水间），**不得把跨核 CrossCoreSetFlag/WaitFlag 一并替换掉**。

---

## Step 2.4：★ API 差异适配（所有层级 MUST 执行）

**基于阶段 1 扫描结果和已加载的 `api-diff-guide.md`，逐项适配。**

### 2.4.1 精度适配（Subnormal）

**适用条件**：阶段 1 扫描发现算子使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal **且**输入可能包含 subnormal。

按 `api-diff-guide.md` §1.3 选择适配策略：

#### 策略 A：eps 规避（首选，性能无损）

```cpp
// 351x 版本 — eps 规避（首选，性能无损）：
constexpr float EPS = 1.0e-38f;  // 大于 FP32 最小正常数
Adds(safeSrc, srcLocal, EPS, count);  // src += eps，避免落入 subnormal
AscendC::Ln(dstLocal, safeSrc, count);  // 默认 INTRINSIC，高性能
```

#### 策略 B：API 模板参数（次选，按需分支）

```cpp
// 351x 版本 — 高精度模式（支持 subnormal，软件模拟）：
constexpr AscendC::LnConfig LN_CONFIG_PRECISE = { AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE };
AscendC::Ln<T, LN_CONFIG_PRECISE>(dstLocal, srcLocal, count);

// 分支选择（按 tiling 参数或属性切换）：
if (highPrecisionMode) {
    AscendC::Ln<T, LN_CONFIG_PRECISE>(dstLocal, srcLocal, count);
} else {
    AscendC::Ln<T, LN_CONFIG_FAST>(dstLocal, srcLocal, count);
}
```

#### 策略 C：Register-based 模式（L2 时）

```cpp
__VEC_SCOPE__
{
    constexpr MicroAPI::LnSpecificMode LN_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    MicroAPI::Ln<float, &LN_SUBNORMAL_MODE>(regDst, regSrc, pMask);
}
```

**适配检查点**：
- [ ] 选定适配策略并实现
- [ ] high_precision 分支仅在需要时启用，非默认路径
- [ ] 不影响非 subnormal 输入的精度

### 2.4.2 兼容性适配（int4 / Mmad）

**适用条件**：阶段 1 扫描发现算子使用 int4b_t 作为 Mmad 输入。

按 `api-diff-guide.md` §2.2 适配：

```cpp
// 351x 版本（Mmad 不支持 int4，需先 cast）：
Cast(int8Tensor, int4Tensor, RoundMode::CAST_RINT, count);
// ... Mmad 使用 int8 输入 ...
```

**适配检查点**：
- [ ] int4 → int8 cast 已实现
- [ ] 已在 tiling 中评估 cast 带来的性能劣化
- [ ] 4:2 稀疏（如有）已移除或替换

### 2.4.3 性能适配（vnchwconv / VSLDB / BilinearInterpolation）

**适用条件**：阶段 1 扫描发现算子命中性能退化路径。

#### 适配项 A：TransDataTo5HD / vnchwconv 退化（P0）

按 `api-diff-guide.md` §3.2.5 优化方向：

- 围绕 UB 写口瓶颈**减少转置总写入量和正反转换次数**
- 尽量**融合转置与后续计算**，通过合理 tile 和流水安排降低同步与写口竞争
- 纯转置类算子考虑替代实现路径（如直接用 DataCopy + 重组）

**适配检查点**：
- [ ] 评估了 UB 写口瓶颈对算子端到端的影响
- [ ] 减少了不必要的转置调用次数
- [ ] 考虑了转置与后续计算的融合

#### 适配项 B：高阶 ReduceSum / VSLDB 退化（P0）

按 `api-diff-guide.md` §3.3.7 优化方向：

- 为 R=16/24 比较 `VSLDB` 与普通 load/手工重排替代实现
- 按 A 大小选择策略（A 大时批量处理更有效）
- 考虑 tiling 调整使 R 不落入 16/24 窄分支

**适配检查点**：
- [ ] 评估了 R=16/24 是否命中 VSLDB 窄分支
- [ ] 如命中，比较了替代实现
- [ ] 考虑了 tiling 调整使 R 不落入窄分支

#### 适配项 C：BilinearInterpolation 退化（P1）

按 `api-diff-guide.md` §3.4.6 优化方向：

- 增加跨 vRepeat 的软件流水/批处理
- 预取 index（index/Gather 双缓冲）
- 采用多累加器或树形累加（减少 Mul→Add 串行依赖链）
- `dstBlkStride=1` 时考虑普通连续 Store 替代 VSSTB

**适配检查点**：
- [ ] 评估了 Reg API 双层循环的影响
- [ ] 考虑了 index/Gather 双缓冲
- [ ] 考虑了树形累加减少 RAW 依赖

---

## Gate 输出条件

- [ ] 对应层级的实现指南已 LOADED
- [ ] L2 时 regbase_development_guide 已 LOADED
- [ ] cube 类算子时 cube-migration-guide 已 LOADED
- [ ] **★ `api-diff-guide` 已 LOADED**
- [ ] 所有改动文件已列出
- [ ] 代码改造检查点全部通过（含跨核同步点处置：沿用/禁用，无遗漏）
- [ ] **★ API 差异三维度适配已完成并勾选检查点**

**全部通过 → 输出 `[GATE-2] FILES_MODIFIED=[file1, file2, ...]` → 进入阶段 3**
