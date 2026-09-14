# L2：RegBase API 重写（MicroAPI）详细指南

> **cube 类算子定级 L2 时**：本指南用于 **AIV 侧 Vector 路径**的评估与重写（L2 语义 ③，见 `cube-migration-guide.md`「性能关键 cube 算子的 L2 语义」）；AIC 侧计算路径（低阶直跑/同步协议）见 `cube-migration-guide.md`。下述触发条件与改动对 cube 类算子仅作用于 AIV 侧代码。

## 触发条件（满足任一）

1. **性能关键路径**：RMSNorm/RoPE 等高频算子，需要极致性能
2. **量化 Cast 链路复杂**：FP32→FP8/HiFloat8/INT8 多步量化
3. **需要溢出模式控制**：RMSNorm/Softmax 等数值范围可控的算子
4. **950 新增数据类型**：FP8_E4M3FN/FP8_E5M2/HiFloat8 的 Cast 需 MicroAPI

## L2 的本质

950 的 Vector Core 从 MemBase（220x）变为 RegBase（351x）。数据不再必须在 UB 内存中流转，可以直接在寄存器中操作。L2 是对核心计算路径的**全部重写**，从 Memory-based 编程模式切换到 Register-based MicroAPI 编程模式。

## API 速览与知识真源

> **完整 API 知识不在本 skill**：RegBase API 的家族级签名地图、白名单、同步约定由 RegBase 最佳实践技能承载（`cannbot-skills/ops/ascendc-regbase-best-practice/`）。下表是 L2 改造最常用的对照速览，细节一律回该技能与 SDK header 确认。

## 迁移视角速览（L2 改造最常用）

| 迁移场景 | RegBase 用法 | 详见 |
|---------|-------------|------|
| 数据声明 | `RegTensor<T>` 替代 `LocalTensor<T>`；VL=256B（950PR） | regbase_api_reference「核心类型」 |
| 显式掩码 | `MaskReg` + `CreateMask/UpdateMask` 替代隐式 count | 同上「Reg Compute 代表签名」 |
| UB↔Reg 搬运 | `LoadAlign` / `StoreAlign`（带 mask 搬出必须在 `__simd_vf__` 内，见 l2-guide「VF 封装」） | l2-guide / datacopy_best_practices |
| 类型转换 | `CastTrait` 四要素替代 `RoundMode`；FP32→INT8 三步 | regbase_api_reference「CastTrait」 |
| 归约 | `Reduce<SUM/MAX/MIN>`；A5 新增 Welford 一步统计 | regbase_api_reference「Welford」 |
| 量化压缩 | 两级 `Pack` 后 `DataCopyUnAlign` | l2-guide「量化 Pack」 |
| 核内同步 | `LocalMemBar<MemType::UB>` 替代 SetFlag/WaitFlag | regbase_api_sync |
| 溢出模式 | `GetCtrlSpr/SetCtrlSpr`（必须保存/恢复） | regbase_api_reference「SPR 访问」 |

## 改动 1：MicroAPI 命名空间引入

```cpp
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::UpdateMask;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::CastTrait;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskMergeMode;
```

## 改动 2：寄存器向量长度

```cpp
// 由平台决定，不再硬编码
constexpr static uint32_t VL_FP32 =
    static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);
```

## 改动 3：CastTrait 配置（替代传统 RoundMode）

950 中类型转换不再使用 `RoundMode` 枚举，改用 `CastTrait` 结构体（四要素：RegLayout / SatMode / MaskMergeMode / RoundMode）。

**CastTrait 完整定义、预定义快捷常量与全部转换方向速查表已沉淀至 RegBase 最佳实践**：`cannbot-skills/ops/ascendc-regbase-best-practice/references/api/regbase_api_reference.md`（CastTrait 家族条目）+ SDK header（`basic_api/reg_compute/`）。

迁移决策速记（详见上表）：

- **量化（高精度→低精度，含 FP32→FP8/INT8）**：`SatMode::SAT` 防溢出；**反量化：`NO_SAT`**
- FP32→INT8 无直接指令：三步 FP32→INT16→FP16→INT8（完整路径与 Pack 见本文件改动 6 与 api-mapping）
- 常量：`CAST_B16_TO_B32` / `CAST_FP32_TO_FP16` / `CAST_FP32_TO_FLOAT8` / `CAST_FP32_TO_HIFLOAT8` / `CAST_FP32_TO_INT16` / `CAST_INT16_TO_FP16` / `CAST_FP16_TO_INT8`


## 改动 4：Register-based 核心计算

> **⚠ 编译约束**：下方示例中的 `StoreTensorForDtypeTOut` / `StoreUnAlignOneTensor` 若内部使用带 mask 的 Reg 搬出（`DataCopy<StoreDist>`/`StoreAlign`），**不能在普通 `__aicore__` 函数中直接调用**（后端报 `Do not know how to split the result of this operator!`），必须按本节末尾「VF 封装：Reg 级搬移的标准结构」处理。

### 完整示例：RMSNorm + Mul

```cpp
__aicore__ inline void RmsNormPostWithMul(...) {
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
        MaskReg pMask = UpdateMask<float>(count);
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopTimes; ++i) {
            LoadTensorForDtypeT<T_KV>(x, reg0, pMask, i * stride);
            LoadTensorForDtypeT<T_KV>(gamma, reg1, pMask, 0);
            MicroAPI::Mul(reg2, reg0, reg0, pMask);       // x * x
            ReduceSum(reg2, reg2, pMask);                   // ReduceSum(x*x)
            MicroAPI::Muls(reg3, reg2, reciprocal, pFull);  // * 1/N
            MicroAPI::Adds(reg4, reg3, epsilon, pFull);     // + eps
            MicroAPI::Sqrt(reg5, reg4, pFull);              // sqrt(...)
            MicroAPI::Div(reg6, reg0, reg5, pMask);         // x / sqrt(...)
            MicroAPI::Mul(reg7, reg1, reg6, pMask);         // gamma * norm
            StoreTensorForDtypeTOut<float>(dst, reg7, pMask, i * stride);
        }
    }
}
```

### 关键要素

1. **`__VEC_SCOPE__`**：必须包裹所有 MicroAPI 计算代码
2. **`RegTensor<T>`**：寄存器张量声明，替代 `LocalTensor<T>`
3. **`MaskReg`**：显式掩码，替代隐式 count 参数
   - `UpdateMask<float>(count)`：按实际元素数创建掩码
   - `CreateMask<float, MaskPattern::ALL>()`：全 1 掩码
4. **`LoadTensorForDtypeT<T_KV>`**：带自动类型转换的加载（half/bf16→fp32）
5. **`StoreTensorForDtypeTOut<T>`**：带自动类型转换的存储（fp32→half/bf16/fp8）
6. **`MicroAPI::` 前缀**：所有向量计算 API 都在 MicroAPI 命名空间下

### VF 封装：Reg 级搬移的标准结构

带 mask 的 Reg 级搬出 API（`StoreAlign` / `DataCopy<StoreDist>`）**必须在 `__simd_vf__` 向量函数中调用**。它携带 mask 且具 repeat 合并语义，编译器需在调用点静态推导 mask 展开——写在普通 `__aicore__` 函数中时 mask 来自运行时标量，推导失败，后端报 `error in backend: Do not know how to split the result of this operator!`（无源码定位信息，模板共有代码可能三 dtype 全挂）。`LoadAlign`（无 mask 参数）不受此限，但建议与 Store 统一封装。

三条必守规则：

1. **搬移 + 计算 + 归约 + 写回完整序列都放进 `__simd_vf__` 函数**，参数用 `__ubuf__` 指针承接 UB 地址，由外层 `__aicore__` 包装函数从 `LocalTensor::GetPhyAddr()` 转换后调用。
2. **VF 内循环必须预计算迭代次数**，禁止乘法比较作循环条件（`for (v = 0; v * vecLen < count; ++v)`）——VF 展开需静态推导 repeat 数，乘法形式推导失败且 clang 前端无诊断崩溃（`exit code 139`）：
   ```cpp
   uint32_t repeatTimes = (count + vecLen - 1) / vecLen;
   for (uint16_t i = 0; i < repeatTimes; ++i) { ... }
   ```
3. **UB 地址不能 `reinterpret_cast<__ubuf__ T*>`**（AS 限定符不能作 reinterpret_cast 目标）——包装函数内用 `GetPhyAddr()` 转换，VF 函数声明参数为 `__ubuf__ T*`。

标准结构模板：

```cpp
// VF 函数：__ubuf__ 指针参数 + 函数内 CreateMask/UpdateMask + 全套 Reg 级 API
__simd_vf__ inline void RowMaxVF(__ubuf__ float *row, __ubuf__ float *mOut, uint32_t cols)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(float));
    Reg::RegTensor<float> vreg;
    Reg::RegTensor<float> rowMaxReg;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::Duplicate(rowMaxReg, AscendC::NumericLimits<float>::NegativeInfinity(), fullMask);
    uint32_t remaining = cols;
    uint32_t repeatTimes = (cols + oneRepElm - 1) / oneRepElm;
    for (uint16_t v = 0; v < repeatTimes; ++v) {
        Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
        const uint32_t offset = v * oneRepElm;
        Reg::LoadAlign<float>(vreg, row + offset);
        Reg::Max<float, Reg::MaskMergeMode::MERGING>(rowMaxReg, rowMaxReg, vreg, mask);
    }
    Reg::ReduceMax(rowMaxReg, rowMaxReg, fullMask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(mOut, rowMaxReg, maskOne);
    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
}

// 外层包装：LocalTensor → __ubuf__ 指针 → 调 VF
extern "C" __global__ __aicore__ void test_kernel(...)
{
    RowMaxVF(reinterpret_cast<float*>(rowTensor.GetPhyAddr()),
             reinterpret_cast<float*>(mOutTensor.GetPhyAddr()), cols);
}
```

**RegBase 编译错误速查**：

| 错误现象 | 根因 | 修复 |
|---------|------|------|
| `error in backend: Do not know how to split the result of this operator!` | 带 mask 搬出（`StoreAlign`/`DataCopy<StoreDist>`）写在普通 `__aicore__` 函数中 | 包进 `__simd_vf__` 函数（上方模板） |
| `clang frontend command failed with exit code 139`（无诊断） | VF 内循环乘法比较条件 `v * vecLen < count` | 预计算 `repeatTimes`，用 `v < repeatTimes` |
| `reinterpret_cast` 目标带地址空间限定符 | `reinterpret_cast<__ubuf__ T*>` | 包装函数内 `GetPhyAddr()` 转换，VF 参数声明 `__ubuf__ T*` |
| `__ubuf__` 未定义 / 类型未知 | 编译器自动 -include 的 `__clang_cce_runtime_wrapper.h` 被改动 | 从同版本 CANN 包恢复该头文件 |
| VF 代码在 `.i` 快速编译回路与真实编译结果不一致 | `-E` 展开把 `__ubuf__` 变成 attribute 形式，语义不等价 | VF 相关改动必须走真实编译链路（opc / .mk -c 命令）验证 |

## 改动 5：溢出模式控制

950 可通过 SPR 寄存器控制溢出检测，对 RMSNorm/Softmax 等数值范围可控的算子，关闭溢出检测可提升性能：

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

// 保存原始溢出模式
#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

// 设置为 0（关闭溢出检测，提升性能）
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif

// ... 核心计算 ...

// 恢复原始溢出模式
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
```

### 使用原则

- **必须保存/恢复**：溢出模式是全局状态，修改后必须恢复
- **仅限数值范围可控的算子**：RMSNorm（输出始终有界）、Softmax（输出 [0,1]）
- **必须用 `__NPU_ARCH__ == 3510` 保护**：220x 平台不支持此 SPR

## 改动 6：量化 Pack + 非对齐 Store（仅 INT8 量化算子需要）

> **注意**：本节内容仅适用于输出包含 INT8 类型的量化算子。如果算子只涉及 FP16/BF16/FP8/HiFloat8，Cast 都是一步完成，无需三步转换。

### FP32 → INT8 三步量化

**910b 和 950 都不支持 FP32 直接转 INT8**，硬件没有这个 Cast 指令，必须经过中间类型。但两个平台的中间类型和同步方式不同：

**950 Register-based 路径**（FP32 → INT16 → FP16 → INT8）：

```cpp
MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
```

**910b Memory-based 路径**（FP32 → INT32 → FP16 → INT8）：

```cpp
Cast(src.ReinterpretCast<int32_t>(), src, RoundMode::CAST_RINT, size);     // FP32→INT32
PipeBarrier<PIPE_V>();
SetDeqScale((half)1.0f);                                                  // 设置反量化缩放因子
PipeBarrier<PIPE_V>();
Cast(src.ReinterpretCast<half>(), src.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);  // INT32→FP16
PipeBarrier<PIPE_V>();
Cast(dst, src.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);      // FP16→INT8
```

**关键差异**：
- 950 用 INT16 作中间类型（更省寄存器），910b 用 INT32（4字节对齐）
- 950 不需要 `SetDeqScale` 和 `PipeBarrier`，910b 每步之间需要同步
- 950 量化后需要 Pack 压缩才能 Store，910b 不需要

### 两级 Pack 压缩

量化后需要 Pack 将数据压缩到更窄的位宽：

```cpp
Pack((RegTensor<uint16_t>&)tmpInt16, (RegTensor<uint32_t>&)quantInt8);
Pack((RegTensor<uint8_t>&)quantInt8, (RegTensor<uint16_t>&)tmpInt16);
```

### 非对齐 Store

当输出数据长度不是对齐倍数时，使用 `DataCopyUnAlign`：

```cpp
DataCopyUnAlign(output, quantInt8, uValue, postUpdateStride);
```

## Register-based vs Memory-based 完整对照

| 操作 | Memory-based（220x） | Register-based（351x） |
|------|---------------------|----------------------|
| 数据声明 | `LocalTensor<float> xLocal` | `RegTensor<float> reg0` |
| GM→UB 加载 | `DataCopy(xLocal, xGm, count)` | `DataCopy<LoadDist::DIST_UNPACK_B16>(reg0, addr)` |
| 类型转换 | `Cast<float, half>(dst, src, RoundMode::CAST_RINT, count)` | `Cast<float, half, CAST_B16_TO_B32>(dst, src, maskReg)` |
| 向量计算 | `Mul(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Mul(regDst, regSrc1, regSrc2, maskReg)` |
| 标量计算 | `Muls(dstLocal, srcLocal, scalar, count)` | `MicroAPI::Muls(regDst, regReg, scalar, maskFull)` |
| 归约 | `ReduceSumCustom(dst, src, work, count)` | `ReduceSum(regDst, regSrc, maskReg)` |
| UB→GM 存储 | `DataCopy(yGm, yLocal, count)` | `DataCopy<StoreDist::DIST_NORM>(addr, reg, maskReg)` |
| 非对齐存储 | `DataCopyPad(yGm, yLocal, padParams)` | `DataCopyUnAlign(addr, reg, unalignReg, stride)` |
| 掩码 | 隐式（通过 count 参数） | 显式 `MaskReg` + `CreateMask/UpdateMask` |
| 内存屏障 | `SetFlag<HardEvent::MTE2_V>` / `WaitFlag` | `LocalMemBar<MemType::UB>` |
| 计算域 | 默认 | `__VEC_SCOPE__ { ... }` |
| 量化 Pack | 不需要 | `Pack((RegTensor<uint16_t>&)dst, (RegTensor<uint32_t>&)src)` |

## L2 迁移检查清单

- [ ] `RegTensor` / `MaskReg` / `CastTrait` 是否正确使用
- [ ] `LoadDist` / `StoreDist` 是否正确配置
- [ ] `LocalMemBar<MemType::UB>` 内存屏障是否正确使用
- [ ] `__VEC_SCOPE__` 计算域是否包裹核心计算
- [ ] 溢出模式控制（GetCtrlSpr/SetCtrlSpr）是否需要，是否正确保存/恢复
- [ ] 量化 Pack（两级 Pack 压缩）是否正确实现
- [ ] `DataCopyUnAlign` 非对齐存储是否正确使用
- [ ] FP8/HiFloat8 量化 CastTrait 的 SatMode 是否正确（量化用 SAT，反量化用 NO_SAT）
- [ ] FP32→INT8 是否经过三步量化（FP32→INT16→FP16→INT8）
- [ ] `__NPU_ARCH__ == 3510` 条件编译保护是否正确

## 架构差异全景（L2 迁移决策依据）

> 注：下表"影响"列的 L4/L4+ 为旧分级口径（Tiling 适配等）；现行层级模型只有 L1/L2/L3，L4+ 条目表示"超出 L1-L3 指南范围"，需升级评估（见 SKILL.md 决策树）。

| 维度 | 220x（MemBase） | 351x（RegBase） | 影响 |
|------|----------------|----------------|------|
| Vector Core 架构 | MemBase（数据在 UB 内存中流转） | RegBase（数据可直接在寄存器中操作） | L2 |
| 编程模式 | 仅 SIMD | SIMD + SIMT + SIMD/SIMT 混合 | L4 |
| UB 容量 | 128KB（910b）/ 192KB（910_93） | 256KB | L4 |
| SIMT DCache | 无 | 复用 UB 空间，需预留 40KB | L4 |
| L1→GM 通路 | ✓ 支持 | ✗ 删除 | L4+ |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ 删除 | L4+ |
| UB→L1 通路 | ✗ 不支持 | ✓ 新增 | L4+ |
| L0C→UB 通路 | ✗ 不支持 | ✓ 新增 | L4+ |
| Subnormal 支持 | ✓ 硬件支持 | ✗ 默认不支持（需软仿） | L4 |
| 新数据类型 | - | fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t | L3/L5 |
| 溢出模式控制 | - | GetCtrlSpr/SetCtrlSpr（SPR 寄存器） | L2 |

> 上表 **L4+ 条目（L1→GM / GM→L0A/L0B / UB→L1 / L0C→UB 数据通路变更）** 属于 Cube 侧差异，完整展开（含分形 ZZ→NZ、SetLoadDataBoundary 删除、L0A/L0B 初始化删除、int4b/稀疏删除、CrossCoreSetFlag 模式等）见 `references/impl/cube-migration-guide.md`「官方数据通路变更总表」。

## 代表算子

**KvRmsnormRopeCache**：唯一真正使用 Register-based 模式的算子，包含：
- MicroAPI 重写核心计算（RMSNorm + RoPE）
- 溢出模式控制（关闭溢出检测提升性能）
- FP8/HiFloat8 量化 CastTrait
- 量化 Pack + DataCopyUnAlign
- 26 种数据类型组合（910b 仅 8 种）

---

## 补充 1：通用 Load/Store 函数模板

### LoadTensorForDtypeT\<T\>：带自动类型转换的加载

支持 `half`/`bfloat16_t`/`float` 三种输入类型自动推导，half/bf16→fp32 自动转换：

```cpp
template <typename T>
__aicore__ inline void LoadTensorForDtypeT(
    __local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + offset));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + offset));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + offset));
    }
}
```

**设计要点**：
- 使用 `if constexpr` 编译期分支，零运行时开销
- half/bf16 加载后立即 Cast 为 fp32，后续计算全部在 fp32 寄存器中进行
- float 类型直接 `DataCopy`，无需 Cast

### StoreTensorForDtypeTOut\<T\>：带自动类型转换的存储

支持 `float`/`half`/`bfloat16_t` 输出，fp32→half/bf16 自动转换：

```cpp
template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T> xFp16;
        Cast<T, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}
```

**设计要点**：
- float 输出直接 `DataCopy`，使用 `StoreDist::DIST_NORM`
- half/bf16 输出先 Cast 再 `DataCopy`，使用 `StoreDist::DIST_PACK_B32`
- 注意 Load 用 `DIST_UNPACK_B16`，Store 用 `DIST_PACK_B32`，方向相反

---

## 补充 2：非对齐 Store 模板

### StoreUnAlignOneTensor\<T\>：支持量化类型的非对齐存储

支持 `int8_t`/`hifloat8_t`/`fp8_e5m2_t`/`fp8_e4m3fn_t` 等量化类型，内部自动选择量化路径：

```cpp
template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(
    __local_mem__ T*& output, MicroAPI::RegTensor<float>& src,
    MicroAPI::UnalignReg& uValue, MicroAPI::MaskReg& preg,
    uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        // FP32 → INT16 → FP16 → INT8 三步量化
        AscendC::MicroAPI::RegTensor<half> tmpHalf;
        AscendC::MicroAPI::RegTensor<int16_t> tmpInt16;
        AscendC::MicroAPI::RegTensor<int8_t> quantInt8;
        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)tmpInt16, (MicroAPI::RegTensor<uint32_t>&)quantInt8);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantInt8, (MicroAPI::RegTensor<uint16_t>&)tmpInt16);
        DataCopyUnAlign(output, quantInt8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        // FP32 → HiFloat8 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<hifloat8_t> quantHifloat8;
        AscendC::MicroAPI::Cast<hifloat8_t, float, CAST_FP32_TO_HIFLOAT8>(quantHifloat8, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantHifloat8, (MicroAPI::RegTensor<uint32_t>&)quantHifloat8);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantHifloat8, (MicroAPI::RegTensor<uint16_t>&)quantHifloat8);
        DataCopyUnAlign(output, quantHifloat8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
        // FP32 → FP8_E5M2 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> quantFloat8E5m2;
        AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E5m2, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantFloat8E5m2, (MicroAPI::RegTensor<uint32_t>&)quantFloat8E5m2);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantFloat8E5m2, (MicroAPI::RegTensor<uint16_t>&)quantFloat8E5m2);
        DataCopyUnAlign(output, quantFloat8E5m2, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        // FP32 → FP8_E4M3FN 直接量化 + Pack + DataCopyUnAlign
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3;
        AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E4m3, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t>&)quantFloat8E4m3, (MicroAPI::RegTensor<uint32_t>&)quantFloat8E4m3);
        Pack((MicroAPI::RegTensor<uint8_t>&)quantFloat8E4m3, (MicroAPI::RegTensor<uint16_t>&)quantFloat8E4m3);
        DataCopyUnAlign(output, quantFloat8E4m3, uValue, postUpdateStride);
    }
}
```

### 量化路径对比

| 输出类型 | 量化路径 | Cast 步数 | 需要 Pack | SatMode |
|---------|---------|----------|----------|---------|
| int8_t | FP32→INT16→FP16→INT8 | 3 步 | ✓ 两级 Pack | SAT |
| hifloat8_t | FP32→HiFloat8 | 1 步 | ✓ 两级 Pack | NO_SAT |
| fp8_e5m2_t | FP32→FP8_E5M2 | 1 步 | ✓ 两级 Pack | SAT |
| fp8_e4m3fn_t | FP32→FP8_E4M3FN | 1 步 | ✓ 两级 Pack | SAT |

**关键差异**：
- INT8 是唯一需要三步量化的类型（硬件不支持 FP32→INT8 直接转换）
- FP8/HiFloat8 可一步量化，但仍需 Pack 压缩后才能 `DataCopyUnAlign`
- 所有量化类型输出均使用 `DataCopyUnAlign`（非对齐存储），因为量化后数据长度通常不是对齐倍数

---

## 补充 3：Subnormal 适配（L2 改造动作）

L2 重写时，若算子使用 **Exp / Ln / Sqrt / Rsqrt / Div / Reciprocal**，需在改代码的同时落实 subnormal 策略（是否处理、怎么处理由阶段 1 扫描与迁移方案决定，见 `api-diff-guide.md` §1）。

**L2 侧的动作只有两种写法**：

| 策略 | Memory-based 路径（L2 内的外层 shell 等） | Register-based 路径（VF 内） |
|---|---|---|
| 保持默认（高性能，subnormal→0） | 不传 Config，即 `Ln(dst, src, count)` | 不传 mode，即 `Reg::Ln(dst, src, mask)` |
| 高精度（软件模拟，支持 subnormal，**性能影响大**） | 传 algo Config：`Ln<T, LN_CONFIG>(dst, src, count)` | 传 `LnSpecificMode` 指针：`Reg::Ln<float, &LN_SUBNORMAL_MODE>(dst, src, mask)` |

- **优先级**：能保持默认就保持默认（迁移方案判定存在结构性排除时）；确需高精度时再按 tiling 参数/属性开分支，不要默认全开
- **完整 API 知识**（六类 API 的 Config 结构、algo 取值含义、220x/351x 代码对比、SpecificMode 定义与示例）见 API 最佳实践：`cannbot-skills/ops/ascendc-api-best-practices/references/api-cross-gen-migration.md`
- **策略选择与证据要求**（何时可用策略 0 结构性排除、eps 可检查性）见 `api-diff-guide.md` §1

## 补充 4：架构差异全景表（220x vs 351x 完整对照）

| 维度 | 220x（MemBase，910b/910_93） | 351x（RegBase，950） | 影响层级 |
|------|---------------------------|---------------------|---------|
| Vector Core 架构 | MemBase（数据在 UB 内存中流转） | RegBase（数据可直接在寄存器中操作） | L2 |
| 编程模式 | 仅 SIMD | SIMD + SIMT + SIMD/SIMT 混合 | L4 |
| UB 容量 | 128KB（910b）/ 192KB（910_93） | 256KB | L4 |
| UB 结构 | 16 bank group × 3 bank × 4KB | 8 bank group × 2 bank × 16KB | L4 |
| SIMT DCache | 无 | 复用 UB 空间，需预留 40KB | L4 |
| L1→GM 通路 | ✓ 支持 | ✗ 删除 | L4+ |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ 删除 | L4+ |
| UB→L1 通路 | ✗ 不支持 | ✓ 新增 | L4+ |
| L0C→UB 通路 | ✗ 不支持 | ✓ 新增 | L4+ |
| Subnormal 支持 | ✓ 硬件支持 | ✗ 默认不支持（需软仿） | L4 |
| int4b_t Cube 计算 | ✓ 支持 | ✗ 不支持 | L4 |
| 4:2 稀疏 | ✓ 支持 | ✗ 不支持 | L4 |
| L0A 分形格式 | ZZ | NZ | L4+ |
| SetLoadDataBoundary | ✓ 支持 | ✗ 删除 | L4+ |
| 新数据类型 | - | fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t | L3/L5 |
| 溢出模式控制 | - | GetCtrlSpr/SetCtrlSpr（SPR 寄存器） | L2 |

> 上表 **L4+ 条目（L1→GM / GM→L0A/L0B / UB→L1 / L0C→UB 数据通路变更、L0A 分形 ZZ→NZ、SetLoadDataBoundary 删除）** 属于 Cube 侧差异，落地方法（装载/回写 API 改写、NZ 分形适配、L1 循环读取手动绕回等）见 `references/impl/cube-migration-guide.md`「官方数据通路变更总表」与「改动 2/3/5」。L4+ 表示"超出 L1-L3 指南范围"，并非"改造深度为 L4"。 |

---

## 补充 5：相关参考文档路径

| 文档类别 | 路径 | 说明 |
|---------|------|------|
| Register-based Vector / MicroAPI | KB 快照 `reg-base-vector/` | Reg 矢量计算编程指南 |
| Memory-based Vector 操作 | KB 快照 `memory-base-vector/` | 掩码操作、归约计算、高维切分、连续计算 API |
| 迁移相关官方文档 | KB 快照 `migration/` | 220x→351x 架构变更、基础/高阶 API 迁移指导、算子编译迁移、兼容性说明 |
| SIMD 编程参考 | KB 快照 `simd/` | SIMD BuiltIn 关键字、语言扩展层 C API |
| SIMT 编程参考 | KB 快照 `simt/` | SIMT 线程架构、同步机制、AI Core SIMT 编程 |
| API 概述 | KB 快照 `api-overview/` | 编程接口概述、高阶/基础 API 概述 |

> 上表"KB 快照"路径位于 `cannbot-skills/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/`（本方法论 2026-05 旧版存档）。
