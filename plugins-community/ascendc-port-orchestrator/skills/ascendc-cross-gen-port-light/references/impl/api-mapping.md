# Memory-based → Register-based API 映射表

> L2 改造时 MUST READ。本文件是 Memory-based API 到 Register-based (MicroAPI) 的快速映射索引。
> 详细使用方法和完整示例见 `l2-guide.md`。
> **适用对象**：vector 类算子整体；cube 类算子中 AIV 侧 Vector 路径（VF/vector 代码）的 Memory→Reg 改造同样适用本表，AIC 侧见 `cube-migration-guide.md`。
> **与 RegBase 最佳实践的分工**：本表是**迁移期对照**（旧调用 → 新写法，含量化路径与 Subnormal 映射）；RegBase 技能（`cannbot-skills/ops/ascendc-regbase-best-practice`）的 `references/api/compute_api_membase_vs_regbase.md` 是**设计与审查视角**（名称相同但对象/签名不同的审查点）。写代码前以本表对照、以 SDK header 为准。

## 一、数据类型转换映射

### 1.1 CastTrait 预定义常量（替代 RoundMode）

| 转换方向 | RoundMode（220x） | CastTrait（351x） | SatMode | RoundMode |
|---------|-------------------|-------------------|---------|-----------|
| half/bf16 → fp32 | `CAST_RINT` | `CAST_B16_TO_B32` | UNKNOWN | UNKNOWN |
| fp32 → fp16 | `CAST_RINT` | `CAST_FP32_TO_FP16` | NO_SAT | CAST_RINT |
| fp32 → fp8_e5m2 | 不支持 | `CAST_FP32_TO_FLOAT8` | SAT | CAST_RINT |
| fp32 → fp8_e4m3fn | 不支持 | `CAST_FP32_TO_FLOAT8` | SAT | CAST_RINT |
| fp32 → hifloat8 | 不支持 | `CAST_FP32_TO_HIFLOAT8` | NO_SAT | CAST_ROUND |
| fp32 → int16 | `CAST_RINT` | `CAST_FP32_TO_INT16` | SAT | CAST_RINT |
| int16 → fp16 | `CAST_NONE` | `CAST_INT16_TO_FP16` | NO_SAT | CAST_RINT |
| fp16 → int8 | `CAST_TRUNC` | `CAST_FP16_TO_INT8` | SAT | CAST_RINT |

### 1.2 FP32→INT8 三步量化路径

**硬件不支持 FP32 直接转 INT8**，两个平台都必须经过中间类型。

**950 Register-based 路径**：
```
FP32 → INT16 (CastTrait: CAST_FP32_TO_INT16, SatMode::SAT)
     → FP16  (CastTrait: CAST_INT16_TO_FP16, SatMode::NO_SAT)
     → INT8  (CastTrait: CAST_FP16_TO_INT8,  SatMode::SAT)
```

```cpp
MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
Pack((RegTensor<uint16_t>&)tmpInt16, (RegTensor<uint32_t>&)quantInt8);
Pack((RegTensor<uint8_t>&)quantInt8, (RegTensor<uint16_t>&)tmpInt16);
```

**910b Memory-based 路径**（对照参考）：
```
FP32 → INT32 (ReinterpretCast + CAST_RINT)
     → FP16  (ReinterpretCast + CAST_NONE + SetDeqScale)
     → INT8  (CAST_TRUNC)
```

### 1.3 量化路径总结

| 输出类型 | 量化路径 | Cast 步数 | 需要 Pack | SatMode |
|---------|---------|----------|----------|---------|
| int8_t | FP32→INT16→FP16→INT8 | 3 步 | 两级 Pack | SAT |
| hifloat8_t | FP32→HiFloat8 | 1 步 | 两级 Pack | NO_SAT |
| fp8_e5m2_t | FP32→FP8_E5M2 | 1 步 | 两级 Pack | SAT |
| fp8_e4m3fn_t | FP32→FP8_E4M3FN | 1 步 | 两级 Pack | SAT |

## 二、核心操作映射

### 2.1 数据声明

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `LocalTensor<float> xLocal` | `RegTensor<float> reg0` |
| `LocalTensor<half> xLocal` | `RegTensor<half> reg0` |
| count 参数（隐式掩码） | `MaskReg pMask = UpdateMask<float>(count)` |
| — | `MaskReg pFull = CreateMask<float, MaskPattern::ALL>()` |

### 2.2 数据搬运

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `DataCopy(xLocal, xGm, count)` | `DataCopy<half, LoadDist::DIST_UNPACK_B16>(reg, addr)` |
| `DataCopy(yGm, yLocal, count)` | `DataCopy<T, StoreDist::DIST_NORM>(addr, reg, maskReg)` |
| `DataCopy(yGm, yLocal, count)` (half/bf16) | `DataCopy<T, StoreDist::DIST_PACK_B32>(addr, reg, maskReg)` |
| `DataCopyPad(yGm, yLocal, padParams)` | `DataCopyUnAlign(addr, reg, uValue, stride)` |
| `DataCopyPad` 非对齐存储 | `DataCopyUnAlign` + Pack 压缩 |

### 2.3 类型转换

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `Cast<float, half>(dst, src, RoundMode::CAST_RINT, count)` | `Cast<float, half, CAST_B16_TO_B32>(dst, src, maskReg)` |
| `Cast<half, float>(dst, src, RoundMode::CAST_RINT, count)` | `Cast<half, float, CAST_FP32_TO_FP16>(dst, src, maskReg)` |
| `SetDeqScale` + `PipeBarrier` + ReinterpretCast 链 | 直接 `Cast` + `Pack`（无需中间同步） |

### 2.4 向量计算

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `Mul(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Mul(regDst, regSrc1, regSrc2, maskReg)` |
| `Add(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Add(regDst, regSrc1, regSrc2, maskReg)` |
| `Sub(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Sub(regDst, regSrc1, regSrc2, maskReg)` |
| `Div(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Div(regDst, regSrc1, regSrc2, maskReg)` |
| `Max(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Max(regDst, regSrc1, regSrc2, maskReg)` |
| `Min(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Min(regDst, regSrc1, regSrc2, maskReg)` |
| `And(dstLocal, src1Local, src2Local, count)` | `MicroAPI::And(regDst, regSrc1, regSrc2, maskReg)` |
| `Or(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Or(regDst, regSrc1, regSrc2, maskReg)` |
| `Abs(dstLocal, srcLocal, count)` | `MicroAPI::Abs(regDst, regSrc, maskReg)` |
| `Exp(dstLocal, srcLocal, count)` | `MicroAPI::Exp(regDst, regSrc, maskReg)` |
| `Ln(dstLocal, srcLocal, count)` | `MicroAPI::Ln(regDst, regSrc, maskReg)` |
| `Sqrt(dstLocal, srcLocal, count)` | `MicroAPI::Sqrt(regDst, regSrc, maskReg)` |
| `Rsqrt(dstLocal, srcLocal, count)` | `MicroAPI::Rsqrt(regDst, regSrc, maskReg)` |
| `Reciprocal(dstLocal, srcLocal, count)` | `MicroAPI::Reciprocal(regDst, regSrc, maskReg)` |
| `Not(dstLocal, srcLocal, count)` | `MicroAPI::Not(regDst, regSrc, maskReg)` |

### 2.5 标量计算

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `Muls(dstLocal, srcLocal, scalar, count)` | `MicroAPI::Muls(regDst, regSrc, scalar, maskFull)` |
| `Adds(dstLocal, srcLocal, scalar, count)` | `MicroAPI::Adds(regDst, regSrc, scalar, maskFull)` |
| `Muls(dstLocal, srcLocal, (half)1.0f, count)` | `MicroAPI::Muls(regDst, regSrc, (float)1.0f, maskFull)` |

### 2.6 归约计算

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `ReduceSumCustom(dst, src, work, count)` | `ReduceSum(regDst, regSrc, maskReg)` |
| `ReduceMax(dst, src, work, count)` | `ReduceMax(regDst, regSrc, maskReg)` |
| `ReduceMin(dst, src, work, count)` | `ReduceMin(regDst, regSrc, maskReg)` |

### 2.7 同步机制

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `SetFlag<HardEvent::MTE2_V>()` | `LocalMemBar<MemType::UB>` |
| `WaitFlag<HardEvent::MTE2_V>()` | `LocalMemBar<MemType::UB>` |
| `SetFlag<HardEvent::V_MTE3>()` | `LocalMemBar<MemType::UB>` |
| `WaitFlag<HardEvent::V_MTE3>()` | `LocalMemBar<MemType::UB>` |
| `PipeBarrier<PIPE_V>()` | 不需要（MicroAPI 内部管理） |
| `PipeBarrier<PIPE_MTE2>()` | 不需要（MicroAPI 内部管理） |

### 2.8 计算域

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| 默认（无包裹） | `__VEC_SCOPE__ { ... }` |
| 直接调用 API | 所有 MicroAPI 必须在 `__VEC_SCOPE__` 内 |

### 2.9 排序/特殊指令

| Memory-based（220x） | Register-based（351x） |
|---------------------|----------------------|
| `Sort(dstIdx, dstVal, srcVal, count)` | `Sort(dstIdx, dstVal, srcVal, maskReg)`（保持不变，接口签名适配） |
| `MrgSort4(...)` | `MrgSort4(...)`（保持不变） |
| `Extract(dst, src, idx, count)` | `Extract(dst, src, idx, maskReg)`（保持不变） |
| `GatherMask(dst, src, idx, count)` | `GatherMask(dst, src, idx, maskReg)`（保持不变） |

## 三、Subnormal 处理映射

| API | 220x 调用（默认支持） | 351x 支持 Subnormal | 351x 高性能模式 |
|-----|---------------------|-------------------|---------------|
| Exp | `Exp(dst, src, count)` | `Exp<T, EXP_CONFIG>(dst, src, count)` | 同 220x（默认 INTRINSIC） |
| Ln | `Ln(dst, src, count)` | `Ln<T, LN_CONFIG>(dst, src, count)` | 同 220x |
| Sqrt | `Sqrt(dst, src, count)` | `Sqrt<T, SQRT_CONFIG>(dst, src, count)` | 同 220x |
| Rsqrt | `Rsqrt(dst, src, count)` | `Rsqrt<T, RSQRT_CONFIG>(dst, src, count)` | 同 220x |
| Div | `Div(dst, s1, s2, count)` | `Div<T, DIV_CONFIG>(dst, s1, s2, count)` | 同 220x |
| Reciprocal | `Reciprocal(dst, src, count)` | `Reciprocal<T, RCP_CONFIG>(dst, src, count)` | 同 220x |

> Config 结构体定义、algo 取值含义与完整示例见 `cannbot-skills/ops/ascendc-api-best-practices/references/api-cross-gen-migration.md`；Reg 路径的 subnormal 写法见 `l2-guide.md` 补充 3。

## 四、溢出模式控制（351x 独有）

220x 无此能力。351x 可通过 SPR 寄存器控制溢出检测：

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

// 保存
int64_t saved = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
// 关闭溢出检测（提升性能）
AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
// ... 核心计算 ...
// 恢复
AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(saved);
```

**适用算子**：RMSNorm（输出有界）、Softmax（输出 [0,1]）等数值范围可控的算子。

## 五、全量仓参考路径

如需查阅某 API 的完整签名和约束，按 `search-rules.md` 路由到全量仓：

| 查阅需求 | 全量仓路径 |
|---------|-----------|
| C-API Reg 矢量计算 API | `$DEVKIT_PATH/docs/zh/api/SIMD-API/c_api/reg_compute/` |
| C-API Memory 矢量计算 API | `$DEVKIT_PATH/docs/zh/api/SIMD-API/c_api/vector_compute/` |
| 基础 API Memory 矢量计算 | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/` |
| 基础 API Reg 矢量计算 | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/` |
| API 头文件声明 | `$DEVKIT_PATH/include/` |
| API 实现源码 | `$DEVKIT_PATH/impl/` |
| 2201→3510 架构变更 | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.md` |
| 基础 API 迁移指导 | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_guide/` |
