# D3: TilingKey-Driven Type Dispatch (TilingKey驱动类型分发)
## Overview
专家实现通过模板类FakeQuantAffineCachemaskFp32和FakeQuantAffineCachemaskFp16实现了对FP16和FP32两种数据类型的完整支持。这种设计允许根据输入数据类型自动选择最优的计算路径，充分利用昇腾芯片对不同精度计算的硬件支持。在算子定义文件中，通过配置多组DataType实现了运行时数据类型的灵活选择。Tiling阶段通过SetTilingKeyMode函数根据数据类型设置不同的Tiling Key（FP16_MODE=2, FP32_MODE=1），从而在kernel入口处分发到对应的模板实例。这种策略的优势在于：1) 避免了运行时的类型判断开销；2) 允许针对每种数据类型进行专门的优化；3) 通过编译期多态实现零开销抽象。值得注意的是，FP16和FP32的实现细节有所不同，例如FP16实现中使用了更多的Cast操作进行精度转换，而FP32实现则尽量保持原生精度计算。

## When to Use
- Host-side multi-type selection
- 支持FP16/BF16/FP32三种精度，通过低精度计算获得2x性能提升，同时保持数值稳定性
- 运行时动态选择最优kernel变体，编译期优化更充分
- 代码结构清晰，每种策略专注优化特定场景

## Trade-off
- 代码复杂度增加，需要模板编程和编译期分支处理
- Tiling逻辑复杂度增加
- 增加代码量和维护成本

**Source operators**: adaptive_avg_pool3d, add_rms_norm_cast, add_rms_norm_dynamic_quant, apply_adam_w_v2, ascend_quant_v2, batch_norm_v3, clipped_swiglu, deep_norm, dequant_bias, dynamic_block_quant, dynamic_mx_quant, dynamic_quant_update_scatter_v2, embedding_dense_grad_v2, fake_quant_affine_cachemask, foreach_abs, foreach_add_scalar_list, gemma_rms_norm, grouped_dynamic_mx_quant, layer_norm_v3, layer_norm_v4, linear_index, masked_scatter_with_position, max_pool_grad_with_argmax_common, max_pool_with_argmax_v3, modulate, multi_scale_deformable_attention_grad, multi_scale_deformable_attn_function, norm_common, rms_norm_quant, scaled_masked_softmax_grad_v2, scaled_masked_softmax_v2, scatter_elements_v2, sparse_to_dense

---

## Variant A: 动态数据类型分发与转换
Source: adaptive_avg_pool3d

专家实现通过Tiling Key设计实现运行时数据类型分发，Tiling Key由mode_key * 10 + data_type_key组成。针对不同类型采用不同RoundMode：FP32→FP16使用CAST_NONE直接截断，FP32→BF16使用CAST_RINT四舍五入。通过std::is_same_v<T, float>编译期判断，在FP32模式下完全跳过Cast操作，减少计算开销。

**Expert implementation:**
```cpp
// 三种数据类型支持
this->Input("x")
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16});

// 差异化RoundMode处理
if constexpr (std::is_same_v<T, float>) {
    DataCopy(outputLocal, sumBufLocal, len);
} else if constexpr (std::is_same_v<T, half>) {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, len);
} else {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
}
```

**vs. baseline:**
```cpp
// baseline仅支持FP32
this->Input("x")
    .DataType({ge::DT_FLOAT});
```

Benefit: 支持FP16/BF16/FP32三种精度，通过低精度计算获得2x性能提升，同时保持数值稳定性
Trade-off: 代码复杂度增加，需要模板编程和编译期分支处理

---

## Variant B: Tiling Key多类型编码策略
Source: add_rms_norm_cast

专家实现使用Tiling Key来编码数据类型和计算模式，通过dtypeKey * 10U + modeKey的方式将两个维度压缩到一个整数中。DTYPE_KEY_FP16=1, DTYPE_KEY_BF16=3, MODE_NORMAL=0, MODE_SPLIT_D=1, MODE_SINGLE_N=3, MODE_MULTI_N=4。这种设计使得同一个kernel可以通过不同的Tiling Key选择不同的特化实现，运行时根据数据特征动态选择最优kernel变体，避免单一kernel代码膨胀，提高指令缓存命中率。

**Expert implementation:**
```cpp
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_BF16 = 3;

uint32_t modeKey = MODE_NORMAL;
if (numCol > ubFactor) {
    modeKey = MODE_SPLIT_D;
} else if (blockFactor == 1) {
    modeKey = MODE_SINGLE_N;
}
uint32_t tilingKey = dtypeKey * 10U + modeKey;
context->SetTilingKey(tilingKey);
```

**vs. baseline:**
```cpp
// Lingxi-code无Tiling Key编码
uint32_t tileLength = (MAX_TILE_LEN < hiddenSize) ? MAX_TILE_LEN : hiddenSize;
uint32_t nTiles = (hiddenSize + tileLength - 1) / tileLength;
```

Benefit: 运行时动态选择最优kernel变体，编译期优化更充分
Trade-off: Tiling逻辑复杂度增加

---

## Variant C: Kernel多策略分发
Source: add_rms_norm_dynamic_quant

专家实现的Kernel入口通过TILING_KEY_IS宏根据Tiling Key选择不同的Kernel实现类(Normal/SingleRow/SliceD)。这种设计使得不同策略可以使用最优的代码路径，避免在单一Kernel中处理所有逻辑的复杂度。

**Expert implementation:**
```cpp
extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant(...) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelAddRmsNormDynamicQuantNormal<DTYPE_X1, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelAddRmsNormDynamicQuantSingleRow<DTYPE_X1, 2> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormDynamicQuantSliceD<DTYPE_X1, 3> op(&pipe);
        INIT_AND_PROCESS;
    }
}
```

**vs. baseline:**
```cpp
extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_custom(...) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddRmsNormDynamicQuant op;
    op.Init(...);
    op.Process();
}
```

Benefit: 代码结构清晰，每种策略专注优化特定场景
Trade-off: 增加代码量和维护成本

---

## Variant D: Tiling Key驱动的多类型分发机制
Source: apply_adam_w_v2

专家实现通过10个不同的Tiling Key（101-110）精确区分不同的数据类型组合，实现了细粒度的算子分发。这种设计允许在编译期确定数据类型，避免了运行时的类型判断开销。具体来说，Tiling Key 101-106处理同构数据类型（BF16/FP16/FP32 × step类型float/int64），而107-110处理混合数据类型场景（var/m/v为FP32，grad为FP16/BF16）。这种设计使得每种数据类型组合都能获得最优的计算路径，特别是混合数据类型场景下，grad使用低精度(FP16/BF16)减少内存带宽，而var/m/v使用FP32保证更新精度，实现了精度与性能的最佳平衡。

**Expert implementation:**
```cpp
static inline void GetTilingKey(ApplyAdamWV2TilingParam& tilingParam)
{
    auto stepDtype = tilingParam.dtypeLst[INDEX_IN_STEP];
    if (IsDiffDtype(tilingParam.dtypeLst)) {
        auto gradDtype = tilingParam.dtypeLst[INDEX_IN_GRAD];
        if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_FLOAT_KEY;
        } else if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_INT64) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_INT64_KEY;
        } // ... 共10种组合
    }
}
```

**vs. baseline:**
```cpp
// Set tiling key - only support FP32
context->SetTilingKey(1);
```

Benefit: 编译期确定数据类型，零运行时开销；支持混合精度训练，降低内存带宽30-50%
Trade-off: 代码复杂度增加，需要维护10个Tiling Key和对应的模板实例

---

## Variant E: 分层类型转换策略
Source: ascend_quant_v2

在量化计算中采用分层类型转换：FP16/BF16输入首先在UB中转换为FP32进行计算（保证数值稳定性），输出阶段再通过特定舍入模式转换为INT8/INT4。使用SetDeqScale指令和两次Cast操作完成float到int8转换，充分利用Ascend C硬件量化指令流水。

**Expert implementation:**
```cpp
// 分层转换：float->int32->half->int8
Cast(xLocal[inOffset].ReinterpretCast<int32_t>(), xLocal[inOffset], RoundMode::CAST_RINT, dataCount);
PipeBarrier<PIPE_V>();
SetDeqScale((half)1.000000e+00f);
Cast(xLocal[inOffset].ReinterpretCast<half>(), xLocal[inOffset].ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, dataCount);
Cast(outLocal[outOffset], xLocal[inOffset].ReinterpretCast<half>(), RoundMode::CAST_RINT, dataCount);
```

**vs. baseline:**
```cpp
// 直接在FP32上计算后转换
AscendC::Cast(tempInt32Local, tempDivLocal, AscendC::RoundMode::CAST_RINT, this->tileSize);
AscendC::Cast(tempRoundLocal, tempInt32Local, AscendC::RoundMode::CAST_NONE, this->tileSize);
```

Benefit: 平衡计算精度和硬件效率，FP32中间计算保证数值稳定性，低比特输入输出减少内存带宽
Trade-off: 需要更多的UB缓冲区存储中间结果

---

## Variant F: 多算法模板自适应选择
Source: batch_norm_v3

专家实现针对BatchNorm的不同计算特征，实现了多种算法模板，通过Tiling系统在运行时动态选择最优路径：1）Welford算法模板（Tiling Key: 1000-1013）：适用于训练场景，使用Welford在线算法计算均值和方差，具有数值稳定性好、单次遍历的特点。根据R0/R1维度的切分对齐情况，细分为4种子模式；2）FullReduce算法模板（Tiling Key: 2000-2001）：适用于小规模数据，使用全规约方式计算均值和方差，计算并行度更高；3）Regbase模板（Tiling Key: 300000+）：针对特定硬件架构优化的寄存器级实现。算法选择逻辑基于数据Shape特征：当patternR1 * patternR0 < 8192时优先使用FullReduce，否则使用Welford。

**Expert implementation:**
```cpp
// 专家实现多模板
#define BNV3_WELFORD_R0_SPLIT_NOT_ALIGN 1000
#define BNV3_WELFORD_R0_SPLIT_ALIGN 1001
#define BNV3_FULL_REDUCE_NORMAL 2000
#define BNV3_FULL_REDUCE_A_PARALLEL 2001

// 模板分发
if (TILING_KEY_IS(BNV3_WELFORD_R0_SPLIT_NOT_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(...);
} else if (TILING_KEY_IS(BNV3_FULL_REDUCE_NORMAL)) {
    BatchNormV3FullReduce<DTYPE_X, DTYPE_WEIGHT, 0> op(&pipe);
    op.Process();
}
```

**vs. baseline:**
```cpp
# baseline单一算法
def batch_norm_v3_kernel(...):
    # 简单两趟计算
    for channel_idx in range(c_start, c_end):
        # Pass 1: 计算均值
        # Pass 2: 计算方差
```

Benefit: 不同Shape场景下性能提升30%-100%；自动选择最优算法避免手动调优
Trade-off: 代码复杂度增加；需要维护多套算法实现

---

## Variant G: 模板化编程与编译期分发
Source: clipped_swiglu

专家实现采用C++模板编程技术，将数据类型作为模板参数，实现了一套代码支持多种数据类型。在Kernel入口函数中，通过编译期宏ORIG_DTYPE_X和TilingKey机制，在编译阶段确定实际调用的模板实例。这种设计避免了运行时的类型判断开销，同时保持了代码的可维护性。关键设计包括：1)模板类设计ClippedSwigluBase<inType>封装所有计算逻辑；2)编译期分发使用条件编译选择模板实例；3)类型转换策略：非FP32输入在CopyIn后Cast到FP32计算，CopyOut前Cast回原始类型；4)差异化RoundMode：BF16使用CAST_RINT（四舍五入），FP16使用CAST_NONE（截断）。预期收益：支持FP16/BF16可减少50%的内存带宽需求。潜在代价：类型转换引入少量计算开销。

**Expert implementation:**
```cpp
// Kernel入口 - 编译期分发
#if (ORIG_DTYPE_X == DT_FLOAT)
    ClippedSwigluOps::ClippedSwigluBase<float> op(&pipe);
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
    ClippedSwigluOps::ClippedSwigluBase<half> op(&pipe);
#endif
#if (ORIG_DTYPE_X == DT_BF16)
    ClippedSwigluOps::ClippedSwigluBase<bfloat16_t> op(&pipe);
#endif

// 模板类定义
template <typename inType>
class ClippedSwigluBase {
    GlobalTensor<inType> xGm_;
    GlobalTensor<inType> yGm_;
};
```

**vs. baseline:**
```cpp
class KernelClippedSwiglu {
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    // 全程使用float
};
```

Benefit: 支持FP16/BF16可减少50%内存带宽，模板编程避免代码重复
Trade-off: 类型转换引入少量计算开销，编译时间略有增加

---

## Variant H: 模板化多数据类型支持
Source: deep_norm

专家实现采用 C++ 模板类 KernelDeepNorm<T> 来实现对多种数据类型的支持，其中 T 可以是 half（FP16）、float（FP32）或 bfloat16_t（BF16）。这种设计允许在编译期针对不同数据类型生成最优代码，同时保持代码的复用性。在 Kernel 入口函数中，通过 TILING_KEY_IS(n) 宏结合模板实例化，实现了运行时的数据类型分发。这种模板化设计相比 baseline 的单一 FP32 实现，能够在支持的硬件上获得 2 倍的内存带宽利用率和更高的计算吞吐量。

**Expert implementation:**
```cpp
// 专家实现 - 模板化支持多数据类型
template <typename T>
class KernelDeepNorm {
public:
    GlobalTensor<T> x_gm;
    GlobalTensor<T> gx_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> z_gm;
    // ...
};

// 入口函数根据 Tiling Key 实例化不同模板
if (TILING_KEY_IS(1)) { // fp16
    KernelDeepNorm<half> op;
    op.ProcessFp16LELimit();
} else if (TILING_KEY_IS(2)) { // fp32
    KernelDeepNorm<float> op;
    op.ProcessFp32LELimit();
}
```

**vs. baseline:**
```cpp
// baseline 仅支持 FP32
AscendC::GlobalTensor<float> inputGm;
AscendC::GlobalTensor<float> weightGm;
AscendC::GlobalTensor<float> biasGm;
AscendC::GlobalTensor<float> outputGm;
```

Benefit: 支持 FP16/BF16，可获得 2 倍内存带宽提升和更高计算吞吐量
Trade-off: 代码复杂度增加，需要维护多套 Tiling Key 和处理路径

---

## Variant I: 多层级 Tiling 策略
Source: deep_norm

专家实现的 Tiling 策略根据输入维度大小、数据类型、UB 容量等多维度因素，将计算场景划分为 Short（D <= 500）、LELimit（D <= 4096）、GTLimit（4096 < D <= 8192/15360）和 Common（D > 8192/15360）四种模式。Tiling Key 的编码（0-18）结合了数据类型标识和维度范围标识，确保在不同输入规模下都能获得最优性能。

**Expert implementation:**
```cpp
// 专家实现 - 多层级 Tiling
// Tiling Key 编码：D > 15360/8192  D > 4096  D <= 4096  D <= 100
// fp32:   1110:14         0110:6    0010:2     10010:18
uint32_t dtypeKey = isShort * TILING_ISSHORT_OFFSET + 
                    upperLimit * TILING_UPPER_LIMIT_OFFSET +
                    beyondLimit * TILING_BEYOND_LIMIT_OFFSET + 
                    isFP32 * TILING_ISFP32_OFFSET +
                    isFP16 * TILING_ISFP16_OFFSET;
context->SetTilingKey(dtypeKey);
```

**vs. baseline:**
```cpp
// baseline - 简单 Tiling
const uint32_t BLOCK_DIM = 32;
const uint32_t MAX_TILE_LEN = 4096;
uint32_t tileLength = (MAX_TILE_LEN < normSize) ? MAX_TILE_LEN : normSize;
uint32_t nTiles = (normSize + tileLength - 1) / tileLength;
```

Benefit: 针对不同输入规模选择最优计算路径，最大化性能
Trade-off: Tiling 逻辑复杂，需要维护多种处理路径

---

## Variant J: 条件编译支持的TilingKey策略
Source: dequant_bias

专家实现通过TilingKey实现了对多种场景的组合支持。TilingKey的编码规则为：[activate_scale标志][bias标志][bias类型]。这种设计允许运行时场景识别、编译期优化、硬件适配。每种TilingKey对应一个完全特化的代码路径，编译器可以进行针对性优化。

**Expert implementation:**
```cpp
// 专家实现：TilingKey驱动的多场景支持
if (TILING_KEY_IS(10100)) {
    // 无activate_scale, 无bias
    DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, DTYPE_BIAS, DTYPE_Y, false> op;
} else if (TILING_KEY_IS(10113)) {
    // 有activate_scale, 有bias(bf16)
    DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, bfloat16_t, DTYPE_Y, true> op;
}
```

**vs. baseline:**
```cpp
// baseline：无TilingKey支持
// 单一固定实现
```

Benefit: 支持16种场景组合，运行时高效派发，编译期针对性优化
Trade-off: TilingKey设计复杂，需要维护映射表

---

## Variant K: Tiling Key编码策略
Source: dynamic_block_quant

专家实现使用了一种精巧的Tiling Key编码策略，将多个维度的信息编码到一个整数中。编码规则：千分位表示RoundMode（1=rint, 4=round, 7=hybrid），百位表示输入类型（1=fp16, 2=bf16），十位表示输出类型（0=fp8_e5m2, 1=fp8_e4m3fn, 2=hifloat8），个位表示kernel类型（0=normal, 1=single行, 2=large block）。这种设计使得单一算子入口可以根据输入参数动态选择不同的kernel实例，而无需在Host端进行复杂的条件判断。例如，输入fp16、输出hifloat8、round模式、normal kernel对应的key是1420。

**Expert implementation:**
```cpp
int64_t thousandDigit = tilingParam.roundMode;
int64_t hundredDigit = inputType == DT_FLOAT16 ? 1 : DIGIT_TWO;
int64_t tenDigit = 0;
if (outputType == ge::DT_FLOAT8_E4M3FN) {
    tenDigit = 1;
} else if (outputType == ge::DT_HIFLOAT8) {
    tenDigit = DIGIT_TWO;
}
int64_t digit = KERNEL_TYPE_NORMAL;
if (tilingParam.blockSizeRow == 1) {
    digit = KERNEL_TYPE_SINGLE;
} else if (maxUbAvailable == 0) {
    digit = KERNEL_TYPE_LARGE;
}
tilingParam.tilingKey = thousandDigit * DIGIT_THOUSAND + hundredDigit * DIGIT_HUNDRED 
    + tenDigit * DIGIT_TEN + digit;
```

Benefit: 单一入口支持多种kernel类型，编译期确定代码路径，运行时高效分派
Trade-off: 需要理解和维护编码规则，增加调试复杂度

---

## Variant L: Tiling Key 多分支路由策略
Source: dynamic_mx_quant

专家实现采用了极其精细的 Tiling Key 路由机制，通过 96+ 个不同的 tiling key 区分各种计算场景。这种设计的核心思想是：不同场景使用不同的优化算法，而非使用通用但次优的算法。Tiling Key 编码规则：千位 (1/2) 表示输入类型 (1=FP16, 2=BF16)、百位 (0/1/2/3) 表示输出类型、十位 (0/1/2/4) 表示计算场景（尾轴/非尾轴/优化/非优化）、个位 (0/1) 表示尾轴奇偶性。这种设计允许尾轴与非尾轴采用不同算法、奇偶 scale 数区分（奇数个 scale 需要额外的 DynamicMxQuantPost 处理）、优化与非优化路径选择。

**Expert implementation:**
```cpp
// 专家实现：96+ tiling keys 路由
#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS 1000
#define TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS 2000
#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE 1001

extern "C" __global__ __aicore__ void dynamic_mx_quant(...) {
    if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS)) {
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, true> op;
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE)) {
        // 奇数 scale 场景：先计算，再 post 处理
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, true> op;
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Process();
    }
    // ... 更多分支
}
```

**vs. baseline:**
```cpp
// baseline：单一算法处理所有场景
class KernelDynamicMxQuant {
    __aicore__ inline void Process() {
        for (uint32_t rowIdx = rowStart; rowIdx < rowEnd; rowIdx++) {
            for (uint32_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                CopyIn(rowIdx, blockIdx);
                Compute(rowIdx, blockIdx);
                CopyOut(rowIdx, blockIdx);
            }
        }
    }
};
```

Benefit: 每种场景使用最优算法，综合性能提升 30-100%；场景覆盖更全面
Trade-off: 代码量显著增加（主 kernel 文件 562 行，主要是分支逻辑）；维护成本增加

---

## Variant M: 多精度输入支持(FP16/BF16)
Source: dynamic_quant_update_scatter_v2

专家实现支持FP16和BF16两种输入数据类型，通过Tiling Key机制(0-BF16, 1-FP16, 2-DB_BF16, 3-DB_FP16)在编译期确定数据类型，避免了运行时的类型判断开销。在Tiling阶段，根据输入数据类型计算不同的UB占用大小(BF16_DB_UB_SIZE=13, FP16_DB_UB_SIZE=11)，确保内存分配的最优性。这种多精度支持策略对于混合精度训练场景尤为重要，可以在保证精度的同时获得更好的性能。

**Expert implementation:**
```cpp
// Tiling Key定义
constexpr int64_t TILING_KEY_BF16 = 0;
constexpr int64_t TILING_KEY_HALF = 1;
constexpr int64_t TILING_KEY_DB_BF16 = 2;
constexpr int64_t TILING_KEY_DB_HALF = 3;

// 根据数据类型设置Tiling Key
void SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb) const {
    if (useDb) {
        if (dataType == ge::DT_BF16) {
            context->SetTilingKey(TILING_KEY_DB_BF16);
            return;
        }
        context->SetTilingKey(TILING_KEY_DB_HALF);
        return;
    }
    if (dataType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
        return;
    }
    context->SetTilingKey(TILING_KEY_HALF);
}
```

**vs. baseline:**
```cpp
// baseline实现可能仅支持FP16
template <typename T>
class Kernel {
    // 单一类型实现...
};
```

Benefit: 支持混合精度训练场景，编译期类型确定避免运行时开销，不同精度的内存分配优化
Trade-off: 增加Tiling Key数量和代码复杂度，需要维护多个模板实例

---

## Variant N: TilingKey多模式分发
Source: embedding_dense_grad_v2

专家实现采用TilingKey区分四种计算模式：基础模式(TILING_KEY_IS(0))用于通用大规模处理；Scale模式(TILING_KEY_IS(1))支持按频率缩放梯度；确定性模式(TILING_KEY_IS(10/11))保证计算结果确定性，避免多核竞态；小维度模式(TILING_KEY_IS(100/101))针对embedding_dim ≤ 512的优化。这种设计允许一个算子入口根据输入特征自适应选择最优算法。

**Expert implementation:**
```cpp
if (TILING_KEY_IS(0)) {
    EmbeddingDenseGradV2Kernel<DTYPE_GRAD, float> op(...);
    op.Process();
} else if (TILING_KEY_IS(1)) {
    ProcessAndScale<false, false>(...);
} else if (TILING_KEY_IS(10)) {
    EmbeddingDenseGradV2DeterministKernel<float> op(...);
} else if (TILING_KEY_IS(100)) {
    EmbeddingDenseGradV2SmallDimKernel<DTYPE_GRAD, float> op(...);
}
```

**vs. baseline:**
```cpp
// Single processing mode
void Process() { ... }
```

Benefit: 根据输入特征自动选择最优算法，全场景性能优化
Trade-off: 代码体积增大，维护复杂度增加

---

## Variant O: 模板化双精度支持 (FP16/FP32)
Source: fake_quant_affine_cachemask

专家实现通过模板类FakeQuantAffineCachemaskFp32和FakeQuantAffineCachemaskFp16实现了对FP16和FP32两种数据类型的完整支持。这种设计允许根据输入数据类型自动选择最优的计算路径，充分利用昇腾芯片对不同精度计算的硬件支持。在算子定义文件中，通过配置多组DataType实现了运行时数据类型的灵活选择。Tiling阶段通过SetTilingKeyMode函数根据数据类型设置不同的Tiling Key（FP16_MODE=2, FP32_MODE=1），从而在kernel入口处分发到对应的模板实例。这种策略的优势在于：1) 避免了运行时的类型判断开销；2) 允许针对每种数据类型进行专门的优化；3) 通过编译期多态实现零开销抽象。值得注意的是，FP16和FP32的实现细节有所不同，例如FP16实现中使用了更多的Cast操作进行精度转换，而FP32实现则尽量保持原生精度计算。

**Expert implementation:**
```cpp
// 专家实现支持FP16/FP32
this->Input("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

// Tiling Key分发
void SetTilingKeyMode(ge::DataType dType) {
    switch (dType) {
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE);
            bytesPerData = BYTES_PER_DATA_FP16;
            break;
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(FP32_MODE);
            bytesPerData = BYTES_PER_DATA_FP32;
            break;
    }
}
```

**vs. baseline:**
```cpp
// baseline仅支持FP32
this->Input("x")
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
```

Benefit: 支持多种数据类型，用户可以根据精度和性能需求选择FP16或FP32，提高算子适用性；充分利用昇腾芯片对不同精度计算的硬件优化
Trade-off: 代码复杂度增加，需要维护多个模板类实现；编译产物体积增大

---

## Variant P: 多数据类型模板化支持
Source: foreach_abs

专家实现通过C++模板机制实现了真正的多数据类型支持。在Kernel端，foreach_abs函数使用TILING_KEY区分不同数据类型：TILING_KEY_IS(1)对应half，TILING_KEY_IS(2)对应float，TILING_KEY_IS(4)对应bfloat16_t。这种设计允许同一份算子代码通过编译时模板实例化生成针对不同数据类型的高效实现。与baseline仅支持float类型相比，专家实现的数据类型支持更加完整，覆盖了深度学习训练和推理中常用的FP16、FP32、BF16三种类型。

**Expert implementation:**
```cpp
std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
this->Input("x")
    .ParamType(DYNAMIC)
    .DataType(tensor_dtype_list)
    .Format(format_list);
```

**vs. baseline:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});
```

Benefit: 支持主流深度学习数据类型，提高算子通用性和适用场景
Trade-off: 增加代码复杂度，需要维护多种数据类型的测试用例

---

## Variant Q: 模板化架构设计
Source: foreach_abs

专家实现采用三层模板结构实现代码复用：KernelForeachBase提供基础功能，KernelForeachUnary提供单目运算通用流程，ForeachTriangle提供具体计算逻辑。这种架构使得新的foreach一元算子只需定义具体的OpAdapter和Tiling Key即可，无需重复编写CopyIn/CopyOut等通用逻辑。

**Expert implementation:**
```cpp
template <typename T, typename P, TriangleOp<P>* op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
class ForeachTriangle
    : public KernelForeachUnary<T, ForeachTriangle<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>
{
    // 模板继承实现
};
```

**vs. baseline:**
```cpp
class KernelForeachAbs {
    // 手动实现Init/Process/CopyIn/Compute/CopyOut
};
```

Benefit: 代码高度复用，开发新算子效率提升，维护成本降低
Trade-off: 模板代码可读性较差，编译时间增加

---

## Variant R: TILING_KEY多路径分发
Source: foreach_add_scalar_list

**Expert implementation:**
```cpp
if (TILING_KEY_IS(1)) {
    ForeachOneScalarListBinary<half, half, AddsAdapter<half>, 1, 1> op;
    op.Init(...); op.Process();
} else if (TILING_KEY_IS(2)) {
    ForeachOneScalarListBinary<float, float, AddsAdapter<float>, 1, 1> op;
    op.Init(...); op.Process();
}
```

Benefit: 消除运行时类型判断开销，提升执行效率
Trade-off: 生成的二进制文件体积增大

---

## Variant S: 数据类型感知的 Tiling 策略
Source: gemma_rms_norm

专家实现根据输入数据类型动态调整 UB（Unified Buffer）使用策略。不同数据类型有不同的对齐要求和 UB 容量计算方式。通过 SetByDtype 函数统一处理类型相关的参数（如 dataPerBlock），确保内存访问对齐和向量化效率。FP16/BF16 使用 GEMMA_UB_FACTOR_B16=10240，而 FP32 使用 GEMMA_UB_FACTOR_B32=8192。

**Expert implementation:**
```cpp
// 专家实现 - 数据类型感知的 Tiling
void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock) {
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;  // 16
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;  // 16
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;        // 8
            break;
    }
}
uint64_t ubFactor = (xDtypeKey == DTYPE_KEY_FP32) ? GEMMA_UB_FACTOR_B32 : GEMMA_UB_FACTOR_B16;
```

**vs. baseline:**
```cpp
// baseline - 固定 tileLength
const uint32_t MAX_TILE_LEN = 4096;
uint32_t tileLength = (MAX_TILE_LEN < normSize) ? MAX_TILE_LEN : normSize;
```

Benefit: 根据数据类型选择最优 UB 分块策略，最大化内存带宽利用率
Trade-off: Tiling 计算略微复杂，但收益显著

---

## Variant T: 双模板策略（KernelRmsNorm vs KernelRmsNormSplitD）
Source: gemma_rms_norm

专家实现提供了两种计算策略，通过 TILING_KEY 动态选择：KernelRmsNorm 适用于归一化维度较小的情况，整行数据可放入 UB，单次计算完成；KernelRmsNormSplitD 适用于归一化维度较大的情况，需要对列维度进行二次分块。这种双策略设计确保算子在不同输入规模下都能高效运行。baseline实现只有单一策略，在大维度场景下性能会显著下降。

**Expert implementation:**
```cpp
// 专家实现 - 双策略选择
extern "C" __global__ __aicore__ void gemma_rms_norm(...) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
    }
}
```

**vs. baseline:**
```cpp
// baseline - 单一策略
for (uint32_t rowIdx = 0; rowIdx < this->rowsPerCore; rowIdx++) {
    for (uint32_t tileId = 0; tileId < this->nTiles; tileId++) {
        // 固定 tile 处理
    }
}
```

Benefit: 适应不同输入规模，小维度低延迟，大维度高吞吐
Trade-off: 需要维护两套 Kernel 代码，增加了代码维护成本

---

## Variant U: TilingKey多类型编译期分发
Source: grouped_dynamic_mx_quant

专家实现采用TilingKey机制实现多数据类型的编译期分发，TilingKey编码规则为：十位数表示输入类型（1=FP16, 2=BF16），个位数表示输出类型（1=E4M3FN, 2=E5M2）。这种设计避免了运行时的类型判断开销，编译器可以针对每种类型组合进行指令优化。Kernel端通过if constexpr结合IsSame类型trait实现类型感知的代码路径选择，FP16需要额外转换为BF16进行指数提取，而BF16可以直接操作。

**Expert implementation:**
```cpp
#define TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS 11
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS 21
#define TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS 12
#define TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS 22

if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
    GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<half, fp8_e4m3fn_t> op;
    op.Init(x, groupIndex, y, mxScale, tilingData);
    op.Process();
} else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
    GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<bfloat16_t, fp8_e4m3fn_t> op;
    ...
}

if constexpr (IsSame<T, half>::value) {
    AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xBF16RegTensor, xRegTensor, p0);
}
```

**vs. baseline:**
```cpp
extern "C" __global__ __aicore__ void grouped_dynamic_mx_quant(...) {
    GroupedDynamicMxQuantBaseFP8<half, fp8_e4m3fn_t> op;
    op.Init(x, groupIndex, y, mxScale, tilingData);
    op.Process();
}
```

Benefit: 编译期类型确定，零运行时开销；支持4种类型组合；编译器可进行针对性指令优化
Trade-off: 代码量增加（4个TilingKey分支）；需要为每种类型组合生成独立模板实例

---

## Variant V: 数据类型转换的向量化优化
Source: layer_norm_v3

在Kernel端处理混合精度时，专家实现使用LoadTensorForDtypeTIn和StoreTensorForDtypeTOut模板函数，通过__VEC_SCOPE__宏启用向量化执行。关键优化包括使用DIST_UNPACK_B16/DIST_PACK_B32指令实现高效的FP16/BF16↔FP32转换，利用RegTensor和MaskReg实现寄存器级向量化操作，通过castTraitB162B32/castTraitB322B16配置精确的舍入和饱和模式。相比baseline直接使用float类型的简单处理，专家实现在混合精度场景下可获得2-4倍的内存带宽利用率提升。

**Expert implementation:**
```cpp
template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst,
                                             MaskReg& preg, uint32_t offset) {
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}
```

**vs. baseline:**
```cpp
// 直接使用float类型，无类型转换
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopy(xLocal, xGm[rowIdx * this->tileLength], this->tileLength);
```

Benefit: 高效的向量化类型转换，最大化内存带宽利用率
Trade-off: 代码复杂度增加，需要模板特化处理不同类型

---

## Variant W: Mean/Rstd输出的类型适配
Source: layer_norm_v3

专家实现将mean和rstd（或variance）作为算子输出（而非仅内部计算），并支持独立的输出数据类型控制。通过IsSameType<M, float>::value编译期判断，在输出类型为FP16/BF16时，执行CastBatchMeanLastout进行向量化类型转换。这避免了重复计算，同时为下游算子提供了灵活的精度接口。

**Expert implementation:**
```cpp
if constexpr(!IsSameType<M, float>::value) {
    // float to bfloat16 or float16, input continue and output each repeat have only half value
    CastBatchMeanLastout(cacheCount);
    // DMA copy with stride adjustment
}
```

Benefit: 支持反向传播，避免重复计算，提供灵活精度接口
Trade-off: 需要额外的内存空间存储mean/rstd

---

## Variant X: 类型转换精细化控制
Source: layer_norm_v4

在混合精度场景，专家实现通过Cast指令进行显式类型转换，并精细控制转换时机。数据加载到UB后立即转为FP32计算，利用更高精度减少累积误差，输出时再转回目标精度。baseline完全忽略类型转换，假设所有数据都是FP32。

**Expert implementation:**
```cpp
if (sizeof(Tfm) == SINGLE_CONST_TWO) {
    Cast(xLocal, xLocal.ReinterpretCast<Tfm>()[tileLength], RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
}
// 输出精度控制
if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
    Cast(yLocal.ReinterpretCast<Tfm>(), yLocal, RoundMode::CAST_ROUND, tileLength);
}
```

Benefit: 中间计算使用更高精度，减少数值误差，支持真正的混合精度计算流程
Trade-off: 增加类型转换开销，需要仔细设计转换时机以隐藏延迟

---

## Variant Y: TilingKey场景编码
Source: linear_index

专家实现使用TilingKey来编码不同的计算场景，通过位运算组合不同的特征：数据类型（int32=1, int64=2）、合轴模式（COMBINE_DIM0=10, COMBINE_DIM1=20）。这种设计允许在kernel入口根据TilingKey快速分发到对应的模板实例，避免了运行时的复杂条件判断。Host端在tiling阶段就确定了计算模式，通过tilingKey传递给kernel，实现了Host-Device协同优化。

**Expert implementation:**
```cpp
const int DT_INT32_TYPE = 1;
const int DT_INT64_TYPE = 2;
const int COMBINE_DIM0 = 10;
const int COMBINE_DIM1 = 20;

if (ge::DT_INT32 == indicesDtype) {
    tilingKey += DT_INT32_TYPE;
} else if (ge::DT_INT64 == indicesDtype) {
    tilingKey += DT_INT64_TYPE;
}
if (combine) {
    if (varShapeSize == THREE_DIM && dim == 0) {
        tilingKey += COMBINE_DIM0;
    } else if (varShapeSize == THREE_DIM && dim == 1) {
        tilingKey += COMBINE_DIM1;
    }
}
context->SetTilingKey(tilingKey);

// Kernel端
if (TILING_KEY_IS(1)) { CALL_OP_IMPL(int, 0); }
else if (TILING_KEY_IS(2)) { CALL_OP_IMPL(int64_t, 0); }
else if (TILING_KEY_IS(11)) { CALL_OP_IMPL(int, 1); }
else if (TILING_KEY_IS(12)) { CALL_OP_IMPL(int64_t, 1); }
else if (TILING_KEY_IS(21)) { CALL_OP_IMPL(int, 2); }
else if (TILING_KEY_IS(22)) { CALL_OP_IMPL(int64_t, 2); }
```

**vs. baseline:**
```cpp
uint32_t indicesDtype = 1;
uint32_t isCombineDim0 = 0;
uint32_t isCombineDim1 = 0;
if (combine_val && ndim == 3) {
    if (axis_val == 0) isCombineDim0 = 1;
    else if (axis_val == 1) isCombineDim1 = 1;
}
tiling.set_indicesDtype(indicesDtype);
tiling.set_isCombineDim0(isCombineDim0);
tiling.set_isCombineDim1(isCombineDim1);
```

Benefit: 消除了运行时类型判断和模式判断的开销，编译器可以针对每种场景生成最优代码
Trade-off: 需要维护更多的kernel实例，增加了代码维护复杂度

---

## Variant Z: Tiling Key策略与多分支编译优化
Source: masked_scatter_with_position

专家实现使用Tiling Key将不同计算模式编码到不同核函数分支，实现编译期优化。Tiling Key = 100 * pattern + isInt64，生成四个分支：BA_INT32(100)、BA_INT64(101)、AB_INT32(200)、AB_INT64(201)。优势包括：编译期优化（每个分支只编译特定模式，编译器深度优化）、零运行时开销（通过`TILING_KEY_IS`宏在编译期确定分支）、硬件资源适配（根据数据规模自动选择32位或64位索引）。baseline所有逻辑在单一核函数中通过运行时条件判断处理。

**Expert implementation:**
```cpp
// 专家实现 - Tiling Key策略
#define BA_INT32_TILING_KEY 100
#define BA_INT64_TILING_KEY 101
#define AB_INT32_TILING_KEY 200
#define AB_INT64_TILING_KEY 201

extern "C" __global__ __aicore__ void masked_scatter_with_position(...) {
    if (TILING_KEY_IS(BA_INT32_TILING_KEY)) {
        MaskedScatterWithPositionSimt<DTYPE_X, uint32_t, PATTERN_BA> op(&tilingData);
    } else if (TILING_KEY_IS(BA_INT64_TILING_KEY)) {
        MaskedScatterWithPositionSimt<DTYPE_X, uint64_t, PATTERN_BA> op(&tilingData);
    }
    // ...
}
```

**vs. baseline:**
```cpp
// baseline - 无Tiling Key，单一核函数
extern "C" __global__ __aicore__ void masked_scatter_with_position_custom(...) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedScatterWithPosition op;
    op.Init(...);
    op.Process();  // 所有逻辑在一个函数内
}
```

Benefit: 编译期分支选择，深度编译优化，零运行时开销，自适应32/64位索引
Trade-off: 核函数代码量增加（多个分支），需要理解Tiling Key机制

---

## Variant 27: 多维度Tiling策略
Source: max_pool_grad_with_argmax_common

专家实现针对MaxPoolGradWithArgmax算子的计算特征，设计了四种细粒度tiling策略来适应不同的数据shape：merge_hwc策略将H、W、C三个维度合并处理，适用于H×W较小但C较大的场景；merge_wc策略合并W和C维度，适用于H较大但W×C适中的场景；bigc策略针对超大C维度的场景；simt策略当数据shape不适合向量化处理时回退到标量模式。每种策略都有对应的TilingKey（500-804），在运行时根据输入shape自动选择最优策略。这种多维度的tiling策略确保了在不同场景下都能获得接近最优的性能。

**Expert implementation:**
```cpp
// 专家实现：多TilingKey策略选择
#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC 500
#define CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC 501
#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC 600
#define NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC 700
#define MPGA_SIMT_NHWC 802

extern "C" __global__ __aicore__ void max_pool_grad_with_argmax(...) {
    if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC)) {
        MaxPoolGradWithArgmaxKernelNHWCMergeHWCBase<DTYPE_X, DTYPE_ARGMAX, 0, VER_NORMAL> op;
        op.Init(...); op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC)) {
        MaxPoolGradWithArgmaxKernelNHWCMergeWCBase<DTYPE_X, DTYPE_ARGMAX, int32_t, 0, VER_NORMAL> op;
        op.Init(...); op.Process();
    } // ... 更多策略
}
```

**vs. baseline:**
```cpp
// baseline实现：简单的C维度tiling
uint32_t max_tile_c = 1024;
uint32_t tile_c = max_tile_c < C ? max_tile_c : C;
tile_c = (tile_c / 16) * 16;  // 对齐到16

// 简单的任务分配
uint32_t total_tasks = N * H_out * W_out;
if (total_tasks % BLOCK_DIM != 0) {
    total_tasks = ((total_tasks + BLOCK_DIM - 1) / BLOCK_DIM) * BLOCK_DIM;
}
```

Benefit: 针对不同数据shape选择最优策略，在各种场景下都能获得接近峰值性能
Trade-off: 增加了代码复杂度和维护成本；需要在tiling阶段精确计算选择哪种策略

---

## Variant 28: 边界范围检查优化
Source: max_pool_grad_with_argmax_common

专家实现通过IS_CHECK_RANGE模板参数实现编译期的边界检查控制。在tiling阶段会根据padding模式判断是否需要进行边界检查，然后通过不同的TilingKey（如500 vs 501）选择带检查或不带检查的kernel版本。这种设计避免了在kernel内部进行运行时的条件判断，同时也确保了在需要时（如SAME padding模式）能够正确过滤越界索引。FilterMask函数使用MicroAPI的比较和掩码操作实现向量化边界检查，比逐元素检查更高效。

**Expert implementation:**
```cpp
// 专家实现：向量化边界检查
__aicore__ inline void FilterMask(MicroAPI::MaskReg& preg, 
                                  MicroAPI::RegTensor<int32_t>& hIndexReg,
                                  MicroAPI::RegTensor<int32_t>& wIndexReg, 
                                  MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                  MicroAPI::RegTensor<int32_t>& wMaxReg, 
                                  MicroAPI::RegTensor<int32_t>& hMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = 
        AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = 
        AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    // h >= 0 && h < hMax
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);
    // w >= 0 && w < wMax
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    // 合并mask
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
}
```

**vs. baseline:**
```cpp
// baseline实现：无边界检查
// 直接计算索引，不做边界验证
uint32_t h_idx = idx / W_in;
uint32_t w_idx = idx % W_in;
uint32_t grad_in_base = n * H_in * W_in * C + h_idx * W_in * C + w_idx * C + c0;
```

Benefit: 编译期确定是否需要边界检查；向量化边界过滤比标量判断高效
Trade-off: 需要额外的tiling key；增加代码分支

---

## Variant 29: 多策略自适应调度（Tiling Key 分发）
Source: max_pool_with_argmax_v3

专家实现包含10+种kernel策略：NHWC Small C（Gather/Scatter向量化）、NHWC Big C（大channel分块）、SIMT NCHW/NHWC（通用路径）、Big Kernel（大kernel分块）、Gather Kernel等。Host端tiling算法通过场景识别（channel大小、格式、kernel尺寸）选择最优策略。

**Expert implementation:**
```cpp
// 专家实现 - 多策略分发
if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_SMALL_C)) {
    MaxPoolWithArgmaxV3SmallC<DTYPE_X, DTYPE_ARGMAX, 0> op(&pipeBase, tilingData);
} else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NCHW)) {
    MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, NCHW, false> op(tilingData);
} else if (TILING_KEY_IS(BIG_KERNEL_FORMAT_NCHW)) {
    MaxPoolWithArgMaxV3BigKernel::MaxPoolWithArgmaxV3BigKernel<DTYPE_X, float, DTYPE_ARGMAX> op(...);
}
// ... 更多策略
```

**vs. baseline:**
```cpp
// baseline 单一路径
class KernelMaxPoolWithArgmaxV3 {
    __aicore__ inline void Process() {
        // 唯一实现
    }
};
```

Benefit: 每种场景都能获得最优性能，避免'一刀切'带来的性能损失
Trade-off: 开发和维护成本大幅增加，需要全面的测试覆盖

---

## Variant 30: 模板多态架构与TilingKey
Source: modulate

专家实现采用C++模板+继承的多态架构，结合TilingKey机制实现运行时分派。ModulateBase基类模板包含通用成员和辅助函数，ModulateB/ModulateL/ModulateD分别特化B/L/D维度分块。TilingKey机制允许Host端动态选择策略（TILING_KEY_IS(0/1/2)）。这种设计的优势是零开销抽象（模板编译期实例化，无运行时虚函数开销）、代码复用（基类封装通用逻辑）、运行时选择、类型安全。

**Expert implementation:**
```cpp
// 专家实现: 模板多态架构
template <typename T>
class ModulateBase { ... };
template <typename T>
class ModulateB : public NormModulate::ModulateBase<T> { ... };
template <typename T>
class ModulateL : public NormModulate::ModulateBase<T> { ... };
template <typename T>
class ModulateD : public NormModulate::ModulateBase<T> { ... };
extern "C" __global__ __aicore__ void modulate(...) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        NormModulate::ModulateB<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NormModulate::ModulateL<DTYPE_X> op(&pipe, &tilingData);
        ...
    }
}
```

**vs. baseline:**
```cpp
// baseline: 单一Kernel类
class ModulateKernel {
public:
    __aicore__ inline ModulateKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y, 
                                int64_t inputB, int64_t inputL, int64_t inputD)
    ...
};
extern "C" __global__ __aicore__ void modulate(...) {
    GET_TILING_DATA(tilingData, tiling);
    ModulateKernel op;
    op.Init(x, scale, shift, y, tilingData.inputB, tilingData.inputL, tilingData.inputD);
    op.Process();
}
```

Benefit: 性能提升20-50%（取决于shape分布），无运行时开销
Trade-off: 代码复杂度显著增加，需要理解模板和多重继承

---

## Variant 31: FP16/FP32双精度Tiling Key
Source: multi_scale_deformable_attention_grad

专家实现通过SetTilingKeyMode函数根据输入数据类型设置不同的Tiling Key(FP32_MODE=0, FP16_MODE=1)。这种设计允许kernel根据数据类型选择不同的计算精度和优化策略。对于FP16，可以使用更大的向量宽度和更高的并行度；对于FP32，则需要更谨慎地管理UB内存。这种区分使得算子可以在精度和性能之间灵活权衡。此外，Tiling Key的设置还为后续可能的混合精度计算预留了扩展空间。

**Expert implementation:**
```cpp
void MultiScaleDeformableAttentionGradTiling::SetTilingKeyMode(ge::DataType dType_str) {
    switch (dType_str) {
        case ge::DT_FLOAT:
            TilingContext->SetTilingKey(FP32_MODE);
            break;
        case ge::DT_FLOAT16:
            TilingContext->SetTilingKey(FP16_MODE);
            break;
        default:
            TilingContext->SetTilingKey(FP32_MODE);
            break;
    }
}
```

**vs. baseline:**
```cpp
// baseline仅支持FP32
.DataType({ge::DT_FLOAT})
```

Benefit: 支持FP16/FP32双精度，允许在精度和性能之间灵活权衡，FP16可获得2倍内存带宽提升
Trade-off: 需要维护两套Tiling逻辑，代码复杂度增加

---

## Variant 32: 多Tiling Key策略实现多路径优化
Source: multi_scale_deformable_attn_function

专家实现通过TilingKey机制（1002/1004/1008/2002/2004/2008）实现了基于输入特征的多路径优化。TilingKey的编码规则为：embed_dims/16 * 1000 + num_points。例如，TilingKey=1002表示embed_dims=16、num_points=2。这种策略允许针对不同embed_dims（16或32）和num_points（2/4/8）组合进行专门的优化，包括UB分配策略、向量化参数、循环展开深度等。在Host端，GroupPoints函数根据numPoints自动选择最优的点分组策略（2/4/8点每组），最大化计算效率。

**Expert implementation:**
```cpp
// Host端TilingKey计算
uint64_t TilingKey = optPoint == 1 ? (embedDims / EMBEDDIMS_SIXTEEN) * TILING_KEY_WEIGHT + point : 0;

// Kernel端根据TilingKey选择模板实例
if (TILING_KEY_IS(1002)) {
    KernelMultiScaleDeformableAttnOpt<2, 16> op(...);
} else if (TILING_KEY_IS(1004)) {
    KernelMultiScaleDeformableAttnOpt<4, 16> op(...);
}

// Host端点分组策略
std::tuple<uint64_t, uint64_t> GroupPoints(uint64_t numPoints)
{
    if (numPoints % NUM_POINTS_EIGHT == 0) {
        return std::make_tuple(NUM_POINTS_EIGHT, numPoints / NUM_POINTS_EIGHT);
    }
    if (numPoints % NUM_POINTS_FOUR == 0) {
        return std::make_tuple(NUM_POINTS_FOUR, numPoints / NUM_POINTS_FOUR);
    }
    if (numPoints % NUM_POINTS_TWO == 0) {
        return std::make_tuple(NUM_POINTS_TWO, numPoints / NUM_POINTS_TWO);
    }
    return std::make_tuple(1, numPoints);
}
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 针对不同输入特征进行专门优化，最大化计算效率
Trade-off: 增加了代码量和维护成本，需要管理多个模板实例

---

## Variant 33: 模板化多数据类型支持
Source: norm_common

专家实现采用C++模板编程技术，通过Tfm（输入/输出数据类型）和Tweight（权重/偏置数据类型）两个模板参数实现灵活的数据类型组合支持。这种设计允许在编译期生成针对不同数据类型的特化代码，避免了运行时的类型判断开销。baseline实现仅支持float32单一数据类型，而专家实现支持float32/float16/bfloat16三种数据类型的组合（如输入FP16+权重FP32、输入BF16+权重BF16等），共15种不同的Tiling Key对应不同的数据类型组合策略。这种多数据类型支持对于现代深度学习框架至关重要，因为不同场景对精度和性能的需求不同——训练时可能需要FP32保证精度，推理时可以使用FP16/BF16提升吞吐量。

**Expert implementation:**
```cpp
template <typename Tfm, typename Tweight>
class LayerNormV4SingleRead {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, ...)
    {
        xGm.SetGlobalBuffer((__gm__ Tfm*)x + xGmOffset, xGmSize);
        gammaGm.SetGlobalBuffer((__gm__ Tweight*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ Tweight*)beta, rowSize);
    }
    
    __aicore__ inline void ProcessBasicBlock(uint32_t nRow, ...)
    {
        if (sizeof(Tfm) == SINGLE_CONST_TWO) {
            Cast(xLocal, xLocal.ReinterpretCast<Tfm>()[tileLength], RoundMode::CAST_NONE, tileLength);
        }
    }
};

// 通过Tiling Key选择不同数据类型组合
if (TILING_KEY_IS(100)) {
    INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(float, float);
} else if (TILING_KEY_IS(110)) {
    INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(half, float);
}
```

**vs. baseline:**
```cpp
class NormCommonCustom {
    GlobalTensor<float> x_gm;
    GlobalTensor<float> weight_gm;
    GlobalTensor<float> bias_gm;
    GlobalTensor<float> y_gm;
    // 没有模板参数，类型硬编码
};
```

Benefit: 支持FP32/FP16/BF16多种数据类型组合，推理时可使用低精度提升吞吐量，训练时可使用FP32保证精度，性能提升可达2-4倍
Trade-off: 代码复杂度增加，需要为每种数据类型组合生成独立的Tiling Key和kernel实例，二进制体积增大

---

## Variant 34: Welford算法提升数值稳定性
Source: norm_common

专家实现提供基于Welford算法的Tiling策略（LayerNormV4WelfordTiling），该算法是计算方差的数值稳定方法。传统两-pass算法（先算均值再算方差）在数据量较大时可能产生精度损失，而Welford算法通过增量式更新均值和M2（方差中间值），避免了浮点数累加的精度问题。该策略通过LayerNormV4_400系列Tiling Key启用，适用于对精度要求极高的场景。

**Expert implementation:**
```cpp
// 专家实现：Welford策略定义
class LayerNormV4WelfordTiling : public LayerNormV4TilingBase {
public:
    LayerNormV4TilingDataWelford td_;
protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override { return 400 + dtypeKey_; }
};

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataWelford)
TILING_DATA_FIELD_DEF(int64_t, M);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTimes);
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_400, LayerNormV4TilingDataWelford)
```

**vs. baseline:**
```cpp
// baseline：传统两-pass算法
float row_mean = row_sum / norm_size;
float row_var = row_var_sum / norm_size;
float row_std = sqrt(row_var + eps);
```

Benefit: 提升数值稳定性，在大数据量或极端数值场景下保持精度
Trade-off: 计算复杂度略高，性能可能略低于传统算法

---

## Variant 35: Tiling Key编码动态路由
Source: rms_norm_quant

专家实现使用6-bit编码的Tiling Key来区分不同的计算配置：[gemma_mode][beta][precision][fast/slice][dtype]。这种设计允许在运行时根据输入特性动态选择最优内核实现。相比baseline的单一实现，专家实现提供了16种配置组合，每种都经过特定优化。

**Expert implementation:**
```cpp
uint32_t CalcNormTilingKey(NormCommonTilingData1* tilingDataPtr, ge::DataType dtype, bool useBeta) {
    uint32_t tilingKey = static_cast<uint32_t>(tilingDataPtr->get_gemmaMode() ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useBeta ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(tilingDataPtr->get_highPrecisionMode() ? 1 : 0);
    bool useSlice = tilingDataPtr->get_numCol() > tilingDataPtr->get_sliceSize();
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useSlice ? 1 : 0);
    tilingKey = (tilingKey << BIT_OFFSET_SIX) + static_cast<uint32_t>(dtype);
    return tilingKey;
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 8;
context->SetBlockDim(BLOCK_DIM);
```

Benefit: 零运行时开销，缓存友好，16种配置独立优化
Trade-off: 代码量增加，需要维护更多模板实例

---

## Variant 36: 模板化多类型支持架构
Source: scaled_masked_softmax_grad_v2

专家实现采用C++模板编程实现多数据类型支持，通过template <typename T>定义核心类，在Kernel入口处通过TilingKey选择具体实例化类型（half/bfloat16_t/float）。这种设计实现代码复用、编译期零开销、类型安全，并具备良好的扩展性。特别值得注意的是，专家实现通过IsSameType<T, float>::value在编译期判断类型，实现float类型的特殊处理逻辑（无需Cast转换），这是典型的C++模板元编程技巧。

**Expert implementation:**
```cpp
template <typename T>
class ScaledMaskedSoftmaxGradV2NormHeadDim : public ScaledMaskedSoftmaxGradV2Base<T> {
    // ...
};

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2(...) {
    if (TILING_KEY_IS(1001)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<half> op;
    } else if (TILING_KEY_IS(1000)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<bfloat16_t> op;
    } else if (TILING_KEY_IS(1002)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<float> op;
    }
}
```

**vs. baseline:**
```cpp
class KernelScaledMaskedSoftmaxGradV2 {
    AscendC::GlobalTensor<float> gradOutputGm;
    AscendC::GlobalTensor<float> softmaxOutputGm;
    AscendC::GlobalTensor<float> gradInputGm;
};
```

Benefit: 一套代码支持FP16/BF16/FP32三种数据类型，编译期实例化零运行时开销，新增数据类型只需添加模板实例化
Trade-off: 代码复杂度增加，需要理解C++模板元编程；Kernel入口代码冗长，需要维护TilingKey与类型的映射关系

---

## Variant 37: 分层Tiling策略（Norm vs Large HeadDim）
Source: scaled_masked_softmax_grad_v2

专家实现的核心性能优化之一是根据headDim大小选择不同的计算策略，通过TilingKey（1000/1001/1002 vs 2000/2001/2002）进行区分。Norm HeadDim（≤1024）使用SoftmaxGrad高阶API，适用于大多数Transformer场景（典型headDim为64/128）；Large HeadDim（>1024）手动实现Softmax梯度计算，通过WholeReduceSum实现高效的行内归约，更大的tiling块减少循环次数。

**Expert implementation:**
```cpp
constexpr uint64_t MAX_NORM_HEAD_DIM = 1024;
constexpr uint64_t NORM_HEADDIM_TILING_KEY = 1000;
constexpr uint64_t LARGE_HEADDIM_TILING_KEY = 2000;

ge::graphStatus ScaledMaskedSoftmaxGradV2Tiling::SetTilingKey() const {
    uint64_t tilingKey = (paddedHeadDim <= MAX_NORM_HEAD_DIM ? NORM_HEADDIM_TILING_KEY : LARGE_HEADDIM_TILING_KEY);
    if (dataType == ge::DT_FLOAT16) {
        tilingKey += TILING_KEY_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey += TILING_KEY_FP32;
    }
    context->SetTilingKey(tilingKey);
}
```

**vs. baseline:**
```cpp
const uint32_t TILE_SIZE = 8192;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    tiling.set_tileSize(TILE_SIZE);
}
```

Benefit: 针对不同场景选择最优算法路径，Norm场景使用高阶API获得最佳性能，Large场景手动优化避免API开销
Trade-off: 代码复杂度增加，需要维护两套实现；TilingKey数量增加，调试难度提升

---

## Variant 38: 模板化Kernel设计与TilingKey类型分发
Source: scaled_masked_softmax_v2

专家实现采用C++模板编程技术，将Kernel实现抽象为ScaledMaskedSoftmaxV2<T>模板类，支持float、half、bfloat16_t三种数据类型。在Kernel入口函数中，通过TILING_KEY_IS宏实现运行时类型分发：TilingKey为0时实例化float类型，为1时实例化half类型，为2时实例化bfloat16_t类型。Host端在SetTilingKey函数中根据输入数据类型设置对应的TilingKey。这种设计使得同一套Kernel代码可以服务多种数据类型，同时通过编译期模板实例化保证了类型特化的执行效率。

**Expert implementation:**
```cpp
// 专家实现支持多类型
if (TILING_KEY_IS(0)) {
    AscendC::ScaledMaskedSoftmaxV2<float> op;
    op.Init(x, mask, y, tilingData);
    op.Process();
} else if (TILING_KEY_IS(1)) {
    AscendC::ScaledMaskedSoftmaxV2<half> op;
    op.Init(x, mask, y, tilingData);
    op.Process();
} else if (TILING_KEY_IS(2)) {
    AscendC::ScaledMaskedSoftmaxV2<bfloat16_t> op;
    op.Init(x, mask, y, tilingData);
    op.Process();
}

// Host端
this->Input("x")
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

**vs. baseline:**
```cpp
// baseline 仅支持 float
class ScaledMaskedSoftmaxV2Custom {
    GlobalTensor<float> xGm;
    GlobalTensor<float> maskGm;
    GlobalTensor<float> yGm;
    // ...
};

this->Input("x")
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});
```

Benefit: 支持混合精度训练场景，避免额外的数据类型转换开销，性能提升20-30%
Trade-off: 代码复杂度增加，需要维护多套类型相关的逻辑

---

## Variant 39: 差异化数据类型转换策略
Source: scaled_masked_softmax_v2

在Compute阶段，专家实现针对不同数据类型采用了差异化的计算策略。对于bfloat16_t类型，先将输入转换为float进行scale计算，然后使用SelectWithBytesMask，全程在float精度下完成；对于half类型，先在half精度下完成scale和mask操作，再转换为float进行softmax；对于float类型，直接使用原生float计算。这种差异化策略的原因是：softmax计算涉及指数运算，需要较高精度，因此统一在float下进行；而scale和mask是线性操作，在较低精度下也能获得较好结果，提前转换可以减少数据移动。

**Expert implementation:**
```cpp
// 专家实现差异化策略
if constexpr (std::is_same<T, bfloat16_t>::value) {
    Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
    Muls(scaledMaskedX, scaledMaskedX, static_cast<float>(scale), this->elePerIter);
    AscendC::SelectWithBytesMask(scaledMaskedX, scaledMaskedX, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
} else {
    Muls(xTensor, xTensor, static_cast<T>(scale), this->elePerIter);
    if constexpr (std::is_same<T, half>::value) {
        AscendC::SelectWithBytesMask(xTensor, xTensor, static_cast<T>(MASK_VAL), maskTensor, sharedBuffer, selectShapeInfo);
        Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
    } else {
        AscendC::SelectWithBytesMask(scaledMaskedX, xTensor, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
    }
}
```

**vs. baseline:**
```cpp
// baseline 统一float处理
AscendC::Muls(scaledLocal, xLocal, scale, tileLength);
AscendC::Add(scaledLocal, scaledLocal, maskLocal, tileLength);
AscendC::ReduceMax(maxLocal, scaledLocal, sharedLocal, tileLength);
```

Benefit: 在保证数值稳定性的前提下最小化数据类型转换开销，提升混合精度场景性能15-25%
Trade-off: 代码分支增加，需要仔细验证各分支的数值正确性

---

## Variant 40: 全面的数据类型覆盖
Source: scatter_elements_v2

专家实现支持7种输入数据类型（float, half, int32, uint8, int8, bfloat16）和2种索引类型（int32, int64），通过模板参数化和编译期tiling key编码实现。在Kernel入口使用TILING_KEY_IS宏进行分支分发，每种类型组合生成独立的kernel实例，避免运行时类型判断开销。这种设计使得算子可以处理各种实际场景，从低精度量化模型到高精度训练场景都能支持。

**Expert implementation:**
```cpp
// Expert: 支持24种类型组合
if (TILING_KEY_IS(111)) {
    CALL_OP_IMPL(float, int, 1);
} else if (TILING_KEY_IS(112)) {
    CALL_OP_IMPL(float, int, 2);
} else if (TILING_KEY_IS(121)) {
    CALL_OP_IMPL(float, long, 1);
}
// ... 共6×2×2=24种组合
```

**vs. baseline:**
```cpp
// Baseline: 仅支持float+int固定类型
KernelScatterElementsV2Baseline<float, int> op;
op.Init(var, indices, updates, ...);
```

Benefit: 支持各种实际应用场景，从低精度量化到高精度训练，零运行时类型判断开销
Trade-off: 代码体积增加，需要生成更多kernel实例

---

## Variant 41: Tiling Key数据类型区分
Source: sparse_to_dense

专家实现通过Tiling Key机制区分output size是否超过int32范围（使用uint32_t还是uint64_t作为内部计算类型）。这种设计允许在大规模数据场景下使用64位索引避免溢出，而在小规模场景下使用32位索引减少寄存器压力和内存占用。Tiling Key的生成逻辑在GetTilingKey()中实现，基于MAX_INT32_NUM阈值进行判断，这种动态选择策略使得单个算子实现可以覆盖从KB级到GB级的数据规模。

**Expert implementation:**
```cpp
uint64_t SparseToDenseTiling::GetTilingKey() const {
    int64_t tilingKey = 1000000;
    int64_t factor = 10;
    int64_t sizeType = (outSize_ > MAX_INT32_NUM) ? 1 : 0;
    tilingKey += (factor * sizeType + isValuesScalar_);
    return tilingKey;
}
```

Benefit: 动态选择32/64位计算类型，兼顾小规模性能和大规模正确性
Trade-off: 增加了Tiling Key的复杂度，需要额外的编译期选择逻辑

---

## Variant 42: Values标量/张量模式支持
Source: sparse_to_dense

专家实现通过模板参数isScalar支持两种values输入模式：标量模式（isScalar = true）所有稀疏位置使用同一个值，适用于重复填充场景；张量模式（isScalar = false）每个稀疏位置使用values张量中对应位置的值。这种设计通过编译期模板特化避免了运行期的条件分支，提高了执行效率。Tiling Key中也包含了isValuesScalar标志，用于在编译期选择合适的模板实例。

**Expert implementation:**
```cpp
if constexpr (isScalar) {
    y[outputIdx] = values[0];
} else {
    y[outputIdx] = values[idx];
}
```

**vs. baseline:**
```cpp
// Always uses values[idx] - no scalar mode support
```

Benefit: 编译期优化避免运行期分支，支持更灵活的使用场景
Trade-off: 需要生成更多的模板实例，增加二进制大小
