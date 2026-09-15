/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_def.cpp  (KB-asset GE op_host IR-registration TEMPLATE)
 * \brief GE op IR registration for the FA-class arch35 op, authored as a KB template, NOT a raw copy
 *        of CANN ops-transformer/.../op_host/flash_attention_score_def.cpp.
 *
 *   ============================================================================================
 *   GENERIC SKELETON vs OP-SPECIFIC (whitebox refactor, flash_attention_score-gehost-3)
 *   ============================================================================================
 *   GENERIC GE-framework skeleton (arch-agnostic, identical shape for EVERY GE op):
 *       #include "register/op_def_registry.h"
 *       class <Op> : public OpDef {
 *           explicit <Op>(const char* name) : OpDef(name) {
 *               this->Input(...).ParamType(...).DataType(...).Format(...).UnknownShapeFormat(...);
 *               this->Output(...)...;
 *               this->Attr(...)...;
 *               OpAICoreConfig cfg; cfg.<flags>...;  this->AICore().AddConfig("<soc>", cfg);
 *           }
 *       };
 *       OP_ADD(<Op>);
 *   OP-SPECIFIC = the OP SIGNATURE: the op name, the ordered (input, output, attr) name list, and the
 *   per-input dtype/format matrix. Marked below with  `// <<< OP-SIGNATURE: parameterize per op >>>`
 *   ... `// <<< END OP-SIGNATURE >>>`  fences. A new GE op re-uses the skeleton and swaps ONLY what is
 *   inside those fences.
 *
 *   A3->A5 TRANSFORM (see GE_HOST_TRANSFORM_RECIPE.md): def.cpp is CARRY + PATCH — the op IR is shared
 *   between A3 (arch22) and A5 (arch35); A5 only ADDS dtype rows (fp8-e4m3 / fp8-e5m2 inputs + fp8
 *   block-wise dequant scale tensors + bf16 fp8 outputs). Carry the customer's A3 def verbatim, then
 *   patch the dtype matrix with the A5-only rows.
 *   ============================================================================================
 *
 *   The CANN production def enumerates a 22-column dtype matrix (every fp16/bf16/fp32/fp8/hifloat8 x
 *   layout permutation the shipped library supports). This TEMPLATE registers the dtype set the
 *   FA-class deliverable actually wires (fp16 / bf16 / fp32 / fp8-e4m3 / fp8-e5m2) in a compact,
 *   per-input helper form — it is the SAME op contract, re-expressed (so it is NOT byte-identical to
 *   the CANN source, per the port_a3 RED LINE) and easy to extend per-op.
 *
 *   RED LINE (port_a3_to_a5): host C++ only — NO `#include "arch35/"`, NO aclnn/aclop.
 *   register/op_def_registry.h is the GE op-build PUBLIC framework surface.
 */
#include <cstddef>
#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

// ============================================================================================
// <<< OP-SIGNATURE: parameterize per op >>>  (the DTYPE MATRIX — op-specific)
//   A3->A5 PATCH note: the A3 base matrix is the first 3 columns {fp16, bf16, fp32}. The A5-only
//   ADD is {fp8-e4m3, fp8-e5m2} (last 2 columns) + the d_scale_*/p_scale scale inputs + the fp8
//   bf16 outputs. To port an A3 def: carry the A3 dtype lists, then append the A5-fp8 columns.
// ============================================================================================
// per-input DataType rows — index alignment: 0=fp16 1=bf16 2=fp32 | 3=fp8e4m3 4=fp8e5m2 (A5-only)
static const std::vector<ge::DataType> FA_IN_DTYPES = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, /*A5-only:*/ ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
};
// atten_mask dtype row (bool/uint8 per fp-input slot).
static const std::vector<ge::DataType> FA_MASK_DTYPES = {
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_UINT8, /*A5-only:*/ ge::DT_UINT8, ge::DT_UINT8,
};
// scale tensors (A5-only fp8 block-wise dequant) — fp32 in every slot.
static const std::vector<ge::DataType> FA_SCALE_DTYPES = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, /*A5-only:*/ ge::DT_FLOAT, ge::DT_FLOAT,
};
// softmax stats — fp32 in every slot.
static const std::vector<ge::DataType> FA_STAT_DTYPES = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, /*A5-only:*/ ge::DT_FLOAT, ge::DT_FLOAT,
};
// attention_out / softmax_out — mirror the input dtype except fp8 -> bf16.
static const std::vector<ge::DataType> FA_OUT_DTYPES = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, /*A5-only:*/ ge::DT_BF16, ge::DT_BF16,
};
// <<< END OP-SIGNATURE (dtype matrix) >>>

// GENERIC skeleton helper (op-agnostic): every input column maps to FORMAT_ND for this op family.
static std::vector<ge::Format> FaNdFormats(size_t n)
{
    return std::vector<ge::Format>(n, ge::FORMAT_ND);
}

struct FaTensorSpec {
    const char *name;
    Option paramType;
    const std::vector<ge::DataType> *dataTypes;
};

static const std::vector<ge::DataType> FA_DROP_MASK_DTYPES(FA_IN_DTYPES.size(), ge::DT_UINT8);

// Tensor metadata is declared by name. Pointer order tables below preserve the GE positional contract.
static const FaTensorSpec ATTEN_MASK_INPUT{"atten_mask", OPTIONAL, &FA_MASK_DTYPES};
static const FaTensorSpec D_SCALE_K_INPUT{"d_scale_k", OPTIONAL, &FA_SCALE_DTYPES};
static const FaTensorSpec D_SCALE_Q_INPUT{"d_scale_q", OPTIONAL, &FA_SCALE_DTYPES};
static const FaTensorSpec D_SCALE_V_INPUT{"d_scale_v", OPTIONAL, &FA_SCALE_DTYPES};
static const FaTensorSpec DROP_MASK_INPUT{"drop_mask", OPTIONAL, &FA_DROP_MASK_DTYPES};
static const FaTensorSpec KEY_INPUT{"key", REQUIRED, &FA_IN_DTYPES};
static const FaTensorSpec PADDING_MASK_INPUT{"padding_mask", OPTIONAL, &FA_IN_DTYPES};
static const FaTensorSpec P_SCALE_INPUT{"p_scale", OPTIONAL, &FA_SCALE_DTYPES};
static const FaTensorSpec QUERY_INPUT{"query", REQUIRED, &FA_IN_DTYPES};
static const FaTensorSpec REAL_SHIFT_INPUT{"real_shift", OPTIONAL, &FA_IN_DTYPES};
static const FaTensorSpec VALUE_INPUT{"value", REQUIRED, &FA_IN_DTYPES};

static const FaTensorSpec ATTENTION_OUT_OUTPUT{"attention_out", REQUIRED, &FA_OUT_DTYPES};
static const FaTensorSpec SOFTMAX_MAX_OUTPUT{"softmax_max", REQUIRED, &FA_STAT_DTYPES};
static const FaTensorSpec SOFTMAX_OUT_OUTPUT{"softmax_out", REQUIRED, &FA_OUT_DTYPES};
static const FaTensorSpec SOFTMAX_SUM_OUTPUT{"softmax_sum", REQUIRED, &FA_STAT_DTYPES};

static const FaTensorSpec *const FA_INPUT_ORDER[] = {
    &QUERY_INPUT, &KEY_INPUT, &VALUE_INPUT, &REAL_SHIFT_INPUT, &DROP_MASK_INPUT, &PADDING_MASK_INPUT,
    &ATTEN_MASK_INPUT, &D_SCALE_Q_INPUT, &D_SCALE_K_INPUT, &D_SCALE_V_INPUT, &P_SCALE_INPUT,
};

static const FaTensorSpec *const FA_OUTPUT_ORDER[] = {
    &SOFTMAX_MAX_OUTPUT, &SOFTMAX_SUM_OUTPUT, &SOFTMAX_OUT_OUTPUT, &ATTENTION_OUT_OUTPUT,
};

static_assert(sizeof(FA_INPUT_ORDER) / sizeof(FA_INPUT_ORDER[0]) == 11, "FA input contract changed");
static_assert(sizeof(FA_OUTPUT_ORDER) / sizeof(FA_OUTPUT_ORDER[0]) == 4, "FA output contract changed");

static void ConfigureFaTensor(OpParamDef &tensor, const FaTensorSpec &spec,
                              const std::vector<ge::Format> &formats, bool autoContiguous)
{
    auto &configured = tensor.ParamType(spec.paramType)
                           .DataType(*spec.dataTypes)
                           .Format(formats)
                           .UnknownShapeFormat(formats);
    if (autoContiguous) {
        configured.AutoContiguous();
    }
}

static void ConfigureFaInputs(OpDef &op, const std::vector<ge::Format> &formats)
{
    for (const FaTensorSpec *spec : FA_INPUT_ORDER) {
        ConfigureFaTensor(op.Input(spec->name), *spec, formats, true);
    }
}

static void ConfigureFaOutputs(OpDef &op, const std::vector<ge::Format> &formats)
{
    for (const FaTensorSpec *spec : FA_OUTPUT_ORDER) {
        ConfigureFaTensor(op.Output(spec->name), *spec, formats, false);
    }
}

using FaAttrSetter = void (*)(OpAttrDef &);

struct FaAttrSpec {
    const char *name;
    Option paramType;
    FaAttrSetter setter;
};

static void SetFloatOne(OpAttrDef &attr)
{
    attr.Float(1.0F);
}

static void SetIntMax(OpAttrDef &attr)
{
    attr.Int(INT64_C(2147483647));
}

static void SetIntOne(OpAttrDef &attr)
{
    attr.Int(INT64_C(1));
}

static void SetIntZero(OpAttrDef &attr)
{
    attr.Int(INT64_C(0));
}

static void SetRequiredInt(OpAttrDef &attr)
{
    attr.Int();
}

static void SetRequiredString(OpAttrDef &attr)
{
    attr.String();
}

static const FaAttrSpec HEAD_NUM_ATTR{"head_num", REQUIRED, SetRequiredInt};
static const FaAttrSpec INNER_PRECISE_ATTR{"inner_precise", OPTIONAL, SetIntZero};
static const FaAttrSpec INPUT_LAYOUT_ATTR{"input_layout", REQUIRED, SetRequiredString};
static const FaAttrSpec KEEP_PROB_ATTR{"keep_prob", OPTIONAL, SetFloatOne};
static const FaAttrSpec NEXT_TOCKENS_ATTR{"next_tockens", OPTIONAL, SetIntMax};
static const FaAttrSpec OFFSET_ATTR{"offset", OPTIONAL, SetIntZero};
static const FaAttrSpec OUT_DTYPE_ATTR{"out_dtype", OPTIONAL, SetIntZero};
static const FaAttrSpec PRE_TOCKENS_ATTR{"pre_tockens", OPTIONAL, SetIntMax};
static const FaAttrSpec PSE_TYPE_ATTR{"pse_type", OPTIONAL, SetIntOne};
static const FaAttrSpec SCALE_VALUE_ATTR{"scale_value", OPTIONAL, SetFloatOne};
static const FaAttrSpec SEED_ATTR{"seed", OPTIONAL, SetIntZero};
static const FaAttrSpec SPARSE_MODE_ATTR{"sparse_mode", OPTIONAL, SetIntZero};

static const FaAttrSpec *const FA_ATTR_ORDER[] = {
    &SCALE_VALUE_ATTR, &KEEP_PROB_ATTR, &PRE_TOCKENS_ATTR, &NEXT_TOCKENS_ATTR,
    &HEAD_NUM_ATTR, &INPUT_LAYOUT_ATTR, &INNER_PRECISE_ATTR, &SPARSE_MODE_ATTR,
    &PSE_TYPE_ATTR, &SEED_ATTR, &OFFSET_ATTR, &OUT_DTYPE_ATTR,
};

static_assert(sizeof(FA_ATTR_ORDER) / sizeof(FA_ATTR_ORDER[0]) == 12, "FA attr contract changed");

static void ConfigureFaAttrs(OpDef &op)
{
    for (const FaAttrSpec *spec : FA_ATTR_ORDER) {
        auto &attr = op.Attr(spec->name).AttrType(spec->paramType);
        spec->setter(attr);
    }
}

// GENERIC SKELETON: class <Op> : public OpDef { ctor registers Input/Output/Attr + AICore config }.
// Only the op NAME (here FlashAttentionScore) and the body inside the OP-SIGNATURE fences are op-specific.
class FlashAttentionScore : public OpDef {   // <<< OP-SIGNATURE: op class name >>>
public:
    explicit FlashAttentionScore(const char *name) : OpDef(name)
    {
        // <<< OP-SIGNATURE: parameterize per op >>>  (input / output / attr name+type list — op-specific)
        const auto nd = FaNdFormats(FA_IN_DTYPES.size());
        ConfigureFaInputs(*this, nd);
        ConfigureFaOutputs(*this, nd);
        ConfigureFaAttrs(*this);
        // <<< END OP-SIGNATURE (input/output/attr list) >>>

        // ---- GENERIC SKELETON: AICore config. The flag set is op-agnostic GE boilerplate; only the
        //      SOC string + opFile.value are parameterized (A3->A5 PATCH: A3 registers ascend910b /
        //      ascend910_93; A5 registers ascend910_95 / Ascend950PR). ----
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true);
        aicoreConfig.DynamicFormatFlag(true);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.NeedCheckSupportFlag(false);
        aicoreConfig.PrecisionReduceFlag(true);
        aicoreConfig.ExtendCfgInfo(
            "opFile.value", "flash_attention_score");  // <<< OP-SIGNATURE: opFile name >>>
        this->AICore().AddConfig("ascend910_95", aicoreConfig);  // <<< OP-SIGNATURE: A5 SOC (A3 PATCH: ascend910b/ascend910_93) >>>
    }
};

OP_ADD(FlashAttentionScore);   // <<< OP-SIGNATURE: op class name >>>

}  // namespace ops
