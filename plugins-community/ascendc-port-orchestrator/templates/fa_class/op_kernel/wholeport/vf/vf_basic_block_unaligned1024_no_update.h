/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_basic_block_unaligned1024_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "../wp_pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1NoUpdateGeneralImpl1024VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ T2 * expUb5, __ubuf__ T2 * expUb6,
    __ubuf__ T2 * expUb7, __ubuf__ T2 * expUb8, __ubuf__ OUTPUT_T * pseUb, __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart,
    __ubuf__ T * srcUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4,
    __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6, __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, __ubuf__ uint32_t * maskUb9,
    __ubuf__ uint32_t * maskUb10, __ubuf__ uint32_t * maskUb11, __ubuf__ uint32_t * maskUb12, __ubuf__ uint32_t * maskUb13, __ubuf__ uint32_t * maskUb14,
    __ubuf__ uint32_t * maskUb15, __ubuf__ uint32_t * maskUb16, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3, const uint32_t oriTailN4, const uint32_t oriTailN5,
    const uint32_t oriTailN6, const uint32_t oriTailN7, const uint32_t oriTailN8, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3,
    const uint32_t tailN4, const uint32_t tailN5, const uint32_t tailN6, const uint32_t tailN7, const uint32_t tailN8, uint32_t pltOriTailN1,
    uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltOriTailN5, uint32_t pltOriTailN6, uint32_t pltOriTailN7, uint32_t pltOriTailN8,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, uint32_t pltTailN5, uint32_t pltTailN6, uint32_t pltTailN7, uint32_t pltTailN8,
    float divValue, const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel1;
    RegTensor<float> vreg_sel2;
    RegTensor<float> vreg_sel3;
    RegTensor<float> vreg_sel4;
    RegTensor<float> vreg_sel5;
    RegTensor<float> vreg_sel6;
    RegTensor<float> vreg_sel7;
    RegTensor<float> vreg_sel8;
    RegTensor<float> vreg_sel9;
    RegTensor<float> vreg_sel10;
    RegTensor<float> vreg_sel11;
    RegTensor<float> vreg_sel12;
    RegTensor<float> vreg_sel13;
    RegTensor<float> vreg_sel14;
    RegTensor<float> vreg_sel15;
    RegTensor<float> vreg_sel16;

    RegTensor<float> vreg_sel9_new;
    RegTensor<float> vreg_sel10_new;
    RegTensor<float> vreg_sel11_new;
    RegTensor<float> vreg_sel12_new;
    RegTensor<float> vreg_sel13_new;
    RegTensor<float> vreg_sel14_new;
    RegTensor<float> vreg_sel15_new;
    RegTensor<float> vreg_sel16_new;


    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_input_x5;
    RegTensor<float> vreg_input_x6;
    RegTensor<float> vreg_input_x7;
    RegTensor<float> vreg_input_x8;
    RegTensor<float> vreg_input_x9;
    RegTensor<float> vreg_input_x10;
    RegTensor<float> vreg_input_x11;
    RegTensor<float> vreg_input_x12;
    RegTensor<float> vreg_input_x13;
    RegTensor<float> vreg_input_x14;
    RegTensor<float> vreg_input_x15;
    RegTensor<float> vreg_input_x16;
    RegTensor<float> vreg_input_x9_new;
    RegTensor<float> vreg_input_x10_new;
    RegTensor<float> vreg_input_x11_new;
    RegTensor<float> vreg_input_x12_new;
    RegTensor<float> vreg_input_x13_new;
    RegTensor<float> vreg_input_x14_new;
    RegTensor<float> vreg_input_x15_new;
    RegTensor<float> vreg_input_x16_new;


    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_max_tmp4;
    RegTensor<float> vreg_max_tmp5;
    RegTensor<float> vreg_max_tmp6;
    RegTensor<float> vreg_max_tmp7;
    RegTensor<float> vreg_max_tmp8;

    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_zero;

    RegTensor<float> vreg_exp_sum1;
    RegTensor<float> vreg_exp_sum2;
    RegTensor<float> vreg_exp_sum3;
    RegTensor<float> vreg_exp_sum4;
    RegTensor<float> vreg_exp_sum5;
    RegTensor<float> vreg_exp_sum6;
    RegTensor<float> vreg_exp_sum7;
    RegTensor<float> vreg_exp_sum8;

    RegTensor<float> vreg_exp_even1;
    RegTensor<float> vreg_exp_odd1;
    RegTensor<float> vreg_exp_even2;
    RegTensor<float> vreg_exp_odd2;
    RegTensor<float> vreg_exp_even3;
    RegTensor<float> vreg_exp_odd3;
    RegTensor<float> vreg_exp_even4;
    RegTensor<float> vreg_exp_odd4;
    RegTensor<float> vreg_exp_even5;
    RegTensor<float> vreg_exp_odd5;
    RegTensor<float> vreg_exp_even6;
    RegTensor<float> vreg_exp_odd6;
    RegTensor<float> vreg_exp_even7;
    RegTensor<float> vreg_exp_odd7;
    RegTensor<float> vreg_exp_even8;
    RegTensor<float> vreg_exp_odd8;

    RegTensor<float> vreg_pse1;
    RegTensor<float> vreg_pse2;
    RegTensor<float> vreg_pse3;
    RegTensor<float> vreg_pse4;
    RegTensor<float> vreg_pse5;
    RegTensor<float> vreg_pse6;
    RegTensor<float> vreg_pse7;
    RegTensor<float> vreg_pse8;
    RegTensor<float> vreg_pse9;
    RegTensor<float> vreg_pse10;
    RegTensor<float> vreg_pse11;
    RegTensor<float> vreg_pse12;
    RegTensor<float> vreg_pse13;
    RegTensor<float> vreg_pse14;
    RegTensor<float> vreg_pse15;
    RegTensor<float> vreg_pse16;

    RegTensor<float> vreg_alibi1;
    RegTensor<float> vreg_alibi2;
    RegTensor<float> vreg_alibi3;
    RegTensor<float> vreg_alibi4;
    RegTensor<float> vreg_alibi5;
    RegTensor<float> vreg_alibi6;
    RegTensor<float> vreg_alibi7;
    RegTensor<float> vreg_alibi8;
    RegTensor<float> vreg_alibi9;
    RegTensor<float> vreg_alibi10;
    RegTensor<float> vreg_alibi11;
    RegTensor<float> vreg_alibi12;
    RegTensor<float> vreg_alibi13;
    RegTensor<float> vreg_alibi14;
    RegTensor<float> vreg_alibi15;
    RegTensor<float> vreg_alibi16;

    RegTensor<bfloat16_t> vreg_exp_even1_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp_even3_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd3_bf16;
    RegTensor<bfloat16_t> vreg_exp_even4_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd4_bf16;
    RegTensor<bfloat16_t> vreg_exp_even5_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd5_bf16;
    RegTensor<bfloat16_t> vreg_exp_even6_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd6_bf16;
    RegTensor<bfloat16_t> vreg_exp_even7_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd7_bf16;
    RegTensor<bfloat16_t> vreg_exp_even8_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd8_bf16;

    RegTensor<bfloat16_t> vreg_exp1_bf16;
    RegTensor<bfloat16_t> vreg_exp2_bf16;
    RegTensor<bfloat16_t> vreg_exp3_bf16;
    RegTensor<bfloat16_t> vreg_exp4_bf16;
    RegTensor<bfloat16_t> vreg_exp5_bf16;
    RegTensor<bfloat16_t> vreg_exp6_bf16;
    RegTensor<bfloat16_t> vreg_exp7_bf16;
    RegTensor<bfloat16_t> vreg_exp8_bf16;

    RegTensor<bfloat16_t> vreg_pse_bf16_src1;
    RegTensor<bfloat16_t> vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse_bf16_src3;
    RegTensor<bfloat16_t> vreg_pse_bf16_src4;
    RegTensor<bfloat16_t> vreg_pse_bf16_src5;
    RegTensor<bfloat16_t> vreg_pse_bf16_src6;
    RegTensor<bfloat16_t> vreg_pse_bf16_src7;
    RegTensor<bfloat16_t> vreg_pse_bf16_src8;

    RegTensor<bfloat16_t> vreg_pse1_bf16;
    RegTensor<bfloat16_t> vreg_pse2_bf16;
    RegTensor<bfloat16_t> vreg_pse3_bf16;
    RegTensor<bfloat16_t> vreg_pse4_bf16;
    RegTensor<bfloat16_t> vreg_pse5_bf16;
    RegTensor<bfloat16_t> vreg_pse6_bf16;
    RegTensor<bfloat16_t> vreg_pse7_bf16;
    RegTensor<bfloat16_t> vreg_pse8_bf16;
    RegTensor<bfloat16_t> vreg_pse9_bf16;
    RegTensor<bfloat16_t> vreg_pse10_bf16;
    RegTensor<bfloat16_t> vreg_pse11_bf16;
    RegTensor<bfloat16_t> vreg_pse12_bf16;
    RegTensor<bfloat16_t> vreg_pse13_bf16;
    RegTensor<bfloat16_t> vreg_pse14_bf16;
    RegTensor<bfloat16_t> vreg_pse15_bf16;
    RegTensor<bfloat16_t> vreg_pse16_bf16;

    RegTensor<half> vreg_exp_even1_f16;
    RegTensor<half> vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16;
    RegTensor<half> vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp_even3_f16;
    RegTensor<half> vreg_exp_odd3_f16;
    RegTensor<half> vreg_exp_even4_f16;
    RegTensor<half> vreg_exp_odd4_f16;
    RegTensor<half> vreg_exp_even5_f16;
    RegTensor<half> vreg_exp_odd5_f16;
    RegTensor<half> vreg_exp_even6_f16;
    RegTensor<half> vreg_exp_odd6_f16;
    RegTensor<half> vreg_exp_even7_f16;
    RegTensor<half> vreg_exp_odd7_f16;
    RegTensor<half> vreg_exp_even8_f16;
    RegTensor<half> vreg_exp_odd8_f16;

    RegTensor<half> vreg_exp1_f16;
    RegTensor<half> vreg_exp2_f16;
    RegTensor<half> vreg_exp3_f16;
    RegTensor<half> vreg_exp4_f16;
    RegTensor<half> vreg_exp5_f16;
    RegTensor<half> vreg_exp6_f16;
    RegTensor<half> vreg_exp7_f16;
    RegTensor<half> vreg_exp8_f16;

    RegTensor<half> vreg_pse_f16_src1;
    RegTensor<half> vreg_pse_f16_src2;
    RegTensor<half> vreg_pse_f16_src3;
    RegTensor<half> vreg_pse_f16_src4;
    RegTensor<half> vreg_pse_f16_src5;
    RegTensor<half> vreg_pse_f16_src6;
    RegTensor<half> vreg_pse_f16_src7;
    RegTensor<half> vreg_pse_f16_src8;

    RegTensor<half> vreg_pse1_f16;
    RegTensor<half> vreg_pse2_f16;
    RegTensor<half> vreg_pse3_f16;
    RegTensor<half> vreg_pse4_f16;
    RegTensor<half> vreg_pse5_f16;
    RegTensor<half> vreg_pse6_f16;
    RegTensor<half> vreg_pse7_f16;
    RegTensor<half> vreg_pse8_f16;
    RegTensor<half> vreg_pse9_f16;
    RegTensor<half> vreg_pse10_f16;
    RegTensor<half> vreg_pse11_f16;
    RegTensor<half> vreg_pse12_f16;
    RegTensor<half> vreg_pse13_f16;
    RegTensor<half> vreg_pse14_f16;
    RegTensor<half> vreg_pse15_f16;
    RegTensor<half> vreg_pse16_f16;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
    MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);

    MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
    MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);

    MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
    MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);

    MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
    MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);

    MaskReg preg_tail_n5 = UpdateMask<float>(pltTailN5);
    MaskReg preg_ori_tail_n5 = UpdateMask<float>(pltOriTailN5);

    MaskReg preg_tail_n6 = UpdateMask<float>(pltTailN6);
    MaskReg preg_ori_tail_n6 = UpdateMask<float>(pltOriTailN6);

    MaskReg preg_tail_n7 = UpdateMask<T>(pltTailN7);
    MaskReg preg_ori_tail_n7 = UpdateMask<T>(pltOriTailN7);

    MaskReg preg_tail_n8 = UpdateMask<T>(pltTailN8);
    MaskReg preg_ori_tail_n8 = UpdateMask<T>(pltOriTailN8);

    MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    MaskReg preg_compare4;
    MaskReg preg_compare5;
    MaskReg preg_compare6;
    MaskReg preg_compare7;
    MaskReg preg_compare8;
    MaskReg preg_compare9;
    MaskReg preg_compare10;
    MaskReg preg_compare11;
    MaskReg preg_compare12;
    MaskReg preg_compare13;
    MaskReg preg_compare14;
    MaskReg preg_compare15;
    MaskReg preg_compare16;

    Duplicate(vreg_min, minValue);
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi1, posShift);
        Adds(vreg_alibi2, vreg_alibi1, floatRepSize, preg_all);
        Adds(vreg_alibi3, vreg_alibi2, floatRepSize, preg_all);
        Adds(vreg_alibi4, vreg_alibi3, floatRepSize, preg_all);
        Adds(vreg_alibi5, vreg_alibi4, floatRepSize, preg_all);
        Adds(vreg_alibi6, vreg_alibi5, floatRepSize, preg_all);
        Adds(vreg_alibi7, vreg_alibi6, floatRepSize, preg_all);
        Adds(vreg_alibi8, vreg_alibi7, floatRepSize, preg_all);
        Adds(vreg_alibi9, vreg_alibi8, floatRepSize, preg_all);
        Adds(vreg_alibi10, vreg_alibi9, floatRepSize, preg_all);
        Adds(vreg_alibi11, vreg_alibi10, floatRepSize, preg_all);
        Adds(vreg_alibi12, vreg_alibi11, floatRepSize, preg_all);
        Adds(vreg_alibi13, vreg_alibi12, floatRepSize, preg_all);
        Adds(vreg_alibi14, vreg_alibi13, floatRepSize, preg_all);
        Adds(vreg_alibi15, vreg_alibi14, floatRepSize, preg_all);
        Adds(vreg_alibi16, vreg_alibi15, floatRepSize, preg_all);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
        LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
        LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
        LoadAlign(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
        LoadAlign(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
        LoadAlign(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
        LoadAlign(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);
        LoadAlign(vreg_input_x9, srcUb + floatRepSize * 8 + i * s2BaseSize);
        LoadAlign(vreg_input_x10, srcUb + floatRepSize * 9 + i * s2BaseSize);
        LoadAlign(vreg_input_x11, srcUb + floatRepSize * 10 + i * s2BaseSize);
        LoadAlign(vreg_input_x12, srcUb + floatRepSize * 11 + i * s2BaseSize);
        LoadAlign(vreg_input_x13, srcUb + floatRepSize * 12 + i * s2BaseSize);
        LoadAlign(vreg_input_x14, srcUb + floatRepSize * 13 + i * s2BaseSize);
        LoadAlign(vreg_input_x15, srcUb + floatRepSize * 14 + i * s2BaseSize);
        LoadAlign(vreg_input_x16, srcUb + floatRepSize * 15 + i * s2BaseSize);


        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
            Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
            Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
            Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
            Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
            Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
            Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

            Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
            Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
            Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
            Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
            Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
            Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
            Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
            Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Abs(vreg_pse1, vreg_alibi1, preg_all);
                Abs(vreg_pse2, vreg_alibi2, preg_all);
                Abs(vreg_pse3, vreg_alibi3, preg_all);
                Abs(vreg_pse4, vreg_alibi4, preg_all);
                Abs(vreg_pse5, vreg_alibi5, preg_all);
                Abs(vreg_pse6, vreg_alibi6, preg_all);
                Abs(vreg_pse7, vreg_alibi7, preg_all);
                Abs(vreg_pse8, vreg_alibi8, preg_all);
                Abs(vreg_pse9, vreg_alibi9, preg_all);
                Abs(vreg_pse10, vreg_alibi10, preg_all);
                Abs(vreg_pse11, vreg_alibi11, preg_all);
                Abs(vreg_pse12, vreg_alibi12, preg_all);
                Abs(vreg_pse13, vreg_alibi13, preg_all);
                Abs(vreg_pse14, vreg_alibi14, preg_all);
                Abs(vreg_pse15, vreg_alibi15, preg_all);
                Abs(vreg_pse16, vreg_alibi16, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse1, vreg_pse1, preg_all);
                    Sqrt(vreg_pse2, vreg_pse2, preg_all);
                    Sqrt(vreg_pse3, vreg_pse3, preg_all);
                    Sqrt(vreg_pse4, vreg_pse4, preg_all);
                    Sqrt(vreg_pse5, vreg_pse5, preg_all);
                    Sqrt(vreg_pse6, vreg_pse6, preg_all);
                    Sqrt(vreg_pse7, vreg_pse7, preg_all);
                    Sqrt(vreg_pse8, vreg_pse8, preg_all);
                    Sqrt(vreg_pse9, vreg_pse9, preg_all);
                    Sqrt(vreg_pse10, vreg_pse10, preg_all);
                    Sqrt(vreg_pse11, vreg_pse11, preg_all);
                    Sqrt(vreg_pse12, vreg_pse12, preg_all);
                    Sqrt(vreg_pse13, vreg_pse13, preg_all);
                    Sqrt(vreg_pse14, vreg_pse14, preg_all);
                    Sqrt(vreg_pse15, vreg_pse15, preg_all);
                    Sqrt(vreg_pse16, vreg_pse16, preg_all);
                }
                Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
                Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
                Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
                Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
                Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
                Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
                Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
                Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
                Muls(vreg_pse9, vreg_pse9, slopes, preg_all);
                Muls(vreg_pse10, vreg_pse10, slopes, preg_all);
                Muls(vreg_pse11, vreg_pse11, slopes, preg_all);
                Muls(vreg_pse12, vreg_pse12, slopes, preg_all);
                Muls(vreg_pse13, vreg_pse13, slopes, preg_all);
                Muls(vreg_pse14, vreg_pse14, slopes, preg_all);
                Muls(vreg_pse15, vreg_pse15, slopes, preg_all);
                Muls(vreg_pse16, vreg_pse16, slopes, preg_all);
                Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
                Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
                Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
                Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
                Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
                Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
                Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
                Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
                Adds(vreg_alibi9, vreg_alibi9, -1.0f, preg_all);
                Adds(vreg_alibi10, vreg_alibi10, -1.0f, preg_all);
                Adds(vreg_alibi11, vreg_alibi11, -1.0f, preg_all);
                Adds(vreg_alibi12, vreg_alibi12, -1.0f, preg_all);
                Adds(vreg_alibi13, vreg_alibi13, -1.0f, preg_all);
                Adds(vreg_alibi14, vreg_alibi14, -1.0f, preg_all);
                Adds(vreg_alibi15, vreg_alibi15, -1.0f, preg_all);
                Adds(vreg_alibi16, vreg_alibi16, -1.0f, preg_all);
            } else {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
                    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
                    Interleave(vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse_bf16_src5, vreg_pse_bf16_src5);
                    Interleave(vreg_pse11_bf16, vreg_pse12_bf16, vreg_pse_bf16_src6, vreg_pse_bf16_src6);
                    Interleave(vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse_bf16_src7, vreg_pse_bf16_src7);
                    Interleave(vreg_pse15_bf16, vreg_pse16_bf16, vreg_pse_bf16_src8, vreg_pse_bf16_src8);


                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_bf16, preg_all_b16);

                } else if constexpr (IsSameType<T2, half>::value) {
                    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    LoadAlign(vreg_pse_f16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
                    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
                    Interleave(vreg_pse9_f16, vreg_pse10_f16, vreg_pse_f16_src5, vreg_pse_f16_src5);
                    Interleave(vreg_pse11_f16, vreg_pse12_f16, vreg_pse_f16_src6, vreg_pse_f16_src6);
                    Interleave(vreg_pse13_f16, vreg_pse14_f16, vreg_pse_f16_src7, vreg_pse_f16_src7);
                    Interleave(vreg_pse15_f16, vreg_pse16_f16, vreg_pse_f16_src8, vreg_pse_f16_src8);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_f16, preg_all_b16);
                }
            }
            Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
            Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
            Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_all);
            Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_all);
            Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_all);
            Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_all);
            Add(vreg_input_x9, vreg_input_x9, vreg_pse9, preg_ori_tail_n1);
            Add(vreg_input_x10, vreg_input_x10, vreg_pse10, preg_ori_tail_n2);
            Add(vreg_input_x11, vreg_input_x11, vreg_pse11, preg_ori_tail_n3);
            Add(vreg_input_x12, vreg_input_x12, vreg_pse12, preg_ori_tail_n4);
            Add(vreg_input_x13, vreg_input_x13, vreg_pse13, preg_ori_tail_n5);
            Add(vreg_input_x14, vreg_input_x14, vreg_pse14, preg_ori_tail_n6);
            Add(vreg_input_x15, vreg_input_x15, vreg_pse15, preg_ori_tail_n7);
            Add(vreg_input_x16, vreg_input_x16, vreg_pse16, preg_ori_tail_n8);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
            Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
            Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
            Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
            Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
            Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
            Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

            Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
            Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
            Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
            Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
            Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
            Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
            Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
            Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
        }

        if constexpr (hasAtten == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare5, (__ubuf__ uint32_t *&)maskUb5, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare6, (__ubuf__ uint32_t *&)maskUb6, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare7, (__ubuf__ uint32_t *&)maskUb7, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare8, (__ubuf__ uint32_t *&)maskUb8, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare9, (__ubuf__ uint32_t *&)maskUb9, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare10, (__ubuf__ uint32_t *&)maskUb10, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare11, (__ubuf__ uint32_t *&)maskUb11, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare12, (__ubuf__ uint32_t *&)maskUb12, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare13, (__ubuf__ uint32_t *&)maskUb13, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare14, (__ubuf__ uint32_t *&)maskUb14, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare15, (__ubuf__ uint32_t *&)maskUb15, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare16, (__ubuf__ uint32_t *&)maskUb16, nPadding);

            Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
            Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
            Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
            Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
            Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
            Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
            Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
            Select(vreg_sel8, vreg_min, vreg_input_x8, preg_compare8);
            Select(vreg_sel9, vreg_min, vreg_input_x9, preg_compare9);
            Select(vreg_sel10, vreg_min, vreg_input_x10, preg_compare10);
            Select(vreg_sel11, vreg_min, vreg_input_x11, preg_compare11);
            Select(vreg_sel12, vreg_min, vreg_input_x12, preg_compare12);
            Select(vreg_sel13, vreg_min, vreg_input_x13, preg_compare13);
            Select(vreg_sel14, vreg_min, vreg_input_x14, preg_compare14);
            Select(vreg_sel15, vreg_min, vreg_input_x15, preg_compare15);
            Select(vreg_sel16, vreg_min, vreg_input_x16, preg_compare16);

            Select(vreg_sel9_new, vreg_sel9, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel10_new, vreg_sel10, vreg_min, preg_ori_tail_n2);
            Select(vreg_sel11_new, vreg_sel11, vreg_min, preg_ori_tail_n3);
            Select(vreg_sel12_new, vreg_sel12, vreg_min, preg_ori_tail_n4);
            Select(vreg_sel13_new, vreg_sel13, vreg_min, preg_ori_tail_n5);
            Select(vreg_sel14_new, vreg_sel14, vreg_min, preg_ori_tail_n6);
            Select(vreg_sel15_new, vreg_sel15, vreg_min, preg_ori_tail_n7);
            Select(vreg_sel16_new, vreg_sel16, vreg_min, preg_ori_tail_n8);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 4 +  i * s2BaseSize, vreg_sel5, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 5 +  i * s2BaseSize, vreg_sel6, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 6 +  i * s2BaseSize, vreg_sel7, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 7 +  i * s2BaseSize, vreg_sel8, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 8 +  i * s2BaseSize, vreg_sel9_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 9 +  i * s2BaseSize, vreg_sel10_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 10 +  i * s2BaseSize, vreg_sel11_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 11 +  i * s2BaseSize, vreg_sel12_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 12 +  i * s2BaseSize, vreg_sel13_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 13 +  i * s2BaseSize, vreg_sel14_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 14 +  i * s2BaseSize, vreg_sel15_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 15 +  i * s2BaseSize, vreg_sel16_new, preg_all);

            Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
            Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
            Max(vreg_max_tmp3, vreg_sel5, vreg_sel6, preg_all);
            Max(vreg_max_tmp4, vreg_sel7, vreg_sel8, preg_all);
            Max(vreg_max_tmp5, vreg_sel9_new, vreg_sel10_new, preg_all);
            Max(vreg_max_tmp6, vreg_sel11_new, vreg_sel12_new, preg_all);
            Max(vreg_max_tmp7, vreg_sel13_new, vreg_sel14_new, preg_all);
            Max(vreg_max_tmp8, vreg_sel15_new, vreg_sel16_new, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
            Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
            Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
            Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp1, preg_all);
        } else {
            Select(vreg_input_x9_new, vreg_input_x9, vreg_min, preg_ori_tail_n1);
            Select(vreg_input_x10_new, vreg_input_x10, vreg_min, preg_ori_tail_n2);
            Select(vreg_input_x11_new, vreg_input_x11, vreg_min, preg_ori_tail_n3);
            Select(vreg_input_x12_new, vreg_input_x12, vreg_min, preg_ori_tail_n4);
            Select(vreg_input_x13_new, vreg_input_x13, vreg_min, preg_ori_tail_n5);
            Select(vreg_input_x14_new, vreg_input_x14, vreg_min, preg_ori_tail_n6);
            Select(vreg_input_x15_new, vreg_input_x15, vreg_min, preg_ori_tail_n7);
            Select(vreg_input_x16_new, vreg_input_x16, vreg_min, preg_ori_tail_n8);


            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x5, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x6, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x7, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x8, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 8 + i * s2BaseSize, vreg_input_x9_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 9 + i * s2BaseSize, vreg_input_x10_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 10 + i * s2BaseSize, vreg_input_x11_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 11 + i * s2BaseSize, vreg_input_x12_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 12 + i * s2BaseSize, vreg_input_x13_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 13 + i * s2BaseSize, vreg_input_x14_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 14 + i * s2BaseSize, vreg_input_x15_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 15 + i * s2BaseSize, vreg_input_x16_new, preg_all);

            Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);
            Max(vreg_max_tmp3, vreg_input_x5, vreg_input_x6, preg_all);
            Max(vreg_max_tmp4, vreg_input_x7, vreg_input_x8, preg_all);
            Max(vreg_max_tmp5, vreg_input_x9_new, vreg_input_x10_new, preg_all);
            Max(vreg_max_tmp6, vreg_input_x11_new, vreg_input_x12_new, preg_all);
            Max(vreg_max_tmp7, vreg_input_x13_new, vreg_input_x14_new, preg_all);
            Max(vreg_max_tmp8, vreg_input_x15_new, vreg_input_x16_new, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
            Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
            Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
            Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

            Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp1, preg_all);
        }
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), ureg_max, 0);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, maxUbStart + i);

        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x9, vreg_input_x10, srcUb + floatRepSize * 8 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x11, vreg_input_x12, srcUb + floatRepSize * 10 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x13, vreg_input_x14, srcUb + floatRepSize * 12 + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x15, vreg_input_x16, srcUb + floatRepSize * 14 + i * s2BaseSize);


        ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even3, vreg_input_x5, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd3, vreg_input_x6, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even4, vreg_input_x7, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd4, vreg_input_x8, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even5, vreg_input_x9, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd5, vreg_input_x10, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even6, vreg_input_x11, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd6, vreg_input_x12, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even7, vreg_input_x13, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd7, vreg_input_x14, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even8, vreg_input_x15, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd8, vreg_input_x16, vreg_max_brc, preg_all);

        Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
        Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
        Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
        Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
        Add(vreg_exp_sum5, vreg_exp_even5, vreg_exp_odd5, preg_all);
        Add(vreg_exp_sum6, vreg_exp_even6, vreg_exp_odd6, preg_all);
        Add(vreg_exp_sum7, vreg_exp_even7, vreg_exp_odd7, preg_all);
        Add(vreg_exp_sum8, vreg_exp_even8, vreg_exp_odd8, preg_all);

        Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
        Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
        Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum6, preg_all);
        Add(vreg_exp_sum7, vreg_exp_sum7, vreg_exp_sum8, preg_all);

        Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum3, preg_all);
        Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum7, preg_all);

        Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum5, preg_all);

        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum1, vreg_exp_sum1, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), vreg_exp_sum1, ureg_exp_sum, 1);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even5_bf16, vreg_exp_even5, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd5_bf16, vreg_exp_odd5, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even6_bf16, vreg_exp_even6, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd6_bf16, vreg_exp_odd6, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even7_bf16, vreg_exp_even7, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd7_bf16, vreg_exp_odd7, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even8_bf16, vreg_exp_even8, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd8_bf16, vreg_exp_odd8, preg_all);


            Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp5_bf16, (RegTensor<uint16_t>&)vreg_exp_even5_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd5_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp6_bf16, (RegTensor<uint16_t>&)vreg_exp_even6_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd6_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp7_bf16, (RegTensor<uint16_t>&)vreg_exp_even7_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd7_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp8_bf16, (RegTensor<uint16_t>&)vreg_exp_even8_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd8_bf16, preg_all_b16);


            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb5), vreg_exp5_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb6), vreg_exp6_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb7), vreg_exp7_bf16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb8), vreg_exp8_bf16, blockStride, repeatStride, preg_all_b16);

        } else if constexpr (IsSameType<T2, half>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even5_f16, vreg_exp_even5, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd5_f16, vreg_exp_odd5, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even6_f16, vreg_exp_even6, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd6_f16, vreg_exp_odd6, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even7_f16, vreg_exp_even7, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd7_f16, vreg_exp_odd7, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even8_f16, vreg_exp_even8, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd8_f16, vreg_exp_odd8, preg_all);

            Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16, (RegTensor<uint16_t>&)vreg_exp_odd3_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16, (RegTensor<uint16_t>&)vreg_exp_odd4_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp5_f16, (RegTensor<uint16_t>&)vreg_exp_even5_f16, (RegTensor<uint16_t>&)vreg_exp_odd5_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp6_f16, (RegTensor<uint16_t>&)vreg_exp_even6_f16, (RegTensor<uint16_t>&)vreg_exp_odd6_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp7_f16, (RegTensor<uint16_t>&)vreg_exp_even7_f16, (RegTensor<uint16_t>&)vreg_exp_odd7_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp8_f16, (RegTensor<uint16_t>&)vreg_exp_even8_f16, (RegTensor<uint16_t>&)vreg_exp_odd8_f16, preg_all_b16);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb5), vreg_exp5_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb6), vreg_exp6_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb7), vreg_exp7_f16, blockStride, repeatStride, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb8), vreg_exp8_f16, blockStride, repeatStride, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

// no update, 512 < originN <= 1024
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl1024(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue, float keepProb)
{
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb5 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 4 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb6 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 5 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb7 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 6 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb8 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 7 * ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb1 = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb2 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t * maskUb3 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 2);
    __ubuf__ uint32_t * maskUb4 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 3);
    __ubuf__ uint32_t * maskUb5 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 4);
    __ubuf__ uint32_t * maskUb6 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 5);
    __ubuf__ uint32_t * maskUb7 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 6);
    __ubuf__ uint32_t * maskUb8 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 7);
    __ubuf__ uint32_t * maskUb9 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 8);
    __ubuf__ uint32_t * maskUb10 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 9);
    __ubuf__ uint32_t * maskUb11 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 10);
    __ubuf__ uint32_t * maskUb12 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 11);
    __ubuf__ uint32_t * maskUb13 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 12);
    __ubuf__ uint32_t * maskUb14 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 13);
    __ubuf__ uint32_t * maskUb15 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 14);
    __ubuf__ uint32_t * maskUb16 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 15);

    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 8 < floatRepSize ? originN - floatRepSize * 8 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 9) <= 0 ? 0 : originN - floatRepSize * 9;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 10) <= 0 ? 0 : originN - floatRepSize * 10;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 11) <= 0 ? 0 : originN - floatRepSize * 11;
    const uint32_t oriTailN5 = static_cast<int32_t>(originN - floatRepSize * 12) <= 0 ? 0 : originN - floatRepSize * 12;
    const uint32_t oriTailN6 = static_cast<int32_t>(originN - floatRepSize * 13) <= 0 ? 0 : originN - floatRepSize * 13;
    const uint32_t oriTailN7 = static_cast<int32_t>(originN - floatRepSize * 14) <= 0 ? 0 : originN - floatRepSize * 14;
    const uint32_t oriTailN8 = static_cast<int32_t>(originN - floatRepSize * 15) <= 0 ? 0 : originN - floatRepSize * 15;

    const uint32_t tailN1 = s2BaseSize - floatRepSize * 8;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 9;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 10;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 11;
    const uint32_t tailN5 = s2BaseSize - floatRepSize * 12;
    const uint32_t tailN6 = s2BaseSize - floatRepSize * 13;
    const uint32_t tailN7 = s2BaseSize - floatRepSize * 14;
    const uint32_t tailN8 = s2BaseSize - floatRepSize * 15;

    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;
    uint32_t pltOriTailN5 = oriTailN5;
    uint32_t pltOriTailN6 = oriTailN6;
    uint32_t pltOriTailN7 = oriTailN7;
    uint32_t pltOriTailN8 = oriTailN8;

    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    uint32_t pltTailN5 = tailN5;
    uint32_t pltTailN6 = tailN6;
    uint32_t pltTailN7 = tailN7;
    uint32_t pltTailN8 = tailN8;
    float divValue = 1.0f / keepProb;

    ProcessVec1NoUpdateGeneralImpl1024VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
        expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8, pseUb, expSumUb, maxUb, maxUbStart, srcUb,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8, maskUb9, maskUb10, maskUb11,
        maskUb12, maskUb13, maskUb14, maskUb15, maskUb16, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2,
        oriTailN3, oriTailN4, oriTailN5, oriTailN6, oriTailN7, oriTailN8, tailN1, tailN2, tailN3, tailN4, tailN5,
        tailN6, tailN7, tailN8, pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4, pltOriTailN5, pltOriTailN6,
        pltOriTailN7, pltOriTailN8, pltTailN1, pltTailN2, pltTailN3, pltTailN4, pltTailN5, pltTailN6, pltTailN7, pltTailN8,
        divValue, m, pseStride, slopes, posShift, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H
