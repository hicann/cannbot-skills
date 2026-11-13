/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3dtranspose_tiling_key.h
 * \brief Conv3DTranspose TilingKey 模板参数声明（ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL）
 *
 * 唯一事实源：design/TilingKey.md §1「TilingKey C++ 结构定义」逐字转写——五参数
 * （顺序 / 位宽 / 取值列表与范式声明及内置 arch35 同族声明一致），ASCENDC_TPL_SEL 仅
 * 实例化本范式当前执行配置的三个组合（loadB2Condition ∈ {1,2,3}，其余四参数固定），
 * 每块末位 ASCENDC_TPL_KERNEL_TYPE_SEL(AIC_ONLY) 声明核类型构建元数据（不占 key 位），
 * 配对「主计算完整写回」独立初始化协议（Kernel.md 9.6）。dtype 与 format 不进入本声明
 * （构建系统按 OpDef 组合以外层 DTYPE_* / FORMAT_* 宏注入，含 bias dtype 的 DTYPE_BIAS，
 * TilingKey.md §3）；bias 存在性经注册 ABI 的 bias GM 指针承载，不占 key 位。
 *
 * key 位域编码（TilingKey.md §1）：UINT 参数取「值在 DECL 枚举列表中的下标」、BOOL 取原值，
 * 按声明顺序自 bit 0 拼接——loadB2Condition bit0-7、kernelSplitMode bit8-15、groupConvMode
 * bit16-23、isBasicBlockTiling bit24、loadB1Condition bit25-32；三组合 key =
 * GET_TPL_TILING_KEY(loadB2Condition, 0, 0, true, 0) = 2^24 + loadB2Condition ∈
 * {16777217, 16777218, 16777219}（后缀是位编码值而非连续实例序号）。
 */

#ifndef CONV3DTRANSPOSE_TILING_KEY_H
#define CONV3DTRANSPOSE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

/* loadB2Condition：L1B→L0B 权重变换（范式 tilingkey-templates §2） */
#define TPL_NO_TRANSPOSE_NO_REVERSE 0
#define TPL_TRANSPOSE_AND_REVERSE 1
#define TPL_REVERSE_ONLY 2
#define TPL_TRANSPOSE_ONLY 3

/* kernelSplitMode：核窗口拆分（本交付执行配置固定不拆分） */
#define TPL_NO_SPLIT_KERNEL 0
#define TPL_SPLIT_KERNEL_HW 1
#define TPL_SPLIT_KERNEL_H 2

/* groupConvMode：分组寻址（本交付执行配置固定原始分组） */
#define TPL_GROUP_MODE_ORIGIN 0
#define TPL_GROUP_MODE_ENLARGE 1

/* loadB1Condition：GM→L1 权重搬入（本交付执行配置固定 GM 直达 L1） */
#define TPL_GM_TO_L1 0
#define TPL_VEC_TO_L1_C04 1
#define TPL_GM_TO_L1_NO_HK 2
#define TPL_GM_TO_L1_NO_HK_WK 3
#define TPL_SMALL_KERNEL 4

/* biasDtypeMode：bias 通路 BIAST 实例维度（Kernel.md 9.9 校验项 9 fp16 扩展义务的
 * key 位收口）。运行期 binary 选择键（dtype/format simplifiedKey）不包含可选输入
 * bias 的 dtype 维度（构建实测），BIAST=half 无法经独立 .o 注册路由；本参数把
 * bias dtype 折入 tilingKey 位域，经 TilingKey.md「运行期按 key 值直接选
 * sub-kernel 符号」的既定机制在同一 .o 内实例化 float/half 两套 BIAST 实例
 * （Kernel.md 模板类 BIAST 形参 + L1→BT 拓宽拷贝重载，内置基线 conv_bp_sub_func.h
 * 同构）。前五参数位域编码不变：fp32-bias 与无 bias 组合 key 仍为
 * 16777217/16777218/16777219（对已全绿组合零扰动）；fp16-bias 组合 key 附加
 * bit33（biasDtypeMode 在 DECL 列表 (0,1) 中下标 1，位宽 8，自 bit 33 起）。 */
#define TPL_BIAS_DTYPE_FLOAT 0
#define TPL_BIAS_DTYPE_HALF 1

/*
 * 模板参数声明：五参数，顺序 / 位宽 / 取值列表与范式声明及内置 arch35 同族声明一致
 * （声明名绑定本算子 OpType Conv3DTranspose，spec op.geir_name 大小写为准）。
 * dtype 与 format 不进入本声明（构建系统按 OpDef 组合以外层 DTYPE_* / FORMAT_* 宏注入，
 * 含 bias dtype 的 DTYPE_BIAS）；bias 存在性经注册 ABI 的 bias GM 指针承载，不占 key 位。
 */
ASCENDC_TPL_ARGS_DECL(Conv3DTranspose,
                      ASCENDC_TPL_UINT_DECL(loadB2Condition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                                            TPL_NO_TRANSPOSE_NO_REVERSE, TPL_TRANSPOSE_AND_REVERSE, TPL_REVERSE_ONLY,
                                            TPL_TRANSPOSE_ONLY),
                      ASCENDC_TPL_UINT_DECL(kernelSplitMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL,
                                            TPL_SPLIT_KERNEL_HW, TPL_SPLIT_KERNEL_H),
                      ASCENDC_TPL_UINT_DECL(groupConvMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN,
                                            TPL_GROUP_MODE_ENLARGE),
                      ASCENDC_TPL_BOOL_DECL(isBasicBlockTiling, 0, 1),
                      ASCENDC_TPL_UINT_DECL(loadB1Condition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1,
                                            TPL_VEC_TO_L1_C04, TPL_GM_TO_L1_NO_HK, TPL_GM_TO_L1_NO_HK_WK,
                                            TPL_SMALL_KERNEL),
                      ASCENDC_TPL_UINT_DECL(biasDtypeMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_FLOAT,
                                            TPL_BIAS_DTYPE_HALF));

/*
 * 模板参数组合：本范式执行配置的六个组合——
 * kernelSplitMode=TPL_NO_SPLIT_KERNEL、groupConvMode=TPL_GROUP_MODE_ORIGIN、
 * isBasicBlockTiling=1、loadB1Condition=TPL_GM_TO_L1 固定，loadB2Condition ∈ {1,2,3}
 * × biasDtypeMode ∈ {TPL_BIAS_DTYPE_FLOAT, TPL_BIAS_DTYPE_HALF}（fp16-bias 扩展义务
 * 的同 .o 双 BIAST 实例，9.9 校验项 9；bias 存在性仍经注册 ABI 的 bias GM 指针承载，
 * biasDtypeMode=FLOAT 覆盖无 bias 与 fp32-bias 组合——无 bias 实例不创建 bias 队列
 * 资源，BIAST 形参无运行期副作用）。
 * 每块末位 ASCENDC_TPL_KERNEL_TYPE_SEL 声明核类型构建元数据（AIC_ONLY，配对
 * 「主计算完整写回」独立初始化协议）；该类参数不占 key 位（encode 段为空），
 * 仅驱动 sub-kernel 按 AIC 通路编译。
 */
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_AND_REVERSE),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_FLOAT),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_AND_REVERSE),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_HALF),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_FLOAT),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_HALF),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_FLOAT),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
                         ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
                         ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1),
                         ASCENDC_TPL_UINT_SEL(biasDtypeMode, ASCENDC_TPL_UI_LIST, TPL_BIAS_DTYPE_HALF),
                         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)));

#endif /* CONV3DTRANSPOSE_TILING_KEY_H */
