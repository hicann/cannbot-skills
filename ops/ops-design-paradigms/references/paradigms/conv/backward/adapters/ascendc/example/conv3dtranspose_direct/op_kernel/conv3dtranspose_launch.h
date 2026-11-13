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
 * \file conv3dtranspose_launch.h
 * \brief conv3dtranspose（Conv3DTranspose，转置卷积，带可选 bias/output_padding）直调工程
 *        Host/Device 接入契约
 *
 * 本文件是测试框架与算子实现（后续节点交付）之间的唯一 ABI 边界：
 *   - Conv3DtSpec：逻辑规格（Host 侧），空间属性取 (D,H,W) 分量，pads 为逐维 (front, back)，
 *     output_padding 为空间分量 (qD,qH,qW)；
 *   - Conv3DtTiling：Host 求解结果（路由头 + 不透明负载），整体拷贝到 GM 后随 launch 传入；
 *   - Conv3DtSolveTiling：Host 求解入口（三 tiling 模板 + loadB2Condition 判定 + bias/初始化判定）；
 *   - LaunchConv3Dt{Fp16,Bf16}{B1,B2,B3}：dtype × loadB2Condition 共 6 个 launch 编译入口。
 *
 * 算子源码接入位：op_kernel/ 下新增的强符号实现（bisheng -xasc 编译）在链接期自动覆盖
 * tests/conv3dtranspose_stub.cpp 提供的同名弱符号占位；弱符号未被覆盖时测试入口按
 * “框架自检模式”（stub）运行，只验证 bin 读写 / 设备内存 / 落盘通路，不验证数值。
 *
 * 口径来源：范式 adapters/ascendc/compile-recipe.md §1.2（launch 形态）与
 * adapters/ascendc/tilingkey-templates.md §2（五参数 tilingKey）；五参数模板与编码已对照
 * CANN 包内置 conv3d_transpose_v2（opp/built-in .../ops_nn/ascendc/conv3d_transpose_v2/
 * conv3d_transpose_v2_arch35_tiling_key.h：ASCENDC_TPL_ARGS_DECL(loadB2Condition, kernelSplitMode,
 * groupConvMode, isBasicBlockTiling, loadB1Condition)，template_argument.h 逐参数按位宽累进编码，
 * (loadB2,0,0,true,0) → 2^24 + loadB2）。
 *
 * input_size 语义：input_size 数组按 data_format 轴序解读（与 GE IR 及 case 表一致：
 * NCDHW → (N, Cin, Di, Hi, Wi)；NDHWC → (N, Di, Hi, Wi, Cin)）；Conv3DtSpec 内部
 * 已归一为逻辑 NCDHW 语义（几何字段 n/ci/di/hi/wi），数组轴序转换由规格装载层完成。
 *
 * bias 语义（范式 scene §2.7 / preprocessing §5.3）：一维 FP32 向量、长度 Cin、沿输出通道
 * 广播；无 bias 用例传 nullptr。bias 为运行时通路（hasBiasFlag 不入 tilingKey，与内置算子
 * 一致——内置五参数模板不编码 bias）。offset_w（第 5 输入槽）为量化占位，20 例全空，
 * 本契约不设形参。
 */

#ifndef CONV3DT_LAUNCH_H
#define CONV3DT_LAUNCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dtype 编码（filter/x/y 三路一致为合法规格；bias 恒 FP32 不编码） */
#define CONV3DT_DTYPE_FP16 0u
#define CONV3DT_DTYPE_BF16 1u
/* E2a 用例使用的“不支持 dtype”（Host 校验须拒绝） */
#define CONV3DT_DTYPE_FP32 2u

/* 特征/输出格式编码 */
#define CONV3DT_FMT_NCDHW 0u
#define CONV3DT_FMT_NDHWC 1u
/* E8 用例使用的“不支持格式”（Host 校验须拒绝） */
#define CONV3DT_FMT_DHWCN 2u

/* tiling 模板编码（与 case 表 remark expected 对应） */
#define CONV3DT_TPL_SMALL_SHAPE 0u
#define CONV3DT_TPL_INNER_PRODUCT 1u
#define CONV3DT_TPL_FULL_LOAD 2u

/* Host 求解返回码：0 成功；<0 参数/资源拒绝（非 0 退出）；CONV3DT_STUB 表示实现未接入 */
#define CONV3DT_OK 0
#define CONV3DT_STUB (-2)

/* loadB2Condition 取值（tilingkey-templates §2：1 转置加逆序 / 2 仅逆序 / 3 仅转置） */
#define CONV3DT_LOADB2_TRANSPOSE_AND_REVERSE 1u
#define CONV3DT_LOADB2_REVERSE_ONLY 2u
#define CONV3DT_LOADB2_TRANSPOSE_ONLY 3u

/* 五参数 tilingKey 预期编码（GET_TPL_TILING_KEY(loadB2,0,0,true,0) = 2^24 + loadB2，
 * 与内置 conv3d_transpose_v2 arch35 五参数模板一致） */
#define CONV3DT_TILING_KEY(loadB2) (0x1000000u + (uint32_t)(loadB2))

/* 逻辑规格：y/dedy(x) 几何按 NCDHW 语义记录；属性取空间分量（升维后统一为 3D 问题）。
 * 对象绑定（scene §1）：x→dedy(N,Cout,Dout,Ho,Wo)、filter→weight(Cout,CinG,Dk,Hk,Wk)、
 * y→dedx(N,Cin,Di,Hi,Wi)、bias(Cin,)。 */
struct Conv3DtSpec {
    uint32_t n;            /* y batch（与 x batch 一致） */
    uint32_t cin;          /* y 通道（= filter CinG × groups，bias 长度） */
    uint32_t di, hi, wi;   /* y 空间几何（显式 input_size 目标形状） */
    uint32_t cout;         /* x 通道（= filter Cout） */
    uint32_t dout, ho, wo; /* x 空间几何 */
    uint32_t kd, kh, kw;   /* 核几何 */
    uint32_t sd, sh, sw;   /* 步长（空间分量） */
    uint32_t padDf, padDb; /* D 轴 pad (front, back) */
    uint32_t padHf, padHb; /* H 轴 pad */
    uint32_t padWf, padWb; /* W 轴 pad */
    uint32_t dd, dh, dw;   /* 空洞率（空间分量） */
    uint32_t qd, qh, qw;   /* output_padding（空间分量；q>=0 且 q<stride or q<dilation） */
    uint32_t groups;       /* 分组（20 例均为 1） */
    uint32_t fmt;          /* x/y 物理格式，CONV3DT_FMT_* */
    uint32_t filterDtype;  /* weight dtype，CONV3DT_DTYPE_* */
    uint32_t dedyDtype;    /* x dtype，CONV3DT_DTYPE_* */
    uint32_t hasBias;      /* 0 = 无 bias；1 = 有 bias（biasElems 必须等于 cin） */
    uint32_t biasElems;    /* bias 元素数（E10 用例变异为 != cin 时求解器拒绝） */
};

/* Host 求解结果：固定布局的路由头 + 不透明负载（负载由算子实现定义，整体随 launch 传 GM） */
struct Conv3DtTiling {
    /* ---- 路由头（框架消费：任务路由与 tag 核对，不参与设备解码须由负载自含） ---- */
    uint32_t blockDim;       /* <<<blocks,nullptr,stream>>> 的 blocks */
    uint32_t tilingKey;      /* 五参数编码：2^24 + loadB2（(loadB2,0,0,true,0) 实例） */
    uint32_t templateId;     /* CONV3DT_TPL_* */
    uint32_t loadB2;         /* CONV3DT_LOADB2_* */
    uint32_t workspaceBytes; /* 设备 workspace 字节数（0 表示无） */
    uint32_t initOutputFlag; /* 1 = 需要输出预初始化（转置无贡献位写 bias 值语义） */
    uint32_t hasBiasFlag;    /* 1 = 本次执行带 bias（与 spec.hasBias 一致，tag 核对用） */
    uint32_t reserved[1];    /* 头部对齐保留，置 0 */
    /* ---- 不透明负载：算子实现自定义的共享字段区（≤ 992 字节） ---- */
    uint8_t payload[992];
};

/* Host 求解入口：成功返回 CONV3DT_OK 并填写 tiling；规格非法（几何/dtype/属性/分组/格式/
 * q 约束/bias 长度）返回 <0（且不产出可 launch 的结果）；算子未接入时弱符号占位返回
 * CONV3DT_STUB。 */
int Conv3DtSolveTiling(const struct Conv3DtSpec* spec, struct Conv3DtTiling* tiling);

/* launch 入口（每 dtype × loadB2Condition 一个编译实例；参数为 GM 地址，tiling 为设备侧拷贝）。
 * bias 为 FP32 一维长度 Cin 的 GM 地址，无 bias 传 nullptr；offset_w 槽全空不设形参。 */
void LaunchConv3DtFp16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);
void LaunchConv3DtFp16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);
void LaunchConv3DtFp16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);
void LaunchConv3DtBf16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);
void LaunchConv3DtBf16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);
void LaunchConv3DtBf16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* bias,
                         void* dedx, void* workspace, void* tiling);

#ifdef __cplusplus
}
#endif

#endif /* CONV3DT_LAUNCH_H */
