/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3ddx_launch.h
 * \brief conv3ddx（Conv3DBackpropInput，输入梯度，无 bias）直调工程 Host/Device 接入契约
 *
 * 本文件是测试框架与算子实现（后续节点交付）之间的唯一 ABI 边界：
 *   - Conv3DdxSpec：逻辑规格（Host 侧），空间属性取 (D,H,W) 分量，pads 为逐维 (front, back)；
 *   - Conv3DdxTiling：Host 求解结果（路由头 + 不透明负载），整体拷贝到 GM 后随 launch 传入；
 *   - Conv3DdxSolveTiling：Host 求解入口（三 tiling 模板 + loadB2Condition 判定）；
 *   - LaunchConv3Ddx{Fp16,Bf16}{B1,B2,B3}：dtype × loadB2Condition 共 6 个 launch 编译入口。
 *
 * 算子源码接入位：op_kernel/ 下新增的强符号实现（bisheng -xasc 编译）在链接期自动覆盖
 * tests/conv3ddx_stub.cpp 提供的同名弱符号占位；弱符号未被覆盖时测试入口按
 * “框架自检模式”（stub）运行，只验证 bin 读写 / 设备内存 / 落盘通路，不验证数值。
 *
 * 口径来源：范式 adapters/ascendc/input/tilingkey-templates.md §2（五参数 tilingKey 编码 2^24+loadB2）。
 *
 * input_size 语义：input_size 数组按 data_format 轴序解读（与 ops-nn proto 及 case 表一致：
 * NCDHW → (N, Cin, Di, Hi, Wi)；NDHWC → (N, Di, Hi, Wi, Cin)）；Conv3DdxSpec 内部
 * 已归一为逻辑 NCDHW 语义（几何字段 n/ci/di/hi/wi），数组轴序转换由规格装载层完成。
 */

#ifndef CONV3DDX_LAUNCH_H
#define CONV3DDX_LAUNCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dtype 编码（filter/dedy/dedx 三路一致为合法规格） */
#define CONV3DDX_DTYPE_FP16 0u
#define CONV3DDX_DTYPE_BF16 1u
/* E2 用例使用的“不支持 dtype”（Host 校验须拒绝） */
#define CONV3DDX_DTYPE_FP32 2u

/* 特征/输出格式编码 */
#define CONV3DDX_FMT_NCDHW 0u
#define CONV3DDX_FMT_NDHWC 1u
/* E8 用例使用的“不支持格式”（Host 校验须拒绝） */
#define CONV3DDX_FMT_DHWCN 2u

/* tiling 模板编码（与 case 表 remark expected 对应） */
#define CONV3DDX_TPL_SMALL_SHAPE 0u
#define CONV3DDX_TPL_INNER_PRODUCT 1u
#define CONV3DDX_TPL_FULL_LOAD 2u

/* Host 求解返回码：0 成功；<0 参数/资源拒绝（非 0 退出）；CONV3DDX_STUB 表示实现未接入 */
#define CONV3DDX_OK 0
#define CONV3DDX_STUB (-2)

/* loadB2Condition 取值（tilingkey-templates §2：1 转置加逆序 / 2 仅逆序 / 3 仅转置） */
#define CONV3DDX_LOADB2_TRANSPOSE_AND_REVERSE 1u
#define CONV3DDX_LOADB2_REVERSE_ONLY 2u
#define CONV3DDX_LOADB2_TRANSPOSE_ONLY 3u

/* 五参数 tilingKey 预期编码（GET_TPL_TILING_KEY(loadB2,0,0,true,0) = 2^24 + loadB2） */
#define CONV3DDX_TILING_KEY(loadB2) (0x1000000u + (uint32_t)(loadB2))

/* 逻辑规格：dedx/dedy 几何按 NCDHW 语义记录；属性取空间分量（升维后统一为 3D 问题） */
struct Conv3DdxSpec {
    uint32_t n;            /* dedx batch（与 dedy batch 一致） */
    uint32_t cin;          /* dedx 通道（= filter CinG × groups） */
    uint32_t di, hi, wi;   /* dedx 空间几何 */
    uint32_t cout;         /* dedy 通道（= filter Cout） */
    uint32_t dout, ho, wo; /* dedy 空间几何 */
    uint32_t kd, kh, kw;   /* 核几何 */
    uint32_t sd, sh, sw;   /* 步长（空间分量） */
    uint32_t padDf, padDb; /* D 轴 pad (front, back) */
    uint32_t padHf, padHb; /* H 轴 pad */
    uint32_t padWf, padWb; /* W 轴 pad */
    uint32_t dd, dh, dw;   /* 空洞率（空间分量） */
    uint32_t groups;       /* 分组（20 例均为 1） */
    uint32_t fmt;          /* dedy/dedx 物理格式，CONV3DDX_FMT_* */
    uint32_t filterDtype;  /* weight dtype，CONV3DDX_DTYPE_* */
    uint32_t dedyDtype;    /* dedy dtype，CONV3DDX_DTYPE_* */
};

/* Host 求解结果：固定布局的路由头 + 不透明负载（负载由算子实现定义，整体随 launch 传 GM） */
struct Conv3DdxTiling {
    /* ---- 路由头（框架消费：任务路由与 tag 核对，不参与设备解码须由负载自含） ---- */
    uint32_t blockDim;       /* <<<blocks,nullptr,stream>>> 的 blocks */
    uint32_t tilingKey;      /* 五参数编码：2^24 + loadB2 */
    uint32_t templateId;     /* CONV3DDX_TPL_* */
    uint32_t loadB2;         /* CONV3DDX_LOADB2_* */
    uint32_t workspaceBytes; /* 设备 workspace 字节数（0 表示无） */
    uint32_t initOutputFlag; /* 1 = 需要输出预清零（DX 无贡献位语义） */
    uint32_t reserved[1];    /* 头部对齐保留，置 0 */
    /* ---- 不透明负载：算子实现自定义的共享字段区（≤ 1008 字节） ---- */
    uint8_t payload[1008];
};

/* Host 求解入口：成功返回 CONV3DDX_OK 并填写 tiling；规格非法返回 <0（且不产出可 launch 的结果）；
 * 算子未接入时弱符号占位返回 CONV3DDX_STUB。 */
int Conv3DdxSolveTiling(const struct Conv3DdxSpec* spec, struct Conv3DdxTiling* tiling);

/* launch 入口（每 dtype × loadB2Condition 一个编译实例；参数为 GM 地址，tiling 为设备侧拷贝） */
void LaunchConv3DdxFp16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);
void LaunchConv3DdxFp16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);
void LaunchConv3DdxFp16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);
void LaunchConv3DdxBf16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);
void LaunchConv3DdxBf16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);
void LaunchConv3DdxBf16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy, void* dedx,
                          void* workspace, void* tiling);

#ifdef __cplusplus
}
#endif

#endif /* CONV3DDX_LAUNCH_H */
