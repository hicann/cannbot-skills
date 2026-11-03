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
 * \file custom_gather_simt.h
 * \brief SIMT kernel implementation for custom_gather
 *
 * MDE 5 设计要点：
 *   - 对标 PyTorch gather CUDA（SE 3.3）：每输出元素一线程，纯数据搬运（DataCopy 语义）
 *   - 全局 Grid-Stride：VF 内建网格（blockIdx.x/gridDim.x/blockDim.x）直接给出线程全局
 *     编号与跨核步长，遍历 [start, outSize)
 *   - 坐标解码：flat i → R 维坐标（用 index 累积步长 outStrides 做固定除数快除），
 *     读 index[i] 得 j → 越界 __trap() 判定（负索引不归一化）→ 用 input 累积步长
 *     inStrides 累加得输入偏移 inOff（dim 维坐标用 j 替代）→ 字节级 PackT 搬移
 *   - 越界语义（SE 5.3）：j < 0 || j >= gatherDimSize 原值比较触发 __trap()（对标 CUDA device
 *     assert / torch IndexError），无兜底分支，合法路径直接搬移
 *   - 纯 GM 直访（无 UB 数据缓存）；TilingData 数组字段与快除 magic/shift 经 UB 中转传 VF
 *   - THREAD_NUM = 1024 编译期常量（MDE 2.2），不入 TilingData
 *
 * 关键设计决策（演进史见 git log）：
 *   - 双重载分派：IDX32（uint32，native 乘法单指令）走编译期 RANK 特化；
 *     IDX64（uint64，乘法高多指令模拟、占双寄存器）高 rank 全展开会寄存器
 *     溢出，保留运行期 rank 截断分支作为调度边界。
 *   - 通用合轴：tiling 侧 MergeAxis 把可合并连续维折叠成 prod 轴后下发，
 *     kernel 收到的即合并后视图，对合轴无感知。
 *   - use64Bit 仅由张量规模（outSize/inputNum > INT32_MAX）决定，int64
 *     index 小张量走 IDX32 特化，越界语义由 kernel 原值比较保障。
 */

#ifndef CUSTOM_GATHER_SIMT_H
#define CUSTOM_GATHER_SIMT_H

#include <climits>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "utils/debug/asc_assert.h"
#include "custom_gather_tiling_data.h"
#include "custom_gather_tiling_key.h"

namespace NsCustomGather {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;                // 编译期常量，__launch_bounds__ 与 dim3 共用（MDE 2.2）
constexpr int32_t MAX_RANK = CUSTOM_GATHER_MAX_RANK; // outStrides/inStrides 数组长度（MDE 3.1）

// ============================================================
// 元素字节宽打包类型（MDE 4.2：纯搬移按位字节操作，bit 级等价；
// gather 仅 6 种 dtype，sizeof ∈ {2,4,8}，对应 PackT ∈ {uint16,uint32,uint64}）
// ============================================================
template <typename T>
struct BytePackType {
    using type = std::conditional_t<sizeof(T) == 2, uint16_t, std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "unsupported element size for gather");
};

// ============================================================
// GatherSimtKernel：VF 核心（IDX32 / uint32 路径，编译期 RANK 特化）
// 坐标解码循环固定 RANK-1 次全展开；dim 保持运行期值（不特化）
// ============================================================
template <typename T, typename INDEX_T, typename IDX_T, int32_t RANK>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void GatherSimtKernel(
    int64_t outSize,              // 输出总元素数（循环上界）
    int32_t dim,                  // 归一化收集轴（运行期值）
    int64_t gatherDimSize,        // input.shape[dim]（越界判定上界）
    int64_t inStrideDim,          // input 的 dim 维 stride（j 贡献累加）
    __ubuf__ int64_t* outStrides, // 坐标解码除数 [0, RANK-2]
    __ubuf__ int64_t* inStrides,  // 偏移累加步长 [0, RANK-2]
    __ubuf__ IDX_T* outMagic,     // 快除 magic [0, RANK-2]
    __ubuf__ IDX_T* outShift,     // 快除 shift [0, RANK-2]
    __gm__ T* inputGm, __gm__ INDEX_T* indexGm, __gm__ T* outputGm)
{
    using PackT = typename BytePackType<T>::type;
    __gm__ PackT* inputBytes = reinterpret_cast<__gm__ PackT*>(inputGm);
    __gm__ PackT* outputBytes = reinterpret_cast<__gm__ PackT*>(outputGm);

    // 全局 Grid-Stride：i = blockIdx.x * blockDim.x + threadIdx.x，步长 gridDim.x * blockDim.x
    IDX_T stride = static_cast<IDX_T>(gridDim.x) * static_cast<IDX_T>(blockDim.x);
    IDX_T start = static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
    for (IDX_T i = start; i < static_cast<IDX_T>(outSize); i += stride) {
        // 1. 读 index 值 j（dim 维坐标）
        INDEX_T j = indexGm[i];

        // 2. 越界检查（负索引不归一化，越界 __trap）
        if (j < 0 || j >= static_cast<INDEX_T>(gatherDimSize)) {
            __trap();
        }
        IDX_T jU = static_cast<IDX_T>(j);

        // 3. 坐标解码（flat i → R 维坐标快除）+ 输入偏移累加
        IDX_T rem = i;
        IDX_T inOff = 0;
#pragma unroll
        for (int32_t d = 0; d < RANK - 1; ++d) {
            IDX_T c = AscendC::Simt::UintDiv<IDX_T>(rem, outMagic[d], outShift[d]);
            rem -= c * static_cast<IDX_T>(outStrides[d]);
            if (d != dim) { // dim 维坐标用 j 替代，此处跳过
                inOff += c * static_cast<IDX_T>(inStrides[d]);
            }
        }

        // 4. 最后一维坐标 c_{R-1} = rem（其 inStrides[R-1] 恒为 1）
        if (RANK - 1 != dim) {
            inOff += rem;
        }

        // 5. dim 维贡献（用 index 值 j 而非解码得到的 c_dim）
        inOff += jU * static_cast<IDX_T>(inStrideDim);

        // 6. 字节级搬移（纯 bit 复制，NaN/Inf/±0 原样透传）
        outputBytes[i] = inputBytes[inOff];
    }
}

// ============================================================
// GatherSimtKernel（IDX64 / uint64 路径，运行期 rank 重载）
// 与 IDX32 重载为 C++ 模板重载（按模板实参个数区分）。
// 解码循环展开 MAX_RANK-1 次 + 运行期 if(d<rank-1) 截断，该截断分支
// 构成调度边界：阻止 64 位乘法高（多指令模拟）跨迭代交错与常量整体
// 上提，使寄存器压力有界——高 rank 全展开会溢出（MDE 9.1）。
// ============================================================
template <typename T, typename INDEX_T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void GatherSimtKernel(
    int64_t outSize,              // 输出总元素数（循环上界）
    int32_t dim,                  // 归一化收集轴
    int32_t rank,                 // R（运行期值，坐标解码截断上界）
    int64_t gatherDimSize,        // input.shape[dim]（越界判定上界）
    int64_t inStrideDim,          // input 的 dim 维 stride（j 贡献累加）
    __ubuf__ int64_t* outStrides, // 坐标解码除数 [0, rank-2]
    __ubuf__ int64_t* inStrides,  // 偏移累加步长 [0, rank-2]
    __ubuf__ IDX_T* outMagic,     // 快除 magic [0, rank-2]
    __ubuf__ IDX_T* outShift,     // 快除 shift [0, rank-2]
    __gm__ T* inputGm, __gm__ INDEX_T* indexGm, __gm__ T* outputGm)
{
    using PackT = typename BytePackType<T>::type;
    __gm__ PackT* inputBytes = reinterpret_cast<__gm__ PackT*>(inputGm);
    __gm__ PackT* outputBytes = reinterpret_cast<__gm__ PackT*>(outputGm);

    IDX_T stride = static_cast<IDX_T>(gridDim.x) * static_cast<IDX_T>(blockDim.x);
    IDX_T start = static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
    for (IDX_T i = start; i < static_cast<IDX_T>(outSize); i += stride) {
        // 1. 读 index 值 j（dim 维坐标）
        INDEX_T j = indexGm[i];

        // 2. 越界检查（负索引不归一化，越界 __trap）
        if (j < 0 || j >= gatherDimSize) {
            __trap();
        }
        IDX_T jU = static_cast<IDX_T>(j);

        // 3. 坐标解码 + 输入偏移累加（运行期 rank 截断）
        IDX_T rem = i;
        IDX_T inOff = 0;
#pragma unroll
        for (int32_t d = 0; d < MAX_RANK - 1; ++d) { // 编译期展开，UB 下标本静态
            if (d < rank - 1) {                      // 运行期 rank 截断（uniform 分支）
                IDX_T c = AscendC::Simt::UintDiv<IDX_T>(rem, outMagic[d], outShift[d]);
                rem -= c * static_cast<IDX_T>(outStrides[d]);
                if (d != dim) { // dim 维坐标用 j 替代，此处跳过
                    inOff += c * static_cast<IDX_T>(inStrides[d]);
                }
            }
        }

        // 4. 最后一维坐标 c_{R-1} = rem（其 inStrides[R-1] 恒为 1）
        if (rank - 1 != dim) {
            inOff += rem;
        }

        // 5. dim 维贡献（用 index 值 j 而非解码得到的 c_dim）
        inOff += jU * static_cast<IDX_T>(inStrideDim);

        // 6. 字节级搬移（纯 bit 复制，NaN/Inf/±0 原样透传）
        outputBytes[i] = inputBytes[inOff];
    }
}

// ============================================================
// Process：scalar 端调度入口（MDE 7.2）
// ============================================================
template <typename T, typename INDEX_T, typename IDX_T>
__aicore__ inline void Process(GM_ADDR input, GM_ADDR index, GM_ADDR output, const CustomGatherTilingData* td)
{
    // 1. 空张量早退（须先于快除预计算：outSize==0 时 outStrides 可能含 0 除数）
    if (td->outSize == 0) {
        return;
    }

    // 2. UB 分配 + TilingData 数组中转 + 快除 magic/shift 预计算
    LocalMemAllocator<Hardware::UB> ubAlloc;
    LocalTensor<int64_t> outStridesUb = ubAlloc.Alloc<int64_t>(MAX_RANK);
    LocalTensor<int64_t> inStridesUb = ubAlloc.Alloc<int64_t>(MAX_RANK);
    LocalTensor<IDX_T> outMagicUb = ubAlloc.Alloc<IDX_T>(MAX_RANK);
    LocalTensor<IDX_T> outShiftUb = ubAlloc.Alloc<IDX_T>(MAX_RANK);
    __ubuf__ int64_t* outStridesPtr = (__ubuf__ int64_t*)outStridesUb.GetPhyAddr();
    __ubuf__ int64_t* inStridesPtr = (__ubuf__ int64_t*)inStridesUb.GetPhyAddr();
    __ubuf__ IDX_T* outMagicPtr = (__ubuf__ IDX_T*)outMagicUb.GetPhyAddr();
    __ubuf__ IDX_T* outShiftPtr = (__ubuf__ IDX_T*)outShiftUb.GetPhyAddr();

    for (int32_t d = 0; d < td->inputRank - 1; ++d) {
        outStridesPtr[d] = td->outStrides[d];
        inStridesPtr[d] = td->inStrides[d];
        // GetUintDivMagicAndShift 出参禁止 __ubuf__ 左值，先算标量再写 UB
        IDX_T magic = 0;
        IDX_T shift = 0;
        AscendC::GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(td->outStrides[d]));
        outMagicPtr[d] = magic;
        outShiftPtr[d] = shift;
    }
    DataSyncBarrier<MemDsbT::UB>(); // scalar 写 UB 后同步，保证 VF 可见

    // 3. GM 指针转换 + 分派（inputRank/dim/strides 均为合轴后视图值）
    __gm__ T* inputGm = reinterpret_cast<__gm__ T*>(input);
    __gm__ INDEX_T* indexGm = reinterpret_cast<__gm__ INDEX_T*>(index);
    __gm__ T* outputGm = reinterpret_cast<__gm__ T*>(output);

    if constexpr (sizeof(IDX_T) == 4) {
        switch (td->inputRank) {
            case 1:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 1>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 2:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 2>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 3:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 3>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 4:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 4>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 5:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 5>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 6:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 6>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 7:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 7>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            case 8:
                asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T, 8>>(
                    dim3(THREAD_NUM), td->outSize, td->dim, td->gatherDimSize, td->inStrideDim, outStridesPtr,
                    inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
                break;
            default:
                break; // 不可达：tiling 已校验 inputRank ∈ [1, 8]
        }
    } else {
        // IDX64（uint64）路径：运行期 rank
        asc_vf_call<GatherSimtKernel<T, INDEX_T, IDX_T>>(
            dim3(THREAD_NUM), td->outSize, td->dim, td->inputRank, td->gatherDimSize, td->inStrideDim, outStridesPtr,
            inStridesPtr, outMagicPtr, outShiftPtr, inputGm, indexGm, outputGm);
    }
}

} // namespace NsCustomGather

#endif // CUSTOM_GATHER_SIMT_H
