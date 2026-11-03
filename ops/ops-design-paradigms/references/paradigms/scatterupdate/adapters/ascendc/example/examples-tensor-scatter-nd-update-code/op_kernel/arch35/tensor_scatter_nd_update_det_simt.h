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
 * \file tensor_scatter_nd_update_det_simt.h
 * \brief SIMT kernel implementation for tensor_scatter_nd_update（确定性模板，v2.0 自
 *        tensor_scatter_nd_update_simt.h 改名，与 host 侧 tensor_scatter_nd_update_det_tiling
 *        对称，MDE §11.8）
 *
 * 三段式确定性算法（完整 N 维坐标语义，MDE §5，inplace：var 原地覆盖）：
 *   Phase 1: 线性索引计算（indices 最后一维 K 连续，直接 indices[pos*K+d] 读取；
 *            越界/负索引写哨兵 key 标记跳过，不用 __trap）
 *   Phase 2: SortLib radix sort（SIMD，升序稳定排序 linear_idx）
 *   Phase 3: 段尾写（段尾线程单写者 last-write-wins，无原子操作）；
 *            通用路径 flat 空间 [N, slice] 快除反解 (k, s) 让 slice 分到相邻线程（coalesced），
 *            slice 轴字节宽打包（W=2/4/8，uint16/uint32/uint64 视图，最大 64 位），
 *            slice==1（K==rank）走无快除快速路径
 *
 * 纯数据搬运算子：Phase3 逐元素 var[dst]=updates[src] 赋值，无算术、无类型提升，
 * 所有 dtype（含 float16/bfloat16/float8/bool/int8/uint8）均 bit-exact 搬运。
 *
 * v2.0 拆分：共享工具（THREAD_NUM/MAX_PACK_BYTES/AlignAllocN/PackType/ComputePackParam）
 * 移至 tensor_scatter_nd_update_common.h，非确定（直接写覆盖）路径移至
 * tensor_scatter_nd_update_nondet_simt.h；本文件仅含确定三段式实现
 * （纯代码移动零语义变更，MDE §11.8）。
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_DET_SIMT_H
#define TENSOR_SCATTER_ND_UPDATE_DET_SIMT_H

#include "tensor_scatter_nd_update_common.h" // v2.0 拆分：共享工具（THREAD_NUM/打包/对齐）
#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "tensor_scatter_nd_update_tiling_data.h"
#include "tensor_scatter_nd_update_tiling_key.h"
#include "sort_lib.h"

namespace NsTensorScatterNdUpdate {
using namespace AscendC;

// ============================================================
// Phase 1: 线性索引计算（完整 N 维坐标，无坐标反解除法）
//   indices 展平行主序，最后一维 K 连续，第 pos 个 update 的 K 维坐标 = indices[pos*K+d]。
//   lin = Σ_{d<K} indices[pos*K+d] · varStrides[d]，即目标 slice 在 var 中的起始元素偏移。
//   任一维越界/负值 → valid=false，写哨兵 key（KeyT 最大值），Phase3 跳过不写入（SE §5.4）。
// ============================================================
template <typename IdxT, typename KeyT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void Phase1LinearIdxKernel(
    int64_t totalIndexNum, int32_t indexDimNum, __gm__ const IdxT* indices, __gm__ KeyT* linearIdxOut,
    __ubuf__ const int64_t* varDims, __ubuf__ const int64_t* varStrides)
{
    const KeyT sentinel = std::numeric_limits<KeyT>::max();
    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * THREAD_NUM + static_cast<int64_t>(threadIdx.x);
         pos < totalIndexNum; pos += static_cast<int64_t>(gridDim.x) * THREAD_NUM) {
        int64_t base = pos * static_cast<int64_t>(indexDimNum);
        int64_t lin = 0;
        bool valid = true;
        for (int32_t d = 0; d < indexDimNum; d++) {
            int64_t j = static_cast<int64_t>(indices[base + d]);
            if (j < 0 || j >= varDims[d]) {
                valid = false;
                break;
            }
            lin += j * varStrides[d];
        }
        linearIdxOut[pos] = valid ? static_cast<KeyT>(lin) : sentinel;
    }
}

// ============================================================
// Phase 3 段尾写（last-write-wins，无原子操作）
//   排序后同 target slice 的 update 连续成段，段内原始顺序唯一（stable sort 保证），
//   段尾元素（perm[k]，原始行号最大）即最后写入，故「段尾线程」直接写回。
//   统一按字节宽打包：slice 轴 W 元素打包，W ∈ {1,2,4,8}（W=1 即标量，PackType<T,1>=T）。
//   sliceVec = sliceSize/W，flat 空间 [N, sliceVec]，快除反解 (k, sw) = (task/sliceVec, task%sliceVec)，
//   相邻线程 → 相邻 sw → 打包元素天然分到相邻线程（coalesced 访存）。
//   var/updates 以 PackT（T / uint16 / uint32 / uint64）视角重解释，一条指令读写 W 元素。
//   dst 下标 = key/W + sw（key 为 sliceSize 整数倍，W 为 2 的幂，key/W 精确）。
// ============================================================
template <typename T, typename KeyT, int W, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void Phase3ScatterUpdateKernelPack(
    IndexT totalIndexNum, IndexT sliceVec, __gm__ const T* updates, __gm__ T* var, __gm__ const KeyT* sortedLinearIdx,
    __gm__ const int32_t* perm, __ubuf__ const uint64_t* divParams)
{
    using PackT = typename PackType<T, W>::type;
    using IndexU = typename std::make_unsigned<IndexT>::type;
    const KeyT sentinel = std::numeric_limits<KeyT>::max();
    const IndexU svMagic = static_cast<IndexU>(divParams[0]);
    const IndexU svShift = static_cast<IndexU>(divParams[1]);
    __gm__ PackT* varVec = reinterpret_cast<__gm__ PackT*>(var);
    __gm__ const PackT* updVec = reinterpret_cast<__gm__ const PackT*>(updates);
    const IndexT totalTask = totalIndexNum * sliceVec;
    for (IndexT task = static_cast<IndexT>(blockIdx.x) * THREAD_NUM + static_cast<IndexT>(threadIdx.x);
         task < totalTask; task += static_cast<IndexT>(gridDim.x) * THREAD_NUM) {
        IndexU kU = Simt::UintDiv<IndexU>(static_cast<IndexU>(task), svMagic, svShift); // ÷sliceVec
        IndexT k = static_cast<IndexT>(kU);
        IndexT sw = task - k * sliceVec;
        KeyT key = sortedLinearIdx[k];
        if (key == sentinel) {
            continue;
        }
        if (k < totalIndexNum - 1 && sortedLinearIdx[k + 1] == key) {
            continue;
        }
        varVec[(static_cast<int64_t>(key) / W) + static_cast<int64_t>(sw)] =
            updVec[static_cast<int64_t>(perm[k]) * sliceVec + static_cast<int64_t>(sw)];
    }
}

// ============================================================
// Phase 3 快速路径（sliceSize == 1，即 K == rank 标量更新）：
//   flat 空间退化为 [N]，无 s 维、无快除，段尾线程直接写单个元素。
// ============================================================
template <typename T, typename KeyT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void Phase3ScatterUpdateKernelSlice1(
    IndexT totalIndexNum, __gm__ const T* updates, __gm__ T* var, __gm__ const KeyT* sortedLinearIdx,
    __gm__ const int32_t* perm)
{
    const KeyT sentinel = std::numeric_limits<KeyT>::max();
    for (IndexT k = static_cast<IndexT>(blockIdx.x) * THREAD_NUM + static_cast<IndexT>(threadIdx.x); k < totalIndexNum;
         k += static_cast<IndexT>(gridDim.x) * THREAD_NUM) {
        KeyT key = sortedLinearIdx[k];
        if (key == sentinel) {
            continue;
        }
        if (k < totalIndexNum - 1 && sortedLinearIdx[k + 1] == key) {
            continue;
        }
        var[static_cast<int64_t>(key)] = updates[static_cast<int64_t>(perm[k])];
    }
}

// ============================================================
// Phase 3 分派（slice==1 最外层快速路径；否则纯按 W 分派，不再感知除法）
//   W==1 → Pack<1>（标量），W==2/4/8 → Pack<2/4/8>（if constexpr 防无效实例化）。
// ============================================================
template <typename T, typename KeyT, typename IndexT>
__aicore__ inline void DispatchPhase3(int64_t W, IndexT sliceSize, IndexT sliceVec, IndexT totalIndexNum,
                                      __gm__ const T* updGm, __gm__ T* varGm, __gm__ const KeyT* sortedGm,
                                      __gm__ const int32_t* permGm, __ubuf__ const uint64_t* ubDivParams)
{
    // slice==1 提到最外层：无除法快速路径
    if (sliceSize == 1) {
        asc_vf_call<Phase3ScatterUpdateKernelSlice1<T, KeyT, IndexT>>(dim3(THREAD_NUM), totalIndexNum, updGm, varGm,
                                                                      sortedGm, permGm);
        return;
    }
    // sliceSize > 1：纯按 W 分派
    if (W == 1) {
        asc_vf_call<Phase3ScatterUpdateKernelPack<T, KeyT, 1, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec, updGm,
                                                                       varGm, sortedGm, permGm, ubDivParams);
        return;
    }
    if constexpr (sizeof(T) <= 4) {
        if (W == 2) {
            asc_vf_call<Phase3ScatterUpdateKernelPack<T, KeyT, 2, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           updGm, varGm, sortedGm, permGm, ubDivParams);
            return;
        }
    }
    if constexpr (sizeof(T) <= 2) {
        if (W == 4) {
            asc_vf_call<Phase3ScatterUpdateKernelPack<T, KeyT, 4, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           updGm, varGm, sortedGm, permGm, ubDivParams);
            return;
        }
    }
    if constexpr (sizeof(T) <= 1) {
        if (W == 8) {
            asc_vf_call<Phase3ScatterUpdateKernelPack<T, KeyT, 8, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           updGm, varGm, sortedGm, permGm, ubDivParams);
            return;
        }
    }
}

// ============================================================
// Process 入口：三阶段编排（inplace：var 原地覆盖，无 Stage A 抄写）
// ============================================================
template <typename T, typename IdxT, typename CountT, typename KeyT>
__aicore__ inline void Process(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_out, GM_ADDR workspace,
                               const TensorScatterNdUpdateTilingData* td)
{
    (void)var_out; // inplace：var_out 与 var 为同一 GM_ADDR，Phase3 直接写 var
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;

    __gm__ T* varGm = reinterpret_cast<__gm__ T*>(var);
    __gm__ IdxT* idxGm = reinterpret_cast<__gm__ IdxT*>(indices);
    __gm__ T* updGm = reinterpret_cast<__gm__ T*>(updates);
    __gm__ char* usrWs = reinterpret_cast<__gm__ char*>(AscendC::GetUserWorkspace(workspace));

    // 空 tensor 短路（SE §5.4）：indices 空或 var 含 0 维，输出 = var 本身
    if (td->totalIndexNum == 0 || td->totalVarElems == 0) {
        return;
    }

    // 准备 Phase1 UB 参数（varDims/varStrides，TPipe 临时 buffer），scalar 写后 DataSyncBarrier
    AscendC::TBuf<AscendC::TPosition::VECCALC> varDimsBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> varStridesBuf;
    pipe.InitBuffer(varDimsBuf, static_cast<uint32_t>(AlignAllocN(8, sizeof(int64_t)) * sizeof(int64_t)));
    pipe.InitBuffer(varStridesBuf, static_cast<uint32_t>(AlignAllocN(8, sizeof(int64_t)) * sizeof(int64_t)));
    __ubuf__ int64_t* ubVarDims = reinterpret_cast<__ubuf__ int64_t*>(varDimsBuf.template Get<int64_t>().GetPhyAddr());
    __ubuf__ int64_t* ubVarStrides =
        reinterpret_cast<__ubuf__ int64_t*>(varStridesBuf.template Get<int64_t>().GetPhyAddr());
    for (int32_t d = 0; d < td->varRank; d++) {
        ubVarDims[d] = td->varDims[d];
        ubVarStrides[d] = td->varStrides[d];
    }
    DataSyncBarrier<MemDsbT::UB>();

    __gm__ KeyT* linearIdxGm = reinterpret_cast<__gm__ KeyT*>(usrWs + td->wsLinearIdxOff);
    __gm__ KeyT* sortedGm = reinterpret_cast<__gm__ KeyT*>(usrWs + td->wsSortedOff);
    __gm__ int32_t* permGm = reinterpret_cast<__gm__ int32_t*>(usrWs + td->wsPermOff);

    // === Phase 1: 计算 linear_idx（workspace 段2） ===
    asc_vf_call<Phase1LinearIdxKernel<IdxT, KeyT>>(dim3(THREAD_NUM), td->totalIndexNum, td->indexDimNum, idxGm,
                                                   linearIdxGm, ubVarDims, ubVarStrides);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    SyncAll();

    // === Phase 2: SortLib radix sort（升序稳定排序 linear_idx） ===
    SortLib::SortParams p;
    p.numTileData = td->numTileData;
    p.tileCount = td->tileCount;
    p.activeCores = td->activeCores;
    p.tmpUbSize = td->tmpUbSize;
    p.totalElements = td->totalIndexNum;
    p.isSingleCore = td->isSingleCore;

    pipe.Reset(); // 调用前释放此前占用的 UB，保证 SortInvoke 可用满 UB
    SortLib::SortInvoke<KeyT, int32_t, CountT, false>(&pipe, linearIdxGm, sortedGm, permGm, usrWs, p);
    pipe.Reset(); // 调用后释放本接口占用的 UB，保证后续接口可用
    SyncAll();

    // 准备 Phase3 快除参数（TPipe 临时 buffer，此时 UB 池已空，无地址重叠）
    //   [0..1] = {sliceVecMagic, sliceVecShift}（÷sliceVec，唯一一组）
    AscendC::TBuf<AscendC::TPosition::VECCALC> divParamsBuf;
    pipe.InitBuffer(divParamsBuf, static_cast<uint32_t>(AlignAllocN(2, sizeof(uint64_t)) * sizeof(uint64_t)));
    __ubuf__ uint64_t* ubDivParams =
        reinterpret_cast<__ubuf__ uint64_t*>(divParamsBuf.template Get<uint64_t>().GetPhyAddr());

    int64_t sliceVec = td->sliceSize;
    int64_t W = ComputePackParam(td->sliceSize, static_cast<int64_t>(sizeof(T)), sliceVec);

    // 快除 magic/shift：按 useInt32Index 选 uint32 / uint64（uint32 存 uint64 低位）
    if (td->useInt32Index != 0) {
        uint32_t svMagic = 0;
        uint32_t svShift = 0;
        if (td->sliceSize > 1) {
            GetUintDivMagicAndShift<uint32_t>(svMagic, svShift, static_cast<uint32_t>(sliceVec));
        }
        ubDivParams[0] = static_cast<uint64_t>(svMagic);
        ubDivParams[1] = static_cast<uint64_t>(svShift);
    } else {
        uint64_t svMagic = 0;
        uint64_t svShift = 0;
        if (td->sliceSize > 1) {
            GetUintDivMagicAndShift<uint64_t>(svMagic, svShift, static_cast<uint64_t>(sliceVec));
        }
        ubDivParams[0] = svMagic;
        ubDivParams[1] = svShift;
    }
    DataSyncBarrier<MemDsbT::UB>();

    // === Phase 3: 段尾写（inplace 写 var，快除反解 + 字节宽打包 + slice==1 快速路径） ===
    if (td->useInt32Index != 0) {
        DispatchPhase3<T, KeyT, int32_t>(W, static_cast<int32_t>(td->sliceSize), static_cast<int32_t>(sliceVec),
                                         static_cast<int32_t>(td->totalIndexNum), updGm, varGm, sortedGm, permGm,
                                         ubDivParams);
    } else {
        DispatchPhase3<T, KeyT, int64_t>(W, td->sliceSize, sliceVec, td->totalIndexNum, updGm, varGm, sortedGm, permGm,
                                         ubDivParams);
    }
}

} // namespace NsTensorScatterNdUpdate

#endif
