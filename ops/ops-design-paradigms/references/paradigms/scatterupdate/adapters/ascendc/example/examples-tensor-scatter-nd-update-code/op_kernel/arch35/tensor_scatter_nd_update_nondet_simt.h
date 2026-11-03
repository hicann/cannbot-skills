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
 * \file tensor_scatter_nd_update_nondet_simt.h
 * \brief v2.0 非确定（直接写覆盖）路径实现（MDE §11.6）
 *
 * 单 VF 直接并发写：task ∈ flat [N, sliceVec]（pack 后）或 [N]（slice==1 快路径）。
 *   覆盖语义核心（MDE §11.1.2）：赋值覆盖不需要任何原子指令——多写者"任意赢"
 *   即正确语义（"丢失"其他写者正是覆盖语义本身）；全部 10 种 data dtype 的
 *   对齐单条 store 均原子安全（MDE §11.6.4 四步论证），无 dtype 白名单。
 *   任务反解（÷sliceVec 快除，magic/shift 经 UB 传 VF）
 *   + 内联坐标线性化（与 det Phase1 同款 UB 数组参数：varStrides/varDims/indices
 *     坐标读取，indices 最后一维 K 连续，直接偏移读取，无除法）
 *   + 越界/负索引 continue 跳过该 task（与 det 哨兵 key 跳过语义一致，SE §5.4；
 *     勿照抄 scatter_add 范式 §四的 __trap 骨架——该基例 SE 语义为硬失败，与本算子不同）
 *   + 直接写 var[target] = updates[...]（W∈{1,2,4,8} 字节宽打包完全复用 det 的
 *     ComputePackParam/PackType，单一事实源在 common.h）。
 *   无排序/段表/用户 ws/SyncAll；grid-stride 兜底：blockDim 不足时每线程多任务。
 *   kernel 末尾 VF 后无 V_S join（范式 §3.4：AIV task 结束协议排空向量管线）。
 *   IndexT 域模板化复用基线：门控 useInt32Index=1（N*sliceSize <= INT32_MAX）时
 *   flat 任务空间全链路 int32 域（uint32 快除），否则 int64 域（零行为变化）。
 * 仅由 apt.cpp 的 TSNU_MODE_NONDET 分支调用；共享工具见 tensor_scatter_nd_update_common.h。
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_NONDET_SIMT_H
#define TENSOR_SCATTER_ND_UPDATE_NONDET_SIMT_H

#include "tensor_scatter_nd_update_common.h"
#include "simt_api/asc_simt.h"                    // 快除/同步（common.h 级联，此处显式声明依赖）
#include "tensor_scatter_nd_update_tiling_data.h" // TensorScatterNdUpdateNondetTilingData

namespace NsTensorScatterNdUpdate {
using namespace AscendC;

// ============================================================
// v2.0 非确定直接写覆盖 VF（通用路径，sliceSize > 1，MDE §11.6.2）
//   任务空间：flat [N, sliceVec]（sliceVec = sliceSize/W，pack 后），相邻线程 → 相邻 sw
//   → coalesced。每个 task（Grid-Stride 兜底）：
//     1. 快除反解：k = task ÷ sliceVec（UintDiv，magic/shift 经 UB 传 VF）；sw = task - k*sliceVec
//     2. 坐标线性化（与 det Phase1 同款，内联）：base = k*K（int64 乘法，无除法）；
//        key = Σ_{d<K} indices[base+d]·varStrides[d]，任一维 j<0 或 j>=varDims[d] → invalid
//     3. 越界/负索引 → continue 跳过（与 det 哨兵 key 跳过语义一致）
//     4. 直接写（无原子、无段首判据、无 perm）：varVec[key/W + sw] = updVec[k*sliceVec + sw]
//   IndexT 域安全（与 det Phase3 同款）：门控 useInt32Index=1 ⇒ N*sliceSize <= INT32_MAX
//   ⇒ totalTask = N*sliceVec ≤ N*sliceSize ≤ INT32_MAX、sw < sliceVec、k*sliceVec+sw <
//   totalTask，全链路无溢出；key 与 key/W+sw（var 侧下标）保持 int64 域计算
//   （totalVarElems 可超 2^31）。
// ============================================================
template <typename T, typename IdxT, int W, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void NondetScatterUpdateKernelPack(
    IndexT totalIndexNum, IndexT sliceVec, int32_t indexDimNum, __gm__ const IdxT* indices, __gm__ const T* updates,
    __gm__ T* var, __ubuf__ const int64_t* varDims, __ubuf__ const int64_t* varStrides,
    __ubuf__ const uint64_t* divParams)
{
    using PackT = typename PackType<T, W>::type;
    using IndexU = typename std::make_unsigned<IndexT>::type;
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
        // 坐标线性化（与 det Phase1 同款：indices 最后一维 K 连续，直接偏移读取，无除法）
        int64_t base = static_cast<int64_t>(k) * static_cast<int64_t>(indexDimNum);
        int64_t key = 0;
        bool valid = true;
        for (int32_t d = 0; d < indexDimNum; d++) {
            int64_t j = static_cast<int64_t>(indices[base + d]);
            if (j < 0 || j >= varDims[d]) {
                valid = false;
                break;
            }
            key += j * varStrides[d];
        }
        if (!valid) {
            continue; // 越界/负索引跳过（与 det 哨兵语义一致，SE §5.4）
        }
        // 直接写（无原子）：key 为 sliceSize 整数倍且 W | sliceSize ⇒ key%W==0 ⇒ key/W 精确（W 为 2 的幂）
        varVec[(key / W) + static_cast<int64_t>(sw)] =
            updVec[static_cast<int64_t>(k) * static_cast<int64_t>(sliceVec) + static_cast<int64_t>(sw)];
    }
}

// ============================================================
// v2.0 非确定直接写覆盖 VF（slice==1 快路径，MDE §11.6.3）
//   sliceSize==1（K==rank，标量更新）时任务空间退化为一维 [N]，无快除、无打包。
// ============================================================
template <typename T, typename IdxT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void NondetScatterUpdateKernelSlice1(
    IndexT totalIndexNum, int32_t indexDimNum, __gm__ const IdxT* indices, __gm__ const T* updates, __gm__ T* var,
    __ubuf__ const int64_t* varDims, __ubuf__ const int64_t* varStrides)
{
    for (IndexT k = static_cast<IndexT>(blockIdx.x) * THREAD_NUM + static_cast<IndexT>(threadIdx.x); k < totalIndexNum;
         k += static_cast<IndexT>(gridDim.x) * THREAD_NUM) {
        int64_t base = static_cast<int64_t>(k) * static_cast<int64_t>(indexDimNum);
        int64_t key = 0;
        bool valid = true;
        for (int32_t d = 0; d < indexDimNum; d++) {
            int64_t j = static_cast<int64_t>(indices[base + d]);
            if (j < 0 || j >= varDims[d]) {
                valid = false;
                break;
            }
            key += j * varStrides[d];
        }
        if (!valid) {
            continue;
        }
        var[key] = updates[static_cast<int64_t>(k)]; // 单元素直接写
    }
}

// ============================================================
// v2.0 W 编译期分派（与 det DispatchPhase3 同构，MDE §11.6.2/§11.11.2）
//   W==1 → Pack<1>（标量），W==2/4/8 → Pack<2/4/8>（if constexpr 防无效实例化）。
//   sliceSize==1 已在 NondetProcess 外层拦截，本函数仅 sliceSize>1 场景可达；
//   sliceSize 参数防御性保留（与 DispatchPhase3 签名对称）。
// ============================================================
template <typename T, typename IdxT, typename IndexT>
__aicore__ inline void DispatchNondetPack(int64_t W, IndexT sliceVec, IndexT totalIndexNum, IndexT sliceSize,
                                          int32_t indexDimNum, __gm__ const IdxT* idxGm, __gm__ const T* updGm,
                                          __gm__ T* varGm, __ubuf__ const int64_t* ubVarDims,
                                          __ubuf__ const int64_t* ubVarStrides, __ubuf__ const uint64_t* ubDivParams)
{
    (void)sliceSize; // 外层已拦截 sliceSize==1，防御性保留参数
    if (W == 1) {
        asc_vf_call<NondetScatterUpdateKernelPack<T, IdxT, 1, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                       indexDimNum, idxGm, updGm, varGm, ubVarDims,
                                                                       ubVarStrides, ubDivParams);
        return;
    }
    if constexpr (sizeof(T) <= 4) {
        if (W == 2) {
            asc_vf_call<NondetScatterUpdateKernelPack<T, IdxT, 2, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           indexDimNum, idxGm, updGm, varGm, ubVarDims,
                                                                           ubVarStrides, ubDivParams);
            return;
        }
    }
    if constexpr (sizeof(T) <= 2) {
        if (W == 4) {
            asc_vf_call<NondetScatterUpdateKernelPack<T, IdxT, 4, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           indexDimNum, idxGm, updGm, varGm, ubVarDims,
                                                                           ubVarStrides, ubDivParams);
            return;
        }
    }
    if constexpr (sizeof(T) <= 1) {
        if (W == 8) {
            asc_vf_call<NondetScatterUpdateKernelPack<T, IdxT, 8, IndexT>>(dim3(THREAD_NUM), totalIndexNum, sliceVec,
                                                                           indexDimNum, idxGm, updGm, varGm, ubVarDims,
                                                                           ubVarStrides, ubDivParams);
            return;
        }
    }
}

// ============================================================
// v2.0 非确定路径入口：单 VF 直接写覆盖（无排序/无段表/无用户 ws/无 SyncAll，MDE §11.6.1）
//   仅由 apt.cpp 的 TSNU_MODE_NONDET 分支调用。
//   与 det Process 的差异清单：KERNEL_TYPE_AIV_ONLY（det 为 MIX_AIV_1_0，因 SortLib）；
//   无 Phase1/Phase2 编排；无 SyncAll；无 SetFlag/WaitFlag<V_S>（det 在 Phase1/Phase3 后
//   有跨核屏障前的 V_S join，nondet 全程单 VF 无屏障）；UB 分配用 LocalMemAllocator
//   （det 与 SortLib 共用 TPipe）。
// ============================================================
template <typename T, typename IdxT>
__aicore__ inline void NondetProcess(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_out, GM_ADDR workspace,
                                     const TensorScatterNdUpdateNondetTilingData* td)
{
    (void)var_out;   // inplace：var_out 与 var 同一 GM_ADDR（Output("var") 与 Input("var") 同名）
    (void)workspace; // 无用户 workspace（系统 ws 框架自管）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 SIMT VF，无 SortLib（两基例 nondet 同款已验证）

    __gm__ T* varGm = reinterpret_cast<__gm__ T*>(var);
    __gm__ IdxT* idxGm = reinterpret_cast<__gm__ IdxT*>(indices);
    __gm__ T* updGm = reinterpret_cast<__gm__ T*>(updates);

    // 空 tensor 早退（与确定路径 Process 一致，SE §5.4）
    if (td->totalIndexNum == 0 || td->totalVarElems == 0) {
        return;
    }

    // ── UB 参数准备（LocalMemAllocator，32B 对齐；无 SortLib 故不依赖 TPipe）──
    //   varDims[8] + varStrides[8]（坐标线性化，与 det Phase1 同源）+ divParams[2]（÷sliceVec 快除）
    LocalMemAllocator<Hardware::UB> ubAlloc;
    LocalTensor<int64_t> varDimsTensor = ubAlloc.Alloc<int64_t>(AlignAllocN(8, sizeof(int64_t)));
    LocalTensor<int64_t> varStridesTensor = ubAlloc.Alloc<int64_t>(AlignAllocN(8, sizeof(int64_t)));
    LocalTensor<uint64_t> divParamsTensor = ubAlloc.Alloc<uint64_t>(AlignAllocN(2, sizeof(uint64_t)));
    __ubuf__ int64_t* ubVarDims = reinterpret_cast<__ubuf__ int64_t*>(varDimsTensor.GetPhyAddr());
    __ubuf__ int64_t* ubVarStrides = reinterpret_cast<__ubuf__ int64_t*>(varStridesTensor.GetPhyAddr());
    __ubuf__ uint64_t* ubDivParams = reinterpret_cast<__ubuf__ uint64_t*>(divParamsTensor.GetPhyAddr());

    for (int32_t d = 0; d < td->varRank; d++) {
        ubVarDims[d] = td->varDims[d];
        ubVarStrides[d] = td->varStrides[d];
    }

    // ── 打包宽度 W 与 sliceVec（与 det Phase3 完全复用同款逻辑，common.h 单一事实源）──
    int64_t sliceVec = td->sliceSize;
    int64_t W = ComputePackParam(td->sliceSize, static_cast<int64_t>(sizeof(T)), sliceVec);

    // ── 快除参数（双宽度，与 det Process 同款）：÷sliceVec 唯一一组 ──
    if (td->useInt32Index != 0) {
        uint32_t svMagic = 0;
        uint32_t svShift = 0;
        if (td->sliceSize > 1) {
            GetUintDivMagicAndShift<uint32_t>(svMagic, svShift, static_cast<uint32_t>(sliceVec));
        }
        ubDivParams[0] = static_cast<uint64_t>(svMagic); // uint32 存 uint64 槽，kernel 端 cast 取回
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
    DataSyncBarrier<MemDsbT::UB>(); // scalar 写 UB 后对 VF 可见

    // ── VF 分派（slice==1 快路径最外层；IndexT 域运行期 if；W 编译期分派）──
    if (td->sliceSize == 1) {
        if (td->useInt32Index != 0) {
            asc_vf_call<NondetScatterUpdateKernelSlice1<T, IdxT, int32_t>>(
                dim3(THREAD_NUM), static_cast<int32_t>(td->totalIndexNum), td->indexDimNum, idxGm, updGm, varGm,
                ubVarDims, ubVarStrides);
        } else {
            asc_vf_call<NondetScatterUpdateKernelSlice1<T, IdxT, int64_t>>(
                dim3(THREAD_NUM), td->totalIndexNum, td->indexDimNum, idxGm, updGm, varGm, ubVarDims, ubVarStrides);
        }
        // 末尾 VF 后无 S 消费：AIV task 结束协议排空向量管线，无需 V_S join（基例已验证）
        return;
    }
    if (td->useInt32Index != 0) {
        DispatchNondetPack<T, IdxT, int32_t>(W, static_cast<int32_t>(sliceVec), static_cast<int32_t>(td->totalIndexNum),
                                             static_cast<int32_t>(td->sliceSize), td->indexDimNum, idxGm, updGm, varGm,
                                             ubVarDims, ubVarStrides, ubDivParams);
    } else {
        DispatchNondetPack<T, IdxT, int64_t>(W, sliceVec, td->totalIndexNum, td->sliceSize, td->indexDimNum, idxGm,
                                             updGm, varGm, ubVarDims, ubVarStrides, ubDivParams);
    }
    // 末尾 VF 后无 S 消费（同上）
}

} // namespace NsTensorScatterNdUpdate

#endif
