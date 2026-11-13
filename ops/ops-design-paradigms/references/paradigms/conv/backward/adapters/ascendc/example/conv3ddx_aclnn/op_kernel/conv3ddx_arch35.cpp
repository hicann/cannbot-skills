/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3ddx_arch35.cpp
 * \brief Conv3ddx 算子 kernel 入口（arch35 架构，被 asc_opc 编译的文件）
 *
 * 实现真值：design/Kernel.md §1「Kernel 入口」逐字实装——
 *   - 入口名 conv3ddx（snake_case，Interface.md「OpDef 代码片段」opFile/opInterface 绑定值），
 *     签名五模板参数 + 六 GM 参数，声明序 input_size、filter、out_backprop、y、workspace、tiling；
 *   - 实例化：包含 arch35/conv3ddx_struct.h 后，ASCENDC_TPL_SEL 以三组实参实例化本入口模板
 *     （loadB2Condition ∈ {1,2,3} → tilingKey 16777217/16777218/16777219，TilingKey.md）；
 *     每个dtype×format 组合（.o）内 3 个 sub-kernel（符号后缀 = key 位编码值）；
 *   - dtype/format 来自构建二重枚举的外层注入（registry-invoke def 驱动模式）：编译单元内
 *     DTYPE_FILTER / DTYPE_OUT_BACKPROP / DTYPE_Y 与 FORMAT_FILTER / FORMAT_OUT_BACKPROP /
 *     FORMAT_Y 宏绑定该组合（dtype 与 format 不进入 TilingKey 位域，TilingKey.md §1）；
 *   - TilingData 读取：REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 整体读入
 *     Conv3ddxTilingData（TilingData.md「TilingData C++ 结构体定义」，三 key 共享无分档）；
 *   - 初始化协议：主计算完整写回变体（Kernel.md 9.5，AIC_ONLY 无融合 AIV 清零）——
 *     needCompute=false 块由 kernel 类 ComputeZeroBlock 显式写零（零权重单次 MMAD +
 *     Fixpipe 通路，见 conv3ddx_kernel.h 机制偏离登记），入口无 InitOutput 调用。
 */

#include "kernel_operator.h"
#include "arch35/conv3ddx_tiling_struct.h"
#include "arch35/conv3ddx_struct.h"
#include "arch35/conv3ddx_kernel.h"

template <uint8_t loadB2Condition, uint8_t kernelSplitMode, uint8_t groupConvMode, bool isBasicBlockTiling,
          uint8_t loadB1Condition>
__global__ __aicore__ void conv3ddx(GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Conv3ddxTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3ddxTilingData, tilingData, tiling);
    /* input_size 的元素值已由 Host 解析装载 tilingData（Interface.md「算子原型」输入 0
     * 值依赖，仅作形状描述符不参与浮点计算）；本执行配置无用户 workspace
     * （HostTiling.md「Host 侧注册」ws[0]=0）。二者设备侧不消费，显式弃置。 */
    (void)input_size;
    (void)workspace;
    /* 后四个 TPL 实参为当前执行配置固定值（kernelSplitMode=0 / groupConvMode=0 /
     * isBasicBlockTiling=1 / loadB1Condition=0，TilingKey.md §1），入口体内不消费，
     * 由 ASCENDC_TPL_SEL 的实例集合保证取值域。 */
    (void)kernelSplitMode;
    (void)groupConvMode;
    (void)isBasicBlockTiling;
    (void)loadB1Condition;
    /* 退化空任务防御性短路（DevView.md「文件组织」空张量与退化任务的处理链）：
     * 未分到主计算任务的参与核（blockIdx ≥ coreNum）与任务量为 0 的防御分支直接返回，
     * 不触计算管线；空张量本体已在 Host 校验层拒绝，不进入路由。 */
    if (GetBlockIdx() >= tilingData.coreNum || tilingData.totalTasks == 0u) {
        return;
    }
    if ASCEND_IS_AIC {
        Conv3ddxKernel<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y,
                       loadB2Condition>
            op;
        op.Init(filter, out_backprop, y, tilingData);
        op.Process();
        op.End();
    }
}
