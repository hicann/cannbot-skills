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
 * \file conv3dtranspose_arch35.cpp
 * \brief Conv3DTranspose 算子 kernel 入口（arch35 架构，被 asc_opc 编译的文件）
 *
 * 实现真值：design/Kernel.md §1「Kernel 入口」逐字实装——
 *   - 入口名 conv3dtranspose（snake_case，Interface.md「OpDef 代码片段」opFile/opInterface 绑定值），
 *     签名五模板参数 + 七 GM 参数，声明序 input_size、x、filter、bias、y、workspace、tiling
 *     （输入按 Interface.md「算子原型」声明序，输出 y，再 workspace、tiling；x 在前、filter 在后——
 *     与 DX 入口位序差异的注册 ABI 恢复，Kernel.md §1 差异登记）；
 *     保留槽 offset_w 不注册为输入（4 输入注册与已验收白盒用例集的 4 输入行对齐——TTK kernel
 *     模式对输入数严格相等校验，5 输入注册使全部用例 INPUT_COUNT_MISMATCH；证据与升级记录见
 *     op_host/conv3dtranspose_def.cpp 集成注记与 tests/whitebox/branch-1/bringup/RECORD.md；
 *     Kernel.md §1 八 GM 参数帧的该保留槽由本注册对齐一并收缩，其余位序不变）；
 *   - 实例化：包含 arch35/conv3dtranspose_tiling_key.h 后，ASCENDC_TPL_SEL 以三组实参实例化
 *     本入口模板（loadB2Condition ∈ {1,2,3} → tilingKey 16777217/16777218/16777219，TilingKey.md）；
 *     每个 dtype×format 组合（.o）内 3 个 sub-kernel（符号后缀 = key 位编码值）；
 *   - dtype/format 来自构建二重枚举的外层注入（registry-invoke def 驱动模式）：编译单元内
 *     DTYPE_FILTER / DTYPE_X / DTYPE_Y 与 FORMAT_FILTER / FORMAT_X / FORMAT_Y 宏绑定该组合
 *     （dtype 与 format 不进入 TilingKey 位域，TilingKey.md §1），另含 bias dtype 的 DTYPE_BIAS
 *     （无 bias 组合的 bias 槽位按 DT_FLOAT 惯性填充，与对应 bias=fp32 组合共用注册元组）；
 *   - TilingData 读取：REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 整体读入
 *     Conv3DTransposeTilingData（TilingData.md「TilingData C++ 结构体定义」，三 key 共享无分档）；
 *   - bias 分路：bias GM 指针承载存在性——bias != nullptr 分路 Process<true> / Process<false>
 *     （存在性不占 key 位、不设 hasBiasFlag 字段）；0 元素空可选张量按无 bias 路由
 *     （HostTiling.md §2 槽位规则：元素数 > 0 即有 bias）——ACLNN 路由 op_api 包装层归一
 *     nullptr，kernel 直调路径（TTK 白盒 (0,) 惰性槽位）依据 Host 已发布的
 *     biasAllocBytes（无 bias 恒 0）在入口等价归一（见入口内注记与 run-20pct RECORD）；
 *   - 初始化协议：主计算完整写回变体（Kernel.md 9.6，AIC_ONLY 无融合 AIV 清零）——
 *     needCompute=false 块由 NoNeedComputeHandle 显式写回（带 bias 经 ComputeForBias 写
 *     bias 值、无 bias 经 ComputeZero 写零），入口无 InitOutput 调用。
 */

#include "kernel_operator.h"
#include "arch35/conv3dtranspose_tiling_data.h"
#include "arch35/conv3dtranspose_tiling_key.h"
#include "arch35/conv3dtranspose.h"

/* biasDtypeMode 模板实参 → BIAST 类型（9.9 校验项 9 fp16 扩展义务：bias dtype 经
 * tilingKey 位域自 Host 发布，运行期按 key 值选 sub-kernel 符号——TilingKey.md
 * 「运行期按 key 值直接选 sub-kernel 符号」既定机制；BIAST=half 实例的 L1→BT
 * 拓宽拷贝与两槽 4B 口径 sizing 见 Kernel.md「Init」/「CopyInBrc」，构建外层
 * DTYPE_BIAS 宏注入在本机制下不再消费（同 .o 内双实例并存）） */
template <uint8_t biasDtypeMode>
struct Conv3DTBiasTypeOf;
template <>
struct Conv3DTBiasTypeOf<TPL_BIAS_DTYPE_FLOAT> {
    using Type = float;
};
template <>
struct Conv3DTBiasTypeOf<TPL_BIAS_DTYPE_HALF> {
    using Type = half;
};

template <uint8_t loadB2Condition, uint8_t kernelSplitMode, uint8_t groupConvMode, bool isBasicBlockTiling,
          uint8_t loadB1Condition, uint8_t biasDtypeMode>
__global__ __aicore__ void conv3dtranspose(GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Conv3DTransposeTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DTransposeTilingData, tilingData, tiling);
    /* input_size 的元素值已由 Host 解析装载 tilingData（Interface.md「算子原型」输入 0
     * 值依赖，仅作形状描述符不参与浮点计算；q+rho 已折入目标几何与反向边界）；
     * 本执行配置无用户 workspace（HostTiling.md「Host 侧注册」workspace 前置，ws[0]=0）。
     * 二者设备侧不消费，显式弃置。保留槽 offset_w 未注册（见文件头注记），无设备位。 */
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
     * 不触计算管线；空张量本体已在 op_api 参数检查与 TilingFunc 异常值校验双层拒绝
     * （HostTiling.md「输入预处理 / 合轴 / 场景预处理」），不进入路由。 */
    if (GetBlockIdx() >= tilingData.coreNum || tilingData.totalTasks == 0u) {
        return;
    }
    if ASCEND_IS_AIC {
        Conv3dtranspose<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y, loadB2Condition,
                        typename Conv3DTBiasTypeOf<biasDtypeMode>::Type>
            op;
        /* 空可选 bias 张量归一（HostTiling.md §2「bias 存在性与 dtype 载体」/Kernel.md §1
         * bias 分路行：0 元素空可选张量按无 bias 路由，非报错）。ACLNN 调用路径该归一由
         * op_api 包装层执行（空张量统一归一 nullptr 后进入注册 ABI）；TTK kernel 直调路径
         * 无该包装层，白盒用例集的无 bias 行以 (0,) 惰性槽位到达——非空指针 + 零字节存储，
         * 而 Host 侧 TilingFunc 按同一槽位规则（元素数 > 0 即有 bias）已判定 hasBias=0。
         * 依据 Host 已发布的 TilingData（UpdateIsBiasFullLoad：无 bias 恒 biasAllocBytes=0、
         * 有 bias 恒 alignUp(biasUse·biasBytes,64) ≥ 64，TilingData.md「字段表」）在入口对齐
         * 同一存在性判定：tiling 判无 bias 时指针归一 nullptr，保持 hasBiasGm_ 与
         * Process<hasBias> 编译期分路恒一致，kernel 不读取零字节 bias 存储（否则经 BT 折入
         * 未初始化数据污染输出——分诊证据见 tests/whitebox/branch-1/run-20pct/RECORD.md）。 */
        GM_ADDR biasEffective = nullptr;
        if (tilingData.biasAllocBytes != 0u) {
            biasEffective = bias;
        }
        op.Init(filter, x, y, tilingData, biasEffective);
        if (biasEffective != nullptr) {
            op.template Process<true>();
        } else {
            op.template Process<false>();
        }
        op.End();
    }
}
