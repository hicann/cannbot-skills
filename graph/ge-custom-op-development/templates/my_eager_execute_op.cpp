/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 命令式待适配骨架：加载与 launch 未实现，明确返回失败。
// 接口和选择条件见 references/interfaces.md、scenarios.md。
#include <cstdio>
#include <limits>
#include <vector>

#include "graph/custom_op.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "graph/types.h"

namespace {
[[maybe_unused]] constexpr const char* kKernelName = "my_kernel";
constexpr int32_t kBlockSize = 1024;    // 与 kernel 侧一致
constexpr int64_t kMaxBlockDim = 65535; // 与官方 CalcBlockDim 的越界判据一致
} // namespace

// 类名必须与 REG_OP 原型名、构图侧 op type 三方一致
class MyEagerExecuteOp : public ge::EagerExecuteOp {
public:
    ge::graphStatus Execute(gert::EagerOpExecutionContext* ctx) override
    {
        // 1) 取输入（const，不可修改）
        auto* x = ctx->GetInputTensor(0);
        auto* y = ctx->GetInputTensor(1);
        if (x == nullptr || y == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 2) 分配输出（生命周期由 ctx 管理，调用者不要释放）
        auto* z = ctx->MallocOutputTensor(0, x->GetShape(), x->GetFormat(), x->GetDataType());
        if (z == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 3) 计算元素个数（注意：GetShapeSize() 是元素个数，GetSize() 是字节数）
        const int64_t n_elements = x->GetShapeSize();
        if (n_elements <= 0 || n_elements > kMaxBlockDim * kBlockSize) {
            return ge::GRAPH_FAILED;
        }
        // 仅对正数且有界的元素数向上取整；kernel 的尾块支持需另行确认。
        const int64_t grid_x_64 = (n_elements + kBlockSize - 1) / kBlockSize;
        const int64_t block_num_64 = grid_x_64 > 0 ? grid_x_64 : 1;
        // 上限 65535（与官方 CalcBlockDim 越界判据一致）：越界直接失败，
        // 不要强转 int32 —— 超了会静默截断成错误值，表现为结果错而不是报错。
        if (block_num_64 > kMaxBlockDim) {
            return ge::GRAPH_FAILED;
        }
        const int32_t grid_x = static_cast<int32_t>(block_num_64);
        const int32_t block_num = grid_x; // 一维网格；多维时 grid_x * grid_y * grid_z
        // n_elements 也要作为 int32 传给 kernel（与 kernel 形参宽度一致时）；
        // 超 INT32_MAX 同样静默截断，显式挡掉。
        if (n_elements <= 0 || n_elements > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            return ge::GRAPH_FAILED;
        }

        // 4) 加载 kernel 二进制（.npubin 场景；同库编译场景可省略加载步骤）
        //    aclrtBinaryLoadFromData(bin_data.data(), bin_data.size(), &opts, &bin_handle);
        //    aclrtBinaryGetFunction(bin_handle, kKernelName, &func_handle);
        //
        //    产物路径规则：不得依赖临时工作目录；
        //    需要 OM 往返时必须实现 PortableOp（见 my_compilable_portable_op.cpp）。

        // 5) 组装 args（顺序必须与 kernel 侧参数顺序逐一对应）
        //    FFTS 场景：首参数需为 ffts_addr（aclrtGetHardwareSyncAddr 获取）
        //
        //    三个必须注意的点：
        //    a. GetInputTensor 返回 const Tensor*，其 GetAddr() 返回 const void*，
        //       赋给 void* 成员会编译报错，必须 const_cast（输出是 Tensor*，无需转换）。
        //    b. 结构体必须 packed：否则编译器会插入填充，host 侧布局与 kernel 形参表错位。
        //    c. 每个槽的宽度必须与 kernel 形参类型**逐一一致**（此处按官方样例用 int32_t；
        //       若你的 kernel 形参是 int64_t，这里也要改，否则取参错位）。
        struct __attribute__((packed)) KernelArgs {
            void* x_addr;
            void* y_addr;
            void* z_addr;
            int32_t n_elements;
            int32_t grid_x;
            int32_t grid_y;
            int32_t grid_z;
        } args{const_cast<void*>(x->GetAddr()),
               const_cast<void*>(y->GetAddr()),
               z->GetAddr(),
               static_cast<int32_t>(n_elements),
               grid_x,
               1,
               1};

        // 6) 取流并 launch（必须用 GE 的 stream，不要自建默认流）
        void* stream = ctx->GetStream();
        if (stream == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // aclrtLaunchKernelWithHostArgs(func_handle, block_num, stream, ..., &args, sizeof(args));

        (void)stream;
        (void)args;
        std::fputs("GE skeleton: implement binary loading and kernel launch before execution\n", stderr);
        return ge::GRAPH_FAILED;
    }
};

// 注册：自动映射类名作为 op type，backend 默认为 kDevice
REG_AUTO_MAPPING_OP(MyEagerExecuteOp);
