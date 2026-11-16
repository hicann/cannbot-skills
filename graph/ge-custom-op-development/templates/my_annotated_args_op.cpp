/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 静态声明式待适配骨架：需填入真实 bin、ABI 并合并推导接口。
// 接口和选择条件见 references/interfaces.md、scenarios.md。
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/custom_op.h"
#include "exe_graph/runtime/annotated_args_context.h"
#include "exe_graph/runtime/op_compile_context.h"
#include "graph/types.h"

namespace {
constexpr const char* kKernelName = "my_kernel";
constexpr int32_t kBlockSize = 1024;
constexpr uint32_t kMaxBlockDim = 65535U; // 与官方 CalcBlockDim 的越界判据一致
} // namespace

class MyAnnotatedArgsOp : public ge::CompilableOp, public ge::AnnotatedArgsOp {
public:
    // 编译期：按输入 shape 准备 kernel 二进制与 block_dim
    ge::graphStatus Compile(gert::OpCompileContext* ctx) override
    {
        const auto* x = ctx->GetInputTensor(0);
        if (x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int64_t n_elements = x->GetShapeSize();
        if (n_elements <= 0 || n_elements > static_cast<int64_t>(kMaxBlockDim) * kBlockSize) {
            return ge::GRAPH_FAILED;
        }
        const auto* key_y = ctx->GetInputTensor(1);
        if (key_y == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const std::string key = BuildKey(*x, *key_y);
        const int64_t grid = (n_elements + kBlockSize - 1) / kBlockSize;

        std::lock_guard<std::mutex> lock(mu_);
        Artifact artifact;
        artifact.kernel_bin = std::make_shared<std::vector<uint8_t>>(); // TODO: 填入真实 kernel 二进制（.npubin 等）
        // 元素数已检查；
        // 上限 65535（与官方 CalcBlockDim 一致），越界直接失败而不是静默截断
        if (grid > static_cast<int64_t>(kMaxBlockDim)) {
            return ge::GRAPH_FAILED;
        }
        artifact.block_dim = static_cast<uint32_t>(grid > 0 ? grid : 1);
        if (n_elements <= 0 || artifact.kernel_bin->empty()) {
            return ge::GRAPH_FAILED;
        }
        artifacts_[key] = std::move(artifact);
        return ge::GRAPH_SUCCESS;
    }

    // 声明式：声明 launch 参数，GE 生成任务并完成地址刷新
    ge::graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext& ctx) override
    {
        // 按当前节点的输入 shape 取对应编译产物：
        // 多 shape 场景下不能共用单槽成员，否则会用到别的 shape 的 bin/block_dim。
        const auto* x = ctx.GetInputTensor(0);
        if (x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto* key_y = ctx.GetInputTensor(1);
        if (key_y == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const std::string key = BuildKey(*x, *key_y);

        uint32_t block_dim = 0U;
        std::shared_ptr<std::vector<uint8_t>> bin;
        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto it = artifacts_.find(key);
            if (it == artifacts_.end()) {
                return ge::GRAPH_FAILED; // 该 shape 未编译
            }
            block_dim = it->second.block_dim;
            bin = it->second.kernel_bin; // shared_ptr 拷贝，O(1)
        }
        if (bin == nullptr || bin->empty()) {
            return ge::GRAPH_FAILED;
        }

        const auto* y = ctx.GetInputTensor(1);
        const auto* z = ctx.GetOutputTensor(0);
        if (y == nullptr || z == nullptr) {
            return ge::GRAPH_FAILED;
        }

        gert::AnnotatedKernelLaunchInfo launch_info;
        launch_info.kernel_name = kKernelName;
        launch_info.kernel_bin = bin->data();
        // kernel_bin_size 必填：漏设时默认 0U，运行期必然失败（官方样例 :142 都赋值）
        launch_info.kernel_bin_size = bin->size();
        launch_info.block_dim = block_dim;
        launch_info.stream_id = ctx.GetStreamId();
        if (launch_info.stream_id == UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }

        // args 顺序 == kernel 侧参数顺序（务必与 kernel 定义一致）
        // InputAddr/OutputAddr 必须同时给 index 与 addr；只给 index 时 addr 为 nullptr。
        gert::AnnotatedKernelArgs args;
        if (args.AppendArg(gert::InputAddr{0U, x->GetAddr()}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        } // x
        if (args.AppendArg(gert::InputAddr{1U, y->GetAddr()}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        } // y
        if (args.AppendArg(gert::OutputAddr{0U, z->GetAddr()}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        } // z
        // 有 workspace 时：AnnotatedArgsContext::MallocWorkSpace(size_t) 直接返回 WorkspaceAddr
        // （注意与 EagerOpExecutionContext::MallocWorkSpace 不同，后者返回 void*）：
        //   gert::WorkspaceAddr ws = ctx.MallocWorkSpace(ws_size);
        //   args.AppendArg(ws);

        // 2 参 AddLaunch 返回 ge::graphStatus（annotated_args_context.h:252），不是 launch token；
        // 只有带 predecessors 的 3 参重载（@since 9.3.0，:265-266）才返回 AnnotatedLaunchToken。
        // 误当 token 判 UINT32_MAX 会静默吞错：GRAPH_SUCCESS=0、失败为非 0 错误码，
        // 永不等于 UINT32_MAX，且 unscoped enum → uint32_t 隐式转换能编译通过。
        // 官方写法是直接透传该状态码（annotated_args_refresh_add_custom/online/cpp/ge/custom_op.cpp:310-315）。
        return ctx.AddLaunch(launch_info, std::move(args));
    }

private:
    struct Artifact {
        uint32_t block_dim{0U};
        std::shared_ptr<std::vector<uint8_t>> kernel_bin;
    };

    // 固定Add骨架无attrs；扩展特化时在此加入所有属性、平台与编译条件。
    static std::string BuildKey(const gert::Tensor& x, const gert::Tensor& y)
    {
        std::string key;
        for (const auto* tensor : {&x, &y}) {
            key += std::to_string(static_cast<int>(tensor->GetDataType())) + ":";
            key += std::to_string(static_cast<int>(tensor->GetOriginFormat())) + ":";
            key += std::to_string(static_cast<int>(tensor->GetStorageFormat())) + ":";
            const auto& expand = tensor->GetFormat().GetExpandDimsType();
            key += std::to_string(expand.GetFullSize()) + ":";
            for (size_t i = 0U; i < expand.GetFullSize(); ++i) {
                key += expand.IsExpandIndex(i) ? '1' : '0';
            }
            key += ":";
            for (const auto* shape : {&tensor->GetOriginShape(), &tensor->GetStorageShape()}) {
                key += "[";
                for (size_t i = 0U; i < shape->GetDimNum(); ++i) {
                    key += std::to_string(shape->GetDim(i)) + ",";
                }
                key += "]";
            }
            key += ";";
        }
        return key;
    }

    std::mutex mu_;
    std::unordered_map<std::string, Artifact> artifacts_;
};

REG_AUTO_MAPPING_OP(MyAnnotatedArgsOp);

/**
 * 多次 launch 示例（带依赖，仅主 stream）：
 *
 *   // 要拿 token 必须走 3 参重载（首个 launch 传空 predecessors）；
 *   // 2 参重载返回 graphStatus，误用会把错误码当依赖传下去（auto 会静默推导错类型）
 *   auto t0 = ctx.AddLaunch(info0, std::move(args0), {});
 *   auto t1 = ctx.AddLaunch(info1, std::move(args1), {t0});   // t1 依赖 t0
 *
 * 规则：token 必须来自本上下文先前 AddLaunch（3 参重载）的返回值；
 *       不可重复使用同一个 token；不可前向依赖（依赖必须已产生）。
 */
