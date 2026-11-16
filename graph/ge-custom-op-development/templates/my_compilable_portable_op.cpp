/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Compile + Portable 待适配骨架：需实现编译、launch 与完整状态校验。
// 接口和选择条件见 references/interfaces.md、scenarios.md。
#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/custom_op.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/op_compile_context.h"
#include "graph/types.h"

namespace {
constexpr int32_t kBlockSize = 1024;
constexpr uint32_t kMaxBlockDim = 65535U;

// 序列化格式的 magic（GE 不解析，只保证原样往返）
constexpr uint32_t kMagic = 0x4D594F50U; // 'MYOP'
} // namespace

class MyCompilablePortableOp : public ge::EagerExecuteOp, public ge::CompilableOp, public ge::PortableOp {
public:
    // (1) 编译期：编译 / 准备 kernel，产物按 shape 缓存
    ge::graphStatus Compile(gert::OpCompileContext* ctx) override
    {
        const auto* x = ctx->GetInputTensor(0);
        if (x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto* key_y = ctx->GetInputTensor(1);
        if (key_y == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const std::string key = BuildKey(*x, *key_y);

        std::lock_guard<std::mutex> lock(mu_);

        // TODO: 在此编译 kernel（使用目标工具链的 API 或外部编译进程）
        //       产物写入 Artifact{block_dim, bin}，不要依赖临时工作目录的路径。
        const int64_t n_elements = x->GetShapeSize();
        if (n_elements <= 0 || n_elements > static_cast<int64_t>(kMaxBlockDim) * kBlockSize) {
            return ge::GRAPH_FAILED;
        }
        const int64_t grid = (n_elements + kBlockSize - 1) / kBlockSize;
        Artifact artifact;
        // 元素数已检查；上限按该骨架约束。
        if (grid > static_cast<int64_t>(kMaxBlockDim)) {
            return ge::GRAPH_FAILED;
        }
        artifact.block_dim = static_cast<uint32_t>(grid > 0 ? grid : 1);
        artifact.bin = std::make_shared<std::vector<uint8_t>>(); // 填入真实 kernel 二进制
        if (n_elements <= 0 || artifact.bin->empty()) {
            return ge::GRAPH_FAILED;
        }
        artifacts_[key] = std::move(artifact);
        return ge::GRAPH_SUCCESS;
    }

    // (2) 序列化：自定义格式，随 OM 落盘
    ge::graphStatus Serialize(std::vector<uint8_t>& buffer) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        buffer.clear();
        if (artifacts_.empty() || artifacts_.size() > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        for (const auto& kv : artifacts_) {
            if (kv.second.bin == nullptr || kv.second.bin->empty() || kv.second.bin->size() > UINT32_MAX ||
                kv.first.size() > UINT32_MAX || kv.second.block_dim == 0U || kv.second.block_dim > kMaxBlockDim) {
                return ge::GRAPH_FAILED;
            }
        }
        WriteU32(buffer, kMagic);
        WriteU32(buffer, static_cast<uint32_t>(artifacts_.size()));
        for (const auto& kv : artifacts_) {
            WriteU32(buffer, static_cast<uint32_t>(kv.first.size()));
            buffer.insert(buffer.end(), kv.first.begin(), kv.first.end());
            const auto& bin = kv.second.bin;
            const uint32_t bin_size = (bin == nullptr) ? 0U : static_cast<uint32_t>(bin->size());
            WriteU32(buffer, kv.second.block_dim);
            WriteU32(buffer, bin_size);
            if (bin_size != 0U) {
                buffer.insert(buffer.end(), bin->begin(), bin->end());
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    // (3) 反序列化：OM 加载时恢复
    ge::graphStatus Deserialize(const std::vector<uint8_t>& buffer) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        std::unordered_map<std::string, Artifact> restored;
        size_t off = 0U;
        uint32_t magic = 0U;
        if (!ReadU32(buffer, off, magic) || magic != kMagic) {
            return ge::GRAPH_FAILED;
        }
        uint32_t count = 0U;
        if (!ReadU32(buffer, off, count) || count == 0U || count > (buffer.size() - off) / 12U) {
            return ge::GRAPH_FAILED;
        }
        for (uint32_t i = 0U; i < count; ++i) {
            uint32_t key_len = 0U;
            if (!ReadU32(buffer, off, key_len) || key_len == 0U || key_len > buffer.size() - off) {
                return ge::GRAPH_FAILED;
            }
            const std::string key(buffer.begin() + static_cast<long>(off),
                                  buffer.begin() + static_cast<long>(off + key_len));
            off += key_len;

            Artifact artifact;
            uint32_t bin_size = 0U;
            if (!ReadU32(buffer, off, artifact.block_dim) || !ReadU32(buffer, off, bin_size) || bin_size == 0U ||
                bin_size > buffer.size() - off || artifact.block_dim == 0U || artifact.block_dim > kMaxBlockDim) {
                return ge::GRAPH_FAILED;
            }
            artifact.bin = std::make_shared<std::vector<uint8_t>>(buffer.begin() + static_cast<long>(off),
                                                                  buffer.begin() + static_cast<long>(off + bin_size));
            off += bin_size;
            if (!restored.emplace(key, std::move(artifact)).second) {
                return ge::GRAPH_FAILED;
            }
        }
        if (off != buffer.size()) {
            return ge::GRAPH_FAILED;
        }
        // 验证完再合并，失败payload不破坏其他节点/模型已有状态。
        for (const auto& kv : restored) {
            const auto existing = artifacts_.find(kv.first);
            if (existing != artifacts_.end() &&
                (existing->second.block_dim != kv.second.block_dim || existing->second.bin == nullptr ||
                 *existing->second.bin != *kv.second.bin)) {
                return ge::GRAPH_FAILED; // 同key不同产物需扩展版本key或隔离实例。
            }
        }
        artifacts_.insert(restored.begin(), restored.end());
        return ge::GRAPH_SUCCESS;
    }

    // (4) 运行时：取已编译产物并 launch
    ge::graphStatus Execute(gert::EagerOpExecutionContext* ctx) override
    {
        auto* x = ctx->GetInputTensor(0);
        if (x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto* z = ctx->MallocOutputTensor(0, x->GetShape(), x->GetFormat(), x->GetDataType());
        if (z == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto* key_y = ctx->GetInputTensor(1);
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
                return ge::GRAPH_FAILED; // 编译产物缺失
            }
            block_dim = it->second.block_dim;
            bin = it->second.bin; // shared_ptr 拷贝，O(1)，不在锁内深拷贝 bin
        }

        void* stream = ctx->GetStream();
        if (stream == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // 使用 bin 加载并 launch（block_num = block_dim）
        (void)stream;
        (void)block_dim;
        std::fputs("GE skeleton: implement binary loading and kernel launch before execution\n", stderr);
        return ge::GRAPH_FAILED;
    }

private:
    struct Artifact {
        uint32_t block_dim{0U};
        // 用 shared_ptr：Execute 里只需要 O(1) 拷贝句柄，避免在锁内深拷贝整个 bin
        std::shared_ptr<std::vector<uint8_t>> bin;
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

    static void WriteU32(std::vector<uint8_t>& buf, uint32_t v)
    {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFU));
        }
    }
    static bool ReadU32(const std::vector<uint8_t>& buf, size_t& off, uint32_t& v)
    {
        if (off + 4U > buf.size()) {
            return false;
        }
        v = 0U;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(buf[off + static_cast<size_t>(i)]) << (8 * i);
        }
        off += 4U;
        return true;
    }

    std::mutex mu_;
    std::unordered_map<std::string, Artifact> artifacts_;
};

REG_AUTO_MAPPING_OP(MyCompilablePortableOp);
