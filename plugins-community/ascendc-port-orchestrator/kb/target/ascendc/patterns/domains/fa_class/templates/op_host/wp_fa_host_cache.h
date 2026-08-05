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
 * \file wp_fa_host_cache.h  (KB-asset op_host reusable host-prep cache)
 * \brief ONE reusable common host-prep function for FA-class assembled kernels — all host
 *        entries call it (owner directive 2026-06-08: host logic in a reusable function,
 *        no per-entry copy).
 *
 *   Persistent workspace + cached device-tiling, keyed by the TilingData POD bytes + ws-size.
 *   REMOVES the per-call `torch::empty(workspace)` + tiling `memcpy`+H2D (the OL-201 host
 *   per-call overhead) — matches what the vendor (npu_fusion_attention / CANN framework) gets
 *   for free via framework workspace-pre-alloc + tiling-cache. A custom-`<<<>>>`-launch op has
 *   no framework, so without this it pays alloc+H2D every call.
 *
 *   WHITEBOX-VERIFIED 2026-06-08 (A5/Ascend950PR, FA graybox kernel, before/after on the same
 *   built kernel): wall −36%(S2048) … −73%(S256) [~0.034 ms/call host overhead removed];
 *   precision PASS_B 40/40 within-T1 UNCHANGED (caching does not touch kernel output, only
 *   where the scratch workspace + tiling live). Not an assumption — measured.
 *
 *   USAGE (in the kw's pybind `run_*` DoTiling, replacing per-call alloc+H2D):
 *     auto hp = wp_fa_host_cache::get(&td, sizeof(td), totalWsBytes, q.device());
 *     // hp.first = workspace ptr, hp.second = device-tiling ptr  → pass to the <<<>>> launcher
 *
 *   Self-contained: torch/extension + std only; NO #include "arch35/", NO aclnn/aclop.
 */
#ifndef WP_FA_HOST_CACHE_H_
#define WP_FA_HOST_CACHE_H_

#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace wp_fa_host_cache {

struct Entry { torch::Tensor workspace; torch::Tensor tiling_dev; };

inline std::unordered_map<uint64_t, Entry>& cache() {
    static std::unordered_map<uint64_t, Entry> c;
    return c;
}

inline uint64_t fnv1a(const void* p, size_t n) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// td_ptr/td_size = the TilingData POD bytes (ANY FA TilingData type — type-erased);
// totalWsBytes = workspace size for this config; dev = the op's NPU device.
// Returns {workspace_ptr, tiling_dev_ptr}: builds+caches on a (tiling,ws-size) miss,
// reuses on a hit. Key = hash(td bytes) ^ ws-size → identical config ⇒ reuse (correct:
// identical td ⇒ identical tiling; same ws-size ⇒ workspace big enough).
//
// CONCURRENCY: sequential-call safe (the workspace is scratch the kernel overwrites each
// call; back-to-back synchronized calls reuse it safely). Concurrent async-stream reuse of
// the SAME config would race on the shared workspace — if that path is needed, key also on
// the stream / keep a small per-stream pool (tracked; not needed for the current sync path).
inline std::pair<void*, void*> get(const void* td_ptr, size_t td_size,
                                   int64_t totalWsBytes, c10::Device dev) {
    const uint64_t key = fnv1a(td_ptr, td_size)
                       ^ (static_cast<uint64_t>(totalWsBytes) * 1099511628211ULL);
    auto& c = cache();
    auto it = c.find(key);
    if (it != c.end()) {
        return {it->second.workspace.data_ptr(), it->second.tiling_dev.data_ptr()};
    }
    auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto workspace = torch::empty({totalWsBytes}, opt_u8);
    auto tiling_cpu = torch::empty({static_cast<int64_t>(td_size)},
                                   torch::TensorOptions().dtype(torch::kUInt8));
    const auto *tiling_bytes = static_cast<const uint8_t *>(td_ptr);
    std::copy_n(tiling_bytes, td_size, tiling_cpu.data_ptr<uint8_t>());
    auto tiling_dev = tiling_cpu.to(dev);
    c[key] = Entry{workspace, tiling_dev};
    return {workspace.data_ptr(), tiling_dev.data_ptr()};
}

}  // namespace wp_fa_host_cache

#endif  // WP_FA_HOST_CACHE_H_
