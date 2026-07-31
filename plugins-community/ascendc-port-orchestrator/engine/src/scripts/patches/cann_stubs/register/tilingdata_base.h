// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// DEBT-110 Scope B: CANN host-side stub for port_a3_to_a5 build.
//
// Replaces CANN's `register/tilingdata_base.h` with a minimal header-only
// shim. The macros below expand to plain C++ class definitions so that
// op_host/<op>_tiling.h files written for CANN can compile against
// the standalone verifier include path WITHOUT pulling in the CANN host runtime.
//
// SCOPE: header-only. No registration semantics — REGISTER_TILING_DATA_CLASS
//        creates a no-op static instance; the factory class is declaration-only.
//        port_a3_to_a5 mode does not exercise the registration path.

#ifndef A5OPS_CANN_STUB_TILINGDATA_BASE_H_
#define A5OPS_CANN_STUB_TILINGDATA_BASE_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>

#define BEGIN_TILING_DATA_DEF(struct_name)         \
class struct_name : public TilingDef {

#define TILING_DATA_FIELD_DEF(type, field_name)    \
 public:                                           \
  void set_##field_name(type field_name) {         \
    this->field_name##_ = field_name;              \
  }                                                \
  type get_##field_name() const {                  \
    return this->field_name##_;                    \
  }                                                \
  type field_name##_ = 0;

#define TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)  \
  struct_type field_name;

#define END_TILING_DATA_DEF };

class TilingDef {};

#define REGISTER_TILING_DATA_CLASS(op_type, class_name)                      \
  static class_name g_##op_type##class_name##init;

#endif  // A5OPS_CANN_STUB_TILINGDATA_BASE_H_
