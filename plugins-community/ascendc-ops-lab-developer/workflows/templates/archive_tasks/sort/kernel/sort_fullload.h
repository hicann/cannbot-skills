/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_FULLLOAD_H
#define SORT_FULLLOAD_H

#include "sort_one_core.h"

namespace KvSort {
using namespace AscendC;

class SortFullLoad : public SortOneCore
{
public:
    __aicore__ inline SortFullLoad() {};
    __aicore__ inline void Process();
};

__aicore__ inline void SortFullLoad::Process()
{
    if (GetBlockIdx() == 0) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
}

}  // namespace KvSort
#endif
