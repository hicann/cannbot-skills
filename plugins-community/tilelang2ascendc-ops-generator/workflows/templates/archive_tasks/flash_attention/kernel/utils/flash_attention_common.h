/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_COMMON_H
#define FLASH_ATTENTION_COMMON_H

#include <cstdint>

inline uint32_t AlignUpHost(uint32_t num, uint32_t align)
{
    if (align == 0) { return num; }
    return (num + align - 1) / align * align;
}

inline uint32_t CeilDivHost(uint32_t a, uint32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

#endif // FLASH_ATTENTION_COMMON_H
