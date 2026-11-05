/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SORT_TILING_KEY_DECL_H_
#define _SORT_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

// Host 的调度编号，保留已有编号；不是 TilingKey 低字节的编码值。
#define SORT_SCHID_0 0 // 归并排序
#define SORT_SCHID_1 1 // 单核 Radix
#define SORT_SCHID_2 2 // 多核 Radix
#define SORT_SCHID_5 5 // 短轴插入
#define SORT_SCHID_6 6 // 两阶段排序

// SDK 按声明顺序拼接三个 8 位字段；UINT 列表编码的是值在列表中的序号。
// schId={0,1,2,5,6} 对应 schIndex={0,1,2,3,4}，不能直接把 schId 当低字节。
// key = schIndex | (isInt32 << 8) | (isDescend << 16)；改动列表顺序会改变已有 key。
#define SORT_TPL_KEY_DECL() \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, SORT_SCHID_0, SORT_SCHID_1, SORT_SCHID_2, \
                          SORT_SCHID_5, SORT_SCHID_6), \
        ASCENDC_TPL_UINT_DECL(isInt32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_DECL(isDescend, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)

#define SORT_TPL_RADIX_MORE_CORE_KEY_SEL() \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_2), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)

#define SORT_TPL_MERGE_SORT_KEY_SEL() \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_0), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 1), \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)

#define SORT_TPL_RADIX_ONE_CORE_KEY_SEL() \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_1), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 1), \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)

#define SORT_TPL_SMALL_AXIS_INSERTION_KEY_SEL() \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_5), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)

#define SORT_TPL_SMALL_AXIS_TWO_STAGE_KEY_SEL() \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_6), \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1), \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)

ASCENDC_TPL_ARGS_DECL(Sort, SORT_TPL_KEY_DECL());

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(SORT_TPL_RADIX_MORE_CORE_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(SORT_TPL_MERGE_SORT_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(SORT_TPL_RADIX_ONE_CORE_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(SORT_TPL_SMALL_AXIS_INSERTION_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(SORT_TPL_SMALL_AXIS_TWO_STAGE_KEY_SEL()));
#endif
