// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

const BRAND_NAME = "CANNBot-Insight"
const BRAND_SLUG = BRAND_NAME.toLowerCase().replace(/[^a-z0-9]+/g, "-")
const BRAND_DESCRIPTION = `${BRAND_NAME} — LLM Agent Session Observability`
const BRAND_CONFIG_DIR_SUFFIX = `.${BRAND_SLUG}`
const BRAND_CLI_ALIAS = "cbi"

// Data format identifier — always "cannbot-insight" regardless of branding
const BRAND_SOURCE_TYPE = "cannbot-insight"

export { BRAND_NAME, BRAND_SLUG, BRAND_DESCRIPTION, BRAND_CONFIG_DIR_SUFFIX, BRAND_CLI_ALIAS, BRAND_SOURCE_TYPE }
