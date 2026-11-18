# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 增量文档生成 — 设计说明

给定一次代码变更（PR 的 before/now 两个版本），只重写受影响 API 文档中受影响的章节，
其余内容保持原文不变。

## 流水线

```
PR(before/now) ──► 变更检测 ──► 证据定位 ──► 章节裁决 ──► 生成/校验 ──► 回填 ──► 导出
                     │            │            │            │          │
                受影响 API    头文件/实现/   哪些 doc_part  按章节写   写回原 md
                与文件清单    文档/示例      需要改        并做一致性校验
```

入口：`src/run_all_cli.py`；批处理入口：`run_by_pr_test.py`（按 PR 跑全链路）。
写作规范的唯一来源是 asc-devkit 仓 `docs/zh/asc_api_writing_guidelines.md`，
由 `src/generation_validation/spec_loader.py` 按章节读取，本仓不保留副本。

## 模块职责

| 模块 | 职责 |
|---|---|
| `src/change_detection/` | 锚点重定向、锚点重建、章节级变更判定 |
| `src/api_change_detect/` | 反向索引构建与增量变更检测（输出受影响 API） |
| `src/api_locator/` | 代码定位、文件重组、按章节提取证据 |
| `src/generation_validation/` | 提示词编排、证据提取、章节生成、一致性校验 |
| `src/md_fill/` | 把生成结果回填到 Markdown，并做确定性后处理 |
| `src/common/`、`src/llm/`、`src/logger.py` | 公共工具、模型客户端、日志 |

## 设计文档

- [API 代码定位与文件重组](DESIGN-api-locator.md)
- [API 变更检测](DESIGN-change-detection.md)
- [锚点重定向流水线](DESIGN-anchor-redirect.md)

> 面向用户的文档（写作规范、术语表、章节提取规则、使用说明）统一放在 docs 仓，
> 本目录只保留设计与代码。
