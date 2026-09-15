---
schema_version: okf.v1
kind: api
type: api_reference
source_family: asc_devkit
title: "数据搬运简介"
description: "数据搬运总览：支持连续与非连续搬运，排布格式可保持不变也可在 NZ Layout 与 ND Layout 之间转换，配两张示意图（每格 16*32Byte）说明两类典型情形。"
tags: [tensor_datamove]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/api/context/tensor_api/data_move/数据搬运简介.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/data_move/数据搬运简介.md

# 数据搬运简介

数据搬运功能支持连续与非连续搬运，搬运时数据排布格式可以保持不变，也可以实现NZ Layout和ND Layout之间的转换。

下文示例中，每格包含16 \* 32Byte的数据：

**图 1**  非连续搬运，数据排布格式不变（NZ Layout -\> NZ Layout）

**图 2**  连续搬运，数据排布格式改变（ND Layout -\> NZ Layout）
