---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: asc_devkit
title: "概述"
description: "高阶API 概述：基于单核对常见算法做抽象封装（数学计算、矩阵计算、激活函数等），内部由多种基础API 组合实现，免去格式转换与数据切分等细节以提高开发效率。"
tags: [api_overview]
resource: https://gitcode.com/cann/asc-devkit/blob/792bc49f7ea06312bdb8964d22b8753e6a5cea30/docs/guide/编程指南/类库API/高阶API/概述.md
created_at: 2026-08-05T14:23:52Z
updated_at: 2026-09-04T23:59:25Z
---
> **原始文档路径**: `asc-devkit/docs/guide/编程指南/类库API/高阶API/概述.md`
# 概述<a name="ZH-CN_TOPIC_0000002490820632"></a>

高阶API基于单核对常见算法进行抽象和封装，实现了一些常用的计算算法，旨在提高编程开发效率。高阶API一般通过调用多种基础API实现。高阶API包括数学计算、矩阵计算、激活函数等API。

如下图所示，实现一个矩阵乘操作，使用基础API需要的步骤较多，需要关注格式转换、数据切分等逻辑；使用高阶API则无需关注这些逻辑，可以快速实现功能。
