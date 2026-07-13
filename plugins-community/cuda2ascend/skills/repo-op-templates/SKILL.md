---
name: repo-op-templates
description: 算子代码模板库，提供代码模板与模板选择规则，作为算子代码开发的起点。触发：开始实现算子代码、搭建工程骨架前，先取模板复制到工作区，以此为起点开发。
---

# 算子代码模板库

实现算子代码前，先取工程模板复制到工作区，以此为起点开发，禁止从零创建工程文件。

本仓的算子代码模板与模板选择规则以共享 skill `ascendc-direct-invoke-template` 为准：**加载 `ascendc-direct-invoke-template`**，按其提供的模板与选择规则复制对应工程模板。

> 本目录是基类**默认模板实现**（属 virtual），各算子仓可 override 为本仓专用模板库；override 时保持 `name:` 与逻辑名 `repo-op-templates` 不变。
