---
name: repo-build-guide
description: 仓库代码结构与构建指南，介绍本仓算子代码的目录/文件结构与编译验证方法。触发：了解算子代码结构、搭建工程、编译与运行验证时加载。
---

# 仓库代码结构与构建指南

本 skill 帮助快速了解本仓各算子的**代码结构**（目录与文件组织），并给出**编译与验证方法**。

本仓的默认算子代码结构以共享 skill `ascendc-direct-invoke-template` 提供的工程模板为准；构建与验证的通用流程与验证程度见 references。

> 本文件只做路由。代码结构说明与构建指南在 `references/` 中按需加载，不在此内联。
> 本目录是基类**默认构建指南**（属 virtual），各算子仓可 override 为本仓专用结构与命令；override 时保持 `name:` 与逻辑名 `repo-build-guide` 不变。

## 路由

| 场景 | 加载 |
|------|------|
| 了解算子代码结构、编译与验证方法及验证程度 | [references/build-guide.md](references/build-guide.md) |
