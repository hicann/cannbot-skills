---
name: repo-build-guide
description: 仓库代码结构与构建指南，介绍本仓算子代码的目录/文件结构与编译验证方法。触发：了解算子代码结构、搭建工程、编译与运行验证时加载。
---

# 仓库代码结构与构建指南

本 skill 帮助快速了解本仓算子工程的**代码结构**（目录与文件组织）与**编译验证方法**。

本仓算子工程采用 direct launch 结构：bisheng 编译 kernel、g++ 编译 plugin、合并为 `_C.abi3.so` 装入 wheel 包。结构与构建方法见 references。

## 路由

| 场景 | 加载 |
|------|------|
| 了解 direct launch 工程结构、编译配置、构建流程与验证程度 | [references/build-guide.md](references/build-guide.md) |
