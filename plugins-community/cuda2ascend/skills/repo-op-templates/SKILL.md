---
name: repo-op-templates
description: 算子代码模板库，提供代码模板与模板选择规则，作为算子代码开发的起点。触发：开始实现算子代码、搭建工程骨架前，先取模板复制到工作区，以此为起点开发。
---

# 算子代码模板库

实现算子代码前，先取工程模板复制到工作区，以此为起点开发，禁止从零创建工程文件。

本仓算子工程采用 direct launch 结构。模板分两层：**工程骨架**（build.sh / setup.py / CMakeLists.txt / cmake/ / python 包 / csrc/extension.cpp——所有算子共享，一次搭建）与**算子模板**（kernel.cpp / launch.h / plugin.cpp / CMakeLists.txt——每个算子复制一份填充）。两层模板见 references。

> 通用 Ascend C kernel 编写范式仍可参照共享 skill `ascendc-direct-invoke-template`；direct launch 工程结构与注册契约以本 skill 的 references 为准。
> 本目录是基类**默认模板实现**（属 virtual），各算子仓可 override 为本仓专用模板库；override 时保持 `name:` 与逻辑名 `repo-op-templates` 不变。

## 路由

| 场景 | 加载 |
|------|------|
| 搭建工程骨架（首次创建提交工程目录） | [references/project-skeleton.md](references/project-skeleton.md) |
| 新增算子（复制算子模板，填充 kernel/plugin/CMakeLists） | [references/operator-template.md](references/operator-template.md) |
