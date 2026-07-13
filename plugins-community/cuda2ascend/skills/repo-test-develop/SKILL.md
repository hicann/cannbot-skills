---
name: repo-test-develop
description: 仓库测试开发指导，介绍本仓测试框架的使用与测试代码的开发方法。触发：实现 golden、编写分级功能用例、补全白盒测试、搭建性能采集框架、执行精度测试时加载。
---

# 仓库测试开发指导

本 skill 介绍本仓测试框架的使用与测试代码的开发方法——golden 实现、分级功能用例、白盒测试补全、性能采集框架的搭建，以及精度测试的执行标准。

> 本文件只做路由。测试框架与精度/性能细则在 `references/` 中按需加载，不在此内联。
> 本目录是基类**默认测试开发指导**（属 virtual），各算子仓可 override 为本仓专用框架；override 时保持 `name:` 与逻辑名 `repo-test-develop` 不变。

## 按场景路由

| 场景 | 加载 |
|------|------|
| 搭建测试工程（golden、gen_data、run、分级用例、白盒补全） | [references/test-framework.md](references/test-framework.md) |
| 精度测试标准与性能采集方法 | [references/precision-and-perf.md](references/precision-and-perf.md) |

## 依赖的共享 skill（逻辑名）

| 逻辑名 | 用途 |
|--------|------|
| `ascendc-st-design` | 测试配置（test_matrix）管理，分级用例的 shape/dtype 组织 |
| `ops-precision-standard` | 各 dtype 的 atol/rtol 精度标准 |
| `ops-profiling` | msprof op 性能采集、CSV 指标解读、达标判定 |
| `ascendc-precision-debug` | 精度不达标时诊断根因 |
