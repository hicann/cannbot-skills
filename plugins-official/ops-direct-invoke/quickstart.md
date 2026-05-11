# CANNBot 算子直调开发快速入门指南

## 概述

CANNBot 算子直调开发模式适用于**快速验证自定义算子**场景，通过 Ascend C API 直接开发和验证算子 Kernel，无需构建完整的 aclnn 接口层，适合原型开发、算子验证、学习研究等场景。

### 与算子仓开发的区别

| 对比维度 | 算子直调（本模式） | 算子仓开发 |
|---------|------------------|-----------|
| 适用场景 | 快速验证、原型开发、学习研究 | 生产级算子、算子仓贡献 |
| 开发内容 | Kernel + Tiling + Host 验证代码 | 完整 aclnn 接口 + Kernel + 测试用例 |
| 目录结构 | 独立工程目录 | 算子仓标准目录结构 |
| 验证方式 | Host 端直接调用 Kernel | aclnn API 调用 |
| 开发周期 | 短（小时级） | 长（天级） |

## 一、环境搭建

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install ops-direct-invoke@cannbot
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-direct-invoke
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode

**首选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-direct-invoke
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### Trae

仅支持项目级安装。

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-direct-invoke
bash init.sh project trae
```

安装后在项目根目录生成 `.trae/` 目录，结构与 Claude/OpenCode 基本一致。

### Cursor

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/ops-direct-invoke
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 ops-direct-invoke@cannbot ✔ enabled

# OpenCode
opencode agent list
# 应看到 ascendc-kernel-architect / ascendc-kernel-developer / ascendc-kernel-reviewer

# Trae
ls .trae/
# 应看到 skills/ agents/ AGENTS.md cannbot-manifest.json

# Cursor
ls .cursor/
# 应看到 skills/ agents/ AGENTS.md cannbot-manifest.json
```

## 二、快速上手

### 启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

> **Trae 用户**：Trae 通过 IDE 启动，`.trae/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。

### 开发算子示例

在交互界面中输入算子开发需求，会自动加载工作流技能并指导开发：

```
帮我开发一个 abs 算子，支持 float16 数据类型，shape 主要是 [1,128]、[4,2048]、[32,4096]
```

### 核心工作流

采用标准工作流，确保算子开发质量：

```
阶段0：环境检查 → 阶段1：需求与设计 → 阶段2：Kernel开发 → 阶段3：验证与优化
```

每个阶段完成后才能进入下一阶段，详见 AGENTS.md。

### 产出物示例

算子直调模式下，CANNBot 会在 `operators/{operator}/` 目录下生成以下文件：

```
operators/add_custom/
├── CMakeLists.txt           # 编译配置
├── build.sh                 # 编译脚本
├── run.sh                   # 运行脚本
├── op_kernel/
│   └── add_custom.cpp       # Kernel 实现
├── op_host/
│   ├── add_custom_tiling.h  # Tiling 数据结构
│   └── add_custom.cpp       # Host 端代码
└── gen_data.py              # 测试数据生成脚本
```

## 三、可用技能

| Skill | 用途 | 触发时机 |
|-------|------|---------|
| `ascendc-kernel-develop-workflow` | 完整开发工作流程 | **强制：所有算子开发任务** |
| `ascendc-docs-search` | 文档资源索引 | 查找 API 文档和示例 |
| `ascendc-api-best-practices` | API 使用最佳实践 | 调用任何 AscendC API 前 |
| `ascendc-tiling-design` | Tiling 设计 | 多核切分、UB切分、Buffer规划 |
| `ascendc-npu-arch` | NPU 架构知识 | 查询芯片特性 |
| `ascendc-precision-debug` | 精度调试 | 算子精度问题 |
| `ascendc-runtime-debug` | 运行时调试 | aclnn 错误、超时 |
| `ascendc-env-check` | 环境检查 | NPU 设备查询 |

## 四、开发资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | `asc-devkit/docs/api/context/` | 约 1022 个 API 文档 |
| 高性能模板 | `asc-devkit/examples/00_introduction/01_add/basic_api_memory_allocator_add/` | 双缓冲+流水线 |
| 各类示例 | `asc-devkit/examples/00_introduction/` | 加法、减法、多输入等 |
| 调试示例 | `asc-devkit/examples/01_utilities/00_printf/printf.asc` | printf 调试方法 |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
# Claude Code
/plugin update ops-direct-invoke@cannbot

# OpenCode (init.sh 方式)
cd cannbot-skills/plugins-official/ops-direct-invoke && bash init.sh

# Trae
cd cannbot-skills/plugins-official/ops-direct-invoke && bash init.sh project trae

# Cursor
cd cannbot-skills/plugins-official/ops-direct-invoke && bash init.sh project cursor
```

### Q: 算子直调模式和算子仓模式如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 快速验证算法思路 | 算子直调 |
| 学习 Ascend C 编程 | 算子直调 |
| 原型开发和概念验证 | 算子直调 |
| 生产级算子开发 | 算子仓开发 |
| 向算子仓贡献代码 | 算子仓开发 |
| 需要完整测试用例 | 算子仓开发 |

---

## 总结

1. 算子直调模式适合快速验证和学习，开发周期短
2. Claude Code 用户用 `/plugin install` 一键安装，OpenCode/Trae/Cursor 用户用 `init.sh` 脚本安装
3. 所有算子开发任务会自动加载工作流技能，按阶段执行
4. 产出物可直接编译运行，快速验证算子功能
