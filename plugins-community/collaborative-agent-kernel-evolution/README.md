# CAKE2

CAKE2 (Collaborative Agent Kernel Evolution 2) 是一个用于昇腾（Ascend）NPU 算子开发的进化式Agent系统。它提供了一套完整的流程，从算子描述生成到最终的AscendC代码实现和评估，并支持通过进化算法自动优化内核性能。

## 快速开始

### 安装 Plugin

#### 方式1: 从 Marketplace 安装

```bash
# 在 Claude Code 中注册 cannbot marketplace（CAKE2 作为社区插件 cake 合入其中）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
# 安装 cake 插件
/plugin install cake@cannbot

# 安装后直接以 agent 启动
claude --agent cake:cake
claude --agent cake:cake-evo
```

Marketplace 是 Claude Code 的插件分发机制。CAKE2 作为 `cake` 插件登记在 cannbot 仓库根的 `.claude-plugin/marketplace.json` 中，使用 `/plugin marketplace add` 注册该 marketplace 后即可安装。

#### 方式2: 开发模式加载

```bash
# 从本地路径临时加载（不写入配置）
claude --plugin-dir /path/to/cake2

# 直接以特定 agent 启动
claude --plugin-dir /path/to/cake2 --agent cake:cake
claude --plugin-dir /path/to/cake2 --agent cake:cake-evo
```

### 使用 Agent

```bash
# 可用 agent：
#   cake         - 标准算子生成
#   cake-evo     - 进化式多变体优化
#   cake-partial - 部分流程生成（DSL → 评估）

# 命令行直接指定 agent 启动
claude --agent cake

# 或进入 Claude Code 后：
#   输入 /agents 选择 agent
#   或直接在对话中请求，如 "用 cake agent 生成一个 Softmax 算子"
```


### 示例用法
```plain
读取 docs/SinkhornKnopp 中的算子描述， 生成 SinkhornKnoppCustom 算子。

读取 docs/MoeInitRouting 中的算子描述， 进化生成 MoeInitRoutingCustom 算子，最大轮数 2，每轮并行数 2，目标加速比 2x 相对于 PyTorch 小算子 eager 实现。
```


## 📚 文档导航

- **[项目概览](docs/PROJECT_OVERVIEW.md)** - 系统架构和设计理念
- **[快速开始](docs/GETTING_STARTED.md)** - 环境配置和首次使用
- **[进化优化](docs/EVO.md)** - 进化式内核优化流程
- **[返回码说明](docs/RETURN_CODE_EXPLANATIONS.md)** - 编译和评估返回码

## 简介

本项目旨在简化昇腾NPU自定义算子的开发流程，通过自动化的方式将算子描述转换为可在昇腾NPU上运行的高性能算子实现。

主要功能包括：
- 算子描述生成（JSON格式）
- PyTorch参考代码生成
- 函数式API转换
- AscendDSL代码生成与优化
- AscendC算子代码生成
- AscendC项目创建与编译
- 算子正确性与性能评估
- 进化式多变体并行优化

## 环境准备

### 方式1: 本地 CANN 环境

需要完整的 CANN + Ascend NPU 环境(无 CPU-only 模式,不支持 macOS/Windows):

| 组件 | 要求 | 说明 |
|------|------|------|
| Ascend NPU | 910B,已装 driver(`npu-smi` 可用) | 编译 / 精度 / 性能评测必需 |
| CANN | >= 8.3.RC1(在 8.5.0 上验证) | 提供 `msopgen` / `msprof` / `acl` 等工具链 |
| Python | >= 3.10 | |
| PyTorch + torch_npu | 二者及 CANN 三者版本需匹配(验证组合:torch 2.9.0 + torch_npu 2.9.0 + CANN 8.5.0) | torch_npu 由昇腾环境提供,需按 CANN 版本选对应 wheel,非 PyPI 通用分发 |
| Node.js + Claude Code | Node 20+,`npm i -g @anthropic-ai/claude-code` | 运行本插件的宿主 |
| tilelang-ascend / torchair | (可选,源码构建) | 仅 TileLang DSL / `torch.compile` 图模式需要 |

创建隔离的 Python 虚拟环境(conda/uv/venv)后,安装纯 Python 依赖:

```sh
pip install -r requirements.txt
# 再安装与你的 CANN 版本匹配的 torch / torch_npu(见上表)
```

**注意：** 该工具在使用过程中会安装`custom-ops.whl`进行测试, 多人环境下会造成冲突，导致算子生成失败。因此必须使用隔离的python环境（例如，使用`conda`创建）。

### 方式2: 远程开发模式

本地没有 NPU 硬件时，可以通过 SSH 连接远程 NPU 服务器进行编译和测试：

```bash
# 1. 配置远程连接
cp ${CLAUDE_SKILL_DIR}/scripts/npus.example.yaml .npus.yaml
# 编辑 .npus.yaml 配置 remotes（docker / hdspace 后端均支持）

# 2. 探测远程平台信息
uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py exec <target> info
```

Agent 启动时会自动检测环境模式（本地/远程），详见 `skills/remote-cann-development/` 。

## 用法

### 作为 Plugin 使用（推荐）

```bash
claude --plugin-dir /path/to/cake2
```

打开 Claude Code 后输入 `/agents` 选择 `cake` agent，然后描述算子需求即可。

### 基于二进制

```sh
./cake.run
```

### 输出

会在当前目录下生成`output/{op_name}`文件夹，生成产物如下：

```
output/
└── {op_name}/
    ├── {op_name}_op_desc.json      # 算子描述
    ├── {op_name}_reference.py      # PyTorch参考实现
    ├── {op_name}_functional.py     # 函数式API实现
    ├── {op_name}_dsl.py            # DSL基线代码
    ├── {op_name}_project.json      # 项目配置
    ├── {op_name}.cpp               # Python绑定代码
    └── {op_name}Custom/            # AscendC项目目录
```

### 支持的算子类型

- **池化算子** (pooling): average_pooling2d 等
- **归约算子** (reduction): sum, mean 等
- **损失函数** (loss): mse_loss 等
- **归一化** (normalization): layer_norm 等
- **激活函数** (activation)
- **卷积** (convolution)
- **矩阵乘法** (matmul)
- **数学运算** (math)
