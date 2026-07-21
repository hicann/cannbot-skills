# CANNBot Agent 开发平台 - 工程师指南

## 核心职责

作为 **Agent 开发工程师**，从 **Agent 架构**视角出发，构建专业、高效、可扩展的 CANNBot 智能体生态：

1. **Skills 开发与优化** - 创建可复用的技能模块，为 Agent 提供专业能力支持
2. **Agents 创建** - 设计专业化子代理，实现职责分工和模块化开发
3. **业务工作流** - 构建多 Agent 协作配置，编排完整的开发流程和业务场景
4. **效果评测** - 建立评测体系，**横向贯穿**所有层级，持续优化 Agent 和 Skills 的实际效果

## 核心原则

开发 Skills、Agents、Plugins 时遵循以下原则：

### 1. 信息来源可信

技术信息必须来自可信源，禁止编造：
- **可信来源**：CANN 官方文档、[社区代码仓](https://gitcode.com/cann)、CANN 安装路径下的文件、用户明确提供
- **禁止行为**：编造 API/参数、推测未验证的行为
- **不确定时**：明确标注，引导用户验证

### 2. 渐进式披露

模块设计分层，按需展开：
- SKILL.md/AGENT.md 只写核心内容
- 详细参考放 references/ 目录
- 用户需要时再深入

### 3. 简洁精炼

输出直击要点：
- 先给结论，再说原因
- 使用列表、表格等结构化表达
- 一个段落只讲一个要点

## 架构设计

```
CANNBot 智能体生态架构（自底向上）
┌─────────────────────────────────────────┐
│        Plugins（应用层）                │  编排多个 Agent 协同工作
│    plugins-*/*/ 目录                    │
├─────────────────────────────────────────┤
│        Agents（角色层）                 │  定义做什么（职责范围）
│    plugins-*/*/agents/ 目录             │
├─────────────────────────────────────────┤
│        Skills（能力层）                 │  定义怎么做（具体实现）
│    {领域}/ 目录下的各 Skill 子目录       │
├─────────────────────────────────────────┤
│        References（知识层）             │  定义如何做得更好
│    内嵌在 Skills 中的最佳实践           │
├─────────────────────────────────────────┤
│    Infrastructure（基础设施层）         │  提供底层工具和文档支持
│    infra/ 目录下的工具类 Skill          │
└─────────────────────────────────────────┘
     ↑ 效果评测横向覆盖所有层级 ↑
```

> 注：领域指业务领域目录，如 `ops/`（算子）、`model/`（模型）、`graph/`（图编译）、`runtime/`（运行时）等

## 项目结构

```
cannbot-skills/
├── ops/                      # 算子 Skills（正式版），每个子目录即一个 Skill
│   ├── ascendc-api-best-practices/
│   ├── ascendc-tiling-design/
│   ├── ops-simulator/
│   └── ...                   # 各 Skill 目录
├── model/                    # 模型推理优化 Skills，每个子目录即一个 Skill
│   ├── model-infer-fusion/
│   ├── model-infer-quantization/
│   └── ...                   # 各 Skill 目录
├── graph/                    # 图编译 Skills
├── runtime/                  # 运行时迁移 Skills
│   └── runtime_migration/
├── infra/                    # 基础设施 Skills（跨领域工具）
│   ├── gitcode-toolkit/
│   └── ...
├── plugins-official/         # 官方 Plugin（Plugin 配置 + Agents + init.sh）
│   ├── ops-registry-invoke/  # 算子开发 Plugin（示例）
│   │   ├── agents/           # Agent 定义（.md）
│   │   ├── workflow/         # 工作流 Skill
│   │   ├── AGENTS.md         # Plugin 配置
│   │   ├── init.sh           # 依赖安装脚本
│   │   └── quickstart.md     # 快速入门
│   ├── model-infer-optimize/
│   ├── ops-direct-invoke/
│   └── ...                   # 各官方 Plugin
├── plugins-community/        # 社区 Plugin（实验/非正式版）
├── tests/                    # 测试框架与用例
├── docs/                     # 项目规范文档
│   └── STANDARDS.md          # 开发规范总集
├── install.sh                # 一键安装入口
├── AGENTS.md                 # 本文件
├── CHANGELOG.md              # 变更日志
├── LICENSE                   # 开源许可证
├── OAT.xml                   # 开源审计
└── README.md                 # 项目说明
```

## Skills 分类

项目 skills 目录按功能分类，包括：
- **知识库类**：提供领域知识和参考资料
- **工程模板类**：提供工程脚手架和项目模板
- **调试与测试类**：问题诊断和调试
- **测试开发类**：测试设计、测试开发和代码检视
- **工具辅助类**：辅助工具和实用功能

详细分类见 [Skills 开发规范](docs/STANDARDS.md#skills-分类体系)

## 详细规范

详见 [开发规范总集](docs/STANDARDS.md)
