# 项目说明书：code-performance-advisor (Skill)

## 1. 项目目录结构 (Project Layout)

```text
code-performance-advisor/
├── SKILL.md                    # Skill 主入口与能力说明（单一事实源）
├── scripts/                    # 能力层：可执行工具
│   ├── README.md               # scripts 说明
│   └── analysis_engine/        # 分析引擎脚本
│       ├── clear.py            # 清理 InputMessages 的便捷脚本
│       └── cli.py              # 命令行入口/胶水脚本
├── assets/                     # 机器可用资产（供脚本读取）
│   ├── conifgs/                # 配置（当前为空，预留）
│   ├── templates/              # 模板资产（当前为空，预留）
│   ├── rules/                  # 机器规则资产
│   │   └── common_rules/       # 通用规则存放
│   ├── manifests/              # 规则索引/清单（当前为空，预留）
│   ├── llm_prompts/            # LLM Prompt 资产（当前为空，预留）
│   └── logs/                   # 运行/演化日志（当前为空，预留）
├── references/                 # 人类可读规范与蓝图
│   ├── README.md               # references 说明
│   ├── architecture/            # 架构/目录说明（本文件所在目录）
│   ├── externel_refs/          # 外部参考资料
│   └── standards/              # 标准与规范文档
├── subskills/                  # 子技能/辅助说明
│   ├── code_tag.md             # 代码打标子技能说明
│   └── tag_code.md             # 标签到代码的映射说明
├── tests/                      # 测试层
│   ├── unit/                   # 单元测试（当前为空，预留）
│   └── integration/            # 集成测试（当前为空，预留）
├── workspace/                  # 运行时输入输出区
│   ├── README.md               # workspace 说明
│   ├── InputMessages/          # 输入消息区
│   │   ├── raw/                # 原始输入
│   │   │   ├── code/           # 原始代码
│   │   │   ├── op_description/ # 原始算子描述
│   │   │   ├── profiling_data/ # 原始 profiling 数据
│   │   │   └── roofline/       # 原始 roofline 相关数据
│   │   ├── curated/            # 清洗后的可复用输入
│   │   │   ├── scenarios/      # 标准化场景
│   │   │   └── paired_examples/# 输入-期望输出对
│   │   └── traces/             # 推理/生成轨迹
│   │       ├── diag_traces/    # 诊断轨迹
│   │       └── rule_gen_traces/# 规则生成轨迹
│   └── OutputMessages/         # 输出消息区
│       └── raw/                # 输出结果（原始）
└── skill_memory.md             # Skill 记忆与注释文档
```

