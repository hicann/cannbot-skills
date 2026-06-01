# CANNBot Skills 开发规范

## 通用命名规范

Skills、Agents 统一采用 **domain 前缀**命名方式：`{domain}-{name}`

**命名建议**：
1. 使用小写字母和连字符（kebab-case）
2. 前缀应简短且具有描述性
3. name 应清晰表达用途/职责
4. 避免使用缩写，除非是广泛认可的

---

## Skills 开发规范

### Skills 分类体系

#### 分类原则

Skills 按功能定位分为五大类，每类技能有明确的职责边界和使用场景。

#### 知识库类

**定位**：提供领域知识和参考资料，作为其他技能的知识源

**特点**：
- 被动查询，不主动执行任务
- 提供设计参考和最佳实践
- 通常是其他技能的依赖

**代表性技能**：

| 技能名称 | 用途 |
|---------|------|
| npu-arch | NPU 架构知识（芯片型号、架构特性） |
| ascendc-tiling-design | Tiling 设计指南（多核切分、Buffer规划） |
| ascendc-api-best-practices | API 使用最佳实践、参数限制 |
| ascendc-docs-search | 文档搜索（本地优先、在线兜底） |

#### 工程模板类

**定位**：提供工程脚手架和项目模板，加速项目初始化

**特点**：
- 提供标准化的工程结构
- 包含可复用的代码模板
- 降低项目搭建门槛

**代表性技能**：

| 技能名称 | 用途 |
|---------|------|
| ascendc-registry-invoke-template | 自定义算子工程模板（Registry Invoke 方式） |
| ascendc-registry-invoke-to-direct-invoke | 注册算子直调改造模板 |

#### 调试与测试类

**定位**：用于问题诊断和调试，帮助定位和解决问题

**特点**：
- 主动执行调试流程
- 提供系统化的问题排查方法
- 包含诊断工具和脚本

**代表性技能**：

| 技能名称 | 用途 |
|---------|------|
| ascendc-precision-debug | 精度调试（数值异常、rtol/atol 不达标） |
| ascendc-runtime-debug | 运行时错误调试（错误码 161xxx/361xxx/561xxx） |
| ascendc-crash-debug | 卡死/崩溃调试（程序无法运行完、Kernel 挂起、Coredump） |

#### 测试开发类

**定位**：用于测试设计、测试开发和代码检视

**特点**：
- 提供测试用例设计方法
- 包含代码检视规范和方法论（4 种工作流：文件检视 / PR检视 / 大型PR检视 / 快速检视）
- 覆盖 UT / ST / 白盒测试

**代表性技能**：

| 技能名称 | 用途 |
|---------|------|
| ascendc-st-design | ST 测试用例设计（aclnn 接口测试） |
| ascendc-ut-develop | UT 开发与覆盖率增强 |
| ascendc-code-review | 代码检视（文件/PR/大型PR/快速检视，假设检验+证据框架+自动路由） |
| ascendc-whitebox-design | 白盒测试设计 |

#### 工具辅助类

**定位**：提供辅助工具和实用功能，提升开发效率

**特点**：
- 提供具体的功能工具
- 可独立使用或配合其他技能
- 包含脚本和自动化工具

**代表性技能**：

| 技能名称 | 用途 |
|---------|------|
| ascendc-env-check | 环境检查（NPU 设备、CANN 配置） |
| ascendc-task-focus | 任务聚焦，解决长任务"迷失在中间"问题 |

> 查看完整技能列表: `ls ops/`

#### 分类使用指南

**选择技能时**：
1. 需要参考资料 → 查询知识库类
2. 需要搭建工程 → 使用工程模板类
3. 遇到问题 → 使用调试与测试类
4. 需要编写测试或检视代码 → 使用测试开发类
5. 需要工具支持 → 使用工具辅助类

**组合使用**：
- 开发流程通常需要组合多个技能
- 知识库类作为基础，模板类加速启动，调试类解决问题，测试开发类保障质量，工具辅助类提升效率

### 创建流程

使用 `skill-creator` 技能创建和优化技能：

```
直接调用 skill-creator 技能即可，它会自动处理：
- 需求分析和规划
- SKILL.md 编写
- 测试和评测
- 迭代优化
- 打包发布
```

详细流程请参考 `skill-creator` 技能

### SKILL.md 结构

```markdown
---
name: skill-name
description: 技能描述（包含触发条件）
---

# {Skill Name}

## 工作流程
1. 步骤一
2. 步骤二

## 脚本工具
- `scripts/main.py` - 主脚本

## 参考资料
- `references/guide.md` - 详细指南
```

### 设计原则

1. **单一职责** - 每个技能专注一个明确的任务
2. **可组合性** - 技能之间可以组合使用
3. **知识依赖单向性** - 标识真源 skill（即某领域知识的唯一上游数据源），下游 skill 消费上游数据，**禁止反向自引用**。修改真源数据时只在真源处更新，避免下游各处漂移。

> 通用原则遵循 [AGENTS.md 核心原则](../AGENTS.md#核心原则)：信息来源可信、渐进式披露、简洁精炼

---

## Agents 开发规范

### 创建 Agent

```markdown
# agents/{agent-name}/AGENT.md
---
name: agent-name
description: Agent 的简短描述
mode: subagent
skills:
  - skill-1
  - skill-2
---

## 职责范围

### 负责
- 任务1

### 不负责
- 任务2
```

### Agent 职责分类

| 分类 | Agent 名称 | 核心职责 | 典型场景 |
|------|-----------|---------|---------|
| **架构类** | `ascendc-ops-architect` | 需求分析、方案设计、技术选型 | 新算子开发启动时 |
| **检视类** | `ascendc-ops-reviewer` | 逐条例假设检验、规范检查 | 开发完成后、上库前 |
| **检视类** | `ascendc-code-summarizer` | 代码结构摘要、侧别识别 | 检视流程 Stage 0 |

### 参考示例

查看 `plugins-official/` 下各 Team 目录中的 Agent 实现：
- `ops-direct-invoke/agents/` - 直调开发子 Agent（architect / developer / reviewer）
- `pypto-op-orchestrator/agents/` - PyPTO 开发子 Agent（analyst / developer / perf-tuner）
- `ops-code-reviewer/agents/` - 代码检视子 Agent

---

## Teams 配置

### 示例：ops-direct-invoke

**核心理念**：Spec-driven Development（规格驱动开发）

**四阶段工作流**：
1. 设计阶段 - 需求分析 → 方案设计 → 测试设计
2. 开发阶段 - 迭代式开发（骨架→整合→全量），算子代码 + ST用例 + UT
3. 验收阶段 - 精度验收 → 性能验收
4. 上库阶段 - 代码检视 → 开发总结

> 详细配置见 `plugins-official/ops-direct-invoke/AGENTS.md`

---

## 代码规范

遵循 [PEP 8](https://peps.python.org/pep-0008/)

**命名约定**：
- 文件名: `skill_loader.py` (小写下划线)
- 类名: `SkillLoader` (大驼峰)
- 函数/方法: `load_skill()` (小写下划线)
- 常量: `MAX_RETRIES` (大写下划线)

---

## 目录结构规范

各层级资源目录的定位和用途：

| 层级 | 目录 | 用途 | 内容示例 |
|------|------|------|---------|
| 领域根目录 | `ops/` | 算子 Skills（正式版：Ascend C + PyPTO） | Ascend C / PyPTO 算子开发相关 |
| 领域根目录 | `ops-lab/` | 算子 Skills/Agents（实验/非正式版） | 实验性技能模块 |
| 领域根目录 | `model/` | 模型优化 | 模型优化相关技能 |
| Skill 内 | `references/` | 按需加载的知识文档 | API 指南、最佳实践、约束说明 |
| Skill 内 | `assets/` | 输出时使用的静态资源 | 模板文件、图标 |
| Skill 内 | `scripts/` | 可执行脚本 | 数据生成、验证脚本 |
| Team 内 | `workflows/` | teams 流程配置文件 | 任务提示词、数据流定义、错误处理指南 |
