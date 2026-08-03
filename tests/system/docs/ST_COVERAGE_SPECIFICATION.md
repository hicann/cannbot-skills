# ST 用例覆盖率规范

## 1. 概述

### 1.1 背景

传统软件测试中，代码覆盖率通过行/分支/函数的精确追踪来度量。AI Agent Skill 的测试对象是自然语言指令（SKILL.md），AI 对 skill 的执行是隐式的、不可追踪的，因此不能直接套用传统代码覆盖率的思路。

本文档定义了一套针对 AI Agent Skill 的三维覆盖率规范，用于衡量 ST 测试用例对 skill 能力的覆盖完整性。

### 1.2 覆盖维度总览

| 编号 | 维度 | 衡量对象 | 数据来源 |
|------|------|---------|---------|
| 维度一 | 触发关键词覆盖率 | SKILL.md 中声明的触发关键词被用例 prompt 覆盖的比例 | SKILL.md description + evals.json prompt |
| 维度二 | 使用场景覆盖率 | SKILL.md 中声明的使用场景被用例覆盖的比例 | SKILL.md description + evals.json prompt/title |
| 维度三 | 内容章节覆盖率 | SKILL.md 的功能章节被用例 expected_output 间接验证的比例 | SKILL.md 章节标题 + evals.json expected_output |

### 1.3 覆盖等级定义

| 等级 | 综合得分 | 含义 |
|------|---------|------|
| 🟢 优秀 | ≥ 90 | 覆盖充分，关注变更时同步 |
| 🟡 良好 | 70–89 | 覆盖基本完整，可补充个别缺口 |
| 🟠 需改善 | 50–69 | 存在明显覆盖缺口，需制定补充计划 |
| 🔴 不足 | < 50 | 覆盖严重不足，需系统性补充用例 |

综合得分计算方式见附录 A。

---

## 2. 维度定义与计算规则

### 2.1 维度一：触发关键词覆盖率（Trigger Keyword Coverage）

**定义**：SKILL.md `description` 中声明的触发关键词，有多少被测试用例的 prompt 文本覆盖。

**输入**：
- `K` = 从 SKILL.md frontmatter `description` 中 `触发关键词：` 后面解析得到的关键词集合
- `P` = 所有用例 prompt 文本拼接后的字符串

**规则**：对于每个关键词 `k ∈ K`，若 `k` 在 `P` 中出现（不区分大小写），则视为已覆盖。

**计算公式**：

```
关键词覆盖率 = |{k ∈ K : k 出现在 P 中}| / |K|
```

**示例**：`ascendc-env-check` 的关键词 `K` = {环境检查, NPU设备, npu-smi, CANN安装, 设备查询, 资源监控, 检查CANN环境变量, NPU架构, npu arch}

| 状态 | 关键词 |
|------|--------|
| ✅ 已覆盖 | 环境检查、npu-smi、NPU设备、CANN安装 |
| ❌ 未覆盖 | 资源监控、NPU架构、npu arch、设备查询、检查CANN环境变量 |

关键词覆盖率 = 4/9。

**约束条件**：
- 校验对象是 prompt 文本（人工编写，可精确控制），而非 AI 回复。这避免了 AI 措辞多样性对判定的干扰。
- 采用子串匹配（不区分大小写）即可，无需语义理解，可静态执行。

### 2.2 维度二：使用场景覆盖率（Use Case Coverage）

**定义**：SKILL.md `description` 中声明的使用场景，有多少有对应的测试用例。

**输入**：
- `S` = 从 SKILL.md description 中 `用于：(1)...(2)...` 提取的场景编号集合
- 每个用例的 `title` 和 `prompt` 文本

**规则**：对于每个场景 `s ∈ S`，若存在至少一个用例，其 `title + prompt` 的语义与场景 `s` 匹配，则视为已覆盖。

**场景提取规则**：

description 中符合 `用于：(N) 文本` 格式的条目按编号提取：

```
用于：(1) 通过 npu-smi 查询 NPU 设备信息，(2) 检查 CANN 环境配置，
(3) 验证开发依赖是否完整，(4) 运行时检测当前设备 NPU 架构。
```

→ `S` = {1, 2, 3, 4}

**计算公式**：

```
场景覆盖率 = |{s ∈ S : ∃ 用例与 s 语义匹配}| / |S|
```

**示例**：

| 场景 | 覆盖状态 | 对应用例 |
|------|---------|---------|
| 查询 NPU 设备信息 | ✅ | Case 1: 检查NPU驱动安装命令 |
| 检查 CANN 环境配置 | ✅ | Case 2: 配置环境变量永久生效 |
| 验证开发依赖完整性 | ✅ | Case 6: 安装前依赖检查 |
| 检测 NPU 架构 | ❌ | — |

场景覆盖率 = 3/4。

**约束条件**：
- 语义匹配可通过 embedding 相似度或关键词重叠判断，也可由用例编写者在 `coverage_manifest` 中人工声明
- 第一阶段推荐人工声明为主，自动化语义匹配为辅

### 2.3 维度三：内容章节覆盖率（Section Coverage）

**定义**：SKILL.md 中的功能章节，有多少被测试用例的 expected_output 间接验证到。

**输入**：
- `H` = SKILL.md 中 `##` 级功能标题集合（过滤掉概述、前置条件、示例等非功能章节）
- 每个用例的 `expected_output` 文本

**规则**：对于每个章节 `h ∈ H`，若存在至少一个用例，其 `expected_output` 的语义涉及该章节的核心内容，则视为已覆盖。

**计算公式**：

```
章节覆盖率 = |{h ∈ H : ∃ 用例的 expected_output 覆盖 h}| / |H|
```

**约束条件**：
- SKILL.md 是流程式或知识库式的，章节间存在上下文关联，一个用例可能间接覆盖多个章节
- 目前 ST 框架不追踪 AI 执行时实际读取了 SKILL.md 的哪些段落，因此该维度的判定依赖 expected_output 的语义分析
- 判定可采用关键词匹配或 AI-as-judge 两种方式，具体由实现工具决定

---

## 3. 数据结构规范

### 3.1 coverage_manifest 字段

在 `evals.json` 顶层增加 `coverage_manifest` 对象，声明该 skill 的覆盖率元数据：

```json
{
  "skill_name": "ascendc-env-check",
  "eval_mode": "text",

  "coverage_manifest": {
    "trigger_keywords_covered": [
      "环境检查",
      "npu-smi",
      "CANN安装",
      "设备查询"
    ],
    "scenarios_covered": [1, 2],
    "sections_covered": [
      "NPU 设备检查",
      "CANN 环境检查"
    ],
    "last_reviewed": "2026-07-30"
  },

  "evals": [ ... ]
}
```

### 3.2 字段定义

| 路径 | 类型 | 级别 | 说明 |
|------|------|------|------|
| `coverage_manifest` | object | 建议 | 覆盖率元数据声明 |
| `coverage_manifest.trigger_keywords_covered` | string[] | 建议 | 该 skill 所有用例共同覆盖的触发关键词列表 |
| `coverage_manifest.scenarios_covered` | int[] | 建议 | 覆盖的场景编号列表（对应 `description` 中的场景序号） |
| `coverage_manifest.sections_covered` | string[] | 建议 | 覆盖的 SKILL.md 章节标题列表 |
| `coverage_manifest.last_reviewed` | date | 建议 | 最近一次覆盖率评审日期 |

### 3.3 用例级覆盖标注

每个用例可增加 `coverage` 字段，声明该用例覆盖的具体范围：

| 路径 | 类型 | 级别 | 说明 |
|------|------|------|------|
| `evals[].coverage` | object | 否 | 单个用例的覆盖标注 |
| `evals[].coverage.keywords` | string[] | 否 | 该用例覆盖的触发关键词 |
| `evals[].coverage.sections` | string[] | 否 | 该用例覆盖的 SKILL.md 章节标题 |
| `evals[].coverage.scenarios` | int[] | 否 | 该用例覆盖的场景编号 |

### 3.4 级别说明

| 级别 | 含义 |
|------|------|
| **建议** | 建议填写该字段。缺少时覆盖率分析工具将自动从 SKILL.md 提取关键词和场景进行估算，但精度可能低于人工声明 |
| **否** | 可选字段。仅在需要精细追踪时填写 |

---

## 4. 门禁规则

### 4.1 规则定义

以下规则在 Phase 1 静态验证（`test_skill_basic.py`）中执行，级别均为 `warn`（提醒不阻断）：

| 规则 ID | 测试项 | 说明 |
|---------|-------|------|
| S-COV-01 | coverage_manifest 存在性 | 鼓励声明覆盖率元数据，无 `coverage_manifest` 时给出提示 |
| S-COV-02 | trigger_keywords_covered 非空 | 至少覆盖 1 个触发关键词 |
| S-COV-03 | 关键词覆盖率 ≥ 60% | 触发关键词是最直接的 skill 触发信号，多数应有用例覆盖 |
| S-COV-04 | scenarios_covered 完整性 | 所有使用场景应至少有一个对应用例 |

### 4.2 规则说明

- 覆盖率校验设为 `warn` 级别而非 `error`。覆盖率不足不意味着 skill 质量有问题，仅提示用例需要补充。
- S-COV-03 和 S-COV-04 仅在 `coverage_manifest` 存在时执行。若未声明 `coverage_manifest`，仅触发 S-COV-01 提示。

---

## 5. 附录

### A. 综合得分计算

```
综合得分 = 关键词覆盖率 × 40 + 场景覆盖率 × 35 + 章节覆盖率 × 25
```

权重分配依据：

| 维度 | 权重 | 理由 |
|------|------|------|
| 关键词覆盖率 | 40% | 直接关联触发准确性，数据来源明确，判定方式确定性强 |
| 场景覆盖率 | 35% | 验证功能完整度，数据来源明确，判定依赖语义匹配 |
| 章节覆盖率 | 25% | 间接验证 SKILL.md 内容，判定依赖预期输出分析，确定性最低 |

### B. 与 L2 行为测试规则的映射

| 覆盖率维度 | 对应 L2 规则 | 关系说明 |
|-----------|-------------|---------|
| 触发关键词覆盖率 | B-TRIG-01（精准触发） | L2 验证单个关键词的触发正确性（质量），覆盖率衡量所有关键词的覆盖全面性（广度） |
| 使用场景覆盖率 | B-TRIG-02（模糊触发） | L2 验证每个场景的触发质量，覆盖率衡量场景覆盖完整性 |

L2 行为测试验证的是 "执行质量"，ST 覆盖率衡量的是 "覆盖广度"，两者互补。

### C. 全量 Skill 目录覆盖状态快照

| 域 | 技能数 | 有 ST 用例 | 覆盖率元数据 |
|----|--------|-----------|-------------|
| ops/ | 44 | 待扫描 | 0 |
| graph/ | 6 | 待扫描 | 0 |
| model/ | 16 | 待扫描 | 0 |
| infra/ | 4 | 待扫描 | 0 |
| plugins-official/ | 7 | 待扫描 | 0 |
| plugins-community/ | 多个 | 待扫描 | 0 |

> `coverage_manifest` 字段为新增规范，当前全仓尚无标注。`trigger_keywords_covered`、`scenarios_covered`、`sections_covered` 均可在首次添加时通过人工声明补全。
