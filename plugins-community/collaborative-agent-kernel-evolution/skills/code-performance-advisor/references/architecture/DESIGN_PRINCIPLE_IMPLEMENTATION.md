# Code-Performance-Advisor 架构改进方案 v2.1
## 设计原则落地方案

**核心思想评估**: ✅ **优秀且可行**

> "CLI 固化流程与数据契约；subskills 固化推理框架与表达结构；具体结论由证据驱动生成"

这个原则完美契合了编译器式设计哲学,并且特别适合 Claude Code CLI 环境。

---

## 一、设计原则详细解读

### 1.1 CLI 固化流程与数据契约

**含义**:
- **流程固化**: 工作流状态机用代码实现,不依赖 LLM 记忆
- **数据契约**: 模块间通过明确的数据结构通信,类型安全

**落地方案**:
```python
# core/orchestration/workflow.py
class WorkflowEngine:
    """
    状态机驱动的工作流引擎

    状态转移完全确定性,不依赖 LLM 判断
    """

    def execute(self, context: WorkflowContext) -> WorkflowResult:
        state = State.INIT

        while state != State.DONE:
            # 状态转移由数据契约驱动
            if state == State.INIT:
                context = self._phase0_analyze(context)
                state = self._route_next_phase(context)

            elif state == State.SUGGEST:
                suggestions = self._generate_suggestions(context)
                context.suggestions = suggestions
                state = State.APPLY if context.mode == "auto" else State.WAIT_USER

            elif state == State.APPLY:
                result = self._apply_suggestion(context)
                context.last_result = result
                state = State.VERIFY

            # ... 状态机继续

        return WorkflowResult(context)
```

**数据契约示例**:
```python
@dataclass
class WorkflowContext:
    """所有阶段共享的数据契约"""
    op_name: str
    tags: Dict[str, Any]              # Phase 0 输出
    route: RouteDecision              # Phase 0 输出
    suggestions: List[Suggestion]     # Phase 1-3 输出
    applied_code: Optional[str]       # Phase 4 输出
    validation: Optional[Validation]  # Phase 4 输出

    # 类型安全,编译时检查
```

**优势**:
- ✅ 不会"忘记调用"某个阶段
- ✅ 数据流可追溯
- ✅ 易于测试(mock 数据契约)

---

### 1.2 Subskills 固化推理框架与表达结构

**含义**:
- **推理框架**: Subskill 不输出结论,而是提供**思维模板**
- **表达结构**: 输出格式固定,LLM 只填充内容

**落地方案**:

#### 错误模式 (当前)
```markdown
<!-- suggest.md -->
根据 profiling 数据和代码,生成优化建议。

[LLM 自由发挥,每次输出格式不同]
```

#### 正确模式 (改进后)
```python
# core/analysis/suggestion_framework.py
class SuggestionFramework:
    """
    固化的推理框架,强制结构化输出
    """

    TEMPLATE = """
    ## 优化建议: {rule_id}

    ### 证据链 (Evidence Chain)
    1. **观察 (Observation)**: {observation}
       - 数据来源: {data_source}
       - 关键指标: {metrics}

    2. **诊断 (Diagnosis)**: {diagnosis}
       - 根因: {root_cause}
       - 置信度: {confidence}

    3. **处方 (Prescription)**: {prescription}
       - 优化策略: {strategy}
       - 预期改善: {expected_improvement}

    ### 实施方案 (Implementation)
    ```cpp
    {code_before}
    ```

    改为:
    ```cpp
    {code_after}
    ```

    ### 验证标准 (Validation Criteria)
    - {criterion_1}
    - {criterion_2}
    """

    def generate(self, rule: Rule, evidence: Evidence) -> Suggestion:
        # LLM 只负责填充槽位,不设计结构
        prompt = f"""
        你是性能优化专家。根据以下证据,填充建议模板的槽位:

        规则: {rule.description}
        证据: {evidence.to_dict()}

        输出 JSON 格式,字段包含:
        - observation (字符串)
        - diagnosis (字符串)
        - prescription (字符串)
        - code_before (字符串)
        - code_after (字符串)
        - expected_improvement (字符串)
        """

        # LLM 调用
        filled_slots = llm_call(prompt)

        # 套用模板
        return Suggestion(
            content=self.TEMPLATE.format(**filled_slots),
            rule_id=rule.id,
            confidence=evidence.confidence
        )
```

**Subskill 重新定义**:
```markdown
<!-- subskills/suggest.md -->
---
name: suggest
type: reasoning_framework
input_contract: Evidence (tags, profiling, matched_rule)
output_contract: Suggestion (structured)
---

## 推理框架 (Reasoning Framework)

你作为 LLM,负责执行以下推理步骤:

### Step 1: Evidence Extraction
从输入中提取:
- Observation: 描述性能瓶颈的客观现象
- Data Source: 引用具体的 CSV 行或代码行号

### Step 2: Root Cause Diagnosis
基于规则知识库,诊断根因:
- 为什么会出现这个瓶颈?
- 哪些因素是主要贡献者?

### Step 3: Prescription Generation
提出优化方案:
- 修改哪些代码?
- 预期改善多少?

### Step 4: Code Transformation
生成 before/after 代码:
- 使用规则模板中的 good_code 作为参考
- 保持代码风格一致

### 输出格式 (强制 JSON)
```json
{
  "observation": "...",
  "diagnosis": "...",
  "prescription": "...",
  "code_before": "...",
  "code_after": "...",
  "expected_improvement": "..."
}
```

**禁止**: 自由文本输出,必须严格遵守 JSON schema
```

**优势**:
- ✅ 输出可预测,易于解析
- ✅ LLM 只做推理,不做格式设计
- ✅ 推理框架可复用,可测试

---

### 1.3 具体结论由证据驱动生成

**含义**:
- **证据 = 触发条件**: 不是"LLM 觉得应该优化",而是"数据显示必须优化"
- **量化驱动**: 每个建议都有量化证据支撑

**落地方案**:

```python
# core/analysis/evidence_extractor.py
class EvidenceExtractor:
    """
    从数据中提取客观证据,不做主观判断
    """

    def extract(self, profiling: ProfilingData, code: str) -> Evidence:
        # 确定性提取
        evidence = Evidence()

        # 1. 量化指标
        evidence.add_metric("aiv_scalar_ratio", profiling.aiv_scalar_ratio)
        evidence.add_metric("task_duration", profiling.task_duration_us)

        # 2. 代码模式
        evidence.add_pattern("has_for_loop", self._detect_loop(code))
        evidence.add_pattern("uses_counter_mode", self._detect_counter(code))

        # 3. 阈值判断 (明确的规则)
        if profiling.aiv_scalar_ratio > 0.4:
            evidence.add_symptom(
                name="HIGH_SCALAR_OVERHEAD",
                severity="critical",
                threshold=0.4,
                actual=profiling.aiv_scalar_ratio,
                source="op_summary.csv:line_5"
            )

        return evidence


# core/analysis/rule_matcher.py
class RuleMatcher:
    """
    基于证据匹配规则,完全确定性
    """

    def match(self, evidence: Evidence) -> List[RuleMatch]:
        matches = []

        for rule in self.rule_library:
            # 计算匹配分数 (确定性算法)
            score = 0.0
            matched_conditions = []

            for condition in rule.conditions:
                if self._evaluate_condition(condition, evidence):
                    score += condition.weight
                    matched_conditions.append(condition)

            if score >= rule.min_score:
                matches.append(RuleMatch(
                    rule=rule,
                    score=score,
                    evidence=evidence,
                    matched_conditions=matched_conditions
                ))

        return sorted(matches, key=lambda m: m.score, reverse=True)
```

**优势**:
- ✅ 可复现 (同样的数据 → 同样的结论)
- ✅ 可审计 (追溯每个结论的证据来源)
- ✅ 无幻觉 (基于客观数据,非 LLM 猜测)

---

## 二、当前问题诊断

### 2.1 问题清单

| 问题 | 表现 | 根因 | 影响 |
|------|------|------|------|
| **调用遗漏** | auto_optimize 未被调用 | 依赖 LLM 记忆流程 | 自动化失效 |
| **输出混乱** | 14 个无组织的 .md 文件 | 无输出契约,自由文本 | 难以追溯 |
| **状态丢失** | iteration_state.json 未更新 | 手动执行,无状态同步 | 无法恢复 |
| **推理重复** | 每次 LLM 重新分析 | 无推理框架,重新思考 | 慢且不稳定 |
| **入口分散** | cli.py, auto_optimize.py, subskills... | 职责不清 | 学习成本高 |

### 2.2 根因分析

```
当前架构:
┌─────────────────────────────────────────┐
│          SKILL.md (文档)                │  ← 描述性的,非执行性的
├─────────────────────────────────────────┤
│  Subskills (markdown)                   │  ← LLM 自由解读,不强制
├─────────────────────────────────────────┤
│  CLI Scripts (分散)                     │  ← 需要手动串联
├─────────────────────────────────────────┤
│  LLM (Claude)                           │  ← 负责流程 + 推理 + 执行
└─────────────────────────────────────────┘

问题:
- LLM 负责太多,容易遗忘或出错
- 没有强制机制确保流程完整
- 状态在多个地方,难以同步
```

**理想架构** (基于新设计原则):
```
┌─────────────────────────────────────────┐
│  advisor CLI (单一入口)                 │  ← 强制流程,状态机驱动
├─────────────────────────────────────────┤
│  WorkflowEngine (编排层)                │  ← 确定性状态转移
├─────────────────────────────────────────┤
│  Modules (功能层)                       │  ← 明确数据契约
│  - Evidence Extractor                   │
│  - Rule Matcher                         │
│  - Suggestion Generator (含 LLM)        │
│  - Code Transformer                     │
│  - Validator                            │
├─────────────────────────────────────────┤
│  Reasoning Frameworks (推理模板)        │  ← Subskills 变为模板
├─────────────────────────────────────────┤
│  Knowledge Base (规则库 + 案例库)       │  ← 证据来源
└─────────────────────────────────────────┘
```

---

## 三、渐进式改进方案 (无需大规模重构)

### 阶段 0: 立即可做 (1-2 天)

#### 目标: 固化当前工作流,防止遗忘

**行动项**:

1. **创建流程编排脚本**:
```bash
# scripts/analysis_engine/workflow.py
"""
固化的工作流引擎,强制执行完整流程
"""

class Phase:
    INIT = "init"
    ANALYZE = "analyze"
    SUGGEST = "suggest"
    APPLY = "apply"
    BUILD = "build"
    EVALUATE = "evaluate"
    UPDATE = "update"
    DONE = "done"

class Workflow:
    def __init__(self, op_name: str):
        self.op_name = op_name
        self.state = Phase.INIT
        self.context = {}

    def run(self, mode="interactive"):
        """强制执行完整流程"""

        # Phase 0
        if self.state == Phase.INIT:
            self._phase_init()
            self.state = Phase.ANALYZE

        if self.state == Phase.ANALYZE:
            route = self._phase_analyze()
            self.context["route"] = route
            self.state = Phase.SUGGEST

        # 后续阶段...
        # 每个阶段完成后强制状态转移

    def _phase_analyze(self):
        """Phase 0: 数据清点 + 路由"""
        # 1. 确保 tag 文件存在
        tag_file = f"workspace/InputMessages/curated/tags/tag_{self.op_name}.json"
        if not os.path.exists(tag_file):
            raise WorkflowError(f"Missing tag file: {tag_file}. Run code_tag first!")

        # 2. 执行评分
        subprocess.run([
            "python", "scripts/analysis_engine/cli.py", "score",
            "--tag-file", tag_file
        ], check=True)

        # 3. 读取评分结果
        scored = json.load(open("workspace/OutputMessages/scored_results.json"))

        # 4. 路由决策 (确定性)
        return self._route_decision(scored)
```

**使用方式**:
```bash
# 替代手动步骤
python scripts/analysis_engine/workflow.py run --op fastgelu --mode interactive

# 内部自动执行:
# - init_workspace (如果需要)
# - code_tag
# - cli.py score
# - 根据路由调用 suggest/deep_research
# - 提示用户确认
# - 应用修改
# - 编译
# - 评测
# - 更新状态
# - 循环或结束
```

2. **添加状态检查点**:
```python
class WorkflowCheckpoint:
    """每个阶段结束时强制写入状态"""

    @staticmethod
    def save(phase: str, op_name: str, data: dict):
        checkpoint_file = f"workspace/OutputMessages/iterations/{op_name}/checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({
                "phase": phase,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }, f, indent=2)

    @staticmethod
    def load(op_name: str) -> dict:
        checkpoint_file = f"workspace/OutputMessages/iterations/{op_name}/checkpoint.json"
        if os.path.exists(checkpoint_file):
            return json.load(open(checkpoint_file))
        return None
```

**收益**:
- ✅ 不会忘记调用某个阶段
- ✅ 中断后可从断点恢复
- ✅ 状态可追溯

---

### 阶段 1: 数据契约标准化 (3-5 天)

#### 目标: 所有模块输入输出类型化

**行动项**:

1. **定义数据模型**:
```python
# core/common/data_models.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class ProfilingMetrics:
    task_duration_us: float
    aiv_scalar_ratio: float
    aiv_vec_ratio: float
    aiv_mte2_ratio: float
    # ... 其他指标

    @classmethod
    def from_csv(cls, csv_path: str) -> "ProfilingMetrics":
        """从 op_summary.csv 解析"""
        df = pd.read_csv(csv_path)
        row = df.iloc[0]
        return cls(
            task_duration_us=float(row["Task Duration(us)"]),
            aiv_scalar_ratio=float(row["aiv_scalar_ratio"]),
            # ...
        )

@dataclass
class CodeTags:
    domain: List[str]      # ["U.Vector", "O.Elementwise"]
    symptoms: List[str]    # ["S.ScalarBound", "S.LowVecUtil"]
    context: List[str]     # ["C.Arch.910B"]

    @classmethod
    def from_json(cls, json_path: str) -> "CodeTags":
        """从 tag_*.json 解析"""
        data = json.load(open(json_path))
        return cls(**data)

@dataclass
class Evidence:
    """证据对象,驱动后续决策"""
    metrics: ProfilingMetrics
    tags: CodeTags
    code_patterns: Dict[str, bool]
    symptoms: List[Symptom]

    def to_dict(self) -> dict:
        """序列化为 JSON"""
        return asdict(self)

@dataclass
class Suggestion:
    id: str
    rule_id: str
    confidence: float
    observation: str
    diagnosis: str
    prescription: str
    code_before: str
    code_after: str
    expected_improvement: str
    evidence: Evidence

    def to_markdown(self) -> str:
        """渲染为 markdown"""
        return SuggestionFramework.TEMPLATE.format(**asdict(self))
```

2. **重构模块接口**:
```python
# scripts/analysis_engine/cli.py
def cmd_score(args) -> RuleMatchResult:
    """返回类型明确的对象,而非 JSON 文件"""
    tags = CodeTags.from_json(args.tag_file)
    matcher = RuleMatcher()
    matches = matcher.match(tags)

    result = RuleMatchResult(
        top_rule=matches[0],
        all_matches=matches,
        routing=decide_route(matches[0])
    )

    # 仍然写入文件(兼容)
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result  # 返回对象,可编程使用
```

**收益**:
- ✅ 类型安全,IDE 自动补全
- ✅ 序列化/反序列化标准化
- ✅ 易于测试 (mock 对象)

---

### 阶段 2: Subskills 转为推理框架 (5-7 天)

#### 目标: Subskill 输出结构化,LLM 只填充内容

**行动项**:

1. **重写 suggest.md**:
```markdown
<!-- subskills/suggest_framework.md -->
---
type: reasoning_framework
input: Evidence (JSON)
output: Suggestion (JSON)
---

## LLM 任务定义

你是性能优化专家。根据输入的证据对象,生成结构化的优化建议。

### 输入格式 (Evidence)
```json
{
  "metrics": {
    "task_duration_us": 8.1,
    "aiv_scalar_ratio": 0.498
  },
  "tags": {
    "domain": ["U.Vector"],
    "symptoms": ["S.ScalarBound"]
  },
  "matched_rule": {
    "id": "R_API_VECTOR_COUNTER_MODE",
    "description": "..."
  }
}
```

### 推理步骤 (强制顺序)

#### Step 1: 观察 (Observation)
从 `metrics` 中提取关键指标:
- 哪个指标超过阈值?
- 引用具体的数值

输出示例:
"aiv_scalar_ratio=49.8%,超过 40% 阈值 (阈值来源: R_API_VECTOR_COUNTER_MODE.threshold)"

#### Step 2: 诊断 (Diagnosis)
结合 `matched_rule.description` 和 `tags.symptoms`,诊断根因:
- 为什么会出现这个症状?
- 哪些代码模式导致?

输出示例:
"代码中存在 for 循环逐 tile 处理,每次迭代产生分支指令,增加标量开销"

#### Step 3: 处方 (Prescription)
基于 `matched_rule.good_code` 提出优化方案:
- 修改哪些代码结构?
- 预期改善多少? (基于历史案例)

输出示例:
"消除 for 循环,使用 Counter 模式一次性处理全部 elements。预期 scalar_ratio 降低至 30% 以下,Task Duration 改善 20-30%"

#### Step 4: 代码转换 (Code Transformation)
生成 before/after 代码:
- 使用实际代码,保持风格
- 标注关键修改点

### 输出格式 (严格 JSON Schema)
```json
{
  "observation": "string",
  "diagnosis": "string",
  "prescription": "string",
  "code_before": "string",
  "code_after": "string",
  "expected_improvement": "string"
}
```

### 约束
- **禁止**: 自由文本,随意格式
- **必须**: 严格遵守 JSON schema
- **引用**: 所有结论必须引用证据中的数据
```

2. **实现框架调用器**:
```python
# core/analysis/suggestion_generator.py
class SuggestionGenerator:
    """
    使用推理框架生成建议
    """

    def generate(self, evidence: Evidence, rule: Rule) -> Suggestion:
        # 1. 准备输入 (结构化)
        input_data = {
            "metrics": asdict(evidence.metrics),
            "tags": asdict(evidence.tags),
            "matched_rule": {
                "id": rule.id,
                "description": rule.description,
                "threshold": rule.threshold,
                "good_code": rule.good_code
            }
        }

        # 2. 加载推理框架
        framework = self._load_framework("suggest_framework.md")

        # 3. 构造 LLM prompt
        prompt = f"""
        {framework.instructions}

        ### 输入数据
        ```json
        {json.dumps(input_data, indent=2)}
        ```

        ### 输出要求
        严格按照 JSON schema 输出,不要额外解释。
        """

        # 4. 调用 LLM
        response = llm_call(prompt, response_format="json")

        # 5. 解析 & 验证
        filled_slots = json.loads(response)
        self._validate_schema(filled_slots, framework.output_schema)

        # 6. 构造 Suggestion 对象
        return Suggestion(
            id=uuid.uuid4().hex,
            rule_id=rule.id,
            confidence=evidence.confidence,
            **filled_slots,
            evidence=evidence
        )
```

**收益**:
- ✅ LLM 输出可预测
- ✅ 易于解析和后处理
- ✅ 推理框架可复用

---

### 阶段 3: 统一 CLI 入口 (5-7 天)

#### 目标: advisor 命令替代所有脚本

**行动项**:

1. **创建 advisor.py**:
```python
#!/usr/bin/env python3
"""
Unified CLI for Code Performance Advisor

Usage:
    advisor analyze <op>       # Phase 0: Analyze and route
    advisor suggest <op>       # Generate suggestions
    advisor apply <op> <id>    # Apply a specific suggestion
    advisor verify <op>        # Verify improvement
    advisor optimize <op>      # End-to-end optimization loop
"""

import click
from core.orchestration import PerformanceAdvisor
from core.common.data_models import *

@click.group()
def cli():
    """Code Performance Advisor - Intelligent performance optimization"""
    pass

@cli.command()
@click.argument("op_name")
@click.option("--mode", default="interactive", type=click.Choice(["auto", "interactive"]))
def optimize(op_name: str, mode: str):
    """
    End-to-end optimization loop

    Example:
        advisor optimize fastgelu --mode interactive
    """
    advisor = PerformanceAdvisor(op_name)
    result = advisor.optimize(mode=mode)

    click.echo(f"✅ Optimization completed!")
    click.echo(f"   Task Duration: {result.baseline_duration}us → {result.final_duration}us")
    click.echo(f"   Improvement: {result.improvement_pct}%")

@cli.command()
@click.argument("op_name")
def analyze(op_name: str):
    """
    Phase 0: Quick analysis and routing

    Example:
        advisor analyze fastgelu
    """
    advisor = PerformanceAdvisor(op_name)
    route = advisor.analyze()

    click.echo(f"🔍 Analysis completed:")
    click.echo(f"   Route: {route.path}")
    click.echo(f"   Confidence: {route.confidence}")
    click.echo(f"   Top Rule: {route.top_rule.id}")

# ... 其他命令

if __name__ == "__main__":
    cli()
```

2. **逐步迁移脚本**:
```bash
# 旧方式 (弃用)
python scripts/analysis_engine/init_workspace.py --op fastgelu
python scripts/analysis_engine/cli.py score --tag-file ...
# ...

# 新方式 (推荐)
advisor optimize fastgelu

# 内部自动调用:
# - init_workspace (如需要)
# - code_tag
# - score
# - suggest
# - apply
# - build
# - evaluate
# - update-baseline
# - 循环或结束
```

**收益**:
- ✅ 单一入口,易学习
- ✅ 内置帮助文档
- ✅ 参数验证

---

## 四、最终架构对比

### Before (v1.0 - 当前)
```
问题:
- 14 个 .md 文件散落各处
- CLI 脚本需要手动串联 (6+ 步骤)
- Subskills 是文档,LLM 自由解读
- 依赖 LLM 记忆完整流程
- 无状态管理,中断即丢失

结果:
❌ auto_optimize 被遗忘
❌ 输出混乱
❌ 自动化失效
```

### After (v2.1 - 改进后)
```
改进:
- 单一 advisor 命令
- WorkflowEngine 强制状态转移
- Subskills 是推理框架 (JSON schema)
- 数据契约类型化 (dataclass)
- 状态持久化 (checkpoint)

结果:
✅ 不会遗忘调用
✅ 输出结构化,可追溯
✅ 自动化程度高
✅ 易于维护和扩展
```

---

## 五、关键设计决策

### 5.1 为什么不立即全面重构?

**理由**:
1. 当前系统 Phase 0 已经工作良好
2. 渐进式改进风险更低
3. 可以逐步验证设计是否正确

**策略**: 并行运行新旧系统,逐步切换

### 5.2 LLM 的正确使用边界

**LLM 应该做**:
- 根据证据推理根因
- 根据模板填充内容
- 生成代码转换逻辑

**LLM 不应该做**:
- 记忆完整工作流
- 设计输出格式
- 判断是否需要某个阶段 (应由数据驱动)

### 5.3 状态管理策略

**原则**:
- 每个阶段结束强制写入状态
- 状态文件是单一真相来源
- 支持断点恢复

**实现**:
```python
# 每个阶段结束
workflow.checkpoint(phase="suggest", data={
    "suggestions": [s.to_dict() for s in suggestions],
    "selected": selected_id
})

# 恢复执行
workflow = Workflow.resume(op_name)
workflow.continue_from_checkpoint()
```

---

## 六、实施优先级

### P0 (本周完成)
- [x] 创建 workflow.py 流程编排脚本
- [x] 添加状态检查点机制
- [ ] 重写 suggest.md 为推理框架
- [ ] 测试 fastgelu 端到端流程

### P1 (下周完成)
- [ ] 定义完整数据模型 (data_models.py)
- [ ] 重构 cli.py 返回类型化对象
- [ ] 实现 SuggestionGenerator
- [ ] 添加单元测试

### P2 (2周后)
- [ ] 创建 advisor.py 统一 CLI
- [ ] 迁移所有命令到 advisor
- [ ] 弃用旧脚本 (标记 deprecated)
- [ ] 更新文档

---

## 七、成功指标

### 短期 (2周)
- [ ] advisor optimize fastgelu 一次成功
- [ ] 不再出现"忘记调用"问题
- [ ] 输出全部结构化 (JSON + Markdown)

### 中期 (1个月)
- [ ] 10+ 算子用新流程优化
- [ ] 旧脚本完全弃用
- [ ] 代码覆盖率 >70%

### 长期 (3个月)
- [ ] 知识闭环运行
- [ ] 规则库扩展到 30+ 条
- [ ] 用户满意度 >4.5/5.0

---

## 八、总结

### 核心思想评价

> "CLI 固化流程与数据契约；subskills 固化推理框架与表达结构；具体结论由证据驱动生成"

**评分**: ⭐⭐⭐⭐⭐ (5/5)

**优势**:
1. ✅ 职责清晰: CLI=流程, Subskills=推理, 证据=触发
2. ✅ 可维护: 数据契约 + 类型安全 + 状态管理
3. ✅ 易用性: 单一入口 + 自动编排
4. ✅ 简洁性: 消除冗余 + 确定性流程

**落地路径**:
- 阶段 0: 流程编排 (1-2天)
- 阶段 1: 数据契约 (3-5天)
- 阶段 2: 推理框架 (5-7天)
- 阶段 3: 统一 CLI (5-7天)

**总工期**: 3-4周,渐进式改进,无需大规模重构

---

**结论**: 这个设计原则完美契合了架构 v2.0 的理念,并且特别适合 Claude Code CLI 环境。建议立即启动阶段 0 实施。

**下一步**: 创建 `scripts/analysis_engine/workflow.py` 流程编排脚本。
