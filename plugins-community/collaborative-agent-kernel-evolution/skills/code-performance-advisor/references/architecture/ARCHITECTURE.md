# Code Performance Advisor - System Architecture Design

**Version**: 2.0
**Date**: 2026-02-25
**Status**: Design Proposal
**Author**: Architecture Team

---

## Executive Summary

Code Performance Advisor is an **intelligent performance optimization system** that transforms low-performance AscendC code into high-performance code through a combination of expert rules, data-driven analysis, and LLM-powered pattern recognition.

### Core Value Proposition

```
Input:  Code + Profiling Data + Performance Goal
Output: Optimized Code + Validated Improvement + New Rules (Knowledge)
```

### Design Philosophy

1. **Compiler-like Precision**: Treat optimization as a compilation process with clear phases
2. **Knowledge Evolution**: Every validated optimization becomes institutional knowledge
3. **Automation First, LLM Enhancement**: Automate deterministic tasks, use LLM for pattern recognition
4. **Verifiable Quality**: No suggestion is "done" until validated by profiling
5. **Progressive Disclosure**: Start simple (rule matching), escalate only when needed

---

## 1. System Architecture Overview

### 1.1 Layered Architecture (5 Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│                     L1: Interface Layer                          │
│  (CLI, API, Web UI) - User-facing entry points                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  L2: Orchestration Layer                         │
│  Advisor (Main Controller), Router, Iteration Manager, State    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    L3: Analysis Layer                            │
│  Rule Matcher, Evidence Extractor, Pattern Recognizer           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   L4: Execution Layer                            │
│  Code Transformer, Builder, Validator, Profiler                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   L5: Knowledge Layer                            │
│  Rule Library, Case Library, Performance Models                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow Pipeline

```
[Raw Input] → [Data Loader] → [Tag Extractor] → [Rule Matcher] → [Suggestion Generator]
                                                                           ↓
[Knowledge Capture] ← [Validator] ← [Profiler] ← [Builder] ← [Code Transformer]
        ↓
[Rule Library Update]
```

---

## 2. Layer-by-Layer Design

### 2.1 L1: Interface Layer

**Purpose**: Provide user-friendly entry points to the system.

**Components**:

#### 2.1.1 CLI (Primary Interface)
```python
# Unified command structure
advisor <command> [options]

Commands:
  analyze <op>       # Phase 0: Analyze and route
  suggest <op>       # Generate suggestions
  apply <op> <id>    # Apply a specific suggestion
  verify <op>        # Verify improvement
  optimize <op>      # End-to-end optimization loop
  rule <action>      # Rule management (list, add, update)
  config <action>    # Configuration management
```

**Design Principles**:
- Single entry point (`advisor` command)
- Verb-first command structure (like git)
- Consistent option naming
- Rich help text with examples

#### 2.1.2 Python API (Future)
```python
from code_performance_advisor import Advisor

advisor = Advisor(op_dir="path/to/operator")
suggestions = advisor.analyze()
result = advisor.apply(suggestions[0])
```

---

### 2.2 L2: Orchestration Layer

**Purpose**: Coordinate system components and manage execution flow.

#### 2.2.1 Advisor (Main Controller)

**Responsibility**: Top-level orchestration of optimization workflow.

```python
class PerformanceAdvisor:
    """
    Main orchestrator for performance optimization.

    Coordinates: Data Loading → Analysis → Execution → Validation → Knowledge Capture
    """

    def __init__(self, op_name: str, config: AdvisorConfig):
        self.op_name = op_name
        self.config = config
        self.state_manager = StateManager(op_name)
        self.router = PhaseRouter()

    def optimize(self, mode: str = "interactive") -> OptimizationResult:
        """
        Execute end-to-end optimization.

        Modes:
        - interactive: User confirms each step
        - auto: Fully automated (stop on goal or max iterations)
        - suggest-only: Generate suggestions without applying
        """
        # Phase 0: Data inventory + routing
        context = self._load_context()
        route = self.router.decide(context)

        # Phase 1-3: Generate suggestions
        suggestions = self._generate_suggestions(route, context)

        # Iteration loop
        for iteration in range(self.config.max_iterations):
            # Select best untried suggestion
            suggestion = self._select_suggestion(suggestions, iteration)

            if mode == "interactive":
                if not self._confirm_apply(suggestion):
                    continue

            # Apply → Build → Validate
            result = self._execute_suggestion(suggestion, context)

            if result.improved:
                # Capture knowledge
                self._capture_knowledge(suggestion, result)

                if result.meets_goal(context.goal):
                    break  # Success!

        return self._generate_report()
```

#### 2.2.2 Phase Router

**Responsibility**: Decide which analysis path to take based on rule confidence.

```python
class PhaseRouter:
    """Route to appropriate analysis phase based on rule matching confidence."""

    def decide(self, context: AnalysisContext) -> RoutingDecision:
        """
        Routing logic:
        - Fast Path: score >= 0.7 OR (score >= 0.55 AND coverage >= 0.8)
        - Moderate Path: 0.3 <= score < 0.55
        - Deep Path: score < 0.3
        """
        scored_results = context.scored_results
        max_score = scored_results.top_score
        coverage = scored_results.top_coverage

        if max_score >= 0.7 or (max_score >= 0.55 and coverage >= 0.8):
            return RoutingDecision(
                path="fast",
                analyzer="RuleMatcher",
                confidence="high"
            )
        elif max_score >= 0.3:
            return RoutingDecision(
                path="moderate",
                analyzer="DeepAnalyzer",
                confidence="medium"
            )
        else:
            return RoutingDecision(
                path="deep",
                analyzer="PatternRecognizer",
                confidence="low"
            )
```

#### 2.2.3 State Manager

**Responsibility**: Persist and restore optimization state for multi-session work.

```python
class StateManager:
    """
    Manage optimization state across sessions.

    State includes:
    - Current iteration
    - Attempted suggestions
    - Performance history
    - Baseline snapshots
    """

    def save_state(self, state: OptimizationState) -> None:
        """Save state to workspace/states/{op}/state.json"""

    def load_state(self) -> OptimizationState:
        """Load latest state or initialize new"""

    def snapshot_baseline(self, code: str, profiling: Dict) -> None:
        """Create immutable baseline snapshot"""

    def update_baseline(self, code: str, profiling: Dict) -> None:
        """Update current baseline after improvement"""
```

---

### 2.3 L3: Analysis Layer

**Purpose**: Extract insights from code and profiling data, generate optimization suggestions.

#### 2.3.1 Rule Matcher

**Responsibility**: Match operator characteristics against expert rule library.

```python
class RuleMatcher:
    """
    Fast rule matching via tag-based scoring.

    Uses: Weighted Jaccard similarity on structured tags
    """

    def __init__(self, rule_index: RuleIndex):
        self.rule_index = rule_index

    def match(self, operator_tags: OperatorTags) -> List[RuleMatch]:
        """
        Score all rules against operator tags.

        Returns: Sorted list of (rule, score, coverage) tuples
        """
        matches = []
        for rule in self.rule_index.rules:
            score = self._weighted_jaccard(operator_tags, rule.tags)
            coverage = self._coverage_ratio(operator_tags, rule.required_tags)
            matches.append(RuleMatch(rule, score, coverage))

        return sorted(matches, key=lambda m: (-m.score, -m.coverage))
```

#### 2.3.2 Evidence Extractor

**Responsibility**: Extract quantitative evidence from profiling data to support suggestions.

```python
class EvidenceExtractor:
    """
    Extract relevant metrics from profiling CSV.

    Supports:
    - Operator-level metrics (task duration, utilization ratios)
    - Bottleneck identification (scalar/vector/memory bound)
    - Comparative analysis (before/after)
    """

    def extract(self, csv_path: Path, operator_name: str) -> Evidence:
        """
        Parse CSV and extract relevant metrics.

        Returns: Evidence object with:
        - Bottleneck type (scalar/vector/memory)
        - Key metrics (aiv_vec_ratio, task_duration, etc.)
        - Comparative deltas (if baseline exists)
        """
```

#### 2.3.3 Suggestion Generator

**Responsibility**: Generate structured optimization suggestions.

```python
class SuggestionGenerator:
    """
    Generate actionable suggestions from matched rules.

    Pipeline:
    1. Load rule documentation
    2. Parse code examples (base_code vs good_code)
    3. Extract evidence from profiling
    4. Match patterns in operator code
    5. Render structured suggestion
    """

    def generate(
        self,
        rule_match: RuleMatch,
        operator_code: str,
        evidence: Evidence
    ) -> Suggestion:
        """
        Generate single suggestion from a matched rule.

        Output: Suggestion with:
        - Problem diagnosis (with evidence)
        - Why this rule applies
        - Code analysis (current vs recommended pattern)
        - Modification plan (specific line changes)
        - Expected improvement (quantified)
        - Verification method
        """
```

#### 2.3.4 Pattern Recognizer (Deep Analysis)

**Responsibility**: Use LLM for pattern recognition when rules don't match.

```python
class PatternRecognizer:
    """
    LLM-powered pattern recognition for novel bottlenecks.

    Use when: Rule matching score < 0.3 (no known patterns)
    """

    def recognize(
        self,
        code: str,
        profiling: Dict,
        flowchart: Optional[Path] = None
    ) -> List[Pattern]:
        """
        Identify optimization patterns via LLM analysis.

        Process:
        1. Prepare structured prompt with code + profiling
        2. LLM identifies potential patterns
        3. Validate patterns against known constraints
        4. Return candidate patterns (not yet rules)
        """
```

---

### 2.4 L4: Execution Layer

**Purpose**: Apply optimizations and validate results.

#### 2.4.1 Code Transformer

**Responsibility**: Apply code modifications based on suggestions.

```python
class CodeTransformer:
    """
    Apply code transformations safely.

    Modes:
    - Patch-based: Apply specific line changes (deterministic)
    - LLM-guided: Use LLM to apply pattern (non-deterministic)
    """

    def apply(
        self,
        original_code: str,
        suggestion: Suggestion,
        mode: str = "patch"
    ) -> TransformResult:
        """
        Apply transformation and return result.

        Safety:
        - Validate syntax after transformation
        - Check for broken dependencies
        - Preserve original as backup
        """
```

#### 2.4.2 Builder

**Responsibility**: Compile and install modified operator.

```python
class OperatorBuilder:
    """
    Build and install AscendC operator.

    Wraps: build_operator.py functionality
    """

    def build(self, op_dir: Path) -> BuildResult:
        """
        Compile operator and install .run package.

        Returns: Success/failure + error logs
        """
```

#### 2.4.3 Validator

**Responsibility**: Verify correctness and performance improvement.

```python
class OptimizationValidator:
    """
    Validate optimization results.

    Two-stage validation:
    1. Correctness: Does output match baseline?
    2. Performance: Does profiling show improvement?
    """

    def validate(
        self,
        baseline: Baseline,
        optimized: ProfilingResult,
        goal: GoalConfig
    ) -> ValidationResult:
        """
        Validate against baseline and goal.

        Returns:
        - Correctness: Pass/Fail
        - Performance delta: +44%, -2%, etc.
        - Goal met: True/False
        """
```

---

### 2.5 L5: Knowledge Layer

**Purpose**: Store and evolve optimization knowledge.

#### 2.5.1 Rule Library

**Structure**:
```
assets/rules/
├── core/                    # Core rules (manually curated)
│   ├── R_API_*/             # API-level optimizations
│   ├── R_ARCH_*/            # Architecture-specific
│   └── R_PATTERN_*/         # General patterns
├── learned/                 # Auto-generated rules
│   └── R_LEARNED_*/
└── experimental/            # Unvalidated patterns
    └── R_EXP_*/
```

**Rule Metadata**:
```yaml
# R_XXX/metadata.yaml
rule_id: R_API_VECTOR_COUNTER_MODE
category: API
confidence: high
success_count: 42        # Validated applications
failure_count: 3
avg_improvement: 0.55    # 55% average speedup
tags: [U.Vector, O.Elementwise, S.LowVecUtil]
required_tags: []
conflicts: [R_API_EXPLICIT_LOOP]
```

#### 2.5.2 Case Library

**Purpose**: Store validated optimization examples for training and regression testing.

```
assets/cases/
├── fastgelu_vector_counter/
│   ├── baseline/
│   │   ├── code.cpp
│   │   └── profiling.csv
│   ├── optimized/
│   │   ├── code.cpp
│   │   └── profiling.csv
│   └── metadata.json    # Rule applied, improvement, tags
```

#### 2.5.3 Performance Models

**Purpose**: Build predictive models for optimization impact.

```python
class PerformanceModel:
    """
    Predict optimization impact before applying.

    Uses: Historical case data + profiling features
    """

    def predict_improvement(
        self,
        rule: Rule,
        operator_features: OperatorFeatures
    ) -> PredictedImprovement:
        """
        Predict: Speedup range, confidence interval
        """
```

---

## 3. Data Models

### 3.1 Core Data Types

```python
# Input Data
@dataclass
class OperatorContext:
    """Complete operator context."""
    op_name: str
    op_dir: Path
    code: str
    profiling: ProfilingData
    tags: OperatorTags
    goal: GoalConfig

@dataclass
class OperatorTags:
    """Structured operator tags."""
    domain: List[str]      # U.*, O.*, T.*
    symptom: List[str]     # S.*
    context: List[str]     # C.*

@dataclass
class GoalConfig:
    """Performance optimization goal."""
    relative_improvement: float  # 0.2 = 20%
    absolute_metrics: Dict[str, float]
    max_iterations: int
    consecutive_failures: int

# Analysis Results
@dataclass
class RuleMatch:
    """Rule matching result."""
    rule: Rule
    score: float
    coverage: float
    matched_tags: List[str]
    missing_tags: List[str]

@dataclass
class Suggestion:
    """Optimization suggestion."""
    id: str
    rule_name: str
    priority: str          # High/Medium/Low
    problem: str           # Problem diagnosis
    solution: str          # Optimization approach
    code_changes: List[CodeChange]
    expected_improvement: str
    evidence: Evidence

@dataclass
class CodeChange:
    """Specific code modification."""
    file: str
    line_start: int
    line_end: int
    old_code: str
    new_code: str
    description: str

# Execution Results
@dataclass
class OptimizationResult:
    """Result of applying an optimization."""
    success: bool
    correctness: bool
    performance_delta: float  # +0.44 = 44% faster
    baseline_duration: float
    optimized_duration: float
    goal_met: bool
    error_message: Optional[str]
```

---

## 4. Module Organization (Refactored)

### 4.1 Proposed Directory Structure

```
code-performance-advisor/
├── advisor.py                    # NEW: Main entry point (replaces scattered scripts)
├── SKILL.md                      # Skill metadata for Claude Code
├── README.md                     # User-facing documentation
│
├── core/                         # NEW: Core system modules
│   ├── __init__.py
│   ├── orchestration/            # L2: Orchestration
│   │   ├── advisor.py            # Main controller
│   │   ├── router.py             # Phase routing
│   │   ├── state_manager.py     # State persistence
│   │   └── iterator.py           # Iteration controller
│   ├── analysis/                 # L3: Analysis
│   │   ├── rule_matcher.py       # Rule matching
│   │   ├── evidence_extractor.py # Evidence extraction
│   │   ├── suggestion_generator.py
│   │   └── pattern_recognizer.py # LLM-based
│   ├── execution/                # L4: Execution
│   │   ├── code_transformer.py
│   │   ├── builder.py
│   │   └── validator.py
│   ├── knowledge/                # L5: Knowledge
│   │   ├── rule_library.py
│   │   ├── case_library.py
│   │   └── performance_model.py
│   └── common/                   # Shared utilities
│       ├── data_loader.py
│       ├── tag_extractor.py
│       ├── profiling_parser.py
│       └── config.py
│
├── assets/                       # Static resources
│   ├── rules/
│   │   ├── core/                 # Curated rules
│   │   ├── learned/              # Auto-generated
│   │   └── experimental/         # Unvalidated
│   ├── cases/                    # Validated examples
│   └── templates/                # Markdown templates
│
├── cli/                          # NEW: CLI implementation
│   ├── __init__.py
│   ├── commands/
│   │   ├── analyze.py
│   │   ├── suggest.py
│   │   ├── apply.py
│   │   ├── verify.py
│   │   ├── optimize.py
│   │   └── rule.py
│   └── utils/
│       └── formatters.py
│
├── workspace/                    # Runtime data
│   ├── operators/                # CHANGED: Per-operator workspaces
│   │   └── {op_name}/
│   │       ├── input/            # Code + profiling
│   │       ├── baseline/         # Initial snapshot
│   │       ├── iterations/       # Iteration history
│   │       ├── output/           # Final results
│   │       └── state.json        # Persistent state
│   └── global/
│       └── index.json            # Rule index cache
│
├── tests/                        # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/                         # NEW: Documentation
│   ├── architecture/             # This file + diagrams
│   ├── user_guide/               # User documentation
│   ├── developer_guide/          # Development docs
│   └── api_reference/            # API docs
│
└── scripts/                      # DEPRECATED: Legacy scripts
    └── migration/                # Migration tools
```

### 4.2 Module Responsibility Matrix

| Module | Responsibility | Dependencies | Output |
|--------|---------------|--------------|--------|
| `advisor.py` | Main entry, CLI parsing | `core.orchestration` | User interaction |
| `core.orchestration.advisor` | Workflow orchestration | All core modules | OptimizationResult |
| `core.analysis.rule_matcher` | Rule scoring | `knowledge.rule_library` | RuleMatch list |
| `core.analysis.suggestion_generator` | Suggestion creation | `rule_matcher`, `evidence_extractor` | Suggestion |
| `core.execution.builder` | Compile operator | External (CANN) | BuildResult |
| `core.execution.validator` | Validate improvement | `profiling_parser` | ValidationResult |
| `core.knowledge.rule_library` | Rule CRUD | `assets/rules/` | Rule objects |

---

## 5. Workflow Redesign

### 5.1 End-to-End Optimization Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ User: advisor optimize fastgelu --mode interactive              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0: Initialization                                         │
│ - Load operator code + profiling                                │
│ - Extract tags (U.Vector, O.Elementwise, S.LowVecUtil, etc.)   │
│ - Load performance goal (default: 20% improvement)              │
│ - Score rules against tags                                      │
│ - Router decides: Fast Path (score=0.625)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1-3: Suggestion Generation                                │
│ - Match: R_API_VECTOR_COUNTER_MODE (score 0.625)                │
│ - Extract evidence: aiv_vec_ratio=0.133, scalar_ratio=0.498     │
│ - Generate suggestion:                                          │
│   * Problem: Scalar overhead from explicit loops                │
│   * Solution: Use COUNTER mode to eliminate loop                │
│   * Code changes: 15 lines (specific diffs)                     │
│   * Expected: 40-60% speedup                                    │
│ - Display to user                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ USER CHECKPOINT: Apply suggestion #1? [Y/n]                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (Y)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Execution                                              │
│ - Apply code transformation                                     │
│ - Build operator (compile + install)                            │
│ - Run profiling                                                 │
│ - Validate:                                                     │
│   * Correctness: ✓ Output matches                               │
│   * Performance: ✓ 44% faster (goal: 20%)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Knowledge Capture                                      │
│ - Save case to case library                                     │
│ - Update rule success_count                                     │
│ - Update performance model                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ REPORT: Optimization complete! ✓                                │
│ - Improvement: 44% (goal: 20%) ✓                                │
│ - Rule applied: R_API_VECTOR_COUNTER_MODE                       │
│ - Before: 8.1us → After: 4.5us                                  │
│ - Report: workspace/operators/fastgelu/report.md                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Command Simplification

**Before** (current state):
```bash
# User has to manually execute multiple steps
python scripts/analysis_engine/init_workspace.py --op fastgelu
python scripts/analysis_engine/cli.py score --tag-file ...
# Read JSON manually
# Ask LLM Agent to generate suggestions
# Manually apply changes
python scripts/analysis_engine/build_operator.py --op fastgelu
# Manually run profiling
# Manually verify
```

**After** (proposed):
```bash
# Single command for end-to-end optimization
advisor optimize fastgelu --mode interactive

# Or step-by-step for debugging
advisor analyze fastgelu              # Phase 0
advisor suggest fastgelu              # Generate suggestions
advisor apply fastgelu --suggestion 1 # Apply specific suggestion
advisor verify fastgelu               # Validate
```

---

## 6. Configuration Management

### 6.1 Configuration Hierarchy

```
1. System defaults (hardcoded in core/common/config.py)
2. Global config (~/.advisor/config.yaml)
3. Project config (workspace/advisor.yaml)
4. Operator config (workspace/operators/{op}/config.yaml)
5. Command-line arguments (highest priority)
```

### 6.2 Configuration Schema

```yaml
# workspace/advisor.yaml (example)
version: "1.0"

# Global settings
max_iterations: 5
consecutive_failures: 2
default_goal:
  relative_improvement: 0.2  # 20%

# Analysis settings
analysis:
  rule_score_threshold: 0.3
  max_suggestions: 3
  exclude_experimental_rules: false

# Execution settings
execution:
  mode: interactive  # interactive | auto | suggest-only
  auto_backup: true
  parallel_build: false

# Profiling settings
profiling:
  warmup_iterations: 3
  test_iterations: 10
  timeout_seconds: 300

# LLM settings (when using pattern recognizer)
llm:
  model: "claude-sonnet-4-5"
  max_tokens: 4000
  temperature: 0.0

# Logging
logging:
  level: INFO
  file: workspace/advisor.log
```

---

## 7. Migration Plan

### 7.1 Phase A: Core Infrastructure (Week 1)

**Goal**: Establish new architecture without breaking existing functionality.

**Tasks**:
1. Create `core/` directory structure
2. Implement core data models (`core/common/data_models.py`)
3. Refactor existing code into new modules:
   - Move CLI logic → `core/orchestration/`
   - Move rule matching → `core/analysis/rule_matcher.py`
   - Move goal_loader → `core/common/config.py`
4. Create `advisor.py` main entry point
5. Implement basic `advisor analyze` command

**Success Criteria**:
- `advisor analyze fastgelu` produces same output as old CLI
- All existing tests pass
- No breaking changes to workspace layout

### 7.2 Phase B: Suggestion Pipeline (Week 2)

**Goal**: Complete suggestion generation automation.

**Tasks**:
1. Implement `core/analysis/suggestion_generator.py` (complete B1.2-B1.5 from previous iteration)
2. Implement `core/analysis/evidence_extractor.py`
3. Create suggestion templates
4. Implement `advisor suggest` command
5. Integration testing with fastgelu

**Success Criteria**:
- `advisor suggest fastgelu` generates high-quality suggestions automatically
- Output matches manual baseline in quality
- Execution time < 10 seconds

### 7.3 Phase C: Execution Pipeline (Week 3)

**Goal**: Automate apply → build → validate cycle.

**Tasks**:
1. Implement `core/execution/code_transformer.py`
2. Implement `core/execution/builder.py` (wrap existing build_operator.py)
3. Implement `core/execution/validator.py`
4. Implement `advisor apply` and `advisor verify` commands
5. State management for iteration tracking

**Success Criteria**:
- `advisor apply fastgelu --suggestion 1` successfully applies transformation
- Build and profiling automated
- Validation compares against baseline correctly

### 7.4 Phase D: End-to-End Integration (Week 4)

**Goal**: Complete optimization loop.

**Tasks**:
1. Implement `core/orchestration/iterator.py`
2. Implement `advisor optimize` command
3. Knowledge capture (`core/knowledge/case_library.py`)
4. Report generation
5. E2E testing with multiple operators

**Success Criteria**:
- `advisor optimize fastgelu --mode auto` runs without user intervention
- Successfully optimizes test operators
- Knowledge captured in case library

### 7.5 Phase E: Cleanup & Documentation (Week 5)

**Goal**: Remove deprecated code, complete documentation.

**Tasks**:
1. Deprecate old `scripts/analysis_engine/` (move to `scripts/legacy/`)
2. Update all documentation
3. Create migration guide
4. Create user guide with examples
5. Code cleanup and linting

---

## 8. Design Principles & Trade-offs

### 8.1 Core Principles

1. **Explicit over Implicit**
   - Clear data flow, no hidden state
   - Explicit dependencies in module interfaces

2. **Verifiable over Fast**
   - Every optimization must be validated
   - Performance over developer convenience

3. **Modular over Monolithic**
   - Each module has single responsibility
   - Modules communicate via well-defined interfaces

4. **Automatable over Manual**
   - Automate deterministic tasks
   - Reserve LLM for truly non-deterministic tasks

5. **Evolvable over Static**
   - Knowledge base grows with each validation
   - System improves through usage

### 8.2 Key Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| 5-layer architecture | Complexity vs Flexibility | Clear separation enables testing, swapping components |
| State persistence | Disk I/O vs Robustness | Enables pause/resume, worth the overhead |
| Interactive mode default | Speed vs Safety | User confirmation prevents bad optimizations |
| Rule library separation (core/learned) | Organization vs Simplicity | Clear distinction between curated and experimental |
| Python-only CLI (no subprocesses) | Startup time vs Maintainability | Easier debugging, dependency management |

---

## 9. Success Metrics

### 9.1 System Performance

- **Suggestion Generation**: < 10 seconds for Fast Path
- **End-to-End Optimization**: < 5 minutes (auto mode, single iteration)
- **Rule Matching Accuracy**: > 90% for high-confidence rules

### 9.2 Code Quality

- **Test Coverage**: > 80%
- **Module Coupling**: Each module imports ≤ 3 other core modules
- **Documentation**: All public APIs documented

### 9.3 User Experience

- **Learning Curve**: New user can optimize first operator in < 10 minutes
- **Error Recovery**: All errors provide actionable suggestions
- **Transparency**: User can trace why each suggestion was made

---

## 10. Future Extensions

### 10.1 Near-term (Q2 2026)

- **Multi-operator optimization**: Optimize multiple operators in batch
- **Performance model training**: Use case library to predict improvements
- **Web UI**: Visual interface for exploration and comparison

### 10.2 Long-term (Q3+ 2026)

- **Collaborative rule development**: Team-based rule curation
- **Cross-architecture support**: GPU, other accelerators
- **CI/CD integration**: Automated regression testing

---

## Appendix A: Glossary

- **Advisor**: Main orchestration component
- **Rule**: Expert knowledge encoded as pattern + transformation
- **Suggestion**: Proposed optimization with evidence
- **Case**: Validated example (baseline + optimized + metadata)
- **Evidence**: Quantitative data from profiling supporting a suggestion
- **Tag**: Structured label (U.Vector, S.LowVecUtil, etc.)
- **Phase**: Stage in optimization workflow (0-5)
- **Route**: Analysis path decision (Fast/Moderate/Deep)

---

## Appendix B: Anti-patterns to Avoid

1. **God Object**: Don't let Advisor do everything; delegate to specialized modules
2. **Leaky Abstractions**: Don't expose implementation details in interfaces
3. **Magic Strings**: Use enums/constants for paths, commands, states
4. **Silent Failures**: Always log and report errors
5. **Premature Optimization**: Prioritize correctness and clarity first
6. **LLM Overuse**: Don't use LLM for tasks that can be automated deterministically
7. **State Mutation**: Prefer immutable data structures; explicit state transitions

---

**END OF ARCHITECTURE DOCUMENT**

**Next Steps**:
1. Review and approve this architecture
2. Begin Phase A implementation (Week 1)
3. Create detailed API specifications for each module
4. Set up project tracking (issues, milestones)
