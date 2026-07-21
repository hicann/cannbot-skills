# Code Performance Advisor - API Specification & Migration Guide

**Version**: 2.0
**Date**: 2026-02-25
**Related**: ARCHITECTURE.md, DIAGRAMS.md

---

## Part I: Core API Specification

### 1. Data Models API

All core data types with their Python type definitions.

#### 1.1 Input Models

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

@dataclass
class GoalConfig:
    """Performance optimization goal configuration."""
    relative_improvement: float = 0.2  # 20% default
    absolute_metrics: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5
    consecutive_failures: int = 2
    notes: str = ""

    def meets_goal(self, improvement: float) -> bool:
        """Check if improvement meets goal."""
        return improvement >= self.relative_improvement

@dataclass
class OperatorTags:
    """Structured operator tags."""
    domain: List[str]      # U.*, O.*, T.*
    symptom: List[str]     # S.*
    context: List[str]     # C.*

    @property
    def all_tags(self) -> List[str]:
        """Return flattened list of all tags."""
        return self.domain + self.symptom + self.context

@dataclass
class ProfilingData:
    """Profiling metrics from CSV."""
    operator_name: str
    task_duration_us: float
    aiv_vec_ratio: float
    aiv_scalar_ratio: float
    aiv_vec_time_us: float
    aiv_scalar_time_us: float
    raw_data: Dict[str, Any]  # Full CSV row

    @property
    def bottleneck_type(self) -> str:
        """Identify bottleneck: scalar/vector/memory."""
        if self.aiv_scalar_ratio > 0.4:
            return "scalar"
        elif self.aiv_vec_ratio < 0.3:
            return "vector_underutilized"
        else:
            return "balanced"

@dataclass
class OperatorContext:
    """Complete operator context for analysis."""
    op_name: str
    op_dir: Path
    code: str
    profiling: ProfilingData
    tags: OperatorTags
    goal: GoalConfig

    @classmethod
    def from_directory(cls, op_dir: Path) -> 'OperatorContext':
        """Factory: Load context from operator directory."""
        # Implementation in core/common/data_loader.py
        pass
```

#### 1.2 Analysis Models

```python
@dataclass
class Rule:
    """Optimization rule metadata."""
    rule_id: str
    rule_path: Path
    category: str          # API, ARCH, PATTERN
    confidence: str        # high, medium, low
    tags: List[str]
    required_tags: List[str]
    conflicts: List[str]   # Conflicting rule IDs

    # Performance tracking
    success_count: int = 0
    failure_count: int = 0
    avg_improvement: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

@dataclass
class RuleMatch:
    """Result of rule matching."""
    rule: Rule
    score: float           # 0.0-1.0
    coverage: float        # 0.0-1.0 (required tags coverage)
    matched_tags: List[str]
    missing_tags: List[str]
    conflict: bool

    @property
    def confidence_level(self) -> str:
        """Map score to confidence level."""
        if self.score >= 0.7:
            return "high"
        elif self.score >= 0.5:
            return "medium"
        else:
            return "low"

@dataclass
class Evidence:
    """Quantitative evidence from profiling."""
    bottleneck_type: str   # scalar/vector/memory
    key_metrics: Dict[str, float]
    thresholds_violated: List[str]
    comparative_deltas: Optional[Dict[str, float]] = None

    def to_markdown(self) -> str:
        """Render evidence as markdown."""
        # Implementation in core/analysis/evidence_extractor.py
        pass

class RoutingDecision(Enum):
    """Analysis path decision."""
    FAST_PATH = "fast"           # High confidence rules
    MODERATE_PATH = "moderate"   # Medium confidence
    DEEP_PATH = "deep"           # Low/no rule match

@dataclass
class Suggestion:
    """Optimization suggestion."""
    id: str                # Unique identifier
    rule_name: str
    priority: str          # High/Medium/Low
    score: float

    # Content
    problem: str           # Problem diagnosis (markdown)
    solution: str          # Optimization approach
    code_changes: List['CodeChange']
    expected_improvement: str

    # Supporting data
    evidence: Evidence
    matched_tags: List[str]

    # Metadata
    attempted: bool = False
    applied_at: Optional[str] = None

@dataclass
class CodeChange:
    """Specific code modification."""
    file: str
    line_start: int
    line_end: int
    old_code: str
    new_code: str
    description: str

    def to_diff(self) -> str:
        """Generate unified diff format."""
        # Implementation in core/execution/code_transformer.py
        pass
```

#### 1.3 Execution Models

```python
class OptimizationMode(Enum):
    """Optimization execution mode."""
    INTERACTIVE = "interactive"  # User approves each step
    AUTO = "auto"               # Fully automated
    SUGGEST_ONLY = "suggest"    # Generate suggestions only

@dataclass
class BuildResult:
    """Operator build result."""
    success: bool
    build_dir: Path
    binary_path: Optional[Path]
    error_log: Optional[str]
    build_time_seconds: float

@dataclass
class ValidationResult:
    """Optimization validation result."""
    correctness_passed: bool
    performance_delta: float      # e.g., +0.44 = 44% faster
    baseline_duration: float
    optimized_duration: float
    goal_met: bool

    # Detailed metrics
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]

    error_message: Optional[str] = None

@dataclass
class OptimizationResult:
    """Complete optimization result."""
    suggestion: Suggestion
    build: BuildResult
    validation: ValidationResult
    iteration: int
    timestamp: str

    @property
    def success(self) -> bool:
        """Overall success check."""
        return (
            self.build.success and
            self.validation.correctness_passed and
            self.validation.performance_delta > 0
        )
```

#### 1.4 State Models

```python
@dataclass
class OptimizationState:
    """Persistent optimization state."""
    op_name: str
    current_iteration: int
    mode: OptimizationMode

    # History
    attempted_suggestions: List[str]  # Suggestion IDs
    results: List[OptimizationResult]

    # Baselines
    initial_baseline: 'Baseline'
    current_baseline: 'Baseline'

    # Config
    goal: GoalConfig

    def save(self, path: Path) -> None:
        """Persist state to JSON."""
        # Implementation in core/orchestration/state_manager.py
        pass

    @classmethod
    def load(cls, path: Path) -> 'OptimizationState':
        """Load state from JSON."""
        pass

@dataclass
class Baseline:
    """Immutable baseline snapshot."""
    code: str
    profiling: ProfilingData
    timestamp: str
    hash: str  # SHA256 of code
```

---

### 2. Core Module APIs

#### 2.1 Orchestration API

```python
# core/orchestration/advisor.py

class PerformanceAdvisor:
    """Main orchestrator for performance optimization."""

    def __init__(self, op_name: str, config: Optional[AdvisorConfig] = None):
        """
        Initialize advisor for an operator.

        Args:
            op_name: Operator name
            config: Optional configuration (uses defaults if None)
        """

    def optimize(
        self,
        mode: OptimizationMode = OptimizationMode.INTERACTIVE,
        resume: bool = False
    ) -> OptimizationResult:
        """
        Execute end-to-end optimization.

        Args:
            mode: Execution mode (interactive/auto/suggest-only)
            resume: Resume from previous state if exists

        Returns:
            Final optimization result

        Raises:
            FileNotFoundError: If operator directory not found
            ValidationError: If data validation fails
        """

    def analyze(self) -> RoutingDecision:
        """
        Phase 0: Analyze and determine routing.

        Returns:
            Routing decision with confidence
        """

    def generate_suggestions(
        self,
        top_n: int = 3
    ) -> List[Suggestion]:
        """
        Phase 1-3: Generate optimization suggestions.

        Args:
            top_n: Number of suggestions to generate

        Returns:
            List of suggestions ranked by priority
        """

    def apply_suggestion(
        self,
        suggestion: Suggestion,
        verify: bool = True
    ) -> OptimizationResult:
        """
        Apply a specific suggestion.

        Args:
            suggestion: Suggestion to apply
            verify: Run validation after applying

        Returns:
            Optimization result with validation
        """
```

#### 2.2 Analysis API

```python
# core/analysis/rule_matcher.py

class RuleMatcher:
    """Rule matching engine."""

    def __init__(self, rule_index: RuleIndex):
        """Initialize with rule index."""

    def match(
        self,
        operator_tags: OperatorTags,
        threshold: float = 0.0
    ) -> List[RuleMatch]:
        """
        Match rules against operator tags.

        Args:
            operator_tags: Structured tags
            threshold: Minimum score (0.0-1.0)

        Returns:
            Sorted list of rule matches
        """

    def score_rule(
        self,
        rule: Rule,
        operator_tags: OperatorTags
    ) -> RuleMatch:
        """
        Score a single rule.

        Args:
            rule: Rule to score
            operator_tags: Operator tags

        Returns:
            Match result with score and coverage
        """

# core/analysis/suggestion_generator.py

class SuggestionGenerator:
    """Generate optimization suggestions."""

    def generate(
        self,
        rule_match: RuleMatch,
        context: OperatorContext
    ) -> Suggestion:
        """
        Generate suggestion from matched rule.

        Args:
            rule_match: Matched rule with score
            context: Operator context

        Returns:
            Structured suggestion
        """

    def generate_batch(
        self,
        rule_matches: List[RuleMatch],
        context: OperatorContext,
        top_n: int = 3
    ) -> List[Suggestion]:
        """Generate multiple suggestions."""

# core/analysis/evidence_extractor.py

class EvidenceExtractor:
    """Extract evidence from profiling data."""

    def extract(
        self,
        profiling: ProfilingData,
        rule: Rule
    ) -> Evidence:
        """
        Extract relevant evidence for a rule.

        Args:
            profiling: Profiling metrics
            rule: Rule being applied

        Returns:
            Evidence object with metrics and violations
        """
```

#### 2.3 Execution API

```python
# core/execution/code_transformer.py

class CodeTransformer:
    """Apply code transformations."""

    def apply(
        self,
        original_code: str,
        code_changes: List[CodeChange],
        validate_syntax: bool = True
    ) -> str:
        """
        Apply code changes.

        Args:
            original_code: Original source code
            code_changes: List of changes to apply
            validate_syntax: Check syntax after transformation

        Returns:
            Transformed code

        Raises:
            SyntaxError: If transformed code has syntax errors
        """

# core/execution/builder.py

class OperatorBuilder:
    """Build and install operators."""

    def build(
        self,
        op_dir: Path,
        clean: bool = False
    ) -> BuildResult:
        """
        Compile and install operator.

        Args:
            op_dir: Operator directory
            clean: Clean build (rebuild from scratch)

        Returns:
            Build result with success status
        """

# core/execution/validator.py

class OptimizationValidator:
    """Validate optimization results."""

    def validate(
        self,
        baseline: Baseline,
        optimized_profiling: ProfilingData,
        goal: GoalConfig
    ) -> ValidationResult:
        """
        Validate optimization.

        Args:
            baseline: Baseline snapshot
            optimized_profiling: New profiling data
            goal: Performance goal

        Returns:
            Validation result with deltas
        """
```

#### 2.4 Knowledge API

```python
# core/knowledge/rule_library.py

class RuleLibrary:
    """Manage rule library."""

    def get_rule(self, rule_id: str) -> Rule:
        """Get rule by ID."""

    def list_rules(
        self,
        category: Optional[str] = None,
        confidence: Optional[str] = None
    ) -> List[Rule]:
        """List rules with optional filters."""

    def add_rule(self, rule: Rule) -> None:
        """Add new rule to library."""

    def update_rule(self, rule: Rule) -> None:
        """Update existing rule."""

    def update_success_metrics(
        self,
        rule_id: str,
        success: bool,
        improvement: float
    ) -> None:
        """Update rule performance metrics."""

# core/knowledge/case_library.py

class CaseLibrary:
    """Manage validated optimization cases."""

    def save_case(
        self,
        case_name: str,
        baseline: Baseline,
        optimized: Tuple[str, ProfilingData],
        rule_applied: str,
        improvement: float
    ) -> None:
        """Save validated case."""

    def get_case(self, case_name: str) -> Dict[str, Any]:
        """Load case by name."""

    def list_cases(self, rule_id: Optional[str] = None) -> List[str]:
        """List all cases, optionally filtered by rule."""
```

---

## Part II: Migration Guide

### 1. Overview

**Goal**: Migrate from current scattered implementation to new 5-layer architecture.

**Timeline**: 5 weeks (see ARCHITECTURE.md Section 7)

**Approach**: Incremental migration with parallel operation during transition.

---

### 2. Pre-Migration Checklist

- [ ] Backup entire project
- [ ] Document current CLI usage patterns
- [ ] Identify all external dependencies
- [ ] List all active operator workspaces
- [ ] Create migration branch: `git checkout -b architecture-v2`

---

### 3. Phase A: Core Infrastructure (Week 1)

#### 3.1 Directory Setup

```bash
# Create new directory structure
mkdir -p core/{orchestration,analysis,execution,knowledge,common}
mkdir -p core/orchestration core/analysis core/execution core/knowledge core/common
mkdir -p cli/commands cli/utils
mkdir -p docs/{architecture,user_guide,developer_guide}

# Initialize Python packages
touch core/__init__.py
touch core/orchestration/__init__.py
touch core/analysis/__init__.py
touch core/execution/__init__.py
touch core/knowledge/__init__.py
touch core/common/__init__.py
touch cli/__init__.py
touch cli/commands/__init__.py
```

#### 3.2 Data Models Migration

**File**: `core/common/data_models.py`

**Source**: Copy from API specification above

**Action**:
1. Create `core/common/data_models.py`
2. Copy all dataclass definitions
3. Add imports and dependencies
4. Test with: `python -m pytest tests/unit/test_data_models.py`

#### 3.3 Configuration Migration

**Old**: Scattered config in scripts, hardcoded paths

**New**: `core/common/config.py`

```python
# core/common/config.py

from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class AdvisorConfig:
    """Advisor configuration."""
    # Paths
    root_dir: Path
    workspace_dir: Path
    assets_dir: Path

    # Analysis settings
    rule_score_threshold: float = 0.3
    max_suggestions: int = 3

    # Execution settings
    mode: str = "interactive"
    max_iterations: int = 5

    @classmethod
    def from_file(cls, config_path: Path) -> 'AdvisorConfig':
        """Load from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def defaults(cls) -> 'AdvisorConfig':
        """Return default configuration."""
        root = Path(__file__).resolve().parents[3]
        return cls(
            root_dir=root,
            workspace_dir=root / "workspace",
            assets_dir=root / "assets"
        )
```

**Migration Steps**:
1. Identify all hardcoded paths in scripts
2. Replace with `config.root_dir`, `config.workspace_dir`, etc.
3. Create default `workspace/advisor.yaml`

#### 3.4 Refactor Existing Code

**Goal**: Move existing functionality into new structure without breaking.

**Mapping**:

| Old | New |
|-----|-----|
| `scripts/utils/goal_loader.py` | `core/common/config.py` (GoalConfig) |
| `scripts/analysis_engine/cli.py` (score logic) | `core/analysis/rule_matcher.py` |
| `scripts/analysis_engine/suggest_template.py` | `core/analysis/suggestion_generator.py` |

**Process**:
1. Copy file to new location
2. Update imports
3. Adapt to new data models
4. Add type hints
5. Write unit tests
6. Mark old file as deprecated (add warning)

**Example**:
```python
# scripts/utils/goal_loader.py (OLD - add deprecation warning)

import warnings

warnings.warn(
    "goal_loader is deprecated. Use core.common.config.GoalConfig instead.",
    DeprecationWarning,
    stacklevel=2
)

# ... rest of old code
```

#### 3.5 Create Main Entry Point

**File**: `advisor.py`

```python
#!/usr/bin/env python3
"""
Code Performance Advisor - Main Entry Point

Usage:
    advisor <command> [options]

Commands:
    analyze   - Analyze operator and determine routing
    suggest   - Generate optimization suggestions
    apply     - Apply a specific suggestion
    verify    - Verify optimization improvement
    optimize  - End-to-end optimization loop
    rule      - Manage rules (list, add, update)
    config    - Configuration management

Examples:
    advisor analyze fastgelu
    advisor optimize fastgelu --mode interactive
    advisor rule list --category API
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.commands import (
    analyze,
    suggest,
    apply_cmd,
    verify,
    optimize,
    rule,
    config as config_cmd
)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Performance Advisor",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register commands
    analyze.register(subparsers)
    suggest.register(subparsers)
    apply_cmd.register(subparsers)
    verify.register(subparsers)
    optimize.register(subparsers)
    rule.register(subparsers)
    config_cmd.register(subparsers)

    args = parser.parse_args()

    # Dispatch to command handler
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
```

#### 3.6 Implement First Command: `advisor analyze`

**File**: `cli/commands/analyze.py`

```python
"""Analyze command implementation."""

from pathlib import Path
from core.orchestration.advisor import PerformanceAdvisor
from core.common.config import AdvisorConfig

def register(subparsers):
    """Register analyze command."""
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze operator and determine routing"
    )
    parser.add_argument("op", help="Operator name")
    parser.add_argument(
        "--op-dir",
        help="Operator directory (default: workspace/operators/{op})"
    )
    parser.set_defaults(func=run)

def run(args):
    """Execute analyze command."""
    try:
        # Initialize advisor
        config = AdvisorConfig.defaults()
        advisor = PerformanceAdvisor(args.op, config)

        # Run analysis
        routing = advisor.analyze()

        # Display results
        print(f"\n✅ Analysis complete for '{args.op}'")
        print(f"   Routing decision: {routing.path}")
        print(f"   Confidence: {routing.confidence}")
        print(f"   Recommended analyzer: {routing.analyzer}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
```

**Stub Implementation**:
```python
# core/orchestration/advisor.py (minimal for Week 1)

from core.common.data_models import RoutingDecision
from core.common.config import AdvisorConfig

class PerformanceAdvisor:
    def __init__(self, op_name: str, config: AdvisorConfig):
        self.op_name = op_name
        self.config = config

    def analyze(self) -> RoutingDecision:
        """Stub: Return fast path for now."""
        return RoutingDecision(
            path="fast",
            analyzer="RuleMatcher",
            confidence="high"
        )
```

**Test**:
```bash
python advisor.py analyze fastgelu
# Should output: "Routing decision: fast"
```

#### 3.7 Week 1 Success Criteria

- [ ] New directory structure created
- [ ] Core data models defined and tested
- [ ] Configuration system working
- [ ] `advisor.py` entry point created
- [ ] `advisor analyze` command working (stub)
- [ ] Existing functionality still works via old scripts
- [ ] All existing tests pass

---

### 4. Phase B: Suggestion Pipeline (Week 2)

See ARCHITECTURE.md Section 7.2 for detailed tasks.

**Key Deliverables**:
- Implement `core/analysis/rule_matcher.py`
- Implement `core/analysis/suggestion_generator.py`
- Implement `core/analysis/evidence_extractor.py`
- Implement `cli/commands/suggest.py`
- Complete integration test with fastgelu

**Migration Strategy**:
1. Port `suggest_template.py` → `suggestion_generator.py`
2. Extract evidence logic into `evidence_extractor.py`
3. Create Jinja2 templates
4. Wire up `advisor suggest` command
5. Compare output quality with manual baseline

---

### 5. Phase C-E: See ARCHITECTURE.md

Continue with phases as outlined in ARCHITECTURE.md Section 7.3-7.5.

---

### 6. Migration Validation

After each phase, run validation:

```bash
# Functional validation
python -m pytest tests/integration/test_migration.py

# Performance validation
time advisor analyze fastgelu
# Should be < 5 seconds

# Output quality validation
advisor suggest fastgelu > new_output.md
diff new_output.md workspace/OutputMessages/fastgelu_test_suggestions.md
# Should be similar (allowing for formatting differences)
```

---

### 7. Rollback Plan

If migration encounters critical issues:

```bash
# Revert to old system
git checkout main

# Or keep new structure but use legacy scripts
export ADVISOR_USE_LEGACY=1
python advisor.py analyze fastgelu  # Falls back to old scripts
```

---

### 8. Post-Migration Cleanup

After all phases complete:

1. **Deprecate old scripts**:
   ```bash
   mv scripts/analysis_engine scripts/legacy
   echo "DEPRECATED: Use 'advisor' command instead" > scripts/legacy/README.md
   ```

2. **Update documentation**:
   - Rewrite SKILL.md with new command structure
   - Update README.md with new usage examples
   - Archive old documentation in `docs/archive/`

3. **Remove technical debt**:
   - Delete duplicate code
   - Remove unused imports
   - Clean up workspace/InputMessages structure

4. **Code quality**:
   ```bash
   # Run linters
   black core/ cli/
   pylint core/ cli/
   mypy core/ cli/

   # Check coverage
   pytest --cov=core --cov=cli --cov-report=html
   ```

---

## Part III: Developer Onboarding

### Quick Start for New Developers

```bash
# 1. Clone repo
git clone <repo-url>
cd code-performance-advisor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize workspace
python advisor.py config init

# 4. Run example
python advisor.py analyze fastgelu

# 5. Read architecture
cat docs/architecture/ARCHITECTURE.md
cat docs/architecture/DIAGRAMS.md
```

### Development Workflow

1. **Feature branches**: `git checkout -b feature/new-analyzer`
2. **Write tests first**: `tests/unit/test_new_analyzer.py`
3. **Implement feature**: `core/analysis/new_analyzer.py`
4. **Run tests**: `pytest tests/`
5. **Update docs**: `docs/developer_guide/new_analyzer.md`
6. **Submit PR**: Include tests, docs, examples

### Code Style Guidelines

- **Type hints**: All public functions must have type hints
- **Docstrings**: Google-style docstrings for all classes and functions
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Imports**: Absolute imports from `core`, `cli`
- **Errors**: Raise specific exceptions, not generic `Exception`

---

**END OF API SPECIFICATION & MIGRATION GUIDE**

This document provides the technical details needed to implement and migrate to the new architecture.
