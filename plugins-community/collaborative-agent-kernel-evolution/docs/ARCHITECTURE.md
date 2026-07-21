# CAKE2 Architecture

This document provides a detailed technical overview of CAKE2's architecture, design decisions, and implementation details.

## System Architecture Overview

CAKE2 follows a multi-stage pipeline architecture in local mode, requiring a CANN environment for compilation.

```
┌─────────────────────────────────────────────────────────────┐
│                     CAKE2 System                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐                                           │
│  │   Local      │                                           │
│  │  Machine     │                                           │
│  │  (NPU+CANN)  │                                           │
│  └──────────────┘                                           │
│         │                                                    │
│         │                                                    │
│    ┌────▼─────┐                                             │
│    │   LLM    │                                             │
│    │  Agent   │                                             │
│    └────┬─────┘                                             │
│         │                                                    │
│    ┌────▼─────┐                                             │
│    │   CANN   │                                             │
│    │ Compiler │                                             │
│    └──────────┘                                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent System (Claude Code SDK)

CAKE2 uses Claude Code's agent framework with specialized agents:

#### cake Agent (Local Mode)
- **Purpose**: Single operator generation with full pipeline
- **Mode**: Local only (CANN environment required)
- **Skills Used**:
  - `op_desc_generation`: Generate operator description JSON
  - `reference_generation`: Create PyTorch reference
  - `functional_conversion`: Convert to functional API
  - `dsl_baseline_generation`: Generate DSL code
  - `ascend_call_generation`: Generate AscendC
  - `ascendc_evaluation`: Evaluate operator

#### cake_evo Agent (Evolution Mode)
- **Purpose**: Multi-round parallel optimization
- **Mode**: Local only (CANN environment required)
- **Process**:
  1. Generate N variants in parallel (Round 1)
  2. Evaluate and classify (Good/Medium/Poor)
  3. Select inspirations using tiered sampling
  4. Generate N new variants with inspirations (Round 2)
  5. Repeat for max_rounds
  6. Return best implementations

### 2. Generation Pipeline

The pipeline transforms natural language to AscendC code through multiple stages:

```
Natural Language Description
         ↓
┌────────────────────────┐
│ 1. Operator Description│  → JSON schema with inputs/outputs/formula
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 2. PyTorch Reference   │  → Functional PyTorch implementation
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 3. Functional API      │  → Convert nn.Module to functional style
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 4. DSL Baseline        │  → High-level DSL (AscendDSL)
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 5. AscendC Generation  │  → Low-level AscendC kernel
│    (4 Passes)          │     Pass 1: Structure
│                        │     Pass 2: Core logic
│                        │     Pass 3: Optimization
│                        │     Pass 4: Polish
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 6. Project Creation    │  → CMake project with host/kernel code
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 7. Local Compilation   │  → Build .so library with CANN
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 8. Evaluation          │  → Correctness + Performance testing
└────────────────────────┘
```

### 3. Evolution System

The evolution module implements parallel exploration with tiered sampling:

```python
┌─────────────────────────────────────────────────────────┐
│              Evolution Orchestrator                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Round 1: Initial Exploration                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Subagent1│ │Subagent2│ │Subagent3│ │Subagent4│ ...   │
│  │(Full    │ │(Full    │ │(Full    │ │(Full    │       │
│  │Pipeline)│ │Pipeline)│ │Pipeline)│ │Pipeline)│       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │           │             │
│       └───────────┴───────────┴───────────┘             │
│                       │                                  │
│                       ▼                                  │
│              ┌─────────────────┐                         │
│              │   Evaluation    │                         │
│              │   & Ranking     │                         │
│              └────────┬────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              ┌─────────────────┐                         │
│              │  Classification │                         │
│              │  Good: 30%      │                         │
│              │  Medium: 40%    │                         │
│              │  Poor: 30%      │                         │
│              └────────┬────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              ┌─────────────────┐                         │
│              │Tiered Sampling  │                         │
│              │(Select 2-3      │                         │
│              │ inspirations)   │                         │
│              └────────┬────────┘                         │
│                       │                                  │
│  Round 2: Guided Exploration                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                    │
│  │Subagent1│ │Subagent2│ │Subagent3│ ...                │
│  │(with    │ │(with    │ │(with    │                    │
│  │inspire) │ │inspire) │ │inspire) │                    │
│  └─────────┘ └─────────┘ └─────────┘                    │
│                                                           │
│  ... (repeat for max_rounds)                             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

#### EvoOrchestrator
- **Responsibility**: Coordinate multi-round evolution
- **Methods**:
  - `run_evolution()`: Main entry point
  - `_run_round()`: Execute one round
  - `_classify_implementations()`: Tier implementations
  - `_select_inspirations()`: Sample from tiers

#### SubagentExecutor
- **Responsibility**: Execute parallel subagents
- **Methods**:
  - `execute_parallel()`: Run N subagents concurrently
  - `_execute_single()`: Run one subagent with full pipeline

#### EvoData
- **Data Structures**:
  - `ImplementationResult`: Single implementation result
  - `RoundResult`: Results from one round
  - `EvolutionResult`: Complete evolution results

### 4. Skill System

CAKE2 uses Claude Code's skill system for modular functionality:

```
.claude/skills/
├── op_desc_generation/          # Generate operator description JSON
│   ├── skill.md
│   └── templates/
│       └── op_desc_template.json
│
├── reference_generation/        # Generate PyTorch reference
│   ├── skill.md
│   └── scripts/
│       └── generate_reference.py
│
├── functional_conversion/       # Convert to functional API
│   ├── skill.md
│   └── scripts/
│       └── convert_functional.py
│
├── dsl_baseline_generation/     # Generate DSL baseline
│   ├── skill.md
│   └── scripts/
│       └── generate_dsl.py
│
├── dsl_lowering/                # DSL → AscendC (local)
│   ├── skill.md
│   └── scripts/
│       └── lower_dsl.py
│
├── ascend_call_generation/      # Generate AscendC project (local)
│   ├── skill.md
│   └── scripts/
│       └── generate_project.py
│
└── ascendc_evaluation/          # Evaluate operator (local)
    ├── skill.md
    └── scripts/
        └── evaluate.py
```

Each skill is self-contained with:
- **skill.md**: Skill definition and instructions
- **scripts/**: Python scripts for execution
- **templates/**: Templates for code generation

## Data Flow

### Local Mode Data Flow

```
User Input
    ↓
Agent (LLM)
    ↓
Skill: op_desc_generation → op_desc.json
    ↓
Skill: reference_generation → reference.py
    ↓
Skill: functional_conversion → functional.py
    ↓
Skill: dsl_baseline_generation → dsl.py
    ↓
Skill: dsl_lowering → ascendc_kernel.cpp (4 passes)
    ↓
Skill: ascend_call_generation → CMake project
    ↓
Local CANN Compiler → custom_op.so
    ↓
Skill: ascendc_evaluation → correctness + performance
    ↓
Results to User
```

## Key Design Patterns

### 1. Pipeline Pattern
Each stage transforms input to output, enabling:
- **Modularity**: Each stage is independent
- **Debugging**: Inspect intermediate outputs
- **Flexibility**: Skip or replace stages as needed

### 2. Factory Pattern
Skill creation for local mode:
```python
skill = "ascend_call_generation"  # Local compilation only
```

### 3. Observer Pattern
Evolution system observes subagent results:
- Subagents execute independently
- Orchestrator collects and analyzes results
- Next round adapts based on observations

## Performance Optimizations

### 1. Parallel Execution
- **Evolution**: N subagents run concurrently
- **Local Compilation**: Multiple threads utilized

### 2. Caching
- **LLM Responses**: Cached by Claude Code SDK
- **Compilation**: Reuse compiled operators when possible

### 3. Resource Management
- **NPU Devices**: Efficient device utilization
- **Memory**: Automatic cleanup of temp files

## Error Handling

### Compilation Errors
```python
try:
    result = compile_operator(...)
except CompilationError as e:
    log_error(e)
    return error_response(e)
```

### Device Errors
```python
try:
    device = allocate_npu_device()
except NoDeviceAvailable:
    queue_request()
    wait_for_device()
```

## Security Considerations

### 1. Resource Limits
- **Max Workers**: Limit concurrent processes
- **Timeouts**: Prevent resource exhaustion
- **Disk Quotas**: Limit temp file usage

### 2. Isolation
- **Temp Directories**: Each compilation gets unique directory
- **Python Environments**: Isolated venv for each compilation

## Testing Strategy

### Unit Tests
- **Evolution Module**: `evolution/test_evolution.py`
- **Local Compilation**: `tests/test_local_compilation.py`
- **Project Generation**: `tests/test_project_generation.py`

### Integration Tests
- **End-to-End**: Full pipeline tests
- **Evolution**: Multi-round optimization

### Performance Tests
- **Benchmarking**: Compare against PyTorch
- **Resource Usage**: Monitor memory and device utilization

## Monitoring and Logging

### Local Logs
```
output/local/compile.log
```

### Metrics
- **Compilation Time**: Time to build operator
- **Evaluation Time**: Time to run tests
- **Speedup**: Performance vs baseline
- **Success Rate**: Percentage of successful compilations

## Future Enhancements

### Planned Features
1. **Advanced Caching**: Enhanced compilation caching
2. **Multi-device Support**: Operators spanning multiple NPUs
3. **Auto-tuning**: Automatic hyperparameter optimization

### Research Directions
1. **Transfer Learning**: Learn from previous operators
2. **Meta-learning**: Optimize evolution strategy itself
3. **Neural Architecture Search**: Discover novel operator designs
4. **Formal Verification**: Prove correctness properties
