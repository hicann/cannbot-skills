# CAKE2 Project Overview

## Introduction

**CAKE2 (Collaborative Ascend Kernel Evolution 2)** is an advanced AI-powered system for developing high-performance operators for Ascend NPU (Neural Processing Unit) hardware. It automates the entire pipeline from natural language operator descriptions to optimized AscendC kernel implementations.

## Project Purpose

CAKE2 simplifies and accelerates the development of custom operators for Ascend NPUs by:

1. **Automating Code Generation**: Converts natural language descriptions into production-ready AscendC code
2. **Evolutionary Optimization**: Uses parallel exploration and multi-round evolution to discover optimal implementations
3. **Local Compilation**: Direct compilation using local CANN environment
4. **Performance Validation**: Automatically tests correctness and measures performance against PyTorch baselines

## Key Features

### 🚀 Core Capabilities

- **Natural Language Input**: Describe operators in plain English/Chinese
- **Multi-Stage Pipeline**: PyTorch Reference → Functional API → DSL → AscendC
- **Evolutionary Generation**: Parallel kernel variants with tiered sampling (30%/40%/30%)
- **Local Mode**: Direct compilation using CANN environment
- **Automatic Evaluation**: Correctness verification and performance benchmarking

### 🏗️ Architecture Mode

#### Local Mode
- **Requirements**: CANN 8.3+, NPU hardware, Python 3.10+
- **Use Case**: Direct development on NPU-equipped machines
- **Advantages**: Full control, no network dependencies

## Project Structure

```
CAKE2/                             # Claude Code plugin
├── .claude-plugin/
│   └── plugin.json                # Plugin manifest
│
├── agents/                        # Agent definitions
│   ├── cake.md                    # Standard CAKE2 agent
│   ├── cake-evo.md                # Evolution agent
│   └── cake-partial.md            # Partial generation agent
│
├── skills/                        # 20+ specialized skills
│   ├── cake-evo/                  # Evolution skill (with references + scripts)
│   │   ├── SKILL.md
│   │   ├── references/meta_prompts/  # Optimization strategies
│   │   └── scripts/               # eval_op.sh, build_strategy_index.py
│   ├── ascendc-evaluation/        # Compilation & evaluation
│   ├── cake-code-review/       # Code review & auto-fix
│   ├── dsl-lowering/              # DSL → AscendC transformation
│   └── ...                        # Other pipeline skills
│
├── docs/                          # Documentation
├── hooks/                         # Claude Code hooks + linters (Claude Code only; see hooks/README.md)
├── tests/                         # Test suite
├── README.md
└── .pre-commit-config.yaml
```

## Supported Operator Types

CAKE2 can generate the following operator categories:

- **Activation Functions**: ReLU, GELU, FastGELU, Swish, etc.
- **Pooling Operations**: Average pooling, max pooling, adaptive pooling
- **Reduction Operations**: Sum, mean, max, min along dimensions
- **Normalization**: LayerNorm, BatchNorm, RMSNorm
- **Loss Functions**: MSE, CrossEntropy, etc.
- **Matrix Operations**: MatMul, BatchMatMul, transpose
- **Convolution**: Conv2D, DepthwiseConv, etc.
- **Element-wise Math**: Add, multiply, divide, power, etc.
- **Fused Operations**: AddLayerNorm, GeluBackward, etc.

## Technology Stack

### Local Environment
- **Python 3.10+**: Core language
- **Claude Code SDK**: AI agent framework
- **CANN 8.3+**: Ascend Computing Architecture
- **torch_npu 2.9.0+**: PyTorch NPU backend
- **numpy, scipy**: Numerical computing

## Workflow Overview

### Standard Generation (cake agent)

```
User Description
    ↓
1. Operator Description (JSON)
    ↓
2. PyTorch Reference Implementation
    ↓
3. Functional API Conversion
    ↓
4. DSL Baseline Generation
    ↓
5. AscendC Code Generation (4 transformation passes)
    ↓
6. Project Creation & Local Compilation
    ↓
7. Evaluation (Correctness + Performance)
    ↓
Result: Speedup metrics, correctness validation
```

### Evolutionary Generation (cake-evo agent)

```
User Description
    ↓
Round 1: Generate N parallel variants
    ↓
Evaluate & Classify (Good/Medium/Poor)
    ↓
Select Inspirations (30%/40%/30% tiered sampling)
    ↓
Round 2: Generate N new variants with inspirations
    ↓
... (repeat for max_rounds)
    ↓
Result: Best implementations ranked by speedup
```

## Key Concepts

### Tiered Sampling
Inspired by CAKE paper, implementations are classified into three tiers:
- **Good (30%)**: Top performers, used as positive examples
- **Medium (40%)**: Average performers, provide diverse approaches
- **Poor (30%)**: Low performers, help avoid bad patterns

### Transformation Passes
AscendC code generation uses 4 sequential passes:
1. **Pass 1**: Basic structure and memory allocation
2. **Pass 2**: Core computation logic
3. **Pass 3**: Optimization (tiling, vectorization)
4. **Pass 4**: Final polish and edge cases

### Local Compilation
Direct compilation using CANN environment:
- **Performance**: No network latency
- **Control**: Full access to compilation process
- **Debugging**: Direct access to build logs

## Performance Characteristics

Typical speedups achieved:
- **Element-wise operations**: 2-5x vs PyTorch
- **Reduction operations**: 1.5-3x vs PyTorch

## Use Cases

1. **Research**: Rapid prototyping of custom operators for novel architectures
2. **Production**: Optimizing critical operators in deployed models
3. **Education**: Learning NPU programming through automated examples
4. **Benchmarking**: Comparing different implementation strategies

## Limitations

- **Complexity**: Very complex operators may require manual refinement
- **Hardware Specific**: Generated code is Ascend NPU specific
- **CANN Dependency**: Requires local CANN environment
- **Single Test Case**: Evolution uses one test case per variant (vs CAKE's comprehensive CSV testing)

## Next Steps

- Read [GETTING_STARTED.md](GETTING_STARTED.md) for installation and first operator
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for deep technical details
- Read [EVO.md](EVO.md) for evolution system details

## Community & Support

- **Issues**: Report bugs and request features via project issue tracker
- **Documentation**: All docs in `/docs` folder

---

**Version**: 2.0 (Local Mode Only)
**Last Updated**: 2026-03-17
**Status**: Production Ready
