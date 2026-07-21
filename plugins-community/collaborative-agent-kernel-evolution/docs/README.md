# CAKE2 Documentation

This directory contains comprehensive documentation for CAKE2 features and capabilities.

## Main Documentation Files

### [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Project Introduction
High-level overview of CAKE2 project, including:
- Project purpose and capabilities
- Architecture overview (local mode)
- Supported operator types
- Technology stack and workflow

**Use this when**: You're new to CAKE2 and want to understand what it does.

### [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and First Steps
Quick start guide for local CANN environment setup, including:
- Prerequisites and installation
- First operator generation
- Output structure understanding
- Troubleshooting common issues

**Use this when**: You want to install CAKE2 and generate your first operator.

### [ARCHITECTURE.md](ARCHITECTURE.md) - System Architecture
Deep technical overview of CAKE2's architecture, including:
- Core components and pipeline stages
- Local compilation architecture
- Skill system organization
- Performance optimizations and design patterns

**Use this when**: You want to understand how CAKE2 works internally.

### [EVO.md](EVO.md) - Evolutionary Kernel Generation
Complete guide to the evolutionary kernel generation system, including:
- Quick start guide
- How to use the cake_evo agent
- How inspirations and diversity work
- Subagent execution and result evaluation
- Troubleshooting evolutionary workflows

**Use this when**: You want to generate multiple kernel variants and optimize through evolutionary search.

### [RETURN_CODE_EXPLANATIONS.md](RETURN_CODE_EXPLANATIONS.md) - Return Codes
Reference for CAKE2 return codes and error handling.

## Quick Navigation

### For Users

**I want to understand what CAKE2 is:**
- Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**I want to generate my first kernel:**
- Follow [GETTING_STARTED.md](GETTING_STARTED.md)
- Use the `cake` agent in Claude Code

**I want to generate multiple kernel variants and optimize:**
- Use the `cake_evo` agent in Claude Code
- See [EVO.md](EVO.md) for complete guide

### For Developers

**I want to understand the system architecture:**
- Read [ARCHITECTURE.md](ARCHITECTURE.md)

**I want to understand the evolutionary algorithm:**
- See [EVO.md](EVO.md) → How Inspirations and Diversity Work

**I want to modify the evolutionary system:**
- See [EVO.md](EVO.md) and study the evolution/ module

## Document Structure

### PROJECT_OVERVIEW.md
- Project introduction and purpose
- Key features and capabilities
- Technology stack overview
- Use cases and limitations

### GETTING_STARTED.md
- Installation prerequisites
- CANN environment setup
- First operator generation
- Troubleshooting guide

### ARCHITECTURE.md
- System architecture overview
- Core components breakdown
- Local compilation pipeline
- Design patterns and optimizations

### EVO.md
- Evolutionary generation workflow
- Tiered sampling and inspirations
- Subagent execution model
- Result evaluation and classification

## Additional Resources

- **Evolution strategies**: `skills/cake-evo/references/meta_prompts/` — optimization strategy index and details

## Getting Started

### 1. Install CANN Environment

Follow [GETTING_STARTED.md](GETTING_STARTED.md) for CANN 8.3+ setup.

### 2. Load CAKE2 Plugin

```bash
claude --plugin-dir /path/to/cake2
```

### 3. Generate Kernels

Type `/agents` to select an agent:
- `cake` — Generate a single high-quality kernel
- `cake-evo` — Evolutionary multi-variant optimization

See [EVO.md](EVO.md) for evolution guide.

## Common Questions

**Q: What's the difference between cake and cake_evo?**

A:
- `cake`: Generates a single high-quality kernel
- `cake_evo`: Generates multiple variants and optimizes through evolution

**Q: Do I need NPU hardware locally?**

A: Yes, CAKE2 requires local CANN environment and NPU hardware for compilation.

**Q: How does the evolutionary system improve across rounds?**

A: It uses tiered sampling to select inspirations from successful implementations, then generates new variants with different optimization strategies. See [EVO.md](EVO.md) → How Inspirations and Diversity Work.

**Q: How are results evaluated?**

A: Results are evaluated on compilation success, correctness (vs PyTorch reference), and performance (speedup). See [EVO.md](EVO.md) → Result Evaluation.

## Contributing

When adding new documentation:
1. Update the appropriate existing file or create a new one
2. Update this README with navigation links

## Support

For issues or questions:
- Check the relevant documentation file
- See main project README for contact information

---

**Last Updated**: 2026-03-16
**Documentation Version**: Local Mode Only