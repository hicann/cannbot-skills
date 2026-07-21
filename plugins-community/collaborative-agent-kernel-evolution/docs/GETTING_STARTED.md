# Getting Started with CAKE2

This guide will help you set up CAKE2 and generate your first Ascend NPU operator.

## Prerequisites

### Local Mode Requirements

**Single Machine:**
- Python 3.10 or higher
- CANN 8.3.RC1 or higher
- Ascend NPU hardware
- torch_npu 2.9.0 or higher
- Internet connection (for LLM API)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd CAKE2
```

### Step 2: Install Dependencies

```bash
# Create isolated environment (REQUIRED to avoid conflicts)
conda create -n cake2 python=3.10
conda activate cake2

# Install dependencies
pip install numpy attrs pyyaml decorator scipy psutil protobuf

# Install CANN dependencies
pip install torch_npu
```

**Important**: Always use an isolated Python environment (conda/venv) to avoid conflicts when installing custom operators.

### Step 3: Set Up CANN Environment

```bash
# Set CANN path (adjust to your installation)
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# Verify CANN installation
npu-smi info
```

## Quick Start: Generate Your First Operator

### Using Claude Code Agent (Recommended)

#### 1. Create Working Directory

```bash
mkdir genop
cp -r .claude genop/
cd genop
```

#### 2. Start Claude Code

```bash
Claude Code
```

#### 3. Select Agent

Press `Tab` key and select:
- `cake` - For standard operator generation
- `cake_evo` - For evolutionary optimization

#### 4. Describe Your Operator

Example prompts:

**Simple Activation Function:**
```
Generate a FastGELU activation function operator.
Formula: y = x / (1 + exp(-1.702 * |x|)) * exp(0.851 * (x - |x|))
```

**Fused Operation:**
```
Generate a fused AddLayerNorm operator that combines element-wise addition
and layer normalization in a single kernel.
```

**Custom Pooling:**
```
Generate an adaptive average pooling 2D operator that reduces spatial
dimensions to a target output size.
```

#### 5. Local Compilation

The agent will automatically detect your CANN environment and compile locally on your NPU.

#### 6. Wait for Results

The agent will:
1. Generate operator description JSON
2. Create PyTorch reference implementation
3. Convert to functional API
4. Generate DSL baseline
5. Generate AscendC kernel (4 transformation passes)
6. Compile and test locally
7. Show performance results

### Using Python API

```python
from pathlib import Path
import subprocess

def compile_local_operator(op_name, project_dir):
    """Compile operator locally with CANN"""
    
    # Build with cmake
    build_dir = project_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure
    subprocess.run([
        "cmake", "..", 
        f"-DCMAKE_CXX_COMPILER=g++",
        f"-DCMAKE_BUILD_TYPE=Release"
    ], cwd=build_dir, check=True)
    
    # Build
    subprocess.run([
        "make", "-j"
    ], cwd=build_dir, check=True)
    
    print(f"✅ {op_name} compiled successfully!")

# Example usage
compile_local_operator(
    op_name="fast_gelu",
    project_dir=Path("output/fast_gelu/FastGeluCustom")
)
```

## Understanding the Output

### Output Directory Structure

```
output/
└── fast_gelu/
    ├── fast_gelu_op_desc.json      # Operator description
    ├── fast_gelu_reference.py      # PyTorch reference
    ├── fast_gelu_functional.py     # Functional API version
    ├── fast_gelu_dsl.py            # DSL baseline
    ├── fast_gelu_project.json      # Project configuration
    ├── fast_gelu.cpp               # Python binding
    └── FastGeluCustom/             # AscendC project
        ├── op_host/
        ├── op_kernel/
        │   └── fast_gelu.cpp       # AscendC kernel
        ├── CMakeLists.txt
        └── build/
            └── libfast_gelu_custom.so  # Compiled library
```

### Evaluation Results

Successful compilation shows:

```
============================================================
CORRECTNESS: [PASS]
Output 0: shape=[1024, 4096], match_rate=100.00% (4194304/4194304),
          max_diff=0.00000e+00, mean_diff=0.00000e+00
PERFORMANCE: ref=0.344ms, custom=0.128ms, speedup=2.69x
============================================================
```

**Metrics Explained:**
- **match_rate**: Percentage of elements matching reference (should be 100%)
- **max_diff**: Maximum absolute difference (should be ~0)
- **mean_diff**: Average absolute difference (should be ~0)
- **speedup**: Custom implementation speed vs PyTorch reference

## Evolutionary Optimization

For better performance, use evolutionary generation:

### Using cake_evo Agent

```bash
cd genop
Claude Code
# Press Tab, select cake_evo
```

Describe your operator, then the agent will:
1. Generate multiple variants in parallel (default: 3-5)
2. Evaluate and classify them (Good/Medium/Poor)
3. Use tiered sampling to inspire next round
4. Repeat for multiple rounds (default: 2-3)
5. Return best implementations

## Troubleshooting

### Compilation Errors

**Check logs:**
```bash
# Check compilation output
cat output/fast_gelu/FastGeluCustom/build/build.log
```

**Common issues:**
- CANN environment not sourced
- NPU device busy or unavailable
- Insufficient memory
- Code generation errors (try regenerating)

### Import Errors

```bash
# Ensure virtual environment is activated
conda activate cake2

# Reinstall dependencies
pip install --upgrade torch_npu
```

### NPU Device Issues

```bash
# Check NPU status
npu-smi info

# Reset NPU if needed
npu-smi -i 0 -r

# Check device availability
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
```

### Environment Issues

```bash
# Check CANN environment
echo $ASCEND_HOME_PATH
echo $LD_LIBRARY_PATH

# Source CANN environment if needed
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Next Steps

1. **Read Architecture**: Understand system design in [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Evolution Details**: Deep dive into evolution system in [EVO.md](EVO.md)

## Best Practices

1. **Always use isolated Python environments** to avoid package conflicts
2. **Verify CANN installation** before starting
3. **Use descriptive operator names** (lowercase, underscores)
4. **Provide clear formulas** in operator descriptions
5. **Start with simple operators** before attempting complex fused operations
6. **Monitor NPU resources** when running multiple compilations
7. **Save successful implementations** for future reference

## Getting Help

- Check documentation in `/docs` folder
- Review examples in `/examples` folder
- Examine test cases in `/tests` folder
- Read inline code comments for implementation details

---

**Next**: [ARCHITECTURE.md](ARCHITECTURE.md) - Understanding CAKE2 Architecture