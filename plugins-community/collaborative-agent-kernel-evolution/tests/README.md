# CAKE2 Tests

Test suite for CAKE2 local mode functionality.

## Prerequisites

- Local CANN environment setup
- NPU hardware available
- Python 3.10+
- torch_npu installed: `pip install torch_npu`

## Running Tests

### Test Local Compilation

This test verifies that CAKE2 can generate and compile AscendC operators locally.

```bash
# Run local compilation test
python3 tests/test_local_compilation.py
```

### GELUCustom Test Case

The main test case validates the complete CAKE2 pipeline using a GELUCustom operator:

1. **Environment Check**
   - Verify CANN installation
   - Check NPU device availability
   - Validate torch_npu setup

2. **Project Generation**
   - Generate AscendC project structure
   - Create CMakeLists.txt and build scripts
   - Generate kernel and host code

3. **Compilation**
   - Build project with cmake
   - Link against CANN libraries
   - Generate custom operator library

4. **Evaluation**
   - Import compiled operator
   - Run correctness tests vs PyTorch
   - Measure performance speedup

### Expected Output

```
============================================================
CAKE2 Local Mode Test Suite
============================================================
Testing: GELUCustom operator
============================================================

============================================================
Test: Environment Check
============================================================

→ Checking CANN installation...
✓ ASCEND_HOME_PATH: /usr/local/Ascend/ascend-toolkit/latest
✓ CANN environment variables set

→ Checking NPU devices...
✓ Found 8 NPU devices
✓ Using device: 0

============================================================
✅ Test PASSED: Environment check
============================================================

============================================================
Test: Project Generation
============================================================

✓ Created test operator: gelu_custom
✓ Generated project structure:
  - CMakeLists.txt
  - build.sh
  - op_host/
  - op_kernel/

→ Verifying project files...
✓ Found: GeluCustom/CMakeLists.txt
✓ Found: GeluCustom/op_host/gelu_custom.cpp
✓ Found: GeluCustom/op_kernel/gelu_custom.cpp

============================================================
✅ Test PASSED: Project generation
============================================================

============================================================
Test: Local Compilation
============================================================

→ Building project with cmake...
✓ CMake configuration successful
✓ Compilation successful
✓ Generated library: libgelu_custom.so

============================================================
✅ Test PASSED: Compilation
============================================================

============================================================
Test: Evaluation
============================================================

→ Testing correctness...
✓ Import successful
✓ Correctness test: 100.00% match
✓ Max difference: 1.2e-06

→ Testing performance...
✓ PyTorch time: 0.234ms
✓ Custom time: 0.087ms
✓ Speedup: 2.69x

============================================================
✅ Test PASSED: Evaluation
============================================================

============================================================
Test Summary
============================================================
✅ PASS: Environment Check
✅ PASS: Project Generation
✅ PASS: Compilation
✅ PASS: Evaluation
============================================================
Results: 4/4 tests passed
============================================================
```

## Test Coverage

These tests ensure that:

1. **CANN Environment**: Proper CANN setup and environment variables
2. **NPU Hardware**: NPU devices are available and accessible
3. **Code Generation**: AscendC projects are generated correctly
4. **Compilation**: Projects compile successfully with CANN
5. **Correctness**: Generated operators produce correct results
6. **Performance**: Operators achieve expected speedup

## Troubleshooting

### CANN Not Found

```
✗ Environment check failed: ASCEND_HOME_PATH not set
```

**Solution**: Set up CANN environment:
```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### NPU Not Available

```
✗ No NPU devices found
```

**Solution**: Check NPU status:
```bash
npu-smi info
# If devices are busy, reset them:
npu-smi -i 0 -r
```

### Compilation Failed

```
✗ Compilation failed: CMake error
```

**Solution**: Check build environment:
```bash
# Verify CANN libraries
ls $ASCEND_HOME_PATH/lib
# Check cmake version
cmake --version  # Should be >= 3.16
```

### Import Failed

```
✗ Import failed: cannot open shared object file
```

**Solution**: Check library path:
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib:$LD_LIBRARY_PATH
# Verify library exists
ls build/libgelu_custom.so
```

## GELUCustom Implementation Details

The test uses a GELU (Gaussian Error Linear Unit) activation function:

**Formula**: `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

**Test Parameters**:
- Input shape: [1024, 4096]
- Data type: float32
- Target accuracy: > 99.99%
- Target speedup: > 1.5x

**Key Features Tested**:
- Element-wise computation
- Memory tiling strategy
- SIMD vectorization
- Numerical stability

## Adding New Test Cases

To add new operator tests, extend `test_local_compilation.py`:

```python
async def test_new_operator():
    """Test new operator implementation"""
    print("=" * 60)
    print("Test: New Operator")
    print("=" * 60)

    # Operator-specific test logic
    # 1. Generate project
    # 2. Compile
    # 3. Evaluate
    # 4. Verify results

    return True  # or False if test fails
```

## Performance Benchmarks

Expected performance ranges for different operator types:

| Operator Type | Expected Speedup |
|---------------|------------------|
| Element-wise (GELU, ReLU) | 2-5x |
| Reductions (Sum, Mean) | 1.5-3x |
| Matrix ops (MatMul) | 1.2-2x |
| Fused ops | 3-10x |

## CI/CD Integration

Local tests can be integrated into CI/CD pipelines:

```yaml
# Example GitLab CI
test_cake2:
  script:
    - source /usr/local/Ascend/ascend-toolkit/set_env.sh
    - python3 tests/test_local_compilation.py
  only:
    - master
    - merge_requests
```