# Return Code Explanations in Evaluation Logs

## Overview

The CAKE2 system automatically adds human-readable explanations for process return codes in evaluation logs. This helps quickly diagnose why an evaluation failed without needing to understand Unix signal codes.

## Feature Details

### What Was Added

When the evaluation script completes (successfully or not), the system now logs:
1. **Return code**: The numeric exit code from the process
2. **Return code meaning**: Human-readable explanation of what the code means
3. **Status**: SUCCESS or FAILED

### Example Output

**Before** (old format):
```
=== Evaluation Script Output ===
Device ID: 2
Return code: -11
Status: FAILED
```

**After** (new format):
```
=== Evaluation Script Output ===
Device ID: 2
Return code: -11
Return code meaning: SIGSEGV (Segmentation Fault) - Invalid memory access in kernel code. Common causes: buffer overflow, null pointer dereference, incorrect tensor shapes/dimensions.
Status: FAILED
```

## Common Return Codes

### Success
- **0**: Success - Evaluation completed without errors

### Normal Errors
- **1-127**: Process exited with error code N
  - Usually indicates Python exception or logical error
  - Check STDERR for error message

### Signal-Based Failures (Process Crashes)

#### Critical Errors (Most Common)
- **-11 or 139**: **SIGSEGV (Segmentation Fault)**
  - **Meaning**: Invalid memory access in kernel code
  - **Common causes**:
    - Buffer overflow/underflow in AscendC kernel
    - Null pointer dereference
    - Incorrect tensor shapes/dimensions
    - Accessing uninitialized memory
    - Tiling calculations resulting in invalid memory access
  - **What to check**:
    - Kernel implementation for memory access bugs
    - Input tensor shapes match kernel expectations
    - Tiling parameters are valid for input sizes

- **-6 or 134**: **SIGABRT (Abort)**
  - **Meaning**: Process aborted
  - **Common causes**:
    - Assertion failure in kernel code
    - Memory corruption detected by runtime
    - std::abort() called explicitly
  - **What to check**:
    - Kernel assertions and preconditions
    - Memory allocation/deallocation patterns

#### Resource Issues
- **-9 or 137**: **SIGKILL**
  - **Meaning**: Process forcefully killed
  - **Common causes**:
    - Out of memory (OOM killer)
    - Manual kill -9 command
    - System resource limits exceeded
  - **What to check**:
    - NPU memory usage
    - System memory availability
    - Check dmesg for OOM messages

#### Other Signals
- **-8 or 136**: **SIGFPE (Floating Point Exception)**
  - Division by zero
  - Invalid arithmetic operation

- **-4 or 132**: **SIGILL (Illegal Instruction)**
  - Incompatible CPU instruction
  - Corrupted binary

- **-15 or 143**: **SIGTERM**
  - Process terminated gracefully
  - Usually from timeout or manual termination

- **-2 or 130**: **SIGINT**
  - Interrupted by user (Ctrl+C)

## Understanding Return Codes

### Why Two Formats?

Unix systems report signal-based termination in two ways:
- **Negative**: `-N` where N is the signal number (e.g., -11 for SIGSEGV)
- **128+N**: `128 + N` where N is the signal number (e.g., 139 = 128 + 11)

The system recognizes both formats and provides the same explanation.

### How to Interpret

1. **Return code 0**: Everything worked ✅
2. **Return code 1-127**: Python error, check STDERR for traceback
3. **Return code < 0 or > 128**: Process crashed, check explanation for cause

## Debugging Workflow

### For SIGSEGV (-11)

1. **Check the kernel implementation**:
   ```bash
   # Look at the generated AscendC code
   cat output/{op_name}/{op_name}Custom/op_kernel/.../*.cpp
   ```

2. **Verify tensor shapes**:
   - Check `{op_name}_custom.py` for input shapes
   - Ensure kernel tiling matches input dimensions

3. **Enable verbose NPU logging**:
   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=1
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

4. **Check for common patterns**:
   - Array indexing without bounds checking
   - Pointer arithmetic errors
   - Incorrect use of `GetBlockIdx()` or `GetBlockNum()`

### For SIGABRT (-6)

1. **Check assertions**:
   - Look for `assert()` statements in kernel code
   - Check preconditions in operator implementation

2. **Memory corruption**:
   - Review memory allocation patterns
   - Check for buffer overruns
   - Verify proper use of UB/L1 memory

### For SIGKILL (-9)

1. **Check memory usage**:
   ```bash
   # Check NPU memory and system status
   dmesg | grep -i "out of memory"
   npu-smi info  # Check NPU memory
   ```

2. **Reduce memory footprint**:
   - Decrease batch size
   - Optimize tiling strategy
   - Check for memory leaks

## Implementation Details

### Code Location

This functionality is integrated into the evaluation workflow and provides automatic return code explanations.

### Supported Return Codes

The system recognizes:
- All common Unix signals (SIGSEGV, SIGKILL, SIGABRT, SIGTERM, SIGFPE, SIGILL, SIGINT)
- Both negative and 128+ formats
- Generic explanations for unknown codes

## Benefits

1. **Faster Debugging**: Immediately understand why evaluation failed
2. **Better Error Messages**: No need to look up signal numbers
3. **Actionable Guidance**: Suggestions for common causes
4. **Consistent Format**: All evaluation logs have same structure

## Example Real-World Case

**Scenario**: index_put operator evaluation

**Log Output**:
```
=== Evaluation Script Output ===
Device ID: 2
Return code: -11
Return code meaning: SIGSEGV (Segmentation Fault) - Invalid memory access in kernel code. Common causes: buffer overflow, null pointer dereference, incorrect tensor shapes/dimensions.
Status: FAILED

=== STDERR ===
INFO - Set ASCEND_CUSTOM_OPP_PATH=...
INFO - Updated LD_LIBRARY_PATH=...
```

**Diagnosis**:
- Process crashed immediately after environment setup
- Likely crash during `import custom_ops_lib` or first operator call
- Kernel has memory access bug

**Next Steps**:
1. Review index_put kernel implementation
2. Check tensor indexing logic
3. Verify bounds checking on index operations
4. Test with smaller input sizes

## Future Enhancements

Potential improvements:
- Add signal handlers to catch crashes and log stack traces
- Integrate with NPU error logs
- Provide operator-specific debugging hints
- Add memory profiling information

## References

- Unix Signal Reference: `man 7 signal`
- Python subprocess documentation: https://docs.python.org/3/library/subprocess.html
- CANN Error Codes: Check CANN documentation