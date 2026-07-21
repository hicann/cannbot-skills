---
name: dsl-lowering
disable-model-invocation: true
description: Translate the operator DSL into AscendC code through multiple passes. Also used when diagnosing compilation errors.
subagent:
  enabled: true
  agent_type: general
  reason: "Multi-pass transformation with compilation requires autonomous error recovery; subagent keeps orchestrator context clean."
---


## What I do

Translate the operator DSL into AscendC code through up to five passes: `tiling_pass`, `init_pass`, `process_pass`, `process_nonaligned_pass`, and (when multi-dtype support is required) `multi_dtype_pass`.
The input AscendC code may already include modifications from earlier passes. Apply **only** the transformation described in the task.


## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Pass 1   tiling_pass                                               │
│           Host Tiling + Workspace + OpDef                           │
│           → compile → error? → error_correction                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Pass 2   kernel_config_pass                                        │
│           Includes / Types / Constants                              │
│           → compile → error? → error_correction                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Pass 3   kernel_pass (KernelOp class)                              │
│           Init() + Process() + member variables                     │
│           → compile → error? → error_correction                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Pass 4   entry_pass (kernel_name function)                         │
│           Dispatch entry + SetBlockDim                              │
│           → compile → error? → error_correction                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Pass 5   multi_dtype_pass  (optional — when multi-dtype required)  │
│           DTYPE_X1 macro + if constexpr + Cast mode fixes           │
│           → compile → error? → error_correction                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Done     All passes compiled successfully                          │
└─────────────────────────────────────────────────────────────────────┘
```

Sub-Agent Strategy

This skill must use a subagent to handle the complex multi-pass transformation:
- The subagent autonomously executes all 4/5 passes sequentially
- Handles compilation errors with automatic retry (up to 3 attempts)
- Manages intermediate file state and error recovery
- Reports progress after each pass

1. Read the operator DSL file `output/{op_name}/{op_name}_dsl.py` and the directory created by the previous AscendC project generation step

2. Based on the operator DSL file, sequentially execute four transformation passes. After each pass generates AscendC code, invoke the `build.sh` script in the AscendC project to perform compilation. If a compilation error occurs:
   - Refer to repair examples in `references/error_correction/` for guidance

   Specifically, the four transformation passes are as follows:

**⚠️ CRITICAL: Always use `source build.sh`, NEVER `bash build.sh`.**
In docker+tmux environments, running `bash build.sh` creates a new process group that receives SIGTTIN/SIGTTOU signals, causing cmake/make to stop silently (Tl state) with no output. Using `source build.sh` executes in the current shell and avoids this issue.

```bash
# Correct:
source output/{op_name}/{op_name}Custom/build.sh
# Wrong (will silently fail in docker+tmux):
# bash output/{op_name}/{op_name}Custom/build.sh
```

### ⚠️ CRITICAL: Process Pass Requirements

The Process() method is the **CORE** of the operator. It must implement the **ACTUAL** computation logic from the DSL.

❌ **NEVER** generate placeholder code such as:
- `Duplicate(output, 0.0f, size)` - just initializing with zeros
- `DataCopy(output, input, size)` - just copying input to output
- Partial computation (e.g., only processing first 32 elements)
- Simplified logic that ignores some inputs
- Approximations that don't match the DSL formula

✅ **ALWAYS** implement the FULL algorithm:
1. Load **ALL** required input tiles (not just one)
2. Perform **ALL** computations specified in DSL
3. Use **ALL** input tensors (check that each input is loaded and used)
4. Accumulate across **ALL** tiles (especially for matmul/reduction)
5. Apply **ALL** activations/transformations correctly

### Verification Checklist for Process()

Before considering Process() complete, verify:
- [ ] Does it match the DSL computation exactly?
- [ ] Are all input tensors loaded and used in computation?
- [ ] Are all loops from DSL present (especially tile loops)?
- [ ] Is accumulation done correctly across tiles?
- [ ] Are activations applied with proper AscendC APIs?
- [ ] Will different inputs produce different outputs?
- [ ] Does the logic make sense for the operator type?

**If ANY answer is "No", the code is INCORRECT and must be regenerated.**

### Special Guidance for Common Patterns

**For Matmul Operations (A @ B or A @ B^T):**
```cpp
// Must implement nested loops with proper accumulation
for (uint32_t tile = 0; tile < num_tiles; tile++) {
    // Load A tile and B tile
    // For each output element (i,j):
    //   - Multiply A[i,:] * B[j,:]  
    //   - Use ReduceSum to compute dot product
    //   - Accumulate to result[i,j]
}
```

**For Element-wise + Reduction:**
```cpp
// Apply element-wise ops first, then reduce
AscendC::Mul(temp, input1, input2, count);
AscendC::ReduceSum(output, temp, workspace, count);
```

**For Activation Functions:**
```cpp
// Use proper AscendC APIs, not approximations
AscendC::Exp(temp, input, count);        // exp(x)
AscendC::Adds(temp, temp, 1.0f, count);  // 1 + exp(x)
AscendC::Log(output, temp, count);       // log(1 + exp(x))
```

2.1 `tiling_pass`'s task description is below：

"""
### Tiling Instruction for DSL → AscendC Conversion

When converting AscendDSL kernels into AscendC, the **tiling logic must be moved to the host side**.  
The purpose of tiling is to compute all scalar parameters, attributes, and workspace requirements on the host and store them in a tiling struct that will be passed to the AscendC kernel.

AscendC kernels follow the SPMD execution model: each core handles one data block.  
Thus, tiling must determine how data is partitioned across cores.

---

### Required Components in Tiling

#### 2.1.1. Define a Tiling Struct
Create a tiling data structure.
This struct stores:
- All scalar values originally computed in DSL.
- Attributes (if any).

#### 2.1.2. Compute and Set Tiling Parameters
All scalar computations that were inside DSL must be moved to the host:
- Core partitioning logic (e.g. rows per core, tiles per core)
- UB tiling parameters (tile sizes, loop counts)
- Shape-derived constants

**⚠️ CRITICAL: Dynamic Core Count — Never Hard-Code BLOCK_DIM**

When the DSL uses `n_cores = tl.num_vec_cores()` (or any fixed constant), the AscendC
tiling function **must** query the hardware at runtime using `GetCoreNumAiv()`:

```cpp
#include "tiling/platform/platform_ascendc.h"

// Inside TilingFunc:
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
if (coreNum == 0) { coreNum = 1; }  // defensive guard

// If there is no work, skip launching the kernel to avoid zero block dim
// and divide-by-zero in basePerCore/pivot computation.
if (totalWork == 0) {
    return;  // Early-exit in the surrounding host function
}

// Cap to actual work to avoid empty cores
uint32_t usedCoreNum = (totalWork < coreNum) ? totalWork : coreNum;
context->SetBlockDim(usedCoreNum);
```

**Do NOT** generate `const uint32_t BLOCK_DIM = 16;` or any hard-coded constant.

**Pivot Distribution for Non-Divisible Workloads**

When `totalWork % usedCoreNum != 0`, use the pivot pattern so every element is covered:

```cpp
uint32_t basePerCore = totalWork / usedCoreNum;   // floor: minimum per core
uint32_t pivot       = totalWork % usedCoreNum;   // first 'pivot' cores get basePerCore+1

// Pass both basePerCore and pivot to the kernel via the tiling struct.
// In the kernel, each core computes:
//   myCount = basePerCore + (blockIdx < pivot ? 1 : 0)
//   myStart = blockIdx * basePerCore + min(blockIdx, pivot)
```

Store `basePerCore` and `pivot` in the tiling struct; the kernel uses them to determine
its own work range without any hard-coded assumption about divisibility.


#### 2.1.3. Set Kernel Attributes
If used attributes, add corresponding fields into the tiling struct.
Access attributes via context->GetAttrs()->GetAttrPointer<float>(index).

#### 2.1.4. Set Workspace Sizes

Workspace should be allocated in AscendC **only when the original DSL host code explicitly allocates a GM buffer for intermediate results**.

In other words:

- If the DSL contains statements like:
```python
workspace = torch.empty(n, dtype=torch.float32, device=...)
```
In such cases:
```cpp
#include "tiling/platform/platform_ascendc.h" // must include for using PlatformAscendC

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;
```
"""

2.2 Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

**API Name:** `GetAttrPointer`  

**Function Prototype:**  
```cpp
template<typename T>
const T* GetAttrPointer(size_t index) const
```

**Usage Example:** 
To obtain the first attribute (negative_slope):
```cpp
namespace ops {
class LeakyReluCustom : public OpDef {
public:
    LeakyReluCustom(const char *name) : OpDef(name)
    {
        // Define the attribute with default value
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.0);
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    gert::TilingContext context;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    // Access the attribute value by index
    const float* negativeSlope = attrs->GetAttrPointer<float>(0);
}
}
```
'''

2.3 Based on the operator category, refer to the most appropriate `tiling` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.


3. `init_pass`' task description is below：

"""
Implement kernel code init part in AscendC kernels based on the AscendDSL code. 

### DSL to AscendC Kernel Key Principles
#### 3.1. Parameter and Core Setup

| DSL Concept | AscendC Implementation | Principle |
| :--- | :--- | :--- |
| **DSL Tiling Parameters** (`rows_per_core`, `tile_length`, etc.) | Store these as **`uint32_t` member variables** within the AscendC class. | These parameters determine the loop bounds and data access size within the `Process()` method. |
| **program_id** | Retrieve using **`GetBlockIdx()`** in the `Init()` method. | This value is used to calculate the starting GM address for the core's data partition. |
| **Global Memory (GM) Access Ranges** | Calculated and configured using **`SetGlobalBuffer`**. | Configure the **`In`** and **`Out`** Tensors (member variables) to ensure the AI Core accesses only its assigned block of data from Global Memory (GM). |

---

#### 3.2. Buffer and Queue Initialization

You must strictly select the type of variable, either TBuf (Calculation Buffer) or TQue (Data Queue), based on its purpose in the data flow. This is a core requirement for AscendC kernel design.

| DSL Buffer Type | AscendC Object Type | VEC TPosition | Initialization and Purpose |
| :--- | :--- | :--- | :--- |
| **Input Buffers** (e.g., ub used in copyin) | **`TQue`** (Tensor Queue) | **VECIN** (Vector Input) | **Purpose:** Used for synchronized data transfer into the `Compute` function. **Initialization:** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`. |
| **Output Buffers** (e.g., ub used in copyout) | **`TQue`** (Tensor Queue) | **VECOUT** (Vector Output) | **Purpose:** Used for synchronized data transfer out of the `Compute` function to `CopyOut`. **Initialization:** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`. |
| **Intermediate/Working Buffers** (`tmp_ub`, `shared_ub`) | **`TBuf`** (Tensor Buffer) | **VECCALC** (Vector Calculation) | **Purpose:** Used for temporary, short-term storage during a single `Compute` phase (e.g., reduction workspace, intermediate results). **Initialization:** `pipe.InitBuffer(TENSOR_BUFFER, SIZE_IN_BYTES)`. |

"""

3.3 Based on the operator category, refer to the most appropriate `init` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.


3.4 Refer to the existing knowledge of AscendC API in `references/ascend_api/`.

4. `process_pass`'s task description is below：

"""
Implement the kernel process part in AscendC kernels based on AscendDSL code.

### DSL to AscendC Kernel Key Principles
#### 4.1. Overall Code Structure and Control Flow

The AscendC code must map the DSL's overall workload execution model onto the AI Core's execution environment.

| DSL Concept | AscendC Implementation | Principle |
| :--- | :--- | :--- |
| **Workload Loop** (Iterating over partitioned data/tiles) | Encapsulated in the **`Process()`** method. | `Process()` manages the execution loop over the data units assigned to the core. |
| **Computational Phases** | `Process()` must call dedicated functions: **`CopyInX`**, **`ComputeX`**, and **`CopyOutX`**. |  Use numbering (e.g., `CopyIn1`, `Compute2`) if the core logic involves multiple distinct passes or data movement steps. |
| **Function Definition** | Each stage function must be defined as `__aicore__ inline` and accept the current loop index (`uint32_t idx`) if needed for global memory (GM) address calculation. | Standard kernel function attributes and structure. |

---

#### 4.2. Data and Buffer Management

This section defines how DSL's memory allocation and movement map to AscendC's Queue and Tensor Buffer (`TBuf`) system for managing the Unified Buffer (UB).

##### 4.2.1. Data Loading (`CopyInX` Functions)

1.  **Allocate UB Space:** Use the appropriate input queue's **`AllocTensor<T>()`** method to reserve a local tensor in the UB.
2.  **Move Data (GM to UB):** Use **`AscendC::DataCopy`** to transfer the data tile from Global Memory (DSL's input tensor) to the local UB tensor.
3.  **Transfer to Compute:** Use the input queue's **`EnQue()`** method to pass the loaded local tensor to the next stage.

##### 4.2.2. Computation (`ComputeX` Functions)

1.  **Acquire All Tensors (Mandatory Start):** At the **very beginning** of every `ComputeX` function, obtain all tensors required for that calculation phase:
    * **Input Tensors:** Dequeue from the input queue(s) (`inQueue.DeQue<T>()`).
    * **Working Buffers:** For internal temporary buffers, use the pre-defined member Tensor Buffers' (`TBuf`) **`Get<T>()`** method (e.g., `sharedBuf.Get<float>()`).
2.  **Execute Logic:** Translate DSL operations to AscendC APIs.
3.  **Flow Control:**
    * If the result is needed by a subsequent stage (`Compute` or `CopyOut`), use the output queue's **`EnQue()`** method.
    * Once an input tensor is processed and no longer needed, use **`FreeTensor()`** on its originating queue.
4.  **Global synchronization**
When different kernels operate on the same global memory and potential data dependency issues arise, synchronization statements `AscendC::SyncAll()` are inserted.

##### 4.2.3. Data Storing (`CopyOutX` Functions)

1.  **Acquire Result:** Use the output queue's **`DeQue<T>()`** method to retrieve the final result tensor from the previous stage.
2.  **Move Data (UB to GM):** Use **`AscendC::DataCopy`** to transfer the result from the local UB tensor to the DSL's output Global Tensor.
3.  **Release UB Space:** Use **`FreeTensor()`** on the output queue to release the local tensor buffer.

---
"""

4.3 Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

'''
API Name: Compare
API Description: Performs element-wise comparison between two tensors. For each element, if the comparison result is true, the corresponding bit in the output tensor is set to 1; otherwise, it is set to 0.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor<uint8_t>
  - src0Local, src1Local: Source operands. Type: LocalTensor
  - cmpMode: Comparison mode of type CMPMODE, which includes the following options: EQ, NE, GE, LE, GT, LT.
  - count: Number of input data elements.
Example:
```cpp
AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, srcDataSize);
```

API Name: CompareScalar
API Description: Performs element-wise comparison between a tensor and a scalar. For each element, if the comparison result is true, the corresponding bit in the output tensor is set to 1; otherwise, it is set to 0.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor<uint8_t>
  - src0Local: Source operand. Type: LocalTensor
  - src1Scalar: Scalar operand.
  - cmpMode: Comparison mode of type CMPMODE, which includes EQ, NE, GE, LE, GT, LT.
  - count: Number of input data elements

API Name: Select
API Description: Generates the destination tensor dst by selecting elements from two source operands (src0 and src1) according to the bit values in selMask.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - selMask: Selection mask tensor. Type: LocalTensor<uint8_t>
  - src0Local: Source operand. Type: LocalTensor
  - src1Local/src1Scalar: Source operand, can be a LocalTensor or a scalar.
  - selMode: VSEL_TENSOR_SCALAR_MODE or VSEL_TENSOR_TENSOR_MODE
  - count: Number of input data elements
Example:
```cpp
AscendC::Select(dstLocal, maskLocal, src0Local, src1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataSize);
```

API Name: LocalTensor.GetValue and GlobalTensor.GetValue
API Description: Retrieves the value at a specific offset in a LocalTensor or GlobalTensor.
Parameter List:
  - offset: Offset by this number of elements
Return Value:
  - Returns an immediate value of type T

API Name: Duplicate
API Description: Duplicates a variable or an immediate value multiple times and fills them into a vector.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor.
  - scalarValue: Source operand to be duplicated. Supports both variables and immediate values. The data type must match the element data type in dstLocal.
  - calCount: Number of data elements to be filled.

API Name: DataCopy
API Description: Data transfer interface for aligned data. 
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - calCount: Number of elements involved in the transfer. The amount of data transferred must be a multiple of 32 bytes.

API Name: TBuf.Get
API Description: Retrieves a LocalTensor from a TBuf object. The tensor can span the entire allocated buffer or a specified portion of it.
Example:
```cpp
// Create a vector calculation buffer and initialize with 1024 bytes
AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
uint32_t byteLen = 1024;
pipe.InitBuffer(calcBuf, byteLen);

// Get a tensor spanning the entire buffer (1024 bytes = 256 int32_t elements)
AscendC::LocalTensor<int32_t> fullTensor = calcBuf.Get<int32_t>();

// Get a tensor with 128 int32_t elements (512 bytes)
AscendC::LocalTensor<int32_t> partialTensor = calcBuf.Get<int32_t>(128);
```
'''


4.4 Based on the operator category, refer to the most appropriate `process` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.



5. `process_nonaligned_pass`'s task description is below：

"""
Process non-32-byte–aligned data transfers in AscendC kernels and **output the full, complete kernel code**.

For all data transfers in the `CopyIn` and `CopyOut` stages:
- If the **data size being transferred** is 32-byte aligned, keep the original `AscendC::DataCopy` implementation.
- If the **data size is not 32-byte aligned**, replace `AscendC::DataCopy` with `AscendC::DataCopyPad`.

⚠️ Important:
- Do NOT output partial code or diffs.
"""

5.1  Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

'''
API Name: DataCopy
API Description: Data transfer interface for aligned data. 
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - calCount: Number of elements involved in the transfer. The amount of data transferred must be a multiple of 32 bytes.

API Name: DataCopyPad
API Description: Data transfer interface for non-aligned data.
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - dataCopyParams, needed for all situations, typically set to {1, count * sizeof(dtype), 0, 0} for contiguous data.
  - padParams, only needed for Global Memory->Local Memory, usually {false, 0, 0, 0} when padding is not required.
Example:
```cpp
AscendC::DataCopyPad(dstGlobal, srcLocal, {1, static_cast<uint16_t>(1 * sizeof(float)), 0, 0, 0}); // four parameters for dataCopyParams: blockCount, blockLen, srcStride, dstStride
AscendC::DataCopyPad(dstLocal, srcGlobal, {1, static_cast<uint16_t>(20 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0}); // Global Memory->Local Memory, needs four parameters
```
'''

5.2  Refer to the examples in the `references/lowering_examples/non_aligned/` directory for the transformation.

6. `multi_dtype_pass` (optional — apply when the operator DSL or op_desc targets float32 + float16 + bfloat16):

"""
Add multi-dtype support so a single kernel source file is compiled three times by the
build system with `-DDTYPE_{INPUT_NAME_UPPER}=float/half/bfloat16_t`.

**Full reference (MUST read before starting)**: `references/multi_dtype/multi_dtype_guide.md`  
**Canonical example**: `references/multi_dtype/add_custom_reference.cpp`

### Transformation checklist

1. **DTYPE macro guard** — read the `*_project.json` to find the first input's name.
   The build system injects `-DDTYPE_{NAME_UPPER}=<type>`. Use that exact name in the
   `#ifndef` guard and throughout the kernel.  **Do NOT hardcode `DTYPE_X1`** unless the
   input is literally named `x1`.  Wrong macro name → silent fallback to float for ALL
   variants. See guide §1 for details and verification command.

2. **Replace hardcoded I/O types** with `DTYPE_{NAME}` in TQue, GlobalTensor, LocalTensor.
   Do NOT replace float32 compute intermediates.

3. **Declare float32 TBuf members** unconditionally. See guide §2.

4. **Init() buffer allocation** — two critical rules (see guide §3 + §Cast Block Alignment):
   - Always allocate conversion TBufs (even for float32 — simplifies code, negligible cost).
   - Float32 buffers need Cast-block-aligned sizes: `max(ALIGN_UP32(N*4), ceil(N/16)*64)`.

5. **CopyIn/CopyOut with type-switching** — use `template <typename T>` helper functions.
   See guide §`if constexpr` Requires a Template Context + §Pattern B Pipeline Barriers.
   Three mandatory rules for non-float paths:
   - Use simple `DataCopyPadParams` API, NOT `DataCopyPadExtParams<T>` (bf16 ABI bug).
   - Add `SetFlag/WaitFlag` pipeline barriers between MTE2↔VEC (TBuf has no implicit sync).
   - Cast modes: bf16→f32 `CAST_NONE`; f32→bf16 `CAST_RINT`; f16↔f32 `CAST_NONE`.

6. **Wrap bf16-incompatible ops** with `if constexpr` (bf16 upcast→f32 compute→downcast).
   `Muls<bf16>`, `Mul<bf16>`, `Add<bf16>` are NOT supported on dav_c220. See guide §4.

7. **Rsqrt precision** — if the kernel uses `Rsqrt`, add Newton-Raphson refinement.
   Hardware Rsqrt returns ~10-bit precision. See guide §Rsqrt Hardware Precision.

8. **Update op_host** type registration to include all three dtypes.

9. **Recompile** all three dtype variants and verify each builds without error.
"""

6.1 **MUST read** `references/multi_dtype/multi_dtype_guide.md` — contains code patterns,
    Cast mode table, buffer alignment formulas, pipeline barrier examples, DataCopyPad API
    rules, and Rsqrt precision fix. The checklist above is intentionally terse; the guide
    has the full rationale and copy-paste code.

6.2 The canonical tested implementation is `references/multi_dtype/add_custom_reference.cpp`.

7. Continue to the next step in agent workflow

## Note

- Apply **only** the transformation described in the task.  
- Follow the logic strictly as defined in the AscendDSL code.  Do **not** introduce extra logic (e.g., conditional checks).  
- Maintain the original code formatting and structure.
- The AscendC code generated by each pass overwrites the existing files directly in the AscendC project.
- In case of compilation failure, attempt up to **three** rounds of automatic repair.
- Provide clear status messages after each pass transformation and after each compilation attempt.