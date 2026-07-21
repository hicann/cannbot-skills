---
name: functional-conversion
disable-model-invocation: true
description: Convert PyTorch nn.Module reference implementation to stateless functional API (module_fn + get_inputs). Use after reference-generation, before ascend-call-generation.
---

## What I do

Convert PyTorch `nn.Module` classes to `torch.nn.functional` style. Removes class definitions, `self` parameters, and uses functional API calls directly.

## Workflow

1. Read input PyTorch code `{op_name}_reference.py`
2. Read the corresponding example files according to op `{category}` from @references:
   - For pool ops: `average_pooling2d`
   - For cum ops: `cumsum`
   - For reduction ops: `sum_reduction_over_a_dimension`
   - For loss ops: `mse_loss`
   - Other: `layer_norm`
3. Generate PyTorch code:
   - Imports: These are always at the top of the file and contains mostly torch imports.
   - Model Definition: A `Model(nn.Module)` class with an init and a forward method.
   - Configurations: Following the `Model` class, there are some configurations and two functions `get_inputs()` and `get_init_inputs()`. Simply copy the code from the original file. They are for test purposes.
   - Members: The `Model` class may have members that are also `nn.Module`, they are defined either in `torch.nn` or before the `Model` class.
4. Save to `output/{op_name}/{op_name}_functional.py`
5. Continue to the next step in agent workflow

**key Points**:
### Imports
You will likely need the following libraries:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
You may import other libraries, but you probably not any other torch modules.
For example, if you do `from torch import _VF`, you can then use `_VF.lstm` and `_VF.gru` functions.
### Model Definition
- Define `nn.Parameter` in `Model.__init__()`, since they are needed for the functional calls.
    - If a parameter is already defined in the `Model` class, you can directly use it.
    - Otherwise, you may extract it from the `nn.Module`s. E.g. `self.conv.weights = nn.Parameter(self.conv1.weight.data.clone)`, where `self.conv1` is an `nn.Module`. If you use `nn.ParameterDict` to extract parameters, remember that names cannot contain dots.
    - Notice that these parameters' specifications are given by the `get_init_inputs()` function call because Model is initialized with `Model(*get_init_inputs())`.
- Modify the `forward()` method to use functional calls instead of `nn.Modules`.
    - This can be achieved by defining a `module_fn()` that takes in all the args that the forward pass was called with as well as any neural network parameters that the Model holds. It should faithfully reproduce the exact forward pass as the original Model class.
    - Then, you can augment the `Model.forward()` method signature, where you add a `fn` argument that defaults to `module_fn`.
    - Important: your goal is to reproduce the original model’s behavior. Do not add unnecessary arguments to module_fn—only include those strictly required to replicate the functionality, even if extra arguments exist in the torch.nn.funtional's interface.
- You need to read the original code carefully. E.g., the code may not implement a residual connection even though the name or comment suggests it.

### Members
- If a member is an `nn.Module` for which there exists a functional call, you can replace its forward pass with the functional call. E.g., if it is a `nn.Conv2d`, you can replace `self.conv(x)` with `F.conv2d(x, ...)` with proper arguments.
- Otherwise, you need to decompose it until the its forward pass can be replaced with a series of functional calls. You can define functions for this, and these functions will be used in the `module_fn()` function.

### Additional Requirement
- If some operation reduces dimensions on a middle or first axis (e.g., batch_norm), you must instead rewrite the computation so that:
  1. You first **transpose** the tensor so the reduction axis becomes a **last axis**,
  2. Then perform the module_fn on the tensor,
  3. Then transpose the tensor back to the original layout.
- Pooling operations: Transpose the tensor so the channel axis becomes the last axis before applying pooling
- Cumulative operations (e.g., cumsum, cummin): Transpose the reduction axis to the first axis before applying the operation.
