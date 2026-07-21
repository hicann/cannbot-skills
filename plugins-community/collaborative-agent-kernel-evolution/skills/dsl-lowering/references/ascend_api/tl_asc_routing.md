reduce_api = '''
API Name: ReduceSum, ReduceMax, ReduceMin
API Description: Computes the sum/max/min of all input data.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - srcLocal: Source operand. Type: LocalTensor
  - workLocal: Used to store intermediate results during instruction execution. It provides the workspace required for internal computation. Pay special attention to the size of this space (see constraint notes). Type: LocalTensor
  - count: Number of input data elements
Tips: To conserve address space, developers can define a single Tensor to be shared (with overlapping addresses) among srcLocal, dstLocal, and workLocal.
'''

unary_api = '''
API Name: Exp, Sqrt, Rsqrt, Abs, Reciprocal, Relu, Tanh, Asin, Sin, Log, Log2, Log10, Erf
API Description: These APIs perform an operation on each element of a single source tensor and write the result to a destination tensor.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - srcLocal: Source operand. Type: LocalTensor
  - count: Number of input data elements
Tips: For scalar operations (e.g., sqrt, log), please defer GetValue and use these APIs with count = 1.
Example:
```cpp
AscendC::Log(LocalTensor0, LocalTensor0, 1);
float log = LocalTensor0.GetValue(0);
```
'''

binary_api = '''
API Name: Add, Sub, Mul, Div, Max, Min
API Description: These APIs perform an element-wise operation between two source tensors and write the result to a destination tensor.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - src0Local, src1Local: Source operand. Type: LocalTensor
  - count: Number of input data elements
'''

binary_scalar_api = '''
API Name: Muls, Adds, Maxs, Mins, ClampMin, ClampMax
API Description: These APIs perform an element-wise operation between a source tensor and a scalar value and write the result to a destination tensor.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - srcLocal: Source operand. Type: LocalTensor
  - scalarValue: Scalar operand. Data type must match the element type of the destination Tensor.
  - count: Number of input data elements
Note: There is no Subs API. Please use Adds, such as Adds(dst, src, -scalar, count) for sub scalar.
Note: There is no Divs API. Please use Muls, such as Muls(dst, src, 1.0f / scalar, count) for Div scalar.
Note: To compute the reciprocal of an integer scalar as a float, use `1.0f / scalar`. Avoid using `static_cast<float>(scalar)` or `float(scalar)`.
'''

gather_api = '''
API Name: Gather
API Description: Given input tensor and offset, gather values into the output tensor.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor.
  - srcLocal: Source operand. Type: LocalTensor.
  - srcOffsetLocal: the offset of every element in the srcLocal. Type: LocalTensor<uint32_t>.
  - srcBaseAddr: the base address of the srcLocal.
  - count: Number of data elements to be gathered.
Example:
```cpp
AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), m_elementCount);
```
'''

tl_group_mappings = {
    'tl.reduce_sum': 'reduce',
    'tl.reduce_max': 'reduce',
    'tl.reduce_min': 'reduce',
    'tl.vexp': 'unary',
    'tl.vsqrt': 'unary',
    'tl.vrsqrt': 'unary',
    'tl.vabs': 'unary',
    'tl.vreciprocal': 'unary',
    'tl.vrelu': 'unary',
    'tl.vtanh': 'unary',
    'tl.vasin': 'unary',
    'tl.vsin': 'unary',
    'tl.vlog': 'unary',
    'tl.vlog2': 'unary',
    'tl.vlog10': 'unary',
    'tl.verf': 'unary',
    'tl.vadd': 'binary',
    'tl.vsub': 'binary',
    'tl.vmul': 'binary',
    'tl.vdiv': 'binary',
    'tl.vmax': 'binary',
    'tl.vmin': 'binary',
    'tl.vmul_scalar': 'binary_scalar',
    'tl.vadd_scalar': 'binary_scalar', 
    'tl.vmin_scalar': 'binary_scalar', 
    'tl.vmax_scalar': 'binary_scalar', 
    'tl.clamp_min': 'binary_scalar', 
    'tl.clamp_max': 'binary_scalar', 
    'tl.gather': 'gather'
}

group_asc_mappings = {
    'reduce': reduce_api,
    'unary': unary_api,
    'binary': binary_api,
    'binary_scalar': binary_scalar_api,
    'gather': gather_api
}
