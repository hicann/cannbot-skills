# Manual Matmul Implementation (Without Mmad API)

## When to Use This

For matmul operations where the Mmad API is too complex or not suitable, use manual dot product implementation with ReduceSum.

**Use Cases:**
- Small matrix sizes
- Matmul followed by activation (e.g., MOE Router: softplus(input @ weight^T))
- Custom matmul variants
- Learning/debugging purposes

## Complete Example: Matrix Multiplication with Activation

### Input DSL
```python
def module_fn(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute: output = activation(input @ weight^T)
    input: [batch, hidden_dim]
    weight: [num_experts, hidden_dim]
    output: [batch, num_experts]
    """
    logits = torch.matmul(input, weight.t())
    output = torch.nn.functional.softplus(logits)
    return output
```

### AscendC Implementation

```cpp
__aicore__ inline void Process()
{
    uint32_t outputSize = block_batch * block_experts;
    
    // Step 1: Initialize result buffer with zeros
    AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();
    AscendC::Duplicate(resultLocal, 0.0f, outputSize);
    
    // Step 2: Perform matmul: input @ weight^T across hidden dimension
    // Loop over hidden dimension tiles
    for (uint32_t hb = 0; hb < hidden_blocks; hb++) {
        uint32_t h_offset = hb * hidden_tile;
        
        // Step 2a: Load input tile [block_batch, hidden_tile]
        AscendC::LocalTensor<float> inputLocal = inputQueue.AllocTensor<float>();
        for (uint32_t b = 0; b < block_batch; b++) {
            AscendC::DataCopyPad(inputLocal[b * hidden_tile], 
                                inputGm[b * hidden_dim + h_offset], 
                                {1, static_cast<uint16_t>(hidden_tile * sizeof(float)), 0, 0}, 
                                {false, 0, 0, 0});
        }
        inputQueue.EnQue(inputLocal);
        
        // Step 2b: Load weight tile [block_experts, hidden_tile]
        AscendC::LocalTensor<float> weightLocal = weightQueue.AllocTensor<float>();
        for (uint32_t e = 0; e < block_experts; e++) {
            AscendC::DataCopyPad(weightLocal[e * hidden_tile], 
                                weightGm[e * hidden_dim + h_offset],
                                {1, static_cast<uint16_t>(hidden_tile * sizeof(float)), 0, 0},
                                {false, 0, 0, 0});
        }
        weightQueue.EnQue(weightLocal);
        
        // Step 2c: Dequeue for computation
        AscendC::LocalTensor<float> inputComp = inputQueue.DeQue<float>();
        AscendC::LocalTensor<float> weightComp = weightQueue.DeQue<float>();
        
        // Step 2d: Compute dot products for each (batch, expert) pair
        AscendC::LocalTensor<float> tempLocal = tempBuf.Get<float>();
        AscendC::LocalTensor<float> sumBuf = expBuf.Get<float>();
        
        for (uint32_t b = 0; b < block_batch; b++) {
            for (uint32_t e = 0; e < block_experts; e++) {
                // Element-wise multiply: input[b, :] * weight[e, :]
                AscendC::Mul(tempLocal, 
                            inputComp[b * hidden_tile], 
                            weightComp[e * hidden_tile], 
                            hidden_tile);
                
                // Sum the products to get dot product
                AscendC::ReduceSum(sumBuf, tempLocal, tempLocal, hidden_tile);
                float dotProduct = sumBuf.GetValue(0);
                
                // Accumulate to result (important for multi-tile case!)
                uint32_t outIdx = b * block_experts + e;
                float currentVal = resultLocal.GetValue(outIdx);
                resultLocal.SetValue(outIdx, currentVal + dotProduct);
            }
        }
        
        // Step 2e: Free tensors
        inputQueue.FreeTensor(inputComp);
        weightQueue.FreeTensor(weightComp);
    }
    
    // Step 3: Apply activation (softplus: log(1 + exp(x)))
    AscendC::LocalTensor<float> expLocal = expBuf.Get<float>();
    AscendC::Exp(expLocal, resultLocal, outputSize);           // exp(x)
    AscendC::Adds(expLocal, expLocal, 1.0f, outputSize);       // 1 + exp(x)
    AscendC::Log(resultLocal, expLocal, outputSize);           // log(1 + exp(x))
    
    // Step 4: Copy result to output queue
    AscendC::LocalTensor<float> outputLocal = outputQueue.AllocTensor<float>();
    AscendC::DataCopy(outputLocal, resultLocal, outputSize);
    outputQueue.EnQue(outputLocal);
    
    // Step 5: Copy to global memory
    AscendC::LocalTensor<float> outputLocalFinal = outputQueue.DeQue<float>();
    for (uint32_t b = 0; b < block_batch; b++) {
        AscendC::DataCopy(outputGm[b * num_experts], 
                         outputLocalFinal[b * block_experts], 
                         block_experts);
    }
    outputQueue.FreeTensor(outputLocalFinal);
}
```

## Key Implementation Points

### 1. Proper Accumulation Across Tiles
```cpp
// WRONG: Overwrites previous tiles
resultLocal.SetValue(outIdx, dotProduct);

// CORRECT: Accumulates across tiles
float currentVal = resultLocal.GetValue(outIdx);
resultLocal.SetValue(outIdx, currentVal + dotProduct);
```

### 2. Use ReduceSum for Dot Product
```cpp
// WRONG: Manual loop (slow and error-prone)
float sum = 0.0f;
for (uint32_t h = 0; h < hidden_tile; h++) {
    sum += tempLocal.GetValue(h);
}

// CORRECT: Use vector reduction
AscendC::ReduceSum(sumBuf, tempLocal, tempLocal, hidden_tile);
float sum = sumBuf.GetValue(0);
```

### 3. Proper Buffer Management
```cpp
// Allocate buffers in Init():
pipe.InitBuffer(tempBuf, block_batch * block_experts * sizeof(float));
pipe.InitBuffer(expBuf, block_batch * block_experts * sizeof(float));
pipe.InitBuffer(resultBuf, block_batch * block_experts * sizeof(float));

// Get buffers in Process():
AscendC::LocalTensor<float> tempLocal = tempBuf.Get<float>();
AscendC::LocalTensor<float> expLocal = expBuf.Get<float>();
AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();
```

### 4. Correct Activation Implementation
```cpp
// WRONG: Approximation
float activated = (x > 10.0f) ? x : 0.693147f + x * 0.5f;

// CORRECT: Use proper AscendC APIs
AscendC::Exp(expLocal, resultLocal, outputSize);
AscendC::Adds(expLocal, expLocal, 1.0f, outputSize);
AscendC::Log(resultLocal, expLocal, outputSize);
```

## Performance Considerations

1. **Tile Size**: Choose `hidden_tile` to fit in L1 cache
   - Typical: 128-256 elements
   - Balance: Larger tiles = fewer iterations, but more memory

2. **Loop Order**: Optimize for memory access patterns
   - Current: batch → expert → hidden (good for row-major)
   - Alternative: expert → batch → hidden (better for some cases)

3. **Buffer Reuse**: Reuse buffers when possible
   - `tempLocal` can be reused for different operations
   - `expBuf` can serve as workspace for ReduceSum

## Common Pitfalls

❌ **Forgetting to accumulate across tiles**
```cpp
// This loses data from previous tiles!
resultLocal.SetValue(outIdx, dotProduct);
```

❌ **Using wrong tensor dimensions**
```cpp
// Wrong: Using hidden_dim instead of hidden_tile
AscendC::Mul(temp, input[b * hidden_dim], weight[e * hidden_dim], hidden_tile);
```

❌ **Not loading all tiles**
```cpp
// Wrong: Only loading first tile
AscendC::DataCopy(inputLocal, inputGm, hidden_tile);

// Correct: Loop over all tiles
for (uint32_t hb = 0; hb < hidden_blocks; hb++) {
    // Load tile hb
}
```

## Verification Checklist

- [ ] All input tiles are loaded (loop over hidden_blocks)
- [ ] Accumulation is done correctly (currentVal + dotProduct)
- [ ] ReduceSum is used for dot products (not manual loops)
- [ ] Activations use proper AscendC APIs
- [ ] Buffer sizes are correct
- [ ] Memory access patterns are efficient
- [ ] All tensors are freed properly

## Expected Accuracy

With this implementation:
- **Max absolute error**: < 1e-5
- **Mean absolute error**: < 1e-6
- **Relative error**: < 1%

If accuracy is worse, check:
1. Accumulation logic
2. Tile loading offsets
3. Activation implementation
4. Buffer initialization
