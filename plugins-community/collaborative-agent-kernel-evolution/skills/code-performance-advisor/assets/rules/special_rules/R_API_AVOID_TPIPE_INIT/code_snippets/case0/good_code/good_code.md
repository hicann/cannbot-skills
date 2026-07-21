# Good Code

```cpp
template <typename ComputeT> class KernelExample {
 public:
     __aicore__ inline KernelExample() {}

     __aicore__ inline void Init(..., TPipe* pipeIn)
     {
         ...
         pipe = pipeIn;
         pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
         ...
     }

 private:
     ...
     TPipe* pipe;
     ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
     ...
     TPipe pipe;
     KernelExample<float> op;
     op.Init(..., &pipe);
     ...
 }
```