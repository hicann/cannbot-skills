# Good Code

```cpp
template <typename ComputeT> class KernelExample {
 public:
   ...
   __aicore__ inline void Process(...)
   {
     for (int i = 0; i < iLen; ++i) {
       ... 
       auto bindLocal = queBind.AllocTensor<ComputeT>();
       DataCopy(bindLocal, inGm[i * 32], size);
       queBind.EnQue(bindLocal);
       auto bindLocal = queBind.DeQue<ComputeT>();
       for (int j = 0; j < len; ++j) {
         ...
         DataCopyPad(outGm[j], bindLocal, ...);
       }
       queBind.FreeTensor(bindLocal);
     }
   }

 private:
   ... 
   TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind; // 使用TQueBind替换原来QueI，QueO
   ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
   ...
   op.Process(...);
 }
```