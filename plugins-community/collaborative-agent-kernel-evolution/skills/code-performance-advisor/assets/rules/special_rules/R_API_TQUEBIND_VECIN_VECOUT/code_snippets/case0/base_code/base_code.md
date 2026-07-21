# Base Code

```cpp
template <typename ComputeT> class KernelExample {
 public:
   ...
   __aicore__ inline void Process(...)
   {
     for (int i = 0; i < iLen; ++i) {
       ... 
       auto iLocal = QueI.AllocTensor<ComputeT>();
       DataCopy(iLocal, inGm[i * 32], size);
       QueI.EnQue(iLocal);
       auto iLocal = QueI.DeQue<ComputeT>();
       for (int j = 0; j < jLen; ++j) { 
         ...
         auto oLocal = QueO.AllocTensor<ComputeT>();
         DataCopy(oLocal, iLocal, size); // LocalTensor -> LocalTensor的DataCopy指令,以实现数据从VECIN到VECOUT的搬移
         QueO.EnQue(oLocal);

         auto oLocal = QueO.DeQue<ComputeT>();
         DataCopyPad(outGm[j], oLocal, ...);
         QueO.FreeTensor(oLocal);
       }
       QueI.FreeTensor(iLocal);
     }
   }

 private:
   ... 
   TQue<QuePosition::VECIN, BUFFER_NUM> QueI;
   TQue<QuePosition::VECOUT, BUFFER_NUM> QueO;
   ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
   ...
   op.Process(...);
 }
```