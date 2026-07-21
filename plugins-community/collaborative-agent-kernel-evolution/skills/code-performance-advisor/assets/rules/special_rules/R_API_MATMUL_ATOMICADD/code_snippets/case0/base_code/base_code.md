# Base Code

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
 __aicore__ inline void MatMulKernel(...)
 {
  ...
  Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
  TPipe pipe;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);

  mm.SetTensorA(gm_a);
  mm.SetTensorB(gm_b);
  mm.SetBias(gm_bias);

  mm.IterateAll(local_c);

  // while (mm.Iterate()) {
    // mm.GetTensorC(local_c);
  // }
    
  DataCopy(local_d, gm_d, d_size);
  event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
  Add(local_d, local_d, local_c, d_size);
  DataCopy(gm_d, local_d, d_size);
  ...
 }

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
   ...
   typedef MatmulType<TPosition::GM, CubeFormat::ND, half> aType; 
   typedef MatmulType<TPosition::GM, CubeFormat::ND, half> bType; 
   typedef MatmulType<TPosition::GM, CubeFormat::ND, float> cType; 
   typedef MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;
   MatMulKernel<aType, bType, cType, biasType)(...);
   ...
 }
```