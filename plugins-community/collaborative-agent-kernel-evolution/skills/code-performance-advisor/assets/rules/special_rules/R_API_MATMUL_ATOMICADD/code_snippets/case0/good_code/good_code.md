# Good Code

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

  mm.IterateAll(gm_d, 1); // IterateAll接口中的enAtomic设为1
    
  // while (mm. Iterate ()) {
    // mm.GetTensorC(gm_d, 1);     // GetTensorC接口中的enAtomic设为1
  // }
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