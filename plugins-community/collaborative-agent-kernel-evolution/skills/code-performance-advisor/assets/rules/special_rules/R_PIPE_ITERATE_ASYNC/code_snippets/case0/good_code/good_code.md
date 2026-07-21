# Good Code

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN>  qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT>  qVecOut;
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.SetWorkspace(workspace, size);//其中，workspace为临时空间的物理地址，size为singleCoreM*singleCoreN大小的矩阵C占用的内存大小：singleCoreM*singleCoreN*sizeof(float)
int16_t scalar = 2;

while(mm.template Iterate<false>()){
  auto cInUB = qVecIn.AllocTensor<float>();
  mm.GetTensorC(cInUB);
  qVecIn.EnQue(cInUB);
  cInUB = qVecIn.Deque<float>();
  auto cOutUB = qVecOut.AllocTensor<float>();
  Muls(cOutUB, cInUB, scalar, baseM*baseN);
  qVecIn.FreeTensor(cInUB);
  ...
}
```