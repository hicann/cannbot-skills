# Base Code

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN>  qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT>  qVecOut;
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
int16_t scalar = 2;

while(mm.template Iterate()){
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