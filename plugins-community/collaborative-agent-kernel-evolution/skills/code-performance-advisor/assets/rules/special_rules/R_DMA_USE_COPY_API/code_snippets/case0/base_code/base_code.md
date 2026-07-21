# Base Code

```cpp
// 搬运数据存在间隔，从GM上每行16KB中搬运2KB数据, 共16行
LocalTensor<float> tensorIn;
GlobalTensor<float> tensorGM;
...
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
// 使用for循环，每次只能搬运2K，重复16次
for (int i = 0, i < imgHeight; i++) {
    DataCopy(tensorIn[i * copyWidth ], tensorGM[i*imgWidth], copyWidth);
}
```