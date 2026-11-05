# aclnndropoutV3

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/random/drop_out_v3)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->
- <term>610Lite</term>：支持

## 功能说明

- 接口功能：训练过程中，按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。

- 计算公式：
  $$
  out_i=\begin{cases}0,&\text { with probability }p \\\frac{1}{1-p}input_i,&\text { with probability }1-p\end{cases}
  $$

- 说明：该接口将从CANN 9.0.0版本开始废弃！

## 函数原型

每个算子分为两段式接口：

```Cpp
aclnnDropoutV3GetWorkspaceSize(
  aclTensor*         input,
  aclTensor*         optionalNoise_Shape,
  double             p,
  int64_t            seedOptional,
  int64_t            offset, aclTensor*       out,
  aclTensor*         maskOut,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnDropoutV3(
  void *workspace,
  uint64_t             workspaceSize,
  const aclOpExecutor *executor,
  aclrtStream          stream)：
```

## aclnnDropoutV3GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 212px">
  <col style="width: 359px">
  <col style="width: 305px">
  <col style="width: 114px">
  <col style="width: 135px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出、属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputs</td>
      <td>变量</td>
      <td>公式中的输入<code>x</code>。</td>
      <td>shape支持0-8维，形如[M, N/2]</td>
      <td>float、FLOAT16、BFLOAT16、INT64</td>
      <td>ND、NC1HW</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>optionalNoise_Shape</td>
      <td>输入</td>
      <td>预留参数，支持传入空指针。</td>
      <td>不支持空Tensor。</td>
      <td>QINT32</td>
      <td>ND</td>
      <td>0</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p</td>
      <td>可选输入</td>
      <td>元素置零的概率。</td>
      <td>默认值为0.5</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seedOptional</td>
      <td>输入</td>
      <td>随机。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>1-4</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数的偏移量，它影响生成的随机数序列的位置。</td>
      <td>取值约束：offset需要被4整除，例如0、4、8 ...，不满足约束会调用失败。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的`out`</td>
      <td>数据类型需要是input可转换的数据类型，shape需要与input一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
      <td>nd</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maskOut</td>
      <td>输出</td>
      <td>bit类型并使用UINT8类型存储的mask数据。</td>
      <td>元素个数需要为(align(input的元素个数,128)/8)；当p=0或p=1场景下，不对传入的maskOut做任何处理；其它场景下以给定的p为置零概率生成mask。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的input、out、maskOut为None。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAMs_INVALID</td>
      <td rowspan="5">161002</td>
      <td>input、out、maskOut的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>p、seed的取值不在支持范围内。</td>
    </tr>
    <tr>
      <td>input维度超过8维。</td>
    </tr>
    <tr>
      <td>input和out的shape不一致。</td>
    </tr>
    <tr>
      <td>maskOut的shape不满足条件。</td>
    </tr>
  </tbody>
  </table>

## aclnnDropoutV3

- **参数说明：**

  - workspace（入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（入参）：在Device侧申请的workspace大小，由第一段接口aclnnDropoutV3GetWorkSpaceSize获取。
  - executor（入参）：op执行器，包含了算子计算流程。
  - stream（入参）：指定执行任务的Stream。

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>不支持BFLOAT16。
- <term>Atlas 训练系列产品</term>：输入参数不支持INT64。

## 支持的产品
<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id8 -->
<!-- npu="910b" id9 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id9 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dropout_v4.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<int64_t> maskOutShape = {16};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* maskOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* optionalNoiseShape = nullptr;
  aclTensor* out = nullptr;
  aclTensor* maskOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  std::vector<uint8_t> maskOutHostData(16, 0);
  double p = 0.5;
  int64_t seed = 1234;
  int64_t offset = 0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maskOut aclTensor
  ret = CreateAclTensor(maskOutHostData, maskOutShape, &maskOutDeviceAddr, aclDataType::ACL_UINT8, &maskOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnDropoutV2第一段接口
  ret = aclnnDropoutV3GetWorkspaceSize(self, optionalNoiseShape, p, seed, offset, out, maskOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDropoutV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnDropoutV2第二段接口
  ret = aclnnDropoutV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDropoutV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(optionalNoiseShape);
  aclDestroyTensor(out);
  aclDestroyTensor(maskOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(maskOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
