/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_dim.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    deviceAddr.reset(rawDeviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    if (rawTensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        return ACL_ERROR_FAILURE;
    }
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    // 如果keepDim的值为true，则indicesShape和outShape的shape为{1, 2}
    std::vector<int64_t> indicesShape = {2};
    std::vector<int64_t> outShape = {2};
    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr indicesDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    TensorPtr self(nullptr, &aclDestroyTensor);
    TensorPtr indices(nullptr, &aclDestroyTensor);
    TensorPtr out(nullptr, &aclDestroyTensor);
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> indicesHostData = {1, 1};
    std::vector<float> outHostData = {1, 1};
    int64_t dim = 0;
    bool keepDim = false;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, indicesShape, aclDataType::ACL_INT32, indicesDeviceAddr, indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMaxDim第一段接口
    ret = aclnnMaxDimGetWorkspaceSize(self.get(), dim, keepDim, out.get(), indices.get(), &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMaxDimGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // 根据第一段接口计算出的workspaceSize申请device内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
        workspaceAddr.reset(rawWorkspaceAddr);
    }
    // 调用aclnnMaxDim第二段接口
    ret = aclnnMaxDim(workspaceAddr.get(), workspaceSize, executor, stream.get());
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMaxDim failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream.get());
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr.get(),
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    return 0;
}
