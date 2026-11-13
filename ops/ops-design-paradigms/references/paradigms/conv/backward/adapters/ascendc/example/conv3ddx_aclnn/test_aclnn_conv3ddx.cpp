/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file test_aclnn_conv3ddx.cpp
 * @brief conv3ddx 算子 aclnn 调用示例（手写 ACLNN 契约版）
 *
 * 调用契约：dev-package 安装头 aclnn_conv3ddx.h（= build/autogen 生成契约的
 * 逐字副本）—— aclnnConv3ddxGetWorkspaceSize 12 形参按头文件顺序逐位传入：
 *   (inputSize, filter, outBackprop, strides, pads, dilationsOptional, groups,
 *    dataFormatOptional, enableHf32, out, workspaceSize, executor)
 *
 * 用例（spec boundary 同族 1D-as-3D 退化几何，解析可推 golden）：
 *   input_size = [1, 3, 2, 2, 2]（NCDHW），filter = (3,3,1,1,1) NCDHW 单位映射
 *   （w[co][ci][0][0][0] = (co == ci) ? 1 : 0），dedy = (1,3,2,2,2) NCDHW，
 *   strides = [1,1,1,1,1]，pads = [0,0,0,0,0,0]，dilations = [1,1,1,1,1]，
 *   groups = 1，data_format = "NCDHW"，enable_hf32 = false，dtype = float16。
 *   1x1x1 单位核 + stride 1 + pad 0 下 dx[n,c,d,h,w] = dedy[n,c,d,h,w]，
 *   dedy 取值 0.25*((i%5)-2) ∈ {-0.5,-0.25,0,0.25,0.5}（fp16 精确可表示，
 *   值域符合用例分布 (-0.5,0.5)），逐元素比对期望即恒等。
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_conv3ddx.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn/opdev/op_errno.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
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

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

// 宿主侧缓冲持有算子 dtype（fp16_t），按 n*sizeof(T) 双向拷贝、比对时升 double
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat format, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor（format 按契约传入 NCDHW）
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出（按 aclnnConv3ddxGetWorkspaceSize 契约）
    //    inputSize（aclIntArray，1D×5，NCDHW 轴序 [N, Cin, Din, Hin, Win]）
    int64_t inputSizeValues[5] = {1, 3, 2, 2, 2};
    aclIntArray* inputSize = aclCreateIntArray(inputSizeValues, 5U);

    //    filter：(3,3,1,1,1) float16 NCDHW —— 单位映射核 w[co][ci] = (co==ci)
    std::vector<int64_t> filterShape = {3, 3, 1, 1, 1};
    const int64_t filterNum = GetShapeSize(filterShape);
    std::vector<op::fp16_t> filterHostData(static_cast<size_t>(filterNum), op::fp16_t(0.0f));
    for (int64_t co = 0; co < 3; co++) {
        for (int64_t ci = 0; ci < 3; ci++) {
            filterHostData[static_cast<size_t>(co * 3 + ci)] = (co == ci) ? op::fp16_t(1.0f) : op::fp16_t(0.0f);
        }
    }
    aclTensor* filter = nullptr;
    void* filterDeviceAddr = nullptr;
    ret = CreateAclTensor(filterHostData, filterShape, &filterDeviceAddr, aclDataType::ACL_FLOAT16,
                          aclFormat::ACL_FORMAT_NCDHW, &filter);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //    outBackprop（dedy）：(1,3,2,2,2) float16 NCDHW，0.25*((i%5)-2) 精确值
    std::vector<int64_t> dedyShape = {1, 3, 2, 2, 2};
    const int64_t dedyNum = GetShapeSize(dedyShape);
    std::vector<op::fp16_t> dedyHostData(static_cast<size_t>(dedyNum), op::fp16_t(0.0f));
    for (int64_t i = 0; i < dedyNum; i++) {
        dedyHostData[static_cast<size_t>(i)] = op::fp16_t(0.25f * static_cast<float>((i % 5) - 2));
    }
    aclTensor* outBackprop = nullptr;
    void* outBackpropDeviceAddr = nullptr;
    ret = CreateAclTensor(dedyHostData, dedyShape, &outBackpropDeviceAddr, aclDataType::ACL_FLOAT16,
                          aclFormat::ACL_FORMAT_NCDHW, &outBackprop);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //    out（y 容器）：shape = inputSize 元素值 [1,3,2,2,2]，dtype/format 随 dedy
    std::vector<int64_t> outShape = {1, 3, 2, 2, 2};
    const int64_t outNum = GetShapeSize(outShape);
    std::vector<op::fp16_t> outHostData(static_cast<size_t>(outNum), op::fp16_t(0.0f));
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_NCDHW,
                          &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //    属性（OpDef 声明序：strides/pads/dilations/groups/data_format/enable_hf32）
    int64_t stridesValues[5] = {1, 1, 1, 1, 1};
    aclIntArray* strides = aclCreateIntArray(stridesValues, 5U);
    int64_t padsValues[6] = {0, 0, 0, 0, 0, 0};
    aclIntArray* pads = aclCreateIntArray(padsValues, 6U);
    int64_t dilationsValues[5] = {1, 1, 1, 1, 1};
    aclIntArray* dilationsOptional = aclCreateIntArray(dilationsValues, 5U);
    int64_t groups = 1;
    char dataFormatOptional[] = "NCDHW"; // 可写缓冲（契约形参为 char*）
    bool enableHf32 = false;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. 调用aclnnConv3ddx第一段接口（12 形参按安装头契约顺序逐位传入）
    ret = aclnnConv3ddxGetWorkspaceSize(inputSize, filter, outBackprop, strides, pads, dilationsOptional, groups,
                                        dataFormatOptional, enableHf32, out, &workspaceSize, &executor);
    if (!(ret == ACLNN_SUCCESS)) {
        LOG_PRINT("aclnnConv3ddxGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("aclnnConv3ddxGetWorkspaceSize ok, workspaceSize = %lu.\n", static_cast<unsigned long>(workspaceSize));

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    // 5. 调用aclnnConv3ddx第二段接口
    ret = aclnnConv3ddx(workspaceAddr, workspaceSize, executor, stream);
    if (!(ret == ACLNN_SUCCESS)) {
        LOG_PRINT("aclnnConv3ddx failed. ERROR: %d\n", ret);
        return ret;
    }

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 7. 获取输出：device→host（n*sizeof(fp16_t)），逐元素与解析 golden 比对
    //    golden：dx[n,c,d,h,w] = dedy[n,c,d,h,w]（1x1x1 单位核、stride 1、pad 0）
    std::vector<op::fp16_t> resultData(static_cast<size_t>(outNum), op::fp16_t(0.0f));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      static_cast<size_t>(outNum) * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    int64_t failedCnt = 0;
    for (int64_t i = 0; i < outNum; i++) {
        double got = static_cast<double>(resultData[static_cast<size_t>(i)]);
        double expect = static_cast<double>(dedyHostData[static_cast<size_t>(i)]);
        if (got != expect) {
            failedCnt++;
            LOG_PRINT("MISMATCH idx[%ld]: dedy = %f, result = %f (expect %f)\n", static_cast<long>(i),
                      static_cast<double>(dedyHostData[static_cast<size_t>(i)]), got, expect);
        }
    }
    for (int64_t i = 0; i < outNum; i++) {
        LOG_PRINT("conv3ddx out_backprop[%ld] is: %f, filter is identity 1x1x1, result[%ld] is: %f\n",
                  static_cast<long>(i), static_cast<double>(dedyHostData[static_cast<size_t>(i)]), static_cast<long>(i),
                  static_cast<double>(resultData[static_cast<size_t>(i)]));
    }

    // 8. 释放aclTensor
    aclDestroyTensor(filter);
    aclDestroyTensor(outBackprop);
    aclDestroyTensor(out);
    aclDestroyIntArray(inputSize);
    aclDestroyIntArray(strides);
    aclDestroyIntArray(pads);
    aclDestroyIntArray(dilationsOptional);

    // 9. 释放device资源
    aclrtFree(filterDeviceAddr);
    aclrtFree(outBackpropDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 10. acl去初始化
    aclFinalize();

    if (failedCnt != 0) {
        LOG_PRINT("conv3ddx aclnn case FAILED: %ld / %ld elements mismatched.\n", static_cast<long>(failedCnt),
                  static_cast<long>(outNum));
        return 1;
    }
    LOG_PRINT("conv3ddx aclnn case PASSED: all %ld elements match the analytic golden (identity 1x1x1 filter).\n",
              static_cast<long>(outNum));
    return 0;
}
