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
 * @file test_aclnn_conv3dtranspose.cpp
 * @brief conv3dtranspose 算子 aclnn 调用示例（手写 ACLNN 契约版）
 *
 * 调用契约：dev-package 安装头 aclnn_conv3dtranspose.h（=
 * REQUIREMENTS §8.2.1 / design target_aclnn_header.h 基准）——
 * aclnnConv3DTransposeGetWorkspaceSize 16 形参按头文件顺序逐位传入：
 *   (inputSize, x, filter, bias, offsetW, strides, pads, dilations, groups,
 *    dataFormat, outputPadding, offsetX, enableHf32, y, workspaceSize, executor)
 * 其中 inputSize 为 1D×5 int32 ND 设备张量（值依赖——本例张量值即
 * [1,3,2,2,2]），bias/offsetW 为可选（本例 nullptr：无 bias；offsetW 保留
 * 不启用）。
 *
 * 用例（spec boundary 同族 1D-as-3D 退化几何，解析可推 golden）：
 *   input_size = [1, 3, 2, 2, 2]（NCDHW），filter = (3,3,1,1,1) NCDHW 单位
 *   映射核（w[co][ci][0][0][0] = (co == ci) ? 1 : 0），x = (1,3,2,2,2)
 *   NCDHW，strides = [1,1,1,1,1]，pads = [0,0,0,0,0,0]，
 *   dilations = [1,1,1,1,1]，groups = 1，data_format = "NCDHW"，
 *   output_padding = [0,0,0,0,0]，offset_x = 0，enable_hf32 = false，
 *   dtype = float16。
 *   1x1x1 单位核 + stride 1 + pad 0 下转置卷积退化为恒等映射：
 *   y[n,ci,d,h,w] = x[n,co=ci,d,h,w]，x 取值 0.25*((i%5)-2) ∈
 *   {-0.5,-0.25,0,0.25,0.5}（fp16 精确可表示，值域符合用例分布
 *   (-0.5,0.5)），逐元素比对期望即恒等。
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_conv3dtranspose.h"
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

// 宿主侧缓冲持有算子 dtype（fp16_t / int32_t），按 n*sizeof(T) 双向拷贝、比对时升 double
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

    // 调用aclCreateTensor接口创建aclTensor（format 按契约传入）
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

    // 2. 构造输入与输出（按 aclnnConv3DTransposeGetWorkspaceSize 契约）
    //    inputSize：1D×5 int32 ND 设备张量（值依赖——张量值即目标形状 [1,3,2,2,2]）
    std::vector<int64_t> inputSizeShape = {5};
    std::vector<int32_t> inputSizeHostData = {1, 3, 2, 2, 2};
    aclTensor* inputSize = nullptr;
    void* inputSizeDeviceAddr = nullptr;
    ret = CreateAclTensor(inputSizeHostData, inputSizeShape, &inputSizeDeviceAddr, aclDataType::ACL_INT32,
                          aclFormat::ACL_FORMAT_ND, &inputSize);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //    x（dedy）：(1,3,2,2,2) float16 NCDHW，0.25*((i%5)-2) 精确值
    std::vector<int64_t> xShape = {1, 3, 2, 2, 2};
    const int64_t xNum = GetShapeSize(xShape);
    std::vector<op::fp16_t> xHostData(static_cast<size_t>(xNum), op::fp16_t(0.0f));
    for (int64_t i = 0; i < xNum; i++) {
        xHostData[static_cast<size_t>(i)] = op::fp16_t(0.25f * static_cast<float>((i % 5) - 2));
    }
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_NCDHW, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

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

    //    bias / offsetW：可选槽位，本例 nullptr（无 bias；offsetW 保留不启用）
    aclTensor* bias = nullptr;
    aclTensor* offsetW = nullptr;

    //    y（输出容器）：shape = inputSize 元素值 [1,3,2,2,2]，dtype/format 随 x
    std::vector<int64_t> yShape = {1, 3, 2, 2, 2};
    const int64_t yNum = GetShapeSize(yShape);
    std::vector<op::fp16_t> yHostData(static_cast<size_t>(yNum), op::fp16_t(0.0f));
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_NCDHW, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //    属性（OpDef 声明序：strides/pads/dilations/groups/data_format/
    //    output_padding/offset_x/enable_hf32）
    int64_t stridesValues[5] = {1, 1, 1, 1, 1};
    aclIntArray* strides = aclCreateIntArray(stridesValues, 5U);
    int64_t padsValues[6] = {0, 0, 0, 0, 0, 0};
    aclIntArray* pads = aclCreateIntArray(padsValues, 6U);
    int64_t dilationsValues[5] = {1, 1, 1, 1, 1};
    aclIntArray* dilations = aclCreateIntArray(dilationsValues, 5U);
    int64_t groups = 1;
    char dataFormat[] = "NCDHW"; // 可写缓冲（契约形参为 char*）
    int64_t outputPaddingValues[5] = {0, 0, 0, 0, 0};
    aclIntArray* outputPadding = aclCreateIntArray(outputPaddingValues, 5U);
    int64_t offsetX = 0;
    bool enableHf32 = false;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. 调用aclnnConv3DTranspose第一段接口（16 形参按安装头契约顺序逐位传入）
    ret = aclnnConv3DTransposeGetWorkspaceSize(inputSize, x, filter, bias, offsetW, strides, pads, dilations, groups,
                                               dataFormat, outputPadding, offsetX, enableHf32, y, &workspaceSize,
                                               &executor);
    if (!(ret == ACLNN_SUCCESS)) {
        LOG_PRINT("aclnnConv3DTransposeGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("aclnnConv3DTransposeGetWorkspaceSize ok, workspaceSize = %lu.\n",
              static_cast<unsigned long>(workspaceSize));

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    // 5. 调用aclnnConv3DTranspose第二段接口
    ret = aclnnConv3DTranspose(workspaceAddr, workspaceSize, executor, stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnConv3DTranspose failed. ERROR: %d\n", ret);
        return ret;
    }

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 7. 获取输出：device→host（n*sizeof(fp16_t)），逐元素与解析 golden 比对
    //    golden：y[n,ci,d,h,w] = x[n,ci,d,h,w]（1x1x1 单位核、stride 1、pad 0
    //    下转置卷积退化为恒等映射）
    std::vector<op::fp16_t> resultData(static_cast<size_t>(yNum), op::fp16_t(0.0f));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      static_cast<size_t>(yNum) * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    int64_t failedCnt = 0;
    for (int64_t i = 0; i < yNum; i++) {
        double got = static_cast<double>(resultData[static_cast<size_t>(i)]);
        double expect = static_cast<double>(xHostData[static_cast<size_t>(i)]);
        if (got != expect) {
            failedCnt++;
            LOG_PRINT("MISMATCH idx[%ld]: x = %f, result = %f (expect %f)\n", static_cast<long>(i),
                      static_cast<double>(xHostData[static_cast<size_t>(i)]), got, expect);
        }
    }
    for (int64_t i = 0; i < yNum; i++) {
        LOG_PRINT("conv3dtranspose x[%ld] is: %f, filter is identity 1x1x1, result[%ld] is: %f\n", static_cast<long>(i),
                  static_cast<double>(xHostData[static_cast<size_t>(i)]), static_cast<long>(i),
                  static_cast<double>(resultData[static_cast<size_t>(i)]));
    }

    // 8. 释放aclTensor
    aclDestroyTensor(inputSize);
    aclDestroyTensor(x);
    aclDestroyTensor(filter);
    aclDestroyTensor(y);
    aclDestroyIntArray(strides);
    aclDestroyIntArray(pads);
    aclDestroyIntArray(dilations);
    aclDestroyIntArray(outputPadding);

    // 9. 释放device资源
    aclrtFree(inputSizeDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(filterDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 10. acl去初始化
    aclFinalize();

    if (failedCnt != 0) {
        LOG_PRINT("conv3dtranspose aclnn case FAILED: %ld / %ld elements mismatched.\n", static_cast<long>(failedCnt),
                  static_cast<long>(yNum));
        return 1;
    }
    LOG_PRINT(
        "conv3dtranspose aclnn case PASSED: all %ld elements match the analytic golden (identity 1x1x1 filter).\n",
        static_cast<long>(yNum));
    return 0;
}
