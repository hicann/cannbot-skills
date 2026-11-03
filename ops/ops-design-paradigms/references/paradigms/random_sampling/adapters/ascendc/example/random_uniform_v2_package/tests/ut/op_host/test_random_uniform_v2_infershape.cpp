/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TestRandomUniformV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "random_uniform_v2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "random_uniform_v2 TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

template <typename T>
gert::Tensor* ConstructInputConstTensor(std::unique_ptr<uint8_t[]>& input_tensor_holder,
                                        const std::vector<T>& const_value, ge::DataType const_dtype)
{
    size_t shape_size = const_value.size();
    std::cout << " shape_size:" << shape_size << std::endl;
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{shape_size}, {shape_size}},       // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        const_dtype,                        // dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (size_t i = 0; i < shape_size; i++) {
        tensor_data[i] = const_value[i];
        std::cout << " const_value:" << const_value[i] << std::endl;
    }

    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor;
}

static void ExeTestCase(std::vector<std::vector<int64_t>> expectResults,
                        const std::vector<gert::StorageShape>& inputShapes, // 存储所有输入StorageShape参数
                        const std::vector<ge::DataType>& dtypes,            // 存储所有DataType参数
                        const std::vector<int32_t> input1Values, const std::vector<int64_t> input2Values,
                        std::vector<gert::StorageShape*>& outStorageShape,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& shapeStorageShape = inputShapes[0];
    const auto& offsetStorageShape = inputShapes[1];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType output1Dtype = dtypes[2];

    auto shape_input_tensor =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * input1Values.size()]);
    auto shape_tensor = ConstructInputConstTensor<int32_t>(shape_input_tensor, input1Values, ge::DT_INT32);
    auto offset_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t)]);
    auto offset_tensor = ConstructInputConstTensor<int64_t>(offset_input_tensor, input2Values, ge::DT_INT64);

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {shape_tensor, offset_tensor};

    std::vector<gert::StorageShape*> outputShapes = outStorageShape;
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("RandomUniformV2")
                             .NodeIoNum(2, 2)
                             .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, output1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("RandomUniformV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(TestRandomUniformV2Infershape, random_uniform_v2_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{32, 512}, {32, 512}},
        {{1}, {1}},
    };
    std::vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};

    std::vector<int32_t> input1Values = {32, 512};
    std::vector<int64_t> input2Values = {0};
    std::vector<std::vector<int64_t>> expectResult = {{32, 512}, {1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase(expectResult, inputShapes, dtypes, input1Values, input2Values, outStorageShape, ge::GRAPH_SUCCESS);
}
