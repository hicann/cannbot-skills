/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_arg_max_with_value_infershape.cpp
 * \brief ArgMaxWithValue InferShape UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ArgMaxWithValueInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArgMaxWithValueInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ArgMaxWithValueInfershape TearDown" << std::endl; }
};

TEST_F(ArgMaxWithValueInfershape, infershape_2d_dim0_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2}, {2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_2d_dim0_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2}, {1, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_3d_dim1_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4}, {2, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_3d_neg_dim_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 4}, {2, 1, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_1d_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_1d_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, infershape_scalar_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxWithValue", {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, argmax_v2_infershape_int64_axis)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr || spaceRegistry->GetOpImpl("ArgMaxV2") == nullptr) {
        GTEST_SKIP() << "ArgMaxV2 not registered, skip";
    }
    std::vector<int64_t> axisValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxV2",
        {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axisValue.data()}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, argmax_v2_infershape_int32_axis)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr || spaceRegistry->GetOpImpl("ArgMaxV2") == nullptr) {
        GTEST_SKIP() << "ArgMaxV2 not registered, skip";
    }
    std::vector<int32_t> axisValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxV2",
        {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axisValue.data()}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ArgMaxWithValueInfershape, argmax_v2_infershape_1d_input)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr || spaceRegistry->GetOpImpl("ArgMaxV2") == nullptr) {
        GTEST_SKIP() << "ArgMaxV2 not registered, skip";
    }
    std::vector<int64_t> axisValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ArgMaxV2",
        {{{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axisValue.data()}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
