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
#include <vector>
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/kernel_utils.h"
#include "reduce/kernel/reduce_op_template.h"
#include "reduce/common/reduce_common.h"
#include "reduce/kernel/reduce_sum_compute.h"

namespace ATVC {
using ReduceOpTraits = OpTraits<OpInputs<float>, OpOutputs<float>>;
using ReduceInt32OpTraits = OpTraits<OpInputs<int32_t>, OpOutputs<int32_t>>;

template <typename Traits, const auto &Policy>
__global__ __aicore__ void ReduceSumCustom(GM_ADDR x, GM_ADDR y, ReduceParam* reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    auto op = Kernel::ReduceOpTemplate<ReduceSumCompute<Traits>, Policy>();
    op.Run(x, y, reduceParam);
}

void TestReduceSumCustom(uint32_t *x, uint32_t *y, ReduceParam *param)
{
    ReduceSumCustom<ReduceOpTraits, REDUCE_POLICY2>(
        reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(x), param);
}

void TestNoReduceCopy(float *x, float *y, ReduceParam *param)
{
    ReduceSumCustom<ReduceOpTraits, REDUCE_POLICY0>(
        reinterpret_cast<uint8_t *>(x), reinterpret_cast<uint8_t *>(y), param);
}

void TestReduceSumInt32(const int32_t *x, int32_t *y)
{
    ReduceParam param{};
    param.tilingData.factorACntPerCore = 1;
    param.tilingData.factorATotalCnt = 1;
    param.tilingData.ubFactorA = 1;
    param.tilingData.factorRCntPerCore = 1;
    param.tilingData.factorRTotalCnt = 1;
    param.tilingData.ubFactorR = 1;
    param.tilingData.groupR = 1;
    param.tilingData.outSize = 1;
    param.tilingData.basicBlock = 49152U;
    param.tilingData.coreNum = 1;
    param.tilingData.shape[0] = 1;
    param.tilingData.shape[1] = 2;
    param.tilingData.stride[0] = 2;
    param.tilingData.stride[1] = 1;
    param.tilingData.dstStride[0] = 1;
    param.tilingData.dstStride[1] = 1;
    param.tilingData.meanVar = 0.5f;
    ReduceSumCustom<ReduceInt32OpTraits, REDUCE_POLICY2>(
        reinterpret_cast<uint8_t *>(const_cast<int32_t *>(x)), reinterpret_cast<uint8_t *>(y), &param);
}

class AtvcReduceSumTestsuite : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2U);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvcReduceSumTestsuite, AtvcReduceSumTestCase)
{
    uint32_t eleNum = 2U;
    uint32_t aGm[eleNum] = {0};
    uint32_t bGm[1] = {0};

    ReduceParam *param = new ReduceParam{};
    param->tilingData.factorACntPerCore = 1;
    param->tilingData.factorATotalCnt = 1;
    param->tilingData.ubFactorA = 1;
    param->tilingData.factorRCntPerCore = 1;
    param->tilingData.factorRTotalCnt = 1;
    param->tilingData.ubFactorR = 1;
    param->tilingData.groupR = 1;
    param->tilingData.outSize = 1;
    param->tilingData.basicBlock = 49152U;
    param->tilingData.coreNum = 1;
    param->tilingData.shape[0] = 1;
    param->tilingData.shape[1] = 2U;
    param->tilingData.stride[0] = 2U;
    param->tilingData.stride[1] = 1;
    TestReduceSumCustom(aGm, bGm, param);
    EXPECT_EQ(bGm[0], 0);
}

TEST_F(AtvcReduceSumTestsuite, NoReduceCopyDoesNotTouchElementsPastOutput)
{
    constexpr float kCanary = -1234.0F;
    std::vector<float> input(16, 99.0F);
    std::vector<float> output(16, kCanary);
    input[0] = 5.25F;

    ReduceParam param{};
    param.tilingData.basicBlock = 49152U;
    param.tilingData.coreNum = 1;
    param.tilingData.shape[0] = 1;

    TestNoReduceCopy(input.data(), output.data(), &param);

    EXPECT_FLOAT_EQ(output[0], input[0]);
    for (size_t i = 1; i < 8; ++i) {
        EXPECT_FLOAT_EQ(output[i], kCanary);
    }
}

TEST_F(AtvcReduceSumTestsuite, NoReduceCopyAlignedControl)
{
    constexpr float kCanary = -1234.0F;
    std::vector<float> input(16, 99.0F);
    std::vector<float> output(16, kCanary);
    for (size_t i = 0; i < 8; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    ReduceParam param{};
    param.tilingData.basicBlock = 49152U;
    param.tilingData.coreNum = 1;
    param.tilingData.shape[0] = 8;

    TestNoReduceCopy(input.data(), output.data(), &param);

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
    for (size_t i = 8; i < 16; ++i) {
        EXPECT_FLOAT_EQ(output[i], kCanary);
    }
}

TEST_F(AtvcReduceSumTestsuite, Int32SumPreservesUnitsAboveFloatPrecisionBoundary)
{
    const int32_t input[2] = {16777216, 1};
    int32_t output = -1;

    TestReduceSumInt32(input, &output);

    EXPECT_EQ(output, 16777217);
}

TEST_F(AtvcReduceSumTestsuite, Int32SumControlAtFloatPrecisionBoundary)
{
    const int32_t input[2] = {16777215, 1};
    int32_t output = -1;

    TestReduceSumInt32(input, &output);

    EXPECT_EQ(output, 16777216);
}
}  // namespace ATVC
