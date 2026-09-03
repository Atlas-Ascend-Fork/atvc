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
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/kernel_utils.h"
#include "reduce/kernel/reduce_op_template.h"
#include "reduce/common/reduce_common.h"
#include "reduce/kernel/reduce_sum_compute.h"
#include "relu_with_reduce_sum/relu_with_reduce_sum.h"

namespace ATVC {
using ReduceOpTraits = OpTraits<OpInputs<float>, OpOutputs<float>, OpTemps<float>>;

void TestReluWithReduceSumCustom(float *x, float *y, ReduceParam *reduceParam)
{
    // 与 test_reduce_sum 相同的 [1,2] 形状/参数组合，使用其已被验证的 REDUCE_POLICY2
    ReluWithReduceSum<ReduceOpTraits, ATVC::REDUCE_POLICY2>(reinterpret_cast<uint8_t *>(x),
        reinterpret_cast<uint8_t *>(y), *reduceParam);
}

class AtvcReluWithReduceSumTestsuite : public testing::Test {
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

TEST_F(AtvcReluWithReduceSumTestsuite, AtvcReluWithReduceSumTestCase)
{
    constexpr uint32_t eleNum = 2U;
    // 输入含负值以覆盖 relu 截断分支：{-1, 2} + 1 = {0, 3}，求和 3.0，relu(3.0) = 3.0
    constexpr float inputVal0 = -1.0F;
    constexpr float inputVal1 = 2.0F;
    constexpr float reluExpected = 3.0F;
    float aGm[eleNum] = {inputVal0, inputVal1};
    float bGm[1] = {};
    float expectResult[1] = {reluExpected};

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

    TestReluWithReduceSumCustom(aGm, bGm, param);
    EXPECT_FLOAT_EQ(bGm[0], expectResult[0]);
    delete param;
}
}  // namespace ATVC