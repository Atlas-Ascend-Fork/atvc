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
#include "elewise/common/elewise_common.h"
#include "elewise/kernel/elewise_op_template.h"

namespace ATVC {
using ADD_OPTRAITS = OpTraits<OpInputs<float, float>, OpOutputs<float>>;

template <typename Traits>
struct AddComputeFunc {
    template <typename T>
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> c)
    {
        AscendC::Add(c, a, b, c.GetSize());
    }
};

template <class Traits>
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, EleWiseParam* param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();
    op.Run(a, b, c, param);
}

void TestAddCustom(float *a, float *b, float *c, EleWiseParam *param)
{
    AddCustom<ADD_OPTRAITS>(reinterpret_cast<uint8_t *>(a),
        reinterpret_cast<uint8_t *>(b),
        reinterpret_cast<uint8_t *>(c),
        param);
}

class AtvcAddTestsuite : public testing::Test {
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

TEST_F(AtvcAddTestsuite, AtvcAddTestCase)
{
    constexpr uint32_t eleNum = 4U;
    float aGm[eleNum] = {1.5f, 2.5f, -1.0f, 0.5f};
    float bGm[eleNum] = {0.5f, -1.0f, 2.0f, 3.5f};
    float cGm[eleNum] = {0.0f};
    float expectResult[eleNum] = {2.0f, 1.5f, 1.0f, 4.0f};

    EleWiseParam *param = new EleWiseParam{};
    param->nBufferNum = 2U;
    param->tilingData.blockNum = 1;
    param->tilingData.numPerBlock = 0;
    param->tilingData.tailBlockCnt = 0;
    param->tilingData.tiledCnt = 32U;
    param->tilingData.tailElemCnt = 4U;
    param->totalCnt = 4U;

    TestAddCustom(aGm, bGm, cGm, param);
    for (uint32_t i = 0; i < eleNum; i++) {
        EXPECT_EQ(cGm[i], expectResult[i]);
    }
    delete param;
}
}  // namespace ATVC