/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "pool/kernel/pool_op_template.h"

namespace ATVC {
namespace {
using PoolTestTraits = OpTraits<OpInputs<float>, OpOutputs<float>>;
constexpr Layout2Dim kControlLayout{16384, 16};
constexpr Layout2Dim kOverflowLayout{16400, 16};

template <typename Traits>
struct PoolCopyCompute {
    static constexpr Layout2Dim TILE_LAYOUT{16, 16};
    static constexpr PoolTilePadding TILE_PADDING{8, 8, 1, 1};

    template <typename T>
    __aicore__ inline void operator()(AscendC::LocalTensor<T> input, AscendC::LocalTensor<T> output)
    {
        AscendC::Adds(output, input, static_cast<T>(0), output.GetSize());
    }
};

template <const Layout2Dim &layout, template <typename> class Compute = PoolCopyCompute>
__global__ __aicore__ void PoolCopyKernel(GM_ADDR input, GM_ADDR output, PoolParam *param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = Kernel::PoolOpTemplate<Compute<PoolTestTraits>, layout>();
    op.Run(input, output, param);
}

template <const Layout2Dim &layout, template <typename> class Compute = PoolCopyCompute>
uint32_t RunPoolCopyAndCountMismatches()
{
    constexpr uint32_t tileHeight = Compute<PoolTestTraits>::TILE_LAYOUT.height;
    constexpr uint32_t tileWidth = Compute<PoolTestTraits>::TILE_LAYOUT.width;
    uint32_t elementCount = layout.height * layout.width;
    std::vector<float> input(elementCount);
    std::vector<float> output(elementCount, -1.0f);
    for (uint32_t i = 0; i < elementCount; ++i) {
        input[i] = static_cast<float>(i);
    }

    PoolParam param{};
    param.nBufferNum = 1;
    param.tilingData.blockNum = 1;
    param.tilingData.numPerBlock = ((layout.height + tileHeight - 1) / tileHeight) *
                                   ((layout.width + tileWidth - 1) / tileWidth);
    param.tilingData.tiledCnt = tileHeight * tileWidth;
    param.tilingData.tiledCntH = tileHeight;
    param.tilingData.tiledCntW = tileWidth;
    param.tilingData.height = layout.height;
    param.tilingData.width = layout.width;
    param.totalCnt = elementCount;

    PoolCopyKernel<layout, Compute>(reinterpret_cast<GM_ADDR>(input.data()),
        reinterpret_cast<GM_ADDR>(output.data()),
        &param);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < elementCount; ++i) {
        mismatches += output[i] != input[i];
    }
    return mismatches;
}

class AtvcPoolTestsuite : public testing::Test {
protected:
    void SetUp() override
    {
        AscendC::SetGCoreType(2U);
    }

    void TearDown() override
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvcPoolTestsuite, OutputStrideFitsDataCopyParams)
{
    EXPECT_EQ(RunPoolCopyAndCountMismatches<kControlLayout>(), 0U);
}

TEST_F(AtvcPoolTestsuite, OutputStrideExceedsDataCopyParams)
{
    EXPECT_EQ(RunPoolCopyAndCountMismatches<kOverflowLayout>(), 0U);
}

}  // namespace
}  // namespace ATVC
