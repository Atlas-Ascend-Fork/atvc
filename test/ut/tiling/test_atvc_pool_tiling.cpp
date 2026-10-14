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
#include "pool/host/pool_host.h"
#include "platform_stub.h"

TEST(TestAtvcPoolTiling, HeightCeilDivNoOverflow)
{
    using PoolOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    const ATVC::Layout2Dim tileLayout{1, 256};
    ATVC::PoolParam param{};

    const ATVC::Layout2Dim exactTileHeight{1, 4294967040U};
    EXPECT_TRUE(ATVC::Host::CalcPoolTiling<PoolOpTraits>(exactTileHeight, tileLayout, param));
    EXPECT_EQ(param.totalCnt, exactTileHeight.height);
    EXPECT_EQ(param.tilingData.blockNum, 48U);
    EXPECT_EQ(param.tilingData.tailBlockCnt, 15U);

    const ATVC::Layout2Dim oneRowTailHeight{1, 4294967041U};
    EXPECT_TRUE(ATVC::Host::CalcPoolTiling<PoolOpTraits>(oneRowTailHeight, tileLayout, param));
    EXPECT_EQ(param.totalCnt, oneRowTailHeight.height);
    EXPECT_EQ(param.tilingData.blockNum, 48U);
    EXPECT_EQ(param.tilingData.tailBlockCnt, 16U);
}

TEST(TestAtvcPoolTiling, TailTileExtentDoesNotWrap)
{
    const uint32_t tailOffset = 4294967040U;
    const uint32_t totalHeight = 4294967041U;
    const uint32_t tailExtent = ATVC::Kernel::GetPoolTileExtent(tailOffset, totalHeight, 256U);
    EXPECT_EQ(tailExtent, 1U);
    EXPECT_EQ(tailOffset + tailExtent, totalHeight);
    EXPECT_EQ(ATVC::Kernel::GetPoolTileExtent(4294967041U, 4294967041U, 256U), 0U);
}
