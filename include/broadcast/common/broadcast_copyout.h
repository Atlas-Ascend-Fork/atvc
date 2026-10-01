/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_copyout.h
 * \brief
 */
#ifndef ATVC_BROADCAST_COPYOUT_H
#define ATVC_BROADCAST_COPYOUT_H

#include <cstdint>
#include "broadcast/common/broadcast_common.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#define ATVC_BROADCAST_COPYOUT_AICORE __aicore__
#else
#define ATVC_BROADCAST_COPYOUT_AICORE
#endif

namespace ATVC {
ATVC_BROADCAST_COPYOUT_AICORE inline uint64_t CalcBroadcastCopyOutOffset(int32_t patternID, uint64_t dimBCount,
                                                                         uint64_t dimASize, uint64_t dimACount,
                                                                         uint64_t dimAStep, uint64_t dimBSize)
{
    if (patternID == ATVC::AB_PATTERN::ABA) {
        return dimBCount * dimASize + dimACount * dimAStep;
    }
    return dimACount * dimAStep * dimBSize + dimBCount;
}
}  // namespace ATVC

#undef ATVC_BROADCAST_COPYOUT_AICORE
#endif  // ATVC_BROADCAST_COPYOUT_H
