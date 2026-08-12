#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(
    std::byte* out,
    const std::byte* in,
    const std::byte* pos_ids,
    llaisysDataType_t type,
    size_t batch,
    size_t seq_len,
    size_t D,
    float theta,
    size_t pos_len
);
}