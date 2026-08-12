#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(
    std::byte* out,
    const std::byte* input,
    const std::byte* gamma,
    llaisysDataType_t type,
    size_t num_tokens,
    size_t hidden_dim,
    float eps
);
}