#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(
    std::byte* attn_val,
    const std::byte* q,
    const std::byte* k,
    const std::byte* v,
    llaisysDataType_t type,
    size_t B,
    size_t S_q,
    size_t S_kv,
    size_t H_q,
    size_t H_kv,
    size_t D,
    float scale
);
}