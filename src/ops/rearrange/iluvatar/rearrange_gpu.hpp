#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::iluvatar {
void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size);
}