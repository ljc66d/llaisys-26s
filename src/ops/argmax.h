#pragma once

#include "llaisys.h"

#ifdef __cplusplus
extern "C" {
#endif

__export void llaisysArgmax(
    llaisysTensor_t max_idx,
    llaisysTensor_t max_val,
    llaisysTensor_t vals);

#ifdef __cplusplus
}
#endif