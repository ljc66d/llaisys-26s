#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t* out_idx, T* out_val, const T* vals, size_t size) {
    T max_val = vals[0];
    int64_t max_idx = 0;
    for (size_t i = 1; i < size; i++) {
        float current = llaisys::utils::cast<float>(vals[i]);
        float best = llaisys::utils::cast<float>(max_val);
        if (current > best) {
            max_val = vals[i]; 
            max_idx = i;
        }
    }
    out_idx[0] = max_idx;
    out_val[0] = max_val;
}
namespace llaisys::ops::cpu {
    void argmax(std::byte* out_idx, std::byte* out_val, const std::byte* vals, llaisysDataType_t type, size_t size) {
        auto *idx_ptr = reinterpret_cast<int64_t *>(out_idx);
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return argmax_(idx_ptr, reinterpret_cast<float *>(out_val), reinterpret_cast<const float *>(vals), size);
        case LLAISYS_DTYPE_BF16:
            return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(out_val),
                        reinterpret_cast<const llaisys::bf16_t *>(vals), size);
        case LLAISYS_DTYPE_F16:
            return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(out_val),
                        reinterpret_cast<const llaisys::fp16_t *>(vals), size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}