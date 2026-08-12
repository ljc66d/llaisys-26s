#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <vector>
template <typename T>

void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t batch, size_t seq_len, size_t D, float theta, size_t pos_len) {
    size_t half = D / 2;
  
    std::vector<float> pow_theta(half);
    for (size_t k = 0; k < half; ++k) {
        pow_theta[k] = std::pow(theta, 2.0f * static_cast<float>(k) / static_cast<float>(D));
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t pos;
            if (pos_len == batch) {
                pos = pos_ids[b];
            } else if (pos_len == seq_len) {
                pos = pos_ids[s];
            } else {
                pos = pos_ids[b * seq_len + s];
            }
            float pos_f = static_cast<float>(pos);
            size_t base = (b * seq_len + s) * D;
            for (size_t k = 0; k < half; ++k) {
                float angle = pos_f / pow_theta[k]; 
                size_t idx0 = base + k;
                size_t idx1 = base + k + half;
                float x0 = llaisys::utils::cast<float>(in[idx0]);
                float x1 = llaisys::utils::cast<float>(in[idx1]);
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                out[idx0] = llaisys::utils::cast<T>(x0 * cos_a - x1 * sin_a);
                out[idx1] = llaisys::utils::cast<T>(x0 * sin_a + x1 * cos_a);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t batch, size_t seq_len, size_t D, float theta, size_t pos_len) {
    const auto *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ptr, batch, seq_len, D, theta, pos_len);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ptr, batch, seq_len, D, theta, pos_len);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ptr, batch, seq_len, D, theta, pos_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu