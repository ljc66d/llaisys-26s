#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t K, float eps) {
    float w0 = llaisys::utils::cast<float>(weight[0]);
    float x0 = llaisys::utils::cast<float>(in[0]);
  
    for (size_t m = 0; m < M; ++m) {
        const size_t row_offset = m * K;

        // 步骤1：计算该行的平方和，全程使用 float 保证精度
        float sum_sq = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float val = llaisys::utils::cast<float>(in[row_offset + k]);
            sum_sq += val * val;
        }

        // 步骤2：计算 RMS 并求倒数，避免重复开方和除法
        float mean_sq = sum_sq / static_cast<float>(K);
        float rms = std::sqrt(mean_sq + eps);
        float inv_rms = 1.0f / rms;

        // 步骤3：逐元素归一化并乘以权重
        for (size_t k = 0; k < K; ++k) {
            float x = llaisys::utils::cast<float>(in[row_offset + k]);
            float w = llaisys::utils::cast<float>(weight[k]);
            float y = x * inv_rms * w;
            out[row_offset + k] = llaisys::utils::cast<T>(y);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t K, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_(reinterpret_cast<float *>(out),
                  reinterpret_cast<const float *>(in),
                  reinterpret_cast<const float *>(weight),
                  M, K, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                  reinterpret_cast<const llaisys::bf16_t *>(in),
                  reinterpret_cast<const llaisys::bf16_t *>(weight),
                  M, K, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                  reinterpret_cast<const llaisys::fp16_t *>(in),
                  reinterpret_cast<const llaisys::fp16_t *>(weight),
                  M, K, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu