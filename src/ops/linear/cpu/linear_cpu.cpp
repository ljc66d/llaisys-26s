#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t K, size_t N) {
    
    for (size_t m = 0; m < M; ++m) {
        // 提前计算当前输入行的起始偏移，避免重复计算
        const size_t in_row_offset = m * K;
        const size_t out_row_offset = m * N;

        for (size_t n = 0; n < N; ++n) {
            // 点积累加，全程使用 float 保证精度
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float in_val = llaisys::utils::cast<float>(in[in_row_offset + k]);
                float w_val = llaisys::utils::cast<float>(weight[n * K + k]);
                sum += in_val * w_val;
            }

            // 仅当 bias 非空时才加偏置，避免空指针访问
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[n]);
            }

            out[out_row_offset + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t K, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_(reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight),
                reinterpret_cast<const float *>(bias),
                M, K, N);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                reinterpret_cast<const llaisys::bf16_t *>(in),
                reinterpret_cast<const llaisys::bf16_t *>(weight),
                reinterpret_cast<const llaisys::bf16_t *>(bias),
                M, K, N);
        break;
    case LLAISYS_DTYPE_F16:
        linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                reinterpret_cast<const llaisys::fp16_t *>(in),
                reinterpret_cast<const llaisys::fp16_t *>(weight),
                reinterpret_cast<const llaisys::fp16_t *>(bias),
                M, K, N);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu