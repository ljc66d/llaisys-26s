#include "rms_norm_gpu.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cmath>

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error("CUDA error: " +                     \
                std::string(cudaGetErrorString(_err)));                   \
        }                                                                 \
    } while(0)

// ========== RMSNorm 核函数 ==========
// 每个 block 处理 1 个 token，block 内 256 线程做规约
// 所有中间计算强制使用 FP32，最后写回原精度
template<typename T>
__global__ void rms_norm_kernel(
    T* output,
    const T* input,
    const T* weight,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    if (block_id >= num_tokens) return;

    __shared__ float s_sum[256];

    const T* src = input + block_id * hidden_dim;
    T* dst = output + block_id * hidden_dim;

    // ========== Step 1: 平方和规约（全程 FP32） ==========
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += 256) {
        float val = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            val = src[i];
        } else if constexpr (std::is_same_v<T, half>) {
            val = __half2float(src[i]);
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            val = __bfloat162float(src[i]);
        }
        sum_sq += val * val;
    }

    s_sum[tid] = sum_sq;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            s_sum[tid] += s_sum[tid + step];
        }
        __syncthreads();
    }

    // ========== Step 2: 计算 rms 倒数（FP32） ==========
    float mean_sq = s_sum[0] / hidden_dim;
    float inv_rms = rsqrtf(mean_sq + eps);

    // ========== Step 3: 归一化并乘权重（全程 FP32） ==========
    for (int i = tid; i < hidden_dim; i += 256) {
        float val = 0.0f;
        float gamma = 0.0f;

        if constexpr (std::is_same_v<T, float>) {
            val = src[i];
            gamma = weight[i];
        } else if constexpr (std::is_same_v<T, half>) {
            val = __half2float(src[i]);
            gamma = __half2float(weight[i]);
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            val = __bfloat162float(src[i]);
            gamma = __bfloat162float(weight[i]);
        }

        float result = val * inv_rms * gamma;

        if constexpr (std::is_same_v<T, float>) {
            dst[i] = result;
        } else if constexpr (std::is_same_v<T, half>) {
            dst[i] = __float2half_rn(result);
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            dst[i] = __float2bfloat16_rn(result);
        }
    }
}

// ========== 模板分发 ==========
template<typename T>
static void rms_norm_impl(
    std::byte* output,
    const std::byte* input,
    const std::byte* weight,
    size_t num_tokens,
    size_t hidden_dim,
    float eps
) {
    const int block_size = 256;
    const int grid_size = static_cast<int>(num_tokens);

    rms_norm_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<T*>(output),
        reinterpret_cast<const T*>(input),
        reinterpret_cast<const T*>(weight),
        static_cast<int>(num_tokens),
        static_cast<int>(hidden_dim),
        eps
    );
    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::iluvatar {

// ========== 对外接口 ==========
void rms_norm(
    std::byte* output,
    const std::byte* input,
    const std::byte* weight,
    llaisysDataType_t type,
    size_t num_tokens,
    size_t hidden_dim,
    float eps
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_impl<float>(output, input, weight, num_tokens, hidden_dim, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_impl<half>(output, input, weight, num_tokens, hidden_dim, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_impl<nv_bfloat16>(output, input, weight, num_tokens, hidden_dim, eps);
        break;
    default:
        throw std::runtime_error("Unsupported data type for RMSNorm");
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::iluvatar