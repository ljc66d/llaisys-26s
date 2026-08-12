#include "rope_gpu.hpp"
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

// ========== RoPE 核函数：半维度旋转 + 正确的多头部位置映射 ==========
template<typename T>
__global__ void rope_kernel(
    T* output,
    const T* input,
    const long long* pos_ids,   // int64 位置 ID，长度为 seq_len
    int total_tokens,           // seq_len * num_heads
    int num_heads,              // 头数，用于将 token_idx 映射到序列位置
    int head_dim,
    float base
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ops = total_tokens * half_dim;   // 需要处理的 (x_i, x_{i+d/2}) 对数
    if (idx >= total_ops) return;

    int token_idx = idx / half_dim;   // 展平后的 token 索引（0 ~ total_tokens-1）
    int i = idx % half_dim;           // 半维度内的频率索引

    // 关键修正：将展平 token 索引映射回序列位置
    int pos_idx = token_idx / num_heads;
    int pos = static_cast<int>(pos_ids[pos_idx]);

    // 标准频率公式：1 / base^(2i / head_dim)
    float inv_freq = 1.0f / powf(base, 2.0f * i / static_cast<float>(head_dim));
    float angle = pos * inv_freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // 计算 x_i 和 x_{i+d/2} 的内存偏移
    int base_offset = token_idx * head_dim;
    int idx1 = base_offset + i;
    int idx2 = base_offset + i + half_dim;

    float x, y;
    if constexpr (std::is_same_v<T, float>) {
        x = input[idx1];
        y = input[idx2];
    } else if constexpr (std::is_same_v<T, __half>) {
        x = __half2float(input[idx1]);
        y = __half2float(input[idx2]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        x = __bfloat162float(input[idx1]);
        y = __bfloat162float(input[idx2]);
    }

    // 旋转矩阵
    float out_x = x * cos_val - y * sin_val;
    float out_y = y * cos_val + x * sin_val;

    // 写回
    if constexpr (std::is_same_v<T, float>) {
        output[idx1] = out_x;
        output[idx2] = out_y;
    } else if constexpr (std::is_same_v<T, __half>) {
        output[idx1] = __float2half(out_x);
        output[idx2] = __float2half(out_y);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        output[idx1] = __float2bfloat16(out_x);
        output[idx2] = __float2bfloat16(out_y);
    }
}

// ========== 模板分发 ==========
template<typename T>
static void rope_impl(
    std::byte* output,
    const std::byte* input,
    const std::byte* pos_ids,
    size_t batch,      // 调用方将其视为序列长度 (seq_len)
    size_t seq_len,    // 调用方将其视为头数 (num_heads)
    size_t head_dim,
    float base
) {
    size_t total_tokens = batch * seq_len;   // seq_len * num_heads
    int half_dim = static_cast<int>(head_dim / 2);
    size_t total_ops = total_tokens * half_dim;

    const int block_size = 1024;
    int grid_size = static_cast<int>((total_ops + block_size - 1) / block_size);

    rope_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<T*>(output),
        reinterpret_cast<const T*>(input),
        reinterpret_cast<const long long*>(pos_ids),
        static_cast<int>(total_tokens),
        static_cast<int>(seq_len),   // 头数，用于位置映射
        static_cast<int>(head_dim),
        base
    );
    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {

// ========== 对外接口（保持与上层 op.cpp 一致） ==========
void rope(
    std::byte* dst,
    const std::byte* src,
    const std::byte* pos_ids,
    llaisysDataType_t type,
    size_t batch,       // 上层传入的是 seq_len
    size_t seq_len,     // 上层传入的是 num_heads
    size_t head_dim,
    float base,
    size_t base_dim     // 保留接口兼容，实现中忽略
) {
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_impl<float>(dst, src, pos_ids, batch, seq_len, head_dim, base);
        break;
    case LLAISYS_DTYPE_F16:
        rope_impl<__half>(dst, src, pos_ids, batch, seq_len, head_dim, base);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_impl<__nv_bfloat16>(dst, src, pos_ids, batch, seq_len, head_dim, base);
        break;
    default:
        throw std::runtime_error("rope: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia