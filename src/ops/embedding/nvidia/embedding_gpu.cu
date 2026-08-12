#include "embedding_gpu.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error("CUDA error: " +                     \
                std::string(cudaGetErrorString(_err)));                   \
        }                                                                 \
    } while(0)

// ========== 逐元素查表核函数 ==========
template<typename T>
__global__ void embedding_kernel(
    T* output,
    const int64_t* token_ids,
    const T* weight,
    int num_tokens,
    int hidden_dim
) {
    int idx = blockIdx.x * 1024 + threadIdx.x;
    int total_elem = num_tokens * hidden_dim;
    if (idx >= total_elem) return;

    int token_idx = idx / hidden_dim;
    int dim_idx = idx % hidden_dim;

    int64_t token = token_ids[token_idx];
    output[idx] = weight[token * hidden_dim + dim_idx];
}

// ========== 模板分发 ==========
template<typename T>
static void embedding_impl(
    std::byte* out,
    const std::byte* token_ids,
    const std::byte* weight,
    size_t num_tokens,
    size_t hidden_dim
) {
    const int block_size = 1024;
    size_t total_elem = num_tokens * hidden_dim;
    const int grid_size = static_cast<int>((total_elem + block_size - 1) / block_size);

    embedding_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<T*>(out),
        reinterpret_cast<const int64_t*>(token_ids),
        reinterpret_cast<const T*>(weight),
        static_cast<int>(num_tokens),
        static_cast<int>(hidden_dim)
    );
    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {

void embedding(
    std::byte* output,
    const std::byte* token_ids,
    const std::byte* weight,
    llaisysDataType_t type,
    size_t num_tokens,
    size_t hidden_dim
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_impl<float>(output, token_ids, weight, num_tokens, hidden_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl<half>(output, token_ids, weight, num_tokens, hidden_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl<nv_bfloat16>(output, token_ids, weight, num_tokens, hidden_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace