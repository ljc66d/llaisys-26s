#include "add_gpu.hpp"
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

__global__ void add_f32_kernel(float* c, const float* a, const float* b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_bf16_kernel(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float fa = __bfloat162float(a[idx]);
        float fb = __bfloat162float(b[idx]);
        c[idx] = __float2bfloat16(fa + fb);
    }
}

__global__ void add_fp16_kernel(half* c, const half* a, const half* b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float fa = __half2float(a[idx]);
        float fb = __half2float(b[idx]);
        c[idx] = __float2half(fa + fb);
    }
}

namespace llaisys::ops::nvidia {

void add(std::byte* c, const std::byte* a, const std::byte* b, llaisysDataType_t type, size_t numel) {
    const int block_size = 256;
    
    const int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float*>(c),
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(b),
            numel
        );
        break;
    case LLAISYS_DTYPE_BF16:
        add_bf16_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<nv_bfloat16*>(c),
            reinterpret_cast<const nv_bfloat16*>(a),
            reinterpret_cast<const nv_bfloat16*>(b),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        add_fp16_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<half*>(c),
            reinterpret_cast<const half*>(a),
            reinterpret_cast<const half*>(b),
            numel
        );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia