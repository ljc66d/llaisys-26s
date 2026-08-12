#include "swiglu_gpu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error("CUDA error: " +                     \
                std::string(cudaGetErrorString(_err)));                   \
        }                                                                 \
    } while(0)

// ========== f32 逐元素 SwiGLU 核函数 ==========
__global__ void swiglu_f32_kernel(float* out, const float* gate, const float* up, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float g = gate[idx];
        float u = up[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = u * g * sig;
    }
}

// ========== f16 逐元素 SwiGLU 核函数（FP32 中间计算）==========
__global__ void swiglu_f16_kernel(__half* out, const __half* gate, const __half* up, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = __float2half(u * g * sig);
    }
}

// ========== bf16 逐元素 SwiGLU 核函数（FP32 中间计算）==========
__global__ void swiglu_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = __float2bfloat16(u * g * sig);
    }
}

namespace llaisys::ops::iluvatar {

void swiglu(std::byte* out, const std::byte* gate, const std::byte* up,
            llaisysDataType_t type, size_t N) {
    const int block_size = 256;
    const int grid_size = static_cast<int>((N + block_size - 1) / block_size);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        const float* dev_gate = reinterpret_cast<const float*>(gate);
        const float* dev_up = reinterpret_cast<const float*>(up);
        float* dev_out = reinterpret_cast<float*>(out);
        
        swiglu_f32_kernel<<<grid_size, block_size>>>(dev_out, dev_gate, dev_up, N);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const __half* dev_gate = reinterpret_cast<const __half*>(gate);
        const __half* dev_up = reinterpret_cast<const __half*>(up);
        __half* dev_out = reinterpret_cast<__half*>(out);
        
        swiglu_f16_kernel<<<grid_size, block_size>>>(dev_out, dev_gate, dev_up, N);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const __nv_bfloat16* dev_gate = reinterpret_cast<const __nv_bfloat16*>(gate);
        const __nv_bfloat16* dev_up = reinterpret_cast<const __nv_bfloat16*>(up);
        __nv_bfloat16* dev_out = reinterpret_cast<__nv_bfloat16*>(out);
        
        swiglu_bf16_kernel<<<grid_size, block_size>>>(dev_out, dev_gate, dev_up, N);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace llaisys::ops::iluvatar