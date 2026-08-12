#include "linear_gpu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(expr)                                                \
    do {                                                                  \
        cublasStatus_t _s = (expr);                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuBLAS error");                     \
        }                                                                 \
    } while(0)

// ========== 线程局部 cuBLAS 句柄，避免重复创建销毁 ==========
struct CublasHandle {
    cublasHandle_t handle;
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    ~CublasHandle() {
        if (handle) cublasDestroy(handle);
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};
static thread_local CublasHandle g_cublas_handle;

// ========== 模板化 Bias 核函数，消除三套冗余实现 ==========
template<typename T>
__global__ void add_bias_kernel(T* out, const T* bias, size_t batch, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * out_dim;
    if (idx < total) {
        size_t bias_idx = idx % out_dim;
        float val = static_cast<float>(out[idx]) + static_cast<float>(bias[bias_idx]);
        if constexpr (std::is_same_v<T, float>) {
            out[idx] = val;
        } else if constexpr (std::is_same_v<T, __half>) {
            out[idx] = __float2half(val);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[idx] = __float2bfloat16(val);
        }
    }
}

namespace llaisys::ops::iluvatar {

void linear(std::byte* out, const std::byte* x, const std::byte* weight, const std::byte* bias,
            llaisysDataType_t type, size_t batch, size_t in_dim, size_t out_dim) {
    cublasHandle_t handle = g_cublas_handle.handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 标准参数：适配 PyTorch 行优先权重 [out_dim, in_dim]
    const int m = static_cast<int>(out_dim);
    const int n = static_cast<int>(batch);
    const int k = static_cast<int>(in_dim);
    const int lda = static_cast<int>(in_dim);
    const int ldb = static_cast<int>(in_dim);
    const int ldc = static_cast<int>(out_dim);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        using T = float;
        const T* dev_x = reinterpret_cast<const T*>(x);
        const T* dev_w = reinterpret_cast<const T*>(weight);
        const T* dev_b = reinterpret_cast<const T*>(bias);
        T* dev_out = reinterpret_cast<T*>(out);

        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T,   // A 转置：W^T
            CUBLAS_OP_N,   // B 不转置：X
            m, n, k,
            &alpha,
            dev_w, lda,
            dev_x, ldb,
            &beta,
            dev_out, ldc
        ));

        dim3 grid(static_cast<int>((batch * out_dim + 255) / 256));
        dim3 block(256);
        add_bias_kernel<T><<<grid, block>>>(dev_out, dev_b, batch, out_dim);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    case LLAISYS_DTYPE_F16: {
        using T = __half;
        const T* dev_x = reinterpret_cast<const T*>(x);
        const T* dev_w = reinterpret_cast<const T*>(weight);
        const T* dev_b = reinterpret_cast<const T*>(bias);
        T* dev_out = reinterpret_cast<T*>(out);

        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dev_w, CUDA_R_16F, lda,
            dev_x, CUDA_R_16F, ldb,
            &beta,
            dev_out, CUDA_R_16F, ldc,
            CUDA_R_32F,        // FP32 累加保证精度
            CUBLAS_GEMM_DEFAULT
        ));

        dim3 grid(static_cast<int>((batch * out_dim + 255) / 256));
        dim3 block(256);
        add_bias_kernel<T><<<grid, block>>>(dev_out, dev_b, batch, out_dim);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        using T = __nv_bfloat16;
        const T* dev_x = reinterpret_cast<const T*>(x);
        const T* dev_w = reinterpret_cast<const T*>(weight);
        const T* dev_b = reinterpret_cast<const T*>(bias);
        T* dev_out = reinterpret_cast<T*>(out);

        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dev_w, CUDA_R_16BF, lda,
            dev_x, CUDA_R_16BF, ldb,
            &beta,
            dev_out, CUDA_R_16BF, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        ));

        dim3 grid(static_cast<int>((batch * out_dim + 255) / 256));
        dim3 block(256);
        add_bias_kernel<T><<<grid, block>>>(dev_out, dev_b, batch, out_dim);
        CUDA_CHECK(cudaGetLastError());
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::iluvatar