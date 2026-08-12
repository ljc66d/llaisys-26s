#include "self_attention_gpu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            char buf[256];                                                \
            snprintf(buf, sizeof(buf), "CUDA error: %s", cudaGetErrorString(_err)); \
            throw std::runtime_error(buf);                                 \
        }                                                                 \
    } while(0)

#define CUBLAS_CHECK(expr)                                                \
    do {                                                                  \
        cublasStatus_t _s = (expr);                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuBLAS error");                     \
        }                                                                 \
    } while(0)

// ========== RAII 封装的 cuBLAS 句柄（线程局部） ==========
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
static cublasHandle_t get_handle() {
    return g_cublas_handle.handle;
}

// ========== 设备内存 RAII 包装 ==========
struct CudaDeleter {
    void operator()(void* ptr) const noexcept { cudaFree(ptr); }
};
template <typename T>
using unique_cuda_ptr = std::unique_ptr<T, CudaDeleter>;

// ========== FP16 <-> FP32 转换核函数 ==========
__global__ void f16_to_f32_kernel(float* dst, const __half* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void f32_to_f16_kernel(__half* dst, const float* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void bf16_to_f32_kernel(float* dst, const __nv_bfloat16* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void f32_to_bf16_kernel(__nv_bfloat16* dst, const float* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ========== 通用的 causal softmax 模板核函数（仅 float 用于精度） ==========
template <typename T_SCORE>
__global__ void causal_softmax_kernel(T_SCORE* out, const T_SCORE* scores,
                                      size_t S_q, size_t S_kv, int mask_offset) {
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;
    int col = blockIdx.x;

    float max_val = -1e30f;
    for (size_t j = tid; j < S_kv; j += blockDim.x) {
        int j_causal = static_cast<int>(j) - mask_offset;
        float val = (j_causal <= col) ? static_cast<float>(scores[col * S_kv + j]) : -1e30f;
        max_val = max(max_val, val);
    }

    s_max[tid] = max_val;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            s_max[tid] = max(s_max[tid], s_max[tid + step]);
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    float sum_exp = 0.0f;
    for (size_t j = tid; j < S_kv; j += blockDim.x) {
        int j_causal = static_cast<int>(j) - mask_offset;
        if (j_causal <= col) {
            float val = static_cast<float>(scores[col * S_kv + j]);
            sum_exp += expf(val - row_max);
        }
    }

    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            s_sum[tid] += s_sum[tid + step];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0] + 1e-9f;

    for (size_t j = tid; j < S_kv; j += blockDim.x) {
        int j_causal = static_cast<int>(j) - mask_offset;
        if (j_causal <= col) {
            float val = static_cast<float>(scores[col * S_kv + j]);
            float res = expf(val - row_max) / row_sum;
            out[col * S_kv + j] = static_cast<T_SCORE>(res);
        } else {
            out[col * S_kv + j] = static_cast<T_SCORE>(0.0f);
        }
    }
}

namespace llaisys::ops::iluvatar {

void self_attention(std::byte* attn_val, const std::byte* q, const std::byte* k, const std::byte* v,
                    llaisysDataType_t type,
                    size_t B, size_t S_q, size_t S_kv,
                    size_t H_q, size_t H_kv, size_t D, float scale) {
    cublasHandle_t handle = get_handle();

    if (H_q % H_kv != 0) {
        throw std::runtime_error("Number of query heads must be a multiple of key/value heads in GQA.");
    }
    size_t n_repeat = H_q / H_kv;
    int mask_offset = static_cast<int>(S_kv) - static_cast<int>(S_q);
    const float beta = 0.0f;
    const float alpha_one = 1.0f;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        // FP32: 全 FP32 路径，无需转换
        size_t score_size = S_q * S_kv * sizeof(float);
        unique_cuda_ptr<void> dev_scores_holder;
        unique_cuda_ptr<void> dev_softmax_holder;
        void* dev_scores = nullptr;
        void* dev_softmax_out = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_scores, score_size));
        dev_scores_holder.reset(dev_scores);
        CUDA_CHECK(cudaMalloc(&dev_softmax_out, score_size));
        dev_softmax_holder.reset(dev_softmax_out);

        using T = float;
        const T* dev_q = reinterpret_cast<const T*>(q);
        const T* dev_k = reinterpret_cast<const T*>(k);
        const T* dev_v = reinterpret_cast<const T*>(v);
        T* dev_out = reinterpret_cast<T*>(attn_val);
        T* scores = static_cast<T*>(dev_scores);
        T* softmax_out = static_cast<T*>(dev_softmax_out);

        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H_q; ++h) {
                size_t h_kv = h / n_repeat;
                const T* q_h = dev_q + b * S_q * H_q * D + h * D;
                const T* k_h = dev_k + b * S_kv * H_kv * D + h_kv * D;
                const T* v_h = dev_v + b * S_kv * H_kv * D + h_kv * D;
                T* out_h = dev_out + b * S_q * H_q * D + h * D;

                int lda = static_cast<int>(H_kv * D);
                int ldb = static_cast<int>(H_q * D);
                int ldc = static_cast<int>(S_kv);

                CUBLAS_CHECK(cublasSgemm(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(S_kv), static_cast<int>(S_q), static_cast<int>(D),
                    &scale, k_h, lda, q_h, ldb, &beta, scores, ldc));

                dim3 grid(static_cast<int>(S_q));
                dim3 block(256);
                causal_softmax_kernel<float><<<grid, block>>>(softmax_out, scores, S_q, S_kv, mask_offset);
                CUDA_CHECK(cudaGetLastError());

                CUBLAS_CHECK(cublasSgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(D), static_cast<int>(S_q), static_cast<int>(S_kv),
                    &alpha_one, v_h, lda, softmax_out, ldc, &beta, out_h, ldb));
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // F16: GEMM 用 F16，softmax 用 FP32 保证精度
        using T = __half;
        size_t score_size_half = S_q * S_kv * sizeof(T);
        size_t score_size_fp32 = S_q * S_kv * sizeof(float);

        unique_cuda_ptr<void> dev_scores_half_holder;
        unique_cuda_ptr<void> dev_scores_fp32_holder;
        unique_cuda_ptr<void> dev_softmax_fp32_holder;
        void* dev_scores_half = nullptr;
        void* dev_scores_fp32 = nullptr;
        void* dev_softmax_fp32 = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_scores_half, score_size_half));
        dev_scores_half_holder.reset(dev_scores_half);
        CUDA_CHECK(cudaMalloc(&dev_scores_fp32, score_size_fp32));
        dev_scores_fp32_holder.reset(dev_scores_fp32);
        CUDA_CHECK(cudaMalloc(&dev_softmax_fp32, score_size_fp32));
        dev_softmax_fp32_holder.reset(dev_softmax_fp32);

        const T* dev_q = reinterpret_cast<const T*>(q);
        const T* dev_k = reinterpret_cast<const T*>(k);
        const T* dev_v = reinterpret_cast<const T*>(v);
        T* dev_out = reinterpret_cast<T*>(attn_val);
        T* scores_half = static_cast<T*>(dev_scores_half);
        float* scores_fp32 = static_cast<float*>(dev_scores_fp32);
        float* softmax_fp32 = static_cast<float*>(dev_softmax_fp32);

        int blocks_convert = (static_cast<int>(S_q * S_kv) + 255) / 256;

        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H_q; ++h) {
                size_t h_kv = h / n_repeat;
                const T* q_h = dev_q + b * S_q * H_q * D + h * D;
                const T* k_h = dev_k + b * S_kv * H_kv * D + h_kv * D;
                const T* v_h = dev_v + b * S_kv * H_kv * D + h_kv * D;
                T* out_h = dev_out + b * S_q * H_q * D + h * D;

                int lda = static_cast<int>(H_kv * D);
                int ldb = static_cast<int>(H_q * D);
                int ldc = static_cast<int>(S_kv);

                // 1. Q * K^T → F16 scores
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(S_kv), static_cast<int>(S_q), static_cast<int>(D),
                    &scale, k_h, CUDA_R_16F, lda, q_h, CUDA_R_16F, ldb,
                    &beta, scores_half, CUDA_R_16F, ldc,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

                // 2. F16 scores → FP32
                f16_to_f32_kernel<<<blocks_convert, 256>>>(scores_fp32, scores_half, S_q * S_kv);
                CUDA_CHECK(cudaGetLastError());

                // 3. Softmax on FP32
                dim3 grid(static_cast<int>(S_q));
                dim3 block(256);
                causal_softmax_kernel<float><<<grid, block>>>(softmax_fp32, scores_fp32, S_q, S_kv, mask_offset);
                CUDA_CHECK(cudaGetLastError());

                // 4. FP32 softmax → F16 (reuse scores_half buffer)
                f32_to_f16_kernel<<<blocks_convert, 256>>>(scores_half, softmax_fp32, S_q * S_kv);
                CUDA_CHECK(cudaGetLastError());

                // 5. Softmax * V → F16 output
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(D), static_cast<int>(S_q), static_cast<int>(S_kv),
                    &alpha_one, v_h, CUDA_R_16F, lda, scores_half, CUDA_R_16F, ldc,
                    &beta, out_h, CUDA_R_16F, ldb,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        // BF16: GEMM 用 BF16，softmax 用 FP32 保证精度
        using T = __nv_bfloat16;
        size_t score_size_half = S_q * S_kv * sizeof(T);
        size_t score_size_fp32 = S_q * S_kv * sizeof(float);

        unique_cuda_ptr<void> dev_scores_half_holder;
        unique_cuda_ptr<void> dev_scores_fp32_holder;
        unique_cuda_ptr<void> dev_softmax_fp32_holder;
        void* dev_scores_half = nullptr;
        void* dev_scores_fp32 = nullptr;
        void* dev_softmax_fp32 = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_scores_half, score_size_half));
        dev_scores_half_holder.reset(dev_scores_half);
        CUDA_CHECK(cudaMalloc(&dev_scores_fp32, score_size_fp32));
        dev_scores_fp32_holder.reset(dev_scores_fp32);
        CUDA_CHECK(cudaMalloc(&dev_softmax_fp32, score_size_fp32));
        dev_softmax_fp32_holder.reset(dev_softmax_fp32);

        const T* dev_q = reinterpret_cast<const T*>(q);
        const T* dev_k = reinterpret_cast<const T*>(k);
        const T* dev_v = reinterpret_cast<const T*>(v);
        T* dev_out = reinterpret_cast<T*>(attn_val);
        T* scores_half = static_cast<T*>(dev_scores_half);
        float* scores_fp32 = static_cast<float*>(dev_scores_fp32);
        float* softmax_fp32 = static_cast<float*>(dev_softmax_fp32);

        int blocks_convert = (static_cast<int>(S_q * S_kv) + 255) / 256;

        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H_q; ++h) {
                size_t h_kv = h / n_repeat;
                const T* q_h = dev_q + b * S_q * H_q * D + h * D;
                const T* k_h = dev_k + b * S_kv * H_kv * D + h_kv * D;
                const T* v_h = dev_v + b * S_kv * H_kv * D + h_kv * D;
                T* out_h = dev_out + b * S_q * H_q * D + h * D;

                int lda = static_cast<int>(H_kv * D);
                int ldb = static_cast<int>(H_q * D);
                int ldc = static_cast<int>(S_kv);

                // 1. Q * K^T → BF16 scores
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(S_kv), static_cast<int>(S_q), static_cast<int>(D),
                    &scale, k_h, CUDA_R_16BF, lda, q_h, CUDA_R_16BF, ldb,
                    &beta, scores_half, CUDA_R_16BF, ldc,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

                // 2. BF16 scores → FP32
                bf16_to_f32_kernel<<<blocks_convert, 256>>>(scores_fp32, scores_half, S_q * S_kv);
                CUDA_CHECK(cudaGetLastError());

                // 3. Softmax on FP32
                dim3 grid(static_cast<int>(S_q));
                dim3 block(256);
                causal_softmax_kernel<float><<<grid, block>>>(softmax_fp32, scores_fp32, S_q, S_kv, mask_offset);
                CUDA_CHECK(cudaGetLastError());

                // 4. FP32 softmax → BF16 (reuse scores_half buffer)
                f32_to_bf16_kernel<<<blocks_convert, 256>>>(scores_half, softmax_fp32, S_q * S_kv);
                CUDA_CHECK(cudaGetLastError());

                // 5. Softmax * V → BF16 output
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(D), static_cast<int>(S_q), static_cast<int>(S_kv),
                    &alpha_one, v_h, CUDA_R_16BF, lda, scores_half, CUDA_R_16BF, ldc,
                    &beta, out_h, CUDA_R_16BF, ldb,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::iluvatar