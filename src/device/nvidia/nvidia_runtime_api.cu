#include "../runtime_api.hpp"
#include <string>   
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
// #include <cublas_v2.h>
#include <stdexcept>
// #include <unordered_map>

namespace llaisys::device::nvidia {

// cuBLAS 句柄按设备ID独立管理，解决多设备切换问题
// static std::unordered_map<int, cublasHandle_t> g_cublas_handles;

namespace runtime_api {

#define CUDA_CHECK(expr)                                                                      \
    do {                                                                                      \
        cudaError_t _err = (expr);                                                            \
        if (_err != cudaSuccess) {                                                            \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(_err))); \
        }                                                                                     \
    } while (0)

// #define CUBLAS_CHECK(expr)                                                                    \
//     do {                                                                                      \
//         cublasStatus_t _s = (expr);                                                           \
//         if (_s != CUBLAS_STATUS_SUCCESS) {                                                    \
//             throw std::runtime_error("cuBLAS error: " + std::to_string(_s));                  \
//         }                                                                                     \
//     } while (0)

    static cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
        switch (kind) {
        case LLAISYS_MEMCPY_H2D:
            return cudaMemcpyHostToDevice;
        case LLAISYS_MEMCPY_D2H:
            return cudaMemcpyDeviceToHost;
        case LLAISYS_MEMCPY_D2D:
            return cudaMemcpyDeviceToDevice;
        case LLAISYS_MEMCPY_H2H:
            return cudaMemcpyHostToHost;
        default:
            throw std::runtime_error("unsupported memcpy kind");
        }
    }

    int getDeviceCount() {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        return count;
    }

    void setDevice(int device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
        // 每个设备独立初始化 cuBLAS 句柄
        //if (g_cublas_handles.find(device_id) == g_cublas_handles.end()) {
            //cublasHandle_t handle = nullptr;
            //CUBLAS_CHECK(cublasCreate(&handle));
            //g_cublas_handles[device_id] = handle;
        //}
        
    }

    void deviceSynchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    llaisysStream_t createStream() {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        return reinterpret_cast<llaisysStream_t>(stream);
    }

    void destroyStream(llaisysStream_t stream) {
        if (stream == nullptr) {
            return;
        }
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        CUDA_CHECK(cudaStreamDestroy(cuda_stream));
    }

    void streamSynchronize(llaisysStream_t stream) {
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    }

    void *mallocDevice(size_t size) {
        void *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void freeDevice(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        CUDA_CHECK(cudaFree(ptr));
    }

    void *mallocHost(size_t size) {
        void *ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, size));
        return ptr;
    }

    void freeHost(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        CUDA_CHECK(cudaFreeHost(ptr));
    }

    void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)));
    }

    void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, to_cuda_memcpy_kind(kind), cuda_stream));
    }

    // 【重要】字段顺序必须与 runtime_api.hpp 中结构体定义完全一致
    static const LlaisysRuntimeAPI RUNTIME_API = {
        &getDeviceCount,
        &setDevice,
        &deviceSynchronize,
        &createStream,
        &destroyStream,
        &streamSynchronize,
        &mallocDevice,
        &freeDevice,
        &mallocHost,
        &freeHost,
        &memcpySync,
        &memcpyAsync
    };

} // namespace runtime_api
const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

// 获取当前设备的 cuBLAS 句柄，供算子层调用
// cublasHandle_t getCublasHandle() {
//     int device_id = 0;
//     cudaGetDevice(&device_id);
//     auto it = g_cublas_handles.find(device_id);
//     if (it == g_cublas_handles.end()) {
//         throw std::runtime_error("cuBLAS not initialized for current device");
//     }
//     return it->second;
// }

} // namespace llaisys::device::nvidia