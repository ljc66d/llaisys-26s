#include "../runtime_api.hpp"
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Iluvatar CoreX API (ix*) - CUDA-compatible interface
// 在实际Iluvatar环境中，需要包含 ix_runtime.h 等头文件
// 这里使用条件编译，在无Iluvatar SDK时提供占位实现
#ifdef HAS_ILUVATAR_SDK
#include <ix_runtime.h>
#endif

namespace llaisys::device::iluvatar {

namespace runtime_api {

#ifdef HAS_ILUVATAR_SDK
#define IX_CHECK(expr)                                                                        \
    do {                                                                                      \
        ixError_t _err = (expr);                                                              \
        if (_err != ixSuccess) {                                                              \
            throw std::runtime_error("Iluvatar error: " + std::string(ixGetErrorString(_err))); \
        }                                                                                     \
    } while (0)

static ixMemcpyKind to_ix_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2D: return ixMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return ixMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return ixMemcpyDeviceToDevice;
    case LLAISYS_MEMCPY_H2H: return ixMemcpyHostToHost;
    default: throw std::runtime_error("unsupported memcpy kind");
    }
}
#endif

int getDeviceCount() {
#ifdef HAS_ILUVATAR_SDK
    int count = 0;
    IX_CHECK(ixGetDeviceCount(&count));
    return count;
#else
    return 0; // 无Iluvatar硬件时返回0
#endif
}

void setDevice(int device_id) {
#ifdef HAS_ILUVATAR_SDK
    IX_CHECK(ixSetDevice(device_id));
#else
    (void)device_id;
#endif
}

void deviceSynchronize() {
#ifdef HAS_ILUVATAR_SDK
    IX_CHECK(ixDeviceSynchronize());
#endif
}

llaisysStream_t createStream() {
#ifdef HAS_ILUVATAR_SDK
    ixStream_t stream = nullptr;
    IX_CHECK(ixStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
#else
    return nullptr;
#endif
}

void destroyStream(llaisysStream_t stream) {
#ifdef HAS_ILUVATAR_SDK
    if (stream != nullptr) {
        ixStream_t ix_stream = reinterpret_cast<ixStream_t>(stream);
        IX_CHECK(ixStreamDestroy(ix_stream));
    }
#else
    (void)stream;
#endif
}

void streamSynchronize(llaisysStream_t stream) {
#ifdef HAS_ILUVATAR_SDK
    ixStream_t ix_stream = reinterpret_cast<ixStream_t>(stream);
    IX_CHECK(ixStreamSynchronize(ix_stream));
#else
    (void)stream;
#endif
}

void *mallocDevice(size_t size) {
#ifdef HAS_ILUVATAR_SDK
    void *ptr = nullptr;
    IX_CHECK(ixMalloc(&ptr, size));
    return ptr;
#else
    (void)size;
    return nullptr;
#endif
}

void freeDevice(void *ptr) {
#ifdef HAS_ILUVATAR_SDK
    if (ptr != nullptr) {
        IX_CHECK(ixFree(ptr));
    }
#else
    (void)ptr;
#endif
}

void *mallocHost(size_t size) {
#ifdef HAS_ILUVATAR_SDK
    void *ptr = nullptr;
    IX_CHECK(ixMallocHost(&ptr, size));
    return ptr;
#else
    (void)size;
    return nullptr;
#endif
}

void freeHost(void *ptr) {
#ifdef HAS_ILUVATAR_SDK
    if (ptr != nullptr) {
        IX_CHECK(ixFreeHost(ptr));
    }
#else
    (void)ptr;
#endif
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
#ifdef HAS_ILUVATAR_SDK
    IX_CHECK(ixMemcpy(dst, src, size, to_ix_memcpy_kind(kind)));
#else
    (void)dst; (void)src; (void)size; (void)kind;
#endif
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
#ifdef HAS_ILUVATAR_SDK
    ixStream_t ix_stream = reinterpret_cast<ixStream_t>(stream);
    IX_CHECK(ixMemcpyAsync(dst, src, size, to_ix_memcpy_kind(kind), ix_stream));
#else
    (void)dst; (void)src; (void)size; (void)kind; (void)stream;
#endif
}

// 字段顺序必须与 runtime_api.hpp 中 LlaisysRuntimeAPI 结构体定义完全一致
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

} // namespace llaisys::device::iluvatar