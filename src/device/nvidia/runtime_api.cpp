#include "../runtime_api.hpp"
#include "cuda_loader.h"

namespace llaisys::device::nvidia {

namespace runtime_api {

    int getDeviceCount() {
        return get_cuda_api()->runtime.get_device_count();
    }

    void setDevice(int device_id) {
        get_cuda_api()->runtime.set_device(device_id);
    }

    void deviceSynchronize() {
        get_cuda_api()->runtime.device_synchronize();
    }

    llaisysStream_t createStream() {
        return get_cuda_api()->runtime.create_stream();
    }

    void destroyStream(llaisysStream_t stream) {
        get_cuda_api()->runtime.destroy_stream(stream);
    }

    void streamSynchronize(llaisysStream_t stream) {
        get_cuda_api()->runtime.stream_synchronize(stream);
    }

    void* mallocDevice(size_t size) {
        return get_cuda_api()->runtime.malloc_device(size);
    }

    void freeDevice(void* ptr) {
        get_cuda_api()->runtime.free_device(ptr);
    }

    void* mallocHost(size_t size) {
        return get_cuda_api()->runtime.malloc_host(size);
    }

    void freeHost(void* ptr) {
        get_cuda_api()->runtime.free_host(ptr);
    }

    void memcpySync(void* dst, const void* src, size_t size, llaisysMemcpyKind_t kind) {
        get_cuda_api()->runtime.memcpy_sync(dst, src, size, kind);
    }

    void memcpyAsync(void* dst, const void* src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
        get_cuda_api()->runtime.memcpy_async(dst, src, size, kind, stream);
    }

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

const LlaisysRuntimeAPI* getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia