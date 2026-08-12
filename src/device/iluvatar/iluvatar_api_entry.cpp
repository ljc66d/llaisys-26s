#include "llaisys/cuda/cuda_api.h"
#include "llaisys.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <stdexcept>

// Iluvatar CoreX SDK provides CUDA-compatible API
// Use standard CUDA API functions

#define ILUVATAR_CHECK(expr)                                                              \
    do {                                                                                   \
        cudaError_t _err = (expr);                                                         \
        if (_err != cudaSuccess) {                                                         \
            throw std::runtime_error("Iluvatar error: " + std::string(cudaGetErrorString(_err))); \
        }                                                                                  \
    } while (0)

// ========== 算子函数原型声明（与各 .cu/.hpp 内核签名严格一致） ==========
namespace llaisys::ops::iluvatar {
    void add(std::byte* c, const std::byte* a, const std::byte* b, llaisysDataType_t type, size_t numel);
    void argmax(std::byte* out_idx, std::byte* out_val, const std::byte* vals, llaisysDataType_t type, size_t size);
    void embedding(std::byte* output, const std::byte* token_ids, const std::byte* weight, llaisysDataType_t type, size_t num_tokens, size_t hidden_dim);
    void linear(std::byte* out, const std::byte* x, const std::byte* weight, const std::byte* bias, llaisysDataType_t type, size_t batch, size_t in_dim, size_t out_dim);
    void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size);
    void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, llaisysDataType_t type, size_t num_tokens, size_t hidden_dim, float eps);
    void rope(std::byte* dst, const std::byte* src, const std::byte* pos_ids, llaisysDataType_t type, size_t batch, size_t seq_len, size_t head_dim, float base, size_t base_dim);
    void self_attention(std::byte* attn_val, const std::byte* q, const std::byte* k, const std::byte* v, llaisysDataType_t type, size_t B, size_t S_q, size_t S_kv, size_t H_q, size_t H_kv, size_t D, float scale);
    void swiglu(std::byte* out, const std::byte* gate, const std::byte* up, llaisysDataType_t type, size_t N);
}

// ========== Runtime 包装层（使用标准 CUDA API） ==========
static int wrap_get_device_count() {
    int count = 0;
    ILUVATAR_CHECK(cudaGetDeviceCount(&count));
    return count;
}

static void wrap_set_device(int device_id) {
    ILUVATAR_CHECK(cudaSetDevice(device_id));
    cudaFree(nullptr);
}

static void wrap_device_synchronize() {
    ILUVATAR_CHECK(cudaDeviceSynchronize());
}

static void* wrap_create_stream() {
    cudaStream_t stream = nullptr;
    ILUVATAR_CHECK(cudaStreamCreate(&stream));
    return stream;
}

static void wrap_destroy_stream(void* stream) {
    if (stream) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

static void wrap_stream_synchronize(void* stream) {
    ILUVATAR_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

static void* wrap_malloc_device(size_t size) {
    void* ptr = nullptr;
    ILUVATAR_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

static void wrap_free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

static void* wrap_malloc_host(size_t size) {
    void* ptr = nullptr;
    ILUVATAR_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

static void wrap_free_host(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

static void wrap_memcpy_sync(void* dst, const void* src, size_t size, int kind) {
    ILUVATAR_CHECK(cudaMemcpy(dst, src, size, static_cast<cudaMemcpyKind>(kind)));
}

static void wrap_memcpy_async(void* dst, const void* src, size_t size, int kind, void* stream) {
    ILUVATAR_CHECK(cudaMemcpyAsync(dst, src, size, static_cast<cudaMemcpyKind>(kind), reinterpret_cast<cudaStream_t>(stream)));
}

// ========== 算子包装层 ==========
static void wrap_add(void* dst, const void* a, const void* b, int dtype, size_t numel) {
    llaisys::ops::iluvatar::add(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(a),
        reinterpret_cast<const std::byte*>(b),
        static_cast<llaisysDataType_t>(dtype), numel);
}

static void wrap_argmax(void* dst, void* indices, const void* src, int dtype, size_t numel) {
    llaisys::ops::iluvatar::argmax(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<std::byte*>(indices),
        reinterpret_cast<const std::byte*>(src),
        static_cast<llaisysDataType_t>(dtype), numel);
}

static void wrap_embedding(void* dst, const void* weight, const void* indices, int dtype, size_t numel, size_t vocab_size) {
    llaisys::ops::iluvatar::embedding(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(weight),
        reinterpret_cast<const std::byte*>(indices),
        static_cast<llaisysDataType_t>(dtype), numel, vocab_size);
}

static void wrap_linear(void* dst, const void* input, const void* weight, const void* bias, int dtype, size_t M, size_t K, size_t N) {
    llaisys::ops::iluvatar::linear(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(input),
        reinterpret_cast<const std::byte*>(weight),
        reinterpret_cast<const std::byte*>(bias),
        static_cast<llaisysDataType_t>(dtype), M, K, N);
}

static void wrap_rearrange(void* dst, const void* src, int dtype, size_t total_size) {
    llaisys::ops::iluvatar::rearrange(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(src),
        static_cast<llaisysDataType_t>(dtype), total_size);
}

static void wrap_rms_norm(void* output, const void* input, const void* weight, int dtype, size_t num_tokens, size_t hidden_dim, float eps) {
    llaisys::ops::iluvatar::rms_norm(
        reinterpret_cast<std::byte*>(output),
        reinterpret_cast<const std::byte*>(input),
        reinterpret_cast<const std::byte*>(weight),
        static_cast<llaisysDataType_t>(dtype), num_tokens, hidden_dim, eps);
}

static void wrap_rope(void* dst, const void* src, const void* pos_ids, int dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta, size_t base_dim) {
    llaisys::ops::iluvatar::rope(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(src),
        reinterpret_cast<const std::byte*>(pos_ids),
        static_cast<llaisysDataType_t>(dtype), seq_len, num_heads, head_dim, theta, base_dim);
}

static void wrap_self_attention(void* dst, const void* q, const void* k, const void* v, int dtype, size_t B, size_t S_q, size_t S_kv, size_t H_q, size_t H_kv, size_t D, float scale) {
    llaisys::ops::iluvatar::self_attention(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(q),
        reinterpret_cast<const std::byte*>(k),
        reinterpret_cast<const std::byte*>(v),
        static_cast<llaisysDataType_t>(dtype), B, S_q, S_kv, H_q, H_kv, D, scale);
}

static void wrap_swiglu(void* dst, const void* gate, const void* up, int dtype, size_t numel) {
    llaisys::ops::iluvatar::swiglu(
        reinterpret_cast<std::byte*>(dst),
        reinterpret_cast<const std::byte*>(gate),
        reinterpret_cast<const std::byte*>(up),
        static_cast<llaisysDataType_t>(dtype), numel);
}

// ========== 构建API表 ==========
static llaisys_cuda_api_table_t build_api_table() {
    llaisys_cuda_api_table_t table;

    table.runtime.get_device_count = wrap_get_device_count;
    table.runtime.set_device = wrap_set_device;
    table.runtime.device_synchronize = wrap_device_synchronize;
    table.runtime.create_stream = wrap_create_stream;
    table.runtime.destroy_stream = wrap_destroy_stream;
    table.runtime.stream_synchronize = wrap_stream_synchronize;
    table.runtime.malloc_device = wrap_malloc_device;
    table.runtime.free_device = wrap_free_device;
    table.runtime.malloc_host = wrap_malloc_host;
    table.runtime.free_host = wrap_free_host;
    table.runtime.memcpy_sync = wrap_memcpy_sync;
    table.runtime.memcpy_async = wrap_memcpy_async;

    table.ops.add = wrap_add;
    table.ops.argmax = wrap_argmax;
    table.ops.embedding = wrap_embedding;
    table.ops.linear = wrap_linear;
    table.ops.rearrange = wrap_rearrange;
    table.ops.rms_norm = wrap_rms_norm;
    table.ops.rope = wrap_rope;
    table.ops.self_attention = wrap_self_attention;
    table.ops.swiglu = wrap_swiglu;

    return table;
}

static llaisys_cuda_api_table_t g_api_table = build_api_table();

// ========== 唯一导出函数 ==========
#ifdef _WIN32
extern "C" __declspec(dllexport) llaisys_cuda_api_table_t* llaisys_iluvatar_get_api_table() {
#else
extern "C" __attribute__((visibility("default"))) llaisys_cuda_api_table_t* llaisys_iluvatar_get_api_table() {
#endif
    return &g_api_table;
}