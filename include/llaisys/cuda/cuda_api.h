#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ========== 设备 Runtime 函数表 ==========
typedef struct {
    int (*get_device_count)();
    void (*set_device)(int device_id);
    void (*device_synchronize)();
    void* (*create_stream)();
    void (*destroy_stream)(void* stream);
    void (*stream_synchronize)(void* stream);
    void* (*malloc_device)(size_t size);
    void (*free_device)(void* ptr);
    void* (*malloc_host)(size_t size);
    void (*free_host)(void* ptr);
    void (*memcpy_sync)(void* dst, const void* src, size_t size, int kind);
    void (*memcpy_async)(void* dst, const void* src, size_t size, int kind, void* stream);
} cuda_runtime_api_t;

// ========== 算子函数表 ==========
typedef struct {
    void (*add)(void* dst, const void* a, const void* b, int dtype, size_t numel);
    void (*argmax)(void* dst, void* indices, const void* src, int dtype, size_t numel);
    void (*embedding)(void* dst, const void* weight, const void* indices, int dtype, size_t numel, size_t vocab_size);
    void (*linear)(void* dst, const void* input, const void* weight, const void* bias, int dtype, size_t M, size_t K, size_t N);
    void (*rearrange)(void* dst, const void* src, int dtype, size_t total_size);
    void (*rms_norm)(void* dst, const void* input, const void* weight, int dtype, size_t numel, size_t hidden_dim, float eps);
    void (*rope)(void* dst, const void* src, const void* pos_ids, int dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta, size_t base_dim);
    void (*self_attention)(void* dst, const void* q, const void* k, const void* v, int dtype, size_t B, size_t S_q, size_t S_kv, size_t H_q, size_t H_kv, size_t D, float scale);
    void (*swiglu)(void* dst, const void* gate, const void* up, int dtype, size_t numel);
} cuda_ops_api_t;

// ========== 总 API 表 ==========
typedef struct {
    cuda_runtime_api_t runtime;
    cuda_ops_api_t ops;
} llaisys_cuda_api_table_t;

#ifdef __cplusplus
}
#endif