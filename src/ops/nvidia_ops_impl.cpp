#include "ops/add/op.hpp"
#include "ops/argmax/op.hpp"
#include "ops/embedding/op.hpp"
#include "ops/linear/op.hpp"
#include "ops/rearrange/op.hpp"
#include "ops/rms_norm/op.hpp"
#include "ops/rope/op.hpp"
#include "ops/self_attention/op.hpp"
#include "ops/swiglu/op.hpp"
#include "device/nvidia/cuda_loader.h"

namespace llaisys::ops::nvidia {

void add(std::byte* dst, const std::byte* a, const std::byte* b, llaisysDataType_t dtype, size_t numel) {
    llaisys::device::nvidia::get_cuda_api()->ops.add(dst, a, b, static_cast<int>(dtype), numel);
}

void argmax(std::byte* dst, std::byte* indices, const std::byte* src, llaisysDataType_t dtype, size_t numel) {
    llaisys::device::nvidia::get_cuda_api()->ops.argmax(dst, indices, src, static_cast<int>(dtype), numel);
}

void embedding(std::byte* dst, const std::byte* weight, const std::byte* indices, llaisysDataType_t dtype, size_t numel, size_t vocab_size) {
    llaisys::device::nvidia::get_cuda_api()->ops.embedding(dst, weight, indices, static_cast<int>(dtype), numel, vocab_size);
}

void linear(std::byte* dst, const std::byte* input, const std::byte* weight, const std::byte* bias, llaisysDataType_t dtype, size_t M, size_t K, size_t N) {
    llaisys::device::nvidia::get_cuda_api()->ops.linear(dst, input, weight, bias, static_cast<int>(dtype), M, K, N);
}

void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size) {
    llaisys::device::nvidia::get_cuda_api()->ops.rearrange(dst, src, static_cast<int>(dtype), total_size);
}

void rms_norm(std::byte* dst, const std::byte* input, const std::byte* weight, llaisysDataType_t dtype, size_t numel, size_t hidden_dim, float eps) {
    llaisys::device::nvidia::get_cuda_api()->ops.rms_norm(dst, input, weight, static_cast<int>(dtype), numel, hidden_dim, eps);
}

void rope(std::byte* dst, const std::byte* src, const std::byte* pos_ids, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta, size_t base_dim) {
    llaisys::device::nvidia::get_cuda_api()->ops.rope(dst, src, pos_ids, static_cast<int>(dtype), seq_len, num_heads, head_dim, theta, base_dim);
}

void self_attention(std::byte* dst, const std::byte* q, const std::byte* k, const std::byte* v, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, size_t kv_len, size_t batch, float scale) {
    llaisys::device::nvidia::get_cuda_api()->ops.self_attention(dst, q, k, v, static_cast<int>(dtype), seq_len, num_heads, num_kv_heads, head_dim, kv_len, batch, scale);
}

void swiglu(std::byte* dst, const std::byte* gate, const std::byte* up, llaisysDataType_t dtype, size_t numel) {
    llaisys::device::nvidia::get_cuda_api()->ops.swiglu(dst, gate, up, static_cast<int>(dtype), numel);
}

} // namespace llaisys::ops::nvidia