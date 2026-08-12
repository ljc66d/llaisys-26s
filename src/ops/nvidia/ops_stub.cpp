#include "llaisys/ops/add/op.hpp"
#include "llaisys/ops/argmax/op.hpp"
#include "llaisys/ops/embedding/op.hpp"
#include "llaisys/ops/linear/op.hpp"
#include "llaisys/ops/rearrange/op.hpp"
#include "llaisys/ops/rms_norm/op.hpp"
#include "llaisys/ops/rope/op.hpp"
#include "llaisys/ops/self_attention/op.hpp"
#include "llaisys/ops/swiglu/op.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {

void add(std::byte* dst, const std::byte* a, const std::byte* b, llaisysDataType_t dtype, size_t numel) {
    throw std::runtime_error("NVIDIA add not implemented");
}

void argmax(std::byte* dst, std::byte* indices, const std::byte* src, llaisysDataType_t dtype, size_t numel) {
    throw std::runtime_error("NVIDIA argmax not implemented");
}

void embedding(std::byte* dst, const std::byte* weight, const std::byte* indices, llaisysDataType_t dtype, size_t numel, size_t vocab_size, size_t hidden_dim) {
    throw std::runtime_error("NVIDIA embedding not implemented");
}

void linear(std::byte* dst, const std::byte* input, const std::byte* weight, const std::byte* bias, llaisysDataType_t dtype, size_t M, size_t K, size_t N) {
    throw std::runtime_error("NVIDIA linear not implemented");
}

void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size) {
    throw std::runtime_error("NVIDIA rearrange not implemented");
}

void rms_norm(std::byte* dst, const std::byte* input, const std::byte* weight, llaisysDataType_t dtype, size_t numel, size_t hidden_dim, float eps) {
    throw std::runtime_error("NVIDIA rms_norm not implemented");
}

void rope(std::byte* dst, const std::byte* src, const std::byte* pos_ids, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim, float theta, size_t base_dim) {
    throw std::runtime_error("NVIDIA rope not implemented");
}

void self_attention(std::byte* dst, const std::byte* q, const std::byte* k, const std::byte* v, llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, size_t kv_len, float scale) {
    throw std::runtime_error("NVIDIA self_attention not implemented");
}

void swiglu(std::byte* dst, const std::byte* gate, const std::byte* up, llaisysDataType_t dtype, size_t numel) {
    throw std::runtime_error("NVIDIA swiglu not implemented");
}

} // namespace llaisys::ops::nvidia