#include "ops/add/op.hpp"
#include "ops/argmax/op.hpp"
#include "ops/embedding/op.hpp"
#include "ops/linear/op.hpp"
#include "ops/rearrange/op.hpp"
#include "ops/rms_norm/op.hpp"
#include "ops/rope/op.hpp"
#include "ops/self_attention/op.hpp"
#include "ops/swiglu/op.hpp"
#include "device/iluvatar/iluvatar_loader.h"

namespace llaisys::ops::iluvatar {

void add(std::byte* c, const std::byte* a, const std::byte* b, llaisysDataType_t type, size_t numel) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.add(c, a, b, static_cast<int>(type), numel);
}

void argmax(std::byte* out_idx, std::byte* out_val, const std::byte* vals, llaisysDataType_t type, size_t size) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.argmax(out_idx, out_val, vals, static_cast<int>(type), size);
}

void embedding(std::byte* output, const std::byte* token_ids, const std::byte* weight, llaisysDataType_t type, size_t num_tokens, size_t hidden_dim) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.embedding(output, token_ids, weight, static_cast<int>(type), num_tokens, hidden_dim);
}

void linear(std::byte* out, const std::byte* x, const std::byte* weight, const std::byte* bias, llaisysDataType_t type, size_t batch, size_t in_dim, size_t out_dim) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.linear(out, x, weight, bias, static_cast<int>(type), batch, in_dim, out_dim);
}

void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.rearrange(dst, src, static_cast<int>(dtype), total_size);
}

void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, llaisysDataType_t type, size_t num_tokens, size_t hidden_dim, float eps) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.rms_norm(output, input, weight, static_cast<int>(type), num_tokens, hidden_dim, eps);
}

void rope(std::byte* dst, const std::byte* src, const std::byte* pos_ids, llaisysDataType_t type, size_t batch, size_t seq_len, size_t head_dim, float theta, size_t base_dim) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.rope(dst, src, pos_ids, static_cast<int>(type), batch, seq_len, head_dim, theta, base_dim);
}

void self_attention(std::byte* attn_val, const std::byte* q, const std::byte* k, const std::byte* v, llaisysDataType_t type, size_t B, size_t S_q, size_t S_kv, size_t H_q, size_t H_kv, size_t D, float scale) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.self_attention(attn_val, q, k, v, static_cast<int>(type), B, S_q, S_kv, H_q, H_kv, D, scale);
}

void swiglu(std::byte* out, const std::byte* gate, const std::byte* up, llaisysDataType_t type, size_t N) {
    llaisys::device::iluvatar::get_iluvatar_api()->ops.swiglu(out, gate, up, static_cast<int>(type), N);
}

} // namespace llaisys::ops::iluvatar