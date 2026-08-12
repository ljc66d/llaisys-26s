#include "qwen2_model.hpp"
#include "core/runtime/runtime.hpp"  
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils/types.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "core/context/context.hpp"
namespace llaisys {

Qwen2Model::Qwen2Model(const LlaisysQwen2Config &config,
                       llaisysDeviceType_t device_type,
                       int *device_ids,
                       int ndevice)
    : config_(config),
      device_type_(device_type),
      past_len_(0) {
    if (device_ids != nullptr && ndevice > 0) {
        device_ids_.assign(device_ids, device_ids + ndevice);
    }

    int L = config_.num_layers;
    attn_norm_weights_.resize(L);
    attn_q_weights_.resize(L);
    attn_k_weights_.resize(L);
    attn_v_weights_.resize(L);
    attn_o_weights_.resize(L);
    mlp_norm_weights_.resize(L);
    mlp_gate_weights_.resize(L);
    mlp_up_weights_.resize(L);
    mlp_down_weights_.resize(L);
    attn_q_bias_.resize(L);
    attn_k_bias_.resize(L);
    attn_v_bias_.resize(L);

    allocateKVCache();
}

void Qwen2Model::allocateKVCache() {
    int L = config_.num_layers;
    int max_seq = config_.max_seq_len;
    int n_kv_heads = config_.num_kv_heads;
    int head_dim = config_.head_dim;

    k_cache_.resize(L);
    v_cache_.resize(L);

    std::vector<size_t> shape = {
        (size_t)max_seq,
        (size_t)n_kv_heads,
        (size_t)head_dim};

    for (int i = 0; i < L; ++i) {
        k_cache_[i] = Tensor::create(shape, config_.dtype,device_type_);
        v_cache_[i] = Tensor::create(shape, config_.dtype,device_type_);
    }
}

void Qwen2Model::resetKVCache() {
    past_len_ = 0;
}

void Qwen2Model::loadWeight(const std::string &name,
                            const void *data,
                            const std::vector<int64_t> &shape_vec) {
    auto createAndLoad = [&](tensor_t &t) {
        if (!t) {
            std::vector<size_t> sz;
            for (auto d : shape_vec) sz.push_back(d);
            t = Tensor::create(sz, config_.dtype, device_type_);
        }
        size_t total_bytes = t->numel() * t->elementSize();
    #ifdef ENABLE_NVIDIA_API
        const auto* rt = llaisys::device::nvidia::getRuntimeAPI();
        rt->memcpy_sync(t->data(), data, total_bytes, LLAISYS_MEMCPY_H2D);
    #elif defined(ENABLE_ILUVATAR_API)
        const auto* irt = llaisys::device::iluvatar::getRuntimeAPI();
        irt->memcpy_sync(t->data(), data, total_bytes, LLAISYS_MEMCPY_H2D);
    #else
        (void)total_bytes;
        t->load(data);
    #endif
    };

    if (name == "model.embed_tokens.weight") {
        createAndLoad(embed_tokens_);
        return;
    }
    if (name == "model.norm.weight") {
        createAndLoad(final_norm_weight_);
        return;
    }
    if (name == "lm_head.weight") {
        createAndLoad(lm_head_weight_);
        return;
    }

    const std::string prefix = "model.layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        fprintf(stderr, "Unknown weight name: %s\n", name.c_str());
        return;
    }

    size_t pos = prefix.size();
    size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        fprintf(stderr, "Invalid layer name: %s\n", name.c_str());
        return;
    }
    int layer_idx = std::stoi(name.substr(pos, dot - pos));
    std::string suffix = name.substr(dot + 1);

    if (suffix == "input_layernorm.weight") {
        createAndLoad(attn_norm_weights_[layer_idx]);
    } else if (suffix == "self_attn.q_proj.weight") {
        createAndLoad(attn_q_weights_[layer_idx]);
    } else if (suffix == "self_attn.q_proj.bias") {
        createAndLoad(attn_q_bias_[layer_idx]);
    } else if (suffix == "self_attn.k_proj.weight") {
        createAndLoad(attn_k_weights_[layer_idx]);
    } else if (suffix == "self_attn.k_proj.bias") {
        createAndLoad(attn_k_bias_[layer_idx]);
    } else if (suffix == "self_attn.v_proj.weight") {
        createAndLoad(attn_v_weights_[layer_idx]);
    } else if (suffix == "self_attn.v_proj.bias") {
        createAndLoad(attn_v_bias_[layer_idx]);
    } else if (suffix == "self_attn.o_proj.weight") {
        createAndLoad(attn_o_weights_[layer_idx]);
    } else if (suffix == "post_attention_layernorm.weight") {
        createAndLoad(mlp_norm_weights_[layer_idx]);
    } else if (suffix == "mlp.gate_proj.weight") {
        createAndLoad(mlp_gate_weights_[layer_idx]);
    } else if (suffix == "mlp.up_proj.weight") {
        createAndLoad(mlp_up_weights_[layer_idx]);
    } else if (suffix == "mlp.down_proj.weight") {
        createAndLoad(mlp_down_weights_[layer_idx]);
    } else {
        fprintf(stderr, "Unknown weight suffix: %s\n", suffix.c_str());
    }
}

tensor_t Qwen2Model::forwardLayer(int layer_idx, const tensor_t &hidden_states) {
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    llaisys::core::context().setDevice(device_type_, device_id);

    auto hs_shape = hidden_states->shape();
    int cur_len = (int)hs_shape[0];
    int hidden_dim = (int)config_.hidden_dim;

    // 1. Pre-Attention RMSNorm
    auto norm_out = Tensor::create(hs_shape, config_.dtype, device_type_);
    ops::rms_norm(norm_out, hidden_states, attn_norm_weights_[layer_idx], config_.epsilon);

    // 2. QKV 投影
    size_t kv_dim = config_.num_kv_heads * config_.head_dim;
    auto q_out = Tensor::create({(size_t)cur_len, (size_t)config_.num_heads * config_.head_dim}, config_.dtype, device_type_);
    auto k_out = Tensor::create({(size_t)cur_len, kv_dim}, config_.dtype, device_type_);
    auto v_out = Tensor::create({(size_t)cur_len, kv_dim}, config_.dtype, device_type_);

    ops::linear(q_out, norm_out, attn_q_weights_[layer_idx], attn_q_bias_[layer_idx]);
    ops::linear(k_out, norm_out, attn_k_weights_[layer_idx], attn_k_bias_[layer_idx]);
    ops::linear(v_out, norm_out, attn_v_weights_[layer_idx], attn_v_bias_[layer_idx]);

    // 3. RoPE：先在三维上应用 RoPE（per-head），再用于 Attention
    auto q_3d = q_out->view({(size_t)cur_len, (size_t)config_.num_heads, (size_t)config_.head_dim});
    auto k_3d = k_out->view({(size_t)cur_len, (size_t)config_.num_kv_heads, (size_t)config_.head_dim});

    auto q = Tensor::create({(size_t)cur_len, (size_t)config_.num_heads, (size_t)config_.head_dim}, config_.dtype, device_type_);
    auto k = Tensor::create({(size_t)cur_len, (size_t)config_.num_kv_heads, (size_t)config_.head_dim}, config_.dtype, device_type_);

    // 创建 pos_ids
    auto pos_ids_cpu = Tensor::create({(size_t)cur_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    {
        int64_t *pos_data = (int64_t *)pos_ids_cpu->data();
        for (int i = 0; i < cur_len; ++i) {
            pos_data[i] = (int64_t)(past_len_ + i);
        }
    }
    auto pos_ids = Tensor::create({(size_t)cur_len}, LLAISYS_DTYPE_I64, device_type_);
    pos_ids->load(pos_ids_cpu->data());

    ops::rope(q, q_3d, pos_ids, config_.theta);
    ops::rope(k, k_3d, pos_ids, config_.theta);

    // V 不需要 RoPE，直接 view 成三维
    auto v = v_out->view({(size_t)cur_len, (size_t)config_.num_kv_heads, (size_t)config_.head_dim});

    // KV 缓存写入
    auto k_cache_slice = k_cache_[layer_idx]->slice(0, past_len_, past_len_ + cur_len);
    auto v_cache_slice = v_cache_[layer_idx]->slice(0, past_len_, past_len_ + cur_len);
    size_t kv_bytes = k->numel() * k->elementSize();

    if ((size_t)(past_len_ + cur_len) > k_cache_[layer_idx]->shape()[0]) {
        fprintf(stderr, "[ERROR] KV cache overflow! layer=%d, cur_total=%zu, max=%zu\n",
               layer_idx, (size_t)(past_len_ + cur_len), k_cache_[layer_idx]->shape()[0]);
        fflush(stderr);
    }
#ifdef ENABLE_NVIDIA_API
	    const auto* runtime_api = llaisys::device::nvidia::getRuntimeAPI();
	    runtime_api->memcpy_sync(k_cache_slice->data(), k->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
	    runtime_api->memcpy_sync(v_cache_slice->data(), v->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
	#elif defined(ENABLE_ILUVATAR_API)
	    const auto* iluvatar_rt = llaisys::device::iluvatar::getRuntimeAPI();
	    iluvatar_rt->memcpy_sync(k_cache_slice->data(), k->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
	    iluvatar_rt->memcpy_sync(v_cache_slice->data(), v->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
	#else
	    std::memcpy(k_cache_slice->data(), k->data(), kv_bytes);
	    std::memcpy(v_cache_slice->data(), v->data(), kv_bytes);
	#endif

    // Self-Attention
    auto full_k = k_cache_[layer_idx]->slice(0, 0, past_len_ + cur_len);
    auto full_v = v_cache_[layer_idx]->slice(0, 0, past_len_ + cur_len);
    auto attn_out = Tensor::create(q->shape(), config_.dtype, device_type_);

    float scale = 1.0f / std::sqrt((float)config_.head_dim);
    ops::self_attention(attn_out, q, full_k, full_v, scale);

    // 7. 输出投影
    auto attn_reshaped = attn_out->view({(size_t)cur_len, (size_t)hidden_dim});
    auto proj_out = Tensor::create({(size_t)cur_len, (size_t)hidden_dim}, config_.dtype, device_type_);
    auto o_bias = Tensor::create({(size_t)hidden_dim}, config_.dtype, device_type_);
    auto zero_bias_cpu = Tensor::create({(size_t)hidden_dim}, config_.dtype, LLAISYS_DEVICE_CPU);
    memset(zero_bias_cpu->data(), 0, o_bias->numel() * o_bias->elementSize());
    o_bias->load(zero_bias_cpu->data());
    ops::linear(proj_out, attn_reshaped, attn_o_weights_[layer_idx], o_bias);

    // 8. 第一次残差
    auto h = Tensor::create(hs_shape, config_.dtype, device_type_);
    ops::add(h, proj_out, hidden_states);

    // 9. Pre-FFN RMSNorm
    auto ffn_norm = Tensor::create(hs_shape, config_.dtype, device_type_);
    ops::rms_norm(ffn_norm, h, mlp_norm_weights_[layer_idx], config_.epsilon);

    // 10. FFN Gate + Up
    std::vector<size_t> ffn_shape = {(size_t)cur_len, (size_t)config_.intermediate_size};
    auto gate_out = Tensor::create(ffn_shape, config_.dtype, device_type_);
    auto gate_bias = Tensor::create({(size_t)config_.intermediate_size}, config_.dtype, device_type_);
    auto zero_gate_cpu = Tensor::create({(size_t)config_.intermediate_size}, config_.dtype, LLAISYS_DEVICE_CPU);
    memset(zero_gate_cpu->data(), 0, gate_bias->numel() * gate_bias->elementSize());
    gate_bias->load(zero_gate_cpu->data());
    ops::linear(gate_out, ffn_norm, mlp_gate_weights_[layer_idx], gate_bias);

    auto up_out = Tensor::create(ffn_shape, config_.dtype, device_type_);
    auto up_bias = Tensor::create({(size_t)config_.intermediate_size}, config_.dtype, device_type_);
    auto zero_up_cpu = Tensor::create({(size_t)config_.intermediate_size}, config_.dtype, LLAISYS_DEVICE_CPU);
    memset(zero_up_cpu->data(), 0, up_bias->numel() * up_bias->elementSize());
    up_bias->load(zero_up_cpu->data());
    ops::linear(up_out, ffn_norm, mlp_up_weights_[layer_idx], up_bias);

    // 11. SwiGLU
    auto swi_out = Tensor::create(ffn_shape, config_.dtype, device_type_);
    ops::swiglu(swi_out, gate_out, up_out);

    // 12. Down 投影
    auto down_out = Tensor::create(hs_shape, config_.dtype, device_type_);
    auto down_bias = Tensor::create({(size_t)hidden_dim}, config_.dtype, device_type_);
    auto zero_down_cpu = Tensor::create({(size_t)hidden_dim}, config_.dtype, LLAISYS_DEVICE_CPU);
    memset(zero_down_cpu->data(), 0, down_bias->numel() * down_bias->elementSize());
    down_bias->load(zero_down_cpu->data());
    ops::linear(down_out, swi_out, mlp_down_weights_[layer_idx], down_bias);

    // 13. 第二次残差
    auto h2 = Tensor::create(hs_shape, config_.dtype, device_type_);
    ops::add(h2, down_out, h);

    return h2;
}

void Qwen2Model::forward(const int64_t *input_ids, int seq_len, float *output_logits) {
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    llaisys::core::context().setDevice(device_type_, device_id);

    auto token_tensor_cpu = Tensor::create({(size_t)seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    memcpy(token_tensor_cpu->data(), input_ids, seq_len * sizeof(int64_t));
    
    auto token_tensor = Tensor::create({(size_t)seq_len}, LLAISYS_DTYPE_I64, device_type_);
    token_tensor->load(token_tensor_cpu->data());

    auto hidden = Tensor::create({(size_t)seq_len, (size_t)config_.hidden_dim},
                                 config_.dtype, device_type_);

    ops::embedding(hidden, token_tensor, embed_tokens_);

    // 逐层计算
    for (int i = 0; i < config_.num_layers; ++i) {
        hidden = forwardLayer(i, hidden);
    }

    past_len_ += seq_len;

    auto normed = Tensor::create(hidden->shape(), config_.dtype, device_type_);
    ops::rms_norm(normed, hidden, final_norm_weight_, config_.epsilon);

    auto logits = Tensor::create({(size_t)seq_len, (size_t)config_.vocab_size},
                                 config_.dtype, device_type_);
    auto lm_bias = Tensor::create({(size_t)config_.vocab_size}, config_.dtype, device_type_);
    auto zero_bias_cpu = Tensor::create({(size_t)config_.vocab_size}, config_.dtype, LLAISYS_DEVICE_CPU);
    memset(zero_bias_cpu->data(), 0, lm_bias->numel() * lm_bias->elementSize());
    lm_bias->load(zero_bias_cpu->data());

    ops::linear(logits, normed, lm_head_weight_, lm_bias);

    // logits D2H 回读
    auto logits_cpu = Tensor::create({(size_t)seq_len, (size_t)config_.vocab_size},
                                     config_.dtype, LLAISYS_DEVICE_CPU);
    size_t total_bytes = logits->numel() * logits->elementSize();
    #ifdef ENABLE_NVIDIA_API
	        const auto* rt_nv = llaisys::device::nvidia::getRuntimeAPI();
	        rt_nv->memcpy_sync(
	            logits_cpu->data(),
	            logits->data(),
	            total_bytes,
	            LLAISYS_MEMCPY_D2H
	        );
	    #elif defined(ENABLE_ILUVATAR_API)
	        const auto* iluvatar_rt = llaisys::device::iluvatar::getRuntimeAPI();
	        iluvatar_rt->memcpy_sync(
	            logits_cpu->data(),
	            logits->data(),
	            total_bytes,
	            LLAISYS_MEMCPY_D2H
	        );
	    #else
	        std::memcpy(logits_cpu->data(), logits->data(), total_bytes);
	    #endif
    size_t offset = (seq_len - 1) * config_.vocab_size;
    uint16_t *logits_raw = (uint16_t *)logits_cpu->data();
    for (int i = 0; i < config_.vocab_size; ++i) {
        uint16_t h = logits_raw[offset + i];
        float val;
        if (config_.dtype == LLAISYS_DTYPE_BF16) {
            // BF16: 1 sign, 8 exponent, 7 mantissa -> shift left 16 bits to F32
            uint32_t bits = ((uint32_t)h) << 16;
            memcpy(&val, &bits, sizeof(float));
        } else {
            // FP16: 1 sign, 5 exponent, 10 mantissa
            int sign = (h >> 15) & 1;
            int exp = (h >> 10) & 0x1f;
            int mant = h & 0x3ff;
            if (exp == 0x1f) {
                val = mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
            } else if (exp == 0) {
                val = (sign ? -1.0f : 1.0f) * ldexpf((float)mant, -24);
            } else {
                val = (sign ? -1.0f : 1.0f) * ldexpf((float)(mant | 0x400), exp - 25);
            }
        }
        output_logits[i] = val;
    }
}

Qwen2Model::~Qwen2Model() {
    // ========== 核心修复：析构前切回主GPU设备 ==========
    // 保证后续成员 Tensor 自动析构时，在正确的硬件上下文执行显存释放
    if (!device_ids_.empty()) {
        llaisys::core::context().setDevice(device_type_, device_ids_[0]);
    } else {
        llaisys::core::context().setDevice(device_type_, 0);
    }
    // 函数体保持为空，成员 vector<Tensor> 会自动执行析构释放
}
} // namespace llaisys