#ifndef LLAISYS_MODELS_QWEN2_MODEL_HPP
#define LLAISYS_MODELS_QWEN2_MODEL_HPP

#include "../../tensor/tensor.hpp" 
#include "llaisys/models/qwen2.h"
#include <string>
#include <vector>

namespace llaisys {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Config &config,
               llaisysDeviceType_t device_type,
               int *device_ids,
               int ndevice);
    ~Qwen2Model();

    void loadWeight(const std::string &name,
                    const void *data,
                    const std::vector<int64_t> &shape);
    void forward(const int64_t *input_ids,
                 int seq_len,
                 float *output_logits);
    void resetKVCache();

private:
    LlaisysQwen2Config config_;
    llaisysDeviceType_t device_type_;
    std::vector<int> device_ids_;


    tensor_t embed_tokens_;
    std::vector<tensor_t> attn_norm_weights_;
    std::vector<tensor_t> attn_q_weights_;
    std::vector<tensor_t> attn_k_weights_;
    std::vector<tensor_t> attn_v_weights_;
    std::vector<tensor_t> attn_o_weights_;
    std::vector<tensor_t> mlp_norm_weights_;
    std::vector<tensor_t> mlp_gate_weights_;
    std::vector<tensor_t> mlp_up_weights_;
    std::vector<tensor_t> mlp_down_weights_;
    std::vector<tensor_t> attn_q_bias_;
    std::vector<tensor_t> attn_k_bias_;
    std::vector<tensor_t> attn_v_bias_;

    tensor_t final_norm_weight_;
    tensor_t lm_head_weight_;


    std::vector<tensor_t> k_cache_;
    std::vector<tensor_t> v_cache_;
    int past_len_ = 0;

    void allocateKVCache();
    tensor_t forwardLayer(int layer_idx, const tensor_t &hidden_states);
    void print_fp16_first_val(const char *tag, tensor_t t);
};

} // namespace llaisys
#endif