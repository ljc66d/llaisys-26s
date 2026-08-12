#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t B, size_t S_q, size_t S_kv,
                     size_t H_q, size_t H_kv, size_t D, float scale) {
    size_t n_repeat = H_q / H_kv;
    int mask_offset = static_cast<int>(S_kv) - static_cast<int>(S_q);

    std::vector<float> scores(S_q * S_kv);
    std::vector<float> weights(S_q * S_kv);

    for (size_t b = 0; b < B; ++b) {
        const T *q_b = q + b * (S_q * H_q * D);
        const T *k_b = k + b * (S_kv * H_kv * D);
        const T *v_b = v + b * (S_kv * H_kv * D);
        T *out_b = attn_val + b * (S_q * H_q * D);

        for (size_t h = 0; h < H_q; ++h) {
            size_t h_kv = h / n_repeat;


            for (size_t i = 0; i < S_q; ++i) {
                for (size_t j = 0; j < S_kv; ++j) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        
                        float q_val = llaisys::utils::cast<float>(
                            q_b[(i * H_q + h) * D + d]);
                        float k_val = llaisys::utils::cast<float>(
                            k_b[(j * H_kv + h_kv) * D + d]);
                        dot += q_val * k_val;
                    }
                    scores[i * S_kv + j] = dot * scale;
                }
            }


            for (size_t i = 0; i < S_q; ++i) {
        
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < S_kv; ++j) {
                    int j_causal = static_cast<int>(j) - mask_offset;
                    if (j_causal <= static_cast<int>(i)) {
                        if (scores[i * S_kv + j] > max_val) {
                            max_val = scores[i * S_kv + j];
                        }
                    }
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j < S_kv; ++j) {
                    int j_causal = static_cast<int>(j) - mask_offset;
                    if (j_causal <= static_cast<int>(i)) {
                        sum_exp += std::exp(scores[i * S_kv + j] - max_val);
                    }
                }


                for (size_t j = 0; j < S_kv; ++j) {
                    int j_causal = static_cast<int>(j) - mask_offset;
                    if (j_causal <= static_cast<int>(i)) {
                        weights[i * S_kv + j] = std::exp(scores[i * S_kv + j] - max_val) / sum_exp;
                    } else {
                        weights[i * S_kv + j] = 0.0f;
                    }
                }
            }


            for (size_t i = 0; i < S_q; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (size_t j = 0; j < S_kv; ++j) {
                        float w = weights[i * S_kv + j];
                        if (w > 0.0f) {
                            float v_val = llaisys::utils::cast<float>(
                                v_b[(j * H_kv + h_kv) * D + d]);
                            val += w * v_val;
                        }
                    }
 
                    out_b[(i * H_q + h) * D + d] = llaisys::utils::cast<T>(val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t B, size_t S_q, size_t S_kv,
                    size_t H_q, size_t H_kv, size_t D, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), 
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),  
                               reinterpret_cast<const float *>(v),
                               B, S_q, S_kv, H_q, H_kv, D, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               B, S_q, S_kv, H_q, H_kv, D, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               B, S_q, S_kv, H_q, H_kv, D, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu