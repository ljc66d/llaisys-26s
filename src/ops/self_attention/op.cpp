#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "llaisys.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/self_attention_gpu.hpp"
#endif

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->shape().size() == 3, "self_attention: q must be 3D");
    ASSERT(k->shape().size() == 3, "self_attention: k must be 3D");
    ASSERT(v->shape().size() == 3, "self_attention: v must be 3D");

    auto q_shape = q->shape();
    auto k_shape = k->shape();


    size_t S_q = q_shape[0];
    size_t H_q = q_shape[1];
    size_t D = q_shape[2];
    size_t S_kv = k_shape[0];
    size_t H_kv = k_shape[1];
    size_t B = 1;
        
    ASSERT(attn_val->shape() == q->shape(), "self_attention: output shape mismatch");
    ASSERT(q->dtype() == k->dtype(), "self_attention: q and k dtype mismatch");
    ASSERT(k->dtype() == v->dtype(), "self_attention: k and v dtype mismatch");
    ASSERT(attn_val->dtype() == q->dtype(), "self_attention: output and q dtype mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: tensors must be contiguous");
    ASSERT(H_q % H_kv == 0, "self_attention: GQA head count mismatch");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), B, S_q, S_kv, H_q, H_kv, D, scale
        );
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), B, S_q, S_kv, H_q, H_kv, D, scale
        );
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), B, S_q, S_kv, H_q, H_kv, D, scale
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops