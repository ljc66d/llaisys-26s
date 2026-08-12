#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rope_gpu.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    ASSERT(in->shape().size() >= 2, "rope: input must be at least 2D");
    auto in_shape = in->shape();
    size_t D = in_shape.back();
    ASSERT(D % 2 == 0, "rope: input last dim must be even");

    ASSERT(pos_ids->shape().size() == 1, "rope: pos_ids must be 1D");
    size_t seq_len = in_shape[in_shape.size() - 2];
    size_t total_tokens = in->numel() / D;
    size_t batch = total_tokens / seq_len;
    size_t pos_len = pos_ids->shape()[0];
    ASSERT(pos_len == batch || pos_len == seq_len || pos_len == total_tokens,
           "rope: pos_ids length must be batch, seq_len, or total_tokens");

    ASSERT(out->shape() == in_shape, "rope: output shape must match input shape");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::rope(out->data(), in->data(), pos_ids->data(),
                  out->dtype(), batch, seq_len, D, theta, pos_len);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rope(out->data(), in->data(), pos_ids->data(),
                  out->dtype(), batch, seq_len, D, theta, pos_len);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::rope(out->data(), in->data(), pos_ids->data(),
                  out->dtype(), batch, seq_len, D, theta, pos_len);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops