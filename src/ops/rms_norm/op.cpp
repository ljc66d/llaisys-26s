#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rms_norm_gpu.hpp"
#endif

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->shape().size() >= 1, "rms_norm: input must be at least 1D");

    auto in_shape = in->shape();
    size_t D = in_shape.back();

    ASSERT(weight->shape().size() == 1 && weight->shape()[0] == D,
           "rms_norm: weight must be 1D with size equal to input last dim");
    ASSERT(out->shape() == in_shape, "rms_norm: output shape must match input shape");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");

    size_t M = in->numel() / D;
    size_t K = D;

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::rms_norm(out->data(), in->data(), weight->data(),
                      out->dtype(), M, K, eps);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rms_norm(out->data(), in->data(), weight->data(),
                      out->dtype(), M, K, eps);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::rms_norm(out->data(), in->data(), weight->data(),
                      out->dtype(), M, K, eps);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops