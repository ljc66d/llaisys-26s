#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#include "llaisys.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/linear_gpu.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }

    ASSERT(in->shape().size() >= 2, "linear: input must be at least 2D");
    ASSERT(weight->shape().size() == 2, "linear: weight must be 2D [out_features, in_features]");

    auto in_shape = in->shape();
    auto wt_shape = weight->shape();
    ASSERT(in_shape.back() == wt_shape[1], "linear: input last dim must equal weight in_features");

    if (bias != nullptr) {
        ASSERT(bias->shape().size() == 1 && bias->shape()[0] == wt_shape[0],
               "linear: bias must be 1D with size out_features");
        CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
    }

    auto expected_out_shape = in_shape;
    expected_out_shape.back() = wt_shape[0];
    ASSERT(out->shape() == expected_out_shape, "linear: output shape mismatch");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: input/weight/output must be contiguous");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous");
    }

    size_t M = in->numel() / wt_shape[1];
    size_t K = wt_shape[1];
    size_t N = wt_shape[0];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::linear(out->data(), in->data(), weight->data(),
                    bias != nullptr ? bias->data() : nullptr,
                    out->dtype(), M, K, N);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::linear(out->data(), in->data(), weight->data(),
                    bias != nullptr ? bias->data() : nullptr,
                    out->dtype(), M, K, N);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::linear(out->data(), in->data(), weight->data(),
                    bias != nullptr ? bias->data() : nullptr,
                    out->dtype(), M, K, N);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops