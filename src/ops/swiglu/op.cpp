#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/swiglu_gpu.hpp"
#endif
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->shape() == gate->shape(), "swiglu: out and gate shape mismatch");
    ASSERT(out->shape() == up->shape(), "swiglu: out and up shape mismatch");
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");

    size_t N = out->numel();

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), N);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), N);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::swiglu(out->data(), gate->data(), up->data(), out->dtype(), N);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops