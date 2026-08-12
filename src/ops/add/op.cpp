#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/add_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/add_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/add_gpu.hpp"
#endif

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(),
           "Add: all tensors must be contiguous.");

    size_t numel = c->numel();

    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());
    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::add(c->data(), a->data(), b->data(), c->dtype(), numel);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::add(c->data(), a->data(), b->data(), c->dtype(), numel);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::add(c->data(), a->data(), b->data(), c->dtype(), numel);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops