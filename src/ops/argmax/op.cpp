#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/argmax_gpu.hpp"
#endif
#include "llaisys.h"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    ASSERT(vals->numel() > 0, "argmax: input tensor must not be empty.");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
           "argmax: max_idx must be int64 type.");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1,
           "argmax: output tensors must have exactly one element.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "argmax: all tensors must be contiguous.");

    size_t size = vals->numel();

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());
    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                    vals->dtype(), size);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                    vals->dtype(), size);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::argmax(max_idx->data(), max_val->data(), vals->data(),
                    vals->dtype(), size);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops