#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_gpu.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/embedding_gpu.hpp"
#endif

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index tensor must be int64 type.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(weight->shape().size() == 2, "embedding: weight must be a 2D matrix [vocab_size, hidden_dim].");

    auto idx_shape = index->shape();
    auto wt_shape = weight->shape();
    auto expected_shape = idx_shape;
    expected_shape.push_back(wt_shape[1]);
    ASSERT(out->shape() == expected_shape,
           "embedding: output shape mismatch. Expected index.shape + [hidden_dim].");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous.");
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            idx_shape[0], // num_tokens
            wt_shape[1]   // hidden_dim
        );
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            idx_shape[0], // num_tokens
            wt_shape[1]   // hidden_dim
        );
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            idx_shape[0], // num_tokens
            wt_shape[1]   // hidden_dim
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops