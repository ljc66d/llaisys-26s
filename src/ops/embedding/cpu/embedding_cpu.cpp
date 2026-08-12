#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_tokens, size_t hidden_dim) {
    for (size_t i = 0; i < num_tokens; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * hidden_dim;
        T *dst = out + i * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys {
namespace ops {
namespace cpu {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_tokens, size_t hidden_dim) {
    const auto *index_ptr = reinterpret_cast<const int64_t *>(index);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), index_ptr,
                          reinterpret_cast<const float *>(weight), num_tokens, hidden_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<bf16_t *>(out), index_ptr,
                          reinterpret_cast<const bf16_t *>(weight), num_tokens, hidden_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<fp16_t *>(out), index_ptr,
                          reinterpret_cast<const fp16_t *>(weight), num_tokens, hidden_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace cpu
} // namespace ops
} // namespace llaisys