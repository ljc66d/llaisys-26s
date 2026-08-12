#include "rearrange_gpu.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error("CUDA error: " +                     \
                std::string(cudaGetErrorString(_err)));                   \
        }                                                                 \
    } while(0)

namespace llaisys::ops::iluvatar {

void rearrange(std::byte* dst, const std::byte* src, llaisysDataType_t dtype, size_t total_size) {
    size_t elem_size = llaisys::utils::dsize(dtype);
    CUDA_CHECK(cudaMemcpy(dst, src, total_size * elem_size, cudaMemcpyDeviceToDevice));
}

} // namespace llaisys::ops::iluvatar