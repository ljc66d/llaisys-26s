#include "nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <climits>

namespace llaisys::device::nvidia {

Resource::Resource(size_t device_id)
    : llaisys::device::DeviceResource(
        LLAISYS_DEVICE_NVIDIA,
        static_cast<int>(device_id)
      ),
      device_id_(device_id)
{
    if (device_id > INT_MAX) {
        throw std::runtime_error("device id exceeds int range");
    }
    // !!! 构造函数严禁调用任何CUDA API !!!
}

void Resource::ensure_init() {
    if (initialized_) return;

    int dev_id = static_cast<int>(device_id_);
    cudaError_t err = cudaSetDevice(dev_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
    }
    cudaFree(nullptr);
    initialized_ = true;
}

std::unique_ptr<llaisys::device::DeviceResource> createResource(size_t device_id) {
    return std::make_unique<Resource>(device_id);
}

} // namespace llaisys::device::nvidia