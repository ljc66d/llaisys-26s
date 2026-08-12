#include "iluvatar_resource.cuh"

#include <stdexcept>
#include <string>
#include <climits>

#ifdef HAS_ILUVATAR_SDK
#include <ix_runtime.h>
#endif

namespace llaisys::device::iluvatar {

Resource::Resource(size_t device_id)
    : llaisys::device::DeviceResource(
        LLAISYS_DEVICE_ILUVATAR,
        static_cast<int>(device_id)
      ),
      device_id_(device_id)
{
    if (device_id > INT_MAX) {
        throw std::runtime_error("device id exceeds int range");
    }
    // 构造函数严禁调用任何Iluvatar API
}

void Resource::ensure_init() {
    if (initialized_) return;

#ifdef HAS_ILUVATAR_SDK
    int dev_id = static_cast<int>(device_id_);
    ixError_t err = ixSetDevice(dev_id);
    if (err != ixSuccess) {
        throw std::runtime_error("ixSetDevice failed: " + std::string(ixGetErrorString(err)));
    }
    ixFree(nullptr);
#endif
    initialized_ = true;
}

std::unique_ptr<llaisys::device::DeviceResource> createResource(size_t device_id) {
    return std::make_unique<Resource>(device_id);
}

} // namespace llaisys::device::iluvatar