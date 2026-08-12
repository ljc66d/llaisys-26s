#pragma once

#include "../device_resource.hpp"
#include <cstddef>
#include <memory>

namespace llaisys::device::nvidia {

class Resource : public llaisys::device::DeviceResource {
public:
    explicit Resource(size_t device_id);
    void ensure_init();
private:
    size_t device_id_;
    bool initialized_ = false;
};

std::unique_ptr<llaisys::device::DeviceResource> createResource(size_t device_id);

} // namespace llaisys::device::nvidia