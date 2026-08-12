#pragma once
#include "llaisys/cuda/cuda_api.h"
#include <stdexcept>

namespace llaisys::device::nvidia {

// 获取 CUDA API 表（首次调用自动加载 DLL）
const llaisys_cuda_api_table_t* get_cuda_api();

// 手动释放 DLL（一般不用主动调用，程序退出自动释放）
void unload_cuda_dll();

} // namespace llaisys::device::nvidia