#pragma once

#include "llaisys/cuda/cuda_api.h"

namespace llaisys::device::iluvatar {

// 获取Iluvatar API表（从 llaisys_iluvatar.dll 动态加载）
const llaisys_cuda_api_table_t* get_iluvatar_api();

// 卸载Iluvatar DLL
void unload_iluvatar_dll();

} // namespace llaisys::device::iluvatar