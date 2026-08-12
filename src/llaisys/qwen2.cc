#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_model.hpp"

__C {

    __export LlaisysQwen2ModelHandle llaisysQwen2ModelCreate(
        const LlaisysQwen2Config *config,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        auto *model = new llaisys::Qwen2Model(*config, device, device_ids, ndevice);
        return reinterpret_cast<LlaisysQwen2ModelHandle>(model);
    }

    __export void llaisysQwen2ModelDestroy(
        LlaisysQwen2ModelHandle model) {
        delete reinterpret_cast<llaisys::Qwen2Model *>(model);
    }

    __export void llaisysQwen2ModelLoadWeight(
        LlaisysQwen2ModelHandle model,
        const char *name,
        const void *data,
        const int64_t *shape,
        int ndim) {
        auto *m = reinterpret_cast<llaisys::Qwen2Model *>(model);
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        m->loadWeight(std::string(name), data, shape_vec);
    }

    __export void llaisysQwen2ModelForward(
        LlaisysQwen2ModelHandle model,
        const int64_t *input_ids,
        int seq_len,
        float *output_logits) {
        auto *m = reinterpret_cast<llaisys::Qwen2Model *>(model);
        m->forward(input_ids, seq_len, output_logits);
    }

    __export void llaisysQwen2ModelResetKVCache(
        LlaisysQwen2ModelHandle model) {
        auto *m = reinterpret_cast<llaisys::Qwen2Model *>(model);
        m->resetKVCache();
    }

} // extern "C"