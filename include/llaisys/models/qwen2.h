#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    typedef struct {
        int num_layers;        // Transformer层数
        int hidden_dim;        // 隐藏层维度
        int num_heads;         // 注意力头数
        int num_kv_heads;      // KV头数
        int head_dim;          // 每个头的维度
        int intermediate_size; // FFN中间层维度
        int vocab_size;        // 词表大小
        int max_seq_len;       // KV缓存最大长度
        float epsilon;         // RMSNorm epsilon
        float theta;           // RoPE theta
        llaisysDataType_t dtype; // 权重数据类型
    } LlaisysQwen2Config;

    

    struct LlaisysQwen2Model;
    typedef struct LlaisysQwen2Model *LlaisysQwen2ModelHandle;

    __export LlaisysQwen2ModelHandle llaisysQwen2ModelCreate(
        const LlaisysQwen2Config *config,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice);

    __export void llaisysQwen2ModelDestroy(
        LlaisysQwen2ModelHandle model);

    __export void llaisysQwen2ModelLoadWeight(
        LlaisysQwen2ModelHandle model,
        const char *name,
        const void *data,
        const int64_t *shape,
        int ndim);

    __export void llaisysQwen2ModelForward(
        LlaisysQwen2ModelHandle model,
        const int64_t *input_ids,
        int seq_len,
        float *output_logits);

    __export void llaisysQwen2ModelResetKVCache(
        LlaisysQwen2ModelHandle model);
}
#endif // LLAISYS_MODELS_QWEN2_H
