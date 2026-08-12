#include"argmax_gpu.hpp"
#include"../../../utils.hpp"
#include<cuda_runtime.h>
#include<stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error("CUDA error: " +                     \
                std::string(cudaGetErrorString(_err)));                   \
        }                                                                 \
    } while(0)
__global__ void argmax_block_kernel_f16(const __half* data, __half* block_val, int64_t* block_idx, int64_t n){
    __shared__ __half s_val[256];
    __shared__ int64_t s_idx[256];

    int tid = threadIdx.x;
    int64_t global_idx = blockIdx.x * 256 + tid;

    if (global_idx < n){
        s_val[tid] = data[global_idx];
        s_idx[tid] = global_idx;
    }else{
        s_val[tid] = __float2half(-1e30f);
        s_idx[tid] = -1;
    }
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1){
        if(tid < step){
            if(__hlt(s_val[tid], s_val[tid + step])){
                s_val[tid] = s_val[tid + step];
                s_idx[tid] = s_idx[tid + step];
            }
        }
        __syncthreads();
    }
    if(tid == 0){
        block_val[blockIdx.x] = s_val[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}
__global__ void argmax_block_kernel_bf16(const __nv_bfloat16* data, __nv_bfloat16* block_val, int64_t* block_idx, int64_t n){
    __shared__ __nv_bfloat16 s_val[256];
    __shared__ int64_t s_idx[256];

    int tid = threadIdx.x;
    int64_t global_idx = blockIdx.x * 256 + tid;

    if (global_idx < n){
        s_val[tid] = data[global_idx];
        s_idx[tid] = global_idx;
    }else{
        s_val[tid] = __float2bfloat16(-1e30f);
        s_idx[tid] = -1;
    }
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1){
        if(tid < step){
            if(__hlt(s_val[tid], s_val[tid + step])){
                s_val[tid] = s_val[tid + step];
                s_idx[tid] = s_idx[tid + step];
            }
        }
        __syncthreads();
    }
    if(tid == 0){
        block_val[blockIdx.x] = s_val[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}
__global__ void argmax_block_kernel(const float* data,float* block_val,int64_t* block_idx,int64_t n){
    __shared__ float s_val[256];
    __shared__ int64_t s_idx[256];

    int tid=threadIdx.x;
    int64_t global_idx=blockIdx.x*256+tid;

    if (global_idx<n){
        s_val[tid]=data[global_idx];
        s_idx[tid]=global_idx;
    }else{
        s_val[tid]=-1e30f;
        s_idx[tid]=-1;
    }
    __syncthreads();
    for (int step=128;step>0;step>>=1){
        if(tid<step){
            if(s_val[tid]<s_val[tid+step]){
                s_val[tid]=s_val[tid+step];
                s_idx[tid]=s_idx[tid+step];
            }
        }
        __syncthreads();
    }
    if(tid==0){
        block_val[blockIdx.x]=s_val[0];
        block_idx[blockIdx.x]=s_idx[0];
    }
}
namespace llaisys::ops::iluvatar{
void argmax(std::byte* out_idx,std::byte* out_val,const std::byte* vals,llaisysDataType_t type,size_t size){
    const int block_size=256;
    const int block_num=static_cast<int>((size+block_size-1)/block_size);

    switch(type){
    case LLAISYS_DTYPE_F32:{
        const float* dev_data=reinterpret_cast<const float*>(vals);
        float* dev_out_val=reinterpret_cast<float*>(out_val);
        int64_t* dev_out_idx=reinterpret_cast<int64_t*>(out_idx);
        
        float* tmp_val;
        int64_t* tmp_idx;
        CUDA_CHECK(cudaMalloc(&tmp_val,block_num*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tmp_idx,block_num*sizeof(int64_t)));

        argmax_block_kernel<<<block_num,block_size>>>(dev_data,tmp_val,tmp_idx,size);
        CUDA_CHECK(cudaGetLastError());

        argmax_block_kernel<<<1,block_size>>>(tmp_val,dev_out_val,dev_out_idx,block_num);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(tmp_val));
        CUDA_CHECK(cudaFree(tmp_idx));
        break;
    }    
    case LLAISYS_DTYPE_F16:{
        const __half* dev_data = reinterpret_cast<const __half*>(vals);
        __half* dev_out_val = reinterpret_cast<__half*>(out_val);
        int64_t* dev_out_idx = reinterpret_cast<int64_t*>(out_idx);
    
        __half* tmp_val;
        int64_t* tmp_idx;
        CUDA_CHECK(cudaMalloc(&tmp_val, block_num * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&tmp_idx, block_num * sizeof(int64_t)));

        argmax_block_kernel_f16<<<block_num, block_size>>>(dev_data, tmp_val, tmp_idx, size);
        CUDA_CHECK(cudaGetLastError());

        argmax_block_kernel_f16<<<1, block_size>>>(tmp_val, dev_out_val, dev_out_idx, block_num);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(tmp_val));
        CUDA_CHECK(cudaFree(tmp_idx));
        break;
    }
    case LLAISYS_DTYPE_BF16:{
        const __nv_bfloat16* dev_data = reinterpret_cast<const __nv_bfloat16*>(vals);
        __nv_bfloat16* dev_out_val = reinterpret_cast<__nv_bfloat16*>(out_val);
        int64_t* dev_out_idx = reinterpret_cast<int64_t*>(out_idx);
    
        __nv_bfloat16* tmp_val;
        int64_t* tmp_idx;
        CUDA_CHECK(cudaMalloc(&tmp_val, block_num * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&tmp_idx, block_num * sizeof(int64_t)));

        argmax_block_kernel_bf16<<<block_num, block_size>>>(dev_data, tmp_val, tmp_idx, size);
        CUDA_CHECK(cudaGetLastError());

        argmax_block_kernel_bf16<<<1, block_size>>>(tmp_val, dev_out_val, dev_out_idx, block_num);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(tmp_val));
        CUDA_CHECK(cudaFree(tmp_idx));
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
}