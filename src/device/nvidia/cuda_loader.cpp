#include "cuda_loader.h"
#include <windows.h>
#include <string>
#include <mutex>
#include <stdexcept>
#include <cstdio>

namespace llaisys::device::nvidia {

static HMODULE g_cudart_dll = nullptr;
static HMODULE g_ops_dll = nullptr;
static const llaisys_cuda_api_table_t* g_api_table = nullptr;
static std::mutex g_mutex;

// CUDA 运行时函数指针类型
typedef int cudaError_t;
typedef cudaError_t (*cuda_set_device_fn)(int);
typedef cudaError_t (*cuda_device_synchronize_fn)();

typedef llaisys_cuda_api_table_t* (*get_api_table_fn)();

// ========== 阶段1：动态加载 cudart 并初始化 GPU ==========
static bool init_cuda_runtime() {
    // 按优先级尝试不同版本的 cudart
    const char* dll_names[] = {
        "cudart64_120.dll",
        "cudart64_12.dll",
        "cudart64_13.dll",
        nullptr
    };

    for (int i = 0; dll_names[i]; i++) {
        g_cudart_dll = LoadLibraryA(dll_names[i]);
        if (g_cudart_dll) break;
    }

    if (!g_cudart_dll) {
        throw std::runtime_error("加载 CUDA 运行时失败，请确认已安装 CUDA Toolkit");
        return false;
    }

    // 获取基础函数
    cuda_set_device_fn set_device = reinterpret_cast<cuda_set_device_fn>(
        GetProcAddress(g_cudart_dll, "cudaSetDevice")
    );
    cuda_device_synchronize_fn sync = reinterpret_cast<cuda_device_synchronize_fn>(
        GetProcAddress(g_cudart_dll, "cudaDeviceSynchronize")
    );

    if (!set_device || !sync) {
        FreeLibrary(g_cudart_dll);
        g_cudart_dll = nullptr;
        throw std::runtime_error("CUDA 运行时库中找不到基础函数");
        return false;
    }

    // 初始化 GPU（关键：这一步完成后，CUDA驱动全部加载，后续算子DLL不会再触发死锁）
    int err = set_device(0);
    if (err != 0) {
        FreeLibrary(g_cudart_dll);
        g_cudart_dll = nullptr;
        throw std::runtime_error("初始化 GPU 失败，错误码：" + std::to_string(err));
        return false;
    }

    sync(); // 同步一次，确保驱动完全就绪
    return true;
}

// ========== 阶段2：加载算子子 DLL ==========
static bool load_ops_dll() {
    // 获取当前 DLL（llaisys.dll）的完整路径，用于定位同目录下的 llaisys_cuda.dll
    char dll_path[MAX_PATH] = {0};
    HMODULE hSelf = GetModuleHandleA("llaisys.dll");
    if (hSelf) {
        GetModuleFileNameA(hSelf, dll_path, MAX_PATH);
        char* last_slash = strrchr(dll_path, '\\');
        if (last_slash) {
            *(last_slash + 1) = '\0';
        }
    }

    // 如果通过模块名获取失败，回退到静态地址方式
    if (dll_path[0] == '\0') {
        static int dummy = 0;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&dummy, &hSelf);
        if (hSelf) {
            GetModuleFileNameA(hSelf, dll_path, MAX_PATH);
            char* last_slash = strrchr(dll_path, '\\');
            if (last_slash) {
                *(last_slash + 1) = '\0';
            }
        }
    }

    std::string ops_path = std::string(dll_path) + "llaisys_cuda.dll";
    g_ops_dll = LoadLibraryA(ops_path.c_str());
    if (!g_ops_dll) {
        throw std::runtime_error("加载 llaisys_cuda.dll 失败");
        return false;
    }

    get_api_table_fn fn = reinterpret_cast<get_api_table_fn>(
        GetProcAddress(g_ops_dll, "llaisys_cuda_get_api_table")
    );
    if (!fn) {
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
        throw std::runtime_error("找不到 llaisys_cuda_get_api_table 入口函数");
        return false;
    }

    g_api_table = fn();
    if (!g_api_table) {
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
        throw std::runtime_error("CUDA API 表无效");
        return false;
    }

    return true;
}

// ========== 对外接口：首次调用自动完成全部初始化 ==========
const llaisys_cuda_api_table_t* get_cuda_api() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_api_table) return g_api_table;

    // 顺序执行：加载cudart → 初始化GPU → 加载算子DLL
    if (!init_cuda_runtime()) {
        throw std::runtime_error("CUDA 运行时初始化失败");
    }

    if (!load_ops_dll()) {
        throw std::runtime_error("CUDA 算子库加载失败");
    }

    return g_api_table;
}

void unload_cuda_dll() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ops_dll) {
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
    }
    if (g_cudart_dll) {
        FreeLibrary(g_cudart_dll);
        g_cudart_dll = nullptr;
    }
    g_api_table = nullptr;
}

} // namespace