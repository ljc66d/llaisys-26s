#include "iluvatar_loader.h"
#include <string>
#include <mutex>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <climits>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <libgen.h>
#endif

namespace llaisys::device::iluvatar {

#ifdef _WIN32
static HMODULE g_ops_dll = nullptr;
#else
static void* g_ops_so = nullptr;
#endif
static const llaisys_cuda_api_table_t* g_api_table = nullptr;
static std::mutex g_mutex;

typedef llaisys_cuda_api_table_t* (*get_api_table_fn)();

// 获取当前模块所在目录（用于定位同目录下的 llaisys_iluvatar.so/dll）
static std::string get_module_dir() {
#ifdef _WIN32
    char path[MAX_PATH] = {0};
    HMODULE hSelf = GetModuleHandleA("llaisys.dll");
    if (!hSelf) {
        static int dummy = 0;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&dummy, &hSelf);
    }
    if (hSelf) {
        GetModuleFileNameA(hSelf, path, MAX_PATH);
        char* last_slash = strrchr(path, '\\');
        if (last_slash) *(last_slash + 1) = '\0';
    }
    return std::string(path);
#else
    // Linux: 使用 dladdr 获取当前 .so 的路径
    Dl_info dl_info;
    if (dladdr((void*)get_module_dir, &dl_info) && dl_info.dli_fname) {
        std::string path(dl_info.dli_fname);
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            return path.substr(0, pos + 1);
        }
    }
    return "./";
#endif
}

// 加载算子子DLL/SO
static bool load_ops_dll() {
    std::string dir = get_module_dir();

#ifdef _WIN32
    std::string ops_path = dir + "llaisys_iluvatar.dll";
    g_ops_dll = LoadLibraryA(ops_path.c_str());
    if (!g_ops_dll) {
        throw std::runtime_error("加载 llaisys_iluvatar.dll 失败，错误码：" + std::to_string(GetLastError()));
        return false;
    }

    get_api_table_fn fn = reinterpret_cast<get_api_table_fn>(
        GetProcAddress(g_ops_dll, "llaisys_iluvatar_get_api_table")
    );
    if (!fn) {
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
        throw std::runtime_error("找不到 llaisys_iluvatar_get_api_table 入口函数");
        return false;
    }
#else
    std::string ops_path = dir + "llaisys_iluvatar.so";
    g_ops_so = dlopen(ops_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_ops_so) {
        throw std::runtime_error("加载 llaisys_iluvatar.so 失败: " + std::string(dlerror()));
        return false;
    }

    get_api_table_fn fn = reinterpret_cast<get_api_table_fn>(
        dlsym(g_ops_so, "llaisys_iluvatar_get_api_table")
    );
    if (!fn) {
        dlclose(g_ops_so);
        g_ops_so = nullptr;
        throw std::runtime_error("找不到 llaisys_iluvatar_get_api_table 入口函数: " + std::string(dlerror()));
        return false;
    }
#endif

    g_api_table = fn();
    if (!g_api_table) {
#ifdef _WIN32
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
#else
        dlclose(g_ops_so);
        g_ops_so = nullptr;
#endif
        throw std::runtime_error("Iluvatar API表无效");
        return false;
    }

    return true;
}

const llaisys_cuda_api_table_t* get_iluvatar_api() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_api_table) return g_api_table;

    if (!load_ops_dll()) {
        throw std::runtime_error("Iluvatar算子库加载失败");
    }
    return g_api_table;
}

void unload_iluvatar_dll() {
    std::lock_guard<std::mutex> lock(g_mutex);
#ifdef _WIN32
    if (g_ops_dll) {
        FreeLibrary(g_ops_dll);
        g_ops_dll = nullptr;
    }
#else
    if (g_ops_so) {
        dlclose(g_ops_so);
        g_ops_so = nullptr;
    }
#endif
    g_api_table = nullptr;
}

} // namespace llaisys::device::iluvatar