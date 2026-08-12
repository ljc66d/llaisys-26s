add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

-- 全局语言标准
set_languages("cxx17")

-- 全局运行库统一：Windows 下强制 MD，彻底解决 MT/MD 冲突
if is_plat("windows") then
    set_runtimes("MD")
end

add_includedirs("include", "src")

-- CPU 平台
includes("xmake/cpu.lua")

-- NVIDIA CUDA 平台开关
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    includes("xmake/nvidia.lua")
    add_defines("ENABLE_NVIDIA_API")
end

-- Iluvatar CoreX 平台开关
option("iluvatar-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Iluvatar GPU")
option_end()

if has_config("iluvatar-gpu") then
    includes("xmake/iluvatar.lua")
    add_defines("ENABLE_ILUVATAR_API")
end

-- ============================================================
-- 层级1：工具库（最底层）
-- ============================================================
target("llaisys-utils")
    set_kind("static")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/utils/*.cpp")
    on_install(function(target) end)
target_end()

-- ============================================================
-- 层级2：核心引擎层（张量、内存管理基类）
-- ============================================================
target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")

    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/core/*/*.cpp")
    on_install(function(target) end)
target_end()

-- ============================================================
-- 层级3：张量层
-- ============================================================
target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/tensor/*.cpp")
    on_install(function(target) end)
target_end()

-- ============================================================
-- 层级4：设备抽象层（CPU/GPU具体实现）
-- ============================================================
target("llaisys-device")
    set_kind("static")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("include", "src")
    add_files("src/device/*.cpp")
    add_files("src/device/cpu/*.cpp")

    -- NVIDIA GPU 支持：纯C++包装层，零CUDA依赖
    if has_config("nv-gpu") then
        add_files("src/device/nvidia/runtime_api.cpp")
        add_files("src/device/nvidia/cuda_loader.cpp")
        add_defines("ENABLE_NVIDIA_GPU")
    end

    -- Iluvatar GPU 支持：纯C++包装层
    if has_config("iluvatar-gpu") then
        add_files("src/device/iluvatar/runtime_api.cpp")
        add_files("src/device/iluvatar/iluvatar_loader.cpp")
        add_defines("ENABLE_ILUVATAR_GPU")
    end

    add_deps("llaisys-core", "llaisys-tensor", "llaisys-utils")
target_end()

-- ============================================================
-- 层级5：算子层（算子分发与具体实现）
-- ============================================================
target("llaisys-ops")
    set_kind("static")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("include", "src")
    add_files("src/ops/*/op.cpp")
    add_files("src/ops/*/cpu/*.cpp")

    -- NVIDIA GPU 算子包装层
    if has_config("nv-gpu") then
        add_files("src/ops/nvidia_ops_impl.cpp")
    end

    -- Iluvatar GPU 算子包装层
    if has_config("iluvatar-gpu") then
        add_files("src/ops/iluvatar_ops_impl.cpp")
    end

    add_deps("llaisys-device", "llaisys-utils")
target_end()

-- ============================================================
-- 层级6：最终推理引擎 DLL（最上层）
-- ============================================================
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils", "llaisys-core", "llaisys-tensor", "llaisys-device", "llaisys-ops")

    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    add_files("src/llaisys/qwen2.cc")
    add_files("src/models/*/*.cpp")
    set_installdir(".")

    -- 主DLL零CUDA依赖，不需要延迟加载
    after_install(function(target)
        print("Copying runtime DLLs to python/llaisys/libllaisys/ ...")
        -- 拷贝主DLL
        os.cp(target:targetfile(), "python/llaisys/libllaisys/")
        -- 拷贝CUDA算子子DLL（仅在启用 nv-gpu 时存在）
        if has_config("nv-gpu") then
            os.cp("build/$(plat)/$(arch)/$(mode)/bin/llaisys_cuda.dll", "python/llaisys/libllaisys/")
        end
        -- 拷贝Iluvatar算子子DLL（仅在启用 iluvatar-gpu 时存在）
        if has_config("iluvatar-gpu") then
            os.cp("build/$(plat)/$(arch)/$(mode)/bin/llaisys_iluvatar.dll", "python/llaisys/libllaisys/")
        end
    end)
target_end()

-- ============================================================
-- GPU 测试程序（仅在启用 nv-gpu 时构建）
-- ============================================================
if has_config("nv-gpu") then

-- GPU 加法测试程序
target("test_gpu_add")
    set_kind("binary")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end

    add_rules("cuda")
    add_cugencodes("native")
    add_syslinks("cudart")

    add_includedirs("$(projectdir)/include")
    add_includedirs("$(projectdir)/src")
    add_files("$(projectdir)/tests/test_gpu_add.cpp")
    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")
target_end()

-- RMSNorm GPU 单测
target("test_rmsnorm_gpu")
    set_kind("binary")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end

    add_rules("cuda")
    add_cugencodes("native")
    add_syslinks("cudart")

    add_includedirs("$(projectdir)/include", "$(projectdir)/src")
    add_files("$(projectdir)/tests/test_rmsnorm_gpu.cpp")
    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")
target_end()

-- Embedding GPU 单测
target("test_embedding_gpu")
    set_kind("binary")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end

    add_rules("cuda")
    add_cugencodes("native")
    add_syslinks("cudart")

    add_includedirs("$(projectdir)/include", "$(projectdir)/src")
    add_files("$(projectdir)/tests/test_embedding_gpu.cpp")
    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")
target_end()

-- RoPE GPU 单测
target("test_rope_gpu")
    set_kind("binary")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end

    add_rules("cuda")
    add_cugencodes("native")
    add_syslinks("cudart")

    add_includedirs("$(projectdir)/include", "$(projectdir)/src")
    add_files("$(projectdir)/tests/test_rope_gpu.cpp")
    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")
target_end()

-- RMSNorm 大尺寸验证单测
target("test_rmsnorm_large_gpu")
    set_kind("binary")
    set_languages("cxx17")
    if is_plat("windows") then
        set_runtimes("MD")
    end

    add_rules("cuda")
    add_cugencodes("native")
    add_syslinks("cudart")

    add_includedirs("$(projectdir)/include", "$(projectdir)/src")
    add_files("$(projectdir)/tests/test_rmsnorm_large_gpu.cpp")
    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")
target_end()

end -- end if has_config("nv-gpu")