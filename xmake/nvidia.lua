target("llaisys-cuda")
    set_kind("shared")
    set_languages("cxx17")

    if is_plat("windows") then
        set_runtimes("MD")
        add_defines("LLAISYS_CUDA_EXPORT")
    end

    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-fPIC")
    end

    -- 启用 CUDA 编译
    add_rules("cuda")
    add_cugencodes("native")
    set_values("cuda.rdc", false)

    -- 关键：开启 NVIDIA API 宏，算子头文件才会暴露 nvidia 命名空间
    add_defines("ENABLE_NVIDIA_API", {public = true})

    -- 头文件路径
    add_includedirs("$(projectdir)/include", {public = true})
    add_includedirs("$(projectdir)/src", {public = true})

    -- 源文件：所有算子层 .cu
    add_files("$(projectdir)/src/ops/*/nvidia/*.cu")
    -- 入口文件
    add_files("$(projectdir)/src/device/nvidia/cuda_api_entry.cu")

    -- 链接 CUDA 运行时 + cuBLAS
    add_syslinks("cudart", "cublas")

    -- 输出文件名：llaisys_cuda.dll
    set_basename("llaisys_cuda")

    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")

    on_install(function(target) end)
target_end()