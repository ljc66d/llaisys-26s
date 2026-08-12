target("llaisys-iluvatar")
    set_kind("shared")
    set_languages("cxx17")

    if is_plat("windows") then
        set_runtimes("MD")
        add_defines("LLAISYS_ILUVATAR_EXPORT")
    end

    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-fPIC")
    end

    -- 启用 CUDA 编译（Iluvatar SDK 使用 clang++ 作为 CUDA 编译器）
    add_rules("cuda")
    -- 使用 CoreX SDK 的 clang 作为 CUDA 编译器
    set_toolchains("clang")
    -- 设置 CUDA SDK 路径为 CoreX
    set_values("cuda.sdk", "/usr/local/corex")
    set_values("cuda.rdc", false)

    -- 开启Iluvatar API宏
    add_defines("ENABLE_ILUVATAR_API", {public = true})

    -- 头文件路径
    add_includedirs("$(projectdir)/include", {public = true})
    add_includedirs("$(projectdir)/src", {public = true})
    add_includedirs("/usr/local/corex/include", {public = true})

    -- 库搜索路径
    add_linkdirs("/usr/local/corex/lib64")

    -- 源文件：所有算子层的iluvatar实现
    add_files("$(projectdir)/src/ops/*/iluvatar/*.cu")
    -- 入口文件
    add_files("$(projectdir)/src/device/iluvatar/iluvatar_api_entry.cpp")

    -- 链接Iluvatar CUDA兼容库
    add_syslinks("cudart", "cublas")

    -- 输出文件名
    set_basename("llaisys_iluvatar")

    set_targetdir("$(projectdir)/build/$(plat)/$(arch)/$(mode)/bin")

    on_install(function(target) end)
target_end()