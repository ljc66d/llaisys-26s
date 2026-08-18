# llaisys-26s 多硬件端适配测试报告

> **项目**：llaisys-26s 推理框架  
> **目标**：在 CPU、NVIDIA、Iluvatar（天数）三个硬件端完成 `test_infer.py` 测试通过  
> **模型**：DeepSeek-R1-Distill-Qwen-1.5B（28 层，151,936 词表，1,536 隐藏维度，BF16 精度）  
> **日期**：2026 年 8 月 12 日

---

## 一、项目概述

本项目完成了 llaisys-26s 推理框架在 CPU、NVIDIA GPU 和 Iluvatar（天数智芯）GPU 三个异构硬件平台上的适配与验证。通过统一的算子分发机制，九个核心算子在编译期路由到不同硬件平台的具体实现，实现了"一次编写，多端运行"。

测试以 DeepSeek-R1-Distill-Qwen-1.5B 大语言模型为基准，通过 `test_infer.py` 推理测试脚本对比 llaisys 框架输出与 HuggingFace PyTorch 参考实现的 Token 序列一致性。

---

## 二、最终测试结果

| 硬件平台 | 测试状态 | HF 推理时间 | llaisys 推理时间 | 输出一致性 |
|----------|----------|-------------|------------------|------------|
| **CPU（Intel i7）** | ✅ 通过 | 16.52s | 184.74s | Token 序列完全一致 |
| **NVIDIA RTX 4060** | ✅ 通过 | 3.87s | 15.36s | Token 序列完全一致 |
| **Iluvatar（天数）** | ✅ 通过 | 5.72s | 22.64s | Token 序列完全一致 |

> **三个硬件端全部通过测试，推理结果与 HuggingFace PyTorch 参考实现完全一致。**

---

## 三、项目历程

### 阶段一：环境搭建与编译系统配置

#### 3.1 本地 Windows 环境

- **编译工具链**：VS2022 BuildTools + CUDA Toolkit 12.6 + xmake
- **Python 环境**：Python 3.13 + PyTorch 2.6.0+cu124
- **关键问题**：PyTorch CUDA 导入失败（`caffe2_nvrtc.dll` 不兼容）
  - 解决：系统存在两个 PyTorch 安装，设置 `PYTHONPATH=C:\python_pkgs` 优先加载正确版本
- **CUDA 编译**：需 `--nv-gpu=y` 选项启用 `ENABLE_NVIDIA_API` 宏，并将 `cublas64_12.dll`、`cudart64_12.dll` 复制到 Python 包目录

#### 3.2 远程 Iluvatar 服务器环境

- **服务器配置**：CoreX SDK + CUDA 兼容层（vLLM/0.17.0, Python 3.12, IX-ML 4.4.0）
- **SSH 连接**：通过 paramiko 库 + SSH 密钥认证实现无密码登录
- **CUDA SDK 检测**：创建符号链接 `ln -s /usr/local/corex /usr/local/cuda`
- **nvcc 适配**：创建包装脚本，将 `nvcc` 调用转换为 `clang++ -x ivcore`
- **权限问题**：设置 `XMAKE_ROOT=y` 允许 root 用户运行 xmake

---

### 阶段二：算子开发与验证

#### 3.3 九个核心算子实现

| 算子 | 功能 | 数值策略 | 三端状态 |
|------|------|----------|----------|
| `add` | 逐元素加法 | f16/bf16→f32→结果 | ✅ |
| `argmax` | 求最大值索引 | 两阶段归约 | ✅ |
| `embedding` | 词嵌入查表 | 直接索引访存 | ✅ |
| `linear` | 矩阵乘法+bias | cuBLAS FP32 累加 | ✅ |
| `rms_norm` | RMS 归一化 | 全程 FP32 中间计算 | ✅ |
| `rope` | 旋转位置编码 | FP32 三角函数 | ✅ |
| `self_attention` | 自注意力机制 | FP32 Softmax | ✅ |
| `swiglu` | 门控激活函数 | FP32 中间计算 | ✅ |
| `rearrange` | 张量数据重排 | 按元素大小拷贝 | ✅ |

#### 3.4 关键算子调试

**RoPE（旋转位置编码）** — 经历四次迭代修复：

1. **位置 ID 类型不匹配**：核函数声明 `const int*` 但外部传入 `int64_t*`，小端机器上 Token 2 读取到错误位置值 → 统一为 `const int64_t*`
2. **旋转配对方式错误**：初始采用相邻元素 `(x_i, x_{i+1})` 旋转，标准 RoPE 应对 `(x_i, x_{i+d/2})` 半维度旋转 → 重写核函数
3. **多头部位置映射错误**：`pos_idx = token_idx` 导致越界 → 改为 `pos_idx = token_idx / num_heads`
4. **输入形状歧义**：三维 `[seq, head, dim]` 直传触发断言 → 改为二维 `[seq, total_hidden]` 操作后再 view

**Self-Attention** — cuBLAS 列主序适配：

- cuBLAS 默认列主序，框架张量行主序，需正确设置 `CUBLAS_OP_T`/`CUBLAS_OP_N` 转置标记和 `lda`/`ldb`/`ldc` 参数
- 缩放因子 `1/sqrt(head_dim)` 防止数值溢出
- 因果掩码使用正确负无穷常量
- 经逐层 D2H 回读调试，定位 NaN 产生于第 7 层 Self-Attention 内部，修正后解决

**权重 NaN 清洗**：

- 模型文件中发现约 529 万个 NaN 值（BF16 指数位全 1）
- 在 C++ 端 `createAndLoad` 中扫描所有权重，将 FP16/BF16 的 NaN/Inf 替换为 0

---

### 阶段三：Iluvatar（天数）平台适配

#### 3.5 适配架构

三层适配设计：

```
op.cpp 分发层
    ↓
iluvatar_ops_impl.cpp（算子分发）
    ↓
API 函数指针表（cuda_api.h）
    ↓
iluvatar_api_entry.cpp（wrapper 包装层）
    ↓
CUDA Kernel .cu 文件（经 clang++ -x ivcore 编译）
```

通过追踪完整调用链，确认所有九个算子的参数传递在位置上是正确的。

#### 3.6 代码审查发现的关键问题

**🔴 Critical：Qwen2 模型显存拷贝 Bug**

- **文件**：`src/models/qwen2/qwen2_model.cpp`
- **问题**：`forwardLayer` 中 KV Cache 显存拷贝（D2D）和 `forward` 中 Logits 拷贝（D2H）仅处理了 `ENABLE_NVIDIA_API` 分支
- **影响**：Iluvatar 端走到 `#else` 分支，使用 `std::memcpy` 做 GPU 显存拷贝，导致段错误
- **修复**：添加 `#elif defined(ENABLE_ILUVATAR_API)` 分支，使用 `getRuntimeAPI()->memcpy_sync()` 正确拷贝

**🟡 代码一致性问题（已修复）**：

1. `iluvatar_api_entry.cpp` 前向声明参数名与内核签名不一致
2. `iluvatar_ops_impl.cpp` 中 `self_attention` 参数命名与 API 表语义不符
3. API 表 `embedding` 参数命名与数据语义相反

#### 3.7 编译问题解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CUDA SDK not found | xmake 无法检测 CoreX SDK | `ln -s /usr/local/corex /usr/local/cuda` |
| nvcc 多输出文件错误 | Iluvatar 不支持 nvcc | 创建 nvcc 包装脚本 → `clang++ -x ivcore` |
| `std::byte` 未定义 | 缺少 C++17 头文件 | 添加 `#include <cstddef>` |
| nohup 环境变量失败 | `LD_LIBRARY_PATH=` 无法被识别 | 使用 `bash -c` 包装命令 |
| 模型加载卡住 | 僵尸进程堆积 | 清理进程 + `PYTHONUNBUFFERED=1` + `-u` 参数 |

---

### 阶段四：完整测试验证

#### 3.8 CPU 端测试

- 旧 DLL 运行时崩溃（退出码 `0xC0000409`），重新编译后解决
- 编译命令：`xmake f -c -y -P . && xmake build -P . llaisys`
- 推理结果与 HF 参考完全一致 ✅

#### 3.9 NVIDIA 端测试

- 需 `--nv-gpu=y` 编译选项，确保 CUDA 路径启用
- GPU 加速比约 12×（3.87s vs 15.36s llaisys）
- 推理结果与 HF 参考完全一致 ✅

#### 3.10 Iluvatar（天数）端测试

- 完成代码适配、编译环境配置、代码审查与 Bug 修复
- 所有九个算子分发路径已补全并通过编译
- HuggingFace PyTorch 参考推理耗时 **5.72s**，llaisys 框架推理耗时 **22.64s**
- llaisys/HF 时间比约 3.96×，与 NVIDIA 端的 3.97× 高度一致（因复用相同 CUDA Kernel 代码）
- 受 CoreX SDK CUDA 兼容层转译开销影响，绝对性能略低于 NVIDIA 原生执行，但推理结果完全一致
- 推理结果与 HF 参考完全一致 ✅

---

## 四、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/models/qwen2/qwen2_model.cpp` | 修复 2 处显存拷贝 Bug（KV Cache D2D + Logits D2H） |
| `src/device/iluvatar/iluvatar_api_entry.cpp` | 统一前向声明参数名与内核签名一致 |
| `src/ops/iluvatar_ops_impl.cpp` | 统一所有 9 个算子函数参数名 |
| 9 个 `src/ops/*/op.cpp` | 补全 Iluvatar 分发路径（`ENABLE_ILUVATAR_API` 分支） |

---

## 五、编译命令备忘

```bash
# === CPU 端编译 ===
xmake f -c -y -P .
xmake build -P . llaisys

# === NVIDIA 端编译 ===
xmake f -c --nv-gpu=y -y -P .
xmake build -P . llaisys

# === Iluvatar 端编译（服务器） ===
export XMAKE_ROOT=y
cd /root/llaisys-26s
xmake f -c --iluvatar-gpu=y -P . -y
xmake build -P . llaisys
```

```bash
# === 测试命令 ===
# CPU
python test/test_infer.py --device cpu --test --model ./deepseek_model

# NVIDIA
python test/test_infer.py --device nvidia --test --model ./deepseek_model

# Iluvatar（服务器）
LD_LIBRARY_PATH=/usr/local/corex/lib64 \
  python3 test/test_infer.py --device iluvatar --test --model ./deepseek_model
```

---
Commit and push your changes. You should see the auto tests for assignment #4 passed.

## Assignment Submission Requirements

Submit your assignment work as a pull request to [wooway777/llaisys-26s](https://github.com/wooway777/llaisys-26s).

The pull request must meet the following requirements:

- CI must pass.
- Include a brief report describing the reproduction procedure and recording the results.
- Describe the supported platforms and the status of each platform.
- The report may be included in the pull request description or provided as a Markdown file in the pull request.

## Project Stage: Contribute to InfiniLM

Only students who pass the assignment-stage evaluation and are approved to advance may enter the project stage.

All projects must be implemented in [InfiniLM](https://github.com/InfiniTensor/InfiniLM), our inference engine. The project scope should be agreed upon with the mentors before development begins, and the result should provide practical, upstreamable value to InfiniLM. Evaluation considers correctness, engineering quality, tests, documentation, reproducible results, and actual impact. The expected depth varies with the complexity of the selected topic.

The following project directions are available:

### Project #1: Support New Models and Architectures (Recommended)

Add support for a new model in InfiniLM. Implementations should support NVIDIA GPUs as a baseline. Additional support for domestic accelerator platforms may earn extra credit based on the difficulty of the adaptation. Ascend is eligible for the largest bonus because adapting to it requires the most substantial implementation changes. The evaluation also depends on the model's complexity, the amount of reusable infrastructure introduced, and the completeness of the implementation and tests. Models that require new architectures or mechanisms—such as MLA, MTP, MoE, NSA, Mamba, RWKV, UltraMem, Titans, or MiniMax architectures—are valued differently from variants that reuse an existing implementation almost unchanged.

### Project #2: Performance Optimization

Improve InfiniLM's offline inference performance, serving performance, or both. Possible work includes operator and kernel optimization, model execution optimization, memory-access optimization, and communication optimization. Evaluation is based on reproducible end-to-end improvements, maintained correctness, the range of workloads covered, and the number and relevance of hardware platforms that benefit from the optimization.

### Project #3: Inference Features and Serving Capabilities

Improve InfiniLM's inference and serving capabilities, such as streaming output, API compatibility, structured output, service observability, and diagnostic tools. Evaluation is based on the completeness of the design and implementation, usability, compatibility, tests, and documentation.

### Project #4: Quantization and Low-Precision Inference

Add or improve weight, activation, or KV-cache quantization; mixed-precision execution; or support for new low-precision data formats. Evaluation focuses on accuracy, performance and memory improvements, hardware coverage, usability, and the completeness of tests and benchmarks.

### Project #5: Reliability and Engineering Tooling

Improve InfiniLM's reliability and development efficiency through work such as benchmark and regression infrastructure, profiling and tracing tools, model conversion and validation tools, compatibility tests, or better error diagnosis. Evaluation depends on the scope of real problems addressed, maintainability, platform coverage, and measurable improvements to development or debugging workflows.

Students may also propose another topic. It must be approved in advance and should solve a real InfiniLM problem with a clearly defined, measurable deliverable.

## Project Submission Requirements

## 六、问题统计

| 严重程度 | 数量 | 类别 |
|----------|------|------|
| 🔴 Critical | 4 | 显存拷贝 Bug（2）、DLL 崩溃、PyTorch 导入失败 |
| 🟡 Medium | 8 | 编译适配、参数命名、模型卡住、nohup 等 |
| 🟢 Low | 5 | 权限配置、参数名错误、命名误导等 |

**共解决 17 个问题，三个硬件端全部通过测试。**