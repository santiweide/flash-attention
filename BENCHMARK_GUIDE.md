# Flash Attention A100 Benchmark 指南

本指南介绍如何对 `csrc/flash_attn/src/flash_fwd_kernel.h` 中的A100实现进行性能测试和分析。

## 📋 前置要求

### 硬件要求
- NVIDIA GPU (推荐 A100, 但 RTX 3090/4090, A6000 等也可以)
- GPU 计算能力 >= 8.0 (Ampere架构)

### 软件要求
- CUDA >= 12.0
- PyTorch >= 2.2
- Python >= 3.8

## 🔧 安装步骤

### 1. 安装依赖
```bash
# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja einops

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. 编译安装 Flash Attention
```bash
cd /Users/michu/Documents/flash-attention

# 从源码编译安装 (首次安装需要3-10分钟)
python setup.py install

# 或者使用pip安装 (如果内存不足，使用MAX_JOBS限制并行编译数)
# MAX_JOBS=4 pip install . --no-build-isolation
```

### 3. 验证安装
```bash
python -c "from flash_attn import flash_attn_func; print('✅ Flash Attention installed successfully!')"
```

## 🚀 快速开始 - 运行Benchmark

### 基础用法
```bash
# 运行默认配置 (batch=2, seqlen=1024, heads=8, dim=64)
python minimal_benchmark.py

# 自定义参数
python minimal_benchmark.py --batch 4 --seqlen 2048 --nheads 16 --headdim 128

# 只测试forward pass (用于profiling)
python minimal_benchmark.py --no-grad

# 使用causal masking
python minimal_benchmark.py --causal

# 使用bf16数据类型
python minimal_benchmark.py --dtype bf16

# 增加重复次数以获得更稳定的结果
python minimal_benchmark.py --repeats 100
```

### 常见配置示例

#### GPT-2/GPT-3 类似配置
```bash
# GPT-2 Small (12层, 12头, 768维度)
python minimal_benchmark.py --batch 8 --seqlen 1024 --nheads 12 --headdim 64 --causal

# GPT-3 类似 (96层, 96头, 12288维度)
python minimal_benchmark.py --batch 2 --seqlen 2048 --nheads 96 --headdim 128 --causal
```

#### BERT 类似配置
```bash
# BERT-Base (12层, 12头, 768维度)
python minimal_benchmark.py --batch 16 --seqlen 512 --nheads 12 --headdim 64

# BERT-Large (24层, 16头, 1024维度)
python minimal_benchmark.py --batch 8 --seqlen 512 --nheads 16 --headdim 64
```

#### 性能测试配置
```bash
# 短序列测试
python minimal_benchmark.py --batch 32 --seqlen 512 --nheads 8 --headdim 64

# 长序列测试
python minimal_benchmark.py --batch 1 --seqlen 8192 --nheads 8 --headdim 64

# 极长序列测试 (需要大显存)
python minimal_benchmark.py --batch 1 --seqlen 16384 --nheads 8 --headdim 64
```

## 📊 性能监测方法

### 1. 基础性能指标 (内置)

脚本会自动输出以下指标：
- **执行时间**: min/avg/max (毫秒)
- **吞吐量**: TFLOPs/s (万亿次浮点运算/秒)
- **内存占用**: GPU显存使用情况

输出示例：
```
Forward Pass Results:
  Time (min):      2.345 ms
  Time (avg):      2.456 ms
  Time (max):      2.678 ms
  TFLOPs/s (peak): 156.32
  TFLOPs/s (avg):  149.87
```

### 2. NVIDIA Nsight Systems (nsys) - 系统级性能分析

**用途**: 分析整体执行流程、找出性能瓶颈、查看kernel调用时间线

```bash
# 基础profiling
nsys profile \
    -o flash_attn_profile \
    --stats=true \
    python minimal_benchmark.py --no-grad --repeats 10

# 查看报告
nsys stats flash_attn_profile.nsys-rep

# 使用GUI查看 (需要在本地电脑安装Nsight Systems)
# 下载 .nsys-rep 文件，用Nsight Systems打开
```

**高级选项**:
```bash
# 包含CUDA API和kernel详细信息
nsys profile \
    -o flash_attn_detailed \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite true \
    python minimal_benchmark.py --batch 4 --seqlen 2048 --no-grad
```

**关键指标**:
- Kernel执行时间占比
- Memory copy时间
- GPU利用率
- SM占用率

### 3. NVIDIA Nsight Compute (ncu) - Kernel级性能分析

**用途**: 详细分析单个CUDA kernel的性能，找出优化机会

```bash
# 基础kernel分析
ncu -o flash_attn_kernel \
    --set full \
    python minimal_benchmark.py --no-grad --repeats 1

# 只分析特定kernel (Flash Attention的forward kernel)
ncu -o flash_attn_fwd \
    --set full \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad --repeats 1

# 快速分析 (只看关键指标)
ncu --set basic \
    --kernel-name regex:"flash" \
    python minimal_benchmark.py --no-grad --repeats 1
```

**分析特定指标**:
```bash
# 内存带宽分析
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad

# 计算单元利用率
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad

# Tensor Core利用率 (FP16/BF16)
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad --dtype fp16
```

**查看报告**:
```bash
# 生成文本报告
ncu --import flash_attn_kernel.ncu-rep

# 使用GUI查看 (更直观)
# 下载 .ncu-rep 文件，用Nsight Compute GUI打开
```

### 4. PyTorch Profiler - Python级性能分析

创建 `profile_benchmark.py`:
```python
import torch
from flash_attn import flash_attn_func

# 配置
batch, seqlen, nheads, headdim = 2, 1024, 8, 64
q = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(10):
    _ = flash_attn_func(q, k, v)

# Profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        _ = flash_attn_func(q, k, v)

# 输出报告
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 保存trace文件
prof.export_chrome_trace("flash_attn_trace.json")
# 使用 chrome://tracing 打开
```

运行:
```bash
python profile_benchmark.py
```

### 5. 使用 nvidia-smi 实时监控

在另一个终端运行：
```bash
# 实时监控GPU使用情况
watch -n 0.5 nvidia-smi

# 或者使用更详细的监控
nvidia-smi dmon -i 0 -s pucvmet -d 1
```

### 6. 使用 nvtop 可视化监控

```bash
# 安装 nvtop (如果没有)
# Ubuntu: apt install nvtop
# 其他: https://github.com/Syllo/nvtop

nvtop
```

## 🔍 性能指标解读

### TFLOPs/s (万亿次浮点运算/秒)
- **A100 (40GB/80GB)**: 理论峰值 ~312 TFLOPs (FP16/BF16)
- **实际性能**: Flash Attention 通常能达到 150-250 TFLOPs
- **更高数值 = 更好的性能**

### Kernel执行时间
- 关注 **最小时间** (min time) - 代表最佳情况
- **平均时间** (avg time) - 代表典型性能
- 如果 max 和 min 差异很大，可能有性能抖动

### GPU利用率
- **目标**: >90% GPU利用率
- **<80%**: 可能有性能瓶颈（内存带宽、kernel launch overhead等）

### 内存带宽利用率
- **A100**: 理论峰值 ~2TB/s (HBM2e)
- **目标**: >80% 内存带宽利用率
- Flash Attention 的优势就在于降低内存访问量

## 🎯 性能优化检查清单

### 基础检查
- [ ] GPU工作频率是否达到最大 (nvidia-smi 查看)
- [ ] 是否有其他进程占用GPU
- [ ] CUDA版本是否最新 (建议 >=12.0)
- [ ] PyTorch版本是否最新 (建议 >=2.2)

### 配置优化
- [ ] Batch size是否充分利用GPU
- [ ] 使用FP16/BF16而非FP32
- [ ] 对于decoder使用causal mask
- [ ] 考虑序列长度对性能的影响

### 高级优化
- [ ] 检查是否使用了CUDA Graph (减少kernel launch开销)
- [ ] 查看kernel fusion情况
- [ ] 分析shared memory使用
- [ ] 检查Tensor Core利用率

## 📈 与标准PyTorch Attention对比

创建对比脚本 `compare_attention.py`:
```python
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
import time

def pytorch_attention(q, k, v, causal=False):
    """标准PyTorch attention"""
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(q.shape[1], k.shape[1]), diagonal=1).bool()
        attn = attn.masked_fill(mask.to(q.device), float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

batch, seqlen, nheads, headdim = 4, 2048, 8, 64
q = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)

# Benchmark Flash Attention
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = flash_attn_func(q, k, v)
torch.cuda.synchronize()
flash_time = (time.time() - start) / 100

# Benchmark PyTorch Attention
q_pt = q.transpose(1, 2)  # PyTorch expects [B, H, L, D]
k_pt = k.transpose(1, 2)
v_pt = v.transpose(1, 2)
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = pytorch_attention(q_pt, k_pt, v_pt)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100

print(f"Flash Attention: {flash_time*1000:.2f} ms")
print(f"PyTorch Attention: {pytorch_time*1000:.2f} ms")
print(f"Speedup: {pytorch_time/flash_time:.2f}x")
print(f"Memory (Flash): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

## 📚 参考资源

### 论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://tridao.me/publications/flash2/flash2.pdf)

### 工具文档
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Cutlass文档
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs)

## 🐛 常见问题

### Q: 编译失败 "ninja: build stopped: subcommand failed"
**A**: 
```bash
# 清理缓存后重试
pip uninstall flash-attn -y
rm -rf build/
MAX_JOBS=4 python setup.py install
```

### Q: CUDA out of memory
**A**: 减小batch size或sequence length:
```bash
python minimal_benchmark.py --batch 1 --seqlen 1024
```

### Q: 性能低于预期
**A**: 
1. 检查GPU频率: `nvidia-smi -q -d CLOCK`
2. 设置GPU为performance模式: `sudo nvidia-smi -pm 1`
3. 确保使用FP16/BF16而非FP32

### Q: nsys/ncu命令不存在
**A**: 需要安装CUDA Toolkit (不只是runtime):
```bash
# 检查是否安装
which nsys
which ncu

# 如果没有，安装CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads
```

## 💡 高级技巧

### 1. 批量测试多个配置
```bash
#!/bin/bash
for seqlen in 512 1024 2048 4096; do
    echo "Testing seqlen=$seqlen"
    python minimal_benchmark.py --seqlen $seqlen --no-grad
done
```

### 2. 导出结果到CSV
修改 `minimal_benchmark.py`，添加：
```python
import csv
with open('results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([batch, seqlen, nheads, headdim, avg_time, tflops_avg])
```

### 3. 自动化性能回归测试
```bash
# baseline
python minimal_benchmark.py > baseline.txt

# 修改代码后
python minimal_benchmark.py > current.txt

# 对比
diff baseline.txt current.txt
```

---

**祝你Benchmark顺利！如有问题，欢迎查看项目Issues或提问。** 🚀

