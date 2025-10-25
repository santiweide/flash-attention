# 快速开始 - Flash Attention Benchmark

> 针对 `csrc/flash_attn/src/flash_fwd_kernel.h` (A100实现) 的最小化benchmark方案

## 🚀 三步快速开始

### 1️⃣ 安装依赖
```bash
# 安装Flash Attention
cd /Users/michu/Documents/flash-attention
pip install ninja packaging einops
python setup.py install
```

### 2️⃣ 运行基础Benchmark
```bash
# 默认配置 (batch=2, seqlen=1024, heads=8, dim=64)
python minimal_benchmark.py

# 自定义配置
python minimal_benchmark.py --batch 4 --seqlen 2048 --nheads 16 --headdim 128
```

### 3️⃣ 对比不同实现
```bash
python compare_implementations.py --batch 4 --seqlen 2048
```

## 📊 工具说明

### 工具1: minimal_benchmark.py
**用途**: 快速测试Flash Attention性能

**执行命令**:
```bash
# 基础测试
python minimal_benchmark.py

# 完整参数示例
python minimal_benchmark.py \
    --batch 4 \
    --seqlen 2048 \
    --nheads 8 \
    --headdim 64 \
    --causal \
    --dtype fp16 \
    --repeats 30
```

**输出指标**:
- ⏱️ **执行时间**: min/avg/max (毫秒)
- 🚀 **吞吐量**: TFLOPs/s (万亿次浮点运算/秒)
- 💾 **内存占用**: GPU显存使用

**示例输出**:
```
Forward Pass Results:
  Time (min):      2.345 ms
  Time (avg):      2.456 ms
  TFLOPs/s (peak): 156.32 TFLOPs/s
  TFLOPs/s (avg):  149.87 TFLOPs/s
```

### 工具2: compare_implementations.py
**用途**: 对比Flash Attention vs PyTorch等实现

**执行命令**:
```bash
python compare_implementations.py --batch 4 --seqlen 2048
```

**输出示例**:
```
Implementation            Time (ms)       Memory (MB)     Speedup
--------------------------------------------------------------------------------
Flash Attention              2.345          512.1         1.00x
PyTorch Standard            15.678         2048.3         0.15x
PyTorch SDPA                 3.456          768.2         0.68x
xFormers                     2.890          624.5         0.81x

🏆 Fastest: Flash Attention (2.345 ms)
💾 Lowest Memory: Flash Attention (512.1 MB)
```

### 工具3: auto_profile.sh
**用途**: 一键运行所有profiling工具

**执行命令**:
```bash
# 使用默认配置
./auto_profile.sh

# 自定义配置
BATCH=4 SEQLEN=2048 NHEADS=16 ./auto_profile.sh
```

**生成的文件**:
```
profile_results/
├── SUMMARY.md                      # 📄 总结报告 (从这里开始看!)
├── baseline_benchmark.txt          # 基础性能数据
├── comparison_results.txt          # 实现对比结果
├── gpu_info.txt                    # GPU信息
├── flash_attn_nsys.nsys-rep       # 🔍 Nsight Systems报告 (GUI打开)
├── flash_attn_ncu_full.ncu-rep    # 🔍 Nsight Compute报告 (GUI打开)
├── nsys_stats.txt                  # Nsight Systems统计
├── ncu_memory_metrics.txt          # 内存带宽分析
└── ncu_compute_metrics.txt         # 计算单元利用率
```

## 🔍 性能分析工具

### A. Nsight Systems (系统级分析)
**查看整体性能、kernel时间线、找瓶颈**

```bash
# 方法1: 使用auto_profile.sh (推荐)
./auto_profile.sh

# 方法2: 手动运行
nsys profile -o flash_profile --stats=true \
    python minimal_benchmark.py --no-grad
```

**如何查看结果**:
1. 下载 `flash_profile.nsys-rep`
2. 在[Nsight Systems GUI](https://developer.nvidia.com/nsight-systems)中打开
3. 查看:
   - CUDA Kernel时间线
   - GPU利用率
   - 内存带宽
   - 找出最慢的kernel

### B. Nsight Compute (Kernel级分析)
**深入分析单个kernel、优化指导**

```bash
# 方法1: 使用auto_profile.sh (推荐)
./auto_profile.sh

# 方法2: 手动运行
ncu --set full -o flash_kernel \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad
```

**如何查看结果**:
1. 下载 `flash_kernel.ncu-rep`
2. 在[Nsight Compute GUI](https://developer.nvidia.com/nsight-compute)中打开
3. 查看:
   - Memory带宽利用率 (目标: >80%)
   - SM效率 (目标: >90%)
   - Tensor Core利用率
   - Warp stall原因
   - 优化建议

### C. 实时监控

**终端1 - 运行benchmark**:
```bash
python minimal_benchmark.py --repeats 1000
```

**终端2 - 实时监控GPU**:
```bash
# 使用nvidia-smi
watch -n 0.5 nvidia-smi

# 或使用nvtop (更好看)
nvtop
```

## 📈 关键性能指标

### 1. TFLOPs/s (吞吐量)
- **A100 理论峰值**: ~312 TFLOPs (FP16)
- **Flash Attention 典型**: 150-250 TFLOPs
- **更高 = 更好**

### 2. 执行时间
| Sequence Length | Batch Size | 典型时间 (A100) |
|----------------|------------|-----------------|
| 512            | 8          | ~1 ms           |
| 1024           | 4          | ~2 ms           |
| 2048           | 2          | ~8 ms           |
| 4096           | 1          | ~30 ms          |

### 3. 内存使用
Flash Attention的优势:
- **标准Attention**: O(N²) 内存
- **Flash Attention**: O(N) 内存
- **长序列下节省明显**

### 4. GPU利用率
- **目标**: >90%
- **查看方法**: `nvidia-smi` 或 Nsight Systems

## 🎯 常见测试场景

### 场景1: GPT模型 (Decoder)
```bash
# GPT-2 Small
python minimal_benchmark.py --batch 8 --seqlen 1024 --nheads 12 --headdim 64 --causal

# GPT-3 Large
python minimal_benchmark.py --batch 2 --seqlen 2048 --nheads 96 --headdim 128 --causal
```

### 场景2: BERT模型 (Encoder)
```bash
# BERT-Base
python minimal_benchmark.py --batch 16 --seqlen 512 --nheads 12 --headdim 64

# BERT-Large
python minimal_benchmark.py --batch 8 --seqlen 512 --nheads 16 --headdim 64
```

### 场景3: 长文本处理
```bash
# 8K context
python minimal_benchmark.py --batch 1 --seqlen 8192 --nheads 8 --headdim 64

# 16K context
python minimal_benchmark.py --batch 1 --seqlen 16384 --nheads 8 --headdim 64
```

### 场景4: 批量推理
```bash
# 高batch size
python minimal_benchmark.py --batch 32 --seqlen 512 --nheads 8 --headdim 64 --no-grad
```

## 🔧 性能优化检查

运行此命令检查系统配置:
```bash
# GPU状态
nvidia-smi

# GPU频率 (应该在最大值)
nvidia-smi -q -d CLOCK | grep "Graphics"

# 设置性能模式
sudo nvidia-smi -pm 1

# 设置最大频率
sudo nvidia-smi -lgc 1410,1410  # A100的值
```

## 📊 结果解读

### 好的性能表现
✅ TFLOPs/s > 150 (A100, FP16)  
✅ GPU利用率 > 90%  
✅ 内存带宽利用率 > 80%  
✅ 相比PyTorch加速 > 3x

### 需要优化的情况
⚠️ TFLOPs/s < 100  
⚠️ GPU利用率 < 70%  
⚠️ 频繁的CUDA OOM错误  
⚠️ 加速比 < 2x

### 优化建议
1. **增加batch size** (如果内存允许)
2. **使用FP16/BF16** 而非FP32
3. **检查GPU频率** 是否被限制
4. **关闭其他GPU进程**
5. **更新CUDA/PyTorch** 到最新版本

## 🐛 故障排查

### 问题1: ImportError: cannot import flash_attn
```bash
# 重新安装
pip uninstall flash-attn -y
python setup.py install
```

### 问题2: CUDA out of memory
```bash
# 减小batch size或sequence length
python minimal_benchmark.py --batch 1 --seqlen 1024
```

### 问题3: 编译失败
```bash
# 限制并行编译数
MAX_JOBS=4 python setup.py install

# 确保ninja已安装
pip install ninja
```

### 问题4: 性能异常低
```bash
# 检查GPU模式
nvidia-smi -q -d PERFORMANCE

# 检查是否有后台进程
nvidia-smi

# 重启GPU
sudo nvidia-smi -r
```

### 问题5: nsys/ncu 命令找不到
```bash
# 需要安装完整CUDA Toolkit (不只是runtime)
# 下载地址: https://developer.nvidia.com/cuda-downloads

# 检查安装
which nsys
which ncu

# 添加到PATH (如果已安装但找不到)
export PATH=/usr/local/cuda/bin:$PATH
```

## 📚 进阶使用

### 批量测试
```bash
#!/bin/bash
for seqlen in 512 1024 2048 4096; do
    echo "Testing seqlen=$seqlen"
    python minimal_benchmark.py --seqlen $seqlen | grep "TFLOPs/s (avg)"
done
```

### 自动化回归测试
```bash
# 保存baseline
python minimal_benchmark.py > baseline.txt

# 修改代码后测试
python minimal_benchmark.py > current.txt

# 对比
diff baseline.txt current.txt
```

### 导出结果
```python
# 在minimal_benchmark.py中添加
import json
results = {
    'time_ms': avg_time,
    'tflops': tflops_avg,
    'memory_gb': memory_allocated
}
with open('results.json', 'w') as f:
    json.dump(results, f)
```

## 🎓 学习资源

### 核心文件
- `csrc/flash_attn/src/flash_fwd_kernel.h` - A100 forward kernel实现
- `csrc/flash_attn/src/flash_bwd_kernel.h` - A100 backward kernel实现  
- `hopper/flash_fwd_kernel_sm90.h` - H100 (Hopper) 优化版本

### 论文
- [FlashAttention v1](https://arxiv.org/abs/2205.14135)
- [FlashAttention v2](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention v3](https://tridao.me/publications/flash3/flash3.pdf)

### Cutlass学习
- [Cutlass GitHub](https://github.com/NVIDIA/cutlass)
- [Cutlass Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

## 💡 最佳实践

1. **从小配置开始**: 先测试小的batch/seqlen，确保正确
2. **使用--no-grad**: 做profiling时只测forward pass
3. **重复多次**: 至少30次以上获得稳定结果
4. **记录环境**: GPU型号、CUDA版本、PyTorch版本
5. **对比baseline**: 总是和PyTorch标准实现对比

## 📞 获取帮助

- **GitHub Issues**: https://github.com/Dao-AILab/flash-attention/issues
- **论文作者**: Tri Dao
- **详细文档**: `BENCHMARK_GUIDE.md`

---

**祝你Benchmark顺利！** 🚀

如有问题，可以查看详细文档: `BENCHMARK_GUIDE.md`

