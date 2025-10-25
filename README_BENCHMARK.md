# Flash Attention Benchmark 工具集

> 完整的性能测试和分析工具，针对 `csrc/flash_attn/src/flash_fwd_kernel.h` 的A100实现

## 📦 工具概览

本工具集提供了一套完整的benchmark和profiling方案：

| 工具 | 用途 | 难度 |
|-----|------|------|
| `minimal_benchmark.py` | 快速性能测试 | ⭐️ 简单 |
| `compare_implementations.py` | 对比不同实现 | ⭐️ 简单 |
| `auto_profile.sh` | 一键全面分析 | ⭐️⭐️ 中等 |
| `QUICK_START_BENCHMARK.md` | 快速入门指南 | 📖 文档 |
| `BENCHMARK_GUIDE.md` | 详细使用手册 | 📖 文档 |

## 🚀 5分钟快速开始

### 步骤1: 安装
```bash
cd /Users/michu/Documents/flash-attention
pip install ninja packaging einops
python setup.py install
```

### 步骤2: 基础测试
```bash
# 运行最简单的benchmark
python minimal_benchmark.py
```

预期输出:
```
================================================================================
Flash Attention Benchmark
================================================================================
GPU: NVIDIA A100-SXM4-40GB
Compute Capability: 8.0
...
Forward Pass Results:
  Time (min):      2.345 ms
  TFLOPs/s (peak): 156.32
✅ Benchmark completed!
```

### 步骤3: 对比测试 (可选)
```bash
# 对比Flash Attention vs PyTorch
python compare_implementations.py
```

预期输出:
```
Implementation            Time (ms)       Speedup
--------------------------------------------------------------------------------
Flash Attention              2.345         1.00x
PyTorch Standard            15.678         0.15x
🏆 Fastest: Flash Attention (2.345 ms)
```

## 📋 详细工具说明

### 1. minimal_benchmark.py - 核心Benchmark工具

**功能**:
- ✅ 测量Forward/Backward执行时间
- ✅ 计算TFLOPs/s吞吐量
- ✅ 监控GPU内存使用
- ✅ 支持多种配置参数

**基础用法**:
```bash
# 默认配置
python minimal_benchmark.py

# 自定义所有参数
python minimal_benchmark.py \
    --batch 4 \
    --seqlen 2048 \
    --nheads 16 \
    --headdim 128 \
    --causal \
    --dtype fp16 \
    --repeats 30
```

**参数说明**:
- `--batch`: Batch size (默认: 2)
- `--seqlen`: 序列长度 (默认: 1024)
- `--nheads`: Head数量 (默认: 8)
- `--headdim`: Head维度 (默认: 64)
- `--causal`: 使用causal masking (用于decoder)
- `--dtype`: 数据类型 fp16/bf16 (默认: fp16)
- `--repeats`: 重复次数 (默认: 30)
- `--no-grad`: 只测forward pass (用于profiling)

**常见场景**:
```bash
# GPT-2 配置
python minimal_benchmark.py --batch 8 --seqlen 1024 --nheads 12 --headdim 64 --causal

# BERT配置
python minimal_benchmark.py --batch 16 --seqlen 512 --nheads 12 --headdim 64

# 长序列测试
python minimal_benchmark.py --batch 1 --seqlen 8192 --nheads 8 --headdim 64
```

**输出指标**:
- 执行时间 (min/avg/max)
- 吞吐量 (TFLOPs/s)
- 内存占用 (GB)

---

### 2. compare_implementations.py - 实现对比工具

**功能**:
- ✅ 对比Flash Attention vs PyTorch
- ✅ 对比Flash Attention vs xFormers
- ✅ 对比Flash Attention vs PyTorch SDPA
- ✅ 显示加速比和内存节省

**用法**:
```bash
# 基础对比
python compare_implementations.py

# 自定义配置
python compare_implementations.py \
    --batch 4 \
    --seqlen 2048 \
    --nheads 16 \
    --headdim 128 \
    --causal
```

**输出示例**:
```
================================================================================
Attention Implementation Comparison
================================================================================
GPU: NVIDIA A100-SXM4-40GB
Config: batch=4, seqlen=2048, nheads=16, headdim=128, causal=False
================================================================================

Results:
================================================================================
Implementation            Time (ms)       Memory (MB)     Speedup
--------------------------------------------------------------------------------
Flash Attention              4.567          1024.5         1.00x
PyTorch Standard            23.456          4096.8         0.19x
PyTorch SDPA                 6.789          1536.2         0.67x
xFormers                     5.234          1152.3         0.87x
================================================================================

🏆 Fastest: Flash Attention (4.567 ms)
💾 Lowest Memory: Flash Attention (1024.5 MB)
```

**关键指标**:
- **Speedup**: Flash Attention相对于其他实现的加速比
- **Memory**: 内存使用对比
- **Time**: 绝对执行时间

---

### 3. auto_profile.sh - 自动化Profiling脚本

**功能**:
- ✅ 一键运行所有profiling工具
- ✅ 生成Nsight Systems报告
- ✅ 生成Nsight Compute报告
- ✅ 收集GPU信息
- ✅ 运行实现对比
- ✅ 生成汇总报告

**用法**:
```bash
# 使用默认配置
./auto_profile.sh

# 自定义配置
BATCH=4 SEQLEN=2048 NHEADS=16 ./auto_profile.sh

# 指定输出目录
OUTPUT_DIR=my_results ./auto_profile.sh
```

**环境变量**:
- `BATCH`: Batch size (默认: 2)
- `SEQLEN`: 序列长度 (默认: 1024)
- `NHEADS`: Head数量 (默认: 8)
- `HEADDIM`: Head维度 (默认: 64)
- `REPEATS`: 重复次数 (默认: 10)
- `OUTPUT_DIR`: 输出目录 (默认: profile_results)

**生成的文件**:
```
profile_results/
├── 📄 SUMMARY.md                      ← 从这里开始！
├── baseline_benchmark.txt            # 基础性能数据
├── comparison_results.txt            # 实现对比
├── gpu_info.txt                      # GPU信息
├── gpu_info_detailed.txt             # 详细GPU信息
│
├── 🔍 Nsight Systems (系统级分析)
│   ├── flash_attn_nsys.nsys-rep     # GUI打开 ← 推荐
│   ├── nsys_stats.txt                # 文本统计
│   └── nsys_output.txt               # 控制台输出
│
└── 🔍 Nsight Compute (Kernel级分析)
    ├── flash_attn_ncu_basic.ncu-rep  # 快速分析
    ├── flash_attn_ncu_full.ncu-rep   # 详细分析 ← 推荐
    ├── ncu_memory_metrics.txt        # 内存带宽
    └── ncu_compute_metrics.txt       # 计算利用率
```

**查看结果**:
```bash
# 1. 查看汇总
cat profile_results/SUMMARY.md

# 2. 使用GUI查看 (更直观)
# - 下载 .nsys-rep 文件，用 Nsight Systems 打开
# - 下载 .ncu-rep 文件，用 Nsight Compute 打开
```

---

## 🔍 性能分析方法

### 方法1: 快速测试 (1分钟)
```bash
python minimal_benchmark.py
```
**适用于**: 快速验证性能、对比不同配置

### 方法2: 对比测试 (2分钟)
```bash
python compare_implementations.py
```
**适用于**: 验证Flash Attention的加速效果

### 方法3: 完整Profiling (10-30分钟)
```bash
./auto_profile.sh
```
**适用于**: 深入性能分析、找出瓶颈、优化kernel

### 方法4: 手动Profiling (高级)

**Nsight Systems** (系统级):
```bash
nsys profile -o profile --stats=true \
    python minimal_benchmark.py --no-grad
```

**Nsight Compute** (Kernel级):
```bash
ncu --set full -o kernel_profile \
    --kernel-name regex:"flash_fwd" \
    python minimal_benchmark.py --no-grad
```

**实时监控**:
```bash
# 终端1: 运行benchmark
python minimal_benchmark.py --repeats 1000

# 终端2: 监控GPU
watch -n 0.5 nvidia-smi
# 或
nvtop
```

---

## 📊 性能指标解读

### TFLOPs/s (吞吐量)

| GPU | 理论峰值 (FP16) | Flash Attn 典型值 | 目标利用率 |
|-----|----------------|-------------------|-----------|
| A100 | ~312 TFLOPs | 150-250 TFLOPs | >50% |
| A6000 | ~154 TFLOPs | 80-130 TFLOPs | >50% |
| RTX 4090 | ~330 TFLOPs | 150-280 TFLOPs | >50% |
| RTX 3090 | ~142 TFLOPs | 70-120 TFLOPs | >50% |

**解读**:
- ✅ **好**: TFLOPs/s > 理论峰值的50%
- ⚠️ **一般**: 30-50%
- ❌ **差**: <30% (需要优化)

### 执行时间参考 (A100, FP16)

| Batch | SeqLen | Heads | HeadDim | 典型时间 |
|-------|--------|-------|---------|---------|
| 1 | 512 | 8 | 64 | ~0.5 ms |
| 2 | 1024 | 8 | 64 | ~2 ms |
| 4 | 2048 | 8 | 64 | ~8 ms |
| 8 | 4096 | 8 | 64 | ~30 ms |
| 1 | 8192 | 8 | 64 | ~15 ms |
| 1 | 16384 | 8 | 64 | ~60 ms |

### 加速比

| 对比实现 | 典型加速比 | 备注 |
|---------|-----------|------|
| PyTorch Standard | 3-8x | 序列越长加速越明显 |
| PyTorch SDPA | 1.2-2x | PyTorch 2.0+ 的优化实现 |
| xFormers | 0.9-1.5x | 另一个高效实现 |

### GPU利用率

**查看方法**:
```bash
nvidia-smi dmon -i 0 -s u -d 1
```

**目标**:
- ✅ **好**: >90%
- ⚠️ **一般**: 70-90%
- ❌ **差**: <70% (有瓶颈)

---

## 🎯 使用场景

### 场景1: 开发调试
```bash
# 快速验证修改是否正确
python minimal_benchmark.py --seqlen 512 --repeats 5
```

### 场景2: 性能回归测试
```bash
# 保存baseline
python minimal_benchmark.py > baseline.txt

# 修改代码后
python minimal_benchmark.py > current.txt

# 对比
diff baseline.txt current.txt
```

### 场景3: 寻找最佳配置
```bash
#!/bin/bash
for bs in 1 2 4 8; do
  for sl in 512 1024 2048; do
    echo "batch=$bs, seqlen=$sl"
    python minimal_benchmark.py --batch $bs --seqlen $sl --no-grad
  done
done
```

### 场景4: 深入性能分析
```bash
# 使用自动化脚本
./auto_profile.sh

# 查看报告
cat profile_results/SUMMARY.md

# 用GUI分析kernel
# 打开 profile_results/flash_attn_ncu_full.ncu-rep
```

### 场景5: 论文/报告
```bash
# 对比多个实现
python compare_implementations.py --batch 4 --seqlen 2048 > paper_results.txt

# 多个配置测试
for seqlen in 512 1024 2048 4096 8192; do
    python minimal_benchmark.py --seqlen $seqlen
done | tee scaling_results.txt
```

---

## 🐛 常见问题

### Q1: ImportError: cannot import flash_attn
```bash
# 重新安装
pip uninstall flash-attn -y
python setup.py install
```

### Q2: CUDA out of memory
```bash
# 减小配置
python minimal_benchmark.py --batch 1 --seqlen 1024
```

### Q3: 编译时间太长
```bash
# 限制并行编译数
MAX_JOBS=4 python setup.py install
```

### Q4: 找不到nsys/ncu命令
```bash
# 需要安装完整CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# 或添加到PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Q5: 性能远低于预期
```bash
# 检查GPU状态
nvidia-smi

# 设置性能模式
sudo nvidia-smi -pm 1

# 检查频率
nvidia-smi -q -d CLOCK | grep Graphics

# 检查后台进程
ps aux | grep python
```

---

## 📚 学习资源

### 必读文档
1. **快速入门**: `QUICK_START_BENCHMARK.md` ← 从这里开始
2. **详细指南**: `BENCHMARK_GUIDE.md`
3. **本文档**: `README_BENCHMARK.md`

### Flash Attention论文
- [Flash Attention v1](https://arxiv.org/abs/2205.14135) - 原始论文
- [Flash Attention v2](https://tridao.me/publications/flash2/flash2.pdf) - 优化版本
- [Flash Attention v3](https://tridao.me/publications/flash3/flash3.pdf) - Hopper架构

### 核心实现文件
- `csrc/flash_attn/src/flash_fwd_kernel.h` - **A100 forward kernel**
- `csrc/flash_attn/src/flash_bwd_kernel.h` - **A100 backward kernel**
- `hopper/flash_fwd_kernel_sm90.h` - H100优化版本
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` - Hopper主循环

### Cutlass资源
- [Cutlass GitHub](https://github.com/NVIDIA/cutlass)
- [Cutlass 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [Cutlass Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

### NVIDIA工具
- [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## 💡 最佳实践

### Benchmark最佳实践
1. ✅ **预热充分**: warmup至少5次
2. ✅ **重复多次**: repeats至少30次
3. ✅ **控制变量**: 每次只改变一个参数
4. ✅ **记录环境**: GPU型号、CUDA版本、PyTorch版本
5. ✅ **对比baseline**: 总是和已知结果对比

### Profiling最佳实践
1. ✅ **用--no-grad**: profiling时只测forward pass
2. ✅ **减少repeats**: profiling时用少量重复
3. ✅ **先用nsys**: 系统级分析找大方向
4. ✅ **再用ncu**: kernel级分析找细节
5. ✅ **用GUI查看**: 比文本报告更直观

### 性能优化检查清单
- [ ] GPU处于最高性能模式
- [ ] 没有其他进程占用GPU
- [ ] 使用FP16/BF16而非FP32
- [ ] Batch size充分利用GPU
- [ ] CUDA和PyTorch版本最新
- [ ] GPU温度正常 (<85°C)

---

## 🤝 贡献

发现问题或有改进建议？

1. 提交Issue到 [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention/issues)
2. 或者修改这些benchmark脚本并分享

---

## 📄 License

这些benchmark工具遵循Flash Attention的LICENSE (BSD-3-Clause)

---

## 🎉 总结

你现在拥有了一套完整的Flash Attention benchmark工具！

**最简单的开始方式**:
```bash
# 1. 安装
python setup.py install

# 2. 测试
python minimal_benchmark.py

# 3. 对比
python compare_implementations.py

# 4. 深入分析 (可选)
./auto_profile.sh
```

**下一步**:
- 📖 阅读 `QUICK_START_BENCHMARK.md` 了解详细用法
- 🔬 运行 `auto_profile.sh` 获得完整性能报告
- 📊 阅读 `BENCHMARK_GUIDE.md` 学习高级技巧

---

**祝你Benchmark顺利！** 🚀

有问题随时查看文档或提Issue！

