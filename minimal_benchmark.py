#!/usr/bin/env python3
"""
最小化的Flash Attention Benchmark脚本
用于测试 csrc/flash_attn/src/flash_fwd_kernel.h 的A100实现

使用方法:
    python minimal_benchmark.py --batch 2 --seqlen 1024 --nheads 8 --headdim 64
    
性能监测:
    # 使用 nsys 分析
    nsys profile -o flash_attn_profile python minimal_benchmark.py
    
    # 使用 ncu 详细分析kernel
    ncu --set full -o flash_attn_ncu python minimal_benchmark.py --no-grad
"""

import argparse
import torch
import math
from flash_attn import flash_attn_func

def flops(batch, seqlen, headdim, nheads, causal=False, mode="fwd"):
    """计算理论FLOPs"""
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def benchmark_forward(func, *args, repeats=30, warmup=5, **kwargs):
    """Benchmark前向传播"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # 使用CUDA events计时
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    
    for i in range(repeats):
        start_events[i].record()
        _ = func(*args, **kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return min(times), sum(times) / len(times), max(times)

def benchmark_backward(func, *args, repeats=30, warmup=5, **kwargs):
    """Benchmark前向+反向传播"""
    # Warmup
    for _ in range(warmup):
        out = func(*args, **kwargs)
        dout = torch.randn_like(out)
        out.backward(dout)
    
    torch.cuda.synchronize()
    
    # 使用CUDA events计时
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    
    for i in range(repeats):
        # 需要每次重新创建tensor以清除梯度
        q = torch.randn_like(args[0], requires_grad=True)
        k = torch.randn_like(args[1], requires_grad=True)
        v = torch.randn_like(args[2], requires_grad=True)
        
        start_events[i].record()
        out = func(q, k, v, **kwargs)
        dout = torch.randn_like(out)
        out.backward(dout)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return min(times), sum(times) / len(times), max(times)

def main():
    parser = argparse.ArgumentParser(description='Flash Attention Minimal Benchmark')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length')
    parser.add_argument('--nheads', type=int, default=8, help='Number of heads')
    parser.add_argument('--headdim', type=int, default=64, help='Head dimension')
    parser.add_argument('--causal', action='store_true', help='Use causal masking')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'], 
                        help='Data type')
    parser.add_argument('--repeats', type=int, default=30, help='Number of repeats')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--no-grad', action='store_true', help='Only run forward pass')
    
    args = parser.parse_args()
    
    # 设置dtype
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    
    print("=" * 80)
    print("Flash Attention Benchmark")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  Batch size:    {args.batch}")
    print(f"  Sequence len:  {args.seqlen}")
    print(f"  Num heads:     {args.nheads}")
    print(f"  Head dim:      {args.headdim}")
    print(f"  Causal:        {args.causal}")
    print(f"  Data type:     {args.dtype}")
    print(f"  Repeats:       {args.repeats}")
    print("=" * 80)
    
    # 创建输入tensors
    q = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, 
                    device=device, dtype=dtype, requires_grad=not args.no_grad)
    k = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, 
                    device=device, dtype=dtype, requires_grad=not args.no_grad)
    v = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim, 
                    device=device, dtype=dtype, requires_grad=not args.no_grad)
    
    # 计算理论FLOPs
    total_flops_fwd = flops(args.batch, args.seqlen, args.headdim, args.nheads, 
                            args.causal, mode="fwd")
    total_flops_bwd = flops(args.batch, args.seqlen, args.headdim, args.nheads, 
                            args.causal, mode="bwd")
    total_flops_fwd_bwd = flops(args.batch, args.seqlen, args.headdim, args.nheads, 
                                args.causal, mode="fwd_bwd")
    
    # Benchmark Forward Pass
    print("\n📊 Benchmarking Forward Pass...")
    min_time, avg_time, max_time = benchmark_forward(
        flash_attn_func, q, k, v, 
        causal=args.causal,
        repeats=args.repeats,
        warmup=args.warmup
    )
    
    # 转换为秒
    min_time_s = min_time / 1000.0
    avg_time_s = avg_time / 1000.0
    max_time_s = max_time / 1000.0
    
    # 计算TFLOPs/s
    tflops_min = (total_flops_fwd / min_time_s) / 1e12
    tflops_avg = (total_flops_fwd / avg_time_s) / 1e12
    tflops_max = (total_flops_fwd / max_time_s) / 1e12
    
    print(f"\nForward Pass Results:")
    print(f"  Time (min):      {min_time:.3f} ms")
    print(f"  Time (avg):      {avg_time:.3f} ms")
    print(f"  Time (max):      {max_time:.3f} ms")
    print(f"  TFLOPs/s (peak): {tflops_min:.2f}")
    print(f"  TFLOPs/s (avg):  {tflops_avg:.2f}")
    
    # Benchmark Forward + Backward Pass
    if not args.no_grad:
        print("\n📊 Benchmarking Forward + Backward Pass...")
        min_time, avg_time, max_time = benchmark_backward(
            flash_attn_func, q, k, v,
            causal=args.causal,
            repeats=args.repeats,
            warmup=args.warmup
        )
        
        # 转换为秒
        min_time_s = min_time / 1000.0
        avg_time_s = avg_time / 1000.0
        max_time_s = max_time / 1000.0
        
        # 计算TFLOPs/s
        tflops_min = (total_flops_fwd_bwd / min_time_s) / 1e12
        tflops_avg = (total_flops_fwd_bwd / avg_time_s) / 1e12
        tflops_max = (total_flops_fwd_bwd / max_time_s) / 1e12
        
        print(f"\nForward + Backward Pass Results:")
        print(f"  Time (min):      {min_time:.3f} ms")
        print(f"  Time (avg):      {avg_time:.3f} ms")
        print(f"  Time (max):      {max_time:.3f} ms")
        print(f"  TFLOPs/s (peak): {tflops_min:.2f}")
        print(f"  TFLOPs/s (avg):  {tflops_avg:.2f}")
    
    # 内存使用
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n💾 Memory Usage:")
    print(f"  Allocated: {memory_allocated:.2f} GB")
    print(f"  Reserved:  {memory_reserved:.2f} GB")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()

