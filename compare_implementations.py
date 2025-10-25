#!/usr/bin/env python3
"""
对比不同Attention实现的性能
- Flash Attention (CUDA)
- PyTorch标准实现
- xFormers (如果安装)
"""

import torch
import torch.nn.functional as F
import time
import argparse
from typing import Optional

def pytorch_attention(q, k, v, causal=False):
    """标准PyTorch scaled dot-product attention"""
    # q, k, v: [B, H, L, D]
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        seqlen = q.shape[2]
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out

def benchmark_impl(name, func, *args, repeats=30, warmup=5, **kwargs):
    """Benchmark一个实现"""
    print(f"  Benchmarking {name}...", end=' ', flush=True)
    
    try:
        # Warmup
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(repeats):
            _ = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / repeats * 1000  # ms
        
        # 测量内存
        torch.cuda.reset_peak_memory_stats()
        _ = func(*args, **kwargs)
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"✓")
        return elapsed, memory_mb, None
        
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")
        return None, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--headdim', type=int, default=64)
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    device = 'cuda'
    
    print("=" * 80)
    print("Attention Implementation Comparison")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: batch={args.batch}, seqlen={args.seqlen}, "
          f"nheads={args.nheads}, headdim={args.headdim}, causal={args.causal}")
    print("=" * 80)
    
    results = {}
    
    # 准备输入
    q_flash = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim,
                          device=device, dtype=dtype)
    k_flash = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim,
                          device=device, dtype=dtype)
    v_flash = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim,
                          device=device, dtype=dtype)
    
    # PyTorch格式 [B, H, L, D]
    q_pt = q_flash.transpose(1, 2).contiguous()
    k_pt = k_flash.transpose(1, 2).contiguous()
    v_pt = v_flash.transpose(1, 2).contiguous()
    
    # 1. Flash Attention
    try:
        from flash_attn import flash_attn_func
        time_ms, mem_mb, error = benchmark_impl(
            "Flash Attention",
            flash_attn_func, q_flash, k_flash, v_flash,
            causal=args.causal, repeats=args.repeats
        )
        results['Flash Attention'] = (time_ms, mem_mb, error)
    except ImportError:
        print("  Flash Attention not installed")
        results['Flash Attention'] = (None, None, "Not installed")
    
    # 2. PyTorch Standard
    time_ms, mem_mb, error = benchmark_impl(
        "PyTorch Standard",
        pytorch_attention, q_pt, k_pt, v_pt,
        causal=args.causal, repeats=args.repeats
    )
    results['PyTorch Standard'] = (time_ms, mem_mb, error)
    
    # 3. PyTorch SDPA (scaled_dot_product_attention)
    if hasattr(F, 'scaled_dot_product_attention'):
        time_ms, mem_mb, error = benchmark_impl(
            "PyTorch SDPA",
            F.scaled_dot_product_attention, q_pt, k_pt, v_pt,
            is_causal=args.causal, repeats=args.repeats
        )
        results['PyTorch SDPA'] = (time_ms, mem_mb, error)
    
    # 4. xFormers
    try:
        import xformers.ops as xops
        # xFormers格式 [B, L, H, D]
        time_ms, mem_mb, error = benchmark_impl(
            "xFormers",
            xops.memory_efficient_attention, q_flash, k_flash, v_flash,
            attn_bias=xops.LowerTriangularMask() if args.causal else None,
            repeats=args.repeats
        )
        results['xFormers'] = (time_ms, mem_mb, error)
    except ImportError:
        results['xFormers'] = (None, None, "Not installed")
    
    # 输出结果表格
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    # 找到最快的实现作为baseline
    baseline_time = None
    for name, (time_ms, mem_mb, error) in results.items():
        if time_ms is not None:
            if baseline_time is None or time_ms < baseline_time:
                baseline_time = time_ms
    
    # 打印结果
    for name, (time_ms, mem_mb, error) in results.items():
        if error is None and time_ms is not None:
            speedup = baseline_time / time_ms if baseline_time else 1.0
            print(f"{name:<25} {time_ms:>12.3f}   {mem_mb:>12.1f}   {speedup:>7.2f}x")
        else:
            reason = error if error else "Failed"
            print(f"{name:<25} {'N/A':>12}   {'N/A':>12}   {reason}")
    
    print("=" * 80)
    
    # 找出最快的
    fastest_name = None
    fastest_time = float('inf')
    for name, (time_ms, mem_mb, error) in results.items():
        if time_ms is not None and time_ms < fastest_time:
            fastest_time = time_ms
            fastest_name = name
    
    if fastest_name:
        print(f"\n🏆 Fastest: {fastest_name} ({fastest_time:.3f} ms)")
    
    # 内存最优
    lowest_mem_name = None
    lowest_mem = float('inf')
    for name, (time_ms, mem_mb, error) in results.items():
        if mem_mb is not None and mem_mb < lowest_mem:
            lowest_mem = mem_mb
            lowest_mem_name = name
    
    if lowest_mem_name:
        print(f"💾 Lowest Memory: {lowest_mem_name} ({lowest_mem:.1f} MB)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

