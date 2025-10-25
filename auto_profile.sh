#!/bin/bash
# 自动化性能分析脚本
# 运行多种profiling工具并生成报告

set -e  # 遇到错误立即退出

# 配置参数（可以根据需要修改）
BATCH=${BATCH:-2}
SEQLEN=${SEQLEN:-1024}
NHEADS=${NHEADS:-8}
HEADDIM=${HEADDIM:-64}
REPEATS=${REPEATS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-"profile_results"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Flash Attention Auto Profiling Script"
echo "=============================================="
echo "Configuration:"
echo "  Batch size:    $BATCH"
echo "  Sequence len:  $SEQLEN"
echo "  Num heads:     $NHEADS"
echo "  Head dim:      $HEADDIM"
echo "  Repeats:       $REPEATS"
echo "  Output dir:    $OUTPUT_DIR"
echo "=============================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# 检查必要的工具
check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

echo "Checking available tools..."
HAS_NSYS=0
HAS_NCU=0
HAS_NVIDIA_SMI=0

check_tool nsys && HAS_NSYS=1 || echo "  → Install CUDA Toolkit for nsys"
check_tool ncu && HAS_NCU=1 || echo "  → Install CUDA Toolkit for ncu"
check_tool nvidia-smi && HAS_NVIDIA_SMI=1 || echo "  → Install NVIDIA drivers"

echo ""

# 基准测试参数
BENCHMARK_CMD="python ../minimal_benchmark.py \
    --batch $BATCH \
    --seqlen $SEQLEN \
    --nheads $NHEADS \
    --headdim $HEADDIM \
    --no-grad \
    --repeats $REPEATS"

# 1. 基础benchmark
echo "=============================================="
echo "1. Running baseline benchmark..."
echo "=============================================="
eval $BENCHMARK_CMD | tee baseline_benchmark.txt
echo ""

# 2. GPU信息收集
if [ $HAS_NVIDIA_SMI -eq 1 ]; then
    echo "=============================================="
    echo "2. Collecting GPU information..."
    echo "=============================================="
    nvidia-smi > gpu_info.txt
    nvidia-smi -q > gpu_info_detailed.txt
    echo -e "${GREEN}✓${NC} GPU info saved to gpu_info*.txt"
    echo ""
fi

# 3. Nsight Systems profiling
if [ $HAS_NSYS -eq 1 ]; then
    echo "=============================================="
    echo "3. Running Nsight Systems profiling..."
    echo "=============================================="
    echo "This may take a few minutes..."
    
    nsys profile \
        -o flash_attn_nsys \
        --stats=true \
        --force-overwrite true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        $BENCHMARK_CMD > nsys_output.txt 2>&1
    
    # 生成统计报告
    if [ -f "flash_attn_nsys.nsys-rep" ]; then
        nsys stats flash_attn_nsys.nsys-rep > nsys_stats.txt 2>&1
        echo -e "${GREEN}✓${NC} Nsight Systems report saved"
        echo "  - flash_attn_nsys.nsys-rep (open with Nsight Systems GUI)"
        echo "  - nsys_stats.txt (text report)"
    else
        echo -e "${RED}✗${NC} Nsight Systems profiling failed"
    fi
    echo ""
fi

# 4. Nsight Compute profiling
if [ $HAS_NCU -eq 1 ]; then
    echo "=============================================="
    echo "4. Running Nsight Compute profiling..."
    echo "=============================================="
    echo "This will take several minutes..."
    echo "Analyzing kernel: flash_fwd"
    
    # 基础分析
    ncu \
        -o flash_attn_ncu_basic \
        --set basic \
        --kernel-name regex:"flash_fwd" \
        --force-overwrite \
        $BENCHMARK_CMD > ncu_basic_output.txt 2>&1 || true
    
    echo "Running detailed analysis (this is slow)..."
    # 详细分析 (可能很慢)
    ncu \
        -o flash_attn_ncu_full \
        --set full \
        --kernel-name regex:"flash_fwd" \
        --force-overwrite \
        --target-processes all \
        --replay-mode kernel \
        --kernel-id ::regex:flash:1 \
        $BENCHMARK_CMD > ncu_full_output.txt 2>&1 || true
    
    # 内存分析
    echo "Analyzing memory bandwidth..."
    ncu \
        --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum \
        --kernel-name regex:"flash_fwd" \
        $BENCHMARK_CMD > ncu_memory_metrics.txt 2>&1 || true
    
    # 计算分析
    echo "Analyzing compute utilization..."
    ncu \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
        --kernel-name regex:"flash_fwd" \
        $BENCHMARK_CMD > ncu_compute_metrics.txt 2>&1 || true
    
    echo -e "${GREEN}✓${NC} Nsight Compute reports saved"
    echo "  - flash_attn_ncu_*.ncu-rep (open with Nsight Compute GUI)"
    echo "  - ncu_*_metrics.txt (key metrics)"
    echo ""
fi

# 5. 对比测试
echo "=============================================="
echo "5. Running implementation comparison..."
echo "=============================================="
if python -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    python ../compare_implementations.py \
        --batch $BATCH \
        --seqlen $SEQLEN \
        --nheads $NHEADS \
        --headdim $HEADDIM \
        --repeats $REPEATS | tee comparison_results.txt
    echo -e "${GREEN}✓${NC} Comparison results saved to comparison_results.txt"
else
    echo -e "${YELLOW}⚠${NC} Flash Attention not installed, skipping comparison"
fi
echo ""

# 6. 生成总结报告
echo "=============================================="
echo "6. Generating summary report..."
echo "=============================================="

cat > SUMMARY.md << EOF
# Flash Attention Profiling Summary

**Date**: $(date)
**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
**CUDA**: $(nvcc --version 2>/dev/null | grep "release" || echo "Unknown")

## Configuration
- Batch size: $BATCH
- Sequence length: $SEQLEN
- Number of heads: $NHEADS
- Head dimension: $HEADDIM
- Repeats: $REPEATS

## Generated Files

### 1. Baseline Benchmark
- \`baseline_benchmark.txt\` - Basic performance metrics

### 2. GPU Information
- \`gpu_info.txt\` - GPU summary
- \`gpu_info_detailed.txt\` - Detailed GPU specs

### 3. Nsight Systems (System-level)
EOF

if [ $HAS_NSYS -eq 1 ] && [ -f "flash_attn_nsys.nsys-rep" ]; then
    cat >> SUMMARY.md << EOF
- \`flash_attn_nsys.nsys-rep\` - **Open with Nsight Systems GUI for visual timeline**
- \`nsys_stats.txt\` - Statistical summary
- \`nsys_output.txt\` - Console output

**Key Metrics from Nsight Systems:**
\`\`\`
$(grep -A 5 "CUDA Kernel Statistics" nsys_stats.txt 2>/dev/null || echo "See nsys_stats.txt for details")
\`\`\`
EOF
else
    echo "- Not available (nsys not found or profiling failed)" >> SUMMARY.md
fi

cat >> SUMMARY.md << EOF

### 4. Nsight Compute (Kernel-level)
EOF

if [ $HAS_NCU -eq 1 ] && [ -f "flash_attn_ncu_basic.ncu-rep" ]; then
    cat >> SUMMARY.md << EOF
- \`flash_attn_ncu_basic.ncu-rep\` - Basic kernel analysis (fast)
- \`flash_attn_ncu_full.ncu-rep\` - **Detailed kernel analysis** (slower but comprehensive)
- \`ncu_memory_metrics.txt\` - Memory bandwidth analysis
- \`ncu_compute_metrics.txt\` - Compute utilization analysis

**Key Metrics:**

Memory Bandwidth:
\`\`\`
$(cat ncu_memory_metrics.txt 2>/dev/null || echo "See ncu_memory_metrics.txt")
\`\`\`

Compute Utilization:
\`\`\`
$(cat ncu_compute_metrics.txt 2>/dev/null || echo "See ncu_compute_metrics.txt")
\`\`\`
EOF
else
    echo "- Not available (ncu not found or profiling failed)" >> SUMMARY.md
fi

cat >> SUMMARY.md << EOF

### 5. Implementation Comparison
\`\`\`
$(cat comparison_results.txt 2>/dev/null || echo "Not available")
\`\`\`

## How to View Results

### Nsight Systems (Recommended for overall analysis)
1. Download \`flash_attn_nsys.nsys-rep\`
2. Install [Nsight Systems](https://developer.nvidia.com/nsight-systems)
3. Open the file in Nsight Systems GUI
4. Look for:
   - Kernel execution timeline
   - Memory transfers
   - GPU utilization
   - Bottlenecks

### Nsight Compute (Recommended for kernel optimization)
1. Download \`flash_attn_ncu_full.ncu-rep\`
2. Install [Nsight Compute](https://developer.nvidia.com/nsight-compute)
3. Open the file in Nsight Compute GUI
4. Look for:
   - SM efficiency
   - Memory bandwidth utilization
   - Warp execution efficiency
   - Instruction mix
   - Occupancy

## Quick Analysis

### Performance Summary
\`\`\`
$(tail -20 baseline_benchmark.txt 2>/dev/null)
\`\`\`

### GPU Status During Run
\`\`\`
$(cat gpu_info.txt 2>/dev/null)
\`\`\`

---
Generated by auto_profile.sh
EOF

echo -e "${GREEN}✓${NC} Summary report saved to SUMMARY.md"
echo ""

# 完成
echo "=============================================="
echo "Profiling Complete!"
echo "=============================================="
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Quick view:"
echo "  cat $OUTPUT_DIR/SUMMARY.md"
echo ""
echo "To analyze further:"
echo "  - Open *.nsys-rep with Nsight Systems GUI"
echo "  - Open *.ncu-rep with Nsight Compute GUI"
echo ""
echo -e "${GREEN}✓${NC} All done!"
echo "=============================================="

