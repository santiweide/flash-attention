# Flash Attention 最小实现 - 架构说明

## 📐 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Flash Attention                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Q [B,H,S,D] │  │  K [B,H,S,D] │  │  V [B,H,S,D] │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                 │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │  Flash Attention Kernel│                     │
│              │                        │                     │
│              │  - Tiling              │                     │
│              │  - Online Softmax      │                     │
│              │  - Fused Operations    │                     │
│              └────────────┬───────────┘                     │
│                           │                                 │
│                           ▼                                 │
│                  ┌──────────────┐                           │
│                  │  O [B,H,S,D] │                           │
│                  └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 算法流程

### 高层流程

```python
for each Q_block in Q:  # 外循环：遍历Q的blocks
    initialize m = -inf, l = 0, O = 0
    
    for each KV_block in K,V:  # 内循环：遍历K,V的blocks
        # 1. 计算attention scores
        S = Q_block @ K_block^T * scale
        
        # 2. 在线更新softmax统计量
        m_old, l_old = m, l
        m = max(m_old, rowmax(S))
        
        # 3. 计算attention weights
        P = exp(S - m)
        l = exp(m_old - m) * l_old + rowsum(P)
        
        # 4. 更新输出（关键：需要修正之前的累加值）
        correction = exp(m_old - m)
        O = correction * O + P @ V_block / l
    
    write O to global memory
```

### 详细执行流程

```
┌────────────────────────────────────────────────────┐
│  Step 1: 加载Q block到共享内存                      │
│  ┌──────────┐                                      │
│  │ Q_block  │  [kBlockM, kHeadDim]                 │
│  └──────────┘                                      │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  Step 2: 循环处理每个K,V block                      │
│                                                    │
│  for each KV_block:                                │
│  ┌──────────┐  ┌──────────┐                       │
│  │ K_block  │  │ V_block  │                       │
│  └──────────┘  └──────────┘                       │
│                                                    │
│  a) 计算 S = Q_block @ K_block^T                   │
│     ┌─────────────────┐                           │
│     │  S [M x N]      │                           │
│     └─────────────────┘                           │
│                                                    │
│  b) 在线更新softmax                                 │
│     - 更新 m (max)                                 │
│     - 更新 l (sum of exp)                          │
│     - 计算 P = exp(S - m)                          │
│                                                    │
│  c) 累加输出                                        │
│     O = correction * O + P @ V_block               │
│                                                    │
└────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────┐
│  Step 3: 归一化并写回全局内存                        │
│  O = O / l                                         │
└────────────────────────────────────────────────────┘
```

## 💾 内存层次

### 全局内存 (HBM)
- **Q, K, V, O**: 输入输出tensors
- **大小**: `batch * num_heads * seq_len * head_dim`
- **带宽**: ~2 TB/s (A100)
- **延迟**: ~100ns

### 共享内存 (SRAM)
- **Q_block**: `[kBlockM, kHeadDim]` = `[64, 64]` = 8KB (FP16)
- **K_block**: `[kBlockN, kHeadDim]` = `[64, 64]` = 8KB (FP16)
- **V_block**: `[kBlockN, kHeadDim]` = `[64, 64]` = 8KB (FP16)
- **S**: `[kBlockM, kBlockN]` = `[64, 64]` = 16KB (FP32)
- **P**: `[kBlockM, kBlockN]` = `[64, 64]` = 16KB (FP32)
- **总计**: ~56KB
- **带宽**: ~19 TB/s (A100)
- **延迟**: ~5ns

### 寄存器
- **线程局部变量**
- **累加器**
- **带宽**: 最快
- **延迟**: 1 cycle

## 🧮 计算分析

### FLOPs计算

对于一个attention操作 `O = softmax(Q @ K^T) @ V`:

1. **Q @ K^T**: `2 * M * N * D` FLOPs
2. **Softmax**: `5 * M * N` FLOPs (exp, max, sum, div)
3. **P @ V**: `2 * M * N * D` FLOPs

**总计**: `4 * M * N * D + 5 * M * N ≈ 4 * M * N * D` FLOPs

对于 `seq_len=S, head_dim=D`:
- **Total FLOPs** = `4 * S^2 * D * batch * num_heads`

### 内存访问分析

#### 标准实现
```
Q: S * D      (读)
K: S * D      (读)
S: S * S      (写+读)  ← 瓶颈！
V: S * D      (读)
O: S * D      (写)
---
Total: 2*S*D + 2*S*S + S*D = S*(3*D + 2*S)
```

当 S >> D 时，主导项是 `2*S*S`，复杂度 **O(S²)**

#### Flash Attention
```
Q: S * D      (读)
K: S * D      (读)
V: S * D      (读)
O: S * D      (写)
---
Total: 4*S*D
```

复杂度 **O(S)**，节省了 `S*S` 的中间存储！

### 理论加速比

```
内存访问时间 = 数据量 / 带宽

标准实现时间 ≈ S*(3*D + 2*S) / BW_HBM
Flash Attn时间 ≈ 4*S*D / BW_SRAM + 计算时间

当 S >> D 时:
加速比 ≈ (2*S*S / BW_HBM) / (4*S*D / BW_SRAM)
      = (S * BW_SRAM) / (2*D * BW_HBM)
      ≈ (S * 19000) / (2*D * 2000)  # A100数据
      = 4.75 * S / D
```

对于 `S=1024, D=64`:
- 理论加速比 ≈ **76x** (仅考虑内存)

实际加速比约 **3-8x**，因为还有计算时间。

## 🎯 关键优化技术

### 1. Tiling (分块)

**目的**: 将大矩阵分成小块，每块fit进共享内存

```
┌────────────────────────────────┐
│  Q [seq_len, head_dim]         │
│  ┌────┬────┬────┐              │
│  │ Q0 │ Q1 │ Q2 │  [BlockM,D]  │
│  ├────┼────┼────┤              │
│  │ Q3 │ Q4 │ Q5 │              │
│  └────┴────┴────┘              │
└────────────────────────────────┘

┌────────────────────────────────┐
│  K [seq_len, head_dim]         │
│  ┌────┬────┐                   │
│  │ K0 │ K1 │  [BlockN, D]      │
│  ├────┼────┤                   │
│  │ K2 │ K3 │                   │
│  ├────┼────┤                   │
│  │ K4 │ K5 │                   │
│  └────┴────┘                   │
└────────────────────────────────┘
```

**块大小选择**:
- 太小：启动开销大
- 太大：共享内存不够
- 典型值：64x64, 128x128

### 2. 在线Softmax

**标准两次遍历**:
```python
# Pass 1: 找max
m = max(x)

# Pass 2: 计算exp和sum
s = sum(exp(x - m))

# Pass 3: 归一化
y = exp(x - m) / s
```

**在线单次遍历** (Flash Attention核心):
```python
m_old, l_old = current_max, current_sum

# 处理新数据
m_new = max(m_old, max(x_new))

# 更新统计量
correction = exp(m_old - m_new)
l_new = correction * l_old + sum(exp(x_new - m_new))

# 修正之前的输出
y_old = y_old * correction * (l_old / l_new)
y_new = exp(x_new - m_new) / l_new

# 合并
y = y_old + y_new
```

### 3. Kernel融合

**标准实现** (多个kernel):
```
kernel1: S = Q @ K^T
kernel2: S_max = rowmax(S)
kernel3: S_exp = exp(S - S_max)
kernel4: S_sum = rowsum(S_exp)
kernel5: P = S_exp / S_sum
kernel6: O = P @ V
```
每个kernel都需要读写全局内存 → **6x 全局内存访问**

**Flash Attention** (单个kernel):
```
kernel_fused: 
    - 计算 S
    - 计算 softmax (在线)
    - 计算 O
    - 全部在共享内存中
```
只在开始和结束时访问全局内存 → **1x 全局内存访问**

### 4. 共享内存优化

**Bank Conflict避免**:
```cpp
// Bad: 所有线程访问同一个bank
shared_mem[threadIdx.x * N]

// Good: 使用padding或swizzling
shared_mem[threadIdx.x * (N + 1)]
```

**Swizzling**: 打乱内存布局以避免bank conflict
```
不使用swizzle:     使用swizzle:
[0][1][2][3]      [0][2][1][3]
[4][5][6][7]  →   [4][6][5][7]
[8][9][A][B]      [8][A][9][B]
```

## 📊 性能特征

### 计算vs内存bound

```
Compute Intensity = FLOPs / Bytes

标准Attention:
  FLOPs = 4*S^2*D
  Bytes = S*(3*D + 2*S)
  Intensity ≈ (4*S*D) / (2*S) = 2*D
  → Memory bound (对于D=64-128)

Flash Attention:
  FLOPs = 4*S^2*D (same)
  Bytes = 4*S*D
  Intensity ≈ S
  → Compute bound (当S足够大)
```

### 可扩展性

| Seq Length | 标准Attn内存 | Flash Attn内存 | 内存节省 |
|-----------|-------------|---------------|---------|
| 512 | 512KB | 128KB | 4x |
| 1024 | 2MB | 256KB | 8x |
| 2048 | 8MB | 512KB | 16x |
| 4096 | 32MB | 1MB | 32x |
| 8192 | 128MB | 2MB | 64x |

**结论**: 序列越长，Flash Attention优势越明显！

## 🔬 与完整实现的差异

| 特性 | 最小实现 | 完整实现 | 影响 |
|-----|---------|---------|------|
| 内存布局 | 简单 | Swizzled | 性能 |
| GEMM | 简单循环 | Cutlass优化 | 性能 |
| 寄存器使用 | 最小 | 优化 | 性能 |
| Bank conflict | 未优化 | 优化 | 性能 |
| Warp级优化 | 无 | 有 | 性能 |
| Causal mask | 无 | 有 | 功能 |
| Dropout | 无 | 有 | 功能 |
| 向后传播 | 无 | 有 | 功能 |

**性能差距**: 最小实现约为完整实现的 **50-70%**

## 🎓 学习建议

### 第1阶段: 理解算法
1. 阅读Flash Attention论文
2. 理解在线softmax的数学原理
3. 手动计算一个小例子 (2x2矩阵)

### 第2阶段: 理解实现
1. 阅读 `flash_attn_minimal.cu`
2. 理解内存布局和数据流
3. 对比参考实现，找出差异

### 第3阶段: 实验
1. 修改块大小，观察性能变化
2. 添加简单的profiling
3. 尝试添加causal mask

### 第4阶段: 优化
1. 学习Cutlass的GEMM实现
2. 实现swizzling
3. 优化bank conflict
4. 使用Nsight Compute分析

## 📚 参考资料

### 论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://tridao.me/publications/flash2/flash2.pdf)

### 代码
- [Flash Attention Official](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [CUTLASS CUTE](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)

### 教程
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Shared Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

希望这份架构说明帮助你深入理解Flash Attention的实现！🚀

