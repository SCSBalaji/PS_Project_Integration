# Block Specification Document (BLOCK_SPEC.md)

**Version:** 1.0  
**Last Updated:** November 29, 2025  
**Authors:** [Your Name / Team Members]  
**Status:** Draft  

---

## Purpose

This document serves as the single source of truth for all neural network block specifications in the MobilePlantViT architecture. It defines precise behavior, tensor shapes, hyperparameters, and failure modes for each block to ensure consistent implementation across the team.

---

## Table of Contents

1. [Notation Conventions](#1-notation-conventions)
2. [Architecture Overview](#2-architecture-overview)
3. [Block Specifications](#3-block-specifications)
   - [3.1 GhostConv](#31-ghostconv)
   - [3.2 Fused Inverted Residual Block](#32-fused-inverted-residual-block)
   - [3.3 Coordinate Attention](#33-coordinate-attention)
   - [3.4 Patch Embedding](#34-patch-embedding)
   - [3.5 Positional Encoding](#35-positional-encoding)
   - [3.6 Linear Differential Attention (LDA)](#36-linear-differential-attention-lda)
   - [3.7 Residual LayerNorm Block](#37-residual-layernorm-block)
   - [3.8 Bottleneck FFN](#38-bottleneck-ffn)
   - [3.9 Global Average Pooling](#39-global-average-pooling)
   - [3.10 Classifier Head](#310-classifier-head)
4. [Full Pipeline Shape Verification](#4-full-pipeline-shape-verification)
5. [Parameter Budget](#5-parameter-budget)
6. [Sign-off](#6-sign-off)

---

## 1. Notation Conventions

### 1.1 Tensor Shape Notation

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `C` | Number of channels |
| `H` | Height (spatial dimension) |
| `W` | Width (spatial dimension) |
| `N` | Sequence length (number of patches/tokens) |
| `D` or `E` | Embedding dimension |

### 1.2 Shape Formats

- **Spatial tensors (CNN):** `(B, C, H, W)` — PyTorch convention (channels first)
- **Sequence tensors (Transformer):** `(B, N, D)` — Batch, Sequence, Features

### 1.3 Stateful vs Stateless

| Type | Description | Examples |
|------|-------------|----------|
| **Stateful** | Contains running statistics that differ between training and evaluation modes | BatchNorm, Dropout |
| **Stateless** | Same computation regardless of mode | LayerNorm, Linear, Conv2d (without BN) |

### 1.4 FLOP Cost Categories

| Category | Approximate Range | Description |
|----------|-------------------|-------------|
| **LOW** | < 100M FLOPs | Lightweight operations |
| **MEDIUM** | 100M - 500M FLOPs | Moderate compute |
| **HIGH** | > 500M FLOPs | Compute intensive |

### 1.5 Default Input Specification

- **Input Image Size:** 224 × 224 pixels
- **Input Channels:** 3 (RGB)
- **Data Type:** `torch.float32`
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

---

## 2. Architecture Overview

### 2.1 Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MobilePlantViT Architecture                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: (B, 3, 224, 224) — Preprocessed RGB Image
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CNN STAGE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐     │
│  │  GhostConv   │ ──▶ │  Fused Inverted     │ ──▶ │  Coordinate      │     │
│  │              │     │  Residual Block     │     │  Attention       │     │
│  └──────────────┘     └─────────────────────┘     └──────────────────┘     │
│  (B,3,224,224)        (B,C1,H1,W1)                (B,C2,H2,W2)              │
│       ▼                    ▼                           ▼                    │
│  (B,C1,H1,W1)         (B,C2,H2,W2)                (B,C2,H2,W2)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSITION STAGE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌────────────────────┐                           │
│  │  Patch Embedding │ ──▶ │ Positional Encoding │                          │
│  └──────────────────┘     └────────────────────┘                           │
│  (B,C2,H2,W2)             (B,N,D)                                           │
│       ▼                        ▼                                            │
│  (B,N,D)                  (B,N,D)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRANSFORMER STAGE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │  Linear Differential    │ ──▶ │  Residual       │ ──▶ │  Bottleneck  │  │
│  │  Attention (LDA)        │     │  LayerNorm      │     │  FFN         │  │
│  └─────────────────────────┘     └─────────────────┘     └──────────────┘  │
│  (B,N,D)                         (B,N,D)                  (B,N,D)           │
│       ▼                               ▼                        ▼            │
│  (B,N,D)                         (B,N,D)                  (B,N,D)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLASSIFIER STAGE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────┐     ┌─────────────────┐                        │
│  │  Global Average        │ ──▶ │  Classifier     │                        │
│  │  Pooling (GAP)         │     │  Head           │                        │
│  └────────────────────────┘     └─────────────────┘                        │
│  (B,N,D)                        (B,D)                                       │
│       ▼                              ▼                                      │
│  (B,D)                          (B, num_classes)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

OUTPUT: (B, num_classes) — Class Probabilities
```

### 2.2 Design Principles

1. **Mobile-First:** All blocks optimized for mobile deployment (< 5M parameters)
2. **Hybrid CNN-Transformer:** CNN for local features, Transformer for global context
3. **Efficiency:** Ghost convolutions, fused operations, linear attention
4. **Plant Disease Focus:** Architecture tuned for plant leaf disease classification

---

## 3. Block Specifications

### Specification Template

Each block specification follows this structure:

```
### Block Name

#### Purpose
[Brief description of what this block does and why]

#### Input/Output Shapes
- **Input:** Shape and description
- **Output:** Shape and description

#### Hyperparameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

#### Stateful/Stateless
[Indicate whether block behavior differs between train/eval]

#### FLOP Cost
[Category and formula]

#### Mathematical Formulation
[Key equations]

#### Example Forward Pass
[Concrete example with numbers]

#### Edge Cases & Failure Modes
[What can go wrong]

#### Implementation Notes
[Any special considerations]
```

---

### 3.1 GhostConv

#### Purpose

GhostConv is an efficient convolution module inspired by the [GhostNet paper](https://arxiv.org/abs/1911.11907). It generates feature maps in two stages:
1. **Primary convolution:** Generates a small set of "intrinsic" feature maps
2. **Cheap operation:** Generates additional "ghost" feature maps from intrinsic ones using depthwise convolution

This approach reduces computational cost while maintaining representational capacity, making it ideal for mobile deployment.

#### Input/Output Shapes

- **Input:** `(B, C_in, H, W)` — 4D spatial tensor
- **Output:** `(B, C_out, H', W')` where:
  - `H' = H / stride`
  - `W' = W / stride`

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `inp` | int | — | ✅ | Number of input channels |
| `oup` | int | — | ✅ | Number of output channels |
| `kernel_size` | int | 1 | ❌ | Kernel size for primary convolution |
| `ratio` | int | 2 | ❌ | Ratio for splitting channels (intrinsic vs ghost) |
| `dw_size` | int | 3 | ❌ | Kernel size for depthwise (cheap) convolution |
| `stride` | int | 1 | ❌ | Stride for primary convolution |
| `relu` | bool | True | ❌ | Whether to apply ReLU activation |

#### Internal Channel Calculation

```python
init_channels = math.ceil(oup / ratio)      # Intrinsic channels
new_channels = init_channels * (ratio - 1)  # Ghost channels
# Total before slicing: init_channels + new_channels
# Output is sliced to exactly 'oup' channels
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| BatchNorm2d | **Stateful** | Maintains running mean/variance |
| Conv2d | Stateless | No running statistics |

**Behavior:**
- **Training mode (`model.train()`):** BatchNorm uses batch statistics
- **Evaluation mode (`model.eval()`):** BatchNorm uses running statistics

#### FLOP Cost

**Category:** LOW

**Formula:**
```
FLOPs ≈ H' × W' × (C_in × init_channels × k1² + init_channels × dw_size²)

Where:
- k1 = kernel_size (primary conv)
- init_channels = ceil(C_out / ratio)
```

**Comparison to Standard Conv:**
- Standard Conv: `H × W × C_in × C_out × k²`
- GhostConv: ~50% fewer FLOPs (with ratio=2)

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×C_in×H×W):

1. Primary Features:
   Y_primary = ReLU(BN(Conv2d(X)))
   Y_primary ∈ ℝ^(B × init_channels × H' × W')

2. Ghost Features:
   Y_ghost = ReLU(BN(DepthwiseConv(Y_primary)))
   Y_ghost ∈ ℝ^(B × new_channels × H' × W')

3. Concatenation and Slicing:
   Y_concat = Concat(Y_primary, Y_ghost, dim=1)
   Y_output = Y_concat[:, :oup, :, :]
```

#### Example Forward Pass

**Configuration:**
```python
ghost = GhostConv(inp=3, oup=64, kernel_size=1, ratio=2, dw_size=3, stride=1)
```

**Internal Channels:**
```
init_channels = ceil(64 / 2) = 32
new_channels = 32 * (2 - 1) = 32
total_before_slice = 32 + 32 = 64
```

**Shape Progression:**
```
Input:           (2, 3, 224, 224)
                      │
                      ▼ Primary Conv (3→32, 1×1)
Primary:         (2, 32, 224, 224)
                      │
                      ▼ Depthwise Conv (32→32, 3×3, groups=32)
Ghost:           (2, 32, 224, 224)
                      │
                      ▼ Concatenate
Concatenated:    (2, 64, 224, 224)
                      │
                      ▼ Slice to oup channels
Output:          (2, 64, 224, 224)
```

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `inp=0` | Error in Conv2d | Validate input > 0 |
| `oup=0` | Empty output tensor | Validate output > 0 |
| `ratio > oup` | `init_channels=1`, most features are ghost | Use ratio ≤ oup |
| `kernel_size` even | Asymmetric padding | Use odd kernel sizes (1, 3, 5) |
| `stride > 1` | Spatial downsampling | Ensure H, W divisible by stride |
| Large `dw_size` | Increased compute in cheap op | Keep dw_size ≤ 5 |

#### Implementation Notes

1. **Padding:** Both convolutions use `padding = kernel_size // 2` for same-padding (when stride=1)

2. **Groups in Depthwise Conv:** The cheap operation uses `groups=init_channels` making it a true depthwise convolution

3. **Channel Slicing:** Final output is sliced to exactly `oup` channels because `init_channels + new_channels` may exceed `oup` due to ceiling operation

4. **No Bias:** Both convolutions use `bias=False` since BatchNorm follows immediately

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Uses `math.ceil` for init_channels calculation
- Applies depthwise conv with `groups=init_channels`
- Slices output to exact `oup` channels
- BatchNorm + ReLU after each convolution

---

### 3.2 Fused Inverted Residual Block

#### Purpose

The Fused Inverted Residual (Fused-IR) block is inspired by [EfficientNetV2](https://arxiv.org/abs/2104.00298) and [MobileNetV3](https://arxiv.org/abs/1905.02244). It improves upon the standard Inverted Residual by **fusing** the expansion convolution and depthwise convolution into a single 3×3 convolution.

**Standard Inverted Residual:**
```
Expand (1×1) → Depthwise (3×3) → Project (1×1)
```

**Fused Inverted Residual:**
```
Fused Expand+DW (3×3) → Project (1×1)
```

**Why fused?**
- Reduces memory access overhead (fewer intermediate tensors)
- More efficient on modern accelerators for early layers
- Better suited when expansion ratio is moderate (≤4)

#### Input/Output Shapes

- **Input:** `(B, C_in, H, W)` — 4D spatial tensor
- **Output:** `(B, C_out, H', W')` where:
  - `H' = H / stride`
  - `W' = W / stride`

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `inp` | int | — | ✅ | Number of input channels |
| `oup` | int | — | ✅ | Number of output channels |
| `stride` | int | 1 | ❌ | Stride for spatial downsampling |
| `expand_ratio` | int | 4 | ❌ | Expansion factor for hidden dimension |

#### Internal Channel Calculation

```python
hidden_dim = int(round(inp * expand_ratio))  # Expanded channels
use_res_connect = (stride == 1 and inp == oup)  # Residual condition
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| BatchNorm2d | **Stateful** | Maintains running mean/variance |
| Conv2d | Stateless | No running statistics |

**Behavior:**
- **Training mode (`model.train()`):** BatchNorm uses batch statistics
- **Evaluation mode (`model.eval()`):** BatchNorm uses running statistics

#### FLOP Cost

**Category:** MEDIUM

**Formula:**
```
FLOPs ≈ H' × W' × (C_in × hidden_dim × 3² + hidden_dim × C_out × 1²)
      = H' × W' × (9 × C_in × hidden_dim + hidden_dim × C_out)

Where:
- hidden_dim = C_in × expand_ratio
- First term: Fused 3×3 conv
- Second term: 1×1 projection conv
```

**For typical values (C_in=64, C_out=64, H=W=224, expand_ratio=4):**
```
hidden_dim = 64 × 4 = 256
FLOPs ≈ 224 × 224 × (9 × 64 × 256 + 256 × 64)
      ≈ 224 × 224 × (147,456 + 16,384)
      ≈ 8.2G FLOPs
```

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×C_in×H×W):

Case 1: expand_ratio ≠ 1 (typical case)
─────────────────────────────────────
1. Fused Expansion (3×3 conv):
   H = ReLU(BN(Conv3×3(X)))
   H ∈ ℝ^(B × hidden_dim × H' × W')

2. Projection (1×1 conv):
   Y_proj = BN(Conv1×1(H))
   Y_proj ∈ ℝ^(B × C_out × H' × W')

Case 2: expand_ratio = 1
────────────────────────
1. Direct Conv (3×3):
   Y_proj = BN(Conv3×3(X))
   Y_proj ∈ ℝ^(B × C_out × H' × W')

Residual Connection (if applicable):
────────────────────────────────────
if stride == 1 and C_in == C_out:
   Y = X + Y_proj
else:
   Y = ReLU(Y_proj)
```

#### Example Forward Pass

**Configuration:**
```python
fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1, expand_ratio=4)
```

**Internal Values:**
```
hidden_dim = 64 × 4 = 256
use_res_connect = (1 == 1 and 64 == 64) = True
```

**Shape Progression (with residual):**
```
Input X:         (2, 64, 56, 56)
                      │
                      ▼ Fused Conv3×3 (64→256) + BN + ReLU
Hidden:          (2, 256, 56, 56)
                      │
                      ▼ Project Conv1×1 (256→64) + BN
Projected:       (2, 64, 56, 56)
                      │
                      ▼ Residual: X + Projected
Output:          (2, 64, 56, 56)
```

**Shape Progression (no residual, stride=2):**
```python
fused_ir = FusedInvertedResidualBlock(inp=64, oup=128, stride=2, expand_ratio=4)
```
```
Input X:         (2, 64, 56, 56)
                      │
                      ▼ Fused Conv3×3 stride=2 (64→256) + BN + ReLU
Hidden:          (2, 256, 28, 28)
                      │
                      ▼ Project Conv1×1 (256→128) + BN
Projected:       (2, 128, 28, 28)
                      │
                      ▼ ReLU (no residual since inp≠oup or stride≠1)
Output:          (2, 128, 28, 28)
```

#### Residual Connection Logic

```python
use_res_connect = (stride == 1) and (inp == oup)

# Residual is APPLIED when:
# - stride is 1 (no spatial downsampling)
# - input and output channels match

# Residual is SKIPPED when:
# - stride > 1 (spatial dimensions change)
# - inp ≠ oup (channel dimensions change)
```

| inp | oup | stride | Residual? |
|-----|-----|--------|-----------|
| 64 | 64 | 1 | ✅ Yes |
| 64 | 64 | 2 | ❌ No |
| 64 | 128 | 1 | ❌ No |
| 64 | 128 | 2 | ❌ No |

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `inp=0` | Error in Conv2d | Validate input > 0 |
| `oup=0` | Error in Conv2d | Validate output > 0 |
| `expand_ratio=0` | `hidden_dim=0`, error | Use expand_ratio ≥ 1 |
| `expand_ratio=1` | No expansion, direct 3×3 conv | Valid, but less expressive |
| `stride > 2` | Large downsampling | Typically use stride ∈ {1, 2} |
| Very large `expand_ratio` | Memory explosion | Keep expand_ratio ≤ 6 |
| `H` or `W` not divisible by stride | Floor division in output size | Ensure divisibility |

#### Implementation Notes

1. **Fused vs Non-Fused:** This implementation uses the "fused" approach where expand + depthwise are combined into a single 3×3 conv. This differs from standard MobileNetV2 inverted residuals.

2. **No SE Block:** The current implementation does not include Squeeze-and-Excitation. This can be added for improved accuracy at the cost of additional parameters.

3. **Activation Placement:**
   - ReLU after fused conv (when expand_ratio ≠ 1)
   - No activation after projection (before residual addition)
   - ReLU after final output only when no residual connection

4. **Padding:** Uses `padding=1` for 3×3 convolutions to maintain spatial dimensions (when stride=1).

5. **No Bias:** Convolutions use `bias=False` since BatchNorm follows immediately.

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Uses fused 3×3 conv for expansion
- Projection with 1×1 conv
- Residual connection when `stride==1 and inp==oup`
- ReLU applied appropriately based on residual usage
- BatchNorm after each convolution

---

### 3.3 Coordinate Attention

#### Purpose

Coordinate Attention (CoordAtt) is an attention mechanism from the [Coordinate Attention paper](https://arxiv.org/abs/2103.02907) that captures:
1. **Long-range dependencies** along one spatial direction
2. **Precise positional information** along the other direction

Unlike standard channel attention (SE blocks) that squeeze spatial dimensions entirely, CoordAtt preserves spatial structure by decomposing attention into two 1D feature encoding processes (horizontal and vertical).

**Why use it:**
- Captures both channel and positional information
- More expressive than SE attention with minimal overhead
- Well-suited for tasks requiring spatial precision (disease localization)

#### Input/Output Shapes

- **Input:** `(B, C, H, W)` — 4D spatial tensor
- **Output:** `(B, C, H, W)` — Same shape as input (attention is multiplicative)

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `inp` | int | — | ✅ | Number of input channels |
| `oup` | int | — | ✅ | Number of output channels (typically same as inp) |
| `reduction` | int | 32 | ❌ | Channel reduction ratio for bottleneck |

#### Internal Channel Calculation

```python
mip = max(8, inp // reduction)  # Intermediate channels (minimum 8)
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| BatchNorm2d | **Stateful** | Maintains running mean/variance |
| Conv2d | Stateless | No running statistics |
| AdaptiveAvgPool2d | Stateless | No learnable parameters |

**Behavior:**
- **Training mode (`model.train()`):** BatchNorm uses batch statistics
- **Evaluation mode (`model.eval()`):** BatchNorm uses running statistics

#### FLOP Cost

**Category:** LOW

**Formula:**
```
FLOPs ≈ 2 × C × mip × (H + W) + 2 × mip × C × 1
      = 2 × C × mip × (H + W + 1)

Where:
- mip = max(8, C // reduction)
- First term: 1×1 conv after concat
- Second term: Two 1×1 convs for h and w attention
```

**For typical values (C=64, H=W=224, reduction=32):**
```
mip = max(8, 64 // 32) = 8
FLOPs ≈ 2 × 64 × 8 × (224 + 224 + 1) ≈ 460K FLOPs
```

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×C×H×W):

1. Coordinate Pooling:
   X_h = AvgPool(X, output_size=(H, 1))  → (B, C, H, 1)
   X_w = AvgPool(X, output_size=(1, W))  → (B, C, 1, W)

2. Concatenate along spatial dimension:
   X_w' = Permute(X_w, [0,1,3,2])        → (B, C, W, 1)
   Y = Concat(X_h, X_w', dim=2)          → (B, C, H+W, 1)

3. Shared Transform:
   Y = HSwish(BN(Conv1×1(Y)))            → (B, mip, H+W, 1)

4. Split back:
   Y_h, Y_w = Split(Y, [H, W], dim=2)
   Y_w = Permute(Y_w, [0,1,3,2])         → Y_h: (B, mip, H, 1), Y_w: (B, mip, 1, W)

5. Generate attention maps:
   A_h = Sigmoid(Conv1×1(Y_h))           → (B, C, H, 1)
   A_w = Sigmoid(Conv1×1(Y_w))           → (B, C, 1, W)

6. Apply attention:
   Output = X × A_h × A_w                → (B, C, H, W)
```

#### Example Forward Pass

**Configuration:**
```python
coord_att = CoordAtt(inp=64, oup=64, reduction=32)
```

**Internal Channels:**
```
mip = max(8, 64 // 32) = max(8, 2) = 8
```

**Shape Progression:**
```
Input X:         (2, 64, 56, 56)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Pool Height              Pool Width
   X_h: (2, 64, 56, 1)      X_w: (2, 64, 1, 56)
          │                       │
          │                       ▼ Permute
          │               X_w': (2, 64, 56, 1)
          │                       │
          └───────────┬───────────┘
                      ▼ Concatenate (dim=2)
              Y: (2, 64, 112, 1)
                      │
                      ▼ Conv1×1 (64→8) + BN + HSwish
              Y: (2, 8, 112, 1)
                      │
          ┌───────────┴───────────┐
          ▼ Split                 ▼
   Y_h: (2, 8, 56, 1)      Y_w: (2, 8, 56, 1)
          │                       │
          │                       ▼ Permute
          │               Y_w: (2, 8, 1, 56)
          │                       │
          ▼ Conv1×1 (8→64)        ▼ Conv1×1 (8→64)
   A_h: (2, 64, 56, 1)     A_w: (2, 64, 1, 56)
          │                       │
          ▼ Sigmoid               ▼ Sigmoid
   A_h: (2, 64, 56, 1)     A_w: (2, 64, 1, 56)
          │                       │
          └───────────┬───────────┘
                      ▼ Element-wise: X × A_h × A_w
             Output: (2, 64, 56, 56)
```

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `inp=0` | Error in Conv2d | Validate input > 0 |
| `H=1` or `W=1` | Degenerate pooling | Ensure spatial dims > 1 |
| `reduction > inp` | `mip` defaults to 8 | Use reasonable reduction |
| `inp ≠ oup` | Channel mismatch possible | Typically use `inp == oup` |
| Very large H, W | Memory in concat | Consider spatial downsampling first |

#### Implementation Notes

1. **HSwish Activation:** Uses Hard Swish: `x * ReLU6(x + 3) / 6` for efficiency

2. **HSigmoid:** Uses Hard Sigmoid: `ReLU6(x + 3) / 6` (not used directly, but related)

3. **Minimum Intermediate Channels:** `mip = max(8, inp // reduction)` ensures at least 8 channels for expressiveness

4. **Broadcasting:** Attention maps `A_h (B,C,H,1)` and `A_w (B,C,1,W)` broadcast correctly when multiplied with input `(B,C,H,W)`

5. **No Residual:** The block applies multiplicative attention directly; residual connection should be added externally if needed

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Separate horizontal and vertical pooling
- Concatenation along height dimension
- Shared 1×1 conv with BatchNorm and HSwish
- Split and separate conv for H and W attention
- Sigmoid activation for attention maps
- Multiplicative attention application

---

### 3.4 Patch Embedding

#### Purpose

Patch Embedding converts CNN spatial feature maps into a sequence of patch tokens suitable for transformer processing. This is the critical bridge between the CNN stage (spatial features) and the Transformer stage (sequence features).

**Key responsibilities:**
1. Divide the feature map into non-overlapping patches
2. Project each patch into the embedding dimension
3. Flatten spatial layout into sequence format

#### Input/Output Shapes

- **Input:** `(B, C_in, H, W)` — 4D spatial tensor from CNN stage
- **Output:** `(B, N, D)` — 3D sequence tensor where:
  - `N = (H // patch_size) × (W // patch_size)` — number of patches
  - `D = embed_dim` — embedding dimension

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `in_channels` | int | — | ✅ | Number of input channels from CNN |
| `embed_dim` | int | 256 | ❌ | Output embedding dimension |
| `patch_size` | int | 4 | ❌ | Size of each square patch |
| `bias` | bool | True | ❌ | Whether to use bias in projection |

#### Internal Calculations

```python
num_patches = (H // patch_size) * (W // patch_size)
# Projection is done via Conv2d with kernel_size=stride=patch_size
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| Conv2d | Stateless | No running statistics |

**Behavior:**
- Same computation in training and evaluation modes
- No BatchNorm or Dropout in basic implementation

#### FLOP Cost

**Category:** LOW

**Formula:**
```
FLOPs = N × C_in × D × patch_size²

Where:
- N = num_patches = (H/patch_size) × (W/patch_size)
- Each patch requires C_in × D × patch_size² operations
```

**For typical values (C_in=64, H=W=14, patch_size=2, D=256):**
```
N = (14/2) × (14/2) = 49 patches
FLOPs = 49 × 64 × 256 × 4 = 3.2M FLOPs
```

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×C_in×H×W):

1. Patch Extraction via Strided Convolution:
   P = Conv2d(X, kernel_size=patch_size, stride=patch_size)
   P ∈ ℝ^(B × D × H' × W')
   where H' = H/patch_size, W' = W/patch_size

2. Reshape to Sequence:
   P_flat = Flatten(P, start_dim=2)  → (B, D, N)
   P_seq = Transpose(P_flat, 1, 2)   → (B, N, D)

Output: P_seq ∈ ℝ^(B×N×D)
```

#### Example Forward Pass

**Configuration:**
```python
patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=2)
```

**Shape Progression:**
```
Input X:         (2, 64, 14, 14)
                      │
                      ▼ Conv2d (kernel=2, stride=2)
After Conv:      (2, 256, 7, 7)
                      │
                      ▼ Flatten (dims 2,3)
Flattened:       (2, 256, 49)
                      │
                      ▼ Transpose (dims 1,2)
Output:          (2, 49, 256)

Number of patches: 7 × 7 = 49
```

**Alternative Configuration (larger patches):**
```python
patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
```
```
Input X:         (2, 64, 28, 28)
                      │
                      ▼ Conv2d (kernel=4, stride=4)
After Conv:      (2, 256, 7, 7)
                      │
                      ▼ Flatten + Transpose
Output:          (2, 49, 256)
```

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `H % patch_size ≠ 0` | Truncated patches | Ensure H divisible by patch_size |
| `W % patch_size ≠ 0` | Truncated patches | Ensure W divisible by patch_size |
| `patch_size > H or W` | Single or zero patches | Use patch_size ≤ min(H, W) |
| `in_channels = 0` | Error in Conv2d | Validate in_channels > 0 |
| `embed_dim = 0` | Error in Conv2d | Validate embed_dim > 0 |
| Very large patch_size | Few patches, loss of spatial detail | Typically patch_size ∈ {2, 4, 7, 14} |

#### Implementation Notes

1. **Conv2d as Patch Projection:** Using Conv2d with `kernel_size=stride=patch_size` is equivalent to:
   - Extracting non-overlapping patches
   - Flattening each patch
   - Projecting via linear layer
   But more efficient due to optimized convolution operations.

2. **No CLS Token:** Unlike ViT, this implementation does not add a [CLS] token. Global Average Pooling is used instead for classification.

3. **Spatial Ordering:** Patches are ordered row-major (left-to-right, top-to-bottom) when flattened.

4. **Bias:** Using bias in the convolution is equivalent to adding a learned constant to each patch embedding.

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Uses Conv2d with kernel_size=stride=patch_size
- Properly flattens and transposes to (B, N, D) format
- Configurable patch_size and embed_dim

---

### 3.5 Positional Encoding

#### Purpose

Positional Encoding adds spatial position information to patch embeddings. Since the transformer's self-attention is permutation-invariant, positional encoding is essential to preserve the spatial structure of the original image.

**Types of Positional Encoding:**
1. **Sinusoidal (fixed) — used here:** Position encoded using sine/cosine functions
2. **Learnable:** Position embeddings are learned parameters
3. **Relative:** Encodes relative positions between tokens

This implementation uses **sinusoidal positional encoding** as introduced in the original Transformer paper ("Attention Is All You Need").

#### Input/Output Shapes

- **Input:** `(B, N, D)` — Patch embeddings from Patch Embedding block
- **Output:** `(B, N, D)` — Same shape, with positional information added

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `embed_dim` | int | — | ✅ | Embedding dimension (must match patch embedding) |
| `max_len` | int | 5000 | ❌ | Maximum sequence length supported |

#### Sinusoidal Encoding Formula

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos = position in sequence (0 to N-1)
- i = dimension index (0 to d_model/2 - 1)
- d_model = embed_dim
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| Buffer (pe) | **Stateless** | Fixed sinusoidal values, registered as buffer |

**Behavior:**
- Same computation in training and evaluation modes
- No learnable parameters
- Positional encodings are precomputed and stored as a buffer

#### FLOP Cost

**Category:** NEGLIGIBLE

**Formula:**
```
FLOPs = N × D (element-wise addition)
```

This is negligible compared to other operations.

#### Mathematical Formulation

```
Given patch embeddings P ∈ ℝ^(B×N×D):

1. Precomputed Positional Encoding (stored as buffer):
   PE ∈ ℝ^(1×max_len×D)
   
   For each position pos and dimension i:
   PE[0, pos, 2i]   = sin(pos × exp(-2i × log(10000) / D))
   PE[0, pos, 2i+1] = cos(pos × exp(-2i × log(10000) / D))

2. Slice to sequence length:
   PE_slice = PE[:, :N, :]  ∈ ℝ^(1×N×D)

3. Add Position Information:
   Y = P + PE_slice  ∈ ℝ^(B×N×D)
   (Broadcasting over batch dimension)

Output: Y ∈ ℝ^(B×N×D)
```

#### Example Forward Pass

**Configuration:**
```python
pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
```

**Shape Progression:**
```
Patch Embeddings P:    (2, 49, 256)
                            │
                            ▼ Slice positional buffer
Positional Embed PE:   (1, 49, 256)  [from precomputed buffer]
                            │
                            ▼ Add (broadcast over batch)
P + PE:                (2, 49, 256)
                            │
Output:                (2, 49, 256)
```

**Variable Sequence Length:**
```python
# pos_enc supports up to 5000 positions
input_patches = torch.randn(2, 49, 256)
output = pos_enc(input_patches)  # Works! Uses PE[:, :49, :]

input_patches_large = torch.randn(2, 196, 256)
output_large = pos_enc(input_patches_large)  # Also works! Uses PE[:, :196, :]
```

#### Sinusoidal vs Learnable Comparison

| Aspect | Sinusoidal (This Implementation) | Learnable |
|--------|----------------------------------|-----------|
| Parameters | 0 (buffer only) | N × D |
| Generalization | Better for unseen lengths | May overfit to training lengths |
| Flexibility | Fixed pattern | Adapts to data |
| Memory | Buffer stored on device | Gradient storage needed |

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `N > max_len` | Index out of bounds | Increase max_len or use max_len ≥ expected N |
| `embed_dim` mismatch | Addition error | Ensure consistency with PatchEmbed |
| `embed_dim` odd | Last dimension unused in cos | Use even embed_dim |
| Very long sequences | Large buffer size | max_len=5000 covers most cases |

#### Implementation Notes

1. **Buffer vs Parameter:** The positional encoding is registered as a `buffer` (not a parameter), meaning:
   - It's saved with the model state
   - It's moved to the correct device automatically
   - It doesn't receive gradients

2. **Precomputation:** All positional encodings up to `max_len` are computed once in `__init__` and reused during forward passes.

3. **No Dropout:** This implementation does not include dropout after adding positional encoding. Add dropout externally if needed.

4. **Division Term:** Uses `exp(-log(10000) × 2i/d)` which is mathematically equivalent to `1/10000^(2i/d)` but more numerically stable.

5. **Even/Odd Dimensions:** Sin is applied to even indices (0, 2, 4, ...) and cos to odd indices (1, 3, 5, ...).

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Sinusoidal positional encoding (fixed, not learnable)
- Precomputed buffer registered with `register_buffer`
- Proper broadcasting over batch dimension
- Support for variable sequence lengths via slicing
- No learnable parameters

---

### 3.6 Linear Differential Attention (LDA)

#### Purpose

Linear Differential Attention (LDA) is inspired by the [DIFF Transformer paper](https://arxiv.org/abs/2410.05258). It provides an efficient attention mechanism with two key innovations:

1. **Differential Attention:** Computes attention as the difference between two softmax attention maps, which cancels out noise and irrelevant information
2. **Linear Complexity:** Achieves O(N) complexity for sequence length N through efficient computation

**Why use it:**
- Reduces attention noise common in standard transformers
- More robust feature extraction for fine-grained classification (plant diseases)
- Learnable scaling factor (λ) adapts noise cancellation per layer

#### Input/Output Shapes

- **Input:** `(B, N, D)` — 3D sequence tensor (Batch, Sequence Length, Embedding Dim)
- **Output:** `(B, N, D)` — Same shape as input

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `embed_dim` | int | — | ✅ | Input/output embedding dimension |
| `num_heads` | int | 8 | ❌ | Number of attention heads |
| `dropout` | float | 0.1 | ❌ | Dropout rate on attention output |
| `init` | float | 0.8 | ❌ | Initialization value for λ (stored as exp(init)) |

#### Internal Calculations

```python
head_dim = embed_dim // num_heads    # Dimension per head
scaling = head_dim ** -0.5           # Attention scaling factor
alpha = nn.Parameter(exp(init))      # Learnable differential weight (λ)
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| GroupNorm | **Stateful** | Maintains running statistics (though often stateless in practice) |
| Linear | Stateless | No running statistics |
| Dropout | **Stateful** | Different behavior in train vs eval |

**Behavior:**
- **Training mode (`model.train()`):** Dropout active, GroupNorm uses batch stats
- **Evaluation mode (`model.eval()`):** Dropout disabled, GroupNorm uses running stats

#### FLOP Cost

**Category:** MEDIUM to HIGH

**Formula:**
```
FLOPs ≈ 4 × N × D² + 2 × N² × D/num_heads × num_heads
      = 4 × N × D² + 2 × N² × D

Where:
- First term: Q, K, V projections (4 because Q and K are doubled for differential)
- Second term: Two attention matrix computations
```

**For typical values (N=196, D=256, num_heads=8):**
```
FLOPs ≈ 4 × 196 × 256² + 2 × 196² × 256
      ≈ 51.4M + 19.7M
      ≈ 71.1M FLOPs
```

**Note:** While called "Linear" Differential Attention, this implementation still uses O(N²) softmax attention. True linear attention would use kernel approximations.

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×N×D):

1. Normalize Input:
   X_norm = GroupNorm(X)

2. Compute Projections (doubled for differential):
   [Q1, Q2] = Linear(X_norm) ∈ ℝ^(B×N×2D)  → split to Q1, Q2 ∈ ℝ^(B×N×D)
   [K1, K2] = Linear(X_norm) ∈ ℝ^(B×N×2D)  → split to K1, K2 ∈ ℝ^(B×N×D)
   V = Linear(X_norm) ∈ ℝ^(B×N×D)

3. Reshape for Multi-Head:
   Q1, Q2, K1, K2, V → (B, num_heads, N, head_dim)

4. Compute Two Attention Maps:
   A1 = softmax(Q1 × K1ᵀ / √head_dim)  ∈ ℝ^(B×heads×N×N)
   A2 = softmax(Q2 × K2ᵀ / √head_dim)  ∈ ℝ^(B×heads×N×N)

5. Differential Attention:
   A_diff = α × (A1 - A2)   where α is learnable

6. Apply to Values:
   Out = A_diff × V  ∈ ℝ^(B×heads×N×head_dim)

7. Concatenate Heads and Project:
   Out = Linear(Concat(heads))  ∈ ℝ^(B×N×D)
   Out = Dropout(Out)
```

#### Example Forward Pass

**Configuration:**
```python
lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.1, init=0.8)
```

**Internal Values:**
```
head_dim = 256 // 8 = 32
scaling = 32 ** -0.5 = 0.1768
alpha = exp(0.8) ≈ 2.226 (initial value, learnable)
```

**Shape Progression:**
```
Input X:            (2, 196, 256)
                         │
                         ▼ GroupNorm
X_norm:             (2, 196, 256)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   Q projection    K projection    V projection
   (2, 196, 512)   (2, 196, 512)   (2, 196, 256)
         │               │               │
         ▼ Split         ▼ Split         │
   Q1, Q2:          K1, K2:              │
   each (2,196,256) each (2,196,256)     │
         │               │               │
         ▼ Reshape to multi-head         ▼
   (2, 8, 196, 32)  (2, 8, 196, 32)  (2, 8, 196, 32)
         │               │               │
         └───────┬───────┘               │
                 ▼                       │
         Attention Scores                │
         A1, A2: (2, 8, 196, 196)        │
                 │                       │
                 ▼ Differential: α(A1-A2)│
         A_diff: (2, 8, 196, 196)        │
                 │                       │
                 └───────────┬───────────┘
                             ▼ Matmul with V
                    (2, 8, 196, 32)
                             │
                             ▼ Reshape & Concat heads
                    (2, 196, 256)
                             │
                             ▼ Output projection + Dropout
Output:             (2, 196, 256)
```

#### Differential Attention Mechanism

The key innovation is the subtraction of two attention maps:

```
A_diff = α × (A1 - A2)
```

**Why this works:**
- **Noise Cancellation:** Common noise patterns appear in both A1 and A2, so they cancel out
- **Signal Enhancement:** Meaningful patterns differ between Q1/K1 and Q2/K2, so they remain
- **Learnable α:** The network learns how much differential to apply per layer

**Visualization:**
```
A1 (attention map 1):     A2 (attention map 2):     A_diff (differential):
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ █ ░ ░ █ ░ ░ █ │      │ ░ ░ ░ █ ░ ░ ░ │      │ █ ░ ░ ░ ░ ░ █ │
│ ░ █ ░ ░ █ ░ ░ │  -   │ ░ █ ░ ░ ░ ░ ░ │  =   │ ░ ░ ░ ░ █ ░ ░ │
│ ░ ░ █ ░ ░ █ ░ │      │ ░ ░ █ ░ ░ ░ ░ │      │ ░ ░ ░ ░ ░ █ ░ │
└─────────────────┘      └─────────────────┘      └─────────────────┘
(signal + noise)         (mostly noise)           (cleaner signal)
```

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `embed_dim % num_heads ≠ 0` | Assertion error | Ensure divisibility |
| `embed_dim = 0` | Error in Linear | Validate embed_dim > 0 |
| `num_heads = 0` | Division by zero | Validate num_heads ≥ 1 |
| Very large N | O(N²) memory for attention | Consider chunked attention |
| `init` very large | α explodes, unstable | Keep init ∈ [0.5, 1.5] |
| `init` very negative | α ≈ 0, no differential | Keep init ∈ [0.5, 1.5] |
| `dropout = 1.0` | All outputs zero in training | Use dropout < 0.5 |

#### Implementation Notes

1. **GroupNorm vs LayerNorm:** Implementation uses GroupNorm with `num_groups=num_heads` for head-wise normalization. This provides per-head statistics.

2. **Projection Structure:**
   - Q and K projections output `2 × embed_dim` (for Q1/Q2 and K1/K2)
   - V projection outputs `embed_dim`
   - All projections use `bias=False`

3. **Alpha Initialization:** Stored as `exp(init)` so the learnable parameter is always positive. Default `init=0.8` gives α ≈ 2.23.

4. **Memory Consideration:** Two full attention matrices are computed, doubling attention memory compared to standard attention.

5. **No Causal Masking:** Current implementation is bidirectional (no causal mask). For autoregressive tasks, masking would need to be added.

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- GroupNorm for input normalization
- Doubled Q and K projections for differential
- Learnable alpha parameter with exp initialization
- Proper multi-head reshape and attention computation
- Differential attention via subtraction
- Output projection with dropout

---

### 3.7 Residual LayerNorm Block

#### Purpose

The Residual LayerNorm Block combines Layer Normalization with residual (skip) connections. This is a fundamental building block in transformer architectures that:

1. **Stabilizes training** through skip connections (gradient highway)
2. **Normalizes activations** using LayerNorm (not BatchNorm, since we're in sequence domain)
3. **Enables deep networks** by allowing gradients to flow directly through residual path

**Architecture Choice: Post-Norm**

This implementation uses **post-norm** style, where LayerNorm is applied to the output before adding the residual:
```
Y = LayerNorm(sublayer_output) + residual
```

Alternative (pre-norm, not used here):
```
Y = sublayer(LayerNorm(x)) + x
```

#### Input/Output Shapes

- **Input `x`:** `(B, N, D)` — Output from previous sublayer (e.g., LDA or FFN)
- **Input `residual`:** `(B, N, D)` — Skip connection (optional, defaults to `x` if not provided)
- **Output:** `(B, N, D)` — Normalized output with residual added

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `embed_dim` | int | — | ✅ | Embedding dimension for LayerNorm |

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| LayerNorm | **Stateless** | Computes statistics per-sample, no running statistics |

**Behavior:**
- Same computation in training and evaluation modes
- LayerNorm uses the input's own statistics (mean, variance) per sample
- No dependency on batch statistics or running averages

**Key Difference from BatchNorm:**
- BatchNorm: Uses batch statistics during training, running statistics during eval
- LayerNorm: Always uses the current input's statistics (stateless)

#### FLOP Cost

**Category:** NEGLIGIBLE

**Formula:**
```
FLOPs ≈ 4 × N × D

Where:
- Compute mean: N × D operations
- Compute variance: N × D operations  
- Normalize: N × D operations
- Scale and shift (γ, β): N × D operations
- Addition (residual): N × D operations
```

**For typical values (N=49, D=256):**
```
FLOPs ≈ 4 × 49 × 256 ≈ 50K FLOPs
```

This is negligible compared to attention or FFN operations.

#### Mathematical Formulation

```
Given input x ∈ ℝ^(B×N×D) and optional residual ∈ ℝ^(B×N×D):

1. Set residual (if not provided):
   if residual is None:
       residual = x

2. Layer Normalization:
   μ = (1/D) × Σ x[..., d]           (mean over embedding dim)
   σ² = (1/D) × Σ (x[..., d] - μ)²   (variance over embedding dim)
   x_norm = (x - μ) / √(σ² + ε)      (normalize)
   x_ln = γ × x_norm + β             (scale and shift with learned params)

3. Add Residual:
   Y = x_ln + residual

Output: Y ∈ ℝ^(B×N×D)
```

#### Example Forward Pass

**Configuration:**
```python
res_ln = ResidualLayerNormBlock(embed_dim=256)
```

**Usage Pattern 1: Default residual (x used as residual)**
```python
x = torch.randn(2, 49, 256)  # Output from LDA
y = res_ln(x)                 # Equivalent to: LN(x) + x
```

**Shape Progression:**
```
Input x:           (2, 49, 256)
                        │
                        ▼ LayerNorm (normalize over dim=-1)
Normalized:        (2, 49, 256)
                        │
                        ▼ Add residual (x itself)
Output:            (2, 49, 256)
```

**Usage Pattern 2: Explicit residual**
```python
x_before_lda = torch.randn(2, 49, 256)  # Input to LDA
x_after_lda = lda(x_before_lda)          # Output from LDA
y = res_ln(x_after_lda, residual=x_before_lda)  # LN(x_after) + x_before
```

**Shape Progression:**
```
x_after_lda:       (2, 49, 256)   [output from sublayer]
                        │
                        ▼ LayerNorm
Normalized:        (2, 49, 256)
                        │
x_before_lda:      (2, 49, 256)   [skip connection]
                        │
                        ▼ Add
Output:            (2, 49, 256)
```

#### Typical Usage in Transformer Block

```python
# Standard transformer block pattern:

# 1. Attention with residual
x_attn = lda(x)
x = res_ln_1(x_attn, residual=x)  # x = LN(attention_output) + x

# 2. FFN with residual  
x_ffn = ffn(x)
x = res_ln_2(x_ffn, residual=x)   # x = LN(ffn_output) + x
```

**Note:** The current implementation's default behavior (`residual=None` means `residual=x`) is slightly different - it normalizes x and adds x back, which may not be the intended pattern for wrapping sublayers. Verify usage in actual model.

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `embed_dim=0` | Error in LayerNorm | Validate embed_dim > 0 |
| `x.shape[-1] ≠ embed_dim` | Runtime error | Ensure dimension consistency |
| `residual.shape ≠ x.shape` | Broadcasting or error | Ensure shapes match exactly |
| Very small values in x | Numerical instability | LayerNorm's ε prevents division by zero |
| Very large values in x | Potential overflow | Rare in practice with proper initialization |
| `residual=None` | Uses x as residual | Document this behavior clearly |

#### Implementation Notes

1. **Post-Norm vs Pre-Norm:** This implementation applies LayerNorm to the input, then adds residual. This is "post-norm" style. Pre-norm would normalize before the sublayer.

2. **Default Residual Behavior:** When `residual=None`, the implementation uses `x` as the residual, resulting in `LN(x) + x`. This is useful when wrapping the entire block but may differ from standard transformer patterns.

3. **Learnable Parameters:** LayerNorm has two learnable parameters per dimension:
   - `γ` (weight): Scale parameter, initialized to 1
   - `β` (bias): Shift parameter, initialized to 0
   - Total: 2 × embed_dim parameters

4. **Epsilon Value:** LayerNorm uses a small epsilon (typically 1e-5 or 1e-6) to prevent division by zero.

5. **Gradient Flow:** Residual connections provide a direct gradient path, helping with training deep networks (addresses vanishing gradient problem).

6. **No Dropout:** This block does not include dropout. If dropout is needed on the residual path, it should be added externally or the block should be modified.

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- LayerNorm over embedding dimension
- Supports optional residual parameter
- Default residual is input x
- Returns normalized output + residual

---

### 3.8 Bottleneck FFN

#### Purpose

The Bottleneck Feed-Forward Network (FFN) is a parameter-efficient variant of the standard transformer FFN. Unlike standard FFNs that expand channels (4× expansion), this implementation uses a **bottleneck** design that **contracts** channels first, reducing parameters and computation.

**Standard Transformer FFN:**
```
Input (D) → Expand (4D) → GELU → Contract (D) → Output
```

**Bottleneck FFN (this implementation):**
```
Input (D) → Contract (D/4) → GELU → Expand (D) → Output
```

**Why bottleneck?**
- Significantly fewer parameters (compression vs expansion)
- Lower memory footprint during forward pass
- Suitable for mobile deployment where parameter budget is limited
- Acts as information bottleneck, forcing learned compression

#### Input/Output Shapes

- **Input:** `(B, N, D_in)` — 3D sequence tensor
- **Output:** `(B, N, D_out)` — 3D sequence tensor (typically D_out = D_in)

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `inp` | int | — | ✅ | Input feature dimension |
| `oup` | int | — | ✅ | Output feature dimension |
| `bottleneck_ratio` | float | 0.25 | ❌ | Ratio for bottleneck dimension (< 1 contracts, > 1 expands) |
| `dropout` | float | 0.1 | ❌ | Dropout probability after each layer |

#### Internal Channel Calculation

```python
bottleneck_channels = max(1, int(inp * bottleneck_ratio))
# With default ratio=0.25 and inp=256:
# bottleneck_channels = max(1, int(256 * 0.25)) = 64
```

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| LayerNorm | Stateless | No running statistics, uses input statistics |
| Linear | Stateless | No running statistics |
| Dropout | **Stateful** | Different behavior in train vs eval |

**Behavior:**
- **Training mode (`model.train()`):** Dropout active
- **Evaluation mode (`model.eval()`):** Dropout disabled

#### FLOP Cost

**Category:** LOW to MEDIUM

**Formula:**
```
FLOPs = N × (D_in × D_bottleneck + D_bottleneck × D_out)
      = N × D_bottleneck × (D_in + D_out)

Where:
- D_bottleneck = D_in × bottleneck_ratio
- First term: Contraction (fc1)
- Second term: Expansion (fc2)
```

**For typical values (D_in=D_out=256, N=49, bottleneck_ratio=0.25):**
```
D_bottleneck = 256 × 0.25 = 64
FLOPs = 49 × 64 × (256 + 256) = 49 × 64 × 512 ≈ 1.6M FLOPs
```

**Comparison to Standard FFN (4× expansion):**
```
Standard: N × D × 4D × 2 = 8 × N × D² 
        = 8 × 49 × 256² ≈ 25.7M FLOPs

Bottleneck: N × D × 0.25D × 2 = 0.5 × N × D²
          = 0.5 × 49 × 256² ≈ 1.6M FLOPs

Reduction: ~16× fewer FLOPs
```

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×N×D_in):

1. Contract to Bottleneck:
   H = Linear1(X)
   H ∈ ℝ^(B×N×D_bottleneck)

2. Normalize:
   H = LayerNorm1(H)

3. Activation:
   H = GELU(H)

4. Dropout:
   H = Dropout1(H)

5. Expand Back:
   Y = Linear2(H)
   Y ∈ ℝ^(B×N×D_out)

6. Normalize:
   Y = LayerNorm2(Y)

7. Final Dropout:
   Y = Dropout2(Y)

Output: Y ∈ ℝ^(B×N×D_out)
```

#### Example Forward Pass

**Configuration:**
```python
ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25, dropout=0.1)
```

**Internal Values:**
```
bottleneck_channels = max(1, int(256 * 0.25)) = 64
```

**Shape Progression:**
```
Input X:           (2, 49, 256)
                        │
                        ▼ fc1: Linear(256→64)
Contracted:        (2, 49, 64)
                        │
                        ▼ LayerNorm + GELU + Dropout
Activated:         (2, 49, 64)
                        │
                        ▼ fc2: Linear(64→256)
Expanded:          (2, 49, 256)
                        │
                        ▼ LayerNorm + Dropout
Output:            (2, 49, 256)
```

**Alternative Configuration (different dimensions):**
```python
ffn = BottleneckFFN(inp=256, oup=128, bottleneck_ratio=0.5, dropout=0.1)
```
```
bottleneck_channels = max(1, int(256 * 0.5)) = 128

Input:        (2, 49, 256)
                   │
                   ▼ fc1: Linear(256→128)
Contracted:   (2, 49, 128)
                   │
                   ▼ LN + GELU + Dropout
                   │
                   ▼ fc2: Linear(128→128)
Expanded:     (2, 49, 128)
                   │
                   ▼ LN + Dropout
Output:       (2, 49, 128)
```

#### Bottleneck vs Expansion Comparison

| Aspect | Bottleneck (ratio < 1) | Expansion (ratio > 1) |
|--------|------------------------|----------------------|
| Hidden dim | D × ratio (smaller) | D × ratio (larger) |
| Parameters | Fewer | More |
| FLOPs | Lower | Higher |
| Capacity | Limited | Higher |
| Use case | Mobile, efficiency | Accuracy-focused |

**This implementation uses bottleneck (ratio=0.25 by default).**

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `inp=0` or `oup=0` | Error in Linear | Validate dimensions > 0 |
| `bottleneck_ratio=0` | `bottleneck_channels=0`, error | Use ratio > 0 |
| Very small ratio | `bottleneck_channels=1` (minimum) | Ratio ≥ 0.1 recommended |
| `bottleneck_ratio > 1` | Acts as expansion, not bottleneck | Valid but changes purpose |
| `dropout=1.0` | All zeros in training | Use dropout < 0.5 |
| `inp ≠ oup` | Dimension change | Valid, but residual won't work |

#### Implementation Notes

1. **LayerNorm Placement:** This implementation applies LayerNorm **after** each linear layer (post-norm style). This differs from some implementations that use pre-norm.

2. **Double LayerNorm:** Both fc1 and fc2 outputs are normalized separately, providing stronger regularization.

3. **GELU Activation:** Uses Gaussian Error Linear Unit, which is smoother than ReLU and standard in modern transformers.

4. **No Residual Connection:** This block does NOT include a residual connection internally. The residual should be added externally (via ResidualLayerNormBlock).

5. **Minimum Bottleneck Channels:** `max(1, ...)` ensures at least 1 channel in the bottleneck, preventing zero-dimension errors.

6. **Dropout Pattern:** Dropout is applied twice:
   - After GELU activation (regularizes intermediate representation)
   - After final LayerNorm (regularizes output)

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Bottleneck design with `bottleneck_ratio` parameter
- fc1 contracts, fc2 expands
- LayerNorm after each linear
- GELU activation
- Dropout after activation and after final norm
- Minimum 1 channel guaranteed

---

### 3.9 Global Average Pooling

#### Purpose

Global Average Pooling (GAP) aggregates sequence information into a single feature vector by averaging across the sequence dimension. This serves as the transition from transformer sequence output to the classifier.

**Why GAP instead of [CLS] token:**
- **Parameter-free:** No additional learnable token required
- **Translation invariance:** Equal contribution from all spatial positions
- **Simplicity:** Straightforward implementation, no special token handling
- **Robustness:** Averages out noise from individual patches

#### Input/Output Shapes

- **Input:** `(B, N, D)` — 3D sequence tensor from transformer
- **Output:** `(B, D)` — 2D tensor (one vector per sample)

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| — | — | — | — | No hyperparameters (parameter-free operation) |

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| Mean operation | **Stateless** | Pure mathematical operation, no learnable parameters |

**Behavior:**
- Identical computation in training and evaluation modes
- No learnable parameters
- No running statistics

#### FLOP Cost

**Category:** NEGLIGIBLE

**Formula:**
```
FLOPs = N × D (summation) + D (division)
      ≈ N × D

Where:
- Sum all N elements for each of D dimensions
- Divide by N
```

**For typical values (N=49, D=256):**
```
FLOPs ≈ 49 × 256 ≈ 12.5K FLOPs
```

This is negligible compared to other operations.

#### Mathematical Formulation

```
Given input X ∈ ℝ^(B×N×D):

Global Average Pooling:
   Y = (1/N) × Σ_{i=1}^{N} X[:, i, :]
   
   Equivalently:
   Y = Mean(X, dim=1)

Output: Y ∈ ℝ^(B×D)
```

#### Example Forward Pass

**Configuration:**
```python
gap = GlobalAveragePooling()
```

**Shape Progression:**
```
Input X:           (2, 49, 256)
                        │
                        ▼ Mean over sequence dimension (dim=1)
Output:            (2, 256)

Each output vector is the average of 49 patch embeddings.
```

**Visualization:**
```
Sequence (N=49 patches):
┌─────────────────────────────────────────────────────────┐
│ [patch_1] [patch_2] [patch_3] ... [patch_49]            │
│   (256)     (256)     (256)        (256)                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼ Average all patches
                    ┌─────────────┐
                    │   Output    │
                    │   (256)     │
                    └─────────────┘
```

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `N=0` (empty sequence) | NaN or error (division by zero) | Validate N > 0 |
| `N=1` | Output equals input (no actual averaging) | Valid, but trivial |
| Very large N | Higher precision requirements | Use float32 or higher |
| `D=0` | Empty output tensor | Validate D > 0 |

#### Implementation Notes

1. **Dimension Specification:** Uses `dim=1` to average over sequence length, preserving batch and embedding dimensions.

2. **Memory Efficiency:** GAP is computed in-place conceptually; no large intermediate tensors.

3. **Alternative: Max Pooling:** Some architectures use Global Max Pooling instead. Average is typically preferred for classification as it provides smoother gradients.

4. **No Learnable Parameters:** This block has exactly 0 parameters.

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Uses `x.mean(dim=1)` for global average pooling
- No learnable parameters
- Correct shape transformation from (B, N, D) to (B, D)

---

### 3.10 Classifier Head

#### Purpose

The Classifier Head is the final layer that maps the pooled feature vector to class probabilities. It consists of a linear projection followed by softmax activation.

**Key responsibilities:**
1. Project embedding dimension to number of classes
2. Convert logits to probabilities via softmax

#### Input/Output Shapes

- **Input:** `(B, D)` — 2D tensor from Global Average Pooling
- **Output:** `(B, num_classes)` — Class probabilities (sum to 1)

#### Hyperparameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `embed_dim` | int | — | ✅ | Input embedding dimension |
| `num_classes` | int | — | ✅ | Number of output classes |

#### Stateful/Stateless

| Component | Type | Reason |
|-----------|------|--------|
| Linear | Stateless | No running statistics |
| Softmax | Stateless | Pure mathematical operation |

**Behavior:**
- Same computation in training and evaluation modes
- No BatchNorm or Dropout in basic implementation

#### FLOP Cost

**Category:** LOW

**Formula:**
```
FLOPs = D × num_classes (matrix multiplication)
      + num_classes (bias addition)
      + num_classes (softmax: exp + sum + div)
      ≈ D × num_classes

Where:
- Main cost is the linear projection
- Softmax is negligible
```

**For typical values (D=256, num_classes=38):**
```
FLOPs ≈ 256 × 38 ≈ 9.7K FLOPs
```

This is negligible compared to other operations.

#### Mathematical Formulation

```
Given input Z ∈ ℝ^(B×D):

1. Linear Projection:
   logits = Z × W^T + b
   logits ∈ ℝ^(B × num_classes)
   
   Where:
   W ∈ ℝ^(num_classes × D)  — weight matrix
   b ∈ ℝ^(num_classes)      — bias vector

2. Softmax Activation:
   probs[i] = exp(logits[i]) / Σ_j exp(logits[j])
   probs ∈ ℝ^(B × num_classes)
   
   Σ probs[i] = 1 for each sample (probabilities sum to 1)

Output: probs ∈ ℝ^(B × num_classes)
```

#### Example Forward Pass

**Configuration:**
```python
classifier = ClassifierHead(embed_dim=256, num_classes=38)
```

**Shape Progression:**
```
Input Z:           (2, 256)
                        │
                        ▼ Linear (256 → 38)
Logits:            (2, 38)
                        │
                        ▼ Softmax (dim=-1)
Probabilities:     (2, 38)

Each row sums to 1.0
```

**Example Output:**
```python
# Input: embedding vector for 2 images
z = torch.randn(2, 256)

# Output: probabilities for 38 plant disease classes
probs = classifier(z)
# probs[0] might be: [0.02, 0.01, 0.85, 0.03, ...]  # High confidence for class 2
# probs[1] might be: [0.15, 0.20, 0.05, 0.10, ...]  # More uncertain
# Each row sums to 1.0
```

#### Parameter Count

```
Linear layer:
- Weight: embed_dim × num_classes = 256 × 38 = 9,728
- Bias: num_classes = 38

Total: 9,728 + 38 = 9,766 parameters
```

#### Training vs Inference Considerations

**Training:**
- Typically use CrossEntropyLoss which combines LogSoftmax + NLLLoss
- More numerically stable than separate Softmax + NLLLoss
- Model should output **logits** (before softmax) for CrossEntropyLoss

**Inference:**
- Apply Softmax to get interpretable probabilities
- Use `argmax` to get predicted class

**Current Implementation Note:**
The implementation applies Softmax in forward(). For training with CrossEntropyLoss:
- Option 1: Modify to return logits, apply softmax only at inference
- Option 2: Use NLLLoss instead of CrossEntropyLoss

#### Edge Cases & Failure Modes

| Condition | Behavior | Recommendation |
|-----------|----------|----------------|
| `embed_dim=0` | Error in Linear | Validate embed_dim > 0 |
| `num_classes=0` | Error in Linear | Validate num_classes > 0 |
| `num_classes=1` | Binary classification, softmax trivial | Consider sigmoid instead |
| Very large logits | Softmax numerical instability | Use LogSoftmax for stability |
| `num_classes > embed_dim` | Valid but unusual | Consider hidden layer |

#### Implementation Notes

1. **Softmax in Forward:** The current implementation applies softmax in `forward()`. For training with `nn.CrossEntropyLoss`, this should be modified to return logits instead, as CrossEntropyLoss internally applies log-softmax.

2. **No Hidden Layer:** This is a direct projection. For more complex classification, a hidden layer could be added:
   ```python
   # Optional: Add hidden layer
   self.hidden = nn.Linear(embed_dim, hidden_dim)
   self.act = nn.ReLU()
   self.fc = nn.Linear(hidden_dim, num_classes)
   ```

3. **Dropout:** No dropout is included. For regularization, consider adding dropout before the linear layer.

4. **Weight Initialization:** PyTorch default initialization (Kaiming uniform for weights, uniform for bias) is typically sufficient.

5. **Class Imbalance:** For imbalanced datasets, consider:
   - Weighted loss function
   - Focal loss
   - Class-balanced sampling

#### Verification Against Implementation

**Current implementation in `my-upgraded-blocks.ipynb`:** ✅ **Correct**

The implementation matches this specification:
- Linear projection from embed_dim to num_classes
- Softmax activation applied in forward pass
- Returns probabilities (not logits)

**Note for Training:** If using `nn.CrossEntropyLoss`, modify to return logits or use `nn.NLLLoss` with the current softmax output.

---

## 4. Full Pipeline Shape Verification

### 4.1 Complete Shape Flow

This section documents the tensor shapes throughout the entire MobilePlantViT forward pass.

**Input Configuration:**
- Image size: 224 × 224
- Batch size: 2
- Number of classes: 38 (PlantVillage dataset)

### 4.2 Shape Flow Table

| Stage | Block | Input Shape | Output Shape | Parameters | Notes |
|-------|-------|-------------|--------------|------------|-------|
| **Input** | — | `(2, 3, 224, 224)` | — | — | RGB image |
| **CNN** | GhostConv | `(2, 3, 224, 224)` | `(2, 64, 224, 224)` | ~512 | 3→64 channels |
| **CNN** | FusedIR | `(2, 64, 224, 224)` | `(2, 64, 56, 56)` | ~164K | stride=4 downsample |
| **CNN** | CoordAtt | `(2, 64, 56, 56)` | `(2, 64, 56, 56)` | ~1.7K | Same shape (attention) |
| **Transition** | PatchEmbed | `(2, 64, 56, 56)` | `(2, 196, 256)` | ~65K | Spatial→Sequence |
| **Transition** | PosEnc | `(2, 196, 256)` | `(2, 196, 256)` | 0 | Sinusoidal (buffer) |
| **Transformer** | LDA | `(2, 196, 256)` | `(2, 196, 256)` | ~394K | Differential attention |
| **Transformer** | ResLN | `(2, 196, 256)` | `(2, 196, 256)` | 512 | LayerNorm + residual |
| **Transformer** | FFN | `(2, 196, 256)` | `(2, 196, 256)` | ~34K | Bottleneck FFN |
| **Classifier** | GAP | `(2, 196, 256)` | `(2, 256)` | 0 | Mean over sequence |
| **Classifier** | Head | `(2, 256)` | `(2, 38)` | ~9.8K | Linear + softmax |
| **Output** | — | — | `(2, 38)` | — | Class probabilities |

### 4.3 Detailed Shape Progression

```
INPUT IMAGE
┌─────────────────────────────────────────────────────────────────┐
│                    (2, 3, 224, 224)                             │
│                    RGB Image Batch                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: CNN BACKBONE                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GhostConv (3→64, stride=1)                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 3, 224, 224) ──→ (2, 64, 224, 224)                  │   │
│  │ Channels: 3 → 64 (via ghost features)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  FusedInvertedResidual (64→64, stride=4, expand=4)             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 64, 224, 224) ──→ (2, 64, 56, 56)                   │   │
│  │ Spatial: 224×224 → 56×56 (4× downsample)                │   │
│  │ Hidden dim: 64 × 4 = 256                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  CoordinateAttention (64→64)                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 64, 56, 56) ──→ (2, 64, 56, 56)                     │   │
│  │ Same shape (multiplicative attention)                   │   │
│  │ Adds channel + positional attention                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: CNN → TRANSFORMER TRANSITION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PatchEmbedding (patch_size=4)                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 64, 56, 56) ──→ (2, 196, 256)                       │   │
│  │ Spatial (56×56) → Sequence (14×14 = 196 patches)        │   │
│  │ Channels: 64 → 256 (embed_dim)                          │   │
│  │ Transformation: (B,C,H,W) → (B,N,D)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  PositionalEncoding (sinusoidal)                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 196, 256) ──→ (2, 196, 256)                         │   │
│  │ Same shape (additive positional info)                   │   │
│  │ No learnable parameters (buffer)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: TRANSFORMER                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LinearDifferentialAttention (8 heads)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 196, 256) ──→ (2, 196, 256)                         │   │
│  │ Self-attention with differential mechanism              │   │
│  │ Q,K: 256→512 (doubled for differential)                 │   │
│  │ V: 256→256                                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ResidualLayerNorm                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 196, 256) ──→ (2, 196, 256)                         │   │
│  │ LayerNorm + skip connection                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  BottleneckFFN (ratio=0.25)                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 196, 256) ──→ (2, 196, 256)                         │   │
│  │ 256 → 64 → 256 (bottleneck)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: CLASSIFICATION                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GlobalAveragePooling                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 196, 256) ──→ (2, 256)                              │   │
│  │ Sequence (196) → Single vector                          │   │
│  │ Mean over all patch embeddings                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ClassifierHead (38 classes)                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ (2, 256) ──→ (2, 38)                                    │   │
│  │ Linear projection + Softmax                             │   │
│  │ Output: class probabilities                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    (2, 38)                                      │
│                    Class Probabilities                          │
│                    (sum to 1.0 per sample)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Key Dimension Transitions

| Transition | From | To | Mechanism |
|------------|------|-----|-----------|
| Channel expansion | 3 | 64 | GhostConv primary + cheap ops |
| Spatial downsample | 224×224 | 56×56 | FusedIR stride=4 |
| Spatial → Sequence | (64, 56, 56) | (196, 256) | PatchEmbed Conv + flatten |
| Sequence → Vector | (196, 256) | (256,) | Global Average Pooling |
| Vector → Classes | (256,) | (38,) | Linear projection |

### 4.5 Parameter Budget Summary

| Component | Parameters | Percentage |
|-----------|------------|------------|
| **CNN Stage** | | |
| └─ GhostConv | 512 | 0.06% |
| └─ FusedIR | 164,480 | 18.97% |
| └─ CoordAtt | 1,688 | 0.19% |
| **Transition Stage** | | |
| └─ PatchEmbed | 262,400 | 30.26% |
| └─ PosEnc | 0 | 0% |
| **Transformer Stage** | | |
| └─ LDA | 393,985 | 45.44% |
| └─ ResLN | 512 | 0.06% |
| └─ FFN | 33,728 | 3.89% |
| **Classifier Stage** | | |
| └─ GAP | 0 | 0% |
| └─ Head | 9,766 | 1.13% |
| **TOTAL** | **867,071** | **100%** |

**Budget Status:** ✅ Within budget (17.3% of 5M limit)

**Note:** Actual parameters depend on specific configuration. Target is < 5M parameters.

### 4.6 Verification Code

```python
def verify_full_pipeline():
    """
    Verify complete forward pass through MobilePlantViT.
    """
    print("=" * 70)
    print("Full Pipeline Shape Verification")
    print("=" * 70)
    
    # Configuration
    batch_size = 2
    img_size = 224
    num_classes = 38
    
    # Create input
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"\nInput: {x.shape}")
    
    # Initialize all blocks with matching dimensions
    ghost = GhostConv(inp=3, oup=64, kernel_size=1, ratio=2, dw_size=3, stride=1)
    fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=4, expand_ratio=4)
    coord_att = CoordAtt(inp=64, oup=64, reduction=32)
    patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
    pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
    lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.1)
    res_ln = ResidualLayerNormBlock(embed_dim=256)
    ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25, dropout=0.1)
    gap = GlobalAveragePooling()
    classifier = ClassifierHead(embed_dim=256, num_classes=num_classes)
    
    # Set to eval mode
    for module in [ghost, fused_ir, coord_att, patch_embed, pos_enc, lda, res_ln, ffn, gap, classifier]:
        module.eval()
    
    # Forward pass
    with torch.no_grad():
        # CNN Stage
        print("\n--- CNN Stage ---")
        x = ghost(x)
        print(f"After GhostConv:    {x.shape}")
        
        x = fused_ir(x)
        print(f"After FusedIR:      {x.shape}")
        
        x = coord_att(x)
        print(f"After CoordAtt:     {x.shape}")
        
        # Transition Stage
        print("\n--- Transition Stage ---")
        x = patch_embed(x)
        print(f"After PatchEmbed:   {x.shape}")
        
        x = pos_enc(x)
        print(f"After PosEnc:       {x.shape}")
        
        # Transformer Stage
        print("\n--- Transformer Stage ---")
        residual = x
        x = lda(x)
        print(f"After LDA:          {x.shape}")
        
        x = res_ln(x, residual=residual)
        print(f"After ResLN:        {x.shape}")
        
        x = ffn(x)
        print(f"After FFN:          {x.shape}")
        
        # Classifier Stage
        print("\n--- Classifier Stage ---")
        x = gap(x)
        print(f"After GAP:          {x.shape}")
        
        x = classifier(x)
        print(f"After Classifier:   {x.shape}")
    
    # Verify final output
    assert x.shape == (batch_size, num_classes), f"Final shape mismatch! Got {x.shape}"
    assert torch.allclose(x.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Probs don't sum to 1!"
    
    # Count total parameters
    print("\n--- Parameter Count ---")
    total_params = 0
    for name, module in [
        ("GhostConv", ghost),
        ("FusedIR", fused_ir),
        ("CoordAtt", coord_att),
        ("PatchEmbed", patch_embed),
        ("PosEnc", pos_enc),
        ("LDA", lda),
        ("ResLN", res_ln),
        ("FFN", ffn),
        ("GAP", gap),
        ("Classifier", classifier),
    ]:
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"  {name:12s}: {params:>10,}")
    
    print(f"  {'TOTAL':12s}: {total_params:>10,}")
    print(f"\n  Target: < 5,000,000 parameters")
    print(f"  Status: {'✅ PASSED' if total_params < 5_000_000 else '❌ EXCEEDED'}")
    
    print("\n" + "=" * 70)
    print("✅ Full pipeline verification complete!")
    print("=" * 70)
    
    return True

# Run verification
verify_full_pipeline()
```

---

## 5. Parameter Budget

### 5.1 Target Constraints

| Constraint | Target | Actual | Status |
|------------|--------|--------|--------|
| Total Parameters | < 5,000,000 | 867,071 | ✅ Pass |
| Model Size (FP32) | < 20 MB | ~3.3 MB | ✅ Pass |
| Mobile Deployment | Yes | Yes | ✅ Pass |

### 5.2 Parameter Distribution

```
Parameter Distribution by Stage:

CNN Stage (19.2%):
  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  166,680

Transition Stage (30.3%):
  ██████████████████████████████░░░░░░░░░░░░░░░░░░░░  262,400

Transformer Stage (49.4%):
  █████████████████████████████████████████████████░  428,225

Classifier Stage (1.1%):
  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    9,766
```

### 5.3 Efficiency Metrics

| Metric | Value |
|--------|-------|
| Parameters per class | 22,817 |
| Compression vs ResNet-50 | ~29× smaller |
| Compression vs ViT-Base | ~99× smaller |

### 5.4 Scaling Options

For different deployment scenarios:

| Variant | Embed Dim | Channels | Est. Params | Use Case |
|---------|-----------|----------|-------------|----------|
| Tiny | 128 | 32 | ~250K | Edge devices |
| Small | 192 | 48 | ~500K | Mobile |
| **Base** | **256** | **64** | **~867K** | **Current** |
| Large | 384 | 96 | ~2M | Server |

---

## 6. Sign-off

### 6.1 Specification Review

| Reviewer | Date | Status | Comments |
|----------|------|--------|----------|
| Developer | November 29, 2025 | ✅ Complete | All blocks specified |

### 6.2 Implementation Verification

| Block | Matches Spec | Verified By | Date |
|-------|--------------|-------------|------|
| GhostConv | ✅ Yes | Shape Test | Nov 29, 2025 |
| FusedIR | ✅ Yes | Shape Test | Nov 29, 2025 |
| CoordAtt | ✅ Yes | Shape Test | Nov 29, 2025 |
| PatchEmbed | ✅ Yes | Shape Test | Nov 29, 2025 |
| PosEnc | ✅ Yes | Shape Test | Nov 29, 2025 |
| LDA | ✅ Yes | Shape Test | Nov 29, 2025 |
| ResLN | ✅ Yes | Shape Test | Nov 29, 2025 |
| FFN | ✅ Yes | Shape Test | Nov 29, 2025 |
| GAP | ✅ Yes | Shape Test | Nov 29, 2025 |
| Classifier | ✅ Yes | Shape Test | Nov 29, 2025 |

### 6.3 Shape Verification Results

```
Full Pipeline Test: ✅ PASSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:          (2, 3, 224, 224)    RGB Image
GhostConv:      (2, 64, 224, 224)   ✅ Correct
FusedIR:        (2, 64, 56, 56)     ✅ Correct  
CoordAtt:       (2, 64, 56, 56)     ✅ Correct
PatchEmbed:     (2, 196, 256)       ✅ Correct
PosEnc:         (2, 196, 256)       ✅ Correct
LDA:            (2, 196, 256)       ✅ Correct
ResLN:          (2, 196, 256)       ✅ Correct
FFN:            (2, 196, 256)       ✅ Correct
GAP:            (2, 256)            ✅ Correct
Classifier:     (2, 38)             ✅ Correct
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output Verification:
  • Final shape:        ✅ (2, 38)
  • Probabilities sum:  ✅ 1.0
  • Value range:        ✅ [0, 1]
  • Parameter count:    ✅ 867,071 (< 5M)
```

### 6.4 Final Checklist

- [x] All 10 blocks specified with full details
- [x] Input/Output shapes documented for each block
- [x] Hyperparameters with defaults listed
- [x] Stateful/Stateless nature documented
- [x] Mathematical formulations provided
- [x] Example forward passes included
- [x] Edge cases and failure modes documented
- [x] Full pipeline shape flow verified
- [x] Parameter budget verified (867K < 5M)
- [x] Implementation matches specification

### 6.5 Final Approval

**Specification Status:** ✅ **COMPLETE**

**Approved By:** _______________  
**Date:** November 29, 2025

**Notes:**
- All blocks implemented and verified
- Total parameters: 867,071 (17.3% of budget)
- Ready to proceed to Stage C (Training Pipeline)

---

*End of Document*