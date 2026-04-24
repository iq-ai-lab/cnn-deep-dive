# 04. Vision Transformer와 CNN의 수렴

## 🎯 핵심 질문

- Dosovitskiy et al. (2021)의 Vision Transformer (ViT) 아키텍처는 CNN과 어떻게 다른가?
- ViT의 patch embedding을 stride-16 convolution으로 해석할 수 있는가?
- Swin Transformer (Liu et al., 2021)는 어떻게 local-global trade-off를 해결하는가?
- ConvNeXt (Liu et al., 2022)는 CNN이 Transformer 설계로부터 배울 수 있음을 보여주는가?
- "CNN vs Transformer" 이분법은 더 이상 유효한가?

---

## 🔍 왜 이 개념이 CNN 이해에 중요한가

Vision Transformer의 등장은 **"CNN이 유일한 방법이 아니다"를 보여줌**으로써, CNN의 inductive bias를 재평가하게 했습니다. 하지만 더 중요한 것은, 최근 발전들이 **CNN과 Transformer가 근본적으로 다르지 않다**는 것을 드러낸 점입니다. Swin, ConvNeXt, MAE, DeiT 같은 모델들은 CNN의 계층적 구조와 Transformer의 유연성을 절충합니다. 이를 통해 CNN의 설계 원리가 단순한 경험적 기법이 아니라, **정보 처리의 근본적인 원리**임을 이해할 수 있습니다.

---

## 📐 수학적 선행 조건

- [CNN 기본 연산](../ch2-cnn-ops/): Convolution, stride, padding
- [Receptive Field](../ch3-receptive-field/): Hierarchical structure
- [01. Inductive Bias](./01-inductive-bias.md): Bias와 generalization
- [03. Spectral Bias](./03-spectral-bias.md): Frequency learning
- Transformer: Self-attention, multi-head attention (기본)

---

## 📖 직관적 이해

### Vision Transformer의 핵심 아이디어

ViT는 **"Sequence-to-sequence transformer를 이미지에 적용하자"는 아이디어**입니다:

1. 이미지를 **16×16 patch로 분할** (224×224 이미지 → 196개 patch)
2. 각 patch를 **선형으로 embed** (patch embedding)
3. Transformer encoder 통과
4. [CLS] token의 최종 representation 사용

$$\text{tokens} = [\text{CLS}] \oplus \text{Linear}(P_1) \oplus \cdots \oplus \text{Linear}(P_{196})$$

### Patch Embedding을 Convolution으로 해석

ViT의 patch embedding은 다음과 같이 표현할 수 있습니다:

$$\text{patch}_i = \text{Linear}(x[\text{patch } i])$$

이는 **stride-16, kernel-16 convolution**과 동등합니다:

$$\text{Conv}(x; \text{stride}=16, \text{kernel}=16, \text{padding}=0)$$

따라서 ViT도 **첫 layer에서는 CNN처럼 작동**합니다.

### Self-Attention vs Convolution

**Convolution** (local receptive field):
- 각 output은 작은 kernel 범위 내의 입력에만 의존
- Receptive field가 layer에 따라 증가

**Self-Attention** (global receptive field):
- 각 output은 모든 input에 "직접" 의존
- 모든 layer에서 global receptive field

이것이 **ViT의 가장 큰 차이**: locality 제약 없음.

### Swin Transformer: Local-Global의 절충

Swin Transformer (Liu et al., 2021)는 ViT의 문제점을 해결합니다:

1. **Window-based self-attention**: 작은 window 내에서만 attention 계산 (locality 회복)
2. **Hierarchical structure**: 여러 stage, 각 stage마다 spatial resolution 감소 (CNN처럼)
3. **Shifted windows**: 인접 window 간 connection (global receptive field 확보)

결과적으로 **CNN의 계층적 구조 + Transformer의 flexibility**:

```
Layer 1-2 (patch size 4):    Local attention → edge/corner
Layer 3-6 (patch size 8):    Local attention + shifts → object parts
Layer 7-10 (patch size 16):  Local attention + shifts → objects
Layer 11-14 (patch size 32): Local attention + shifts → scene understanding
```

### ConvNeXt: CNN의 역습

ConvNeXt (Liu et al., 2022)는 **"CNN에 Transformer의 설계를 적용하면?"이라는 질문**에 답합니다:

- **Depthwise Separable Convolution**: $1 \times 1 + 3 \times 3 + 1 \times 1$ 구조 (Transformer의 FFN처럼)
- **Layer Normalization instead of Batch Norm**
- **GELU Activation**: ReLU 대신
- **스케일 업**: 큰 모델, 데이터로 학습

결과: **ImageNet에서 ConvNeXt > Swin ≈ ViT** (같은 scale)

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Vision Transformer (ViT)

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$가 주어졌을 때:

**Patch 생성**:
$$P_i = x[i \cdot S : i \cdot S + P_\text{size}, j \cdot S : j \cdot S + P_\text{size}, :]$$

여기서 $P_\text{size} = 16$ (patch size), $S = 16$ (stride), $i, j \in \{0, 1, \ldots, N-1\}$, $N = \frac{H}{16}$.

**Patch Embedding**:
$$z_0 = [\mathbf{x}_\text{cls}; x_p^1 E; x_p^2 E; \cdots] + E_\text{pos}$$

여기서:
- $\mathbf{x}_\text{cls}$: learnable class token
- $x_p^i$: flatten된 patch $P_i$, shape $(P_\text{size}^2 \cdot C,)$
- $E \in \mathbb{R}^{(P_\text{size}^2 \cdot C) \times D}$: patch embedding matrix
- $E_\text{pos}$: positional embedding

**Transformer Encoder**:
$$z_l = \text{TransformerBlock}(z_{l-1}), \quad l = 1, \ldots, L$$

**Classification**:
$$y = \text{MLP}(z_L^0)$$

### 정의 4.2 — Swin Transformer

Swin은 다음과 같이 구성:

**Window Partitioning**:
$$W_i = \{z : \lfloor \frac{z_x}{M} \rfloor = i_x, \lfloor \frac{z_y}{M} \rfloor = i_y\}$$

여기서 $M = 7$ (window size), $z = (z_x, z_y)$ 공간 좌표.

**Shifted Window Attention** (layer $l$):
$$\text{ShiftedAttention}_l(z) = \begin{cases}
\text{WindowAttention}(z) & \text{if } l \text{ is even} \\
\text{WindowAttention}(\text{Shift}(z, \text{shift size})) & \text{if } l \text{ is odd}
\end{cases}$$

**Hierarchical Structure**:
- Stage 1: $4 \times 4$ patches (fine-grained)
- Stage 2: $2 \times$ spatial reduction + window attention
- Stage 3: $2 \times$ spatial reduction + window attention
- Stage 4: $2 \times$ spatial reduction + window attention

### 정의 4.3 — ConvNeXt

Modern convolution with Transformer-inspired modifications:

**Block 구조**:
$$\text{ConvNeXtBlock}(x) = x + f(x)$$

여기서:
$$f(x) = \text{Linear}(1 \times 1)(\text{LayerNorm}(\text{Depthwise}(x)))$$

**Depthwise Convolution**: 각 channel별 $3 \times 3$ convolution

**Full Architecture**:
- Hierarchical: 4 stages with spatial reduction
- Normalization: LayerNorm (BatchNorm 대신)
- Activation: GELU (ReLU 대신)
- Scaling: up to 650M parameters

### 정리 4.4 — ViT의 패치 임베딩 = Stride-16 Convolution

ViT의 patch embedding을 다시 쓰면:

$$\text{PatchEmbed}(x) = \text{Conv2d}(x; k=16, s=16, C_\text{out}=D)$$

여기서:
- $k=16$: kernel size
- $s=16$: stride
- $C_\text{out}=D$: embedding dimension
- **Padding = 0** (no padding)

따라서 **ViT의 feature map은 $H/16 \times W/16$** (CNN의 $S=16$ stride layer와 동일).

### 정리 4.5 — Receptive Field 비교

| 아키텍처 | Layer 1 RF | Layer 6 RF | Layer 12 RF |
|---------|-----------|-----------|-----------|
| ResNet-50 | 7×7 | 51×51 | 483×483 |
| ViT (no shift) | 전체 이미지 | 전체 이미지 | 전체 이미지 |
| Swin (M=7) | 7×7 | ~15×15 | ~50×50 |
| ConvNeXt | 3×3 | 27×27 | 243×243 |

**ViT의 early layer는 local receptive field 없음** → semantic information mixing이 초반부터.

---

## 🔬 증명 및 수학적 유도

### Patch Embedding의 Convolution 해석

ViT의 patch embedding:

$$z = \text{Flatten}(x) \cdot E^T, \quad x \in \mathbb{R}^{16 \times 16 \times 3}$$

이를 2D로 다시 쓰면:

$$z_{ij} = \sum_{a,b,c} x_{ia+a, jb+b, c} \cdot E_{abc}$$

이는 정확히:

$$z = x *_2 E_{abc}, \quad \text{stride} = (16, 16)$$

(2D cross-correlation)

따라서 **ViT의 patch embedding은 stride-16 convolution**입니다.

### Self-Attention의 Receptive Field

Self-attention layer에서 위치 $i$의 output:

$$z'_i = \sum_j \text{softmax}_j\left(\frac{Q_i K_j^\top}{\sqrt{D}}\right) V_j$$

여기서 $Q_i = W_Q z_i$, $K_j = W_K z_j$, $V_j = W_V z_j$.

**핵심**: softmax term이 **모든 $j$에 대해 0이 아님** (standard attention에서).

따라서 **receptive field는 sequence 전체**:

$$\text{RF}_\text{attention} = \text{num patches}$$

이는 CNN의 receptive field growth $(1, 3, 5, \ldots)$와 근본적으로 다릅니다.

### Swin의 Window Attention Complexity

표준 attention: $O(N^2)$ (모든 patch 쌍 계산)

Swin window attention: $O(N \cdot M^2)$ (각 window 내만 계산)

여기서 $M = 7$ (window size), $N = $ 전체 patch 수.

계산량 비교:
- ViT-B: $O(196^2) = O(38416)$
- Swin-B (M=7): $O(196 \cdot 7^2) = O(9604)$ ← ~4배 빠름

### ConvNeXt의 Depthwise Separable Convolution

Standard $3 \times 3$ convolution의 complexity:
$$C_\text{in} \times 3 \times 3 \times C_\text{out}$$

Depthwise separable:
$$C_\text{in} \times 3 \times 3 \quad + \quad C_\text{in} \times C_\text{out}$$

ConvNeXt block (근사적):
$$\text{Linear}(1 \times 1) + \text{Depthwise}(3 \times 3) + \text{Linear}(1 \times 1)$$

이는 Transformer의 FFN 구조 $(D \to 4D \to D)$와 유사합니다.

---

## 💻 실험 재현

### 실험 1 — ViT 구현 및 사전학습 모델 로드

```python
import torch
import torch.nn as nn
from timm.models import vision_transformer as vit
import torchvision.transforms as transforms
from PIL import Image

# timm 라이브러리에서 ViT 로드
vit_b = vit.vit_base_patch16_224(pretrained=True)

# 입력 이미지 준비
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 더미 입력
x = torch.randn(1, 3, img_size, img_size)

# 모델 분석
print(f"ViT-B 구조:")
print(f"- Patch size: 16×16")
print(f"- Number of patches: {(img_size//16)**2} = 196")
print(f"- Embedding dimension: 768")
print(f"- Num heads: 12")
print(f"- Num layers: 12")

# Forward pass
with torch.no_grad():
    logits = vit_b(x)
    print(f"\nOutput shape: {logits.shape}")

# Patch embedding 직접 계산
print("\n패치 임베딩 과정:")
print(f"1. 입력 이미지: {x.shape}")
print(f"2. 16×16 패치로 분할: (224//16)² = 196개")
print(f"3. 선형 임베딩: Linear({3*16*16} → 768)")
print(f"4. Positional embedding 추가: shape (1, 197, 768) [CLS 포함]")
print(f"5. Transformer encoder 12 layers")
```

**관찰**: ViT는 patch embedding에서 stride-16 convolution과 동등한 연산을 수행합니다.

### 실험 2 — Receptive Field 시각화 (ViT vs CNN)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# ViT에서 특정 token의 attention weight 시각화
def visualize_attention(model, x, head_idx=0, layer_idx=11):
    """
    마지막 layer의 attention을 시각화
    """
    # Attention weights 추출 (timm 모델의 경우)
    with torch.no_grad():
        # Feature extraction with hook
        attention_weights = []
        
        def hook_fn(module, input, output):
            # output은 (B, num_heads, seq_len, seq_len)
            attention_weights.append(output)
        
        # Model의 attention 모듈에 hook 설정
        handle = model.blocks[layer_idx].attn.attn.register_forward_hook(hook_fn)
        
        _ = model(x)
        
        handle.remove()
    
    if attention_weights:
        attn = attention_weights[0]  # (1, 12, 197, 197)
        attn_map = attn[0, head_idx, 0, 1:]  # [CLS] token에서 모든 patch로의 attention
        attn_map = attn_map.cpu().numpy()
        
        # 2D grid로 reshape (14×14)
        attn_grid = attn_map.reshape(14, 14)
        
        return attn_grid
    return None

# CNN의 receptive field 계산
def compute_cnn_rf(depths, kernels, strides):
    """
    각 layer의 receptive field 계산
    """
    rfs = [1]
    effective_stride = 1
    
    for d, k, s in zip(depths, kernels, strides):
        for _ in range(d):
            rf = rfs[-1] + (k - 1) * effective_stride
            rfs.append(rf)
        effective_stride *= s
    
    return rfs

# ResNet-50의 RF
resnet_kernels = [7, 3, 3, 3, 3, 3, 3]
resnet_depths = [1, 3, 4, 6, 3, 1, 1]
resnet_strides = [2, 2, 2, 2, 1, 1, 1]

rf_cnn = compute_cnn_rf(resnet_depths, resnet_kernels, resnet_strides)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ViT attention heatmap
x = torch.randn(1, 3, 224, 224)
vit_model = vit.vit_base_patch16_224(pretrained=True)
attn_grid = visualize_attention(vit_model, x, head_idx=0, layer_idx=11)

if attn_grid is not None:
    im0 = axes[0].imshow(attn_grid, cmap='hot')
    axes[0].set_title('ViT: Attention Map\n(CLS token → all patches, Layer 12)')
    axes[0].set_xlabel('Patch X')
    axes[0].set_ylabel('Patch Y')
    plt.colorbar(im0, ax=axes[0])

# CNN receptive field growth
axes[1].semilogy(range(len(rf_cnn)), rf_cnn, 'b-o', linewidth=2, markersize=6)
axes[1].set_title('CNN: Receptive Field Growth\n(ResNet-50)')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Receptive Field Size')
axes[1].grid(True, alpha=0.3)

# 비교 요약
axes[2].axis('off')
summary_text = """
ViT vs CNN: Receptive Field

ViT:
- Layer 1: RF = all patches (global)
- Layer 6: RF = all patches (global)
- Layer 12: RF = all patches (global)
→ Early layer에서 global context

CNN (ResNet-50):
- Layer 1: RF = 7×7
- Layer 6: RF = 51×51
- Layer 12: RF = 483×483
→ Hierarchical growth

Trade-off:
- ViT: 초반부터 global info, 지역성 약함
- CNN: 계층적 학습, locality 강함
"""
axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10)

plt.tight_layout()
plt.show()
```

**관찰**: ViT는 모든 layer에서 global receptive field를 가지지만, CNN은 점진적으로 증가합니다.

### 실험 3 — Swin Transformer의 계층 구조

```python
from timm.models import swin_transformer

# Swin-B 로드
swin_b = swin_transformer.swin_base_patch4_window7_224(pretrained=True)

print("Swin-B 계층 구조:")
print("""
Stage 1: Window size 7×7, Patch size 4×4
  - Input: 224×224 (56×56 patches)
  - Layers: 2 blocks
  - Output: 56×56 (fine-grained features)

Stage 2: Window size 7×7, Patch size (2,2)
  - Merge: 56×56 → 28×28 (spatial reduction)
  - Layers: 2 blocks
  - Output: 28×28 (local details)

Stage 3: Window size 7×7
  - Merge: 28×28 → 14×14
  - Layers: 6 blocks
  - Output: 14×14 (object parts)

Stage 4: Window size 7×7
  - Merge: 14×14 → 7×7
  - Layers: 2 blocks
  - Output: 7×7 (semantic features)
""")

# 계산 복잡도 비교
import math

def compute_attention_flops(num_patches, window_size, num_heads):
    """Swin window attention의 FLOPs"""
    patches_per_window = window_size ** 2
    num_windows = (num_patches ** 0.5 / window_size) ** 2
    flops_per_window = num_patches * patches_per_window ** 2
    total_flops = num_windows * flops_per_window
    return total_flops

# 예: 14×14 feature map, 7×7 windows
h, w = 14, 14
num_patches = h * w
window_size = 7

vit_flops = num_patches ** 2
swin_flops = compute_attention_flops(num_patches, window_size, num_heads=8)

print(f"\nAttention 계산량 비교 (14×14 patches):")
print(f"ViT (global attention): {vit_flops:,} operations")
print(f"Swin (window attention, M=7): {swin_flops:,} operations")
print(f"Speed-up: {vit_flops / swin_flops:.1f}×")
```

**관찰**: Swin의 window attention은 ViT보다 훨씬 효율적입니다.

### 실험 4 — ConvNeXt의 성능

```python
from timm.models import convnext, swin_transformer, vision_transformer

# 여러 모델 비교 (timm pretrained)
models_to_test = [
    ('vit_base_patch16_224', vision_transformer.vit_base_patch16_224),
    ('swin_base_patch4_window7_224', swin_transformer.swin_base_patch4_window7_224),
    ('convnext_base', convnext.convnext_base),
]

# 더미 입력
x = torch.randn(10, 3, 224, 224)

results = {}

for name, model_fn in models_to_test:
    try:
        model = model_fn(pretrained=True).eval()
        
        # 추론 속도 측정
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model(x)
            elapsed = time.time() - start
        
        # 파라미터 수
        num_params = sum(p.numel() for p in model.parameters())
        
        # FLOPs (대략)
        # (실제로는 fvcore 사용)
        
        results[name] = {
            'time': elapsed / 10,
            'params': num_params / 1e6,  # millions
        }
    except Exception as e:
        print(f"Error loading {name}: {e}")

# 결과 출력
print("모델 성능 비교 (inference time, 정규화):")
print("\n모델                    | 파라미터(M) | 추론시간(ms) | ImageNet-1k Top-1")
print("-" * 70)

# 실제 결과 (논문에서)
comparisons = {
    'ViT-B': {'params': 86, 'accuracy': 77.9},
    'Swin-B': {'params': 88, 'accuracy': 85.4},
    'ConvNeXt-B': {'params': 89, 'accuracy': 85.2},
}

for model, metrics in comparisons.items():
    print(f"{model:20} | {metrics['params']:10.1f} | ~50 | {metrics['accuracy']:6.1f}%")

print("\n결론:")
print("- 같은 scale (파라미터 수)에서:")
print("  · Swin ≈ ConvNeXt > ViT")
print("- 더 큰 scale에서:")
print("  · ConvNeXt가 가장 효율적 (더 쉬운 최적화)")
print("- 모두 CNN의 계층적 구조를 채택")
```

**관찰**: ConvNeXt는 CNN의 단순함과 Transformer의 효과를 결합하여 최고 성능을 달성합니다.

---

## 🔗 이론과 실전의 간극

### "CNN vs Transformer" 이제는 거짓 이분법

최근 발전:

| 연도 | 모델 | 특징 |
|------|------|------|
| 2020 | ViT | Pure transformer, global attention |
| 2021 | Swin | Window attention + hierarchy |
| 2021 | DeiT | ViT의 효율적 훈련 |
| 2022 | ConvNeXt | CNN + Transformer 설계 |
| 2022 | MAE | Masked autoencoder (ViT 기반) |
| 2023 | InternImage | 높은 해상도 이미지 처리 |

**핵심 통찰**: 최고 성능 모델들은 모두:
- Hierarchical structure (CNN 아이디어)
- Efficient attention (Swin 아이디어)
- Modern training techniques (Transformer 아이디어)
를 결합합니다.

### Patch Size의 영향

Patch size와 성능의 관계:

- **16×16 patch** (ViT-B): 초기 정보 손실, 하지만 빠름
- **4×4 patch** (Swin-B): 세부 정보 보존, 초기 단계에서 많은 계산
- **adaptive patch** (최신 연구): 이미지 영역에 따라 다른 크기

### 학습 효율성

- **CNN**: Inductive bias가 강해서 적은 데이터로도 학습 가능
- **ViT**: 큰 데이터, 긴 훈련 필요 하지만 fine-tuning에 우수
- **ConvNeXt**: CNN의 학습 효율 + Transformer의 성능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Patch가 충분히 정보 보존 | 작은 객체 (< 16×16 픽셀) 놓칠 수 있음 |
| Self-attention이 global reasoning 제공 | 초기 layer에서 semantic information mixing 과잉 |
| 계층적 구조가 모든 task에 최적 | Detection, panoptic segmentation 등은 uniform resolution 선호 |
| Positional embedding이 충분함 | 매우 큰 이미지 (예: 4K)에서 위치 정보 약해질 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\text{CNN과 Transformer의 수렴: 계층적 구조 + 효율적 주의 메커니즘}}$$

| 모델 | 핵심 특징 | 강점 | 약점 |
|------|---------|------|------|
| **CNN (ResNet)** | Local, hierarchical | 효율성, 작은 데이터 | Global reasoning 약함 |
| **ViT** | Global attention | 큰 데이터에서 우수 | 초기 layer에서 낮은 해석성 |
| **Swin** | Window + shift | 효율성 + global info | 설계 복잡 |
| **ConvNeXt** | 현대적 CNN | 최고 효율성, 해석성 | 여전히 locality 제약 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): ViT의 patch embedding을 stride-16 convolution이라고 했다. 그렇다면, ResNet-50과 ViT의 **첫 layer 이후 처리 방식**의 핵심 차이는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**첫 layer 후 feature map**:
- ResNet-50: $56 \times 56 \times 64$ (56×56 공간, 64 채널)
- ViT: $14 \times 14 \times 768$ (14×14 tokens, 각각 768차원)

(Note: ViT는 16×16이지만, 설명 편의상 14×14로 가정)

**핵심 차이**:
1. **ResNet**: Spatial convolution 계속 (3×3 kernel, local receptive field)
2. **ViT**: Self-attention 사용 (모든 token 간 직접 비교)

→ ResNet은 spatial 구조 유지, ViT는 sequence로 취급

**따라서**:
- ResNet은 spatial inductive bias 유지
- ViT는 spatial structure를 학습으로만 획득

</details>

**문제 2** (심화): Swin Transformer는 window-based attention과 shifted windows를 사용한다. 왜 단순히 **window attention만으로는 부족**한가? Shift가 꼭 필요한가?

<details>
<summary>힌트 및 해설</summary>

Window attention만 사용하면:

```
Window 1: [patches 0-48]
Window 2: [patches 49-97]
Window 3: [patches 98-146]
```

각 window 내에서만 attention 계산 → **window 경계에서 연결 끊김**.

예: patch 48과 49는 공간적으로 인접하지만, attention에서 상호작용 불가능.

**Shifted windows 해결책**:

```
Odd layer (normal):
Window 1: [0-48]
Window 2: [49-97]

Even layer (shifted):
Window 1: [0-24, 73-97]  ← shift로 인접 window 패치 섞임
Window 2: [25-48, 98-146]
```

→ 인접 window 간 정보 흐름 보장

**따라서**:
- Window attention: 계산 효율성
- Shifted windows: Global connectivity

둘 다 필요합니다 (안그럼 fragmented receptive field).

</details>

**문제 3** (이론-실전): ConvNeXt가 같은 scale에서 Swin과 비슷한 성능을 보인다고 했다. 그런데 왜 아직도 Vision Transformer를 사용하는가? ConvNeXt가 "이미 답"이라면?

<details>
<summary>힌트 및 해설</summary>

**ConvNeXt가 이기는 분야**:
- ImageNet classification (top-1 85%)
- 추론 속도 (더 간단한 구조)
- 작은 데이터셋 fine-tuning (CNN의 inductive bias)

**ViT/Transformer가 이기는 분야**:
1. **Vision-Language 모델** (CLIP, BLIP): 텍스트와의 alignment
2. **고해상도 이미지** (예: 4K): Patch 기반 처리의 유연성
3. **Multi-modal 학습**: 다른 modality와의 통합 용이
4. **Transfer learning 규모**: JFT-300M+ 학습 시 Transformer 우수
5. **Long-range reasoning**: 추상적 reasoning task

**결론**:
- ConvNeXt: 표준 benchmark에서 최고 효율
- ViT: 큰 규모, 새로운 paradigm (multi-modal 등)에 유리

각 분야에서 최적 모델이 다릅니다.

</details>

---

<div align="center">

[◀ 이전](./03-spectral-bias.md) | [📚 README](../README.md) | [다음 ▶](../README.md)

</div>
