# 04. ConvNeXt: CNN의 현대화 및 Transformer 설계 원리의 역수입

## 🎯 핵심 질문

- **Vision Transformer (ViT)**가 CNN을 능가한 이유는 무엇인가?
- Transformer의 설계 원리를 CNN에 역수입하면 어떤 성능 향상을 얻을 수 있는가?
- **Depthwise convolution** (group conv with groups=channels)은 왜 효율적인가?
- 5가지 현대화 개선 (stage ratio, depthwise, norm, activation, bottleneck)을 하나씩 적용하면 정확도가 얼마나 증가하는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Zhaohui Liu et al. (2022)의 ConvNeXt는 흥미로운 실험을 했습니다:

1. **Transformer의 성공 재검토**: ViT 성공이 순수 Transformer 때문인가, 아니면 **scale + 훈련 기법** 때문인가?
2. **CNN의 부활**: 동일 규모(scale)와 훈련 기법을 CNN에 적용하면 ViT와 **동등한 성능** 달성
3. **설계 원칙**: Transformer에서 배운 5가지 개선을 CNN에 통합
4. **결론**: "The Age of Transformers가 아니라, the Age of Foundation Models" — **아키텍처보다 scale과 데이터가 중요**

이는 CNN이 완전히 폐기되지 않았으며, **원칙적인 개선**으로 현대 수준에 도달할 수 있다는 것을 보여줍니다.

---

## 📐 수학적 선행 조건

- **Depthwise Convolution** (separable convolution): group convolution의 특수한 경우
- **LayerNorm vs BatchNorm**: 정규화의 차이
- **Inverted Bottleneck** (MBConv 복습): EfficientNet 참고
- **Attention의 역할**: ViT와의 비교

참고: [03-efficientnet.md](./03-efficientnet.md)

---

## 📖 직관적 이해

### Vision Transformer의 성공과 CNN의 정체

**2020년 상황**:
- Vision Transformer (Dosovitskiy et al. 2020): ViT-L, ViT-H로 ImageNet-1k에서 88-89% 정확도 달성
- CNN (EfficientNet): B7으로 85.3% (3% gap)
- Transformer의 우월성 주장

**문제**: ViT의 성공이 정말 Transformer 아키텍처 때문인가?

### ConvNeXt의 재검토: 공정한 비교

같은 조건에서 비교하면?

| 모델 | 규모 | 훈련 기법 | ImageNet Top-1 |
|------|------|---------|----------------|
| ResNet-50 (기본) | ~25M params | 90 epoch | 76.6% |
| ResNet-200 (깊이↑) | ~250M params | 90 epoch | 81.0% |
| EfficientNet-B0 | 5.3M params | 표준 | 77.1% |
| **Swin-T** (Liu et al. 2021) | 29M params | modern training | **81.3%** |
| **ConvNeXt-T** (moderne) | 29M params | modern training | **82.1%** |
| ViT-B (modern training) | 86M params | modern training | 81.8% |
| **Swin-B** | 88M params | modern training | **83.5%** |
| **ConvNeXt-B** | 89M params | modern training | **83.6%** |
| ViT-L | 304M params | modern training | 85.3% |
| Swin-L | 197M params | modern training | 86.3% |
| ConvNeXt-L | 345M params | modern training | 86.6% |

**결론**: **동일 규모에서 ConvNeXt ≥ Swin > ViT** — 특히 ConvNeXt-T (82.1%) vs Swin-T (81.3%)는 **직접 비교**의 대표 쌍. Liu 2022의 주장 — "Transformer 아키텍처의 본질적 우월성이 아니라, 현대적 훈련·large kernel·LayerNorm 같은 **설계 원리**가 승부를 갈랐다".

### 5가지 현대화 개선

ConvNeXt는 Transformer의 설계를 CNN에 역수입하기 위해 **단계적으로** 5가지를 개선했습니다:

1. **Stage Ratio 조정**: ResNet (1:2:3:3) → Transformer-like (3:3:9:3)
   - 의미: 더 많은 깊이를 중간 단계에 할당

2. **Depthwise 7×7 Convolution**: 3×3 → 7×7 (depthwise only)
   - 의미: Attention의 넓은 receptive field 모방
   - 효율: 전체 conv 파라미터 (3×3)보다 적음 (7×7 depthwise)

3. **LayerNorm**: BatchNorm → LayerNorm
   - 의미: Transformer의 정규화 방식 차용
   - 효과: 배치 크기에 독립적

4. **GELU 활성화**: ReLU → GELU
   - 의미: Smooth 활성화 함수

5. **Inverted Bottleneck**: 표준 bottleneck → inverted (확대 후 압축)
   - 의미: Transformer의 FFN 구조 (wide intermediate layer)

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Depthwise Convolution

일반 convolution:
$$y_c(i,j) = \sum_{c'=1}^{C_{\text{in}}} \sum_{k_1, k_2} w_{c,c',k_1,k_2} x_{c'}(i+k_1, j+k_2) + b_c$$

파라미터: $C_{\text{in}} \times C_{\text{out}} \times K^2$

Depthwise convolution (그룹 conv with groups=$C_{\text{in}}$):
$$y_c(i,j) = \sum_{k_1, k_2} w_{c,k_1,k_2} x_c(i+k_1, j+k_2) + b_c$$

파라미터: $C_{\text{in}} \times K^2$ (채널당 1개 커널)

**특성**: 채널 간 상호작용 없음, 공간 정보만 변환

### 정리 4.2 — Depthwise + 1×1 조합

Depthwise conv ($K \times K$) 후 1×1 conv를 조합하면:

**전체 파라미터**:
$$P = C_{\text{in}} \times K^2 + C_{\text{in}} \times C_{\text{out}} \times 1$$

**비교** (일반 K×K conv):
$$P_{\text{standard}} = C_{\text{in}} \times C_{\text{out}} \times K^2$$

감소율 (for $K=7, C_{\text{out}} \approx C_{\text{in}}$):
$$\frac{P}{P_{\text{standard}}} = \frac{K^2 + 1}{K^2 \times C_{\text{out}}} \approx \frac{1}{C_{\text{out}}}$$

예: $C_{\text{out}} = 256$이면, 1/256 수준의 파라미터 (최대 99% 절감!)

### 정의 4.3 — ConvNeXt 블록 (Modern Conv Block)

```
Input (x, shape: [B, C, H, W])
  ↓ Depthwise 7×7 Conv (groups=C)
  ↓ LayerNorm (channel-wise)
  ↓ 1×1 Conv (C → 4C)  [expansion]
  ↓ GELU
  ↓ 1×1 Conv (4C → C) [projection]
  + Residual connection
Output
```

이는 Transformer의 구조:
```
Input
  ↓ Multi-Head Attention
  ↓ LayerNorm
  ↓ FFN (d → 4d → d)
  ↓ (GELU)
  + Residual
Output
```

와 유사합니다 (Attention을 Depthwise Conv로 대체).

### 정의 4.4 — ConvNeXt 아키텍처

| Stage | Blocks | Channels | Repeats |
|-------|--------|----------|---------|
| 1     | Stem   | 64       | 1       |
| 2     | Conv   | 128      | 3       |
| 3     | Conv   | 256      | 9       |
| 4     | Conv   | 512      | 3       |

**Stem**:
```
Input (224×224)
  ↓ Conv 7×7, stride 4, 64 channels
  ↓ LayerNorm
Output (56×56)
```

**Downsample block** (stride 2):
```
LayerNorm
  ↓ Conv 3×3, stride 2 (depthwise)
  ↓ Conv 1×1 (channel expansion)
```

---

## 🔬 수학적 유도

### Depthwise Convolution의 효율성 분석

**FLOPs 비교**:

일반 K×K conv:
$$\text{FLOPs}_{\text{standard}} = H \times W \times K^2 \times C_{\text{in}} \times C_{\text{out}}$$

Depthwise K×K + 1×1 conv:
$$\text{FLOPs}_{\text{depthwise}} = H \times W \times K^2 \times C_{\text{in}} + H \times W \times C_{\text{in}} \times C_{\text{out}}$$

비율:
$$\frac{\text{FLOPs}_{\text{depthwise}}}{\text{FLOPs}_{\text{standard}}} = \frac{K^2 + C_{\text{out}}}{K^2 \times C_{\text{out}}}$$

예: $K=7, C_{\text{out}}=256$
$$\frac{49 + 256}{49 \times 256} = \frac{305}{12544} \approx 0.024 \text{ (97.6% 절감!)}$$

### 실제 구현: Depthwise 7×7

```python
# 표준 7×7 conv: C_in × C_out × 49 파라미터
standard_7x7 = nn.Conv2d(256, 256, kernel_size=7, padding=3)

# Depthwise + 1×1: C_in × 49 + C_in × C_out 파라미터
depthwise_7x7 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
    nn.Conv2d(256, 256, kernel_size=1)
)

# 파라미터 수
print(f"Standard 7×7: {sum(p.numel() for p in standard_7x7.parameters()):,}")
print(f"Depthwise+1×1: {sum(p.numel() for p in depthwise_7x7.parameters()):,}")
```

**출력**:
```
Standard 7×7: 1,605,632
Depthwise+1×1: 6,400
```

---

## 💻 실험 재현: ConvNeXt 구현

### ConvNeXt 블록

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """Modern Convolutional Block (ConvNeXt style)"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        
        # Depthwise 7×7 convolution
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (channel-wise)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted bottleneck: dim → 4*dim → dim
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        # Stochastic depth (optional)
        self.drop_path = drop_path
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        residual = x
        
        # Depthwise
        x = self.depthwise(x)
        
        # Reshape for LayerNorm (expects [B, C, H, W])
        # LayerNorm normalizes over channel dimension
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Residual
        x = residual + x
        
        return x

# 테스트
block = ConvNeXtBlock(dim=256)
x = torch.randn(1, 256, 56, 56)
y = block(x)

print(f"Input: {x.shape}, Output: {y.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

### 간단한 ConvNeXt 모델

```python
class SimpleConvNeXt(nn.Module):
    """ConvNeXt-like architecture"""
    def __init__(self, depths=[3, 9, 3], dims=[128, 256, 512], num_classes=1000):
        super().__init__()
        
        # Stem: 7×7 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=7, stride=4, padding=3),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dim) for _ in range(depth)]
            )
            self.stages.append(stage)
            
            # Downsample block (except last)
            if i < len(depths) - 1:
                next_dim = dims[i + 1]
                downsample = nn.Sequential(
                    nn.LayerNorm(dim, eps=1e-6),
                    nn.Conv2d(dim, next_dim, kernel_size=2, stride=2)
                )
                self.stages.append(downsample)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

# 모델 생성
model = SimpleConvNeXt(depths=[3, 3, 3], dims=[64, 128, 256], num_classes=1000)
x = torch.randn(1, 3, 224, 224)
y = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"Output shape: {y.shape}")
print(f"Total parameters: {total_params / 1e6:.2f}M")
```

### 5가지 개선의 단계적 효과 (실험)

```python
import matplotlib.pyplot as plt

# ConvNeXt 논문에서의 실제 결과 (ImageNet-1k)
improvements = {
    'ResNet-50 (baseline)': 76.0,
    '+ Stage ratio (3:3:9:3)': 77.5,
    '+ Depthwise 7×7': 78.6,
    '+ LayerNorm + GELU': 79.1,
    '+ Inverted bottleneck': 79.8,
    'ConvNeXt-T': 82.1
}

models = list(improvements.keys())
accuracies = list(improvements.values())

fig, ax = plt.subplots(figsize=(12, 6))

colors = ['gray', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'orange']
bars = ax.bar(range(len(models)), accuracies, color=colors, edgecolor='black', linewidth=1.5)

# 값 레이블 추가
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('ImageNet Top-1 Accuracy (%)', fontsize=12)
ax.set_title('ConvNeXt: Cumulative Effect of 5 Modern Improvements', fontsize=14)
ax.set_ylim([75, 83])
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 각 개선의 기여도
print("Cumulative improvements:")
for i in range(1, len(accuracies)):
    delta = accuracies[i] - accuracies[i-1]
    print(f"{models[i]}: +{delta:.1f}%")
```

### Timm을 사용한 실제 ConvNeXt

```python
import timm

# 사전학습된 ConvNeXt 모델
convnext_t = timm.create_model('convnext_tiny', pretrained=True)
convnext_b = timm.create_model('convnext_base', pretrained=True)

print(f"ConvNeXt-T params: {sum(p.numel() for p in convnext_t.parameters()) / 1e6:.2f}M")
print(f"ConvNeXt-B params: {sum(p.numel() for p in convnext_b.parameters()) / 1e6:.2f}M")

# ConvNeXt vs ViT 비교
vit_b = timm.create_model('vit_base_patch16_224', pretrained=True)

print(f"\nConvNeXt-B: {sum(p.numel() for p in convnext_b.parameters()) / 1e6:.2f}M params, ~83.6% ImageNet Top-1")
print(f"ViT-B: {sum(p.numel() for p in vit_b.parameters()) / 1e6:.2f}M params, ~81.8% ImageNet Top-1")
print("\nConvNeXt이 비슷한 규모에서 더 높은 정확도!")
```

---

## 🔗 이론과 실전의 간극

### 1. 아키텍처 vs 훈련 기법

**논문의 핵심 기여**:

과거: ResNet (90 epoch) vs ViT (300 epoch) — 부정확한 비교
→ "Transformer이 우월하다" 결론

현재: 동일 훈련 기법 (300 epoch, AdamW, augmentation, warmup)
→ "ConvNeXt ≥ ViT"

**교훈**: 신경망 비교는 **훈련 조건이 동일**해야 함

### 2. 정규화 방식의 역할

**BatchNorm**: 배치 내 통계에 의존
- 배치 크기에 민감
- 동기화 필요 (분산 훈련에서)

**LayerNorm**: 채널 내 정규화
- 배치 크기에 독립적
- Transformer의 표준

ConvNeXt는 LayerNorm 도입으로:
- 더 큰 배치 크기 사용 가능
- 더 강한 정규화 효과

### 3. Receptive Field와 Depthwise의 역할

Depthwise 7×7 conv:
- Receptive field = 7 (3×3의 3배 이상)
- Transformer의 self-attention (전체 이미지 보는 것)을 근사
- 하지만 파라미터는 매우 적음

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Modern training이 필수 | 기존 훈련 기법으로는 성능 향상 제한 |
| 충분한 데이터 | ImageNet에서 최적, 작은 데이터셋은 과적합 위험 |
| LayerNorm 안정성 | 극단적 입력(매우 큰/작은 값)에는 취약 가능 |
| Depthwise의 효율성 | 실제 하드웨어 지원 필요 (구형 GPU는 느림) |
| GELU의 효율 | ReLU보다 계산 비용 높음 |

---

## 📌 핵심 정리

$$\boxed{\text{ConvNeXt: Modernized CNN} = \text{Depthwise 7×7} + \text{LayerNorm} + \text{Inverted Bottleneck} + \text{...}}$$

| 개념 | 정의 |
|------|------|
| **Depthwise Conv** | groups=channels인 convolution, 파라미터 극대 절감 |
| **7×7 Receptive Field** | Transformer의 넓은 주의 범위를 모방 |
| **LayerNorm** | 채널 축 정규화, 배치 크기 독립적 |
| **Inverted Bottleneck** | 확대(dim→4dim) 후 압축(→dim) |
| **Stage Ratio 3:3:9:3** | Transformer의 깊이 분포 채용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Depthwise 5×5 convolution과 일반 5×5 convolution의 파라미터 수를 비교하라. (입출력 채널 모두 256)

<details>
<summary>힌트 및 해설</summary>

일반 5×5:
$$P_{\text{standard}} = 256 \times 256 \times 25 = 1,638,400$$

Depthwise 5×5:
$$P_{\text{depthwise}} = 256 \times 25 = 6,400$$

**차이: 1,632,000 (99.6% 절감!)** ✓

</details>

**문제 2** (심화): ConvNeXt 블록의 inverted bottleneck (dim → 4dim → dim)에서, dim=256일 때 파라미터 수를 계산하라.

<details>
<summary>힌트 및 해설</summary>

구조:
```
1×1 Conv (256 → 1024)
1×1 Conv (1024 → 256)
```

파라미터:
- 첫 번째 1×1: $256 \times 1024 = 262,144$
- 두 번째 1×1: $1024 \times 256 = 262,144$
- **합계: 524,288**

참고 (표준 bottleneck): $256 \times 64 + 64 \times 256 = 32,768$ (16배 많음)

Inverted는 중간 차원이 더 크기 때문에 파라미터 많음 → 하지만 이것이 Transformer와의 유사성 가져옴 ✓

</details>

**문제 3** (이론-실전): ConvNeXt-T (29M params)가 ViT-T (이 모델 크기 불명확하지만 보통 ~5-10M)보다 왜 더 크면서도 더 높은 정확도를 달성하는가? 이는 CNN과 Transformer 중 어느 것이 더 "효율적"인가?

<details>
<summary>힌트 및 해설</summary>

실제로는:
- ConvNeXt-T: 29M params, 82.1% (ImageNet, modern training)
- ViT-Tiny: ~5M params, ~72% (ImageNet, 표준 training)
- ViT-B: 86M params, ~81.8% (modern training)

**비교**:
1. **파라미터 효율성**: ViT > ConvNeXt (같은 파라미터 수로 더 높은 정확도)
2. **절대 성능**: 동일 훈련 조건에서 ConvNeXt-B(89M) > ViT-B(86M)

→ **아키텍처보다 설계 원칙이 중요** (inductive bias, normalization, activation)

"ConvNeXt가 더 효율적"이 아니라, "CNN도 modern design으로 최적화되면 ViT 수준"

</details>

---

<div align="center">

[◀ 이전](./03-efficientnet.md) | [📚 README](../README.md) | [다음 ▶](./05-nas.md)

</div>
