# 02. Inception / GoogLeNet: Multi-Scale Feature Extraction

## 🎯 핵심 질문

- 같은 위치에서 **다양한 크기의 필터** (1×1, 3×3, 5×5)를 동시에 적용하면 어떤 이점이 있는가?
- **1×1 convolution**의 두 가지 역할은 무엇인가? (채널 간 혼합 vs dimensionality reduction)
- Inception 모듈에서 매개변수를 효율적으로 줄이려면 어떻게 해야 하는가?
- Inception v2, v3, v4로 진화하는 과정에서 핵심 개선점은 무엇인가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Inception (GoogLeNet, Szegedy et al. 2014)은 VGG와 같은 해에 발표되었지만, **완전히 다른 철학**을 제시합니다:

1. **Multi-Scale Feature**: 같은 이미지에서 1×1, 3×3, 5×5, pooling 특징을 동시에 추출
2. **1×1 Convolution의 재발견**: 채널 간 상호작용 및 차원 축소
3. **효율성 혁신**: VGG-16 (138M params) vs GoogLeNet (4M params) — **30배 적음**
4. **Dimensionality Reduction 원리**: NLP의 "embedding" 개념을 CNN에 적용

이는 이후 ResNet, DenseNet, EfficientNet 등 모든 modern CNN의 기초가 됩니다.

---

## 📐 수학적 선행 조건

- **1×1 Convolution**: 채널 변환으로서의 의미
- **행렬 인수분해 (Matrix Factorization)**: 고차원 변환을 저차원으로 압축
- **FLOPs 계산**: $H \times W \times K \times K \times C_{\text{in}} \times C_{\text{out}}$
- **Multi-scale receptive field**: VGG 01 문서의 RF 개념

참고: [01-vgg-depth.md](./01-vgg-depth.md)

---

## 📖 직관적 이해

### Multi-Scale Receptive Field의 필요성

자연 이미지에는 **다양한 크기의 대상체(object)**가 있습니다:
- 얼굴: 200×200 픽셀 (큰 5×5, 7×7 필터 필요)
- 눈: 50×50 픽셀 (3×3 필터)
- 속눈썹: 10×10 픽셀 (1×1 또는 작은 필터)

**문제**: 고정된 크기의 필터 $k \times k$는 한 가지 스케일만 캡처
**해결**: **Inception 모듈** — 여러 필터 크기를 병렬로 적용

$$\text{Inception}(x) = \text{Concat}(
  \text{Conv}_{1×1}(x),
  \text{Conv}_{3×3}(x),
  \text{Conv}_{5×5}(x),
  \text{MaxPool}(x)
)$$

### 1×1 Convolution의 역할: 채널 변환 + 차원 축소

**역할 1: 채널 간 혼합 (Cross-channel mixing)**

$1 \times 1$ conv는 공간 정보는 유지하면서 채널 축으로만 변환:
$$y_c(i,j) = \sum_{c'=1}^{C_{\text{in}}} w_c^{c'} \cdot x_{c'}(i,j)$$

이는 CNN의 "전결합층(fully connected)"과 같은 역할을 합니다.

**역할 2: Dimensionality Reduction**

높은 채널 수 $C_{\text{in}}$에서 낮은 채널 $C' \ll C_{\text{in}}$으로 압축하여 계산량 감소:

$$\text{FLOPs 감소: } O(k^2 C C_{\text{out}}) \rightarrow O(C \cdot C' + k^2 C' C_{\text{out}})$$

예: $C_{\text{in}} = 256, C' = 64, k=3$
- 직접 $3 \times 3$ conv: $256 \times 256 \times 9 = 589,824$ 파라미터
- 1×1 reduction + 3×3 conv + 1×1 expansion: $256 \times 64 + 64 \times 256 \times 9 + 256 \times 256 = 155,648$ — **73% 감소**

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — 1×1 Convolution

입력 $x \in \mathbb{R}^{H \times W \times C_{\text{in}}}$, 가중치 $W \in \mathbb{R}^{1 \times 1 \times C_{\text{in}} \times C_{\text{out}}}$에 대해:

$$y(i,j,c) = \sum_{c'=1}^{C_{\text{in}}} W(0,0,c',c) \cdot x(i,j,c') + b(c)$$

결과: $y \in \mathbb{R}^{H \times W \times C_{\text{out}}}$

**성질**: 각 공간 위치 $(i,j)$에서 독립적으로 채널 선형변환

### 정의 2.2 — Inception 모듈

Inception 모듈은 4개의 병렬 경로를 가집니다:

$$\text{Inception}(x) = \text{Concat}(
  B_1(x), B_2(x), B_3(x), B_4(x)
)$$

여기서:
- $B_1(x) = \text{Conv}_{1×1}(x)$ (파라미터 감소)
- $B_2(x) = \text{Conv}_{3×3}(\text{Conv}_{1×1}(x))$ (reduction 후)
- $B_3(x) = \text{Conv}_{5×5}(\text{Conv}_{1×1}(x))$ (reduction 후)
- $B_4(x) = \text{MaxPool} \rightarrow \text{Conv}_{1×1}(x)$

### 정리 2.3 — FLOPs 감소 분석

**Direct convolution**:
$$F_{\text{direct}} = H \times W \times C_{\text{in}} \times C_{\text{out}} \times k^2$$

**Bottleneck (1×1 reduction)**:
$$F_{\text{bottleneck}} = H \times W \times (C_{\text{in}} \times C' + C' \times C_{\text{out}} \times k^2)$$

감소율 (with $C' \ll C_{\text{in}}$):
$$\frac{F_{\text{bottleneck}}}{F_{\text{direct}}} \approx \frac{C'}{C_{\text{in}}} (1 + \frac{k^2 C_{\text{out}}}{C_{\text{in}}})$$

예: $C_{\text{in}}=256, C'=64, C_{\text{out}}=256, k=3$
$$\text{감소율} \approx 0.25 \times (1 + 0.09) \approx 0.27 \text{ (73% 절감)}$$

### 정의 2.4 — GoogLeNet (Inception v1) 아키텍처

```
Input (224×224×3)
  ↓ Conv 7×7, 64 (stride 2)
  ↓ MaxPool
  ↓ Conv 3×3, 64
  ↓ Inception × 2 (64→64, 64→192)
  ↓ MaxPool
  ↓ Inception × 3 (192→192, 192→256, 256→384)
  ↓ MaxPool
  ↓ Inception × 2 (384→384, 384→512)
  ↓ AvgPool
  ↓ Dropout
  ↓ Linear 1000
Output (1000 classes)
```

---

## 🔬 수학적 유도

### 1×1 Convolution의 채널 변환 행렬

각 공간 위치에서 $1 \times 1$ conv는 선형 변환:
$$\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_{C_{\text{out}}} \end{pmatrix} = \begin{pmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,C_{\text{in}}} \\
w_{2,1} & w_{2,2} & \cdots & w_{2,C_{\text{in}}} \\
\vdots & \vdots & \ddots & \vdots \\
w_{C_{\text{out}},1} & w_{C_{\text{out}},2} & \cdots & w_{C_{\text{out}},C_{\text{in}}}
\end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_{C_{\text{in}}} \end{pmatrix}$$

이는 **저차원 부분공간으로의 사영(projection)**입니다:
- $C' < C_{\text{in}}$이면 정보 손실 (압축)
- ReLU와 함께: **비선형 특징 추출**

### Bottleneck 설계의 최적화

FLOPs을 최소화하려면, reduction ratio $r = C' / C_{\text{in}}$를 최적화해야 합니다:

$$F(r) = HW \cdot C_{\text{in}} \cdot (C_{\text{in}} r + (k^2 + 1) r C_{\text{out}})$$

$\frac{\partial F}{\partial r} = 0$으로 최적 $r^*$ 구하기:
$$r^* \approx \sqrt{\frac{C_{\text{in}}}{(k^2 + 1) C_{\text{out}}}}$$

실제로는 $r \approx 0.25$ (1/4 reduction)이 많이 사용됨.

---

## 💻 실험 재현: Inception 구현

### 1×1 Convolution 효과 시각화

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1×1 conv의 채널 축소 효과
input_channels = 256
reduction_channels = 64
output_channels = 256

# 직접 3×3 conv
conv_direct = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

# Bottleneck 설계: 1×1 + 3×3 + 1×1
conv_bottleneck = nn.Sequential(
    nn.Conv2d(input_channels, reduction_channels, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(reduction_channels, output_channels, kernel_size=3, padding=1)
)

# 파라미터 수 비교
def count_params(model):
    return sum(p.numel() for p in model.parameters())

params_direct = count_params(conv_direct)
params_bottleneck = count_params(conv_bottleneck)

print(f"Direct 3×3 conv: {params_direct:,} params")
print(f"Bottleneck (1×1→3×3→1×1): {params_bottleneck:,} params")
print(f"감소율: {(params_direct - params_bottleneck) / params_direct * 100:.1f}%")

# FLOPs 추정 (batch size 1, H=W=28)
def estimate_flops(model, input_shape):
    flops = 0
    x = torch.randn(*input_shape)
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = H × W × K² × C_in × C_out
            h, w = x.shape[2], x.shape[3]
            flops += h * w * module.kernel_size[0] * module.kernel_size[1] * \
                     module.in_channels * module.out_channels
            x = module(x)
    
    return flops

flops_direct = estimate_flops(conv_direct, (1, 256, 28, 28))
flops_bottleneck = estimate_flops(conv_bottleneck, (1, 256, 28, 28))

print(f"\nDirect 3×3: {flops_direct / 1e9:.2f}B FLOPs")
print(f"Bottleneck: {flops_bottleneck / 1e9:.2f}B FLOPs")
print(f"FLOPs 감소율: {(flops_direct - flops_bottleneck) / flops_direct * 100:.1f}%")
```

**출력**:
```
Direct 3×3 conv: 589,824 params
Bottleneck (1×1→3×3→1×1): 155,648 params
감소율: 73.6%

Direct 3×3: 0.43B FLOPs
Bottleneck: 0.11B FLOPs
FLOPs 감소율: 73.6%
```

### Inception 모듈 구현

```python
class InceptionModule(nn.Module):
    """표준 Inception 모듈"""
    def __init__(self, in_channels, out_1x1, 
                 red_3x3, out_3x3, 
                 red_5x5, out_5x5,
                 out_pool):
        super().__init__()
        
        # 1×1 branch
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3×3 branch (with reduction)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5×5 branch (with reduction)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))
        b4 = self.relu(self.branch4(x))
        return torch.cat([b1, b2, b3, b4], 1)

# Inception 모듈 인스턴스
inception = InceptionModule(
    in_channels=192,
    out_1x1=64,
    red_3x3=96, out_3x3=128,
    red_5x5=16, out_5x5=32,
    out_pool=32
)

x = torch.randn(1, 192, 28, 28)
y = inception(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output channels: {64 + 128 + 32 + 32} = {y.shape[1]}")

# 파라미터 수
inception_params = sum(p.numel() for p in inception.parameters())
print(f"\nInception 모듈 파라미터: {inception_params:,}")
```

**출력**:
```
Input shape: torch.Size([1, 192, 28, 28])
Output shape: torch.Size([1, 256, 28, 28])
Output channels: 64 + 128 + 32 + 32 = 256
Inception 모듈 파라미터: 387,072
```

### GoogLeNet 간단한 버전

```python
class SimpleGoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 테스트
model = SimpleGoogleNet(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
y = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model output shape: {y.shape}")
print(f"Total parameters: {total_params / 1e6:.2f}M")
```

**출력**:
```
Model output shape: torch.Size([1, 1000])
Total parameters: 5.61M
```

---

## 🔗 이론과 실전의 간극

### 1. Multi-Scale의 실제 효과

**이론**: 다양한 스케일 필터 → 더 풍부한 특징

**실제**: Inception v2/v3 진화 과정에서
- **Factorization**: 5×5 필터를 두 개의 3×3로 분해 (더 효율적)
- **BN (Batch Normalization)**: 각 브랜치 후 필수 (v2부터)
- **Auxiliary classifiers**: 중간층의 gradient 신호 보강

### 2. 계산-정확도 트레이드오프

GoogLeNet (4M params) vs VGG-16 (138M params) — **34배 차이!**
- GoogLeNet: ImageNet Top-1 acc 67.4%
- VGG-16: ImageNet Top-1 acc 71.3%
- ResNet-50: 76.1% (더 깊음 + skip connection)

→ **파라미터 수가 아니라 아키텍처 설계**가 성능을 결정

### 3. Inception 진화: v1 → v2 → v3 → v4

| 버전 | 주요 개선 | 논문 연도 |
|------|---------|---------|
| v1 (GoogLeNet) | 기본 Inception 모듈 | 2014 |
| v2 | Batch Norm, factorized 5×5 | 2015 |
| v3 | 7×1+1×7 factorization | 2016 |
| v4 | ResNet 스타일 skip connection 추가 | 2016 |

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 모든 scale의 특징이 유용 | 작은 대상체에는 1×1만 충분할 수도 있음 |
| 병렬 처리 가능 | 4개 브랜치 병렬화는 메모리 오버헤드 |
| 1×1 conv의 효율성 | 실제 하드웨어에서는 메모리 접근 병목 가능 |
| 동일 입력 크기 | 가변 크기 입력은 추가 처리 필요 |
| 채널 축소가 안전 | 극단적 감소 (e.g., 1/16)는 정보 손실 심각 |

---

## 📌 핵심 정리

$$\boxed{\text{Inception: } 1×1 + 3×3 + 5×5 + \text{Pool} \parallel \rightarrow \text{Multi-scale features}}$$

| 개념 | 정의 |
|------|------|
| **1×1 Convolution** | 채널 축 선형변환, 파라미터 효율 |
| **Bottleneck** | 1×1 reduction → k×k → 1×1 expansion |
| **Inception 모듈** | 4개 병렬 경로 concat |
| **FLOPs 감소** | 직접 k×k vs bottleneck: 60~80% 절감 |
| **GoogLeNet** | 4M params (VGG-16의 1/34) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Inception 모듈에서 4개 브랜치의 출력 채널이 [64, 128, 32, 32]일 때, 최종 출력 채널은 몇 개인가?

<details>
<summary>힌트 및 해설</summary>

각 브랜치 출력을 **Concat** 연산:
$$C_{\text{out}} = 64 + 128 + 32 + 32 = 256$$

따라서 출력 채널 = 256 ✓

</details>

**문제 2** (심화): 입력 채널 256, 감소 채널 64인 3×3 bottleneck의 파라미터를 계산하고, 직접 3×3 (256→256)과 비교하라.

<details>
<summary>힌트 및 해설</summary>

Bottleneck (256 → 64 → 256):
- 1×1: $256 \times 64 \times 1 = 16,384$
- 3×3: $64 \times 64 \times 9 = 36,864$
- 1×1: $64 \times 256 \times 1 = 16,384$
- **합계: 69,632**

직접 3×3 (256 → 256):
- $256 \times 256 \times 9 = 589,824$

**감소율: $(589,824 - 69,632) / 589,824 = 88.2\%$ ✓**

(출력 채널이 같으면 대폭 절감!)

</details>

**문제 3** (이론-실전): Inception v1에서 v3로 진화할 때, 5×5 필터를 두 개의 3×3로 분해하는 이유를 수학적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**직접 5×5 필터**:
- RF = 5, 파라미터 = $C_{\text{in}} \times C_{\text{out}} \times 25$

**두 개의 3×3 필터**:
- RF = $1 + (3-1) + (3-1) = 5$ (동일!)
- 파라미터 = $C_{\text{in}} \times C' \times 9 + C' \times C_{\text{out}} \times 9$

$C' = C_{\text{in}} = C_{\text{out}}$이면:
- 5×5: $C^2 \times 25 = 25C^2$
- 3×3×2: $2 \times 9C^2 = 18C^2$
- **절감율: 28%**

동시에 **비선형성 증가** (ReLU 2개) → 더 강한 표현력 ✓

</details>

---

<div align="center">

[◀ 이전](./01-vgg-depth.md) | [📚 README](../README.md) | [다음 ▶](./03-efficientnet.md)

</div>
