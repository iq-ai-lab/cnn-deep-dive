# 01. VGG와 Depth의 효과: 깊은 신경망의 가능성

## 🎯 핵심 질문

- 합성곱 신경망에서 **network depth**가 중요한 이유는 무엇인가?
- 작은 $3 \times 3$ 필터를 연속으로 적용하면, 왜 큰 필터를 쓰는 것과 같은 receptive field를 얻을 수 있는가?
- 2개의 $3 \times 3$ 필터 스택이 1개의 $5 \times 5$ 필터보다 왜 더 효율적인가?
- VGG는 최대 19 layer까지 가능하지만, 152 layer plain CNN은 왜 수렴하지 못하는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

VGG (Simonyan & Zisserman, 2014)는 ImageNet 분류에서 AlexNet 이후 가장 큰 도약을 이룬 아키텍처입니다. VGG의 핵심 기여는:

1. **Depth의 중요성 입증** — 깊이가 정확도에 직접적 영향을 미친다는 실험적 증거 제공
2. **$3 \times 3$ convolution 표준화** — 효율성과 표현력 간 최적 균형
3. **Degradation problem 발견** — plain deep networks가 수렴 실패하는 현상 관찰

이는 이후 ResNet의 동기가 되었으며, modern CNN 설계의 기본 원칙이 됩니다: **층이 많을수록 더 복잡한 특징을 학습할 수 있다**는 것입니다.

---

## 📐 수학적 선행 조건

- **합성곱 기본**: Padding, stride, dilation
- **Receptive Field (RF)** 개념: 한 뉴런이 입력에서 "본" 영역의 크기
- **파라미터 수 계산**: $K \times K \times C_{\text{in}} \times C_{\text{out}}$
- **신경망 근사 이론**: depth ↔ universal approximation

참고: [CNN 기초](../ch1-fundamentals)

---

## 📖 직관적 이해

### Receptive Field와 Filter 크기

합성곱 신경망에서 한 뉴런의 **receptive field (RF)**는 출력이 영향을 받는 입력의 영역입니다.

초기 이미지 입력에서:
- $k_1 \times k_1$ filter → RF = $k_1$
- 그 위에 $k_2 \times k_2$ filter → RF = $k_1 + k_2 - 1$
- 총 $L$ layers, 각각 $3 \times 3$ → RF = $1 + 2 \times L$

따라서:
- 19-layer VGG (stacking $3 \times 3$): RF ≈ $1 + 2 \times 19 = 39$
- 비교: 단일 $39 \times 39$ filter는 거의 같은 RF이지만, 파라미터는 훨씬 많음

### 파라미터 효율성: 3×3 vs 5×5

$C_{\text{in}} = C_{\text{out}} = C$일 때:

**Option A**: 1개의 $5 \times 5$ 필터
$$\text{Parameters} = 5^2 \times C^2 = 25C^2$$

**Option B**: 2개의 $3 \times 3$ 필터 (중간 채널도 $C$)
$$\text{Parameters} = 2 \times (3^2 \times C^2) = 2 \times 9C^2 = 18C^2$$

효율성 향상: $(25 - 18) / 25 = 28\%$ 파라미터 절감

동시에 **비선형성 증가**:
- Option A: 1개 활성화함수 → 깊이 1
- Option B: 2개 활성화함수 → 깊이 2

더 깊은 신경망은 더 복잡한 함수를 근사할 수 있습니다.

### Degradation Problem

일반적 직관: "더 깊은 네트워크 = 더 큰 표현력" → 더 낮은 오류율

하지만 VGG 논문의 실험에서:
- VGG-11 (8 conv + 3 fc): 69.0% top-1 accuracy (ImageNet)
- VGG-16: 71.3%
- VGG-19: 71.4%
- **Plain CNN-16** (no skip connections): 훈련 오류도 높음 → 수렴 실패

이는 단순히 최적화 문제입니다: gradient vanishing으로 인해 깊은 층의 가중치가 거의 업데이트되지 않음.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Receptive Field

$l$번째 층에서 출력 $(i, j)$에 대해, receptive field $RF_l$은 이 출력에 기여하는 입력 이미지의 영역 크기입니다.

다음 점화식:
$$RF_{l} = RF_{l-1} + (K_l - 1) \prod_{m=1}^{l-1} s_m$$

여기서 $K_l$은 $l$ 층의 필터 크기, $s_m$은 $m$ 층의 stride.

### 정리 1.2 — 연속 3×3 필터의 RF

$s_m = 1$ (stride 1)인 경우, $n$개의 $3 \times 3$ 필터를 스택하면:
$$RF_n = 1 + 2n$$

증명: 귀납법으로 $RF_1 = 3$, $RF_n = RF_{n-1} + 2$ → $RF_n = 1 + 2n$. $\square$

### 정의 1.3 — VGG 아키텍처

VGG는 다음과 같이 정의됩니다:
$$\text{VGG}(D) = [C_1, M, C_2, M, C_3, M, C_4, M, C_5, M, FC(4096), FC(4096), FC(1000)]$$

여기서:
- $M$ = Max-pooling $2 \times 2$, stride 2
- $C_d$ = $d$번째 블록 (3×3 conv layers)
- VGG-11: $d$ = [1, 1, 2, 2, 2] (총 8 conv)
- VGG-13: $d$ = [2, 2, 2, 2, 2] (총 10 conv)
- VGG-16: $d$ = [2, 2, 3, 3, 3] (총 13 conv)
- VGG-19: $d$ = [2, 2, 4, 4, 4] (총 16 conv)

### 정리 1.4 — 파라미터 수 비교

$C_{\text{in}} = C_{\text{out}} = C$, 채널 수 일정할 때:

**$k \times k$ 단일 필터**: $k^2 C^2$ 파라미터

**$3 \times 3$ 필터 $m$개 스택** (중간에도 $C$ 채널): $m \times 9C^2$ 파라미터

$m = \lceil k/2 \rceil$일 때:
$$\text{절감율} = 1 - \frac{9m}{k^2} = 1 - \frac{9 \lceil k/2 \rceil}{k^2}$$

$k=5$: $1 - 18/25 = 28\%$ 절감. $\square$

---

## 🔬 수학적 유도 및 분석

### Receptive Field의 재귀 공식

$l$층의 receptive field는:
$$RF_l = 1 + \sum_{m=1}^{l-1} (K_m - 1) \prod_{i=m+1}^{l-1} s_i$$

stride가 모두 1인 경우:
$$RF_l = 1 + (K_1 - 1) + (K_2 - 1) + \cdots + (K_{l-1} - 1) = \sum_{m=1}^{l} K_m - (l-1)$$

$K_m = 3$이면:
$$RF_l = 3l - (l-1) = 2l + 1$$

따라서 19-layer VGG: $RF = 2 \times 19 + 1 = 39$

### 파라미터-FLOPs 분석

VGG-16 (모든 conv가 $C=64, 128, 256, 512$ 채널):

| Block | Layers | Channel | RF  | Conv Params | FLOPs (1 forward pass) |
|-------|--------|---------|-----|-------------|----------------------|
| 1     | 2      | 64      | 3   | 9K          | 107M                 |
| 2     | 2      | 128     | 7   | 147K        | 109M                 |
| 3     | 3      | 256     | 15  | 883K        | 218M                 |
| 4     | 3      | 512     | 31  | 1.9M        | 219M                 |
| 5     | 3      | 512     | 39  | 1.9M        | 110M                 |
| **Total** | **16** | — | **39** | **7.6M (+FC: 138M)** | **~15.5B** |

---

## 💻 실험 재현: VGG 구현 및 Depth 효과

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VGGBlock(nn.Module):
    """VGG 단일 블록: 여러 3x3 conv + pooling"""
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGG(nn.Module):
    """VGG 아키텍처"""
    def __init__(self, num_classes=1000, depth_config=[2,2,3,3,3]):
        """
        depth_config: 각 블록의 conv 개수
        - [2,2,3,3,3]: VGG-16 (13 conv + maxpool)
        - [2,2,4,4,4]: VGG-19 (16 conv + maxpool)
        """
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, depth_config[0]),
            VGGBlock(64, 128, depth_config[1]),
            VGGBlock(128, 256, depth_config[2]),
            VGGBlock(256, 512, depth_config[3]),
            VGGBlock(512, 512, depth_config[4])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 사전학습된 VGG 로드
vgg16 = models.vgg16(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

print(f"VGG-16 파라미터 수: {sum(p.numel() for p in vgg16.parameters()):,}")
print(f"VGG-19 파라미터 수: {sum(p.numel() for p in vgg19.parameters()):,}")
```

**출력**:
```
VGG-16 파라미터 수: 138,357,544
VGG-19 파라미터 수: 144,237,544
```

---

## 🔗 이론과 실전의 간극

### 1. 최적화 관점

**이론**: 더 깊은 네트워크는 더 넓은 함수 클래스를 근사

**실전의 문제**:
- **Gradient vanishing**: 역전파 중 gradient가 기하급수적으로 감소
- Sigmoid/tanh 활성화에서 심각 ($\partial \sigma / \partial x \leq 0.25$)
- ReLU 도입으로 완화되었지만, 여전히 매우 깊으면 문제 (→ ResNet의 동기)

### 2. 계산 복잡도

VGG-16: ~138M 파라미터, ~15.5B FLOPs
- AlexNet (60M params)의 2배 이상
- ImageNet 학습: V100 GPU에서 수일 소요
- Mobile/Edge 환경에서는 실제로 사용 불가

→ 이후 MobileNet, SqueezeNet 같은 경량화 아키텍처 개발

### 3. 일반화 (Generalization)

VGG는 매우 깊어 **과적합 위험** 높음:
- ImageNet (1.2M 이미지) 학습 후 CIFAR-10 (60K)에서 세밀한 조정 필요
- Dropout, 정규화, 데이터 증강 필수

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 충분한 메모리 & 계산 자원 | 138M params → GPU 메모리 부족 가능 |
| 모든 활성화함수 동등 | ReLU 필수, sigmoid면 gradient vanishing 심화 |
| ImageNet-scale 데이터 | 작은 데이터셋에서는 과적합 |
| 정확한 gradient 계산 가능 | Batch norm 없으면 learning rate 민감 |
| 고정 입력 해상도 (224×224) | 다른 크기는 재조정 또는 아키텍처 수정 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{VGG: Depth} \rightarrow \text{RF, Expressiveness} \uparrow \quad \text{But:} \quad \text{Optimization} \downarrow \text{ (gradient vanishing)}}$$

| 개념 | 정의 |
|------|------|
| **Receptive Field** | 출력 1픽셀에 기여하는 입력의 영역 크기 |
| **RF (n개 3×3)** | $RF_n = 1 + 2n$ (stride=1) |
| **VGG-16** | 13 conv + 3 fc layers, 138M params |
| **Degradation** | 깊이 증가 → 훈련 손실 증가 (수렴 실패) |
| **3×3 효율성** | 1개 5×5 vs 2개 3×3: 28% 파라미터 절감, 비선형성 2배 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): VGG-16의 첫 블록(2개 3×3 conv)의 receptive field를 계산하고, 이에 해당하는 단일 필터 크기를 구하라.

<details>
<summary>힌트 및 해설</summary>

각 conv 후 receptive field 성장:
- Layer 0 (입력): RF = 1
- 1번째 3×3: RF = 1 + (3-1) = 3
- 2번째 3×3: RF = 3 + (3-1) = 5

따라서 블록 1의 RF = 5 ✓

단일 필터로 같은 RF: 5×5 필터 필요

</details>

**문제 2** (심화): VGG-16과 VGG-19의 FLOPs 차이가 약 15~20%인 이유를 설명하라. (힌트: Block 5의 차이)

<details>
<summary>힌트 및 해설</summary>

VGG-16 Block 5: 3개 3×3 conv
VGG-19 Block 5: 4개 3×3 conv

Block 5는 최고 해상도 (7×7) + 최고 채널 (512)에서 실행되므로, 1개 층 추가 = 상당한 FLOPs 증가:
$$\Delta \text{FLOPs} = 7 \times 7 \times 512 \times 512 \approx 12.8 \text{B}$$

전체 15.5B FLOPs 중 ~12%, 충분히 설명 가능 ✓

</details>

**문제 3** (이론-실전): "깊이가 깊을수록 표현력이 높다"는 명제는 언제 거짓이 되는가? ResNet이 이를 어떻게 해결하는지 논의하라.

<details>
<summary>힌트 및 해설</summary>

깊이의 표현력 이득이 최적화 비용으로 상쇄될 때:
- Gradient vanishing: 각 층마다 $\partial L/\partial \theta_i$ 감소 → 역전파 불가능
- 훈련 동역학: 깊은 층은 거의 업데이트 안 됨 → 실질적으로 얕은 네트워크

**ResNet 솔루션** (다음 Ch): Skip connection $x_{l+1} = x_l + f(x_l)$
- Gradient highway: $\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} (1 + \frac{\partial f}{\partial x_l})$
- "1"항이 gradient를 보존 → deep networks도 학습 가능
- 따라서 VGG 깊이 19 vs ResNet 깊이 152 가능

</details>

---

<div align="center">

[◀ 이전](../ch4-resnet/06-stochastic-depth.md) | [📚 README](../README.md) | [다음 ▶](./02-inception.md)

</div>
