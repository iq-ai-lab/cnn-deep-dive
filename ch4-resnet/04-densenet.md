# 04. DenseNet: Dense Connections과 Feature Reuse

## 🎯 핵심 질문

- ResNet의 additive skip connection vs DenseNet의 concatenative connection의 근본적 차이는?
- Dense block에서 각 층이 이전 모든 층의 output을 받으면, 파라미터 수는 어떻게 변하는가?
- Growth rate $k$란 무엇이고, 왜 작은 값 (예: 12, 24)이 효과적인가?
- DenseNet이 ResNet-50과 유사한 성능을 **1/3 파라미터로** 달성하는 메커니즘은?

---

## 🔍 왜 이 개념이 CNN에 중요한가

DenseNet (Huang et al., 2017)은 ResNet 이후 CNN 아키텍처 설계의 중요한 방향을 제시했습니다. Skip connection의 개념을 극단화하여 **모든 이전 층으로부터의 연결**을 만들었습니다. 결과적으로 파라미터 효율, 정규화 효과, feature reuse가 극대화됩니다. 이는 modern CNN 설계에서 "더 깊은 것보다 더 연결된 것"이라는 철학을 제시했습니다.

---

## 📐 수학적 선행 조건

- [Ch4-01: Residual Block](./01-residual-block.md)
- [Ch4-02: Gradient Flow](./02-gradient-flow.md)
- 선형대수: Concatenation operation, feature dimension growth
- 미적분: Parameter counting, computational complexity analysis

---

## 📖 직관적 이해

### ResNet vs DenseNet: 연결 구조

**ResNet**: $y = x + F(x)$ (addition)
- 정보를 합치기 (element-wise)
- 계산 효율적
- 정보 손실 가능성

**DenseNet**: $y = [x, F(x)]$ (concatenation)
- 정보를 이어붙이기
- 정보 손실 없음
- 채널 수가 빠르게 증가

### Growth Rate의 의미

Dense block에서 각 층의 출력 채널이 $k$ (growth rate)만큼 증가합니다.

Layer $l$ 입력 채널: $c_0 + l \cdot k$

예시 ($c_0 = 24$, $k = 12$):
- Layer 0: 24 channels input
- Layer 1: 24 + 12 = 36 channels input
- Layer 2: 36 + 12 = 48 channels input
- ...

작은 $k$ (12, 24)를 사용하는 이유:
1. 파라미터 수 제어 (충분히 작음)
2. 충분한 표현력 (growth로 누적)
3. Regularization 효과 (narrow bottleneck)

### Bottleneck 구조

각 layer: $1 \times 1 \to 3 \times 3$ 구조
```
Input (C channels) 
  ↓ BN-ReLU-Conv 1×1 (4k channels)
  ↓ BN-ReLU-Conv 3×3 (k channels)
  ↓ Output (k channels)
```

이 구조로 파라미터를 크게 줄입니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 4.1 — Dense Block

$L$개의 layer를 가진 dense block:

$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

여기서:
- $H_l$은 BN-ReLU-Conv의 composite function
- $[\cdot]$은 concatenation
- 각 layer 출력: $k$ channels (growth rate)

### 정의 4.2 — 채널 수 성장

Dense block 시작: $c_0$ channels

Layer $l$ 입력 채널: $C_l = c_0 + l \cdot k$

$L$ layers 후 출력: $c_0 + L \cdot k$ channels

### 정의 4.3 — Transition Layer

Dense block 사이에 spatial dimension 감소:

$$y = \text{AvgPool2d}(\text{Conv}(1 \times 1, \theta \times C \text{ channels}, BN(x)))$$

Compression rate $\theta = 0.5$ (채널 수 절반)

### 정리 4.4 — 파라미터 효율성

ResNet-50: ~25.5M parameters
DenseNet-121: ~7.0M parameters (ResNet의 27%)

동일 정확도 달성하면서 파라미터 1/3 이하

**이유**:
1. Bottleneck으로 각 layer의 contribution 제한
2. Dense connection으로 depth 효과 (모든 이전 layer 접근)
3. Compression으로 채널 수 감소

---

## 🔬 수학적 유도

### Dense Connection의 Gradient

Dense block: $x_l = [x_0, x_1, \ldots, x_{l-1}]$

Backpropagation:
$$\frac{\partial L}{\partial x_{l-1}} = \frac{\partial L}{\partial x_l} \frac{\partial x_l}{\partial x_{l-1}}$$

중요: $x_{l-1}$이 $x_l$의 구성 요소이므로, gradient는 직접 전파.

Gradient path 개수: $L$개 (ResNet의 1개보다 많음)

따라서 gradient vanishing이 더 어려움.

### 파라미터 수 계산

Bottleneck layer $l$:
- Input: $C_{l-1} = c_0 + (l-1)k$ channels
- $1 \times 1$ conv: $C_{l-1} \times 4k$ parameters
- $3 \times 3$ conv: $4k \times k \times 9$ parameters

Parameters:
$$P_l = C_{l-1} \times 4k + 36k^2$$

Dense block total ($L$ layers):
$$\sum_{l=0}^{L-1} P_l = O(L^2) \text{ 이론적, 하지만 } k \text{가 작아서 실제로는 } O(L)$$

Transition 후 채널 압축으로 추가 감소.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — DenseNet Block 구현

```python
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rate=12, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = input_channels + i * growth_rate
            
            # Bottleneck: BN-ReLU-Conv 1x1 → Conv 3x3
            layer = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, 4 * growth_rate, kernel_size=1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, 
                         padding=1, bias=False)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, 1))
            features.append(new_feat)
        return torch.cat(features, 1)

# Test
block = DenseBlock(24, growth_rate=12, num_layers=6)
x = torch.randn(4, 24, 32, 32)
y = block(x)
print(f"Input: {x.shape}, Output: {y.shape}")
# Output: torch.Size([4, 96, 32, 32])  (24 + 6*12)
```

### 실험 2 — 채널 수 추적

```python
def trace_channel_growth():
    growth_rate = 12
    block_config = (6, 12, 24, 16)  # 각 block의 layers
    
    for block_idx, num_layers in enumerate(block_config):
        print(f"\nDense Block {block_idx + 1} ({num_layers} layers):")
        in_ch = 24 if block_idx == 0 else 48
        
        for layer in range(num_layers):
            print(f"  Layer {layer}: {in_ch} → {in_ch + growth_rate} channels")
            in_ch += growth_rate
        
        print(f"  Block output: {in_ch} channels")
        in_ch = int(in_ch * 0.5)  # Compression
        print(f"  After compression: {in_ch} channels")

trace_channel_growth()
```

**출력**:
```
Dense Block 1 (6 layers):
  Layer 0: 24 → 36 channels
  Layer 1: 36 → 48 channels
  ...
  Layer 5: 84 → 96 channels
  Block output: 96 channels
  After compression: 48 channels
```

### 실험 3 — 파라미터 비교

```python
def compare_parameters():
    # ResNet-50 style
    resnet_params = {
        'conv1': 9 * 64,
        'layer1': 3 * (64*64*9 + 64*64*1) * 2,
        'layer2': 4 * (128*128*9 + 128*64*1 + 64*128*1),
        'layer3': 6 * (256*256*9 + 256*128*1 + 128*256*1),
        'layer4': 3 * (512*512*9 + 512*256*1 + 256*512*1),
        'fc': 512 * 1000
    }
    
    total_resnet = sum(resnet_params.values())
    
    print(f"ResNet-50 estimated params: {total_resnet/1e6:.1f}M")
    print(f"DenseNet-121 params: ~7.0M")
    print(f"Ratio: {total_resnet/1e6 / 7.0:.1f}x")

compare_parameters()
```

---

## 🔗 이론과 실전의 간극

### 메모리 오버헤드

DenseNet의 문제:
- 모든 intermediate 보관 (gradient 계산용)
- 각 layer가 concatenation으로 커짐
- 실제 메모리 사용은 ResNet과 유사할 수 있음

해결책:
- Sequential inference
- Checkpointing (일부 feature 재계산)

### 계산 복잡도

Concatenation 비용:
- Theoretical parameter는 적지만
- Actual computation은 더 많을 수 있음
- GPU VRAM bandwidth 병목

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Growth rate가 작음 | 큰 $k$는 파라미터 이점 상쇄 |
| Concatenation 비용 무시 | 실제로는 메모리 bandwidth 고려 필요 |
| Batch Normalization 필수 | BN 없이는 성능 저하 |

---

## 📌 핵심 정리

$$\boxed{x_l = [x_0, x_1, \ldots, x_{l-1}] \text{ — 모든 이전 층과 연결}}$$

| 개념 | 설명 |
|------|------|
| **Dense Block** | 각 층이 이전 모든 층을 concatenation으로 수용 |
| **Growth Rate** | 각 층의 output channels (보통 12, 32) |
| **Bottleneck** | $1 \times 1 \to 3 \times 3$ 구조로 파라미터 제어 |
| **Transition** | Compression으로 채널 수 감소 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Dense block ($c_0=24, k=12, L=6$)에서 마지막 layer 입력 채널 수는?

<details>
<summary>힌트 및 해설</summary>

$C_l = c_0 + l \times k = 24 + l \times 12$

Layer 5 입력: $24 + 5 \times 12 = 84$ channels

</details>

**문제 2** (심화): Bottleneck ratio를 2k 대신 4k로 하는 이유는?

<details>
<summary>힌트 및 해설</summary>

4k로 확장:
- Feature interaction space 증가
- 3x3 conv의 효과 향상
- 경험적으로 최적

He et al. extensive experiments로 4k 확인.

</details>

**문제 3** (이론-실전): DenseNet이 ResNet보다 메모리를 더 쓸 수 있는 이유는?

<details>
<summary>힌트 및 해설</summary>

1. 모든 intermediate feature 보관
2. Concatenation으로 크기 증가
3. Backward pass에서 모든 이전 layer 필요

Checkpointing으로 해결 가능.

</details>

---

<div align="center">

[◀ 이전](./03-identity-approximation.md) | [📚 README](../README.md) | [다음 ▶](./05-highway.md)

</div>
