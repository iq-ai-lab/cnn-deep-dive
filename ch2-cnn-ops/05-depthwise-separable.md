# 05. Depthwise Separable Convolution

## 🎯 핵심 질문

- Standard convolution의 계산량 $O(K^2 C_{in} C_{out} H W)$을 depthwise separable convolution이 어떻게 분해하는가?
- Depthwise ($k \times k$, 채널별 독립) + Pointwise ($1 \times 1$, 채널 혼합)의 두 단계는 무엇을 하는가?
- 연산량 절감 비율 $\frac{1}{C_{out}} + \frac{1}{k^2}$는 언제 유리한가?
- MobileNet, EfficientNet, ConvNeXt에서 depthwise separable이 어떻게 재해석되었는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

2010년대 중후반부터 CNN은 깊어지고 커졌습니다. AlexNet(8층)에서 ResNet-152(152층), EfficientNet(수백층 equivalent)으로. 하지만 이렇게 커진 모델을 실제 모바일 기기나 임베디드 시스템에 배포하기는 어렵습니다.

Depthwise separable convolution (Chollet 2017, Xception)은 **같은 성능을 유지하면서 모델 크기와 계산량을 급격히 감소**시킵니다. MobileNet V1-V3, EfficientNet, ConvNeXt 등 최신 경량 모델의 핵심입니다. 

이 문서에서는 depthwise separable의 수학적 정의, 효율성 분석, 그리고 이후 모델들이 어떻게 이를 재해석했는지를 이해합니다.

---

## 📐 수학적 선행 조건

- [Ch2-01: Convolution Forward/Backward](./01-conv-forward-backward.md): Multi-channel convolution, backward
- [Ch2-04: Stride, Dilation, Transposed](./04-stride-dilation-transposed.md): Receptive field, depthwise concept 선행
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Asymptotic complexity, Big-O
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Rank, tensor decomposition

---

## 📖 직관적 이해

### Standard Convolution의 비효율

Standard $3 \times 3$ convolution, $C_{in} = 64$ input channel, $C_{out} = 128$ output channel일 때:

$$\text{FLOPs} = K^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W = 9 \cdot 64 \cdot 128 \cdot H \cdot W = 73,728 \cdot H \cdot W$$

이는 매우 많습니다. 특히 모바일 기기에서는 전력 소비와 배터리 수명 문제가 됩니다.

### Separable Convolution: 두 단계로 분해

Depthwise separable convolution은 이를 두 개의 더 효율적인 연산으로 분해합니다:

**1단계: Depthwise Convolution (채널별 독립 convolution)**

각 input channel에 대해 **독립적으로** $K \times K$ convolution을 수행합니다:

$$Y[c, i, j] = \sum_{m, n} K[c, m, n] \cdot X[c, i+m, j+n] + b[c]$$

Output channel수 = input channel수. Kernel shape: $(C_{in}, 1, K, K)$.

$$\text{FLOPs} = K^2 \cdot C_{in} \cdot H \cdot W = 9 \cdot 64 \cdot H \cdot W$$

**2단계: Pointwise Convolution (채널 혼합)**

$1 \times 1$ convolution으로 channels를 혼합합니다:

$$Z[c_o, i, j] = \sum_{c_i} W[c_o, c_i] \cdot Y[c_i, i, j] + b'[c_o]$$

Kernel shape: $(C_{out}, C_{in}, 1, 1)$.

$$\text{FLOPs} = 1 \cdot C_{in} \cdot C_{out} \cdot H \cdot W = 64 \cdot 128 \cdot H \cdot W$$

**총 FLOPs:**

$$\text{Depthwise Separable} = (K^2 C_{in} + C_{in} C_{out}) H W = (9 \cdot 64 + 64 \cdot 128) H W = 8,640 \cdot H W$$

**효율성 비율:**

$$\frac{\text{Separable FLOPs}}{\text{Standard FLOPs}} = \frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

위 예제: $\frac{1}{128} + \frac{1}{9} \approx 0.008 + 0.111 = 0.119 \approx 1/8.4$.

즉, **약 8배 계산량 감소**!

### 인수분해 관점

Depthwise separable convolution은 3D 텐서 (또는 4D including batch)를 다음처럼 인수분해합니다:

$$K^{\text{sep}} \approx K^{\text{dw}} \otimes K^{\text{pw}}$$

여기서 $\otimes$는 적절한 tensor product. 이는 **low-rank approximation**으로도 해석할 수 있습니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.19 — Depthwise Convolution

Input $X \in \mathbb{R}^{C \times H \times W}$, depthwise kernel $K_{\text{dw}} \in \mathbb{R}^{C \times 1 \times K_h \times K_w}$:

$$Y[c, i, j] := \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} K_{\text{dw}}[c, 0, m, n] \cdot X[c, i+m, j+n] + b[c]$$

각 channel $c$는 독립적으로 처리. Output shape: $C \times H' \times W'$ (stride/padding에 따라).

### 정의 2.20 — Pointwise Convolution

$Y \in \mathbb{R}^{C_{in} \times H \times W}$에서 시작, pointwise kernel $K_{\text{pw}} \in \mathbb{R}^{C_{out} \times C_{in} \times 1 \times 1}$:

$$Z[c_o, i, j] := \sum_{c_i=0}^{C_{in}-1} K_{\text{pw}}[c_o, c_i, 0, 0] \cdot Y[c_i, i, j] + b'[c_o]$$

Output shape: $C_{out} \times H \times W$ (spatial size 유지).

### 정의 2.21 — Depthwise Separable Convolution

Depthwise → Pointwise의 순차 적용. 전체 효과:

$$Z[c_o, i, j] = \sum_{c_i} K_{\text{pw}}[c_o, c_i] \left( \sum_{m, n} K_{\text{dw}}[c_i, m, n] X[c_i, i+m, j+n] \right) + b'[c_o]$$

### 정리 2.22 — 계산량 감소 비율

Standard convolution (FLOPs per output element): $K^2 C_{in} C_{out}$.

Depthwise separable (FLOPs per output element): $K^2 C_{in} + C_{in} C_{out}$.

감소 비율:

$$r = \frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

증명: 직접 FLOPs 계산. $\square$

### 정리 2.23 — Depthwise Separable의 Expressivity

Depthwise separable은 standard convolution을 **항상 표현 가능**합니다.

증명: Pointwise에서 rank를 충분히 크게 하면 (입력의 차원까지) 임의의 채널 혼합이 가능. Depthwise는 spatial pattern을 추출, pointwise는 이들을 선형결합하므로, 적절한 가중치를 선택하면 standard convolution을 재현 가능. $\square$

---

## 🔬 증명 또는 수학적 유도

### FLOPs 계산 상세

**Standard $K \times K$ convolution:**
- 각 output pixel마다: $K^2$ spatial positions $\times$ $C_{in}$ input channels = $K^2 C_{in}$ 곱셈.
- 이를 $C_{out}$ output channels로 반복.
- 총 output pixels: $H W$.
- 총 FLOPs: $K^2 C_{in} C_{out} H W$.

**Depthwise separable:**
- Depthwise: 각 output pixel마다 $K^2 C_{in}$ (모든 channel이 독립이지만 같은 kernel size).
- 총 depthwise FLOPs: $K^2 C_{in} H W$.
- Pointwise: 각 output pixel마다 $C_{in}$ 곱셈 $\times$ $C_{out}$ output channels.
- 총 pointwise FLOPs: $C_{in} C_{out} H W$.
- 총합: $(K^2 C_{in} + C_{in} C_{out}) H W$. $\square$

### Expressivity 증명 스케치

$X \in \mathbb{R}^{C_{in} \times H \times W}$에서 depthwise는 각 channel을 개별 $K \times K$ convolution으로 처리:

$$Y = \text{Depthwise}(X) \in \mathbb{R}^{C_{in} \times H \times W}$$

각 spatial position $(i, j)$마다 $C_{in}$ 차원 벡터가 있습니다.

Pointwise는 이를 dense layer로 처리:

$$Z[c_o, :, :] = W[c_o, :] @ Y[:, :, :]$$

여기서 $W[c_o, :]$는 $C_{in}$ 차원 가중치 벡터.

Standard convolution $K^2 C_{in} C_{out}$ 파라미터를 표현하려면, depthwise ($K^2 C_{in}$) + pointwise ($C_{in} C_{out}$) = 합: $K^2 C_{in} + C_{in} C_{out} < K^2 C_{in} C_{out}$ (보통).

하지만 **표현 능력**은 동일합니다. 왜냐하면 pointwise의 rank가 충분하면, 임의의 mixing이 가능하기 때문입니다. $\square$

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — FLOPs 비교

```python
import torch
import torch.nn as nn

def count_conv_flops(in_channels, out_channels, kernel_size, h, w, stride=1):
    """
    Convolution FLOPs 계산 (forward pass)
    """
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    flops = kernel_size ** 2 * in_channels * out_channels * out_h * out_w
    return flops

# 예제: MobileNet처럼 small model
C_in, C_out, K = 64, 128, 3
H, W = 32, 32

# Standard convolution
flops_std = count_conv_flops(C_in, C_out, K, H, W)

# Depthwise + Pointwise
flops_dw = count_conv_flops(C_in, C_in, K, H, W)  # Depthwise는 output channel = input channel
flops_pw = count_conv_flops(C_in, C_out, 1, H, W)  # Pointwise
flops_sep = flops_dw + flops_pw

print(f"Standard Conv FLOPs: {flops_std:,}")
print(f"Depthwise FLOPs: {flops_dw:,}")
print(f"Pointwise FLOPs: {flops_pw:,}")
print(f"Depthwise Separable Total: {flops_sep:,}")
print(f"Reduction ratio: {flops_sep / flops_std:.3f}")
print(f"Speedup: {flops_std / flops_sep:.2f}x")

# 이론적 비율
theoretical_ratio = 1 / C_out + 1 / K**2
print(f"\nTheoretical ratio: {theoretical_ratio:.3f}")
```

**예상 결과**: 약 8-9배 speedup.

### 실험 2 — PyTorch에서 Depthwise Separable 구현

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Test
in_ch, out_ch = 32, 64
model = DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, padding=1)

X = torch.randn(2, in_ch, 28, 28)
Y = model(X)

print("Input shape:", X.shape)
print("Output shape:", Y.shape)
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} ({param.numel()} params)")

# 파라미터 개수 비교
total_params = sum(p.numel() for p in model.parameters())
std_params = in_ch * out_ch * 3 * 3 + out_ch  # Standard conv
sep_params = (in_ch * 1 * 3 * 3 + in_ch) + (in_ch * out_ch * 1 + out_ch)  # Sep conv
print(f"\nTotal sep params: {total_params}")
print(f"Standard equiv params: {std_params}")
print(f"Reduction: {std_params / total_params:.2f}x")
```

**예상**: Depthwise separable이 훨씬 적은 파라미터.

### 실험 3 — Backward 검증

```python
# Gradient flow 확인
X = torch.randn(2, 32, 16, 16, requires_grad=True)
model = DepthwiseSeparableConv(32, 64)

Y = model(X)
loss = Y.sum()
loss.backward()

print("Input gradient shape:", X.grad.shape)
print("Input gradient norm:", X.grad.norm().item())
print("Depthwise grad norm:", model.depthwise[0].weight.grad.norm().item())
print("Pointwise grad norm:", model.pointwise[0].weight.grad.norm().item())
print("\nBackward pass successful!")
```

### 실험 4 — MobileNet style block

```python
class MobileNetBlock(nn.Module):
    """
    MobileNet V1-style block: Depthwise Separable + ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sep_conv = DepthwiseSeparableConv(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.sep_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Build small MobileNet
blocks = nn.Sequential(
    MobileNetBlock(3, 32, stride=1),
    MobileNetBlock(32, 64, stride=2),
    MobileNetBlock(64, 128, stride=1),
    MobileNetBlock(128, 128, stride=2),
)

X = torch.randn(2, 3, 224, 224)
Y = blocks(X)

print("Input shape:", X.shape)
print("Output shape:", Y.shape)
print("Model size: {:.2f}M parameters".format(
    sum(p.numel() for p in blocks.parameters()) / 1e6
))
```

---

## 🔗 이론과 실전의 간극

### MobileNet의 진화

**MobileNet V1 (2017):**
- Depthwise separable을 주 빌딩블록으로 사용.
- Width multiplier, resolution multiplier로 크기 조정.

**MobileNet V2 (2018):**
- **Inverted Residual Block**: Pointwise로 expansion, depthwise로 processing, pointwise로 projection.
  - Standard: $C_{in} \to C_{mid} \to C_{out}$ (중간에 많음).
  - Inverted: $C_{in} \to C_{exp} \to C_{out}$ (처음엔 expand, 나중에 compress).
  - Mobile device에서 더 효율적 (메모리 접근 패턴).

**MobileNet V3 (2019):**
- Neural Architecture Search (NAS)로 최적 구조 탐색.
- Squeeze-and-Excitation (SE) blocks 추가.

### ConvNeXt와의 재해석

2022년 ConvNeXt는 modern vision transformer의 성능을 CNN으로 달성하려 시도.

이 과정에서:
- Depthwise $7 \times 7$ convolution을 중앙에 배치 (넓은 receptive field).
- Inverted bottleneck structure 활용.
- Batch norm 대신 LayerNorm.

이는 depthwise separable의 원리를 확장한 것입니다.

### 실전 Trade-off

**Depthwise separable의 장점:**
- 계산량 급감.
- 파라미터 수 감소 → 메모리 효율, 빠른 inference.

**Depthwise separable의 단점:**
- 각 channel이 독립적으로 처리 → channel 간 정보 교환이 pointwise에만 의존.
- Early layers에서는 채널이 적어서 효율 감소 (예: $C_{in} = 3$).
- Pointwise가 bottleneck이 될 수 있음.

**해결책**: Inverted residual, channel shuffling (ShuffleNet), grouped convolution 등.

---

## ⚖️ 가정과 한계

| 가정 | 설명 | 한계 |
|------|------|------|
| Channel independence 적절 | Depthwise가 spatial features 추출 | 초기 layer (채널 적음)에서는 비효율 |
| Spatial smoothness | 이미지 픽셀들이 국소적 상관 | 랜덤 패턴이나 구조화되지 않은 데이터는 덜 유리 |
| Pointwise rank sufficiency | 1x1 convolution이 충분한 표현력 | 매우 복잡한 채널 상호작용은 부족할 수 있음 |
| Depthwise 먼저 | Depthwise 후 pointwise 순서 | Inverted (pointwise expansion) 가능 |

---

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| **Depthwise Conv** | 각 channel별 독립 $K \times K$ convolution |
| **Pointwise Conv** | $1 \times 1$ convolution으로 channels 혼합 |
| **Depthwise Separable** | Depthwise + Pointwise 순차 적용 |
| **FLOPs 감소** | $K^2 C_{in} + C_{in} C_{out}$ vs $K^2 C_{in} C_{out}$ |
| **Reduction Ratio** | $r = \frac{1}{C_{out}} + \frac{1}{K^2}$ |
| **Expressivity** | Standard conv를 항상 표현 가능 (rank 충분할 때) |

---

## 🤔 생각해볼 문제

**문제 1 (기초).** Depthwise separable convolution에서 $C_{in}=32, C_{out}=64, K=3$일 때 FLOPs 감소 비율을 계산하라.

<details>
<summary>해설</summary>

Reduction ratio: $r = \frac{1}{C_{out}} + \frac{1}{K^2} = \frac{1}{64} + \frac{1}{9} \approx 0.0156 + 0.111 = 0.127$.

즉, standard conv의 약 12.7%, 약 **8배 감소**.

더 정확히: Standard = $9 \times 32 \times 64 = 18,432$ FLOPs/pixel.

Depthwise sep = $(9 \times 32 + 32 \times 64) = 288 + 2,048 = 2,336$ FLOPs/pixel.

감소율: $18,432 / 2,336 \approx 7.88$x.

</details>

**문제 2 (심화).** Inverted residual block (MobileNet V2)에서 expansion ratio 6일 때, $C_{in}=64, C_{out}=64$의 구조를 그리고 FLOPs를 계산하라.

<details>
<summary>해설</summary>

구조: $C_{in}=64 \xrightarrow{\text{pw expand}} C_{exp}=64 \times 6=384 \xrightarrow{\text{dw}} 384 \xrightarrow{\text{pw project}} C_{out}=64$.

FLOPs:
- Pointwise expand: $1 \times 1 \times 64 \times 384 = 24,576$ FLOPs/pixel.
- Depthwise: $3 \times 3 \times 384 = 3,456$ FLOPs/pixel.
- Pointwise project: $1 \times 1 \times 384 \times 64 = 24,576$ FLOPs/pixel.
- 총합: $52,608$ FLOPs/pixel.

Standard bottleneck (같은 채널 변화): $\approx$ expansion-depthwise-projection의 여러 조합. Inverted의 장점은 메모리 접근 패턴 개선입니다.

</details>

**문제 3 (논문 비평).** Depthwise separable convolution이 "separable"이라고 불리는 이유와, 이것이 rank decomposition과 어떤 관계가 있는지 설명하라.

<details>
<summary>해설</summary>

**Separable 이유**: Spatial 연산 (depthwise $K \times K$)과 channel 연산 ($1 \times 1$)이 **분리 가능**하기 때문입니다. Standard convolution은 이 둘이 강하게 얽혀 있습니다.

**Rank decomposition 관점**:
Standard convolution kernel $K \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$는 4D tensor입니다.

Depthwise separable은 이를 두 개 tensor의 "곱"으로 분해:
- Depthwise: $K_d \in \mathbb{R}^{C_{in} \times K_h \times K_w}$ (각 channel마다).
- Pointwise: $K_p \in \mathbb{R}^{C_{out} \times C_{in}}$ (spatial은 없음).

이는 **low-rank factorization**입니다. Rank-1 approximation이 아니지만, tensor network 이론에서 보면 spatial과 channel 차원을 분리하는 것입니다.

따라서 "separable"은 단순히 효율성이 아니라, **tensor 구조의 분해**를 의미합니다.

</details>

---

<div align="center">

[◀ 이전](./04-stride-dilation-transposed.md) | [📚 README](../README.md) | [다음 ▶](../ch3-receptive-field/01-theoretical-rf.md)

</div>
