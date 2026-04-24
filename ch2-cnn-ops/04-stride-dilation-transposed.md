# 04. Stride, Dilation, Transposed Convolution

## 🎯 핵심 질문

- Stride $s > 1$이 receptive field를 곱셈적으로 확장하는 이유는?
- Dilated convolution (atrous convolution)의 공식 $(I *_d K)[i] = \sum_m K[m] I[i + dm]$은 무엇을 의미하는가?
- Transposed convolution이 왜 "역 convolution"이 아니라 **"convolution이 아닌 다른 선형 변환"**인가?
- Odena 2016의 checkerboard artifact는 stride와 kernel size의 관계에서 왜 생기는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Convolution은 기본적으로 공간 해상도를 감소시킵니다. 하지만 많은 작업에서 **공간을 확대**해야 합니다 (예: image upsampling, super-resolution, semantic segmentation). 또한 receptive field를 빠르게 확대하려면 stride를 사용해야 합니다.

이 장에서 다루는 세 기법은 이러한 문제를 해결합니다:

1. **Stride**: Downsampling과 receptive field 확대.
2. **Dilation**: Receptive field 확대 without downsampling.
3. **Transposed Convolution**: Upsampling.

이들은 최신 CNN 아키텍처 (ResNet, U-Net, DeepLab)의 필수 요소입니다. 이 문서는 **각 기법의 수학적 정의, 차이점, 그리고 artifact 문제**를 정확히 다룹니다.

---

## 📐 수학적 선행 조건

- [Ch2-01: Convolution Forward/Backward](./01-conv-forward-backward.md): Convolution 기초, chain rule
- [Ch2-02: Pooling](./02-pooling.md): Downsampling, receptive field
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Strided indexing, transposed operations
- [Signal Processing Deep Dive](https://github.com/iq-ai-lab/signal-processing-deep-dive): Upsampling, decimation

---

## 📖 직관적 이해

### Stride: Skip every s-th position

일반 convolution은 kernel을 이미지 위에 1픽셀씩 이동시킵니다. Stride $s > 1$이면 $s$ 픽셀씩 이동합니다.

$$Y[i, j] = \sum_{m, n} K[m, n] \cdot X[i \cdot s + m, j \cdot s + n]$$

**효과**:
1. Output size가 $1/s$로 감소 (downsampling).
2. 다음 layer의 1픽셀이 이전 layer의 $s \times s$ 영역을 봄 (receptive field *= $s$).
3. 계산량이 약 $1/s^2$로 감소.

**문제점**: Pooling과 달리 information이 "skipped" — 모든 위치의 정보를 사용하지 않음.

### Dilation (Atrous Convolution): Sparse kernel

Stride 대신, kernel 자체의 원소들을 sparse하게 배치합니다:

$$(I *_d K)[i] := \sum_{m=0}^{K-1} K[m] \cdot I[i + d \cdot m]$$

여기서 $d$는 dilation rate. $d=1$이면 일반 convolution, $d=2$이면 kernel의 원소들 사이에 1개 gap.

**효과**:
1. Output size는 유지 (downsampling 없음).
2. Receptive field는 $K + (K-1)(d-1) = K + Kd - d = K(1 + d) - d$ (stride와 유사하게 확대).
3. 모든 입력 위치의 정보를 (다르게) 사용.

**적용**: Semantic segmentation에서 pooling으로 resolution을 잃지 않으면서도 receptive field를 확대.

### Transposed Convolution: "Backward" convolution?

Transposed convolution은 이름과 달리 convolution의 역연산이 **아닙니다**. 대신, convolution의 backward pass를 forward로 사용합니다.

만약 forward convolution이 $Y = W * X$ (linear 연산)이면:

$$\frac{\partial L}{\partial X} = (W^T *_{\text{full}} \frac{\partial L}{\partial Y})$$

Transposed convolution은 이를 forward로 재해석하여 **upsampling**을 수행합니다.

**주의**: Transposed convolution은 convolution의 정확한 역이 아닙니다. 정보 손실이 있기 때문입니다.

### Checkerboard Artifact (Odena 2016)

Stride > 1인 convolution 후 upsampling (또는 transposed convolution)하면, kernel size가 stride로 나누어떨어지지 않을 때 **체크판 패턴** artifact가 생깁니다.

**원인**: Upsampling 시 각 pixel이 overlapping하는 receptive field 크기가 불균일. 예를 들어 stride 2, kernel 3이면, 일부 output pixel은 2개, 일부는 1개의 upsampled pixel에 의해 정의됨.

**해결책**:
1. Kernel size를 stride의 배수로 설정 (예: stride 2이면 kernel 4).
2. Bilinear upsampling + convolution 순서 사용.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.14 — Strided Convolution

Input $X \in \mathbb{R}^{C \times H \times W}$, kernel $K \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$, stride $s$:

$$Y[c_o, i, j] := \sum_{c_i, m, n} K[c_o, c_i, m, n] \cdot X[c_i, i \cdot s + m, j \cdot s + n]$$

Output shape: $H_{out} = \lfloor (H - K_h) / s \rfloor + 1$.

### 정의 2.15 — Dilated Convolution (1D 표기)

$$(I *_d K)[i] := \sum_{m=0}^{K-1} K[m] \cdot I[i + d \cdot m]$$

2D로 확장:

$$Y[c_o, i, j] := \sum_{c_i, m, n} K[c_o, c_i, m, n] \cdot X[c_i, i + d_h \cdot m, j + d_w \cdot n]$$

Output size: 일반 convolution과 동일 (downsampling 없음).

### 정리 2.16 — Receptive Field with Dilation

Dilation rate $d$, kernel size $K$일 때:

$$\text{RF} = K + (K - 1)(d - 1) = 1 + K \cdot d - d$$

증명: Kernel의 가장 먼 두 원소 사이의 거리 = $(K-1) \cdot d + 1$. $\square$

### 정의 2.17 — Transposed Convolution

Stride $s$인 convolution의 형식적 adjoint:

$$Y[i, j] := \sum_{c_i, m, n} K[c_o, c_i, m, n] \cdot X[c_i, (i - m) / s, (j - n) / s]$$

(단, indices가 정수이고 유효한 범위에서만)

더 정확하게는, padded upsampled input에 대한 convolution:

$$X_{\text{up}}[c, i', j'] := \begin{cases} X[c, i'/s, j'/s] & \text{if } s | i' \text{ and } s | j' \\ 0 & \text{otherwise} \end{cases}$$

$$Y = \text{conv}(X_{\text{up}}, K, \text{padding}, \text{stride}=1)$$

### 정리 2.18 — Checkerboard Artifact 조건

Stride $s$, kernel size $K$일 때, checkerboard artifact가 심한 경우:

$$\gcd(s, K) < s$$

즉, kernel size가 stride의 배수가 아닐 때. $\square$

---

## 🔬 증명 또는 수학적 유도

### Receptive Field 계산: Stride vs Dilation

**Stride $s=2$, kernel $3 \times 3$:**

Forward에서 output[0]은 input[0:3]을 봅니다. Output[1]은 input[2:5]를 봅니다.

따라서 한 output pixel = input의 3픽셀 range. 하지만 다음 layer에서는 output[0]과 [1]이 모두 현재 output pixel과 연결되므로, 실제 RF = (3-1)*2 + 1 = 5. (stride의 곱셈적 효과)

**Dilation $d=2$, kernel $3 \times 3$:**

Output = $\sum K[m] \cdot X[i + 2m]$.

Kernel indices: $m = 0, 1, 2$ → input indices: $i, i+2, i+4$ (range = 5).

따라서 RF = $1 + 2 \cdot (3-1) = 5$. 

결과적으로 RF는 유사하지만, **stride는 downsampling**, **dilation은 spatial resolution 유지**. $\square$

### Transposed Convolution의 수학

Forward convolution을 Jacobian으로 쓰면:

$$\frac{\partial Y_{ij}}{\partial X_{kl}} = K[\text{offset}(i, j, k, l)]$$

Backward (adjoint):

$$\frac{\partial L}{\partial X} = K^T @ \frac{\partial L}{\partial Y}$$

Transposed convolution은 이를 forward로 해석 (다시 convolution 형태로). $\square$

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Stride의 효과

```python
import torch
import torch.nn.functional as F

X = torch.randn(1, 3, 16, 16)
K = torch.randn(16, 3, 3, 3)

# stride = 1
Y_s1 = F.conv2d(X, K, padding=1, stride=1)

# stride = 2
Y_s2 = F.conv2d(X, K, padding=1, stride=2)

# stride = 4
Y_s4 = F.conv2d(X, K, padding=1, stride=4)

print("Input shape:", X.shape)
print("Stride 1 output shape:", Y_s1.shape)
print("Stride 2 output shape:", Y_s2.shape)
print("Stride 4 output shape:", Y_s4.shape)

# Receptive field 확인
# RF를 명시적으로 계산하지 말고, 각 output pixel이 몇개 input을 skip했는지 확인
print("\nSkip amount (stride 지표):")
print("Stride 2 → 매 2 픽셀 skip")
print("Stride 4 → 매 4 픽셀 skip")
```

**예상**: Output size가 stride에 반비례.

### 실험 2 — Dilation 비교

```python
# Dilation rate의 영향
X = torch.randn(1, 3, 16, 16)
K = torch.randn(8, 3, 3, 3)

Y_d1 = F.conv2d(X, K, padding=1, dilation=1)
Y_d2 = F.conv2d(X, K, padding=2, dilation=2)  # padding도 dilation에 맞춰야 함
Y_d4 = F.conv2d(X, K, padding=4, dilation=4)

print("Input shape:", X.shape)
print("Dilation 1 output shape:", Y_d1.shape)
print("Dilation 2 output shape (padding=2):", Y_d2.shape)
print("Dilation 4 output shape (padding=4):", Y_d4.shape)

# 모두 같은 spatial size → stride와 다름!
print("\nAll dilation outputs have same spatial size (no downsampling)")
```

**예상**: 모든 dilation 출력이 같은 공간 크기 (padding을 조정했으므로).

### 실험 3 — Transposed Convolution (Upsampling)

```python
X = torch.randn(1, 16, 8, 8)
K = torch.randn(3, 16, 3, 3)

# Transposed convolution: stride 2, kernel 3
Y_trans = F.conv_transpose2d(X, K, stride=2, padding=1, output_padding=1)

print("Input shape (downsampled):", X.shape)
print("Transposed conv output shape:", Y_trans.shape)
print("Expected ~16x16 (2배 upsampling)")

# 비교: naive upsampling + convolution
X_up = F.interpolate(X, scale_factor=2, mode='nearest')
Y_up_conv = F.conv2d(X_up, K, padding=1)
print("Naive upsampling + conv output shape:", Y_up_conv.shape)
print("(Transposed conv와는 달 수 있음 - output_padding 때문)")
```

**예상**: Transposed convolution이 spatial size를 증가.

### 실험 4 — Checkerboard Artifact

```python
# Stride와 kernel size의 불일치로 artifact 생성
import matplotlib.pyplot as plt

# Stride 2, kernel 3 (GCD(2,3) = 1 < 2) → artifact 예상
X = torch.ones(1, 1, 1, 1)
K1 = torch.ones(1, 1, 3, 3)

# Transposed conv로 upsampling
Y_trans1 = F.conv_transpose2d(X, K1, stride=2, padding=1)
print("Stride 2, Kernel 3 output:\n", Y_trans1[0, 0])
print("(Uneven overlapping → checkerboard pattern 가능성)")

# Stride 2, kernel 4 (GCD(2,4) = 2 = stride) → artifact 감소
K2 = torch.ones(1, 1, 4, 4)
Y_trans2 = F.conv_transpose2d(X, K2, stride=2, padding=1)
print("\nStride 2, Kernel 4 output:\n", Y_trans2[0, 0])
print("(More uniform overlapping)")

# 시각화 (큰 입력에서)
X_large = torch.randn(1, 16, 4, 4)
K_bad = torch.randn(3, 16, 3, 3)
K_good = torch.randn(3, 16, 4, 4)

Y_bad = F.conv_transpose2d(X_large, K_bad, stride=2, padding=1)
Y_good = F.conv_transpose2d(X_large, K_good, stride=2, padding=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(Y_bad[0, 0].detach(), cmap='gray')
axes[0].set_title('Stride 2, Kernel 3 (Artifact!)')
axes[1].imshow(Y_good[0, 0].detach(), cmap='gray')
axes[1].set_title('Stride 2, Kernel 4 (Better)')
plt.show()
```

---

## 🔗 이론과 실전의 간극

### Stride vs Dilation Trade-off

**Stride**: 
- 장점: 계산량 감소, RF 빠르게 확대.
- 단점: 정보 손실, checkerboard artifact.

**Dilation**:
- 장점: 정보 손실 없음, flexible RF.
- 단점: 계산량은 변 없음, grid artifacts (일부 위치는 skip됨).

### Modern Architectures

- **ResNet**: Stride 2로 downsampling + residual connections.
- **WaveNet** (van den Oord et al. 2016): Causal dilated conv를 **exponential dilation** ($d_l = 2^{l-1}$)로 쌓아 $L$층만으로 RF $= 2^L$ 달성. $k=2$ causal conv 30층 → $2^{30} \approx 10^9$ sample (약 1분 오디오)를 단일 output이 바라봄. Ch3-03에서 WaveNet의 RF 확장을 정밀 분석.
- **Atrous Spatial Pyramid Pooling (ASPP)**: 다양한 dilation으로 multi-scale RF.
- **U-Net**: Stride downsampling + transposed conv upsampling + skip connections.

### Checkerboard Artifact 해결

실제 구현에서:
1. Stride/kernel size를 신중하게 설정.
2. Bilinear interpolation + convolution 사용.
3. Sub-pixel convolution (shuffling after upsampling).

---

## ⚖️ 가정과 한계

| 가정 | 설명 | 한계 |
|------|------|------|
| Square kernels | 정사각형 kernel | Rectangular kernel도 가능 |
| Uniform stride/dilation | 모든 방향 같은 값 | 비대칭 stride도 가능하지만 드물음 |
| Transposed = adjoint | Transposed conv가 정확한 역 | Information이 손실되므로 정확한 역 아님 |
| Regular grid | 정규 격자 입력 | Sparse/irregular input은 다른 방식 |

---

## 📌 핵심 정리

| 개념 | 공식 | 효과 |
|------|------|------|
| **Stride** | $Y[i,j] = \sum K[m,n] X[i \cdot s + m, j \cdot s + n]$ | Downsampling (1/$s^2$ size) + RF *= $s$ |
| **Dilation** | $Y[i,j] = \sum K[m,n] X[i + d \cdot m, j + d \cdot n]$ | RF = $K + (K-1)(d-1)$, no downsampling |
| **Transposed Conv** | Upsampling (backward convolution 형태) | $s$배 upsampling, checkerboard artifact 주의 |
| **Checkerboard** | $\gcd(s, K) < s$ | Stride의 배수 kernel 사용으로 완화 |

---

## 🤔 생각해볼 문제

**문제 1 (기초).** Stride 2, kernel 3으로 $8 \times 8$ 입력을 convolution하면 output size는?

<details>
<summary>해설</summary>

$H_{out} = \lfloor (8 - 3) / 2 \rfloor + 1 = \lfloor 5/2 \rfloor + 1 = 2 + 1 = 3$.

Output: $3 \times 3$.

</details>

**문제 2 (심화).** Dilation 2, kernel 3에서 'same' convolution을 하려면 padding은 몇일까?

<details>
<summary>해설</summary>

Effective kernel size (dilation 고려) = $K + (K-1)(d-1) = 3 + 2 \cdot 2 = 7$.

'Same' padding: $p = (7 - 1) / 2 = 3$.

따라서 padding = 3.

PyTorch에서는: `F.conv2d(X, K, padding=3, dilation=2)`.

</details>

**문제 3 (논문 비평).** Odena 2016 "Deconvolution and Checkerboard Artifacts"에서 제시한 checkerboard artifact의 근본 원인을 설명하고, 두 가지 해결책을 제시하라.

<details>
<summary>해설</summary>

**원인**: Transposed convolution에서 upsampled input의 각 pixel이 output과 다양한 정도로 overlap됨. Stride와 kernel size가 맞지 않으면 일부 output pixel은 적게, 일부는 많이 cover됨.

**해결책**:
1. **Resize-convolution**: 먼저 interpolation (bilinear)으로 target size로 upsample, 그 후 일반 convolution.
   - 예: `F.interpolate(X, scale_factor=2, mode='bilinear') → F.conv2d(..., stride=1)`.

2. **Stride 조정**: Kernel size를 stride의 배수로 설정.
   - 예: stride 2이면 kernel 4, stride 3이면 kernel 6.

대부분 최신 모델은 resize-convolution을 선호합니다 (더 균일한 gradient).

</details>

---

<div align="center">

[◀ 이전](./03-padding.md) | [📚 README](../README.md) | [다음 ▶](./05-depthwise-separable.md)

</div>
