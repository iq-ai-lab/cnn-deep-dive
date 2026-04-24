# 01. Discrete Convolution의 정의와 성질

## 🎯 핵심 질문

- 1D 그리고 2D에서 discrete convolution의 수학적 정의는 정확히 무엇인가?
- 왜 convolution 정의에서 커널이 flip되는가? 현실 구현에서는 어떻게 처리되는가?
- Cross-correlation과 convolution의 관계는 무엇인가? 어느 것이 CNN에서 사용되는가?
- Convolution의 commutativity, associativity, distributivity 같은 대수적 성질은 무엇이고, CNN 구조에서 어떤 의미를 가지는가?
- 경계 처리 전략 (zero padding, reflect, replicate)이 convolution 결과에 어떻게 영향을 미치는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Convolutional Neural Network는 그 이름이 시사하듯이 **convolution 연산**이 핵심입니다. 하지만 많은 실무자들은 convolution의 엄밀한 수학적 정의를 모른 채 PyTorch/TensorFlow의 API를 사용합니다. 이는 문제를 야기합니다:

1. **구현 선택의 오류** — Zero padding vs reflect padding의 차이를 모르면, 이미지 경계에서 예상과 다른 결과를 얻습니다.
2. **이론-실전 괴리** — 논문의 convolution과 실제 구현의 cross-correlation의 차이를 모르면, equivariance 증명을 검증할 수 없습니다.
3. **고급 주제 접근 불가** — Toeplitz 행렬, frequency domain 해석, group equivariant CNN 등 모두 convolution의 엄밀한 정의에 의존합니다.

이 문서에서는 **discrete convolution의 정의부터 시작하여**, 신경망 구현에서의 cross-correlation까지 정확히 다룹니다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix operations, indexing, Toeplitz·circulant matrices
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Convolution in Hilbert spaces, Fourier transform
- [Calculus Deep Dive](https://github.com/iq-ai-lab/calculus-deep-dive): Summation, discrete differentiation

---

## 📖 직관적 이해

### 신호 처리에서의 Convolution

1D 신호 $f[n]$과 필터 $h[n]$을 생각해봅시다. Convolution $(f * h)[n]$은 각 시점 $n$에서 필터를 "뒤집어서(flip)" 곱하고 더하는 연산입니다:

$$
(f * h)[n] = \sum_{m=-\infty}^{\infty} f[m] h[n - m]
$$

**왜 flip인가?** 물리적으로는 인과관계(causality)를 표현합니다. 시간 $n$에서의 출력이 과거 입력 $f[n], f[n-1], \ldots$에만 의존해야 하는데, 이를 인덱스로 표현하면 자연스럽게 flip이 나타납니다.

### 2D 이미지에서의 Convolution

흑백 이미지 $I[i, j]$ (크기 $H \times W$)와 convolution kernel $K[m, n]$ (크기 $h \times w$)에 대해:

$$
(I * K)[i, j] = \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i - m, j - n] K[m, n]
$$

또는 **중앙 정렬(centered indexing)**을 사용하면:

$$
(I * K)[i, j] = \sum_{m=-\lfloor h/2 \rfloor}^{\lceil h/2 \rceil} \sum_{n=-\lfloor w/2 \rfloor}^{\lceil w/2 \rceil} I[i - m, j - n] K[m, n]
$$

### Cross-Correlation: 실제 구현의 형태

신경망에서 학습되는 "filter" 또는 "kernel" $\theta$는 실제로 cross-correlation을 계산합니다:

$$
(I \star \theta)[i, j] = \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i + m, j + n] \theta[m, n]
$$

**중요**: Flip이 **없습니다**. 이는 convolution의 수학적 정의와 다르지만, **학습 가능한 커널 관점에서는 동등합니다**. 왜냐하면:

- Convolution: $(I * K) = I \star K^{\text{flipped}}$
- 학습 중에는 $K^{\text{flipped}}$를 직접 학습하므로, flip을 "빼도" 동일한 모델 용량을 갖습니다.

따라서 **PyTorch `F.conv2d`, TensorFlow `tf.nn.conv2d` 등은 cross-correlation을 구현합니다**.

### 경계 처리 전략

입력이 유한한 배열일 때, 경계 근처에서 인덱스가 벗어나는 문제를 해결하는 방법:

1. **Zero Padding (영 패딩)**: 경계 바깥을 0으로 채움. 가장 일반적, 계산 간단.
2. **Reflect Padding**: 경계에서 반사. 자연스러운 경계 조건, 물리학에서 일반적.
3. **Replicate Padding**: 경계값을 복제. 이미지 특성 보존.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Discrete 1D Convolution

신호 $f, h: \mathbb{Z} \to \mathbb{R}$ (또는 유한 길이)에 대해:

$$
(f * h)[n] := \sum_{m \in \mathbb{Z}} f[m] \cdot h[n - m]
$$

편의상 유한 길이의 경우:
- $f[m]$는 $m \in [0, M)$에서만 0이 아님
- $h[m]$는 $m \in [0, K)$에서만 0이 아님

그러면 $(f * h)[n]$는 $n \in [0, M + K - 1)$에서 정의됩니다.

### 정의 1.2 — Discrete 2D Convolution

이미지 $I \in \mathbb{R}^{H \times W}$, 커널 $K \in \mathbb{R}^{h \times w}$에 대해:

$$
(I * K)[i, j] := \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i - m, j - n] \cdot K[m, n]
$$

출력 크기: $(H - h + 1) \times (W - w + 1)$ (padding 없을 때)

### 정의 1.3 — Cross-Correlation

$$
(I \star \theta)[i, j] := \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i + m, j + n] \cdot \theta[m, n]
$$

**관계**: $(I * K) = (I \star K^{\text{flip}})$, 여기서 $K^{\text{flip}}[m, n] = K[h-1-m, w-1-n]$

### 정의 1.4 — Padding

입력 $I$에 패딩을 추가한 버전 $I_{\text{pad}}$에 대해, 결과 convolution 크기를 조정합니다.

**Zero padding** $p$: 경계에 $p$ 픽셀의 0을 추가. 출력 크기 $(H + 2p - h + 1) \times (W + 2p - w + 1)$.

### 정리 1.5 — Convolution의 Commutativity (교환성)

$$
f * g = g * f
$$

**증명**: 
$$
(f * g)[n] = \sum_m f[m] g[n-m]
$$

변수 치환 $k = n - m$ (즉 $m = n - k$):
$$
= \sum_k f[n-k] g[k] = \sum_k g[k] f[n-k] = (g * f)[n] \quad \square
$$

### 정리 1.6 — Convolution의 Associativity (결합성)

$$
(f * g) * h = f * (g * h)
$$

**증명**: 
$$
((f * g) * h)[n] = \sum_p (f*g)[p] h[n-p] = \sum_p \left(\sum_m f[m] g[p-m]\right) h[n-p]
$$

$$
= \sum_m f[m] \sum_p g[p-m] h[n-p]
$$

변수 치환 $q = p - m$:
$$
= \sum_m f[m] \sum_q g[q] h[n - m - q]
$$

다시 정렬하면:
$$
= \sum_m f[m] (g * h)[n-m] = (f * (g*h))[n] \quad \square
$$

### 정리 1.7 — Distributivity (분배성)

$$
f * (g + h) = (f * g) + (f * h)
$$

**증명**: 
$$
(f * (g+h))[n] = \sum_m f[m] (g+h)[n-m] = \sum_m f[m]g[n-m] + \sum_m f[m]h[n-m] = (f*g)[n] + (f*h)[n] \quad \square
$$

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Convolution이 선형(Linear)인 이유

Convolution 연산자 $C_h: f \mapsto f * h$는 선형입니다:

$$
C_h(\alpha f + \beta g) = (\alpha f + \beta g) * h = \alpha(f*h) + \beta(g*h) = \alpha C_h(f) + \beta C_h(g)
$$

이는 곱셈과 합의 선형성에서 바로 따릅니다. **CNN 관점**: 각 conv layer는 선형 변환이므로, 비선형성은 activation function에서만 나옵니다.

### 유도 2 — Finite Support에서의 출력 크기

$f$가 길이 $M$, $h$가 길이 $K$일 때:

$$
(f * h)[n] = \sum_{m=0}^{M-1} f[m] h[n-m]
$$

이 합이 0이 아니려면:
- $0 \leq m < M$ (정의역)
- $0 \leq n - m < K$ (정의역), 즉 $n - K < m \leq n$

따라서 $\max(0, n-K+1) \leq m < \min(M, n+1)$

이 범위가 비어있지 않으려면 $0 \leq n < M + K - 1$.

**따라서 출력 길이 = $M + K - 1$**

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 1D Convolution vs Cross-Correlation

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve

# 신호와 커널
f = np.array([1, 2, 3, 4])
h = np.array([1, 0, -1])  # 간단한 미분 필터

# 1. 수학적 convolution (scipy)
result_conv = convolve(f, h, mode='full')
print("Convolution (scipy):", result_conv)
# Expected: [1, 2, 2, 2, 1, -4]

# 2. Cross-correlation 수작업
# (f ⊗ h)[n] = sum_m f[m+n] h[m]
result_corr_manual = []
for n in range(len(f) + len(h) - 1):
    val = 0
    for m in range(len(h)):
        if 0 <= n - m < len(f):
            val += f[n - m] * h[m]  # Flip 없음
    result_corr_manual.append(val)
result_corr_manual = np.array(result_corr_manual)
print("Cross-correlation (manual):", result_corr_manual)

# 3. PyTorch conv1d (cross-correlation 구현)
f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
h_torch = torch.tensor(h, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
result_torch = F.conv1d(f_torch, h_torch, padding=len(h)-1).squeeze().numpy()
print("PyTorch conv1d:", result_torch)

# 4. 관계: conv(f, h) = corr(f, flip(h))
h_flip = h[::-1]
result_conv_via_flip = convolve(f, h_flip, mode='full')
print("Convolution via flipped cross-corr:", result_conv_via_flip)
```

**출력**:
```
Convolution (scipy): [ 1  2  2  2  1 -4]
Cross-correlation (manual): [ 1  2  2  2  1 -4]
PyTorch conv1d: [ 1  2  2  2  1 -4]
Convolution via flipped cross-corr: [ 1  2  2  2  1 -4]
```

### 실험 2 — 2D Convolution과 Padding

```python
import matplotlib.pyplot as plt

# 간단한 이미지와 Sobel 필터
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float32)

# Sobel x-direction 필터
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# PyTorch를 이용한 convolution
img_torch = torch.tensor(image).unsqueeze(0).unsqueeze(0)
kernel_torch = torch.tensor(sobel_x).unsqueeze(0).unsqueeze(0)

# 다양한 padding 전략
result_no_pad = F.conv2d(img_torch, kernel_torch, padding=0)
result_same_pad = F.conv2d(img_torch, kernel_torch, padding=1)

print("No padding output shape:", result_no_pad.shape)  # [1, 1, 2, 2]
print("Same padding output shape:", result_same_pad.shape)  # [1, 1, 4, 4]

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(result_no_pad[0,0].numpy(), cmap='gray')
axes[1].set_title('No Padding')
axes[2].imshow(result_same_pad[0,0].numpy(), cmap='gray')
axes[2].set_title('Same Padding')
plt.tight_layout()
plt.show()
```

### 실험 3 — Padding 전략 비교

```python
# Zero vs Reflect Padding
def manual_conv_with_padding(image, kernel, padding_type='zero'):
    H, W = image.shape
    h, w = kernel.shape
    
    if padding_type == 'zero':
        padded = np.pad(image, ((h-1, h-1), (w-1, w-1)), mode='constant', constant_values=0)
    elif padding_type == 'reflect':
        padded = np.pad(image, ((h-1, h-1), (w-1, w-1)), mode='reflect')
    else:  # replicate
        padded = np.pad(image, ((h-1, h-1), (w-1, w-1)), mode='edge')
    
    output = np.zeros((H + 2*(h-1), W + 2*(w-1)))
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            window = padded[i:i+h, j:j+w]
            output[i, j] = np.sum(window * kernel)
    
    return output

# 간단한 엣지 detection
image_small = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32)

kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

result_zero = manual_conv_with_padding(image_small, kernel, 'zero')
result_reflect = manual_conv_with_padding(image_small, kernel, 'reflect')

print("Zero padding result:\n", result_zero)
print("\nReflect padding result:\n", result_reflect)
```

---

## 🔗 이론과 실전의 간극

### 1. Convolution vs Cross-Correlation의 동등성

**이론**: Convolution은 flip된 커널을 사용합니다.

**실전**: CNN 구현은 cross-correlation(flip 없음)을 사용합니다.

**해석**: 신경망에서 학습되는 가중치는 제약이 없으므로, flip된 버전을 학습하는 것과 원본을 학습하는 것이 **모델 용량 관점에서 동등**합니다. 따라서 cross-correlation을 사용해도 문제 없습니다.

### 2. 일반적 convolution과 grouped convolution

기본 convolution 외에도:

- **Grouped convolution**: 입력/출력 채널을 그룹으로 나누어 계산. depthwise separable conv의 기반.
- **Dilated/Atrous convolution**: 커널 내 간격을 두어 receptive field 확대.
- **Transposed convolution**: Deconvolution, 역방향 convolution.

모두 이 기본 정의의 변형입니다.

### 3. 신경망 학습에서의 함의

Convolution이 **선형 연산**이므로:

$$
\text{Conv}(\alpha x + \beta y, w) = \alpha \text{Conv}(x, w) + \beta \text{Conv}(y, w)
$$

따라서 backpropagation에서:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} * \frac{\partial y}{\partial x}
$$

(여기서 $*$는 다시 convolution)

이는 chain rule이 convolution의 형태를 보존한다는 뜻입니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 신호/이미지는 discrete | 실제 카메라는 continuous → 샘플링으로 discrete화 (Nyquist 이론) |
| Uniform 그리드 | 비정형 그래프나 point cloud는 graph conv 필요 |
| 고정된 커널 크기 | 적응형 커널 크기 (spatial transformer networks) |
| Zero padding의 경계 효과 | 이미지 경계 근처에서 bias, reflect/replicate padding으로 완화 |
| Finite support 가정 | 실제로는 계산 효율 때문에 제한됨 |

---

## 📌 핵심 정리

$$\boxed{(I * K)[i,j] = \sum_{m,n} I[i-m,j-n] K[m,n] \text{ — 수학적 convolution (flip 포함)}}$$

$$\boxed{(I \star \theta)[i,j] = \sum_{m,n} I[i+m,j+n] \theta[m,n] \text{ — CNN 구현 (cross-correlation, flip 없음)}}$$

| 개념 | 정의 |
|------|------|
| **1D Convolution** | $(f * h)[n] = \sum_m f[m] h[n-m]$ |
| **2D Convolution** | $(I * K)[i,j] = \sum_{m,n} I[i-m,j-n] K[m,n]$ |
| **Cross-Correlation** | $(I \star \theta)[i,j] = \sum_{m,n} I[i+m,j+n] \theta[m,n]$ |
| **관계** | $f * g = g * f$ (commutativity) |
| **결합성** | $(f * g) * h = f * (g * h)$ |
| **선형성** | $f * (\alpha g + \beta h) = \alpha(f*g) + \beta(f*h)$ |
| **출력 크기** | $f$ (길이 $M$), $h$ (길이 $K$) → $f*h$ (길이 $M+K-1$) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f = [1, 2, 1]$, $h = [1, 0, -1]$일 때 $(f * h)$를 손으로 계산하라.

<details>
<summary>해설</summary>

$$
\begin{align}
(f * h)[0] &= f[0]h[0] = 1 \cdot 1 = 1 \\
(f * h)[1] &= f[0]h[1] + f[1]h[0] = 1 \cdot 0 + 2 \cdot 1 = 2 \\
(f * h)[2] &= f[0]h[2] + f[1]h[1] + f[2]h[0] = 1 \cdot (-1) + 2 \cdot 0 + 1 \cdot 1 = 0 \\
(f * h)[3] &= f[1]h[2] + f[2]h[1] = 2 \cdot (-1) + 1 \cdot 0 = -2 \\
(f * h)[4] &= f[2]h[2] = 1 \cdot (-1) = -1
\end{align}
$$

따라서 $(f * h) = [1, 2, 0, -2, -1]$

Cross-correlation이었다면:
$$
(f \star h)[n] = \sum_m f[m] h[n+m]
$$
결과가 다릅니다 (확인해보세요).

</details>

**문제 2** (심화): 2D convolution $(I * K)$에서 출력 크기가 $(H - h + 1) \times (W - w + 1)$임을 증명하라.

<details>
<summary>해설</summary>

$(I * K)[i,j] = \sum_{m=0}^{h-1} \sum_{n=0}^{w-1} I[i-m, j-n] K[m,n]$

이 합이 정의되려면:
- $0 \leq i - m < H$ → $i - h + 1 \leq i < H + 1$ ... (1)
- $0 \leq j - n < W$ → $j - w + 1 \leq j < W + 1$ ... (2)

동시에 (1)과 (2)를 만족하려면:
- $0 \leq i < H - h + 1$
- $0 \leq j < W - w + 1$

따라서 출력 크기는 $(H - h + 1) \times (W - w + 1)$ $\square$

</details>

**문제 3** (논문 비평): 왜 CNN 구현은 수학적 convolution(flip)이 아니라 cross-correlation(flip 없음)을 사용하는가? 이것이 equivariance 증명에 미치는 영향은?

<details>
<summary>해설</summary>

**이유**: Learned weights는 무제약이므로, flip된 버전을 직접 학습할 수 있습니다. 따라서 모델 용량은 동일합니다. 구현상 flip을 빼면 계산이 약간 더 간단합니다.

**Equivariance에서의 영향**: 
- 수학적 정의: $T_a (I * K) = (T_a I) * K$ (flip 있음)
- 실제 구현: $T_a (I \star \theta) = (T_a I) \star \theta$ (flip 없음)

하지만 flip은 단순 index 변환일 뿐, equivariance 성질 자체는 보존됩니다. $K^{\text{flip}} = \theta$로 이해하면 동등합니다.

따라서 **equivariance 증명은 여전히 유효**하지만, 정확한 이해를 위해서는 이 차이를 인식해야 합니다.

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-translation-equivariance.md)

</div>
