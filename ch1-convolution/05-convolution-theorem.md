# 05. Convolution Theorem과 Frequency Domain 해석

## 🎯 핵심 질문

- Convolution Theorem이 정확히 무엇이고, 왜 중요한가?
- Time/space domain의 convolution이 frequency domain의 곱셈이 되는 이유는?
- CNN의 각 layer를 frequency response로 어떻게 해석할 수 있는가?
- Low-pass, high-pass, edge detector 같은 필터를 frequency domain에서 이해할 수 있는가?
- 신경망의 "spectral bias" (저주파 선호)는 무엇이고, CNN에서 어떻게 나타나는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

CNN이 무엇을 학습하는지 이해하려면 **frequency domain 관점**이 필수입니다:

1. **필터의 의미** — Sobel edge detector는 high-pass filter. 학습된 필터는 어떤 주파수대를 감지하는가?
2. **Spectral Bias** — 신경망은 자연스럽게 저주파를 먼저 학습합니다 (Rahaman et al. 2019). CNN도 동일한가?
3. **정규화 이해** — Batch norm, layer norm 등이 frequency spectrum에 미치는 영향
4. **데이터-모델 매칭** — 자연 이미지는 저주파 성분이 많습니다. CNN이 이에 적응하는가?
5. **고급 주제** — Neural ODE, diffusion models 등 최신 방법들이 frequency 관점을 중심으로 합니다.

이 문서에서는 **Convolution Theorem을 유도하고, CNN의 frequency 해석을 수행**합니다.

---

## 📐 수학적 선행 조건

- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Fourier transform, DFT, inverse transform
- [Signal Processing Deep Dive](https://github.com/iq-ai-lab/signal-processing-deep-dive): Frequency response, filters
- 이전 문서: 04-toeplitz-matrix.md (Circulant matrix, FFT)

---

## 📖 직관적 이해

### Fourier Transform: Time → Frequency

**직관**: 복잡한 신호를 "사인파들의 합"으로 분해합니다.

$$f(t) = \sum_k \hat{f}(k) e^{2\pi i k t}$$

여기서 $\hat{f}(k)$는 주파수 $k$의 "강도"입니다.

### Convolution Theorem (직관)

시간 영역에서:
$$y(t) = (f * g)(t) = \int f(\tau) g(t - \tau) d\tau$$

이는 "모든 시점에서 필터를 곱하고 이동시켜 더하기"입니다 (무거운 계산).

주파수 영역에서:
$$\hat{y}(k) = \hat{f}(k) \cdot \hat{g}(k)$$

단순 곱셈! (매우 빠름)

**물리적 의미**:
- $f$: 입력 신호
- $g$: 필터
- $\hat{f}(k)$: 입력의 주파수 $k$ 성분
- $\hat{g}(k)$: 필터의 주파수 응답 (frequency response)
- 출력: 각 주파수 성분이 $\hat{g}(k)$배 증폭/감소

### CNN 필터의 Frequency Response

Sobel x-direction 필터:
$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

이것의 frequency response는?
- 수직 엣지 (낮은 수평 주파수): 강하게 반응
- 매끄러운 영역 (모든 주파수 약함): 약하게 반응
- → **High-pass filter in 수평 방향**

### Spectral Bias: 신경망의 저주파 선호

**관찰** (Rahaman et al. 2019): 신경망은 학습 초기에 저주파 함수를 배웁니다. 고주파는 나중에 배웁니다.

**이유**:
1. ReLU, sigmoid 같은 activation은 smooth → 저주파 선호
2. 가중치 초기화가 small → 고주파 신호는 약함
3. SGD가 "쉬운" (loss landscape smooth) 저주파부터 학습

**CNN에서**: Early layers는 low-pass (blur-like features), deep layers는 high-pass (detailed features)

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Fourier Transform (Continuous)

함수 $f: \mathbb{R}^d \to \mathbb{C}$에 대해:

$$\hat{f}(\omega) = \int_{\mathbb{R}^d} f(x) e^{-2\pi i \omega \cdot x} dx$$

**역 변환**:
$$f(x) = \int_{\mathbb{R}^d} \hat{f}(\omega) e^{2\pi i \omega \cdot x} d\omega$$

### 정의 5.2 — Discrete Fourier Transform (DFT)

길이 $N$ 신호 $x[n]$에 대해:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-2\pi i k n / N}$$

**역 변환**:
$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{2\pi i k n / N}$$

### 정의 5.3 — Frequency Response

필터 $h[n]$의 **frequency response**는 그 DFT:

$$H[k] = \sum_{n=0}^{N-1} h[n] e^{-2\pi i k n / N}$$

**해석**: 주파수 $k$의 신호에 대한 "이득(gain)"과 "위상 편이(phase shift)"

### 정리 5.4 — Convolution Theorem (이산)

신호 $x, h: [0, N) \to \mathbb{C}$의 circular convolution:

$$(x *_c h)[n] = \sum_{m=0}^{N-1} x[m] h[(n-m) \mod N]$$

의 DFT는:

$$\mathcal{F}(x *_c h)[k] = \mathcal{F}(x)[k] \cdot \mathcal{F}(h)[k] = X[k] \cdot H[k]$$

**증명**:

$$\mathcal{F}(x *_c h)[k] = \sum_{n=0}^{N-1} \left(\sum_{m=0}^{N-1} x[m] h[(n-m) \mod N]\right) e^{-2\pi i k n / N}$$

변수 치환 $\ell = (n-m) \mod N$:

$$= \sum_{m=0}^{N-1} x[m] \sum_{\ell=0}^{N-1} h[\ell] e^{-2\pi i k (\ell+m) / N}$$

$$= \sum_{m=0}^{N-1} x[m] e^{-2\pi i k m / N} \sum_{\ell=0}^{N-1} h[\ell] e^{-2\pi i k \ell / N}$$

$$= X[k] \cdot H[k] \quad \square$$

### 정리 5.5 — Parseval의 정리

시간과 주파수 영역에서의 에너지:

$$\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2$$

**의미**: 신호의 에너지는 주파수 영역에서도 보존됩니다.

### 정리 5.6 — 2D Separable Convolution과 Frequency Domain

2D convolution이 separable하면 (가로×세로로 분해):

$$K(x,y) = K_x(x) K_y(y)$$

frequency domain에서도:

$$\hat{K}(u,v) = \hat{K}_x(u) \hat{K}_y(v)$$

**효율성**: 2D FFT $O(N^2 \log N)$ 대신 1D FFT 2번 $2 O(N \log N)$

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Non-circular Convolution과 Zero Padding

선형 convolution (non-circular)을 DFT로 계산하려면, zero padding이 필요합니다.

두 신호 $x$ (길이 $N$), $h$ (길이 $K$)의 선형 convolution 결과는 길이 $N+K-1$입니다.

**트릭**: 둘 다 길이 $M \geq N+K-1$로 zero-pad한 후 circular convolution:

$$\text{LinearConv}(x,h) = \text{CircularConv}_M(x_{\text{pad}}, h_{\text{pad}})$$

DFT를 길이 $M$으로 수행하면 convolution theorem 적용 가능합니다.

### 유도 2 — Filter의 Magnitude Response와 Phase Response

Frequency response $H[k]$를 극형식으로:

$$H[k] = |H[k]| e^{i \phi[k]}$$

- **Magnitude response** $|H[k]|$: 주파수 $k$의 신호가 얼마나 증폭되는가?
- **Phase response** $\phi[k]$: 주파수 $k$의 신호가 얼마나 위상 이동되는가?

CNN에서는 보통 magnitude response만 관심 있습니다 (non-linear activation이 위상을 보존 안 함).

### 유도 3 — Natural Images의 저주파 성분

실제 자연 이미지는 frequency spectrum이:

$$S(f) \propto 1/f^{\alpha}$$

여기서 $\alpha \approx 2$ (평균적으로).

즉, **저주파 성분이 고주파보다 훨씬 강합니다**.

따라서 CNN이 저주파를 먼저 학습하는 것은 이미지 통계와 일치합니다 (자연 선택).

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Convolution Theorem 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 신호와 필터
x = np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.complex128)
h = np.array([1, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.complex128)

# 방법 1: 직접 circular convolution
def circular_convolve(x, h):
    n = len(x)
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            y[i] += x[j] * h[(i - j) % n]
    return y

y_direct = circular_convolve(x, h)

# 방법 2: DFT로 계산
X = np.fft.fft(x)
H = np.fft.fft(h)
Y = X * H  # Element-wise 곱
y_fft = np.fft.ifft(Y).real

print("Direct convolution:", y_direct.real[:4])
print("FFT convolution:", y_fft[:4])
print("Difference:", np.abs(y_direct - y_fft).max())
```

### 실험 2 — 필터의 Frequency Response

```python
import torch
import torch.nn.functional as F

# 여러 필터들
filters = {
    'Identity': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'Blur': np.ones((3, 3)) / 9,
    'Sobel-X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel-Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
}

fig, axes = plt.subplots(len(filters), 3, figsize=(12, 10))

for idx, (name, kernel) in enumerate(filters.items()):
    # Zero-pad to larger size for better frequency resolution
    kernel_padded = np.zeros((32, 32))
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    
    # 2D FFT
    fft_result = np.fft.fft2(kernel_padded)
    magnitude = np.abs(fft_result)
    magnitude_log = np.log1p(magnitude)  # Log scale
    
    # Shift zero frequency to center
    magnitude_shifted = np.fft.fftshift(magnitude_log)
    
    # 시각화
    axes[idx, 0].imshow(kernel, cmap='gray')
    axes[idx, 0].set_title(f'{name} Filter')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(magnitude_shifted, cmap='viridis')
    axes[idx, 1].set_title(f'{name} - Magnitude Response')
    axes[idx, 1].axis('off')
    
    # Radial frequency profile
    center = 16
    radial = []
    for r in range(1, 16):
        circle = magnitude_shifted[center-r:center+r, center-r:center+r]
        radial.append(np.mean(circle))
    axes[idx, 2].plot(radial)
    axes[idx, 2].set_title(f'{name} - Radial Profile')
    axes[idx, 2].set_xlabel('Frequency (radius)')

plt.tight_layout()
plt.show()
```

### 실험 3 — CNN의 Spectral Bias

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# 학습 과정에서 frequency response 모니터링
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 합성 데이터: 특정 주파수 신호
H, W = 32, 32
target_freq = 2  # 주파수 2만 활성화

t_x = np.linspace(0, 1, W)
t_y = np.linspace(0, 1, H)
X_mesh, Y_mesh = np.meshgrid(t_x, t_y)

# 순수 고주파 신호
low_freq_signal = np.sin(2 * np.pi * 1 * X_mesh) * np.sin(2 * np.pi * 1 * Y_mesh)
high_freq_signal = np.sin(2 * np.pi * 4 * X_mesh) * np.sin(2 * np.pi * 4 * Y_mesh)

x_low = torch.tensor(low_freq_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
x_high = torch.tensor(high_freq_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 여러 epoch 에서 출력의 spectral profile 측정
epochs = 50
spectral_profiles = {'low_freq': [], 'high_freq': []}

for epoch in range(epochs):
    # Dummy training (실제로는 특정 target에 맞게)
    out_low = model(x_low)
    out_high = model(x_high)
    
    # Spectral analysis
    fft_low = np.abs(np.fft.fft2(out_low.detach().numpy()[0, 0]))
    fft_high = np.abs(np.fft.fft2(out_high.detach().numpy()[0, 0]))
    
    spectral_profiles['low_freq'].append(np.mean(fft_low))
    spectral_profiles['high_freq'].append(np.mean(fft_high))
    
    # Backward pass (간단한 loss)
    loss = out_low.mean() - out_high.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(spectral_profiles['low_freq'], label='Low Freq (1 Hz)', marker='o')
plt.semilogy(spectral_profiles['high_freq'], label='High Freq (4 Hz)', marker='s')
plt.xlabel('Training Epoch')
plt.ylabel('Mean Spectral Magnitude (log scale)')
plt.title('CNN Spectral Bias: Low-Freq First')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("관찰: 초기에 low-freq가 더 빠르게 증가 → spectral bias 확인")
```

### 실험 4 — Natural Image의 Spectrum

```python
from PIL import Image
from torchvision import datasets, transforms

# MNIST 또는 다른 데이터셋 로드
mnist = datasets.MNIST(root='./data', train=True, download=True)
img, _ = mnist[0]
img_array = np.array(img)

# 2D FFT
fft_img = np.fft.fft2(img_array)
magnitude = np.abs(fft_img)
magnitude_log = np.log1p(magnitude)
magnitude_shifted = np.fft.fftshift(magnitude_log)

# 방사형 주파수 프로필 (radial frequency power)
center_y, center_x = magnitude_shifted.shape[0]//2, magnitude_shifted.shape[1]//2
radii = []
powers = []

for r in range(1, min(center_y, center_x)):
    # Annular region: r-1 to r
    y, x = np.ogrid[:magnitude_shifted.shape[0], :magnitude_shifted.shape[1]]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist >= r-1) & (dist < r)
    power = magnitude_shifted[mask].mean()
    radii.append(r)
    powers.append(power)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(magnitude_shifted, cmap='viridis')
axes[1].set_title('Magnitude Spectrum (log scale)')
axes[1].axis('off')

axes[2].loglog(radii, powers, 'b-', label='Natural Image')
# Theoretical: 1/f^2
axes[2].loglog(radii, 1000/np.array(radii)**2, 'r--', label='1/f² Reference')
axes[2].set_xlabel('Frequency (radius)')
axes[2].set_ylabel('Power (log scale)')
axes[2].set_title('Radial Power Spectrum')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Natural images typically follow 1/f^α power law (α ≈ 2)")
```

---

## 🔗 이론과 실전의 간극

### 1. Circular vs Linear Convolution

**이론 (Convolution Theorem)**: Circular convolution은 DFT에서 정확히 곱셈

**실전**:
- Zero padding으로 linear convolution을 circular로 변환
- 대부분의 CNN 구현은 spatial domain에서 직접 (FFT 사용 안 함)

### 2. Phase Information의 무시

Frequency domain: $H[k] = |H[k]| e^{i\phi[k]}$

하지만 CNN 학습 후 activation (ReLU 등)이 **위상 정보를 부분적으로 파괴**합니다.

따라서 단순 magnitude analysis는 근사일 수 있습니다.

### 3. Non-linear의 Frequency Domain 표현

Activation function이 있으면 정확한 frequency domain 분석이 어렵습니다.

**대안**:
- Harmonic analysis (고급)
- Empirical frequency analysis (실험적)
- Neural ODE의 frequency interpretation

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear filtering | CNN은 nonlinear (activation 때문) → frequency 해석이 정확하지 않음 |
| Stationary signals | 이미지의 주파수 특성이 위치마다 다름 → 시간-주파수 분석 필요 |
| Circular/periodic | Actual images: rectangular, zero-padded → boundary artifacts |
| White noise 가정 | Natural images: colored noise (1/f power law) → signal-dependent |
| Continuous signals | Discrete → aliasing 가능 (Nyquist 이론) |

---

## 📌 핵심 정리

$$\boxed{\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g) \text{ — Convolution Theorem}}$$

$$\boxed{y[n] = \text{IFFT}(\text{FFT}(x) \odot \text{FFT}(h)) \text{ — Frequency domain 계산}}$$

$$\boxed{\text{CNN은 low-frequency를 먼저 학습 — Spectral Bias}}$$

| 개념 | 정의 |
|------|------|
| **Fourier Transform** | $\hat{f}(k) = \int f(x) e^{-2\pi i kx} dx$ |
| **Convolution Theorem** | $\mathcal{F}(f*g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$ |
| **Frequency response** | 필터 $h$의 DFT: $H[k] = \sum_n h[n] e^{-2\pi i kn/N}$ |
| **Magnitude response** | $\|H[k]\|$ — 각 주파수의 증폭 정도 |
| **Phase response** | $\angle H[k]$ — 위상 이동 |
| **Spectral bias** | 신경망이 저주파 함수를 먼저 학습 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 길이 4인 신호 $x = [1, 0, 0, 0]$과 필터 $h = [1, 1, 0, 0]$의 circular convolution을 DFT를 사용해 계산하라.

<details>
<summary>해설</summary>

$$X[k] = \text{DFT}([1, 0, 0, 0]) = [1, 1, 1, 1]$$ (모든 k에 대해)

$$H[k] = \text{DFT}([1, 1, 0, 0]) = [2, 1-i, 0, 1+i]$$

$$Y[k] = X[k] \cdot H[k] = [2, 1-i, 0, 1+i]$$

$$y[n] = \text{IDFT}(Y) = [1, 1, 0, 0]$$

확인: 직접 계산
$(x *_c h)[0] = 1 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 = 1$
$(x *_c h)[1] = 1 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 0 \cdot 1 = 1$
$(x *_c h)[2] = 0$
$(x *_c h)[3] = 0$

따라서 $y = [1, 1, 0, 0]$ ✓

</details>

**문제 2** (심화): Parseval의 정리를 증명하라: $\sum_n |x[n]|^2 = \frac{1}{N} \sum_k |X[k]|^2$

<details>
<summary>해설</summary>

$$\sum_n |x[n]|^2 = \sum_n x[n] \overline{x[n]} = \sum_n x[n] \sum_k \overline{X[k]} e^{-2\pi i kn/N}$$

$$= \sum_n \sum_k x[n] \overline{X[k]} e^{-2\pi i kn/N}$$

Swap order:
$$= \sum_k \overline{X[k]} \sum_n x[n] e^{-2\pi i kn/N}$$

하지만 $X[k] = \sum_n x[n] e^{-2\pi i kn/N}$이므로:

$$= \sum_k \overline{X[k]} X[k] = \sum_k |X[k]|^2$$

Inverse DFT normalization을 고려하면 $\frac{1}{N}$ 인수가 나타납니다. $\square$

</details>

**문제 3** (논문 비평): "CNN은 frequency domain에서 low-pass filter처럼 동작한다"는 주장이 얼마나 정확한가? 반례를 들어보라.

<details>
<summary>해설</summary>

**주장의 정확성**: 부분적으로 참

**Early layers (낮은 깊이)**:
- 대부분의 가중치가 작음 → 약한 필터 → low-pass처럼 동작

**Deep layers (깊은 깊이)**:
- High-frequency도 학습 가능 (edge, texture 감지)
- Activation functions (ReLU)가 고주파를 생성 가능

**반례**:
- Edge detection layers → high-pass
- Texture discrimination → mid-to-high frequency

**더 정확한 설명**:
- Early: 저주파 (물체 형태, 색상)
- Middle: 중간 주파수 (object parts)
- Deep: 고주파 (fine details, textures)

즉, **계층별로 다르며**, 단순 "low-pass"라고 말하기는 부정확합니다.

</details>

---

<div align="center">

[◀ 이전](./04-toeplitz-matrix.md) | [📚 README](../README.md) | [다음 ▶](../ch2-cnn-ops/01-conv-forward-backward.md)

</div>
