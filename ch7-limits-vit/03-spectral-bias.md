# 03. Spectral Bias와 신경망의 학습 편향

## 🎯 핵심 질문

- Rahaman et al. (2019)의 "spectral bias"는 무엇인가? 왜 신경망은 저주파를 먼저 학습하는가?
- Target function의 Fourier 분해를 통해 학습 수렴 속도를 정량화할 수 있는가?
- CNN과 완전 연결층(fully connected network)이 spectral bias를 동등하게 보이는가, 아니면 다른가?
- Fourier feature와 positional encoding (NeRF, Vision Transformer)은 spectral bias를 극복하는가?

---

## 🔍 왜 이 개념이 CNN 이해에 중요한가

**Spectral bias**는 신경망이 함수 공간을 "균등하게" 학습하지 않는다는 발견입니다. CNN이나 MLP 모두 저주파(low-frequency) 성분을 먼저 학습하고, 고주파(high-frequency) 성분은 나중에 학습합니다. 이는 adversarial examples의 원인 중 하나이며, 또한 CNN이 특정 작업 (예: 고주파 텍스처)에 약한 이유를 설명합니다. NeRF, Vision Transformer와 같은 최신 아키텍처들도 이 bias를 극복하려고 노력하고 있습니다.

---

## 📐 수학적 선행 조건

- [Gradient Descent와 최적화](../ch1-gd-geometry/): Convergence analysis, step size
- [CNN 기본 연산](../ch2-cnn-ops/): Convolution, kernel의 주파수 특성
- Fourier 분석: DFT, FFT, frequency decomposition
- 미분방정식: ODE 해석, stability analysis
- 선형대수: Eigenvalue, spectral decomposition

---

## 📖 직관적 이해

### Spectral Bias의 직관

신경망이 함수 $f: \mathbb{R}^d \to \mathbb{R}$를 학습할 때, target을 Fourier 급수로 분해하면:

$$f_\text{target}(x) = \sum_k a_k e^{i \omega_k \cdot x}$$

신경망은 이 **고주파 성분들을 다른 속도로** 학습합니다:

- **저주파 ($|\omega_k|$ 작음)**: 빠르게 수렴 (initial loss 빠르게 감소)
- **고주파 ($|\omega_k|$ 큼)**: 느리게 수렴 (plateau 단계가 오래 지속)

### 왜 이런 현상이 일어나는가?

**1. 신경망의 implicit regularization**

신경망은 training loss를 최소화할 때, **가장 간단한 (smooth) 함수부터 학습**합니다. 이는 마치 early stopping과 유사한 효과입니다.

**2. Gradient flow의 속도**

저주파 성분은 gradient가 크고 (큰 activation), 고주파 성분은 gradient가 작습니다 (세밀한 진동). 따라서 gradient descent 관점에서, 저주파 성분이 더 빠르게 업데이트됩니다.

**3. Receptive field와의 관계 (CNN 특화)**

CNN의 kernel size가 작으면 (예: 3×3), 고주파 특징을 직접 학습하기 어렵습니다. 초기 층은 저주파 (edge 같은 큰 구조)를 학습하고, 깊은 층이 고주파 (texture 같은 세부)를 학습합니다.

### Fourier Feature를 통한 해결책

NeRF (Mildenhall et al., 2021)는 다음과 같은 **Fourier positional encoding**을 도입:

$$\gamma(x) = [\sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)]$$

이렇게 **사전에 고주파를 도입**하면, 신경망이 고주파 성분도 빨리 학습할 수 있습니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Spectral Bias

신경망 $f_\theta(x)$가 target 함수 $f_\text{target}(x)$를 학습할 때, 주파수 $\omega$에 따른 loss 수렴 속도를 $\tau(\omega)$라 하면:

$$\tau(\omega) = \text{convergence time for frequency } \omega$$

**Spectral bias**는 $\tau(\omega)$가 $\omega$에 대해 단조증가한다는 성질:

$$|\omega_1| < |\omega_2| \Rightarrow \tau(\omega_1) < \tau(\omega_2)$$

### 정의 3.2 — Fourier Decomposition

Target 함수를 Fourier 급수로 분해:

$$f_\text{target}(x) = \sum_k a_k \phi_k(x)$$

여기서 $\phi_k$는 Fourier basis (사인 또는 코사인), $a_k$는 계수.

**Error at frequency** $k$:

$$e_k(t) = |a_k - \hat{a}_k(t)|$$

여기서 $\hat{a}_k(t)$는 시간 $t$에 학습된 계수.

### 정의 3.3 — Implicit Regularization

신경망의 early stopping 효과:

$$\min_\theta L(\theta) = \min_\theta \|f_\theta - f_\text{target}\|_2^2$$

에서, gradient descent는 다음과 같이 implicit하게 정규화:

$$\theta(t) = \arg\min_{\|\theta\|_\text{eff} \leq R(t)} L(\theta)$$

여기서 $R(t)$는 시간에 따라 증가하는 implicit regularization budget.

### 정리 3.4 — Spectral Convergence Rate (Rahaman 2019)

ReLU MLP에서 1차원 데이터 $x \in [0, 1]$, target $f_\text{target}(x) = \sum_k a_k e^{2\pi i k x}$에 대해:

**수렴 시간**:

$$\tau(k) \approx C \cdot \frac{1}{\lambda_\min(\text{NTK}(\omega_k))}$$

여기서 NTK는 Neural Tangent Kernel, $\lambda_\min$은 최소 고유값.

**실제로**, $k$가 크면 (고주파) $\tau(k) \propto k^2$ (또는 그 이상).

### 정리 3.5 — Positional Encoding의 효과

Fourier positional encoding을 사용하면:

$$\tau(k) \propto 1$$

즉, **모든 주파수가 거의 동등하게 빠르게 수렴**.

---

## 🔬 증명 및 수학적 유도

### Spectral Bias의 Gradient Flow 분석

1차원 least-square loss:

$$L(t) = \frac{1}{2} \int_0^1 (f_\theta(x; t) - f_\text{target}(x))^2 dx$$

Gradient flow:

$$\frac{\partial \theta}{\partial t} = -\nabla_\theta L$$

Neural Tangent Kernel (NTK) 근사 하에서:

$$\frac{\partial f_\theta(x; t)}{\partial t} = -\int_0^1 K(x, x') (f_\theta(x'; t) - f_\text{target}(x')) dx'$$

여기서 $K(x, x') = \nabla_\theta f_\theta(x) \cdot \nabla_\theta f_\theta(x')$는 kernel.

Fourier 기저로 전개하면:

$$\hat{a}_k(t) = a_k (1 - e^{-\lambda_k t})$$

여기서 $\lambda_k = K(\omega_k, \omega_k)$는 주파수 $k$의 kernel eigenvalue.

**핵심**: $\lambda_k$가 $k$에 대해 감소하면, 고주파가 느리게 수렴합니다.

실제로 ReLU MLP에서:

$$\lambda_k \propto 1/k^2$$

따라서:

$$\tau(k) = 1/\lambda_k \propto k^2$$

### CNN과 Spectral Bias

CNN의 convolution:

$$f(x) = \sum_{\omega} \hat{K}(\omega) \hat{x}(\omega) e^{i\omega \cdot x}$$

여기서 $\hat{K}(\omega)$는 kernel의 Fourier 변환.

작은 kernel (예: 3×3)은 저주파에 대해서는 큰 응답이지만, 고주파에 대해서는 작은 응답을 가집니다:

$$|\hat{K}(\omega)| \propto \frac{1}{1 + \alpha |\omega|^2}$$

이것이 **CNN의 spectral bias**입니다.

### Fourier Positional Encoding의 원리

입력을 다음과 같이 encoding:

$$\phi_L(x) = [\sin(2^0 \pi x), \cos(2^0 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)]^T$$

이렇게 사전에 $2^L$까지의 모든 주파수를 도입하면, 신경망이 각 주파수를 **독립적으로 학습**할 수 있습니다.

결과적으로, NTK가 더 균등해지고:

$$\lambda_k \approx \text{const}$$

따라서 모든 주파수가 거의 동등한 속도로 수렴합니다.

---

## 💻 실험 재현

### 실험 1 — 1D 함수에서의 Spectral Bias 시각화

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target function: 다양한 주파수의 합
def target_function(x):
    """
    f(x) = sin(2π·1·x) + sin(2π·5·x) + sin(2π·10·x)
    저주파, 중주파, 고주파 성분 혼합
    """
    return (torch.sin(2*np.pi*1*x) + 
            torch.sin(2*np.pi*5*x) + 
            torch.sin(2*np.pi*10*x)) / 3

# 신경망 정의 (간단한 MLP)
class SimpleNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 데이터 생성
x_train = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
y_train = target_function(x_train)

# 모델 훈련
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 주파수별 error 추적
frequencies = [1, 5, 10]
error_history = {f: [] for f in frequencies}

num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    # 주파수별 error 계산 (FFT 사용)
    if epoch % 50 == 0:
        with torch.no_grad():
            y_pred_np = y_pred.squeeze().cpu().numpy()
            y_target_np = y_train.squeeze().cpu().numpy()
            
            # FFT 계산
            fft_pred = np.fft.fft(y_pred_np)
            fft_target = np.fft.fft(y_target_np)
            
            for freq in frequencies:
                # 해당 주파수 component의 error
                error = np.abs(fft_pred[freq] - fft_target[freq])
                error_history[freq].append(error)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 좌측: 주파수별 error 수렴 곡선
ax1.semilogy(error_history[1], label='f=1 (저주파)', linewidth=2)
ax1.semilogy(error_history[5], label='f=5 (중주파)', linewidth=2)
ax1.semilogy(error_history[10], label='f=10 (고주파)', linewidth=2)
ax1.set_xlabel('Epoch (×50)')
ax1.set_ylabel('Frequency Component Error')
ax1.set_title('Spectral Bias: Low-Frequency First Learning')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 우측: 최종 prediction vs target
with torch.no_grad():
    y_final = model(x_train).squeeze().cpu().numpy()

ax2.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 
         'b-', linewidth=2, label='Target', alpha=0.7)
ax2.plot(x_train.cpu().numpy(), y_final, 
         'r--', linewidth=2, label='Predicted', alpha=0.7)
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('Function Learning Result')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("관찰:")
print("- 저주파 (f=1): 초반에 빠르게 수렴")
print("- 고주파 (f=10): 훨씬 천천히 수렴")
print("- 이것이 spectral bias 현상")
```

**관찰**: 저주파는 빠르게 정확해지지만, 고주파는 훨씬 오래 걸립니다.

### 실험 2 — Fourier Positional Encoding의 효과

```python
# Fourier positional encoding을 사용하는 네트워크
class FourierNet(nn.Module):
    def __init__(self, hidden_dim=128, L=4):
        """
        L: Fourier 레이어 수 (2^L까지의 주파수)
        """
        super().__init__()
        self.L = L
        # Positional encoding 차원: 2L (sin과 cos 쌍)
        encoding_dim = 2 * L
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Fourier positional encoding
        x_encoded = []
        for i in range(self.L):
            x_encoded.append(torch.sin(2**i * np.pi * x))
            x_encoded.append(torch.cos(2**i * np.pi * x))
        x_encoded = torch.cat(x_encoded, dim=-1)
        return self.net(x_encoded)

# 동일한 데이터로 훈련
model_fourier = FourierNet(hidden_dim=128, L=4).to(device)
optimizer_fourier = optim.Adam(model_fourier.parameters(), lr=0.01)

error_history_fourier = {f: [] for f in frequencies}

for epoch in range(num_epochs):
    optimizer_fourier.zero_grad()
    y_pred = model_fourier(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer_fourier.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            y_pred_np = y_pred.squeeze().cpu().numpy()
            y_target_np = y_train.squeeze().cpu().numpy()
            
            fft_pred = np.fft.fft(y_pred_np)
            fft_target = np.fft.fft(y_target_np)
            
            for freq in frequencies:
                error = np.abs(fft_pred[freq] - fft_target[freq])
                error_history_fourier[freq].append(error)

# 비교 시각화
fig, ax = plt.subplots(figsize=(12, 6))

ax.semilogy(error_history[10], 'r-', linewidth=2, label='Standard MLP (f=10)', marker='o', markersize=4)
ax.semilogy(error_history_fourier[10], 'b-', linewidth=2, label='Fourier Net (f=10)', marker='s', markersize=4)

ax.set_xlabel('Epoch (×50)')
ax.set_ylabel('High-Frequency Component Error')
ax.set_title('Fourier Positional Encoding Overcomes Spectral Bias')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("관찰:")
print("- Standard MLP: 고주파 수렴 매우 느림")
print("- Fourier Net: 고주파도 빠르게 수렴")
print("- Fourier encoding이 spectral bias를 크게 완화")
```

**관찰**: Fourier positional encoding을 사용하면 고주파도 빠르게 학습됩니다.

### 실험 3 — CNN의 Spectral Bias

```python
# 2D 이미지에서의 CNN spectral bias
class SimpleCNN2D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# 2D Target: low-freq와 high-freq 혼합
def create_2d_target(size=64):
    """
    저주파: 큰 sinusoid
    고주파: 세밀한 checkerboard
    """
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Low-freq component
    low_freq = torch.sin(2*np.pi*2*X) * torch.cos(2*np.pi*2*Y)
    
    # High-freq component
    high_freq = torch.sin(2*np.pi*8*X) * torch.cos(2*np.pi*8*Y)
    
    return (low_freq + 0.3*high_freq).unsqueeze(0).unsqueeze(0)

# 데이터 생성
target_2d = create_2d_target(size=64).to(device)

# CNN 훈련
model_cnn = SimpleCNN2D(kernel_size=3).to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

loss_history = []
for epoch in range(1000):
    optimizer_cnn.zero_grad()
    pred = model_cnn(target_2d)
    loss = criterion(pred, target_2d)
    loss.backward()
    optimizer_cnn.step()
    
    if epoch % 100 == 0:
        loss_history.append(loss.item())
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 결과 시각화
with torch.no_grad():
    pred_2d = model_cnn(target_2d)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(target_2d.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Target (Low + High Freq)')
axes[0].axis('off')

axes[1].imshow(pred_2d.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title('CNN Prediction')
axes[1].axis('off')

axes[2].imshow((target_2d - pred_2d).squeeze().cpu().numpy(), cmap='RdBu')
axes[2].set_title('Error (CNN못 배운 고주파)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("관찰:")
print("- CNN은 저주파(큰 구조) 잘 학습")
print("- 고주파(세부) 부분에서 높은 오차")
print("- Small kernel이 고주파 학습 어렵게 함")
```

**관찰**: CNN도 spectral bias를 보이며, 고주파 세부는 못 배웁니다.

### 실험 4 — Kernel Size와 Spectral Bias의 관계

```python
# 다양한 kernel size에서의 spectral response
kernel_sizes = [3, 5, 7, 11]
fig, axes = plt.subplots(1, len(kernel_sizes), figsize=(15, 4))

for idx, k in enumerate(kernel_sizes):
    # kernel 생성 (Gaussian)
    x = torch.linspace(-1, 1, k)
    kernel = torch.exp(-x**2 * 2)
    kernel = kernel / kernel.sum()
    
    # Frequency response 계산 (FFT)
    freq_response = np.abs(np.fft.fft(kernel.numpy(), n=256))[:128]
    
    axes[idx].semilogy(freq_response)
    axes[idx].set_title(f'Kernel Size = {k}')
    axes[idx].set_xlabel('Frequency')
    axes[idx].set_ylabel('Magnitude')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("관찰:")
print("- Kernel size 작으면: 고주파 응답 급격히 감소")
print("- Kernel size 크면: 고주파도 어느 정도 응답")
print("- 따라서 큰 kernel이 spectral bias 약함")
print("  (하지만 계산량과 parameter 증가)")
```

**관찰**: Kernel size가 크면 고주파를 더 잘 전달합니다.

---

## 🔗 이론과 실전의 간극

### Vision Transformer에서의 활용

ViT는 patch embedding 후에 **절대 positional encoding**을 사용합니다:

```python
# ViT-style positional encoding
pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
```

이는 Fourier와 다르지만, 비슷하게 **모든 patch를 명시적으로 구분**하므로 spectral bias를 완화합니다.

### NeRF의 성공

NeRF (Neural Radiance Fields)는 Fourier positional encoding을 통해:

- 매우 고주파의 세부 표면 구조도 학습 가능
- 이전 기법들 (volumetric rendering)보다 훨씬 우수한 품질

이는 spectral bias 극복의 직접적인 성과입니다.

### CNN의 한계

작은 kernel (3×3)의 CNN은 본질적으로 spectral bias를 가집니다:

- 고주파 texture 학습에 약함
- 따라서 adversarial perturbations에 취약 (고주파 노이즈)
- Object shape (저주파)는 잘 배우지만, fine details (고주파)는 못 배움

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Target이 smooth (low frequency 지배) | 매우 jagged한 함수는 느린 수렴 |
| Fourier 분해가 적절함 | 데이터가 frequency domain에서 구조 없으면 부정확 |
| NTK 근사 유효 | 큰 learning rate에서는 NTK 근사 깨짐 |
| Positional encoding이 도움 | 학습 가능한 encoding이 잘못되면 오히려 해로움 |
| 모든 신경망에 동등하게 적용 | Attention mechanism은 spectral bias 다를 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\tau(f) \text{ increases with frequency } f \quad \text{— Lower frequencies learned first}}$$

| 개념 | 정의 | 영향 |
|------|------|------|
| **Spectral Bias** | 신경망이 저주파를 빨리, 고주파를 느리게 학습 | Smooth function 학습 유리, 고주파 텍스처 약함 |
| **NTK Eigenvalue** | $\lambda_f \propto 1/f^2$ (ReLU MLP) | 고주파 수렴 시간 $\propto f^2$ |
| **Positional Encoding** | Fourier basis 사전 도입 | 모든 주파수 균등하게 빠르게 학습 |
| **CNN의 Spectral Bias** | 작은 kernel의 저주파 응답 크고 고주파 응답 작음 | 큰 구조는 잘 배우고 세부는 약함 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Target 함수 $f(x) = \sin(2\pi k x)$ (순수 $k$-주파수)를 MLP가 학습할 때, 수렴 시간이 $k^2$에 비례한다고 했다. 이는 무엇을 의미하는가? $k=1$과 $k=10$에서 수렴 시간의 비율은?

<details>
<summary>힌트 및 해설</summary>

수렴 시간이 $k^2$에 비례하면:

$$\tau(k) = C \cdot k^2$$

따라서:

$$\frac{\tau(10)}{\tau(1)} = \frac{C \cdot 100}{C \cdot 1} = 100$$

즉, **고주파는 저주파보다 100배 오래 걸림**.

만약 저주파는 100 epoch에서 수렴하면, 고주파는 10,000 epoch이 필요합니다.

</details>

**문제 2** (심화): CNN과 MLP는 모두 spectral bias를 보이지만, 다른 방식이다. CNN은 kernel의 Fourier 변환이 저주파에 큰 응답을 가지므로, 그리고 MLP는 신경망 구조 자체가 spectral bias를 가진다. 이 두 가지의 **본질적 차이**는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**CNN의 spectral bias**:
- Inductive bias (아키텍처의 구조)
- 모든 kernel이 동일하게 저주파 편향
- 큰 kernel을 쓰면 개선 가능

**MLP의 spectral bias**:
- Implicit bias (학습 과정, gradient descent의 성질)
- 신경망 깊이, 너비, 초기화에 따라 다름
- Kernel size 개념 없음

**중요한 차이**:
- CNN: 구조적 제약 (convolution이 필연적으로 저주파 편향 유발)
- MLP: 학습 방식의 편향 (gradient descent가 자연스럽게 smooth 함수 먼저 학습)

**따라서**:
- CNN은 큰 kernel로 "해결 가능"
- MLP는 positional encoding으로만 해결 가능

</details>

**문제 3** (이론-실전): Vision Transformer (ViT)의 positional embedding은 Fourier 기반이 아니다. 그럼에도 ViT가 텍스처 같은 고주파 세부를 CNN보다 잘 배운다고 알려져 있다. 왜일까?

<details>
<summary>힌트 및 해설</summary>

ViT의 positional embedding은 **학습 가능한 파라미터**입니다:

$$\text{pos}_\text{embedding} \in \mathbb{R}^{N_\text{patches} \times D}$$

이는 다음과 같이 작동합니다:

1. 각 patch는 **독립적인 embedding vector** 할당
2. 모든 patch를 explicitly 구분 가능
3. 따라서 **모든 주파수의 patch 조합을 직접 표현** 가능

**Fourier와의 차이**:
- Fourier: 명시적 사인/코사인 함수
- ViT: 암묵적 embedding (더 유연함)

**ViT가 고주파를 잘 배우는 이유**:
- Self-attention이 global receptive field (CNN과 달리 locality 제약 없음)
- Patch 기반 처리 (픽셀 레벨이 아닌 patch 레벨, 고주파 제약 덜함)
- 충분한 데이터와 파라미터로 모든 주파수 학습 가능

</details>

---

<div align="center">

[◀ 이전](./02-adversarial.md) | [📚 README](../README.md) | [다음 ▶](./04-vit.md)

</div>
