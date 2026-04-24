# 02. 유효 Receptive Field (Effective RF)

## 🎯 핵심 질문

- 이론적 RF와 실제 유효 RF(ERF)의 차이는 무엇인가?
- Luo et al. (2016)의 "Understanding the Effective Receptive Field in Deep CNNs"에서 주장하는 중심 집중 현상은 왜 발생하는가?
- Gradient propagation을 통해 어떤 픽셀이 최종 출력에 실질적으로 영향을 미치는지 측정할 수 있는가?
- 깊이 $L$에 따라 ERF가 $1/\sqrt{L}$로 감소한다는 이론적 예측은 맞는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

이전 문서의 이론적 RF 계산은 "최악의 경우(worst-case)" 가정입니다. 즉:
- 모든 경로(path)가 동등하게 기여
- 모든 가중치가 동일한 크기

현실에는:
- **Gradient flow**: 역전파 시 gradient가 경로마다 다르게 흐름
- **가중치 분포**: 중앙 근처 가중치 >> 주변부 가중치
- **Non-linearity**: ReLU, normalization이 gradient를 감소시킴

따라서 "실제로 영향 있는 영역"은 이론적 RF보다 훨씬 작습니다. 이를 정확히 이해하면, 네트워크 설계 및 성능 분석이 더 현실적입니다.

---

## 📐 수학적 선행 조건

- 편미분, 연쇄 법칙 (Chain rule)
- Central limit theorem (CLT)
- Gaussian 분포, standard deviation
- Gradient descent, backpropagation

---

## 📖 직관적 이해

### 중심 집중 현상 (Center Concentration)

다층 convolution을 생각해봅시다:

$$y = \text{Conv}_L(\cdots \text{Conv}_2(\text{Conv}_1(x)) \cdots)$$

각 convolution이 kernel을 중심 기준으로 적용합니다. 만약 입력 $x$의 경계 픽셀이 최종 출력 $y$에 영향을 주려면:

1. 경계에서 시작한 신호가 여러 층을 거쳐야 함
2. 각 층마다 중심으로부터의 거리가 "noise"처럼 누적
3. 결과적으로 경계 신호는 중앙 신호에 비해 크기가 지수적으로 감소

### Gradient 관점에서의 이해

출력 $y_c$ (중심 위치)에 대한 입력 $x_i$ (위치 $i$)의 gradient:

$$\frac{\partial y_c}{\partial x_i} = \text{Backprop gradient}$$

이 gradient는:
- $i = c$ (중심): 큼 (직접 경로 많음)
- $i \neq c$ (경계): 작음 (신호가 여러 비선형 변환 거침)

Central limit theorem 관점에서, 이 gradient의 분포는 Gaussian에 수렴합니다:
$$\frac{\partial y_c}{\partial x_i} \sim \mathcal{N}(0, \sigma^2(i))$$

여기서 $\sigma(i)$는 $i$와 중심 $c$ 사이의 거리에 지수적으로 감소합니다.

### 정량화: 1/sqrt(L) 감소

직관적으로, $L$개 층을 지나면서:
- 각 층에서 약 $1/\sqrt{L}$만큼의 "신호 감쇠" 발생
- 총 감쇠: $(1/\sqrt{L})^L \approx 0$에 가까워짐 (경계에서)
- 중앙에서만 신호 유지

따라서 **실제 ERF는 이론적 RF의 약 $1/\sqrt{L}$ 배**입니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.5 — 유효 Receptive Field (ERF)

출력 뉴런 $y_c$에 대한 입력 위치 $i$의 기여도를 다음으로 정의합니다:

$$R(i) = \left| \frac{\partial y_c}{\partial x_i} \right|$$

**유효 receptive field**는 $R(i)$가 유의미한 크기 이상인 위치들의 집합:

$$\text{ERF}_\alpha = \{i : R(i) > \alpha \cdot \max_j R(j)\}$$

보통 $\alpha = 0.5$ (최대값의 50% 이상)를 사용합니다.

### 정리 3.6 — Central Limit Theorem을 통한 ERF 분석

각 입력 위치 $i$에 대해, gradient $\partial y_c / \partial x_i$는 다음과 같이 근사됩니다:

$$\frac{\partial y_c}{\partial x_i} \approx \mathcal{N}(0, \sigma^2_L(i))$$

여기서 $\sigma_L(i)$는:

$$\sigma_L(i) \approx C \cdot \rho^{|i - c|/L}$$

$C$는 상수, $\rho < 1$은 감쇠율. 따라서:

$$E[\sigma_L(i)] \sim e^{-|i-c|/\sqrt{L}} \quad (\text{Gaussian-like})$$

### 정의 3.7 — 유효 RF 범위 (Full Width at Half Maximum)

$$\text{ERF}_{\text{FWHM}} = \{i : \sigma_L(i) \geq 0.5 \cdot \sigma_L(c)\}$$

통상적으로:
$$|\text{ERF}_{\text{FWHM}}| \approx \sqrt{L} \cdot \text{const}$$

---

## 🔬 증명 또는 수학적 유도

### Gradient Flow와 Central Limit Theorem

$L$층 네트워크의 경우, backpropagation은 다음과 같이 전개됩니다:

$$\frac{\partial y_c}{\partial x_i} = \sum_{\text{paths from } i \text{ to } c} \prod_{l=1}^L w_l$$

각 경로의 가중치 곱 $\prod w_l$는:
1. 각 경로마다 다름
2. 거리 $|i - c|$에 따라 경로 개수가 지수적으로 증가
3. 하지만 각 경로의 크기는 감소

Central limit theorem에 의해, 이 합은 Gaussian 분포에 수렴:

$$\frac{\partial y_c}{\partial x_i} \xrightarrow{L \to \infty} \mathcal{N}(0, \sigma_L^2(i))$$

### Gaussian 근사

Gaussian의 표준편차가 거리에 대해:

$$\sigma_L(i) = \sigma_0 \cdot \exp\left(-\frac{(i-c)^2}{2\sigma_L^2}\right)$$

로 모델링되면, FWHM (Full Width at Half Maximum)는:

$$\text{FWHM} = 2\sqrt{2\ln 2} \cdot \sigma_L \approx 2.355 \cdot \sigma_L$$

Luo et al.의 실험 데이터에 따르면:

$$\sigma_L \propto \sqrt{L}$$

따라서:

$$\text{ERF} \propto \sqrt{L} \quad \square$$

---

## 💻 실험 재현 / PyTorch 구현

### 구현 1: ResNet-50 ERF 측정

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# Pre-trained ResNet-50
model = models.resnet50(pretrained=True)
model.eval()

def compute_gradient_sensitivity(model, input_size=224, num_samples=100):
    """
    Compute gradient-based receptive field
    Average over multiple random input patches
    """
    sensitivity = np.zeros((input_size, input_size))
    
    for sample_idx in range(num_samples):
        # Create random input
        x = torch.randn(1, 3, input_size, input_size, 
                       requires_grad=True)
        
        # Forward pass
        with torch.no_grad():
            x_no_grad = x.detach()
        x.requires_grad = True
        
        # Forward again with gradient tracking
        y = model(x)
        
        # Compute gradient w.r.t central output location
        # For simplicity, use average pool output
        loss = y.mean()  # Aggregate loss
        loss.backward()
        
        # Extract gradient w.r.t input
        grad_input = x.grad.abs().mean(dim=1)[0].cpu().numpy()  # (H, W)
        
        sensitivity += grad_input / num_samples
    
    return sensitivity

# Compute ERF
print("Computing ERF for ResNet-50...")
erf_data = compute_gradient_sensitivity(model, input_size=224, num_samples=50)

# Normalize and plot
erf_normalized = erf_data / erf_data.max()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap
im = axes[0].imshow(erf_normalized, cmap='hot', origin='upper')
axes[0].set_title('ERF Heatmap (ResNet-50)', fontsize=12)
axes[0].set_xlabel('Width')
axes[0].set_ylabel('Height')
plt.colorbar(im, ax=axes[0], label='Normalized gradient magnitude')

# Cross-section along center
center = 224 // 2
cross_section = erf_normalized[center, :]

axes[1].plot(cross_section, linewidth=2, label='Horizontal cross-section')
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, 
               label='50% threshold (FWHM)')
axes[1].set_title('ERF Cross-section at Center', fontsize=12)
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Normalized sensitivity')
axes[1].legend()
axes[1].grid()

# Compute FWHM
above_threshold = np.where(cross_section > 0.5)[0]
if len(above_threshold) > 0:
    fwhm = above_threshold[-1] - above_threshold[0]
    print(f"ERF FWHM (1D): {fwhm}")
    print(f"Theoretical RF: 212 (from previous document)")
    print(f"Ratio ERF/Theoretical RF: {fwhm / 212:.3f}")

plt.tight_layout()
plt.savefig('erf_resnet50.png', dpi=150)
plt.show()
```

### 구현 2: 깊이에 따른 ERF 변화

```python
def build_simple_cnn(depth, kernel_size=3):
    """Build simple CNN with given depth"""
    layers = []
    
    in_channels = 3
    out_channels = 64
    
    for i in range(depth):
        layers.append(nn.Conv2d(in_channels, out_channels, 
                               kernel_size, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    
    return nn.Sequential(*layers)

def measure_erf_by_depth(depths=[5, 10, 15, 20, 25]):
    """Measure ERF for networks of varying depth"""
    erf_measurements = []
    input_size = 224
    
    for depth in depths:
        print(f"Processing depth={depth}...")
        model = build_simple_cnn(depth)
        model.eval()
        
        # Compute gradient-based ERF
        x = torch.randn(1, 3, input_size, input_size, 
                       requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        
        # Measure ERF as FWHM
        grad = x.grad.abs()[0].mean(dim=0).detach().numpy()
        center = input_size // 2
        cross = grad[center, :]
        
        # Find FWHM
        max_val = cross.max()
        above_half = np.where(cross > 0.5 * max_val)[0]
        if len(above_half) > 0:
            fwhm = above_half[-1] - above_half[0]
        else:
            fwhm = 1
        
        erf_measurements.append(fwhm)
    
    return erf_measurements

depths = [5, 10, 15, 20, 25, 30]
erfs = measure_erf_by_depth(depths)

# Plot ERF vs Depth
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(depths, erfs, 'o-', linewidth=2, markersize=8, label='Measured ERF')

# Theoretical prediction: ERF ~ sqrt(depth)
theoretical = np.array(depths) ** 0.5 * (erfs[0] / (depths[0] ** 0.5))
ax.plot(depths, theoretical, 's--', linewidth=2, markersize=6, 
       label='Theoretical: ERF ~ sqrt(L)', color='red')

ax.set_xlabel('Network Depth (L)', fontsize=12)
ax.set_ylabel('Effective Receptive Field', fontsize=12)
ax.set_title('ERF Growth with Network Depth', fontsize=12)
ax.legend(fontsize=11)
ax.grid()

plt.savefig('erf_vs_depth.png', dpi=150)
plt.show()

print("\nERF measurements:")
for d, e in zip(depths, erfs):
    print(f"Depth {d:2d}: ERF = {e:6.1f}")
```

### 구현 3: 이론적 RF vs 유효 RF 비교

```python
def compare_theoretical_vs_effective():
    """
    Compare theoretical RF from document 01 with effective RF
    """
    
    # AlexNet-like architecture
    alexnet_config = [
        (11, 4), (5, 1), (3, 1), (3, 1), (3, 1)
    ]
    
    # Compute theoretical RF
    def compute_rf(config):
        rf = 1
        product_s = 1
        rfs = [rf]
        for k, s in config:
            rf = rf + (k - 1) * product_s
            product_s *= s
            rfs.append(rf)
        return rfs[-1]
    
    theo_rf = compute_rf(alexnet_config)
    
    # Build and measure effective RF
    model = models.alexnet(pretrained=True)
    model.eval()
    
    # Simplified: just measure on center output
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    y = model.features(x)  # Feature extraction
    
    loss = y.mean()
    loss.backward()
    
    grad = x.grad.abs()[0].mean(dim=0).numpy()
    center = 224 // 2
    cross = grad[center, :]
    
    max_val = cross.max()
    above_half = np.where(cross > 0.5 * max_val)[0]
    erf = above_half[-1] - above_half[0] if len(above_half) > 0 else 1
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Theoretical RF as circle
    x_grid = np.linspace(-224//2, 224//2, 224)
    y_grid = np.linspace(-224//2, 224//2, 224)
    XX, YY = np.meshgrid(x_grid, y_grid)
    
    theo_rf_2d = np.sqrt(XX**2 + YY**2) <= theo_rf / 2
    axes[0].imshow(theo_rf_2d, cmap='Blues', origin='upper')
    axes[0].set_title(f'Theoretical RF (radius={theo_rf//2})', fontsize=12)
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    
    # Effective RF as heatmap
    erf_2d = np.exp(-((XX**2 + YY**2) / ((erf/2)**2)))
    im = axes[1].imshow(erf_2d, cmap='Reds', origin='upper')
    axes[1].set_title(f'Effective RF (FWHM={erf})', fontsize=12)
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('theo_vs_erf.png', dpi=150)
    plt.show()
    
    print(f"Theoretical RF: {theo_rf}")
    print(f"Effective RF (FWHM): {erf}")
    print(f"Ratio (ERF/Theoretical): {erf / theo_rf:.3f}")

compare_theoretical_vs_effective()
```

**예상 결과**:
- Theoretical RF: ~51 (AlexNet)
- Effective RF: ~13-15 (약 25-30%)
- 깊이에 따라 ERF는 sqrt(L) 비례 증가

---

## 🔗 이론과 실전의 간극

### Gradient Propagation의 현실

실제 신경망에서:
1. **Gradient vanishing**: 깊이 증가 시 gradient 크기 감소
2. **Normalization layers**: BatchNorm이 gradient 흐름 개선
3. **Skip connections**: ResNet의 skip이 peripheral gradient 유지

따라서:
- Plain CNN: ERF ≈ sqrt(depth) 효과 뚜렷
- ResNet/DenseNet: Skip 덕분에 ERF 더 크게 유지

### 설계 연구

Semantic segmentation에서:
- **Naive FCN**: RF 부족 → 작은 객체 놓침
- **Dilated convolution**: 이론적 RF 증가 → ERF도 증가
- **Multi-scale**: 다양한 RF 레벨 학습

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Gradient magnitude = 실제 기여도 | Gradient direction도 중요 |
| Gaussian 근사 | 실제 분포는 비대칭일 수 있음 |
| Independent paths | Skip connection, residual 경로 있음 |
| 고정된 가중치 | 학습 중 가중치 분포 변함 |
| Single-pixel output | 실제로는 feature map 크기 큼 |

---

## 📌 핵심 정리

$$\boxed{\text{ERF} \approx \frac{\text{Theoretical RF}}{\sqrt{L}} \text{ (Gaussian 분포로 수렴)}}$$

| 개념 | 정의 |
|------|------|
| **유효 RF (ERF)** | Gradient 기반 실제 기여도가 유의미한 영역 |
| **중심 집중** | 경계 픽셀은 exponential decay로 기여도 감소 |
| **Central Limit Theorem** | Gradient 분포가 Gaussian에 수렴 (깊이 증가) |
| **FWHM** | Full Width Half Maximum, ERF 측정 기준 |
| **sqrt(L) 감쇠** | 깊이 L에 따라 ERF는 sqrt(L)에 비례 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 이론적 RF=51, 깊이=25인 네트워크의 예상 ERF(FWHM)를 계산하라 (sqrt(25)=5 가정).

<details>
<summary>힌트 및 해설</summary>

ERF ≈ Theoretical RF / sqrt(L) = 51 / sqrt(25) = 51 / 5 = 10.2

따라서 예상 ERF는 약 10-11 픽셀 범위.

</details>

**문제 2** (심화): ResNet의 skip connection이 ERF를 증가시키는 메커니즘을 설명하라.

<details>
<summary>힌트 및 해설</summary>

Skip connection은 입력을 직접 깊은 층으로 전파합니다:

$$y = x + f(x)$$

따라서 gradient는:
$$\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$$

항등 행렬 부분 $I$는 경계 픽셀의 gradient를 직접 전달하므로, vanishing이 적게 됩니다. 결과적으로 ERF가 더 크게 유지됩니다.

이것이 ResNet이 매우 깊은 네트워크(152 layers)도 잘 학습할 수 있는 이유 중 하나입니다.

</details>

**문제 3** (논문 비평): Luo et al. (2016)의 "이론적 RF는 과대 추정"이라는 주장이 semantic segmentation 설계에 어떤 영향을 미쳤는가?

<details>
<summary>힌트 및 해설</summary>

이전: "RF가 크면 충분하다"고 단순 가정 → DeepLab v1에서 매우 큰 dilation rate

현재: ERF가 이론적 RF보다 훨씬 작음을 알게 됨 → 더 전략적 dilation 설계

예: DeepLab v3+의 ASPP (Atrous Spatial Pyramid Pooling)는 다양한 dilation rate의 조합으로 실제 effective multi-scale context 확보.

</details>

---

<div align="center">

[◀ 이전](./01-theoretical-rf.md) | [📚 README](../README.md) | [다음 ▶](./03-dilated-rf.md)

</div>
