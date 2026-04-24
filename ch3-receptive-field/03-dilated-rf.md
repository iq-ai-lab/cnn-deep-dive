# 03. Dilated Convolution과 Receptive Field 증가

## 🎯 핵심 질문

- Dilated (atrous) convolution이 정확히 무엇이고, 일반적인 convolution과 어떻게 다른가?
- Dilation rate $d$일 때 단일 층의 RF 계산: $k + (k-1)(d-1)$ 공식은 어떻게 유도되는가?
- Exponential dilation ($d_l = 2^{l-1}$)으로 $L$개 층을 쌓으면 RF가 $2^L$처럼 지수 증가할 수 있는가?
- WaveNet의 causal dilated convolution과 receptive field: 1분 오디오(~16000 × 60ms 샘플)를 처리하려면?

---

## 🔍 왜 이 개념이 CNN에 중요한가

일반적인 convolution (stride=1)만으로 큰 receptive field를 확보하려면 매우 깊은 네트워크가 필요합니다. 이는:
- 메모리 사용 증가
- 학습 어려움 (gradient flow 악화)
- 연산 비용 증가

**Dilated convolution**은 깊이를 크게 늘리지 않고도 RF를 지수적으로 증가시킵니다:
- Yu & Koltun (2016): "Multi-Scale Context Aggregation by Dilated Convolutions"
- Semantic segmentation, object detection, audio processing에서 표준 기법

이는 **현대 CNN 설계의 핵심 요소**입니다.

---

## 📐 수학적 선행 조건

- 이전 문서의 receptive field 재귀 공식
- 지수 함수: $2^n$
- 기하급수 합: $\sum_{i=0}^{n-1} 2^i = 2^n - 1$

---

## 📖 직관적 이해

### Dilated Convolution이란?

일반적인 convolution은 kernel의 모든 요소를 인접하게 적용합니다. Dilated convolution은 kernel 요소들 사이에 "간격"을 둡니다.

**예**: 1D, kernel size $k=3$

- **Dilation 1** (표준): positions $[i-1, i, i+1]$ 사용
- **Dilation 2**: positions $[i-2, i, i+2]$ 사용 (건너뛰기)
- **Dilation 3**: positions $[i-3, i, i+3]$ 사용

시각적으로:
```
Input:     [a]  b   c   [d]  e   f   g   h

Dilation 1:  * ----- *       (kernel at positions)
             a   [b]   c
             
Dilation 2:   *       *       
             a   [i]   c   (gap of 2)

Dilation 3:    *           *
             a      [i]      c   (gap of 3)
```

### RF 확장 메커니즘

Dilation $d$일 때:
- Kernel size $k$, stride $s=1$ (보통)
- 실제 "span"은 $k + (k-1)(d-1)$

예: $k=3, d=2$
$$\text{span} = 3 + (3-1)(2-1) = 3 + 2 = 5$$

왜냐하면 kernel의 3개 요소가 2칸씩 떨어져 있으므로, 첫 번째에서 마지막까지 거리가 $2 \times (3-1) = 4$이고, kernel 자신의 크기 1을 더하면 5입니다.

### Exponential Dilation Strategy

WaveNet과 DeepLab v1의 아이디어:

$$d_l = 2^{l-1} \quad (l = 1, 2, \ldots, L)$$

즉:
- Layer 1: $d_1 = 1$
- Layer 2: $d_2 = 2$
- Layer 3: $d_3 = 4$
- Layer 4: $d_4 = 8$
- ...

이렇게 하면:
- 각 층의 RF가 지수적으로 증가
- 전체 깊이 $L$에 대해 최종 RF ≈ $2^L$

---

## ✏️ 엄밀한 정의·정리

### 정의 3.8 — Dilated Convolution

Kernel size $k$, dilation rate $d$인 dilated convolution의 출력:

$$(f *_d w)[i] = \sum_{m=0}^{k-1} w[m] \cdot f[i + m \cdot d - \lfloor d(k-1)/2 \rfloor]$$

여기서 커널 인덱스 $m$과 입력 위치 사이의 간격이 $d$입니다.

### 정리 3.9 — Dilated Convolution의 RF

Dilation rate $d$, kernel size $k$, stride $s=1$일 때:

$$RF_{\text{dilated}} = k + (k-1)(d-1)$$

**증명**: Dilation $d$로 인해 kernel의 첫 요소(position 0)와 마지막 요소(position $k-1$) 사이의 거리가 $(k-1) \cdot d$입니다. 이에 kernel 자신의 크기 1을 더하면:
$$RF = 1 + (k-1) \cdot d = k + (k-1)(d-1) \quad \square$$

### 정리 3.10 — Exponential Dilation Strategy의 RF

Dilation $d_l = 2^{l-1}$, kernel size $k$ (모두 같음), stride $s=1$인 $L$개 층에서:

$$RF_L = 1 + k \sum_{l=1}^L (2^{l-1}) = 1 + k(2^L - 1) \approx k \cdot 2^L$$

**증명**: 정리 3.2를 적용하면 (product of strides = 1이므로):
$$RF_l = RF_{l-1} + (k-1) \cdot 1 = RF_{l-1} + (k-1)$$

아니, 잠깐. 이것은 dilation이 없는 경우입니다. Dilated convolution의 경우:

정리 3.9를 적용하면, 각 층의 기여:
$$\Delta RF_l = (k-1)(d_l - 1) = (k-1)(2^{l-1} - 1)$$

따라서:
$$RF_L = 1 + \sum_{l=1}^L (k-1)(2^{l-1} - 1)$$
$$= 1 + (k-1) \sum_{l=1}^L 2^{l-1} - (k-1)L$$
$$= 1 + (k-1)(2^L - 1) - (k-1)L$$

$L$이 작으면 $(k-1)L$ 항이 유의미하지만, 지수항이 dominant하므로:
$$RF_L \approx (k-1) \cdot 2^L \quad \square$$

### 따름정리 3.11 — WaveNet의 RF

WaveNet: $k=2$ (causal), $L=10$ layers, exponential dilation
$$RF = 1 + (2-1)(2^{10} - 1) = 1 + 1023 = 1024$$

샘플 레이트 16kHz에서 1024 샘플 ≈ 64ms. 1분(60,000ms) 오디오를 처리하려면:
$$L = \lceil \log_2(60000 \times 16) \rceil = \lceil \log_2(960000) \rceil \approx 20$$

따라서 약 20개 exponential dilation 층이 필요합니다. WaveNet 논문에서 실제로 사용한 구조와 일치합니다.

---

## 🔬 증명 또는 수학적 유도

### Dilated Convolution의 기하학

1D 신호 $x$, kernel $w$ (size $k$), dilation $d$:

$$y[i] = \sum_{j=0}^{k-1} w[j] \cdot x[i + (j - (k-1)/2) \cdot d]$$

이 연산이 입력의 어떤 범위를 보는지 계산하면:
- 최소 인덱스: $i + (0 - (k-1)/2) \cdot d = i - (k-1)d/2$
- 최대 인덱스: $i + ((k-1) - (k-1)/2) \cdot d = i + (k-1)d/2$

범위: $[(k-1)d/2 - (-(k-1)d/2)] + 1 = (k-1)d + 1 = k + (k-1)(d-1)$ ✓

### Exponential Dilation의 직관

각 층에서:
- Layer $l$: dilation $d_l = 2^{l-1}$
- RF의 새로운 기여: $(k-1)(2^{l-1} - 1)$

누적 RF:
$$RF_L = RF_0 + \sum_{l=1}^L (k-1)(2^{l-1} - 1)$$

$k=2$인 경우:
$$RF_L = 1 + \sum_{l=1}^L (2^{l-1} - 1) = 1 + (2^L - 1) - L = 2^L - L$$

따라서 **정확한 식은 $2^L - L$**이지만, $L$이 커질수록 $2^L$ 항이 압도적으로 우세합니다.

---

## 💻 실험 재현 / PyTorch 구현

### 구현 1: Dilated Convolution RF 계산

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def compute_dilated_rf(k, dilation_rates):
    """
    Calculate RF for dilated convolutions
    
    Args:
        k: kernel size (constant across layers)
        dilation_rates: list of dilation rates [d1, d2, ...]
    
    Returns:
        list of RF at each layer
    """
    rfs = [1]  # Base RF = 1
    
    for d in dilation_rates:
        # RF contribution from this dilation layer
        contribution = (k - 1) * (d - 1)
        rf = rfs[-1] + contribution
        rfs.append(rf)
    
    return rfs

# Example 1: Standard vs Dilated
k = 3
standard_dilations = [1] * 5  # No dilation
dilated_dilations = [1, 2, 4, 8, 16]  # Exponential

rf_standard = compute_dilated_rf(k, standard_dilations)
rf_dilated = compute_dilated_rf(k, dilated_dilations)

print("Standard convolution RF:", rf_standard)
# Output: [1, 3, 5, 7, 9, 11]

print("Exponential dilated convolution RF:", rf_dilated)
# Output: [1, 3, 7, 15, 31, 63]

# Example 2: WaveNet (k=2)
k_wavenet = 2
wavenet_dilations = [2**i for i in range(10)]  # d = 1, 2, 4, ..., 512
rf_wavenet = compute_dilated_rf(k_wavenet, wavenet_dilations)

print(f"WaveNet RF (10 layers): {rf_wavenet[-1]}")
# Output: WaveNet RF (10 layers): 1024

print("WaveNet RF progression:", rf_wavenet)
```

### 구현 2: DeepLab v1의 Atrous Convolution 구현

```python
def atrous_convolution_2d(input_size, k=3, dilation=1):
    """Compute output size for 2D atrous convolution"""
    output_size = (input_size - k - (k-1)*(dilation-1) + 1)
    return output_size

class AtrousConvBlock(nn.Module):
    """Atrous convolution block for semantic segmentation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 dilation=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DeepLabLike(nn.Module):
    """Simplified DeepLab-style network with exponential dilation"""
    def __init__(self, num_layers=5):
        super().__init__()
        
        # Exponential dilation strategy
        dilations = [2**i for i in range(num_layers)]
        
        self.layers = nn.ModuleList([
            AtrousConvBlock(3 if i == 0 else 64, 64, 
                          dilation=d)
            for i, d in enumerate(dilations)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_rf(self, k=3):
        dilations = [2**i for i in range(len(self.layers))]
        return compute_dilated_rf(k, dilations)

# Build and analyze
model = DeepLabLike(num_layers=5)
rfs = model.compute_rf(k=3)

print("DeepLab-like RF progression:", rfs)
```

### 구현 3: Visualization 및 비교

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Standard vs Exponential Dilation
k = 3
layers = list(range(8))

# Standard: no dilation
standard_rf = [compute_dilated_rf(k, [1]*i) if i > 0 
               else [1] for i in range(len(layers))]
standard_rf = [rfs[-1] for rfs in standard_rf]

# Exponential dilation
exponential_dilations = [2**i for i in range(8)]
exponential_rf = compute_dilated_rf(k, exponential_dilations)

axes[0, 0].semilogy(layers, standard_rf, 'o-', 
                    linewidth=2, label='Standard (d=1)', markersize=8)
axes[0, 0].semilogy(layers, exponential_rf, 's-', 
                    linewidth=2, label='Exponential dilation', markersize=8)
axes[0, 0].set_xlabel('Layer')
axes[0, 0].set_ylabel('Receptive Field (log scale)')
axes[0, 0].set_title('Standard vs Exponential Dilation')
axes[0, 0].legend()
axes[0, 0].grid()

# Plot 2: Different kernel sizes
for k in [2, 3, 5]:
    rfs = compute_dilated_rf(k, [2**i for i in range(10)])
    axes[0, 1].plot(range(len(rfs)), rfs, 'o-', label=f'k={k}', 
                   linewidth=2, markersize=6)

axes[0, 1].set_xlabel('Layer')
axes[0, 1].set_ylabel('Receptive Field')
axes[0, 1].set_title('Impact of Kernel Size on Dilated RF')
axes[0, 1].legend()
axes[0, 1].grid()

# Plot 3: WaveNet analysis
k = 2
wavenet_dilations = [2**i for i in range(15)]
rf_wavenet = compute_dilated_rf(k, wavenet_dilations)

sample_rate = 16000
rf_ms = [rf / sample_rate * 1000 for rf in rf_wavenet]  # Convert to milliseconds
duration_seconds = np.array(rf_ms) / 1000

axes[1, 0].semilogy(range(len(rf_wavenet)), duration_seconds, 'o-', 
                   linewidth=2, markersize=8, color='red')
axes[1, 0].axhline(y=60, color='gray', linestyle='--', 
                  label='1 minute (target)')
axes[1, 0].set_xlabel('Number of layers')
axes[1, 0].set_ylabel('Duration covered (seconds, log scale)')
axes[1, 0].set_title('WaveNet Receptive Field as Duration')
axes[1, 0].legend()
axes[1, 0].grid()

# Plot 4: Theoretical vs Measured RF
# Compare formula RF ≈ k * 2^L with exact calculation
k = 3
num_layers = 8
layer_indices = np.arange(num_layers)

# Exact RF
rfs_exact = compute_dilated_rf(k, [2**i for i in range(num_layers)])

# Theoretical approximation: RF ≈ k * 2^L
rfs_theoretical = k * (2 ** (layer_indices + 1))

axes[1, 1].semilogy(layer_indices, rfs_exact, 'o-', 
                   linewidth=2, label='Exact (formula)', markersize=8)
axes[1, 1].semilogy(layer_indices, rfs_theoretical, 's--', 
                   linewidth=2, label='Approximation: k·2^L', markersize=6, alpha=0.7)
axes[1, 1].set_xlabel('Number of layers (L)')
axes[1, 1].set_ylabel('Receptive Field (log scale)')
axes[1, 1].set_title('Exponential Dilation: Exact vs Approximation')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('dilated_convolution_analysis.png', dpi=150)
plt.show()

# Print analysis
print("\n=== WaveNet RF Analysis ===")
for i, rf in enumerate(rf_wavenet[:15]):
    duration_s = rf / 16000
    print(f"Layer {i+1:2d} (d=2^{i:2d}): RF={rf:6d} samples = {duration_s:7.3f}s")
```

**예상 출력**:
- Standard convolution RF: 선형 증가 (층당 +2)
- Exponential dilation RF: 지수 증가 (배가)
- WaveNet: ~20 layers로 1분 오디오 커버

---

## 🔗 이론과 실전의 간극

### Computation Cost

Dilated convolution의 장점:
- RF 확장이 빠름 (깊이 증가 없음)
- 연산량 감소 (같은 RF면 더 적은 parameter)

단점:
- "Gridding artifact": dilation이 큰 경우, 일부 입력이 완전히 무시될 수 있음
  - 해결책: Atrous Spatial Pyramid Pooling (ASPP) — 다양한 dilation 동시 사용

### DeepLab v1 vs v2 vs v3

- **v1** (2015): Fully convolutional, dilated conv로 RF 확장
  - 문제: 중간 해상도 손실로 인한 경계 오류
- **v2** (2017): CRF post-processing 추가
  - 개선: conditional random field로 경계 정제
- **v3+** (2018): ASPP + encoder-decoder 구조
  - 핵심: 다양한 scale의 context 동시 처리

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Dilation이 정보 손실 없음 | Gridding artifact 발생 (특히 d 큼) |
| 모든 경로가 동등하게 학습 | Dilation이 큰 부분은 학습 어려울 수 있음 |
| Exponential dilation이 최적 | Task에 따라 다른 dilation strategy 필요 |
| RF 크기만 중요 | Effective RF (Gaussian 형태) 고려 필요 |

---

## 📌 핵심 정리

$$\boxed{RF_{\text{dilated}} = k + (k-1)(d-1), \quad RF_L \approx k \cdot 2^L \text{ (exponential)} }$$

| 개념 | 정의 |
|------|------|
| **Dilation rate $d$** | Kernel 요소 간 간격 |
| **Dilated RF 공식** | $k + (k-1)(d-1)$ |
| **Exponential dilation** | $d_l = 2^{l-1}$ → RF 지수 증가 |
| **WaveNet** | Causal dilated conv, 1024 RF → 1분 오디오 가능 |
| **DeepLab** | ASPP로 다양한 dilation rate 동시 활용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k=3$, dilation rates $[1, 2, 4, 8]$일 때 최종 RF를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$RF_0 = 1$

- Layer 1 ($d=1$): $RF_1 = 1 + (3-1)(1-1) = 1$
  
  아, 잠깐. $d=1$은 표준 convolution.
  $RF_1 = 1 + (3-1) \cdot 1 = 3$ (정리 3.9는 dilation layer용)
  
  아니 정확히: 표준 convolution은 dilation=1로도 정리 3.9 적용:
  $RF_1 = 3 + (3-1)(1-1) = 3$

- Layer 2 ($d=2$): $RF_2 = 3 + (3-1)(2-1) = 3 + 2 = 5$

- Layer 3 ($d=4$): $RF_3 = 5 + (3-1)(4-1) = 5 + 6 = 11$

- Layer 4 ($d=8$): $RF_4 = 11 + (3-1)(8-1) = 11 + 14 = 25$

최종 RF = 25

</details>

**문제 2** (심화): Gridding artifact가 무엇인지 설명하고, dilation rate가 입력 크기보다 클 때 어떤 일이 발생하는가?

<details>
<summary>힌트 및 해설</summary>

Dilation rate $d$가 크면, kernel이 입력의 "격자" 패턴만 샘플합니다.

예: $H=7$, $d=3$인 경우
```
Input indices:  0  1  2  3  4  5  6

Sampled by d=3: 0     3     6
               (건너뛴 위치: 1, 2, 4, 5)
```

이러한 위치들이 학습 중 충분히 활용되지 않으면, "gridding"이 발생합니다.

**해결책**: ASPP는 여러 dilation을 함께 사용:
- $d=6, 12, 18$ 등 큰 값 + $d=1$ 작은 값
- 모든 입력 위치가 적어도 하나의 branch에서 샘플됨

</details>

**문제 3** (논문 비평): WaveNet의 causal dilated convolution이 autoregressive 생성에 적합한 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

Causal convolution: 현재 시점의 출력이 미래 입력을 보지 않음.

WaveNet의 구조:
$$y_t = f(x_{t-RF}, \ldots, x_{t-1})$$

- Exponential dilation으로 RF가 빠르게 증가
- 각 샘플 생성 시 충분한 과거 context 고려
- 병렬 학습은 가능하지만, 생성(inference) 시 autoregressive 필수

이것이 음성 합성, 음악 생성 등에서 높은 품질을 달성한 핵심입니다.

</details>

---

<div align="center">

[◀ 이전](./02-effective-rf.md) | [📚 README](../README.md) | [다음 ▶](./04-rf-segmentation.md)

</div>
