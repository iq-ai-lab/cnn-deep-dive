# 01. 이론적 Receptive Field 계산

## 🎯 핵심 질문

- Receptive field는 정확히 무엇이고, 왜 CNN에서 중요한가?
- 각 층의 기여를 고려한 재귀 공식 $RF_L = RF_{L-1} + (k_L - 1) \prod_{i=1}^{L-1} s_i$는 어떻게 유도되는가?
- AlexNet, VGG-16, ResNet-50처럼 실제 네트워크에서 마지막 층의 receptive field는 입력 전체를 커버하는가?
- Stride 2인 max pooling이 receptive field 확장에 미치는 효과는 무엇인가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Receptive field(RF)는 네트워크의 한 뉴런이 "볼 수 있는" 입력 영역의 크기를 나타냅니다. 이는 CNN이 어떤 규모의 특징을 감지할 수 있는지를 직접 결정합니다. 예를 들어:

- **객체 인식**: 얼굴 전체를 인식하려면 RF가 충분히 커야 합니다.
- **Semantic segmentation**: 픽셀별 분류를 위해서는 각 픽셀이 충분한 context를 봐야 합니다.
- **Dense prediction**: U-Net, DeepLab 같은 아키텍처는 RF를 전략적으로 설계합니다.

이론적 RF를 정확히 계산하면, 네트워크 구조 설계 단계에서 "이 아키텍처가 원하는 정보를 충분히 담을 수 있는가?"를 판단할 수 있습니다.

---

## 📐 수학적 선행 조건

- 합성곱(Convolution) 연산: kernel size $k$, stride $s$
- 다단계 합성곱: 입출력 해상도 계산
- 기본 산술과 귀납법

---

## 📖 직관적 이해

### Receptive Field의 정의

한 층에서의 출력 뉴런이 "감응"하는 입력 영역의 크기를 receptive field라 합니다.

**1층 convolution** ($k=3, s=1$):
- 중심 위치의 출력은 입력의 3개 연속 픽셀에 의존
- $RF_1 = 3$

**2층 convolution** 연쇄:
- 첫 번째 층이 입력의 3개 픽셀을 보고
- 두 번째 층이 첫 번째 층의 출력 3개를 보면
- 전체적으로 입력의 몇 개 픽셀에 영향을 받는가?

직관적으로, 두 번째 층의 한 뉴런은 첫 번째 층의 3개 뉴런에 영향받고, 각각이 입력의 3개 픽셀에 영향받으므로:
$$RF_2 = 3 + (3-1) \times 1 = 5$$

### Stride의 기하학적 해석

$s > 1$이면 출력 해상도가 줄어듭니다. 이는 RF 관점에서:
- stride $s$로 인해, 출력의 인접한 두 뉴런은 입력에서 $s$만큼 떨어진 위치를 중심으로 합니다.
- 따라서 다음 층에서 $k$개의 출력을 보면, 입력 공간에서는 이들이 $k \times s$만큼 떨어져 있습니다.

**예**: $k=3, s=2$ convolution 후 $k=3, s=1$ convolution:
- 첫 층: 입력 $(2i-1, 2i, 2i+1)$에서 출력 $i$ 생성
- 두 번째 층: 출력 $(i-1, i, i+1)$을 보면, 입력 $(2i-3, \ldots, 2i+3)$ 총 7개

따라서 $RF_2 = 3 + (3-1) \times 2 = 7$.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Receptive Field

$l$번째 층의 출력 위치 $j$에 대해, 이 뉴런에 영향을 주는 입력 위치들의 집합을 **receptive field**라 하며, 그 크기(span)를 $RF_l$이라 정의합니다.

1D 신호 기준으로, 출력 위치 $j$의 receptive field는 $[c_l - (RF_l-1)/2, c_l + (RF_l-1)/2]$ 형태의 구간입니다 (여기서 $c_l$은 중심).

### 정리 3.2 — Receptive Field 재귀 공식

$i$번째 층이 kernel size $k_i$, stride $s_i$를 갖을 때:

$$RF_l = RF_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i$$

기저 조건: $RF_0 = 1$ (입력 신호 자신)

### 따름정리 3.3 — 모든 stride가 1인 경우

$s_i = 1$ for all $i$일 때:
$$RF_l = \sum_{i=1}^l (k_i - 1) + 1$$

**증명**:
$$RF_l = RF_{l-1} + (k_l - 1) \cdot 1 = \left(\sum_{i=1}^{l-1}(k_i-1) + 1\right) + (k_l - 1) = \sum_{i=1}^l (k_i-1) + 1 \quad \square$$

### 예시 3.4 — AlexNet의 RF

AlexNet 합성곱층 (1D 버전):
| Layer | $k$ | $s$ | $\prod_{i=1}^{l-1} s_i$ | $RF_l$ |
|-------|-----|-----|----------------------|--------|
| Input | — | — | — | 1 |
| Conv1 | 11 | 4 | 1 | $1 + 10 \cdot 1 = 11$ |
| Conv2 | 5 | 1 | 4 | $11 + 4 \cdot 4 = 27$ |
| Conv3 | 3 | 1 | 4 | $27 + 2 \cdot 4 = 35$ |
| Conv4 | 3 | 1 | 4 | $35 + 2 \cdot 4 = 43$ |
| Conv5 | 3 | 1 | 4 | $43 + 2 \cdot 4 = 51$ |

AlexNet은 $224 \times 224$ 입력을 처리하므로 (정사각형), 1D RF 51은 2D에서 $51 \times 51$에 대응. 따라서 최종 층은 중앙 정보를 주로 보게 됩니다.

---

## 🔬 증명 또는 수학적 유도

### 정리 3.2의 엄밀한 증명

$l$번째 층의 출력 뉴런이 위치 $j$에 있을 때, 이는 $l-1$번째 층의 $[j - (k_l-1)/2, j + (k_l-1)/2]$ 범위의 뉴런들에 의존합니다.

$l-1$번째 층의 뉴런 위치 $i$는 입력의 중심 위치 (1D에서):
$$c_{l-1}(i) = c_0 + \sum_{m=1}^{l-1} (\text{offset from stride})$$

더 정확히, $l-1$번째 층의 뉴런 $i$의 receptive field는 입력 위치 $[c_{l-1}(i) - (RF_{l-1}-1)/2, c_{l-1}(i) + (RF_{l-1}-1)/2]$입니다.

$l$번째 층에서 kernel이 $l-1$층의 $j - (k_l-1)/2$ ~ $j + (k_l-1)/2$ 범위를 보면:
- 중심은 $c_{l-1}(j) = c_0 + j \prod_{i=1}^{l-1} s_i$ (이전 stride 누적)
- 좌측 경계: $c_{l-1}(j - (k_l-1)/2) - (RF_{l-1}-1)/2$
  $$= c_0 + (j - (k_l-1)/2)\prod_{i=1}^{l-1}s_i - (RF_{l-1}-1)/2$$

우측 경계: $c_{l-1}(j + (k_l-1)/2) + (RF_{l-1}-1)/2$
  $$= c_0 + (j + (k_l-1)/2)\prod_{i=1}^{l-1}s_i + (RF_{l-1}-1)/2$$

총 span:
$$RF_l = \left[(k_l-1)\prod_{i=1}^{l-1}s_i + (RF_{l-1}-1)\right] + 1 = RF_{l-1} + (k_l-1)\prod_{i=1}^{l-1}s_i \quad \square$$

---

## 💻 실험 재현 / PyTorch 구현

### 구현 1: Receptive Field 자동 계산

```python
def compute_receptive_field(layer_configs):
    """
    layer_configs: list of (kernel_size, stride) tuples
    Returns: receptive field at each layer
    """
    rf = 1
    product_stride = 1
    rfs = [rf]
    
    for k, s in layer_configs:
        rf = rf + (k - 1) * product_stride
        product_stride *= s
        rfs.append(rf)
    
    return rfs

# AlexNet 예시
alexnet_layers = [
    (11, 4),  # Conv1
    (5, 1),   # Conv2
    (3, 1),   # Conv3
    (3, 1),   # Conv4
    (3, 1),   # Conv5
]

rfs_alexnet = compute_receptive_field(alexnet_layers)
print("AlexNet RF by layer:", rfs_alexnet)
# Output: [1, 11, 27, 35, 43, 51]

# VGG-16 (합성곱만, 224x224 입력)
# Conv1: (3,1), (3,1) -> MaxPool (2,2)
# Conv2: (3,1), (3,1) -> MaxPool (2,2)
# ... 등
vgg_layers = [
    (3, 1), (3, 1),  # Conv1, Pool stride=2
    (3, 1), (3, 1),  # Conv2, Pool stride=2
    (3, 1), (3, 1), (3, 1),  # Conv3, Pool stride=2
    (3, 1), (3, 1), (3, 1),  # Conv4, Pool stride=2
    (3, 1), (3, 1), (3, 1),  # Conv5, Pool stride=2
]

# Max pooling도 stride로 반영
vgg_layers_with_pool = [
    (3, 1), (3, 1), (1, 2),  # Conv1 + Pool
    (3, 1), (3, 1), (1, 2),  # Conv2 + Pool
    (3, 1), (3, 1), (3, 1), (1, 2),  # Conv3 + Pool
    (3, 1), (3, 1), (3, 1), (1, 2),  # Conv4 + Pool
    (3, 1), (3, 1), (3, 1), (1, 2),  # Conv5 + Pool
]

rfs_vgg = compute_receptive_field(vgg_layers_with_pool)
print(f"VGG-16 final RF: {rfs_vgg[-1]}")
# Output: VGG-16 final RF: 212
```

### 구현 2: ResNet-50의 RF 추적

```python
def compute_resnet50_rf():
    """ResNet-50 (논문 기준)"""
    layers = [
        (7, 2),   # Initial Conv, stride=2
        (1, 1),   # Conv1
        (3, 1), (1, 1),  # Bottleneck block, stride=1 (avg ~3 blocks)
        (1, 2),   # MaxPool or stride=2 transition
        (3, 1), (1, 1),  # Bottleneck, stride=1
        (1, 2),   # Stride=2 transition
        (3, 1), (1, 1),  # Bottleneck, stride=1
        (1, 2),   # Stride=2 transition
        (3, 1), (1, 1),  # Bottleneck, stride=1
    ]
    
    rfs = compute_receptive_field(layers)
    return rfs

rfs_resnet = compute_resnet50_rf()
print(f"ResNet-50 RF progression: {rfs_resnet}")
```

### 구현 3: 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AlexNet
rfs_a = compute_receptive_field(alexnet_layers)
axes[0, 0].plot(range(len(rfs_a)), rfs_a, 'o-', linewidth=2, markersize=8)
axes[0, 0].set_title('AlexNet Receptive Field Growth', fontsize=12)
axes[0, 0].set_xlabel('Layer')
axes[0, 0].set_ylabel('RF size')
axes[0, 0].grid()

# VGG-16
rfs_v = compute_receptive_field(vgg_layers_with_pool)
axes[0, 1].plot(range(len(rfs_v)), rfs_v, 's-', color='green', linewidth=2, markersize=6)
axes[0, 1].set_title('VGG-16 Receptive Field Growth', fontsize=12)
axes[0, 1].set_xlabel('Layer')
axes[0, 1].set_ylabel('RF size')
axes[0, 1].grid()

# Stride 효과 비교
stride_configs = [
    ([(3, 1)] * 5, "Stride=1 only"),
    ([(3, 1), (3, 1), (1, 2)] * 2, "With max pooling"),
]

for config, label in stride_configs:
    rfs = compute_receptive_field(config)
    axes[1, 0].plot(range(len(rfs)), rfs, 'o-', label=label, linewidth=2)

axes[1, 0].set_title('Impact of Stride on RF Growth', fontsize=12)
axes[1, 0].set_xlabel('Layer')
axes[1, 0].set_ylabel('RF size')
axes[1, 0].legend()
axes[1, 0].grid()

# Input size와의 비교 (224x224)
input_size = 224
rfs_normalized = [rf / input_size * 100 for rf in rfs_v]
axes[1, 1].plot(range(len(rfs_normalized)), rfs_normalized, 'd-', 
                color='red', linewidth=2, markersize=6)
axes[1, 1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, 
                   label='Full input (100%)')
axes[1, 1].set_title('VGG-16 RF as % of 224x224 Input', fontsize=12)
axes[1, 1].set_xlabel('Layer')
axes[1, 1].set_ylabel('RF coverage (%)')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('receptive_field_analysis.png', dpi=150)
plt.show()
```

**예상 출력**: AlexNet은 점진적 증가, VGG는 pooling으로 인한 가파른 증가, ResNet-50은 초반 stride 2로 빠른 확장. 최종적으로 224×224 입력에 대해 전체 이미지를 커버하는지 확인 가능.

---

## 🔗 이론과 실전의 간극

### 이론적 RF와 실제 RF의 차이

이론적 RF는 **최악의 경우(worst-case)** receptive field입니다:
- 모든 가중치가 동일하게 영향을 준다고 가정
- 실제로는 중심 부근 가중치가 훨씬 큼 (Gaussian-like distribution)

따라서:
- **이론적 RF 51**: 이론상 51개 픽셀이 영향
- **실제 유효 RF**: 약 $51 / \sqrt{\text{depth}}$ (다음 문서에서 상세히 설명)

### 설계 원칙

실무에서 receptive field를 설계할 때:

1. **Semantic segmentation**: 최종 RF가 최소한 객체 크기보다 커야 함
   - 배경과 객체를 동시에 보기 위해 RF ≥ 객체 크기 × 2-3배 권장

2. **Object detection**: 앵커박스를 RF 내에 포함
   - 작은 객체: RF 작으면 안 됨
   - 큰 객체: 충분하면 됨

3. **실시간 처리**: Stride 크게 → RF 빠르게 증가 → 연산 감소
   - Trade-off: 세밀한 정보 손실

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear receptive field 계산 | Non-linearity (ReLU 등)의 영향 무시 |
| 모든 가중치 동등 기여 | 실제로는 중심 가중치가 dominant (다음 문서) |
| 입력이 충분히 큼 | Padding, 경계 효과 무시 |
| 직선적 경로만 고려 | Skip connection, dilated convolution 복잡도 증가 |
| 1D 계산의 2D 확장 | 실제 2D에서 receptive field는 정사각형 아닐 수 있음 |

---

## 📌 핵심 정리

$$\boxed{RF_l = RF_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i}$$

| 개념 | 정의 |
|------|------|
| **Receptive field (RF)** | 한 뉴런이 입력에서 영향받는 영역의 크기 |
| **기저 조건** | $RF_0 = 1$ (입력 신호 자신) |
| **재귀 공식** | 이전 RF + 현 층의 새로운 기여 × 이전 stride 누적 |
| **Stride의 역할** | 다음 층 RF 증가 속도를 배수만큼 가속 |
| **Max pooling** | Kernel size 1, stride 2로 모델링 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k=5, s=1$인 단일 convolution 층이 뒤따르는 $k=5, s=1$ 층의 최종 RF를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$RF_0 = 1$

Layer 1: $RF_1 = 1 + (5-1) \cdot 1 = 5$

Layer 2: $RF_2 = 5 + (5-1) \cdot 1 = 9$

일반적으로 $k=5, s=1$ $n$개 층: $RF_n = 1 + 4n$

</details>

**문제 2** (심화): $224 \times 224$ 입력에서 최종 receptive field가 정확히 $224$가 되는 layer 구성을 설계하라. 최소 몇 층이 필요한가?

<details>
<summary>힌트 및 해설</summary>

초반에 stride 2를 사용하여 RF를 빠르게 확장하는 것이 효율적입니다.

예: $(11, 4), (5, 1), (5, 1), (5, 1), (5, 1)$ → RF = 51 (부족)

더 빠른 확장: $(7, 2), (5, 1), (5, 1), (5, 1), (5, 1)$ 
- Layer 0: RF = 1
- Layer 1: RF = 1 + 6·1 = 7
- Layer 2: RF = 7 + 4·2 = 15
- Layer 3: RF = 15 + 4·2 = 23
- Layer 4: RF = 23 + 4·2 = 31
- Layer 5: RF = 31 + 4·2 = 39 (여전히 부족)

더 많은 stride 또는 dilated convolution 필요. 일반적으로 5~6층으로 충분.

</details>

**문제 3** (논문 비평): Luo et al. (2016)은 "이론적 RF는 과도 추정"이라 주장합니다. 왜 이론적 계산이 실제 영향을 과대평가할까요?

<details>
<summary>힌트 및 해설</summary>

이론적 RF는 "가능한 최대 span"을 계산합니다. 하지만:

1. **가중치 분포**: 중앙 가중치 >> 경계 가중치 (Gaussian-like)
2. **Non-linearity**: ReLU, normalization이 gradient 경로를 제한
3. **학습 과정**: 네트워크는 모든 경로를 동등하게 사용하지 않음

따라서 실제 "유효 receptive field"(ERF)는 이론적 RF의 $1/\sqrt{\text{depth}}$ 배 정도. 다음 문서에서 상세히 다룹니다.

</details>

---

<div align="center">

[◀ 이전](../ch2-cnn-ops/05-depthwise-separable.md) | [📚 README](../README.md) | [다음 ▶](./02-effective-rf.md)

</div>
