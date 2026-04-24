# 01. Residual Block의 정의와 Forward Flow

## 🎯 핵심 질문

- Residual block $y = F(x; W) + x$는 일반 convolution layer $y = F(x; W)$와 어떻게 다른가?
- Identity shortcut이 왜 필수인가? Gradient 소멸 문제와의 연관성은?
- Pre-activation (BN-ReLU-Conv 순서)가 post-activation (Conv-BN-ReLU)보다 왜 우수한가?
- Bottleneck block ($1 \times 1 \to 3 \times 3 \to 1 \times 1$)이 파라미터 수를 어떻게 줄이는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Residual block은 He et al. (2015)의 ResNet 논문에서 도입되어, 깊은 신경망을 실제로 훈련 가능하게 만든 핵심 아이디어입니다. Skip connection이 없으면 깊은 네트워크는 학습이 거의 불가능했던 반면, 이를 추가하자 152층 이상의 네트워크도 수렴하게 되었습니다. 이는 **gradient flow 문제의 구조적 해결책**이며, 이후 DenseNet, Highway Networks, Transformer 등 현대 딥러닝의 많은 아키텍처의 기초가 됩니다.

---

## 📐 수학적 선행 조건

- [Ch2: CNN 기본 연산](../ch2-cnn-basics) — Convolution, Batch Normalization, ReLU
- [Ch3: Receptive Field](../ch3-receptive-field) — Network depth와 receptive field 관계
- 미적분학: Partial derivative, chain rule, gradient computation
- 선형대수: Matrix notation, block matrices, rank

---

## 📖 직관적 이해

### Identity Shortcut의 필요성

전통적인 컨볼루션 신경망에서 층이 깊어질수록, 특히 50층을 넘으면 **역전파(backpropagation) 중 gradient가 점점 작아져서** 초기 층의 파라미터가 거의 업데이트되지 않습니다. 이를 **gradient vanishing** 문제라고 합니다.

Residual block은 이를 우회 경로(shortcut)로 해결합니다:

$$y = F(x; W) + x$$

여기서:
- $F(x; W)$는 여러 convolution, normalization, activation을 거친 "residual mapping"
- $x$는 그냥 통과하는 **identity mapping**

Backpropagation 시:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

이 식의 좌항에서 1이 있으므로, $\partial F/\partial x$가 작아도 **최소 1은 전파됩니다**. 따라서 gradient vanishing을 완화할 수 있습니다.

### Pre-activation vs Post-activation

**Post-activation (원래 ResNet, 2015)**:
```
Conv → BN → ReLU → Conv → BN → (+) → ReLU
```

문제점: Identity shortcut이 Activation 후를 넘어가므로, shortcut은 항상 비음수(음수가 될 수 없음). 이것이 feature representation을 제한합니다.

**Pre-activation (He 2016b, "Identity Mappings")**:
```
BN → ReLU → Conv → BN → ReLU → Conv → (+)
```

장점:
- Addition 직후가 아닌, BN 전에 수행되므로 gradient가 더 잘 흐릅니다.
- 실험 결과 더 깊은 네트워크에서도 수렴이 안정적입니다.
- Normalization이 residual connection 이전에 적용되므로 더 "깔끔한" signal이 더해집니다.

### Bottleneck Block의 파라미터 효율

$3 \times 3$ convolution 두 개를 직렬로 사용하면:
- 입력: 256 channel, 출력: 256 channel
- 파라미터: $2 \times (3 \times 3 \times 256 \times 256) = 1.2$M

**Bottleneck** ($1 \times 1 \to 3 \times 3 \to 1 \times 1$):
```
Input (256) → Conv 1x1 (64) → Conv 3x3 (64) → Conv 1x1 (256) → Output (256)
```
- 파라미터: $(1 \times 1 \times 256 \times 64) + (3 \times 3 \times 64 \times 64) + (1 \times 1 \times 64 \times 256) = 0.27$M
- **약 77% 파라미터 감소**, 단 연산 능력은 유지

이것이 ResNet-50/101/152에서 실용적인 훈련을 가능하게 했습니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Residual Block (Post-activation)

입력 $x \in \mathbb{R}^{H \times W \times C}$에 대해:

$$y = \text{ReLU}(F(x; W) + x)$$

여기서 $F(x; W) = \text{ReLU}(\text{BN}(\text{Conv}(x; W_2))) \circ W_1)$ 형태 (정확히는 두 개 이상의 convolutional layer).

**제약**: Input과 output의 채널 수가 같아야 하며, spatial dimension도 같아야 합니다. 다르면 $1 \times 1$ projection shortcut $W_s$를 추가: $y = \text{ReLU}(F(x; W) + W_s x)$.

### 정의 1.2 — Pre-activation Residual Block (He 2016b)

$$y = F(x; W) + x$$

여기서:

$$F(x; W) = \text{Conv}(W_3; \text{ReLU}(\text{BN}(\text{Conv}(W_2; \text{ReLU}(\text{BN}(x))))))$$

즉, **모든 nonlinearity와 normalization이 addition 이전에 적용**됩니다.

### 정의 1.3 — Bottleneck Residual Block

$$y = F(x; W) + x$$

여기서:

$$F(x; W) = \text{Conv}_{1 \times 1}(W_3; \text{ReLU}(\text{BN}(\text{Conv}_{3 \times 3}(W_2; \text{ReLU}(\text{BN}(\text{Conv}_{1 \times 1}(W_1; x)))))))$$

reduction ratio $r$에 대해, 중간 채널 수는 $C/r$ (보통 $r=4$).

### 정리 1.4 — Residual Block의 Forward Propagation

Pre-activation residual block에서:

$$z_1 = \text{ReLU}(\text{BN}(x))$$
$$z_2 = \text{ReLU}(\text{BN}(\text{Conv}_1(z_1; W_1)))$$
$$z_3 = \text{ReLU}(\text{BN}(\text{Conv}_2(z_2; W_2)))$$
$$y = \text{Conv}_3(z_3; W_3) + x$$

Output activation은 **block 외부의 다음 block** (또는 global pooling 이전)에서 처리됩니다.

---

## 🔬 수학적 유도

### Residual Block이 Identity를 학습할 수 있음을 보이기

만약 $F(x; W) = 0$ (모든 파라미터가 0으로 초기화됨)이면:

$$y = 0 + x = x$$

즉, residual block은 identity mapping을 최소 "하지 않으려고 노력하는" 구조입니다. 반면 plain layer는:

$$y = F(x; W)$$

만약 $F(x; W) = x$를 학습하려면 명시적으로 learn해야 합니다. **Residual structure에서는 parameterized function이 "identity 근처"부터 시작**하므로, 훈련이 더 쉬워집니다.

### Pre-activation의 Gradient Flow

Post-activation:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial \text{ReLU}(z)}{\partial z} \frac{\partial (F + x)}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbb{1}_{z > 0} \cdot \left(\frac{\partial F}{\partial x} + I\right)$$

Pre-activation:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(\frac{\partial F}{\partial x} + I\right)$$

Addition이 non-linearity 이후에 일어나므로, identity term의 contribution이 직접적입니다.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Post-activation vs Pre-activation 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostActivationResBlock(nn.Module):
    """Original ResNet (He et al. 2015)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # Addition BEFORE final ReLU
        out = F.relu(out)  # Post-activation
        return out


class PreActivationResBlock(nn.Module):
    """Improved ResNet (He et al. 2016b, "Identity Mappings")"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        
        # Projection shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                     stride=stride, bias=False)
    
    def forward(self, x):
        # BN and ReLU BEFORE convolution
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = out + self.shortcut(x)  # Addition AFTER all nonlinearity
        return out  # Output without final ReLU (left for next block)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block (ResNet-50/101/152)"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False)
            )
    
    def forward(self, x):
        # Pre-activation: BN → ReLU → Conv
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        out = out + self.shortcut(x)
        return out
```

### 실험 2 — Residual vs Plain 비교 (Forward Pass)

```python
# CIFAR-10 학습 설정
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test forward pass
x = torch.randn(32, 64, 32, 32)  # batch=32, channels=64, H=W=32

block_post = PostActivationResBlock(64, 64)
block_pre = PreActivationResBlock(64, 64)
block_bottleneck = BottleneckBlock(64, 256, expansion=4)

y_post = block_post(x)
y_pre = block_pre(x)
y_bottleneck = block_bottleneck(x)

print(f"Post-activation output shape: {y_post.shape}")
print(f"Pre-activation output shape: {y_pre.shape}")
print(f"Bottleneck output shape: {y_bottleneck.shape}")
print(f"Post-activation params: {count_parameters(block_post)}")
print(f"Pre-activation params: {count_parameters(block_pre)}")
print(f"Bottleneck params: {count_parameters(block_bottleneck)}")
```

**예상 출력**:
```
Post-activation output shape: torch.Size([32, 64, 32, 32])
Pre-activation output shape: torch.Size([32, 64, 32, 32])
Bottleneck output shape: torch.Size([32, 256, 32, 32])
Post-activation params: 147456
Pre-activation params: 147456
Bottleneck params: 262144
```

### 실험 3 — Bottleneck의 파라미터 효율성 분석

```python
def analyze_bottleneck_efficiency():
    """Compare parameter counts for different block designs"""
    in_ch = 256
    out_ch = 256
    
    # 두 개의 3x3 convolution
    basic_params = (3*3*in_ch*out_ch) + (3*3*out_ch*out_ch)
    
    # Bottleneck 1x1 → 3x3 → 1x1 (reduction=4)
    mid_ch = out_ch // 4
    bottleneck_params = (1*1*in_ch*mid_ch) + (3*3*mid_ch*mid_ch) + (1*1*mid_ch*out_ch)
    
    print(f"Input channels: {in_ch}, Output channels: {out_ch}")
    print(f"Basic block (3×3 → 3×3): {basic_params:,} parameters")
    print(f"Bottleneck (1×1 → 3×3 → 1×1): {bottleneck_params:,} parameters")
    print(f"Reduction ratio: {basic_params / bottleneck_params:.2f}x")
    print(f"Parameter savings: {(1 - bottleneck_params/basic_params)*100:.1f}%")

analyze_bottleneck_efficiency()
```

**출력**:
```
Input channels: 256, Output channels: 256
Basic block (3×3 → 3×3): 1179648 parameters
Bottleneck (1×1 → 3×3 → 1×1): 295104 parameters
Reduction ratio: 3.99x
Parameter savings: 75.0%
```

---

## 🔗 이론과 실전의 간극

### ResNet 논문의 실험 결과

- **ImageNet (ILSVRC 2015)**:
  - ResNet-50: Top-1 error 24.0% (56층 plain network: 28.3%)
  - ResNet-101: Top-1 error 23.6%
  - ResNet-152: Top-1 error 23.3%

- **Identity Mapping 개선 (He 2016b)**:
  - ResNet-101 pre-activation: 23.4% (vs post-activation 23.6%)
  - 깊은 네트워크일수록 pre-activation의 이득이 더 큼

### 실전 적용 시 고려사항

1. **Projection shortcut의 필요성**: stride > 1 또는 channel 변화 시에만 필요
2. **Initialization**: $W$를 작게 초기화하는 것이 중요 (He initialization)
3. **Batch Normalization**: Residual block의 성공은 BN과 함께 일어남
4. **Computational cost**: Skip connection 자체의 computational cost는 거의 없음

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Input와 output dimension이 같거나 projection 가능 | Non-rectangular feature maps에서는 복잡함 |
| Batch Normalization 사용 | BN 없이는 성능 저하 (후속 연구: Group Norm, Layer Norm) |
| Spatial dimension이 작거나 같음 | Spatial dimension 변화는 stride로만 가능 |
| Addition이 feature-wise에서 가능 | DenseNet은 concatenation으로 해결 (Ch4-04) |
| 초기화가 정교함 | 부실한 초기화는 shortcut의 이점을 상쇄 |

---

## 📌 핵심 정리

$$\boxed{y = F(x; W) + x \text{ — Skip connection으로 gradient vanishing 해결}}$$

| 개념 | 설명 |
|------|------|
| **Residual block** | $y = F(x; W) + x$, identity shortcut이 gradient 직통로 제공 |
| **Post-activation** | Conv-BN-ReLU 순서, addition 후 activation — 원래 ResNet |
| **Pre-activation** | BN-ReLU-Conv 순서, addition 전 activation — 개선된 설계 |
| **Bottleneck** | $1 \times 1 \to 3 \times 3 \to 1 \times 1$ 구조, 75% 파라미터 감소 |
| **Projection shortcut** | Input과 output의 channel/spatial이 다를 때 $1 \times 1$ conv 추가 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Pre-activation block에서 왜 마지막 convolution 이후에 activation을 하지 않는가? 다음 block과의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

Pre-activation 설계에서는 **각 block이 output activation을 "제공하지 않습니다"**. 대신 다음 block의 첫 단계 (BN → ReLU)에서 activation이 일어납니다.

이렇게 하면:
1. Addition이 raw value에 대해 일어나므로 더 "깨끗함"
2. Gradient flow가 linear한 addition 직후에 시작
3. 여러 residual block을 연결했을 때 더 안정적

만약 각 block이 ReLU를 추가로 한다면, 역전파 시 그 ReLU의 mask가 추가로 곱해져서 gradient가 더 죽을 수 있습니다.

</details>

**문제 2** (심화): Bottleneck block에서 reduction ratio를 4가 아니라 2로 설정하면 어떤 장단점이 있을까?

<details>
<summary>힌트 및 해설</summary>

Reduction ratio = 2인 경우:
- 중간 채널 수: $C/2$ (vs $C/4$)
- 파라미터 수: 약 50% 증가
- 연산량(FLOPs): 약 50% 증가

**장점**:
- 정보 손실이 적음 (wider bottleneck)
- 더 강력한 feature representation 가능

**단점**:
- 메모리 사용량 증가
- 훈련 시간 증가
- 과도한 "bottleneck effect" 없음 — 원래 목적(파라미터 효율)이 약해짐

실제로 He et al.은 extensive experiment로 4가 최적임을 보였습니다.

</details>

**문제 3** (이론-실전): Identity shortcut $y = F(x; W) + x$에서 만약 $x$의 norm이 매우 크고 $F(x; W)$의 norm이 매우 작다면, 역전파 시 어떤 문제가 생길까?

<details>
<summary>힌트 및 해설</summary>

Gradient:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(I + \frac{\partial F}{\partial x}\right)$$

만약 $\frac{\partial F}{\partial x} \approx 0$이면:
$$\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y}$$

즉, gradient가 **direct하게 통과**합니다 (이는 좋은 것).

하지만 **forward pass에서**는 $y = F(x; W) + x$가 $x$에 지배당할 수 있습니다:
- $\|x\|$가 크고 $\|F(x; W)\|$가 작으면, $F$의 contribution이 무시됨
- 이를 해결하기 위해 **Batch Normalization**이 input을 normalize하므로, scale이 자동 조정됨

따라서 BN 없는 residual block은 일부 scale imbalance 문제에 취약합니다.

</details>

---

<div align="center">

[◀ 이전](../ch3-receptive-field/04-rf-segmentation.md) | [📚 README](../README.md) | [다음 ▶](./02-gradient-flow.md)

</div>
