# 03. EfficientNet과 Compound Scaling

## 🎯 핵심 질문

- CNN을 **확대(scale)** 할 때, depth, width, resolution 중 어느 것을 늘려야 정확도가 최대화되는가?
- **Compound scaling** 제약 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ 는 어디서 나왔는가?
- EfficientNet-B0부터 B7까지 스펙트럼을 어떻게 설계했는가?
- **FLOPs-정확도 Pareto frontier**를 실제로 달성할 수 있는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Tan & Le (2019)의 EfficientNet은 **일반적 진리**를 실험적으로 증명했습니다:

1. **Scaling의 체계화**: 임의로 layer 늘리기 (VGG 방식) 또는 채널 늘리기 (ResNet 방식) 대신, 원리적으로 설계
2. **Compound Scaling Law**: 세 가지 차원 (depth, width, resolution)을 균형있게 증가
3. **효율성의 극대화**: 같은 FLOPs에서 이전 모델들보다 4-8.4% 더 높은 정확도
4. **실용성**: B0 (5.3M params) ~ B7 (66M params) 다양한 크기로 배포

이는 **AutoML의 성공 사례**이자, **신경망 설계의 원리**를 드러냅니다.

---

## 📐 수학적 선행 조건

- **FLOPs 계산**: $N = H \times W \times K^2 \times C_{\text{in}} \times C_{\text{out}}$ (depth, width, resolution의 함수)
- **지수 스케일링 법칙**: $y = ax^b$ 형태의 power-law
- **Constraint optimization**: Lagrange 승수법
- **Neural Architecture Search (NAS)** 기초

참고: [02-inception.md](./02-inception.md)

---

## 📖 직관적 이해

### Scaling 문제의 정의

기본 CNN 아키텍처가 주어졌을 때, 더 큰 모델을 만들려면:

**Depth scaling**: layer 개수 $d$ 증가
- Effect: receptive field 증가, 더 복잡한 특징 학습
- Cost: 선형 증가 (layer 2배 → FLOPs 2배)

**Width scaling**: channel 수 $w$ 증가
- Effect: 더 많은 특징 맵, 더 세밀한 표현
- Cost: 제곱 증가 (channel 2배 → FLOPs 4배, 왜냐하면 $\text{FLOPs} \propto w^2$)

**Resolution scaling**: 입력 이미지 해상도 $r$ 증가
- Effect: 세밀한 세부 정보 캡처
- Cost: 제곱 증가 (해상도 2배 → FLOPs 4배)

**문제**: 세 가지를 어떻게 조합하면 최선인가?

### Compound Scaling의 직관

실험적 관찰:
- Depth만 증가: receptive field 증가하지만 채널 수 부족 → low accuracy
- Width만 증가: 채널 풍부하지만 receptive field 제한 → accuracy ceiling
- Resolution만 증가: 세부 정보 포착하지만 큰 구조 이해 불가

**해결**: 세 가지를 **균형있게** 증가

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi$$

여기서 $\phi$는 scaling coefficient, $\alpha, \beta, \gamma$는 기본 아키텍처에 대해 grid search로 결정된 상수.

### FLOPs 제약 조건

FLOPs는 대략:
$$\text{FLOPs} \propto d \times w^2 \times r^2$$

(depth는 선형, width와 resolution은 제곱)

일정 FLOPs 예산 $F_{\text{target}}$ 하에서:
$$d \times w^2 \times r^2 \approx \text{const}$$

따라서:
$$\alpha^\phi \times (\beta^\phi)^2 \times (\gamma^\phi)^2 \approx \text{const}$$
$$\alpha \times \beta^2 \times \gamma^2 \approx 2 \text{ (정규화)}$$

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Compound Scaling

기본 네트워크 $N(d, w, r)$에 대해, compound scaling은:
$$\hat{N} = N(\alpha^\phi \cdot d, \beta^\phi \cdot w, \gamma^\phi \cdot r)$$

여기서:
- $\alpha, \beta, \gamma \geq 1$: 상수 (grid search로 결정)
- $\phi \geq 1$: scaling parameter (사용자가 지정)
- 최적 값: $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$ (EfficientNet-B0에서 grid search)

### 정리 3.2 — FLOPs Scaling 관계

CNN의 FLOPs는:
$$\text{FLOPs}(d, w, r) = H \times W \times r^2 \times K^2 \times d \times (w \cdot C_0)^2$$

여기서 $H, W$는 기본 해상도, $C_0$는 기본 채널.

따라서:
$$\text{FLOPs} \propto d \times w^2 \times r^2$$

### 정의 3.3 — EfficientNet-B0 (기본 모델)

| Stage | Resolution | Channels | Repeats |
|-------|-----------|----------|---------|
| 1     | 224×224   | 32       | 1       |
| 2     | 112×112   | 16       | 2       |
| 3     | 56×56     | 24       | 2       |
| 4     | 28×28     | 40       | 3       |
| 5     | 14×14     | 80       | 3       |
| 6     | 7×7       | 112      | 4       |
| 7     | 7×7       | 192      | 1       |
| 8     | 1×1       | 1280     | 1       |

각 블록: **MBConv** (Mobile Inverted Bottleneck) + SE (Squeeze-Excitation)

### 정리 3.4 — Compound Scaling Grid Search

입력: $\phi \in \{1, 2, \ldots, 7\}$ (EfficientNet-B0 ~ B7)

각 $\phi$에 대해:
$$d_\phi = 1.2^\phi, \quad w_\phi = 1.1^\phi, \quad r_\phi = 224 + 30^\phi$$

결과:

| Model | $\phi$ | Depth | Width | Resolution | Params (M) | FLOPs (B) | Acc |
|-------|--------|-------|-------|-----------|-----------|----------|-----|
| B0    | 1      | 1.0   | 1.0   | 224       | 5.3       | 0.39     | 77.1 |
| B1    | 1.5    | 1.3   | 1.1   | 240       | 7.8       | 0.71     | 79.8 |
| B2    | 2      | 1.6   | 1.2   | 260       | 9.1       | 1.03     | 80.6 |
| B3    | 3      | 1.8   | 1.2   | 300       | 12        | 1.86     | 82.2 |
| B4    | 4      | 2.2   | 1.4   | 380       | 19        | 4.3      | 83.4 |
| B5    | 5      | 2.6   | 1.6   | 456       | 30        | 9.9      | 84.3 |
| B6    | 6      | 3.1   | 1.8   | 528       | 43        | 19       | 84.8 |
| B7    | 7      | 3.2   | 2.0   | 600       | 66        | 37       | 85.3 |

---

## 🔬 수학적 유도

### FLOPs의 Scaling 법칙 유도

입력 해상도 $H_r \times W_r = (r/224) \times H_0 \times (r/224) \times W_0$일 때:

$$\text{FLOPs}(d, w, r) = \sum_{i=1}^L K_i^2 \times (r/224)^2 H_0 W_0 \times (d \cdot w \cdot C_i)^2$$

approximation하면:
$$\text{FLOPs} \propto d \times w^2 \times r^2$$

따라서:
$$\log(\text{FLOPs}) = \log(k) + \log(d) + 2\log(w) + 2\log(r)$$

### Compound Scaling 최적화

고정 FLOPs $F$ 예산 하에서, **정확도** $\text{Acc}(d, w, r)$를 최대화:

$$\max_{d, w, r} \text{Acc}(d, w, r) \quad \text{s.t.} \quad d \times w^2 \times r^2 \leq F$$

Lagrange 승수법:
$$\nabla \text{Acc} = \lambda \nabla (d \times w^2 \times r^2)$$

실제로는 **grid search**로 상수 $\alpha, \beta, \gamma$ 결정:
- 작은 $\phi$ (e.g., $\phi=1$)에서 여러 조합 시도
- 각 (depth 배수, width 배수, resolution)에 대해 정확도 측정
- Pareto frontier 추출
- 최적 비율 $\alpha, \beta, \gamma$ 결정

---

## 💻 실험 재현: EfficientNet 구현

### MBConv 블록 (Mobile Inverted Bottleneck)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-Excitation (SE) block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1)
    
    def forward(self, x):
        # Squeeze: global average pooling
        se = F.adaptive_avg_pool2d(x, 1)
        # Excitation
        se = self.fc1(se)
        se = F.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, expand_ratio=6, drop_rate=0.0):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expand_ratio
        
        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                                 stride=stride, padding=kernel_size//2, 
                                 groups=mid_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_channels)
        
        # Squeeze-Excitation
        self.se = SEBlock(mid_channels)
        
        # Projection phase
        self.proj_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        # Expansion
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = F.relu(out)
        
        # Depthwise
        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = F.relu(out)
        
        # SE
        out = self.se(out)
        
        # Projection
        out = self.proj_conv(out)
        out = self.proj_bn(out)
        
        # Skip
        if self.skip:
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
            out = out + x
        
        return out

# 테스트
mbconv = MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
y = mbconv(x)
print(f"Input: {x.shape}, Output: {y.shape}")
print(f"Parameters: {sum(p.numel() for p in mbconv.parameters()):,}")
```

### EfficientNet 기본 구조

```python
class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, 
                 resolution=224, num_classes=1000, drop_rate=0.2):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32 * width_mult), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks (simplified)
        channels_config = [16, 24, 40, 80, 112, 192, 320, 1280]
        repeats_config = [1, 2, 2, 3, 3, 4, 1, 1]
        
        self.blocks = nn.ModuleList()
        
        in_channels = int(32 * width_mult)
        for stage, (out_ch, num_repeats) in enumerate(zip(channels_config, repeats_config)):
            out_channels = int(out_ch * width_mult)
            num_repeats = int(num_repeats * depth_mult)
            
            # First block: stride 2 (except stem)
            stride = 2 if stage > 0 else 1
            
            for i in range(num_repeats):
                block = MBConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    expand_ratio=6,
                    drop_rate=drop_rate
                )
                self.blocks.append(block)
            
            in_channels = out_channels
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.head[:-1](x)  # Conv + BN + ReLU + AvgPool + Dropout
        x = torch.flatten(x, 1)
        x = self.head[-1](x)  # Linear
        
        return x

# EfficientNet-B0
efficientnet_b0 = EfficientNet(width_mult=1.0, depth_mult=1.0, resolution=224)
print(f"EfficientNet-B0 params: {sum(p.numel() for p in efficientnet_b0.parameters()) / 1e6:.2f}M")

# EfficientNet-B3
efficientnet_b3 = EfficientNet(width_mult=1.2, depth_mult=1.8, resolution=300)
print(f"EfficientNet-B3 params: {sum(p.numel() for p in efficientnet_b3.parameters()) / 1e6:.2f}M")
```

### Compound Scaling 시뮬레이션

```python
import matplotlib.pyplot as plt
import numpy as np

# EfficientNet 스펙
phi_values = np.array([0, 1, 2, 3, 4, 5, 6, 7])
alpha, beta, gamma = 1.2, 1.1, 1.15

depth_mult = alpha ** phi_values
width_mult = beta ** phi_values
resolution = 224 + 30 * phi_values

# FLOPs 추정 (상대적)
flops_relative = depth_mult * (width_mult ** 2) * (resolution / 224) ** 2

# 정확도 (실제 값)
accuracies = np.array([77.1, 79.8, 80.6, 82.2, 83.4, 84.3, 84.8, 85.3])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scaling factors
ax1.plot(phi_values, depth_mult, 'o-', label='Depth ($\\alpha^\\phi$)', markersize=6)
ax1.plot(phi_values, width_mult, 's-', label='Width ($\\beta^\\phi$)', markersize=6)
ax1.plot(phi_values, resolution / 224, '^-', label='Resolution ($\\gamma^\\phi$)', markersize=6)
ax1.set_xlabel('Scaling Coefficient $\\phi$')
ax1.set_ylabel('Multiplier')
ax1.set_title('Compound Scaling Factors')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: FLOPs vs Accuracy
ax2.plot(flops_relative, accuracies, 'o-', markersize=8, color='green', linewidth=2)
for i, phi in enumerate(phi_values):
    ax2.annotate(f'B{int(phi)}', (flops_relative[i], accuracies[i]), 
                 fontsize=9, ha='center', va='bottom')
ax2.set_xlabel('FLOPs (relative to B0)')
ax2.set_ylabel('ImageNet Top-1 Accuracy (%)')
ax2.set_title('EfficientNet: Pareto Frontier')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**출력**: 세 scaling factor가 조화롭게 증가하며, FLOPs 증가에 따라 정확도가 단조 증가

### Timm을 사용한 실제 사용

```python
import timm

# 사전학습된 EfficientNet 로드
model_b0 = timm.create_model('efficientnet_b0', pretrained=True)
model_b7 = timm.create_model('efficientnet_b7', pretrained=True)

# 파라미터 수
params_b0 = sum(p.numel() for p in model_b0.parameters())
params_b7 = sum(p.numel() for p in model_b7.parameters())

print(f"EfficientNet-B0: {params_b0 / 1e6:.2f}M params")
print(f"EfficientNet-B7: {params_b7 / 1e6:.2f}M params")
print(f"Ratio: {params_b7 / params_b0:.1f}x")

# 이미지 분류
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 추론
img = Image.new('RGB', (224, 224), color='red')
x = transform(img).unsqueeze(0)

with torch.no_grad():
    y_b0 = model_b0(x)
    y_b7 = model_b7(x)

print(f"\nB0 logits shape: {y_b0.shape}")
print(f"B7 logits shape: {y_b7.shape}")
```

---

## 🔗 이론과 실전의 간극

### 1. Grid Search vs NAS

**이론**: Compound scaling의 최적 비율 $(\alpha, \beta, \gamma)$ 계산

**실제**: Grid search (수작업) vs Neural Architecture Search (자동화)
- EfficientNet: 작은 기본 모델에서 grid search로 $(1.2, 1.1, 1.15)$ 결정
- EfficientNet v2: NAS로 새로운 기본 모델 + 다시 grid search

### 2. 이론적 FLOPs vs 실제 처리 시간

**이론**: FLOPs $\propto d \times w^2 \times r^2$ (완벽한 선형성 가정)

**실제**:
- 메모리 접근 패턴 (cache 효율)
- 하드웨어 병렬화 (GPU의 warp 크기)
- 배치 크기에 따른 비효율

→ B3 (12M params) vs B7 (66M params): FLOPs는 19배 차이지만, 실제 추론 시간은 8-10배

### 3. 정확도의 diminishing return

```
B0: +2.7% (77.1 → 79.8)
B1: +0.8% (79.8 → 80.6)
B2: +1.6% (80.6 → 82.2)
B3: +1.2% (82.2 → 83.4)
...
B6: +0.5% (84.8 → 85.3)
```

→ 깊은 스케일링일수록 정확도 향상이 감소

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 모든 이미지에 최적 | 특정 도메인 (의료, 위성)은 재조정 필요 |
| FLOPs 관점의 최적성 | 메모리, 레이턴시는 FLOPs와 비례하지 않음 |
| ImageNet 기준 | 다른 데이터셋에서는 최적 비율이 다를 수 있음 |
| 정규화 $\alpha \beta^2 \gamma^2 = 2$ | 실제는 근사값, 모든 경우에 적용 안 됨 |
| MBConv 블록 고정 | 다른 블록 타입은 다른 scaling factor 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{EfficientNet: } d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi \quad \text{with} \quad \alpha \beta^2 \gamma^2 \approx 2}$$

| 개념 | 정의 |
|------|------|
| **Compound Scaling** | 세 차원 (depth, width, resolution) 균형 조정 |
| **Scaling Exponents** | $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$ (EfficientNet-B) |
| **FLOPs 법칙** | $\text{FLOPs} \propto d \times w^2 \times r^2$ |
| **MBConv** | 역 병목(inverted bottleneck) + SE block |
| **Pareto Frontier** | B0~B7에서 계속 증가하는 정확도 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): EfficientNet-B0의 scaling factor가 (1.0, 1.0, 1.0)일 때, B2의 scaling factor $(\alpha^2, \beta^2, \gamma^2)$를 계산하라. ($\alpha=1.2, \beta=1.1, \gamma=1.15$)

<details>
<summary>힌트 및 해설</summary>

$$d_2 = 1.2^2 = 1.44$$
$$w_2 = 1.1^2 = 1.21$$
$$r_2 = 1.15^2 = 1.3225$$

따라서 B2는:
- Depth: 44% 증가
- Width: 21% 증가
- Resolution: 32% 증가 ✓

</details>

**문제 2** (심화): EfficientNet-B1과 B3의 FLOPs 비율을 계산하라. ($\phi_1=1.5, \phi_3=3$)

<details>
<summary>힌트 및 해설</summary>

$$\text{FLOPs} \propto d \times w^2 \times r^2 = \alpha^\phi \times \beta^{2\phi} \times \gamma^{2\phi}$$

$$= \alpha^\phi \beta^{2\phi} \gamma^{2\phi} = (\alpha \beta^2 \gamma^2)^\phi \approx 2^\phi$$

따라서:
$$\frac{\text{FLOPs}_3}{\text{FLOPs}_1} \approx 2^3 / 2^{1.5} = 8 / 2.83 \approx 2.83$$

실제 FLOPs (문제에서): 1.86 / 0.71 ≈ 2.62

(근사이지만 비슷한 수준) ✓

</details>

**문제 3** (이론-실전): Compound scaling이 모든 CNN 아키텍처 (VGG, ResNet, Inception)에 동일하게 적용되지 않는 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

각 아키텍처의 특성 차이:

1. **VGG**: 스택 구조 → depth 증가 효율 낮음
   - 깊이의 표현력 이득 < 최적화 비용

2. **ResNet**: Skip connection → depth 증가 효율 높음
   - depth 2배는 가능

3. **Inception**: Multi-scale feature → width 증가 시 redundancy
   - width 증가의 효율 낮음

4. **EfficientNet**: MBConv + SE → 모든 차원 효율 좋음
   - $(\alpha=1.2, \beta=1.1, \gamma=1.15)$ 균형

→ **아키텍처가 다르면 scaling factor도 다름**!

EfficientNet v2는 다른 기본 모델에서 새로 grid search → 다른 $\alpha', \beta', \gamma'$

</details>

---

<div align="center">

[◀ 이전](./02-inception.md) | [📚 README](../README.md) | [다음 ▶](./04-convnext.md)

</div>
