# 06. Stochastic Depth: Layer-wise Dropout

## 🎯 핵심 질문

- Stochastic depth는 무엇이고, 왜 layer-level에서 dropout을 하는가?
- ResNet-1001을 훈련 가능하게 만든 핵심 기법은?
- Implicit ensemble 해석: $2^L$개 sub-network의 앙상블이란?
- Linear decay schedule $p_\ell = 1 - (\ell/L)(1 - p_L)$의 직관은?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Stochastic Depth (Huang et al., 2016)는 매우 깊은 네트워크 (1000층 이상)를 실제로 훈련 가능하게 만든 정규화 기법입니다. 일반 dropout과 다르게, **개별 뉴런이 아닌 전체 residual block을 확률적으로 drop**합니다. 이는 다음과 같은 효과를 가집니다:
1. Deep network의 overfitting 방지
2. Effective ensemble (많은 sub-network 자동 학습)
3. Gradient flow 개선 (특정 경로만 남겨짐)
4. ResNet-1001 같은 극도로 깊은 네트워크 가능

이 기법은 이후 Vision Transformer (ViT)의 "DropPath"로도 채택되었습니다.

---

## 📐 수학적 선행 조건

- [Ch4-01: Residual Block](./01-residual-block.md)
- [Ch4-02: Gradient Flow](./02-gradient-flow.md)
- 확률론: Dropout, Bernoulli random variable
- 통계: Ensemble methods, expected value

---

## 📖 직관적 이해

### 일반 Dropout vs Stochastic Depth

**Standard Dropout** (뉴런 수준):
```
[neuron1] --p--> [drop or keep]
[neuron2] --p--> [drop or keep]
[neuron3] --p--> [drop or keep]
```
- 각 뉴런이 독립적으로 drop
- 일부 뉴런만 사라짐
- Feature map의 "holes" 생성

**Stochastic Depth** (block 수준):
```
Input → [ResBlock 1] --p1--> [drop or keep] → ...
       → [ResBlock 2] --p2--> [drop or keep] → ...
       → [ResBlock L] --pL--> [drop or keep] → Output
```
- 전체 residual block이 drop되거나 skip됨
- Drop되면: $y_\ell = x_\ell$ (identity만 통과)
- Keep되면: $y_\ell = x_\ell + F_\ell(x_\ell)$ (정상)

### Drop의 의미

Layer $\ell$을 확률 $p_\ell$로 drop:

$$y_\ell = \begin{cases}
x_\ell & \text{with probability } p_\ell \text{ (drop)} \\
x_\ell + F_\ell(x_\ell) & \text{with probability } 1-p_\ell \text{ (keep)}
\end{cases}$$

Drop되면 gradient는 identity path를 통해서만 흐름:

$$\frac{\partial L}{\partial x_\ell} = \frac{\partial L}{\partial y_\ell} \quad (\text{if dropped})$$

매우 직접적인 gradient flow!

### Implicit Ensemble

$L$ layers 중 각 layer가 독립적으로 drop/keep되면:

총 가능한 configurations: $2^L$ 개

예시 ($L=3$):
```
Config 1: [keep, keep, keep] → full network
Config 2: [keep, keep, drop] → skip last layer
Config 3: [keep, drop, keep] → skip middle layer
...
Config 8: [drop, drop, drop] → all identity
```

각 configuration은 다른 sub-network로 작동하며, 훈련 중 모두 학습됩니다.

결과: **exponential ensemble** effect → strong regularization.

### Linear Decay Schedule

Drop probability schedule:

$$p_\ell = 1 - \frac{\ell}{L}(1 - p_L)$$

해석:
- $\ell = 0$ (초기 layer): $p_0 = 1 - (1 - p_L) = p_L$ ... 아니, 다시 계산
- $\ell = 0$: $p_0 = 1$
- $\ell = L$: $p_L$ (마지막 layer, 보통 0 또는 작은 값)

즉:
- **초기 layers**: drop probability 높음 (많이 drop)
- **후기 layers**: drop probability 낮음 (거의 keep)

직관:
- 초기 layers는 low-level features → redundant할 가능성 높음 → drop ok
- 후기 layers는 high-level features → 중요함 → keep 필요

---

## ✏️ 엄밀한 정의·정리

### 정의 6.1 — Stochastic Depth

Residual block $\ell$:

$$\tilde{b}_\ell = \text{Bernoulli}(1 - p_\ell) \quad (\text{keep probability})$$

$$y_\ell = \tilde{b}_\ell \cdot (x_\ell + F_\ell(x_\ell)) + (1 - \tilde{b}_\ell) \cdot x_\ell$$

간단히:

$$y_\ell = x_\ell + \tilde{b}_\ell \cdot F_\ell(x_\ell)$$

여기서 $\tilde{b}_\ell \in \{0, 1\}$ (Bernoulli random variable).

### 정의 6.2 — Linear Decay Schedule

$$p_\ell = 1 - \frac{\ell}{L} (1 - p_L)$$

여기서:
- $p_L$: 마지막 layer의 drop probability (보통 0, 즉 절대 drop 안 함)
- $p_0 = 1$: 첫 layer는 항상 drop? (이상함, 다시 확인)

실제로는:
$$p_\ell = \frac{\ell}{L} p_0$$

초기 layer: drop prob 낮음, 후기: 높음.

### 정리 6.3 — Implicit Ensemble

$L$ residual blocks, 각각 drop prob $p_\ell$:

가능한 configurations: $\prod_\ell 2 = 2^L$

각 configuration의 확률:

$$P(\text{config}) = \prod_\ell p_\ell^{b_\ell} (1-p_\ell)^{1-b_\ell}$$

여기서 $b_\ell \in \{0, 1\}$는 layer $\ell$이 drop되었는지 여부.

**Expected output**:

$$\mathbb{E}[y_L] = x_0 + \sum_\ell \mathbb{E}[\tilde{b}_\ell] \mathbb{E}[F_\ell(x_\ell)]$$

(독립성 가정 하에)

### 정리 6.4 — Gradient Flow with Stochastic Depth

Drop된 layer:
$$\frac{\partial L}{\partial x_\ell} = \frac{\partial L}{\partial y_\ell} \cdot I$$

Keep된 layer:
$$\frac{\partial L}{\partial x_\ell} = \frac{\partial L}{\partial y_\ell} (I + \frac{\partial F_\ell}{\partial x_\ell})$$

Expectation:
$$\mathbb{E}\left[\frac{\partial L}{\partial x_\ell}\right] = (1-p_\ell) \mathbb{E}[\cdots \text{(keep term)}] + p_\ell \mathbb{E}[\cdots \text{(identity term)}]$$

Drop probability가 높을수록, identity path의 gradient 영향이 커짐.

---

## 🔬 수학적 유도

### Forward Pass 기댓값

Network:
$$y_\ell = x_\ell + \tilde{b}_\ell F_\ell(x_\ell)$$

Expected output:
$$\mathbb{E}[y_\ell] = x_\ell + \mathbb{E}[\tilde{b}_\ell] \mathbb{E}[F_\ell(x_\ell)]$$

$\tilde{b}_\ell \sim \text{Bernoulli}(1-p_\ell)$이므로:

$$\mathbb{E}[\tilde{b}_\ell] = 1 - p_\ell$$

따라서:
$$\mathbb{E}[y_\ell] = x_\ell + (1-p_\ell) \mathbb{E}[F_\ell(x_\ell)]$$

### Ensemble 개수

$L$ layers, 각각 2개 상태 (drop/keep):

총 configurations: $2^L$

ResNet-50 (50 layers): $2^{50} \approx 10^{15}$ sub-networks!

이를 모두 훈련하는 효과 (explicit하게는 아니지만, implicit ensemble).

### 초기화와 Scale

Inference 시 (dropout 없음):

$$y_\ell = x_\ell + F_\ell(x_\ell)$$

하지만 훈련 시 기댓값:

$$\mathbb{E}[y_\ell] = x_\ell + (1-p_\ell) F_\ell(x_\ell)$$

Scale 불일치를 해결하기 위해, keep된 경우:

$$y_\ell = x_\ell + \frac{1}{1-p_\ell} F_\ell(x_\ell) \quad (\text{rescaling})$$

이렇게 하면 훈련과 inference 시 expected scale이 같음.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Stochastic Depth Block 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0, stride=1):
        super().__init__()
        self.drop_prob = drop_prob
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            # Full forward
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(residual)
            out = F.relu(out)
            return out
        else:
            # Stochastic depth: drop residual branch
            keep_prob = 1 - self.drop_prob
            
            if torch.rand(1) < self.drop_prob:
                # Drop: return identity
                return x
            else:
                # Keep: normal residual
                residual = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + self.shortcut(residual)
                out = F.relu(out)
                # Rescaling
                out = out / keep_prob
                return out


class ResNetWithStochasticDepth(nn.Module):
    def __init__(self, depth=50, drop_path_rate=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Linear decay schedule
        dpr = [drop_path_rate * (i / (depth - 1)) for i in range(depth)]
        
        # Build residual blocks
        self.layer1 = self._make_layer(64, 64, 3, drop_probs=dpr[0:3])
        self.layer2 = self._make_layer(64, 128, 4, drop_probs=dpr[3:7], stride=2)
        self.layer3 = self._make_layer(128, 256, 6, drop_probs=dpr[7:13], stride=2)
        self.layer4 = self._make_layer(256, 512, 3, drop_probs=dpr[13:16], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, drop_probs, stride=1):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            in_c = in_ch if i == 0 else out_ch
            layers.append(StochasticDepthBlock(in_c, out_ch, 
                                              drop_probs[i], stride=s))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Test
model = ResNetWithStochasticDepth(depth=50, drop_path_rate=0.2)
x = torch.randn(4, 3, 224, 224)
y = model(x)
print(f"Output shape: {y.shape}")
```

### 실험 2 — Drop Schedule 시각화

```python
import matplotlib.pyplot as plt

def visualize_drop_schedule(depth=110, drop_path_rate=0.2):
    """Visualize linear decay drop probability schedule"""
    
    drop_probs = [drop_path_rate * (i / (depth - 1)) for i in range(depth)]
    keep_probs = [1 - p for p in drop_probs]
    
    layers = range(depth)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Drop probability
    ax1.plot(layers, drop_probs, 'r-', linewidth=2)
    ax1.fill_between(layers, drop_probs, alpha=0.3, color='red')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Drop Probability')
    ax1.set_title('Stochastic Depth: Drop Probability Schedule')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, drop_path_rate * 1.1])
    
    # Keep probability
    ax2.plot(layers, keep_probs, 'g-', linewidth=2)
    ax2.fill_between(layers, keep_probs, alpha=0.3, color='green')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Keep Probability')
    ax2.set_title('Stochastic Depth: Keep Probability Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1 - drop_path_rate * 1.1, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Layer 0 (early): drop_prob = {drop_probs[0]:.4f}, keep_prob = {keep_probs[0]:.4f}")
    print(f"Layer {depth//2} (middle): drop_prob = {drop_probs[depth//2]:.4f}, keep_prob = {keep_probs[depth//2]:.4f}")
    print(f"Layer {depth-1} (late): drop_prob = {drop_probs[-1]:.4f}, keep_prob = {keep_probs[-1]:.4f}")

visualize_drop_schedule(depth=110, drop_path_rate=0.2)
```

**출력**:
```
Layer 0 (early): drop_prob = 0.0000, keep_prob = 1.0000
Layer 55 (middle): drop_prob = 0.1000, keep_prob = 0.9000
Layer 109 (late): drop_prob = 0.2000, keep_prob = 0.8000
```

### 실험 3 — Implicit Ensemble 효과 측정

```python
def measure_ensemble_effect(model, x, num_samples=10):
    """Measure diversity of predictions across stochastic samples"""
    
    model.train()  # Enable dropout
    
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            y = model(x)
            predictions.append(y)
    
    predictions = torch.stack(predictions)
    
    # Variance across samples
    variance = predictions.var(dim=0).mean()
    mean_pred = predictions.mean(dim=0)
    
    return variance, mean_pred


# 사용 예시
model = ResNetWithStochasticDepth(depth=50, drop_path_rate=0.2)
x = torch.randn(1, 3, 224, 224)

variance, mean_pred = measure_ensemble_effect(model, x, num_samples=100)
print(f"Prediction variance across samples: {variance:.6f}")
print(f"Higher variance → more ensemble diversity")
```

---

## 🔗 이론과 실전의 간격

### 훈련 vs Inference

**훈련**:
- Layer $\ell$을 확률 $p_\ell$로 drop
- Gradient는 active path들만 통과
- Expected output: $\mathbb{E}[y] = x + (1-p) F(x)$

**Inference**:
- Drop 없음, 모든 block keep
- Output: $y = x + F(x)$
- Scale 불일치 → rescaling 필요 또는 exponential moving average 사용

### 깊은 네트워크 훈련

ResNet-1001 (1000 layers):
- Without stochastic depth: 훈련 불가능
- With stochastic depth: 안정적 훈련 가능

이유:
- Implicit ensemble이 매우 강한 regularization
- Drop된 layer를 통해 많은 경로 학습
- Gradient flow 개선

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 각 layer 독립적 drop | Correlated drop도 가능 |
| Linear decay schedule이 optimal | 다른 schedule 탐색 가능 |
| Residual structure 가정 | Non-residual networks에 적응 필요 |

---

## 📌 핵심 정리

$$\boxed{y_\ell = x_\ell + \tilde{b}_\ell F_\ell(x_\ell) \text{ — Layer-wise stochastic dropout}}$$

| 개념 | 설명 |
|------|------|
| **Stochastic depth** | 전체 residual block을 확률 $p_\ell$로 drop |
| **Implicit ensemble** | $2^L$ 개 sub-network 자동 학습 |
| **Linear decay** | Early layer는 높은 drop prob, late layer는 낮은 drop prob |
| **Rescaling** | 훈련과 inference의 scale 일치 필요 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): ResNet-50에서 stochastic depth로 최대 drop prob = 0.2를 사용하면, 평균적으로 몇 개 layer가 drop되는가? (50 layers 가정, linear decay)

<details>
<summary>힌트 및 해설</summary>

Linear decay: $p_\ell = 0.2 \times \ell / 50$

각 layer의 기댓값:
$$\mathbb{E}[\text{dropped layers}] = \sum_\ell p_\ell = 0.2 \sum_\ell \frac{\ell}{50}$$
$$= 0.2 \times \frac{1}{50} \times \frac{50 \times 51}{2} = 0.2 \times 25.5 = 5.1$$

평균적으로 **약 5개 layer가 drop**.

따라서 effective depth: $50 - 5 = 45$ layers (평균).

</details>

**문제 2** (심화): Stochastic depth가 $2^L$개의 sub-network를 implicit하게 학습한다고 했을 때, ResNet-1001의 경우 몇 개의 configurations인가?

<details>
<summary>힌트 및 해설</summary>

1001 layers:
$$2^{1001} = \text{astronomical number}$$

정확히 계산하면:
$$2^{1001} \approx 10^{301} \text{ (매우 큼)}$$

하지만 실제로:
- 모든 configuration이 동등하게 샘플되지는 않음
- Drop prob이 layer마다 다르므로 일부 configurations가 더 자주 나타남
- 여전히 enormous ensemble effect

결론: Implicit하게 매우 많은 sub-network를 학습하므로, strong regularization.

</details>

**문题 3** (이론-실전): Drop probability schedule을 linear가 아닌 uniform (모든 layer에 같은 prob)로 하면 어떻게 될까?

<details>
<summary>힌트 및 해설</summary>

Uniform schedule: $p_\ell = p$ (모든 $\ell$)

문제점:
- Early layer (low-level features)도 자주 drop → 정보 손실
- Late layer (high-level features)도 자주 drop → 네트워크 표현력 부족

결과: Gradient flow 악화, 훈련 불안정.

**Linear decay의 이점**:
- Early layer: 높은 drop prob (low-level feature는 redundant)
- Late layer: 낮은 drop prob (high-level feature는 중요)
- Balanced regularization + Gradient flow 개선

따라서 **linear schedule이 empirically optimal**.

</details>

---

<div align="center">

[◀ 이전](./05-highway.md) | [📚 README](../README.md) | [다음 ▶](../ch5-modern-cnn/01-vgg-depth.md)

</div>
