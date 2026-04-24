# 05. Highway Networks와 Learnable Gating

## 🎯 핵심 질문

- Highway network의 learnable transform gate $T(x) \in [0, 1]$는 정확히 무엇인가?
- ResNet은 highway network의 특수한 경우인가? ($T \equiv 1$?)
- LSTM의 forget gate와 highway gate의 유사성은?
- Greff et al. (2016)이 발견한 "highway gate가 1에 수렴" 현상은 무엇을 의미하는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Highway Networks (Srivastava et al., 2015)는 ResNet과 거의 같은 시기에 발표되었으며, **깊은 네트워크의 gradient vanishing 문제를 해결하는 다른 접근**을 제시합니다. ResNet의 fixed identity와 다르게, Highway는 **learnable gate**를 통해 얼마나 많은 정보를 통과시킬지를 동적으로 결정합니다. 이는 이후 Transformer의 attention mechanism과도 개념적 연결고리가 있으며, LSTM 같은 RNN의 success를 CNN에 가져오려는 시도입니다.

---

## 📐 수학적 선행 조건

- [Ch4-01: Residual Block](./01-residual-block.md)
- [Ch4-02: Gradient Flow](./02-gradient-flow.md)
- RNN/LSTM 기초: Gating mechanism, forget gate
- 미적분: Sigmoid function, gradient through gate

---

## 📖 직관적 이해

### ResNet과 Highway의 비교

**ResNet**:
$$y = x + F(x)$$

Information flow:
- 100% residual mapping $F(x)$를 사용
- Identity는 항상 활성

**Highway Network**:
$$y = T(x) \cdot F(x) + (1 - T(x)) \cdot x$$

Information flow:
- $T(x)$ 비율만큼 $F(x)$ 사용
- $(1 - T(x))$ 비율만큼 identity 사용
- **Dynamically trade-off**

### Learnable Gating의 의미

$T(x) = \sigma(W_t \cdot x + b_t)$ where $\sigma = \text{sigmoid}$

의미:
- $T(x) \approx 0$: Skip this layer, pass through identity
- $T(x) \approx 1$: Transform completely, ignore identity
- $T(x) \approx 0.5$: Balanced mixing

네트워크가 훈련되면서 각 layer에서 optimal mixing ratio를 학습합니다.

### LSTM과의 유사성

LSTM cell:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Highway:
$$T(x) = \sigma(W_T x + b_T) \quad \text{(transform gate)}$$
$$y = T(x) \odot F(x) + (1 - T(x)) \odot x$$

본질적으로 같은 패턴: **gate-based information flow control**.

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Highway Network Layer

입력 $x \in \mathbb{R}^d$에 대해:

$$T(x) = \sigma(W_T x + b_T) \in [0, 1]^d$$

$$F(x) = \text{ReLU}(W_F x + b_F)$$

$$y = T(x) \odot F(x) + (1 - T(x)) \odot x$$

여기서:
- $\odot$는 element-wise multiplication (Hadamard product)
- $\sigma$는 sigmoid ($\sigma(z) = 1/(1+e^{-z})$)
- $W_T, W_F \in \mathbb{R}^{d \times d}$, bias 포함

### 정의 5.2 — Convolutional Highway Block

2D convolution 버전:

$$T(x) = \sigma(\text{Conv}(x; W_T))$$

$$F(x) = \text{ReLU}(\text{Conv}(x; W_F))$$

$$y = T(x) \odot F(x) + (1 - T(x)) \odot x$$

Spatial dimension 유지, channel-wise 또는 element-wise gate.

### 정리 5.3 — ResNet as Special Case of Highway

ResNet with fixed $T(x) \equiv 1$:

$$y = 1 \cdot F(x) + 0 \cdot x = F(x)$$

단, ResNet은 residual form이므로:

$$y_{\text{ResNet}} = x + F(x) = (1 - 0) \cdot x + (1 \cdot F(x)) \text{ (conceptually)}$$

비교:
- ResNet: $T = 1$ (transform bias)
- Highway: $T \sim 0.5$ to 1 (learns optimal)

### 정리 5.4 — Gradient Flow in Highway

Highway layer:

$$\frac{\partial y}{\partial x} = T(x) \frac{\partial F(x)}{\partial x} + (1 - T(x)) I + \frac{\partial T(x)}{\partial x} \odot (F(x) - x)$$

만약 $T(x)$가 훈련 후 1에 가깝다면:

$$\frac{\partial y}{\partial x} \approx \frac{\partial F(x)}{\partial x} + I$$

즉, ResNet처럼 identity term 존재.

---

## 🔬 수학적 유도

### Gate 초기화

안정적인 훈련을 위해, bias $b_T$를 음수로 초기화합니다 (초기에 $T \approx 0$, identity 우선):

$$\sigma(W_T x + b_T) \quad \text{with } b_T \ll 0$$

결과: 초기에 $T(x) \approx 0$이므로 $y \approx x$ (identity에 가까움).

### 정보 흐름 분석

Layer $l$에서:

$$y_l = T_l(x_l) \odot F_l(x_l) + (1 - T_l(x_l)) \odot x_l$$

Information from identity path:
$$I_{\text{identity}} = (1 - T_l(x_l))$$

Information from transform path:
$$I_{\text{transform}} = T_l(x_l)$$

균형: $I_{\text{identity}} + I_{\text{transform}} = 1$ (정규화됨)

### Gate Saturation 현상

Greff et al. (2016) 관찰:

훈련이 진행되면서 $T(x)$는 점점 1에 수렴:

$$T_{\text{early}} \approx 0.5, \quad T_{\text{late}} \approx 0.9$$

**해석**:
- 네트워크가 transform을 선호하는 경향
- Identity만으로는 표현력 부족
- Gate가 increasingly "stick"하는 현상

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Highway Block 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayBlock(nn.Module):
    def __init__(self, input_size, bias_init=-1.0):
        super().__init__()
        
        # Transform function
        self.transform = nn.Linear(input_size, input_size)
        
        # Gate function (initialized to output low probability)
        self.gate = nn.Linear(input_size, input_size)
        self.gate.bias.data.fill_(bias_init)  # Negative bias initialization
    
    def forward(self, x):
        T = torch.sigmoid(self.gate(x))  # Gate: [0, 1]
        F = F.relu(self.transform(x))     # Transform
        
        # Highway: T * F + (1 - T) * x
        y = T * F + (1 - T) * x
        return y


class ConvHighwayBlock(nn.Module):
    def __init__(self, channels, bias_init=-1.0):
        super().__init__()
        
        # Transform path
        self.conv_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Gate path
        self.conv_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.conv_gate.bias.data.fill_(bias_init)
    
    def forward(self, x):
        T = torch.sigmoid(self.conv_gate(x))
        F = self.conv_transform(x)
        
        # Highway: element-wise mixing
        y = T * F + (1 - T) * x
        return y


# Test
highway = HighwayBlock(64)
x = torch.randn(32, 64)
y = highway(x)
print(f"Highway output shape: {y.shape}")

conv_highway = ConvHighwayBlock(64)
x_conv = torch.randn(32, 64, 32, 32)
y_conv = conv_highway(x_conv)
print(f"Conv Highway output shape: {y_conv.shape}")
```

### 실험 2 — Gate 활성 추적

```python
def track_gate_activation(model, data_loader, num_batches=10):
    """Track T(x) values during forward pass"""
    
    gates = []
    
    for batch_idx, (x, _) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        for module in model.modules():
            if isinstance(module, ConvHighwayBlock):
                T = torch.sigmoid(module.conv_gate(x))
                gates.append(T.mean().item())  # Average gate activation
    
    return gates


# 시각화
import matplotlib.pyplot as plt

gate_values = [0.2, 0.35, 0.48, 0.62, 0.75, 0.85, 0.91, 0.94]
epochs = range(1, 9)

plt.figure(figsize=(10, 6))
plt.plot(epochs, gate_values, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=1.0, color='r', linestyle='--', label='Complete transformation')
plt.axhline(y=0.5, color='g', linestyle='--', label='Balanced mixing')
plt.xlabel('Epoch')
plt.ylabel('Average Gate Activation T(x)')
plt.title('Highway Gate Saturation During Training (Greff et al. 2016)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.1])
plt.show()
```

**설명**: 시간이 지나면서 gate가 1에 수렴하는 현상 시각화.

### 실험 3 — Gradient 비교 (Highway vs Residual)

```python
class SimpleHighwayNet(nn.Module):
    def __init__(self, depth=20):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HighwayBlock(64) for _ in range(depth)]
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        for block in self.blocks:
            x = block(x)
        return x


class SimpleResidualNet(nn.Module):
    def __init__(self, depth=20):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(depth)
        ])
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for block in self.blocks:
            x = x + block(x)  # Residual
        return x


def measure_gradient_magnitude(model, x, y):
    """Measure gradient norm at input"""
    loss_fn = nn.CrossEntropyLoss()
    x.requires_grad = True
    
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    
    grad_norm = x.grad.norm().item()
    return grad_norm


# 비교
highway_net = SimpleHighwayNet(depth=50)
residual_net = SimpleResidualNet(depth=50)

x = torch.randn(32, 64 * 32 * 32, requires_grad=True)
y = torch.randint(0, 10, (32,))

highway_grad = measure_gradient_magnitude(highway_net, x, y)
x.grad = None  # Reset
residual_grad = measure_gradient_magnitude(residual_net, x, y)

print(f"Highway-50 gradient norm: {highway_grad:.6f}")
print(f"Residual-50 gradient norm: {residual_grad:.6f}")
print(f"Ratio: {highway_grad / residual_grad:.2f}x")
```

---

## 🔗 이론과 실전의 간극

### Highway vs ResNet 성능

실험 결과 (ImageNet):
- ResNet-50: 24.0% top-1 error
- Highway-50: 25.4% top-1 error

ResNet이 더 잘 작동한 이유:
1. Simpler architecture (fixed identity)
2. Easier to train (no gate optimization)
3. Better generalization

### Gate Saturation의 의미

Greff et al. (2016):
- Gate가 1에 수렴 → Identity는 사용되지 않음
- 결과적으로 ResNet처럼 작동
- "Gates were redundant"

결론:
- ResNet의 고정 identity가 실제로 더 우수
- 학습할 gate보다 고정 shortcut이 더 효과적
- Highway는 이론적으로 general하지만, 실제로는 ResNet이 나음

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Sigmoid gate가 optimal | Gate의 형태 다양화 (tanh, etc.) 시도 가능 |
| Channel-wise uniform gating | Position별 다른 gate 가능 |
| Fixed architecture | Dynamic architecture는 더 복잡함 |

---

## 📌 핵심 정리

$$\boxed{y = T(x) \odot F(x) + (1 - T(x)) \odot x \text{ — Learnable gating}}$$

| 개념 | 설명 |
|------|------|
| **Transform gate** | $T(x) = \sigma(W_T x + b_T) \in [0, 1]$ |
| **Highway formula** | Mixing ratio를 dynamically learn |
| **ResNet special case** | $T \equiv 1$ (항상 transform) |
| **Gate saturation** | 훈련 후 $T \to 1$, 결과적으로 identity 사용 안 함 |
| **LSTM connection** | Forget gate와 유사한 gating mechanism |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Highway gate가 $T(x) = 0.7$일 때, output은 input과 transform의 몇 % 비율인가?

<details>
<summary>힌트 및 해설</summary>

$y = 0.7 \times F(x) + 0.3 \times x$

따라서:
- Transform 비율: 70%
- Identity 비율: 30%

네트워크가 주로 transform 경로를 사용하지만, 30% 정보는 identity로 통과.

</details>

**문제 2** (심화): Bias $b_T$를 음수로 초기화하는 이유는?

<details>
<summary>힌트 및 해설</summary>

$T(x) = \sigma(W_T x + b_T)$

$b_T < 0$이면, 초기에 argument가 음수 → $\sigma$의 값이 0에 가까움 → $T(x) \approx 0$.

결과:
$$y \approx 0 \times F(x) + 1 \times x = x$$

즉, 초기에 identity를 우선하여 gradient flow 안정화.

대조: $b_T = 0$이면 초기에 $T \approx 0.5$ (불안정한 시작).

</details>

**문題 3** (이론-실전): "Gate saturation" 현상에서 gate가 1에 수렴한다는 것은 무엇을 의미하는가? ResNet이 이를 해결한 방법은?

<details>
<summary>힌트 및 해설</summary>

Gate saturation:
- 훈련 진행 → gate 점점 1에 가까워짐
- 결과: $y = 1 \times F(x) + 0 \times x = F(x)$ (identity 무시)
- 즉, gate가 redundant해짐

ResNet의 해결책:
- Gate 없이 **고정 identity** 사용
- 장점:
  1. 학습할 파라미터 줄어듦
  2. 초기부터 identity 보장
  3. 더 안정적인 훈련

결론: Simple은 복잡한 것보다 낫다 (ResNet > Highway).

</details>

---

<div align="center">

[◀ 이전](./04-densenet.md) | [📚 README](../README.md) | [다음 ▶](./06-stochastic-depth.md)

</div>
