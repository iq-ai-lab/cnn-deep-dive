# 02. Gradient Flow 분석: Residual vs Plain

## 🎯 핵심 질문

- Plain network에서 gradient magnitude가 깊이에 따라 어떻게 변하는가?
- Residual block의 skip connection이 gradient flow에 어떤 영향을 주는가?
- $\partial y_L / \partial x_0$을 계산할 때 identity term이 왜 중요한가?
- He et al. (2016) Fig 1 — "deeper networks have larger gradients"를 재현할 수 있는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

ResNet의 진정한 혁신은 gradient flow를 **구조적으로 개선**했다는 점입니다. Plain network에서는 역전파 중 gradient magnitude가 지수적으로 감소하여, 초기 층의 파라미터가 거의 학습되지 않습니다 (gradient vanishing). 반면 residual block의 skip connection은 identity 경로를 제공하여 gradient가 "최소한 1은 전파"되도록 합니다. 이 문서에서는 이 현상을 수학적으로 분석하고, PyTorch로 재현합니다.

---

## 📐 수학적 선행 조건

- [Ch1: Gradient Descent 기초](../ch1-optimization/01-gd-geometry.md)
- [Ch4-01: Residual Block의 정의](./01-residual-block.md)
- 미적분: Chain rule, 다변수 미분
- 선형대수: Matrix norm, spectral properties

---

## 📖 직관적 이해

### Plain Network에서의 Gradient Vanishing

길이 $L$인 plain network:

$$x_{l+1} = \sigma(W_l x_l + b_l), \quad l = 0, 1, \ldots, L-1$$

여기서 $\sigma = \text{ReLU}$라고 하자. Backpropagation:

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial \sigma(W_l x_l + b_l)}{\partial x_l}$$

연쇄 법칙으로:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \prod_{l=0}^{L-1} \frac{\partial \sigma(W_l x_l + b_l)}{\partial x_l}$$

만약 $\|\partial \sigma / \partial x\| < 1$인 경우가 많으면, $L$층을 거치면서 product가 지수적으로 감소합니다:

$$\left\|\frac{\partial L}{\partial x_0}\right\| \approx \left\|\frac{\partial L}{\partial x_L}\right\| \cdot \gamma^L, \quad \gamma < 1$$

이것이 gradient vanishing입니다.

### Residual Network에서의 개선

Residual block:

$$y_l = x_l + F_l(x_l)$$

Gradient:

$$\frac{\partial y_l}{\partial x_l} = I + \frac{\partial F_l}{\partial x_l}$$

연쇄 법칙:

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial y_l} \left(I + \frac{\partial F_l}{\partial x_l}\right)$$

역전파를 전개하면:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial y_L} \prod_{l=0}^{L-1} \left(I + \frac{\partial F_l}{\partial x_l}\right)$$

이 product를 Taylor 전개하면:

$$\prod_{l} (I + \Delta_l) = I + \sum_l \Delta_l + O(\|\Delta_l\|^2)$$

따라서 1차 항 $I$가 반드시 남아있습니다. 비록 $\Delta_l$이 작더라도 gradient는 최소 그 크기를 유지합니다.

### Gradient Highway 메타포

이 identity term $I$를 직관적으로 이해하면 — 각 residual block이 **gradient를 감쇠 없이 통과시키는 "고속도로(highway)"**를 제공하는 셈입니다. Plain network의 gradient가 **매 층마다 일반도로의 신호등에서 감쇠**되는 것과 달리, residual block에서는:

- **Highway 차선** ($I$ 경로): gradient가 $\partial L / \partial y_L$ 크기 그대로 $x_0$까지 직통
- **Local 차선** ($\partial F_l / \partial x_l$ 경로): residual function의 학습된 변형

깊이가 $L$이 되어도 highway 차선의 gradient는 $\approx \partial L / \partial y_L$을 유지 — 이것이 He et al. 2016이 plain-56의 훈련 실패를 ResNet-56으로 해결한 기계적 이유입니다.

### 정량적 비교

Plain network, $L = 50$:
- $\gamma = 0.9$ (각 layer의 jacobian singular value 평균)
- Gradient magnitude: $\gamma^{50} = 0.9^{50} \approx 0.005$ — **99.5% 감소**

Residual network, $L = 50$:
- 각 layer에서 $\|I + \Delta_l\|_{\text{spectral}} \approx 1 + \epsilon$ (작은 $\epsilon$)
- Gradient magnitude: $\approx 1$ — **거의 변화 없음**

---

## ✏️ 엄밀한 정의·정리

### 정리 2.1 — Plain Network의 Gradient Vanishing

길이 $L$인 plain feedforward network에서:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \prod_{l=0}^{L-1} J_l$$

여기서 $J_l = \partial \sigma(W_l x_l) / \partial x_l$는 Jacobian. 만약 모든 $l$에 대해 $\|J_l\|_{\text{spectral}} \leq \gamma < 1$이면:

$$\left\|\frac{\partial L}{\partial x_0}\right\| \leq \left\|\frac{\partial L}{\partial x_L}\right\| \cdot \gamma^L$$

따라서 $L$이 충분히 크면 gradient magnitude는 지수적으로 감소합니다.

**증명**: Spectral norm의 곱셈 성질 $\|AB\| \leq \|A\| \|B\|$에서 직접 따릅니다. $\square$

### 정리 2.2 — Residual Block의 Gradient 안정성

Pre-activation residual block:

$$y_l = x_l + F_l(x_l; W_l)$$

에 대해, Jacobian:

$$\frac{\partial y_l}{\partial x_l} = I + \frac{\partial F_l}{\partial x_l}$$

만약 $\left\|\frac{\partial F_l}{\partial x_l}\right\|_{\text{spectral}} \leq \mu < \infty$이면:

$$\left\|I + \frac{\partial F_l}{\partial x_l}\right\|_{\text{spectral}} \in [1 - \mu, 1 + \mu]$$

따라서 역전파 시 gradient norm은 최소한 $(1-\mu)^L$까지만 감소합니다.

**증명**: Spectral norm의 삼각 부등식. 최악의 경우 eigenvalue들이 모두 $-\mu$ 근처인데도 $I + \Delta$의 smallest eigenvalue는 $1 - \mu > 0$ (if $\mu < 1$). $\square$

### 정의 2.3 — Effective Gradient Depth

네트워크 $L$층 중 gradient가 "살아있는" 유효 깊이:

$$d_{\text{eff}} := \arg \max_l \left\|\frac{\partial L}{\partial x_l}\right\| / \left\|\frac{\partial L}{\partial x_L}\right\| > \lambda$$

여기서 $\lambda$는 threshold (보통 0.1 또는 0.01).

Plain network: $d_{\text{eff}} \ll L$ (대부분의 초기 층의 gradient가 무시됨)
Residual network: $d_{\text{eff}} \approx L$ (모든 층이 충분한 gradient를 받음)

---

## 🔬 수학적 유도

### Plain Network의 Explicit Gradient 계산

2층 plain network를 예시로:

$$x_1 = \sigma(W_1 x_0)$$
$$x_2 = \sigma(W_2 x_1)$$
$$L = \frac{1}{2}\|x_2 - y_{\text{true}}\|^2$$

역전파:

$$\frac{\partial L}{\partial x_2} = x_2 - y_{\text{true}}$$

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_2} \cdot \text{diag}(\sigma'(W_2 x_1)) \cdot W_2^T$$

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_1} \cdot \text{diag}(\sigma'(W_1 x_0)) \cdot W_1^T$$

$\sigma = \text{ReLU}$인 경우, $\sigma'(z) = \mathbb{1}_{z > 0}$. 따라서:

$$\frac{\partial L}{\partial x_0} = (x_2 - y_{\text{true}})^T W_2^T \text{diag}(\sigma'(W_2 x_1)) W_1^T \text{diag}(\sigma'(W_1 x_0))$$

ReLU의 mask가 두 번 곱해지므로, 많은 뉴런이 zero gradient를 받습니다.

### Residual Network의 Explicit Computation

2개 residual block:

$$y_0 = x_0 + F_0(x_0)$$
$$y_1 = y_0 + F_1(y_0)$$

역전파:

$$\frac{\partial L}{\partial y_1} = \nabla_{\text{output}} L$$

$$\frac{\partial L}{\partial y_0} = \frac{\partial L}{\partial y_1} \left(I + \frac{\partial F_1}{\partial y_0}\right) = \frac{\partial L}{\partial y_1} + \frac{\partial L}{\partial y_1} \frac{\partial F_1}{\partial y_0}$$

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial y_0} \left(I + \frac{\partial F_0}{\partial x_0}\right)$$

여기서 중요한 점: $\frac{\partial L}{\partial y_0}$의 첫 항 $\frac{\partial L}{\partial y_1}$은 **$F_1$을 거치지 않고 직접 전파**됩니다. 따라서 gradient magnitude가 보존됩니다.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Plain vs Residual Network의 Gradient Magnitude 추적

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PlainNet(nn.Module):
    def __init__(self, depth=56, width=16):
        super().__init__()
        layers = []
        in_ch = 3
        for i in range(depth):
            layers.append(nn.Conv2d(in_ch, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            in_ch = width
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(width, 10))
        self.features = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))
    
    def relu(self, x):
        return torch.nn.functional.relu(x)


class ResNet(nn.Module):
    def __init__(self, depth=56, width=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        
        # depth-1 residual blocks
        blocks = []
        in_ch = width
        for i in range(depth - 1):
            blocks.append(ResidualBlock(in_ch, width))
        
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def measure_gradient_norms(model, x, y):
    """Measure gradient norm at each layer"""
    loss_fn = nn.CrossEntropyLoss()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    return grad_norms, loss.item()


# 테스트
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depths = [20, 32, 44, 56]
x = torch.randn(32, 3, 32, 32, device=device)
y = torch.randint(0, 10, (32,), device=device)

plain_grad_norms = []
resnet_grad_norms = []

for depth in depths:
    # Plain network
    plain = PlainNet(depth=depth, width=16).to(device)
    grad_norms, _ = measure_gradient_norms(plain, x, y)
    plain_grad_norms.append(np.mean(grad_norms))
    
    # ResNet
    resnet = ResNet(depth=depth, width=16).to(device)
    grad_norms, _ = measure_gradient_norms(resnet, x, y)
    resnet_grad_norms.append(np.mean(grad_norms))

# 시각화
plt.figure(figsize=(10, 6))
plt.semilogy(depths, plain_grad_norms, 'o-', label='Plain Network', linewidth=2)
plt.semilogy(depths, resnet_grad_norms, 's-', label='ResNet', linewidth=2)
plt.xlabel('Network Depth')
plt.ylabel('Average Gradient Norm')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.title('Gradient Flow: Plain vs ResNet (He et al. 2016 Fig 1 재현)')
plt.show()
```

**예상 결과**:
- Plain network: Depth 증가에 따라 gradient norm이 지수적으로 감소
- ResNet: Depth 변화에 관계없이 gradient norm이 거의 일정

### 실험 2 — Layer-wise Gradient 추적

```python
def track_layerwise_gradients(model, x, y):
    """Track gradient magnitude at each layer"""
    loss_fn = nn.CrossEntropyLoss()
    
    # Forward pass with hook
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward and backward
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    
    # Extract gradients
    layer_grad_norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            layer_grad_norms[name] = param.grad.norm().item()
    
    return layer_grad_norms

# ResNet-20 분석
resnet20 = ResNet(depth=20, width=16)
grad_dict = track_layerwise_gradients(resnet20, x, y)

# 시각화
layer_names = list(grad_dict.keys())[:10]  # 처음 10개 layer
grad_values = [grad_dict[name] for name in layer_names]

plt.figure(figsize=(12, 6))
plt.bar(range(len(layer_names)), grad_values)
plt.xlabel('Layer Index')
plt.ylabel('Gradient Norm')
plt.title('Layer-wise Gradient Distribution in ResNet-20')
plt.xticks(range(len(layer_names)), ['L'+str(i) for i in range(len(layer_names))], rotation=45)
plt.tight_layout()
plt.show()
```

**해석**: 초기 layer부터 마지막 layer까지 gradient norm이 균등하게 분포. Plain network와 대조적.

### 실험 3 — Backward Pass 속도 비교

```python
import time

def measure_backward_time(model, x, y, iterations=100):
    """Measure backward pass speed"""
    loss_fn = nn.CrossEntropyLoss()
    times = []
    
    for _ in range(iterations):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        start = time.time()
        loss.backward()
        end = time.time()
        times.append(end - start)
        
        model.zero_grad()
    
    return np.mean(times[10:])  # Skip first 10 (warmup)

# 비교
plain_time = measure_backward_time(PlainNet(depth=56), x, y)
resnet_time = measure_backward_time(ResNet(depth=56), x, y)

print(f"Plain-56 backward time: {plain_time*1000:.3f}ms")
print(f"ResNet-56 backward time: {resnet_time*1000:.3f}ms")
print(f"Ratio: {resnet_time/plain_time:.2f}x")
```

---

## 🔗 이론과 실전의 간극

### Batch Normalization의 중요성

BN이 없으면 내부 covariate shift로 인해 gradient flow가 악화됩니다:
- BN은 각 layer의 input을 normalize하여 gradient 전파를 안정화
- ResNet의 성공은 실제로 "BN + Residual connection" 조합에서 비롯됨
- 최근: Group Norm, Layer Norm 등으로 BN 대체 가능

### Initialization의 역할

He initialization ($\mathcal{N}(0, 2/n)$):
- Residual block의 초기 $F(x) \approx 0$에 가까우므로, gradient path가 identity를 통해 흐름
- 잘못된 초기화는 이 이점을 상쇄

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\|\partial F_l/\partial x_l\|$ 이 bounded | 매우 깊은 네트워크에서는 gradient explosion 가능성 |
| Batch Normalization이 layer를 normalize | BN 없이는 gradient flow 개선 미미 |
| Small learning rate | Large LR에서는 gradient instability 여전히 가능 |

---

## 📌 핵심 정리

$$\boxed{\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial y_L} \prod_{l=0}^{L-1} (I + \frac{\partial F_l}{\partial x_l}) \text{ — Identity term이 gradient 안정성의 핵}}$$

| 개념 | 설명 |
|------|------|
| **Gradient vanishing** | Plain network에서 역전파 시 gradient magnitude가 지수적으로 감소 |
| **Skip connection의 역할** | $y = x + F(x)$에서 $\partial y/\partial x = I + \partial F/\partial x$로 identity 경로 제공 |
| **Effective gradient depth** | Residual network에서는 모든 층의 gradient가 충분하게 전파됨 |
| **He et al. Fig 1** | Plain-56은 gradient가 0.005 이하로 감소, ResNet-56은 거의 불변 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Plain network에서 50층을 거친 gradient의 크기가 약 0.005로 줄어든다고 했을 때, 이것의 의미를 설명하라. 초기 layer의 파라미터는 어떻게 학습될 것인가?

<details>
<summary>힌트 및 해설</summary>

Gradient가 0.005 수준이면 학습률(learning rate) $\eta = 0.01$에서 파라미터 업데이트:

$$W_{\text{new}} = W_{\text{old}} - 0.01 \times 0.005 = W_{\text{old}} - 0.00005$$

즉, 초기 층의 파라미터는 거의 업데이트되지 않습니다. 따라서:
- 초기 층은 random state에 가까운 feature extraction
- 주로 후기 층만 의미 있게 학습
- 전체 네트워크의 표현 능력이 제한됨

반면 ResNet에서 gradient는 $\approx 1$이므로:
$$W_{\text{new}} = W_{\text{old}} - 0.01 \times 1 = W_{\text{old}} - 0.01$$

모든 층이 균등하게 학습되어, 깊은 네트워크도 효과적입니다.

</details>

**문제 2** (심화): Residual block에서 $\frac{\partial F}{\partial x}$의 spectral norm이 2라면, $I + \frac{\partial F}{\partial x}$의 smallest eigenvalue는 몇일까? Gradient가 안정적일까?

<details>
<summary>힌트 및 해설</summary>

$\|\frac{\partial F}{\partial x}\|_{\text{spectral}} = 2$이면, $\frac{\partial F}{\partial x}$의 eigenvalue는 최대 2 크기.

$I + \frac{\partial F}{\partial x}$의 eigenvalue:
- 최대: $1 + 2 = 3$
- 최소: $1 + (-2) = -1$ (worst case) 또는 $1 - 2 = -1$

**문제**: Smallest eigenvalue가 음수 또는 0 근처이면, spectral norm은 3이 되고, gradient norm이 3배 증가 (exploding).

실제로 깊은 ResNet에서 **gradient explosion** 문제가 발생할 수 있으므로, 이를 제어하기 위해:
- Gradient clipping
- Learning rate decay
- Careful initialization (small $\sigma^2$ for $W$)

이런 조치들이 필요합니다.

</details>

**문제 3** (이론-실전): Batch Normalization 없이 Residual block을 사용하면 어떤 일이 일어날까?

<details>
<summary>힌트 및 해설</summary>

BN이 없는 residual block:
$$y = \text{Conv}(x) + x$$

문제점:
1. Forward pass: $\text{Conv}(x)$의 scale이 $x$의 scale과 달라서, addition이 불균형
   - 예: $\|x\| = 1$, $\|\text{Conv}(x)\| = 100$이면, $y \approx 100$
   - 이후 layer의 input이 매우 크거나 작아짐

2. Backward pass: Scale imbalance로 인해 gradient도 불균형
   - Residual term의 gradient와 Conv term의 gradient 크기가 크게 다름
   - Effective learning이 어려움

따라서 **BN은 ResNet의 필수 요소**입니다. 최근 연구들도 BN 대체물(Group Norm, Layer Norm)을 찾으려고 노력하는 이유입니다.

</details>

---

<div align="center">

[◀ 이전](./01-residual-block.md) | [📚 README](../README.md) | [다음 ▶](./03-identity-approximation.md)

</div>
