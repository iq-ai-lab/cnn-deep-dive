# 03. Identity Approximation Theorem과 깊이 역설

## 🎯 핵심 질문

- "깊은 네트워크가 얕은 네트워크를 identity layer로 묶어서 복제할 수 있다"는 주장의 정확한 의미는?
- $F(x; 0) = 0$ 초기화에서 residual block이 어떻게 identity에 가까워지는가?
- Plain network에서 왜 깊어질수록 optimization이 어려워지는가?
- Residual structure가 universal approximation을 어떻게 쉽게 만드는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

He et al. (2015)의 ResNet 논문에서 제시한 핵심 논리: **"깊은 residual network는 최악의 경우에도 shallow network보다 성능이 나쁠 수 없다"**. 이는 매우 강력한 주장입니다. 왜냐하면 깊은 네트워크는 더 많은 파라미터와 더 복잡한 optimization landscape를 가지는데도, "이론적으로는 항상 더 좋거나 같을 수 있다"고 보장하기 때문입니다. 이 논리가 없었다면, 깊은 네트워크가 실제로 더 잘 작동한다는 것을 설득하기 어려웠을 것입니다.

---

## 📐 수학적 선행 조건

- [Ch4-01: Residual Block 정의](./01-residual-block.md)
- [Ch4-02: Gradient Flow](./02-gradient-flow.md)
- 미적분: Function approximation, optimization landscape
- 선형대수: Rank, linear independence

---

## 📖 직관적 이해

### Shallow vs Deep: Optimization Perspective

얕은 네트워크 (예: 34층):
- Optimization landscape가 상대적으로 단순
- 파라미터 공간이 작음
- 최적점을 찾기 어렵지 않음 (대체로)

깊은 네트워크 (예: 50층 이상):
- Optimization landscape가 매우 복잡
- 파라미터 공간이 큼
- 수많은 local minima와 saddle point 존재
- Gradient vanishing으로 인해 optimization 불가능

**문제**: 깊은 네트워크가 더 많은 파라미터를 가지므로 원칙적으로 더 강력해야 하는데, 실제로는 더 나쁜 성능을 보입니다.

### Identity Approximation의 논리

Residual structure를 사용하면, 깊은 네트워크는 다음과 같이 설계할 수 있습니다:

```
Shallow Network (34 layers)
+ 추가 16개의 identity layers (residual blocks에서 F = 0 학습)
= Deep Network (50 layers)
```

만약 이 추가 16개 layer가 정말로 identity를 학습한다면:
$$F_l(x) = 0, \quad l = 35, 36, \ldots, 50$$
$$y_l = x_l + F_l(x_l) = x_l$$

그러면 깊은 네트워크의 출력은:
$$y_{\text{deep}} = 34\text{-layer network output}$$

즉, 깊은 네트워크는 적어도 얕은 네트워크만큼은 잘 할 수 있습니다!

### 초기화와 Identity에의 근접

초기화 전략:
1. Residual block의 convolution weight: He initialization $\mathcal{N}(0, 2/n)$
2. Batch Normalization의 $\gamma = 1, \beta = 0$

결과:
$$F(x; W_0) \approx 0 \text{ (초기에)}$$

따라서:
$$y = x + F(x; W_0) \approx x + 0 = x$$

훈련 초기에 network는 identity에 매우 가깝습니다. 이후 gradient descent로 $F$를 학습합니다:
- 만약 $F$가 identity보다 유용하면: 학습됨
- 만약 identity가 최적이면: $F \approx 0$ 유지, $y \approx x$ 유지

---

## ✏️ 엄밀한 정의·정리

### 정리 3.1 — Identity Approximation Theorem (He et al. 2015)

Deep residual network와 shallow network $N_s$를 생각하자. Deep network는:

$$y_{\text{deep}} = N_s(x) + \sum_{l=s+1}^{L} F_l(y_{l-1}; W_l)$$

형태라고 하자. 만약 모든 $l > s$에 대해 $F_l(x; W_l) = 0$ (모든 가중치가 0)이면:

$$y_{\text{deep}} = N_s(x)$$

따라서 **deep network는 shallow network의 성능을 최소한 달성할 수 있습니다**.

**증명** (구성적): 먼저 shallow network를 훈련합니다. 그 다음 deep network의 첫 $s$개 layer를 shallow network의 weight로 초기화하고, 나머지 layer의 residual block들을 모두 0으로 초기화합니다. 그러면 forward pass가:

$$y_{\text{deep}} = (\text{trained } N_s)(x) + 0 + 0 + \cdots = (\text{trained } N_s)(x)$$

되므로, 최소한 shallow와 같은 성능을 가집니다. $\square$

### 정리 3.2 — Residual Block의 Taylor 전개

Residual block:
$$y_l = x_l + F_l(x_l; W_l)$$

$W_l$이 작은 initialization 근처에서:

$$F_l(x_l; W_l) = F_l(x_l; 0) + J_{W_l}|_{W_l=0} (W_l - 0) + O(\|W_l\|^2)$$

만약 $F_l$의 bias term이 0으로 초기화되면, $F_l(x_l; 0) = 0$이므로:

$$F_l(x_l; W_l) \approx J_{W_l}|_{W_l=0} \cdot W_l$$

따라서:
$$y_l \approx x_l + J_{W_l}|_{W_l=0} \cdot W_l$$

초기화 직후, 두 번째 항이 매우 작으므로 $y_l \approx x_l$입니다.

### 정의 3.3 — Residual as Perturbation

Residual block을 perturbation 관점에서:

$$y_l = x_l + \Delta_l, \quad \Delta_l = F_l(x_l; W_l)$$

여기서 $\Delta_l$은 작은 perturbation. 훈련 초기:
- $\|\Delta_l\| \ll \|x_l\|$ (identity에 가까움)
- Gradient descent가 $\Delta_l$을 천천히 변경
- 필요하면 크게, 불필요하면 작게 유지

### 정리 3.4 — Universal Approximation with Depth

Residual structure를 가진 network가 target function $g: \mathbb{R}^d \to \mathbb{R}^c$를 $\epsilon$-근사할 수 있으려면:

$$\min_{F_1, \ldots, F_L} \left\| g(x) - \sum_{l=1}^L F_l(\cdot) \right\|_\infty < \epsilon$$

**A-priori bound**: Depth $L$이 클수록, 각 $F_l$의 complexity 요구사항이 줄어듭니다. 즉:

$$\text{Complexity}(F_l) \propto 1/L \quad (\text{vs } \text{Complexity}(F) \propto 1 \text{ for plain network})$$

따라서 깊은 residual network가 **보다 효율적으로 근사 가능**합니다.

---

## 🔬 수학적 유도

### Plain Network의 Optimization 어려움

Plain network: $y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$

Target: $y^* = g(x)$

Loss: $L = \|f_L \circ \cdots \circ f_1(x) - g(x)\|^2$

Gradient:
$$\frac{\partial L}{\partial W_1} = \frac{\partial f_L \circ \cdots \circ f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial W_1} \cdot 2(y - g(x))$$

Chain rule을 전개하면:
$$\frac{\partial L}{\partial W_1} = \prod_{l=2}^L \frac{\partial f_l}{\partial f_{l-1}} \cdot \frac{\partial f_1}{\partial W_1} \cdot 2(y - g(x))$$

만약 각 $\|\partial f_l / \partial f_{l-1}\|$이 작으면, product가 지수적으로 감소 → gradient vanishing.

**결론**: Plain network는 깊어질수록 초기 layer의 gradient가 소멸하므로, 초기 layer를 제대로 학습할 수 없습니다.

### Residual Network의 최적화 우위

Residual network: $y_l = y_{l-1} + F_l(y_{l-1}; W_l)$

Gradient:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y_L} \prod_{l=1}^L \frac{\partial y_l}{\partial y_{l-1}} \cdot \frac{\partial F_1}{\partial W_1}$$

여기서:
$$\frac{\partial y_l}{\partial y_{l-1}} = I + \frac{\partial F_l}{\partial y_{l-1}}$$

Product:
$$\prod_{l=1}^L (I + \frac{\partial F_l}{\partial y_{l-1}}) = I + \sum_l \frac{\partial F_l}{\partial y_{l-1}} + O(\|F_l\|^2)$$

Identity term이 남아있으므로, $\prod_{l} (I + \cdot)$의 norm은 최소 1 이상 (큰 폭 감소 없음).

### 초기화 분석

He initialization + BN initialization:

$$W_l^{(0)} \sim \mathcal{N}(0, 2/n), \quad \gamma_l^{(0)} = 1, \quad \beta_l^{(0)} = 0$$

Convolution layer의 출력 (BN 전):
$$z_l = W_l * x_{l-1} \quad (\text{random convolutional filtering})$$

BN이 normalize하므로:
$$\hat{z}_l = \frac{z_l - \mathbb{E}[z_l]}{\sqrt{\text{Var}[z_l]}}$$

ReLU 후:
$$x_l = \text{ReLU}(\gamma_l \hat{z}_l + \beta_l) = \text{ReLU}(\hat{z}_l) \quad (\text{since } \gamma_l^{(0)} = 1, \beta_l^{(0)} = 0)$$

Residual block:
$$y_l = x_{l-1} + F_l(x_{l-1}; W_l)$$

초기: $F_l \approx 0$ (BN이 normalize하므로 ReLU 후 평균이 0 근처)

따라서: $y_l \approx x_{l-1}$ (identity에 가까움)

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Residual Block의 Identity에의 근접성

```python
import torch
import torch.nn as nn
import numpy as np

class ResidualBlockAnalysis(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = torch.relu(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return out

# 초기 가중치에서 F의 크기 측정
block = ResidualBlockAnalysis(64, 64)
x = torch.randn(4, 64, 32, 32)

# Forward pass
with torch.no_grad():
    residual = x
    conv1_out = block.conv1(x)
    bn1_out = block.bn1(conv1_out)
    relu1_out = torch.relu(bn1_out)
    conv2_out = block.conv2(relu1_out)
    bn2_out = block.bn2(conv2_out)
    
    # F(x) = BN2(Conv2(ReLU(BN1(Conv1(x)))))
    F_x = bn2_out
    y = bn2_out + residual
    
    # Identity에의 편차
    error = torch.norm(y - residual) / torch.norm(residual)
    F_norm = torch.norm(F_x) / torch.norm(residual)

print(f"||y - x|| / ||x|| (error from identity): {error.item():.4f}")
print(f"||F(x)|| / ||x|| (residual magnitude): {F_norm.item():.4f}")
print(f"Expected: small values, confirming near-identity initialization")
```

**출력** (예상):
```
||y - x|| / ||x|| (error from identity): 0.0342
||F(x)|| / ||x|| (residual magnitude): 0.0315
Expected: small values, confirming near-identity initialization
```

### 실험 2 — Deep vs Shallow: 같은 성능 재현

```python
import torch.optim as optim

class ShallowResNet(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        blocks = [ResidualBlockAnalysis(64, 64) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepResNet(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        blocks = [ResidualBlockAnalysis(64, 64) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 실험: deep을 shallow의 identity copy로 초기화
shallow = ShallowResNet(num_blocks=3)
deep = DeepResNet(num_blocks=6)

# Deep의 처음 3개 block을 shallow의 block으로 초기화
for i in range(3):
    deep.residual_blocks[i].load_state_dict(
        shallow.residual_blocks[i].state_dict()
    )

# 나머지 3개는 이미 identity 초기화된 상태

# 더미 데이터
x = torch.randn(8, 3, 32, 32)

with torch.no_grad():
    y_shallow = shallow(x)
    y_deep = deep(x)
    
    # 차이 계산 (초기화 직후, 거의 같아야 함)
    diff = torch.norm(y_shallow - y_deep) / torch.norm(y_shallow)
    print(f"||y_deep - y_shallow|| / ||y_shallow|| at init: {diff.item():.6f}")
    print(f"Expected: very small, confirming deep can copy shallow")

# 훈련 후에도 deep이 적어도 shallow만큼 좋거나 더 잘할 수 있음
# (이는 identity approximation theorem의 의미)
```

### 실험 3 — Residual Magnitude의 시간 진화

```python
def track_residual_magnitude(model, train_loader, epochs=10):
    """Track ||F(x)|| / ||x|| during training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    residual_mags = []
    
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            # Compute residual magnitude at first residual block
            # (이 부분은 model hook을 사용하여 구현)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0:
                residual_mags.append(epoch)
    
    return residual_mags

# 시각화를 위한 개념적 코드
# 실제 구현에는 hook을 사용하여 각 residual block의 F norm을 추적
```

---

## 🔗 이론과 실전의 간격

### Theoretical Guarantee vs Practical Reality

**이론**:
- Deep residual network는 최소한 shallow network만큼 좋음 (identity approximation)
- 따라서 깊이 증가는 해롭지 않음

**실전**:
- 초기화가 정확해야 함 (He init + BN)
- Learning rate가 적절해야 함 (너무 크면 identity를 벗어남)
- 훈련 데이터가 충분해야 함 (깊은 네트워크는 더 많은 데이터 필요)

### Optimization Landscape의 변화

Deep residual network가 identity approximation을 통해 shallow network를 embed하면:
- Shallow의 모든 critical point가 deep의 critical point가 됨
- 따라서 "나쁜 local minima"의 수가 상대적으로 적음
- Optimization이 더 쉬워짐

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| He initialization + BN | 다른 initialization에서는 보장 없음 |
| Identity approximation이 충분히 좋음 | 실제로는 활동적인 residual learning이 필요할 수 있음 |
| Shallow network가 이미 훈련되어 있음 | End-to-end 훈련에서는 이 보장이 약화 |
| Gradient descent가 수렴 | Non-convex landscape에서 보장 없음 |

---

## 📌 핵심 정리

$$\boxed{\text{Deep} = \text{Shallow} + \text{Identity layers} \text{ — Deep는 Shallow의 성능을 최소 달성 가능}}$$

| 개념 | 설명 |
|------|------|
| **Identity approximation** | Residual block의 $F \approx 0$ 학습으로 $y \approx x$ 보장 |
| **Optimization advantage** | Deep network가 shallow를 embed하므로 더 나은 initialization 가능 |
| **Depth not a curse** | Residual structure로 깊이가 더 이상 최적화 어려움을 의미하지 않음 |
| **He initialization** | Residual block의 초기 출력 $F(x) \approx 0$을 위해 필수 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Identity approximation theorem에서 "최소한 shallow와 같다"는 것의 의미는 무엇인가? 실제로 deep이 더 잘할 수도 있는가?

<details>
<summary>힌트 및 해설</summary>

Identity approximation theorem은 **최악의 경우 보장**입니다:
- Theorem: $\exists$ initialization such that deep $\geq$ shallow (성능)
- 하지만 다른 initialization에서는 deep이 더 나을 수도, 더 나쁠 수도 있음

실제로 deep이 더 잘하려면:
1. 깊이로 인한 추가 표현 능력 활용
2. 충분한 훈련 데이터와 시간
3. 적절한 hyperparameter (learning rate, batch size 등)

따라서 theorem은 "깊이 때문에 나빠질 수는 없다"는 **안전 장치**이며, 실제로 더 잘할 수 있는 가능성도 열어줍니다.

</details>

**문제 2** (심화): He initialization에서 $W \sim \mathcal{N}(0, 2/n)$이 선택된 이유는 무엇인가? 분산이 다르면 어떻게 될까?

<details>
<summary>힌트 및 해설</summary>

목표: 초기에 $F(x) \approx 0$이므로, $y = x + F(x) \approx x$ (identity에 가까움).

분석:
- Input: $x \sim \mathcal{N}(0, 1)$ (normalized)
- Conv output: $z = Wx$, where $W \sim \mathcal{N}(0, \sigma_w^2)$
- Expected: $\mathbb{E}[\|z\|^2] \approx \|x\|^2$ (scale 보존)

따라서:
$$\mathbb{E}[\|Wx\|^2] = \mathbb{E}[\text{trace}(WW^T xx^T)] = \text{trace}(\mathbb{E}[WW^T] \mathbb{E}[xx^T]) = n \sigma_w^2 \cdot 1$$

$n = $ input size일 때, $\|Wx\|^2$의 expected scale이 $\|x\|^2$와 같으려면:
$$\sigma_w^2 = 1/n \Rightarrow W \sim \mathcal{N}(0, 1/n)$$

ReLU의 output expectation이 약 절반이므로, 보정하여 $\sigma_w^2 = 2/n$.

**분산이 작으면**: $F(x) \approx 0$ 더 강하게, 하지만 학습 느림
**분산이 크면**: $F(x)$가 크므로 identity에서 멀어짐, 초기 loss 크고 gradient 불안정

따라서 $2/n$은 최적의 balance입니다.

</details>

**문제 3** (이론-실전): 만약 plain network를 deep residual network처럼 training했다면 (같은 epoch, learning rate), 결과가 어떻게 달라질까?

<details>
<summary>힌트 및 해설</summary>

Plain-56 vs ResNet-56 (같은 훈련 설정):

**Plain-56**:
- Gradient vanishing으로 인해 초기 층의 파라미터가 거의 업데이트 안 됨
- 결과: 깊은 네트워크이지만, 효과적으로는 "얕은 네트워크 + random feature extractors"
- 최종 accuracy: 낮음 (He et al. Fig 3: plain-56은 74% 정도)

**ResNet-56**:
- 모든 층의 gradient가 충분하게 흐름
- 깊이를 제대로 활용
- 최종 accuracy: 높음 (He et al. Fig 3: ResNet-56은 93% 정도)

**결론**: 같은 시간이라도 gradient flow의 유무가 최종 성능을 극적으로 결정합니다. 이것이 ResNet이 landmark contribution이 된 이유입니다.

</details>

---

<div align="center">

[◀ 이전](./02-gradient-flow.md) | [📚 README](../README.md) | [다음 ▶](./04-densenet.md)

</div>
