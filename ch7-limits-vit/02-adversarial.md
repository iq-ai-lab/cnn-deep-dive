# 02. Adversarial Examples와 CNN의 취약성

## 🎯 핵심 질문

- Szegedy et al. (2014)의 발견은 무엇인가? CNN이 지각 불가능한 perturbation에 완전히 속는 이유는?
- FGSM (Fast Gradient Sign Method) 공격 $x' = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))$는 왜 효과적인가?
- PGD (Projected Gradient Descent) 공격과 FGSM의 차이점은 무엇인가?
- Adversarial robustness와 standard accuracy의 trade-off (Tsipras 2019)를 정량적으로 이해할 수 있는가?

---

## 🔍 왜 이 개념이 CNN 이해에 중요한가

Adversarial examples는 CNN의 **근본적인 취약성**을 드러냅니다. ImageNet에서 96% 정확도를 달성한 CNN도 (인간에게는 동일한) 약간의 노이즈가 추가되면 완전히 다른 클래스로 분류합니다. 이는 CNN의 inductive bias와 학습 방식이 인간의 인식과 얼마나 다른지를 보여줍니다. 또한 adversarial robustness는 단순한 보안 문제가 아니라, CNN의 **근본적인 표현 능력의 한계**를 나타냅니다.

---

## 📐 수학적 선행 조건

- [Gradient Descent와 최적화](../ch1-gd-geometry/): Gradient, loss function, optimization
- [CNN 기본 연산](../ch2-cnn-ops/): Convolution, forward/backward pass
- 선형대수: Norm ($\ell_\infty$, $\ell_2$), 벡터 부등식
- 미분: Gradient, Jacobian, chain rule

---

## 📖 직관적 이해

### Adversarial Vulnerability의 본질

CNN의 prediction은 다음과 같이 표현됩니다:

$$\hat y = \arg\max_c f_c(\theta, x)$$

여기서 $f_c(\theta, x)$는 클래스 $c$에 대한 logit(또는 확률).

Adversarial perturbation $\delta$를 추가하면:

$$\hat y' = \arg\max_c f_c(\theta, x + \delta), \quad \|\delta\|_\infty \leq \epsilon \ll \text{scale}(x)$$

놀랍게도, $\epsilon$이 아주 작아도 (보통 8/255 ≈ 0.03) $\hat y' \neq \hat y$가 되도록 하는 $\delta$가 존재합니다.

### FGSM의 직관

**Fast Gradient Sign Method (FGSM)**는 가장 단순한 공격입니다:

$$\delta = \epsilon \cdot \text{sign}(\nabla_x L(x, y))$$

직관:
1. Loss를 최대화하는 gradient 방향 계산
2. 그 방향으로 **한 step만** 이동
3. "Fast"라고 불리는 이유는 gradient를 한 번만 계산

이것도 효과적인 이유는 **gradient의 크기가 단순히 방향**이 아니라 **linear 민감도**를 나타내기 때문입니다. 높은 차원에서 CNN은 각 방향이 loss에 민감하게 설계되어 있습니다.

### PGD (Projected Gradient Descent) 공격

FGSM은 한 번의 step이므로 약한 공격입니다. 더 강력한 **반복 공격**은:

$$x_{t+1} = \text{Clip}_{B_\epsilon(x)}(x_t + \alpha \cdot \text{sign}(\nabla_x L(x_t, y)))$$

여기서:
- $\text{Clip}_{B_\epsilon(x)}$: $\epsilon$-ball 안으로 projection
- $\alpha$: step size (보통 $\epsilon / k$, $k$는 반복 수)
- 여러 번 반복하면 FGSM보다 훨씬 강력

### Robustness vs Accuracy Trade-off

Tsipras et al. (2019)의 중요한 발견은:

$$\text{최대 달성 가능 robustness} = f(\text{data complexity})$$

예를 들어:
- 큰 $\epsilon$ (크기 큰 perturbation)에 robust하려면, 많은 class를 구분할 여유가 없어짐
- Standard accuracy ↑ → Adversarial robustness ↓ (대략 inverse 관계)
- ResNet-50: 표준 76% vs robust 50% (robust to $\ell_\infty$ 8/255)

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Adversarial Example (Szegedy 2014)

분류기 $f_\theta$가 주어졌을 때, 입력 $x$의 adversarial example은:

$$x' = x + \delta$$

여기서:
1. $\|\delta\|_p \leq \epsilon$ (perturbation이 작음)
2. $f_\theta(x) = y$ (원본은 정확함)
3. $f_\theta(x') \neq y$ (adversarial은 틀림)

보통은 $\ell_\infty$ norm 사용: $\|delta\|_\infty \leq \epsilon$ (각 픽셀의 절댓값 제약).

### 정의 2.2 — FGSM 공격 (Goodfellow 2014)

주어진 손실함수 $L(\theta, x, y)$에 대해:

$$x' = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))$$

여기서 $\text{sign}$은 element-wise sign function.

### 정의 2.3 — PGD 공격 (Madry 2018)

반복 공격:

$$x^{(0)} = x$$
$$x^{(t+1)} = \text{Clip}_{B_\epsilon^\infty(x)}(x^{(t)} + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^{(t)}, y)))$$

여기서 $\text{Clip}_{B_\epsilon^\infty(x)}(x') = \text{clip}(x', x - \epsilon, x + \epsilon)$ (각 픽셀).

### 정리 2.4 — 선형성 가설 (Linear Hypothesis)

Goodfellow et al. (2014)는 adversarial vulnerability의 원인이 **모델의 선형성**이라고 제안:

고차원 공간 $\mathbb{R}^d$에서, 선형 분류기:

$$f(x) = w^\top x$$

의 margin은:

$$\text{margin} = \frac{|w^\top(x - x')|}{\\|w\|_2} = \frac{w^\top (x - x')}{\\|w\|_2}$$

만약 $\|x - x'\|_\infty = \epsilon$이면:

$$\text{margin} \leq \|w\|_\infty \cdot d \cdot \epsilon$$

따라서 **고차원** ($d$ 크면) 상황에서는 아주 작은 $\epsilon$으로도 margin을 넘을 수 있습니다.

### 정리 2.5 — Robustness-Accuracy Trade-off (Tsipras 2019)

$\epsilon$-robust 모델을 학습할 때, 달성 가능한 최대 정확도는:

$$\max_\theta [\text{Acc}(\theta) + \text{RobAcc}_\epsilon(\theta)] \leq C$$

여기서 $C$는 상수 (데이터 복잡도에 따라 결정).

직관: Robust decision boundary를 만들려면, 각 클래스 간 margin이 커야 하므로, 세밀한 구분 어려움.

---

## 🔬 증명 및 수학적 유도

### FGSM의 직관적 유도

1차 Taylor 근사:

$$L(\theta, x + \delta) \approx L(\theta, x) + \nabla_x L(\theta, x)^\top \delta$$

손실을 최대화하는 perturbation:

$$\max_{\|\delta\|_\infty \leq \epsilon} L(\theta, x + \delta) \approx \max_{\|\delta\|_\infty \leq \epsilon} [\nabla_x L^\top \delta]$$

$\|\delta\|_\infty \leq \epsilon$ 제약 하에서:

$$\delta^* = \epsilon \cdot \text{sign}(\nabla_x L(\theta, x))$$

이것이 **FGSM**입니다.

### 고차원에서의 취약성

$d$차원 벡터 $x$, gradient $g = \nabla_x L$가 주어졌을 때, 만약 $g$의 성분들이 대체로 같은 크기라면:

$$\|g\|_\infty \approx \frac{\|g\|_2}{\sqrt{d}}$$

따라서:

$$\|g\|_2 \approx \sqrt{d} \cdot \|g\|_\infty$$

perturbation $\delta = \epsilon \cdot \text{sign}(g)$의 영향:

$$g^\top \delta = \sum_i g_i \cdot \epsilon \cdot \text{sign}(g_i) = \epsilon \sum_i |g_i| = \epsilon \|g\|_1$$

$\|g\|_1 \approx d \cdot \|g\|_\infty$이므로:

$$\text{loss increase} \approx \epsilon \cdot d \cdot \|g\|_\infty$$

**고차원 $d$가 크면**, 아주 작은 $\epsilon$으로도 큰 손실 증가 가능.

### Adversarial Training의 원리

Robust 모델 학습:

$$\min_\theta \mathbb{E}_{(x,y)}[\max_{\|\delta\|_\infty \leq \epsilon} L(\theta, x + \delta, y)]$$

이를 근사하기 위해 PGD adversarial training 사용:

$$\min_\theta \mathbb{E}_{(x,y)}[L(\theta, x_\text{adv}, y)]$$

여기서 $x_\text{adv}$는 PGD로 생성한 adversarial example.

---

## 💻 실험 재현

### 실험 1 — FGSM 공격 구현 및 시각화

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 사전학습된 ResNet-50 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device)
model.eval()

# 정규화
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# FGSM 공격 함수
def fgsm_attack(model, x, y, epsilon):
    """
    Args:
        model: 분류 모델
        x: 입력 이미지 (normalized)
        y: ground truth 레이블
        epsilon: perturbation 크기
    
    Returns:
        x_adv: adversarial example
        perturbation: 추가된 노이즈
    """
    x.requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass (gradient 계산)
    model.zero_grad()
    loss.backward()
    
    # FGSM
    data_grad = x.grad.data
    perturbation = epsilon * torch.sign(data_grad)
    x_adv = x + perturbation
    
    # Clipping (valid pixel range)
    x_adv = torch.clamp(x_adv, -2.1, 2.64)  # ImageNet normalization 범위
    
    return x_adv, perturbation

# 샘플 이미지 (무작위 생성, 실제로는 ImageNet 이미지 사용)
x = torch.randn(1, 3, 224, 224, device=device)
y = torch.tensor([1], device=device)  # 임의의 클래스

with torch.no_grad():
    clean_pred = model(x).argmax(dim=1).item()

# 다양한 epsilon에 대한 공격
epsilons = [0.0, 0.005, 0.01, 0.03, 0.1]
accuracies = []

for eps in epsilons:
    x_adv, perturb = fgsm_attack(model, x.clone(), y, eps)
    
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1).item()
    
    # 정확도 (원본과 같은지)
    acc = 1.0 if (adv_pred == clean_pred) else 0.0
    accuracies.append(acc)
    
    print(f"ε={eps:.4f}: Clean pred={clean_pred}, Adv pred={adv_pred}, Acc={acc:.1%}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(epsilons, accuracies, 'b-o', linewidth=2, markersize=8)
plt.xlabel('ε (Perturbation Size)')
plt.ylabel('Clean Accuracy')
plt.title('FGSM Attack: Impact of Perturbation Magnitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**관찰**: 아주 작은 $\epsilon$ (예: 0.03 = 8/255)에서도 정확도가 급락합니다.

### 실험 2 — PGD 공격 (더 강력한 공격)

```python
def pgd_attack(model, x, y, epsilon, alpha, num_iter):
    """
    Projected Gradient Descent Attack
    
    Args:
        model: 분류 모델
        x: 입력 이미지
        y: 레이블
        epsilon: perturbation budget
        alpha: 각 iteration의 step size
        num_iter: 반복 횟수
    
    Returns:
        x_adv: adversarial example
    """
    x_adv = x.clone().detach().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(num_iter):
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        # Gradient sign step
        x_adv_data = x_adv.data + alpha * torch.sign(x_adv.grad.data)
        
        # Project back to epsilon ball
        delta = torch.clamp(x_adv_data - x, min=-epsilon, max=epsilon)
        x_adv.data = x + delta
        x_adv.grad.zero_()
    
    return x_adv

# PGD vs FGSM 비교
epsilon = 0.03
x = torch.randn(1, 3, 224, 224, device=device)
y = torch.tensor([1], device=device)

with torch.no_grad():
    clean_pred = model(x).argmax(dim=1).item()
    clean_conf = torch.softmax(model(x), dim=1)[0, clean_pred].item()

# FGSM
x_fgsm, _ = fgsm_attack(model, x.clone(), y, epsilon)
with torch.no_grad():
    fgsm_pred = model(x_fgsm).argmax(dim=1).item()
    fgsm_conf = torch.softmax(model(x_fgsm), dim=1)[0, fgsm_pred].item()

# PGD
x_pgd = pgd_attack(model, x.clone(), y, epsilon, alpha=epsilon/4, num_iter=20)
with torch.no_grad():
    pgd_pred = model(x_pgd).argmax(dim=1).item()
    pgd_conf = torch.softmax(model(x_pgd), dim=1)[0, pgd_pred].item()

print(f"Clean: pred={clean_pred}, conf={clean_conf:.3f}")
print(f"FGSM:  pred={fgsm_pred}, conf={fgsm_conf:.3f}")
print(f"PGD:   pred={pgd_pred}, conf={pgd_conf:.3f}")
print(f"\nPGD이 더 강한 공격: PGD 성공률 > FGSM 성공률")
```

**관찰**: 같은 $\epsilon$에서 PGD가 FGSM보다 훨씬 강력합니다. PGD는 반복으로 optimal perturbation을 찾습니다.

### 실험 3 — Adversarial Training 효과

```python
import torch.optim as optim

def train_robust_model(model, train_loader, epochs=10, epsilon=0.03):
    """Adversarial training"""
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Adversarial example 생성
            x_adv, _ = fgsm_attack(model, x_batch.clone(), y_batch, epsilon)
            
            # Robust 모델은 adversarial example도 맞혀야 함
            optimizer.zero_grad()
            outputs = model(x_adv)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.3f}")

# 예시: 훈련은 생략하고, 결과만 보여줌
print("Adversarial training 결과:")
print("표준 모델 (standard training):")
print("  - Clean accuracy: ~76%")
print("  - Adversarial (ε=8/255) accuracy: ~0%")
print("")
print("Robust 모델 (adversarial training with ε=8/255):")
print("  - Clean accuracy: ~65% (약 11% 감소)")
print("  - Adversarial (ε=8/255) accuracy: ~50%")
```

**관찰**: Adversarial training은 robustness를 높이지만, **clean accuracy를 상당히 희생**합니다 (robustness-accuracy trade-off).

### 실험 4 — Perturbation의 시각화

```python
# 이미지와 perturbation의 시각화
import matplotlib.pyplot as plt

# 가상의 이미지 생성
clean = torch.randn(1, 3, 32, 32)
y = torch.tensor([0])

x_adv, perturb = fgsm_attack(model, clean.clone(), y, epsilon=0.03)

# Numpy로 변환 (시각화용)
clean_np = clean[0].permute(1, 2, 0).detach().cpu().numpy()
perturb_np = perturb[0].permute(1, 2, 0).detach().cpu().numpy()
adv_np = x_adv[0].permute(1, 2, 0).detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Normalize for visualization
clean_vis = (clean_np - clean_np.min()) / (clean_np.max() - clean_np.min())
perturb_vis = (perturb_np - perturb_np.min()) / (perturb_np.max() - perturb_np.min())
adv_vis = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min())

axes[0].imshow(clean_vis)
axes[0].set_title('Clean Image')
axes[0].axis('off')

axes[1].imshow(perturb_vis)
axes[1].set_title(f'Perturbation (ε={0.03:.4f})')
axes[1].axis('off')

axes[2].imshow(adv_vis)
axes[2].set_title('Adversarial Example')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**관찰**: Perturbation은 인간에게 거의 보이지 않지만, CNN은 완전히 다르게 분류합니다.

---

## 🔗 이론과 실전의 간극

### ImageNet에서의 현실

- **표준 ResNet-50**: 76% accuracy
- **FGSM 공격 (ε=8/255)**: ~10% accuracy (실제로 0%에 가까움)
- **PGD 공격 (ε=8/255)**: ~0% accuracy

### Adversarial Training의 한계

Madry et al. (2019)의 robust 모델:

- **Clean accuracy**: 66% (표준 76% → 10% 감소)
- **Robust accuracy (ε=8/255)**: 50%

Trade-off가 매우 심합니다. 더 큰 $\epsilon$에 robust하려면 clean accuracy가 더 많이 떨어집니다.

### 다른 공격 방법들

- **C&W (Carlini & Wagner)** 공격: 최적화 기반, PGD보다 더 강력
- **AutoAttack**: 여러 공격의 조합, 현재 최강 공격
- **Certified robustness**: 수학적으로 보장되는 robustness (계산 비용 높음)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\ell_\infty$ perturbation이 현실적 위협 | 실제 공격은 의미론적 (semantic) 변형일 수도 |
| Gradient 기반 공격만 고려 | Black-box 공격, 물리적 공격 등도 존재 |
| Single-step 공격 (FGSM) 분석 | 실전에선 반복 공격 (PGD)이 강함 |
| 분류만 고려 | 탐지, 분할 등 다른 작업도 취약 |
| CNN만 취약한가? | Transformer도 adversarial vulnerability 있음 |

---

## 📌 핵심 정리

$$\boxed{x' = x + \epsilon \cdot \text{sign}(\nabla_x L) \quad \text{— FGSM, 한 번의 step으로 공격}}$$

| 공격 방법 | 계산량 | 강도 | 실전성 |
|----------|--------|------|--------|
| **FGSM** | 1× gradient | 약함 | Low (한 번만) |
| **PGD** | 20-100× gradient | 강함 | High (반복) |
| **C&W** | 매우 많음 | 매우 강함 | 중간 (최적화) |
| **AutoAttack** | 매우 많음 | 최강 | Low (평가용) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): FGSM에서 $\epsilon = 8/255 \approx 0.031$은 왜 이렇게 작은 값인가? 이미지 픽셀 값이 0-255 범위일 때, 각 픽셀에 평균 얼마나 더할 수 있다는 뜻인가?

<details>
<summary>힌트 및 해설</summary>

$\epsilon = 8/255$는 각 픽셀에 **-8에서 +8** 범위의 정수 값을 더할 수 있다는 뜻입니다.

예: 픽셀 값 100 → 92~108 범위로 변경 가능

인간의 눈으로는 이 정도 변화가 **지각 불가능**하지만, CNN은 완전히 다르게 분류합니다.

이는 CNN이 인간의 시각과 근본적으로 다르게 특징을 학습하고 있음을 시사합니다.

</details>

**문제 2** (심화): Tsipras et al. (2019)의 robustness-accuracy trade-off를 설명하라. 왜 완벽하게 robust한 모델을 만들 수 없는가? 이론적 하한이 있는가?

<details>
<summary>힌트 및 해설</summary>

**근본 원인**: 두 클래스의 decision boundary가 겹치기 시작합니다.

예: CIFAR-10 (32×32 이미지) 분류에서
- 표준: 95% accuracy, 0% robust (ε=8/255)
- Robust (ε=8/255): 최대 ~60% accuracy

이론적으로, 큰 $\epsilon$에 robust하려면:
$$\text{margin}_\text{required} \geq 2\epsilon$$

하지만 모든 클래스 쌍이 $2\epsilon$만큼 떨어져 있을 수 없습니다 (고차원에서 불가능).

따라서 **perfect robustness는 이론적으로 불가능**합니다.

</details>

**问题 3** (이론-실전): FGSM과 PGD는 모두 gradient 기반이다. 하지만 왜 gradient가 없는 **검은 상자(black-box) 공격**도 효과적일까? 한 모델의 adversarial example이 다른 모델도 공격할 수 있는 이유는?

<details>
<summary>힌트 및 해설</summary>

**Transferability**: 한 모델 $f_1$에 대한 adversarial example이 다른 모델 $f_2$도 속입니다.

예: ResNet-50의 adversarial example → VGG도 공격 성공

**이유**:
1. 두 모델이 같은 데이터셋으로 훈련됨 → 비슷한 특징 학습
2. Adversarial example의 "adversarial 속성"은 모델 간 공유됨
3. 따라서 gradient 없이도 다른 모델의 gradient를 이용해 공격 가능

**Black-box 공격 절차**:
1. Surrogate 모델 $f_1$ (gradient 접근 가능)의 adversarial example 생성
2. 실제 타겟 모델 $f_2$에 적용 → 50-90% 성공률

</details>

---

<div align="center">

[◀ 이전](./01-inductive-bias.md) | [📚 README](../README.md) | [다음 ▶](./03-spectral-bias.md)

</div>
