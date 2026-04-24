# 01. 이미지 분류 (Image Classification) — Pipeline과 최신 기법

## 🎯 핵심 질문

- ImageNet과 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)는 왜 CNN의 발전을 주도했는가?
- Cross-entropy loss에서 softmax의 gradient가 $p - y$로 단순해지는 수학적 이유는?
- Label smoothing $y_c' = (1-\epsilon)y_c + \epsilon/K$는 왜 모델의 calibration 성능을 높이는가?
- MixUp과 CutMix 같은 data augmentation이 정규화(regularization) 관점에서 어떤 효과를 갖는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

이미지 분류는 CNN의 가장 기본적이면서 가장 강력한 응용입니다. ImageNet ILSVRC(2012-2017)는 AlexNet(2012)부터 최근의 Vision Transformer까지 모든 주요 CNN 아키텍처의 성능 기준을 제시했습니다. 이 문서에서는 **분류의 수학적 기초(cross-entropy, softmax gradient)**, **label smoothing의 calibration 효과**, 그리고 **data augmentation의 정규화 이론**을 체계적으로 다룹니다. 이들은 다른 모든 CNN 응용(detection, segmentation)의 foundation입니다.

---

## 📐 수학적 선행 조건

- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Chain rule, softmax, cross-entropy 미분
- [Statistics & Probability Deep Dive](https://github.com/iq-ai-lab/statistics-probability-deep-dive): KL divergence, entropy, expectation
- [Ch5-01 CNN Architecture](../ch5-modern-cnn/01-basic-cnn.md): Basic CNN layers
- 선형대수: One-hot encoding, probability simplex

---

## 📖 직관적 이해

### ImageNet과 ILSVRC의 역사적 역할

ImageNet(Deng et al., 2009)은 14백만 개의 손으로 레이블 된 이미지와 1,000개의 카테고리로 이루어진 대규모 dataset입니다. ILSVRC(2012-2017)는 연간 classification 챌린지를 개최했고:

- **2012**: AlexNet (Top-1 63.3%) — 기존 방법 대비 약 13% 향상, CNN 르네상스 시작
- **2014**: VGGNet (Top-1 72.7%), GoogLeNet (74.8%)
- **2015**: ResNet (76.2%) — deep 네트워크의 new paradigm

Top-1 accuracy와 Top-5 accuracy는 상위 1개, 5개 예측 중에 정답이 있는지 평가. Top-5는 데이터셋이 겹침이 많아서(예: 여러 개 개 종류) 더 관대한 지표.

### Cross-Entropy와 Softmax의 수학적 구조

분류 문제: $n$개 샘플, $K$개 클래스, $y_i \in \{0,1\}^K$ (one-hot), 예측 로짓 $z_i \in \mathbb{R}^K$.

**Softmax**:
$$p_c = \text{softmax}(z)_c = \frac{\exp(z_c)}{\sum_{k=1}^K \exp(z_k)}$$

**Cross-entropy loss**:
$$\ell_i = -\sum_{c=1}^K y_{ic} \log p_c(z_i) = -\log p_{y_i}(z_i)$$
(여기서 $y_i$는 정답 클래스)

**전체 손실**: $L = \frac{1}{n}\sum_{i=1}^n \ell_i$

### Softmax-CrossEntropy Gradient의 단순성

놀랍게도, softmax-cross-entropy 조합의 gradient는:
$$\frac{\partial \ell}{\partial z_c} = p_c - y_c$$

이는 매우 깔끔한 형태로, 실제로는 "예측과 실제의 확률 차이"를 의미합니다. 이렇게 단순해지는 이유는 softmax의 특수한 구조 때문입니다.

### Label Smoothing의 정규화 효과

실제 label은 one-hot이므로 $y_c = 1$ (정답) 또는 0 (오답). 모델은 정답 클래스에 매우 큰 logit을 부여하려고 하고, 이는 **overconfidence**를 초래합니다.

**Label smoothing** (Szegedy et al., 2016):
$$y_c' = (1-\epsilon) y_c + \frac{\epsilon}{K}$$

예: $K=1000$, $\epsilon=0.1$일 때:
- 정답 클래스: $y_c' = 0.9 \cdot 1 + 0.1/1000 = 0.9001$
- 오답 클래스: $y_c' = 0.9 \cdot 0 + 0.1/1000 = 0.0001$

효과:
1. **Calibration 개선**: 예측 확률이 실제 accuracy와 더 일치
2. **Regularization**: 모든 클래스에 작은 가중치를 주어 overfit 억제

### MixUp과 CutMix — Data Augmentation

**MixUp** (Zhang et al., 2017): 두 샘플을 선형 조합
$$\tilde x = \lambda x_i + (1-\lambda) x_j, \quad \tilde y = \lambda y_i + (1-\lambda) y_j$$
여기서 $\lambda \sim \text{Beta}(\alpha, \alpha)$, 보통 $\alpha=1$.

**CutMix** (Yun et al., 2019): 이미지의 일부를 교환
- 두 이미지에서 random region을 선택하여 한 이미지에 붙임
- 라벨은 교환된 면적의 비율로 혼합

**정규화 관점**: 이들 기법은 선형 interpolation 또는 local mixing을 통해 decision boundary를 부드럽게 만들어, 모델의 일반화 능력을 높입니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.1 — Softmax 함수

로짓 벡터 $z = (z_1, \ldots, z_K)$에 대해:
$$\sigma(z)_c = \frac{\exp(z_c)}{\sum_{k=1}^K \exp(z_k)}$$

성질:
1. $\sigma(z)_c > 0$ for all $c$
2. $\sum_c \sigma(z)_c = 1$ (확률 분포)
3. $\sigma(z + c \mathbf{1}) = \sigma(z)$ (scale invariance)

### 정의 6.2 — Cross-Entropy Loss

$y \in \{0,1\}^K$ (one-hot), $p = \sigma(z)$:
$$\ell_{\text{CE}}(y, p) = -\sum_c y_c \log p_c = -\log p_{y_\text{true}}$$

### 정리 6.3 — Softmax-CrossEntropy 최적화 성질

임의의 손실 함수 $\ell(z, y)$에 대해:
$$\frac{\partial}{\partial z_c}[-\log \sigma(z)_c] = \sigma(z)_c - 1$$
$$\frac{\partial}{\partial z_c}[-\sum_k y_k \log \sigma(z)_k] = \sigma(z)_c - y_c$$

**증명 스케치**: Softmax의 미분이 $\frac{\partial \sigma_c}{\partial z_d} = \sigma_c (\delta_{cd} - \sigma_d)$이므로, chain rule로 정리하면 위 결과를 얻습니다. $\square$

### 정의 6.4 — Label Smoothing

원본 one-hot label $y$에 대해, smoothed label:
$$y_{\text{smooth}}^c = (1-\epsilon) y^c + \frac{\epsilon}{K}$$

여기서 $\epsilon \in (0, 1)$은 hyperparameter (보통 0.1).

### 정의 6.5 — 모델 Calibration

예측 확률 $\hat p$와 실제 정확도 $\text{acc}$가 일치하는 정도:
$$\text{ECE} = \frac{1}{n} \sum_i |\hat p_i - \mathbb{1}[\text{prediction}_i = \text{true}_i]|$$
낮을수록 better calibrated.

### 정의 6.6 — MixUp 데이터 증강

$$(\tilde x, \tilde y) = (\lambda x_i + (1-\lambda) x_j, \lambda y_i + (1-\lambda) y_j)$$
$\lambda \sim \text{Beta}(\alpha, \alpha)$.

### 정의 6.7 — CutMix 데이터 증강

두 이미지 $x_i, x_j$와 binary mask $M$에 대해:
$$\tilde x = x_i \odot M + x_j \odot (1-M)$$
$$\tilde y = \frac{|M|}{HW} y_i + \frac{|1-M|}{HW} y_j$$
여기서 $|M|$은 mask의 pixel 개수.

---

## 🔬 증명 또는 수학적 유도

### Softmax-CrossEntropy 미분 유도

Cross-entropy loss:
$$L = -\log p_c = -\log \frac{\exp(z_c)}{\sum_k \exp(z_k)} = -z_c + \log \sum_k \exp(z_k)$$

Softmax $\sigma_d = \frac{\exp(z_d)}{\sum_k \exp(z_k)}$의 미분:
$$\frac{\partial \sigma_d}{\partial z_c} = \delta_{cd} \sigma_d - \sigma_c \sigma_d = \sigma_d(\delta_{cd} - \sigma_c)$$

따라서:
$$\frac{\partial L}{\partial z_c} = -\delta_{cd} + \frac{\exp(z_c)}{\sum_k \exp(z_k)} = -\delta_{cd} + \sigma_c$$

다중 클래스 (one-hot $y$):
$$\frac{\partial}{\partial z_c}\left[-\sum_k y_k \log \sigma_k\right] = -\sum_k y_k \frac{\partial \log \sigma_k}{\partial z_c} = -y_c \frac{1}{\sigma_c} \frac{\partial \sigma_c}{\partial z_c}$$
$$= -y_c \sigma_c(\delta_{cd} - \sigma_d) / \sigma_c = \sigma_c - y_c$$

### Label Smoothing의 Calibration 개선

원본 one-hot label $y$에 대한 cross-entropy:
$$L_{\text{orig}} = -\log p_c$$

이는 모델이 정답에 무한히 큰 확률을 부여하도록 incentivize. 반대로, smoothed label:
$$L_{\text{smooth}} = -\sum_c y_c^{\text{smooth}} \log p_c = -(1-\epsilon) \log p_c - \frac{\epsilon}{K} \sum_k \log p_k$$

두 번째 항 $\sum_k \log p_k$는 **entropy 항**으로, 모든 클래스에 일정 확률을 유지하도록 강제하여 overconfidence 방지. 이는 KL divergence를 정규화하는 효과.

### MixUp 정규화 관점

Mixup에서 $\tilde y = \lambda y_i + (1-\lambda) y_j$는 label space에서의 linear interpolation. 이는 다음을 의미:

Loss:
$$L_{\text{MixUp}} = \lambda L(f(\tilde x), y_i) + (1-\lambda) L(f(\tilde x), y_j)$$

직관: 두 다른 클래스의 "사이"에 있는 샘플들을 학습함으로써, decision boundary를 부드럽게 만들고, 모델이 극단적인 예측을 피하도록 유도.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Softmax-CrossEntropy Gradient 검증

```python
import torch
import torch.nn.functional as F

# 배치 크기 4, 클래스 10
batch_size, num_classes = 4, 10
z = torch.randn(batch_size, num_classes, requires_grad=True)
y = torch.tensor([2, 5, 7, 1])  # 정답 클래스

# 방법 1: 직접 softmax + log
p = F.softmax(z, dim=1)
loss_manual = -torch.log(p[torch.arange(batch_size), y]).mean()

# 방법 2: PyTorch의 CrossEntropyLoss (내부적으로 softmax 포함)
loss_fn = torch.nn.CrossEntropyLoss()
loss_pytorch = loss_fn(z, y)

print(f"Manual loss: {loss_manual.item():.6f}")
print(f"PyTorch loss: {loss_pytorch.item():.6f}")

# Gradient 확인
loss_pytorch.backward()
p_after = F.softmax(z.detach(), dim=1)
p_after[torch.arange(batch_size), y] -= 1  # p - y
expected_grad = p_after / batch_size

print(f"\nGradient norm: {z.grad.norm().item():.6f}")
print(f"Expected (p-y)/batch_size norm: {expected_grad.norm().item():.6f}")
```

**예상 출력**: 두 손실이 거의 같으며, gradient는 $(p - y) / \text{batch\_size}$와 일치.

### 실험 2 — Label Smoothing의 Calibration 효과

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# 합성 데이터: 100개 샘플, 10개 클래스
torch.manual_seed(42)
np.random.seed(42)

def train_with_label_smoothing(epsilon=0.0, num_epochs=50):
    model = torch.nn.Linear(20, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X = torch.randn(100, 20)
    y_true = torch.randint(0, 10, (100,))
    
    for epoch in range(num_epochs):
        z = model(X)
        p = F.softmax(z, dim=1)
        
        if epsilon > 0:
            # Label smoothing
            y_smooth = torch.zeros(100, 10)
            y_smooth.fill_(epsilon / 10)
            y_smooth[torch.arange(100), y_true] = 1 - epsilon
            loss = -(y_smooth * torch.log(p + 1e-8)).mean()
        else:
            loss = F.cross_entropy(z, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calibration 평가
    with torch.no_grad():
        z_final = model(X)
        p_final = F.softmax(z_final, dim=1).numpy()
        pred_prob = p_final.max(axis=1)
        pred_class = p_final.argmax(axis=1)
        acc = (pred_class == y_true.numpy()).astype(float)
    
    return pred_prob, acc

epsilons = [0.0, 0.1, 0.2]
colors = ['blue', 'orange', 'green']

fig, axes = plt.subplots(1, len(epsilons), figsize=(15, 4))

for idx, (eps, color) in enumerate(zip(epsilons, colors)):
    pred_prob, acc = train_with_label_smoothing(epsilon=eps, num_epochs=100)
    
    prob_true, prob_pred = calibration_curve(acc, pred_prob, n_bins=10)
    
    ax = axes[idx]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(prob_pred, prob_true, 'o-', color=color, label=f'ε={eps}')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Label Smoothing ε={eps}')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()
```

**예상 출력**: $\epsilon > 0$일수록 calibration curve가 diagonal에 더 가까움.

### 실험 3 — MixUp 데이터 증강

```python
import torchvision.datasets as datasets
from torchvision import transforms

class MixUpDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]
        # 랜덤 다른 이미지 선택
        idx2 = torch.randint(0, len(self.dataset), (1,)).item()
        x2, y2 = self.dataset[idx2]
        
        # MixUp
        lam = np.random.beta(self.alpha, self.alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = (lam * y1, (1 - lam) * y2, lam)  # 라벨 저장 (학습 중 사용)
        
        return x_mix, y_mix

# CIFAR-10에 적용
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
mixup_dataset = MixUpDataset(cifar10, alpha=1.0)

# 배치 시각화
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i in range(4):
    x, (y1, y2, lam) in mixup_dataset[i]
    ax = axes[0, i]
    ax.imshow(x.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # unnormalize
    ax.set_title(f'λ={lam:.2f}, y1/y2={int(y1)}/{int(y2)}')
    ax.axis('off')

for i in range(4, 8):
    ax = axes[1, i-4]
    x, (y1, y2, lam) = mixup_dataset[i]
    ax.imshow(x.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    ax.set_title(f'λ={lam:.2f}, y1/y2={int(y1)}/{int(y2)}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 🔗 이론과 실전의 간극

### Top-1 vs Top-5 Accuracy의 선택

이론: Top-1이 더 엄격한 기준. Top-5는 dataset 자체의 모호함(예: 다양한 개 종류)을 반영.

실전: ImageNet 카테고리가 매우 세분화되어 (예: 120개 개 종류), Top-5는 semantically 비슷한 오류를 허용.

### Cross-Entropy의 한계

이론: 모든 오분류를 동등하게 취급.

실전: 일부 오분류는 더 심각(예: 개를 고양이로 vs 다른 개 종류로). Focal loss(Ch6-03)는 어려운 샘플에 더 큰 가중치.

### Data Augmentation의 이중성

이론: 정규화 효과로 일반화 능력 향상.

실전: 강한 augmentation이 항상 좋은 것은 아님. Self-supervised learning(Ch6-05)에서는 stronger augmentation 필요.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 클래스가 균형잡혀 있음 | 실제로 long-tail distribution (예: rare disease detection) |
| Softmax는 모든 클래스에 확률 할당 | 일부 응용에선 open-set recognition (미지의 클래스) 필요 |
| Label smoothing의 $\epsilon$ 고정 | 클래스별·샘플별로 다른 값이 더 좋을 수 있음 |
| MixUp/CutMix은 언제나 도움됨 | 일부 task(예: 의료 이미지)에서 과도한 mixing은 정보 손실 |
| ImageNet이 universal visual knowledge | Domain shift (새로운 distribution)에는 취약 |

---

## 📌 핵심 정리

$$\boxed{\frac{\partial L_{\text{CE}}}{\partial z_c} = p_c - y_c \text{ — softmax-cross-entropy의 우아한 구조}}$$

| 개념 | 정의 |
|------|------|
| **Softmax** | 로짓을 확률로: $\sigma(z)_c = \exp(z_c) / \sum_k \exp(z_k)$ |
| **Cross-entropy** | $L = -\sum_c y_c \log p_c$, one-hot이면 $L = -\log p_{\text{true}}$ |
| **Label smoothing** | $y_c' = (1-\epsilon) y_c + \epsilon/K$, calibration & regularization |
| **MixUp** | $\tilde x = \lambda x_i + (1-\lambda) x_j$, boundary smoothing |
| **CutMix** | 이미지 영역 교환, local structure 학습 |
| **ImageNet ILSVRC** | 1,000개 클래스, benchmark for CNN architecture progress |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 10개 클래스 분류에서 one-hot label $y = (0,1,0,\ldots,0)$에 대해, 예측 확률 $p = (0.05, 0.7, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01)$일 때 cross-entropy loss를 계산하라.

<details>
<summary>힌트 및 해설</summary>

Cross-entropy는 정답 클래스(인덱스 1)의 확률만 봄:
$$L = -\log p_1 = -\log(0.7) \approx 0.357$$

만약 정답이 아닌 클래스를 정답이라고 오분류했다면, 예를 들어 $p = (0.7, 0.05, 0.05, \ldots)$:
$$L = -\log(0.05) \approx 2.996$$

같은 오분류라도 confidence가 높을수록 loss가 더 큼.

</details>

**문제 2** (심화): Label smoothing $\epsilon=0.1$을 적용하면, 10개 클래스에서 정답 클래스와 오답 클래스의 smoothed label은 각각 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$$y_{\text{true}}' = (1-0.1) \cdot 1 + \frac{0.1}{10} = 0.9 + 0.01 = 0.91$$

$$y_{\text{false}}' = (1-0.1) \cdot 0 + \frac{0.1}{10} = 0 + 0.01 = 0.01$$

정답 클래스에도 여전히 높은 가중치(0.91)를 주지만, 오답 클래스들도 작지만 0이 아닌 가중치(0.01)를 받음. 이는 모델이 과도한 confidence를 피하도록 유도.

</details>

**문제 3** (이론-실전): MixUp에서 $\lambda \sim \text{Beta}(1, 1)$ (uniform)을 사용하는 이유는 무엇인가? 다른 분포은 어떤 효과를 갖겠는가?

<details>
<summary>힌트 및 해설</summary>

$\text{Beta}(1,1)$은 uniform distribution [0,1]. 즉, 두 샘플 간 모든 가중치가 동등하게 선택됨.

- $\alpha < 1$ (예: Beta(0.5, 0.5)): 극단 값(0 또는 1)에 가까운 혼합, 한쪽 샘플에 더 큰 가중치
- $\alpha > 1$ (예: Beta(2, 2)): 0.5 근처에 집중, 균형잡힌 혼합

실전: $\alpha=1$이 가장 일반적이고 robust. 극단 값은 너무 쉬운 샘플(한쪽에 치우침), 균형은 너무 어려운 샘플(완전히 새로운 입력).

</details>

---

<div align="center">

[◀ 이전](../ch5-modern-cnn/05-nas.md) | [📚 README](../README.md) | [다음 ▶](./02-two-stage-detection.md)

</div>
