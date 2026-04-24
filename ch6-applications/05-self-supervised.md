# 05. 자기지도학습 (Self-Supervised Learning) with CNNs

## 🎯 핵심 질문

- Self-supervised learning은 왜 CNN representation 학습의 새로운 패러다임인가?
- Pretext task(Jigsaw puzzle, rotation prediction, colorization)는 어떻게 label 없이도 유용한 feature를 학습하는가?
- SimCLR (Chen et al., 2020)의 contrastive loss $L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$는 어떤 원리로 작동하는가?
- Strong augmentation이 self-supervised learning에서 왜 필수적인가?
- MoCo (He et al., 2019)의 momentum encoder와 queue는 batch size 의존성을 어떻게 제거하는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

ImageNet 사전학습은 오래도록 transfer learning의 표준이었습니다. 하지만 레이블된 대규모 데이터는 비용이 많이 들고, 새로운 도메인(의료, 위성, 산업)에선 부족합니다. **Self-supervised learning**은 **라벨 없이** CNN 백본을 학습하여, 이 문제를 해결합니다. SimCLR, MoCo, BYOL 등은 ImageNet 사전학습과 비슷한 성능을 달성했고, 현재는:
- Foundation models (CLIP, DINO) 기반
- Contrastive learning의 표준 접근
- Domain shift에 더 robust한 feature

이 문서에서는 pretext task에서 contrastive learning으로 발전하는 흐름을 추적합니다.

---

## 📐 수학적 선행 조건

- [Ch3-Optimization](../ch3-optimization/): Gradient descent, batch processing
- [Ch6-01 Classification](./01-classification.md): Cross-entropy, softmax
- 정보이론: Entropy, mutual information, KL divergence
- 선형대수: Cosine similarity, normalization

---

## 📖 직관적 이해

### Pretext Task 시대 (2015-2019)

Labeled data 없이 CNN을 학습하는 방법:

**1. Jigsaw Puzzle (Noroozi & Favaro, 2016)**
- 이미지를 $3 \times 3$ 또는 $4 \times 4$ grid로 나눔
- Patch 순서를 섞음 (permutation)
- 네트워크가 원래 순서를 복원하도록 학습
- 효과: 객체의 geometric understanding 학습

**2. Rotation Prediction (Gidaris et al., 2018)**
- 이미지를 $\{0°, 90°, 180°, 270°\}$로 회전
- 네트워크가 회전 각도를 예측
- 효과: 방향 정보, semantic structure 학습

**3. Colorization (Zhang et al., 2016)**
- 흑백 이미지가 입력, 칼라 이미지를 예측
- Lab color space에서 ab 채널만 예측
- 효과: 객체의 appearance, texture 학습

**문제**: Pretext task 성능과 downstream task 성능이 일치하지 않음 (task discrepancy).

### Contrastive Learning — SimCLR

**핵심 아이디어** (Chen et al., 2020):
- Pretext task 대신 **instance discrimination**: 같은 이미지의 서로 다른 augmentation은 가깝게, 다른 이미지는 멀게 학습.

**아키텍처**:
```
Image x
  ↓ augment (random crop, color jitter, Gaussian blur, ...)
  ↓ encoder (ResNet-50)
  ↓ projection head (2-layer MLP)
  ↓ ℓ2 normalize
  ↓ z (representation)
  
Contrastive loss: 같은 이미지 2개 augmentation과 배치의 다른 이미지들 비교
```

**Contrastive Loss** (NT-Xent = Normalized Temperature-scaled Cross Entropy):
$$L_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

여기서:
- $z_i, z_j$: 같은 이미지 $x$ 의 두 augmentation의 embedding
- $z_k, k \neq i$: 배치 내 다른 이미지들의 embedding
- $\text{sim}(a, b) = a^\top b / (|a||b|)$: cosine similarity
- $\tau$: temperature parameter (보통 0.07)

**효과**: 배치 크기에 크게 의존 (큰 배치 = 더 나은 성능).

### MoCo — Momentum Contrast

**문제**: SimCLR은 큰 배치가 필수 (4096 이상). 메모리 제약이 있는 환경에서 어려움.

**해결** (He et al., 2019): Momentum encoder + 큐(queue) 메모리.

```
Image x (query)
  ↓ augment
  ↓ encoder (fast update)
  ↓ q (query representation)

Image x' (key)
  ↓ augment
  ↓ momentum encoder (slow update, EMA)
  ↓ k (key representation)

Memory queue: 이전 배치들의 k들을 저장
Contrastive loss: q와 같은 이미지의 k + 배치 내 다른 k들 비교
```

**Momentum encoder update**:
$$\theta_m \gets \alpha \theta_m + (1 - \alpha) \theta$$

여기서 $\alpha \approx 0.999$ (very slow update, EMA).

**효과**:
- 작은 배치 크기에서도 작동 (256)
- 큐에 수만 개 negative sample 저장
- SimCLR 대비 메모리 효율적 (1/4)

### BYOL — Bootstrap Your Own Latent

**혁신**: Contrastive loss 제거!

```
Image x → augment → encoder → projection → predictor → p
Image x → augment → momentum encoder (no grad) → projection → ẑ

MSE loss: ||p - ẑ||²
```

Momentum encoder와 EMA만으로 collapse 방지. Contrastive pair 필요 없음.

### Transfer Learning — CNN Feature의 범용성

사전학습 후 downstream task에 fine-tuning:

**Linear evaluation protocol**:
1. CNN backbone만 고정
2. 분류 head만 학습 (몇 epoch)
3. 성능 평가

이 지표는 **label efficiency** 측정: 라벨 몇 개로 좋은 성능 달성하는가?

---

## ✏️ 엄밀한 정의·정리

### 정의 6.25 — Self-Supervised Learning

라벨 없이 데이터 자체에서 supervision signal을 생성:
$$\text{supervision signal} = f_{\text{pretext}}(\mathcal{D})$$

예: 회전 각도 예측, 색상 복원, 순서 복원 등.

### 정의 6.26 — Pretext Task

원래 관심 task와 무관하게 모양이지만, 유용한 representation을 유도하는 intermediate task.

### 정의 6.27 — Contrastive Learning

두 샘플 간 거리를 최소화/최대화하는 metric learning:
$$L = -\log \frac{p(\text{similar})}{\sum_k p(\text{similar} \text{ or } \text{dissimilar}_k)}$$

### 정의 6.28 — SimCLR Loss (NT-Xent)

배치 크기 $N$, 각 이미지 2개 augmentation (총 $2N$ samples):
$$L = \frac{1}{2N} \sum_{i=1}^{2N} l_i$$
$$l_i = -\log \frac{\exp(\text{sim}(z_i, z_{i'})/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

### 정의 6.29 — Cosine Similarity

정규화된 벡터 $u, v$ (||u|| = ||v|| = 1):
$$\text{sim}(u, v) = u^\top v$$

### 정의 6.30 — Exponential Moving Average (EMA)

매개변수 $\theta_m$ (momentum):
$$\theta_m \gets \alpha \theta_m + (1-\alpha) \theta_{\text{current}}$$

$\alpha$ 크면 느린 업데이트, 작으면 빠른 업데이트.

### 정의 6.31 — Linear Evaluation Protocol

사전학습 CNN backbone $f$에 대해:
1. Feature freeze: $f$ 고정
2. Linear classifier 학습: $y = W f(x)$ (random init)
3. Downstream task 정확도 평가

CNN feature의 "quality" 측정.

---

## 🔬 증명 또는 수학적 유도

### Contrastive Loss의 직관

NT-Xent loss를 다시 쓰면:
$$L = -\log \frac{\exp(s_{i,i'}/\tau)}{Z}$$

여기서 $Z = \sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(s_{i,k}/\tau)$ (partition function, softmax의 분모).

이는 **logistic regression**과 유사:
$$L = -\log p(\text{similar} | \text{pair})$$

확률 관점:
$$p(\text{similar} | z_i) = \frac{\exp(s_{i,i'}/\tau)}{Z}$$

Temperature $\tau$:
- $\tau \to 0$: sharp distribution, easy negatives 무시
- $\tau \to \infty$: uniform distribution, 모든 pair 동등

### Momentum Encoder의 안정성

기본 contrastive learning (SimCLR):
$$\min_{\theta_q, \theta_k} L(z_q(\theta_q), z_k(\theta_k))$$

문제: $\theta_q$와 $\theta_k$를 동시 업데이트하면, representation이 collapse 가능.

Momentum encoder (MoCo):
$$\theta_k \gets \alpha \theta_k + (1-\alpha) \theta_q$$

$\theta_k$는 천천히 업데이트되므로, 충분한 다양성 유지 → collapse 방지.

### Strong Augmentation의 필요성

Weak augmentation (small crop, minor color jitter):
- 같은 이미지의 2개 augmentation이 매우 유사
- 모델이 trivial solution (예: 밝기만 배우기) 학습 가능
- Downstream task와 관계 없는 feature

Strong augmentation (random crop 25%-100%, aggressive color jitter, Gaussian blur):
- Augmentation 후에도 semantic 일관성 유지 필수
- 모델이 **semantic-level** feature 학습 강제
- Downstream transfer 성능 향상

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — SimCLR Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        z_i, z_j: [batch_size, projection_dim]
        normalized vectors (||z|| = 1)
        """
        batch_size = z_i.shape[0]
        
        # Cosine similarity
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, proj_dim]
        similarity = torch.matmul(z, z.T) / self.temperature
        
        # Mask: 자기 자신과 positive pair 제거
        mask = torch.eye(2 * batch_size, dtype=torch.bool)
        # Positive pairs: (i, i+batch) and (i+batch, i)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True
        
        # Loss 계산
        pos = torch.diag(similarity, batch_size)  # positive pairs
        neg = similarity[~mask].reshape(2 * batch_size, -1)
        
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long)
        
        loss = F.cross_entropy(logits, labels)
        return loss

# 테스트
batch_size = 32
proj_dim = 128

z_i = F.normalize(torch.randn(batch_size, proj_dim), dim=1)
z_j = F.normalize(torch.randn(batch_size, proj_dim), dim=1)

loss_fn = SimCLRLoss(temperature=0.07)
loss = loss_fn(z_i, z_j)

print(f"SimCLR Loss: {loss.item():.4f}")
print(f"Expected range: 3-5 (log(256) ≈ 5.55)")
```

### 실험 2 — Strong Augmentation 비교

```python
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
img = Image.new('RGB', (224, 224), color='red')

# Weak augmentation
weak_transform = T.Compose([
    T.RandomCrop(200),
    T.Resize((224, 224)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
])

# Strong augmentation (SimCLR style)
strong_transform = T.Compose([
    T.RandomResizedCrop((224, 224), scale=(0.25, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)], p=0.8),
    T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.1),
    T.ToTensor(),
])

# 시각화
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Weak augmentation
for i in range(5):
    aug_img = weak_transform(img)
    axes[0, i].imshow(aug_img.permute(1, 2, 0))
    axes[0, i].set_title('Weak')
    axes[0, i].axis('off')

# Strong augmentation
for i in range(5):
    aug_img = strong_transform(img)
    axes[1, i].imshow(aug_img.permute(1, 2, 0))
    axes[1, i].set_title('Strong')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

### 실험 3 — MoCo 구현

```python
import torch
import torch.nn as nn
import math

class MomentumContrast(nn.Module):
    def __init__(self, backbone, projection_dim=128, hidden_dim=2048,
                 queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        
        # Encoders
        self.encoder_q = nn.Sequential(
            backbone,
            nn.Linear(backbone.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # momentum encoder (copy)
        self.encoder_k = nn.Sequential(
            backbone,
            nn.Linear(backbone.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Copy weights
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Register queue
        self.register_buffer('queue', torch.randn(projection_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """EMA update momentum encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[1]
        
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:, :remaining]
            self.queue[:, :batch_size - remaining] = keys[:, remaining:]
        
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    
    def forward(self, x_q, x_k):
        """
        x_q, x_k: augmented versions of same image
        """
        # Query
        q = self.encoder_q(x_q)  # [B, C]
        q = nn.functional.normalize(q, dim=1)
        
        # Key
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)  # [B, C]
            k = nn.functional.normalize(k, dim=1)
        
        # Contrast with queue
        l_pos = torch.sum(q * k, dim=1, keepdim=True)  # [B, 1]
        l_neg = torch.matmul(q, self.queue.clone().detach())  # [B, K]
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(k.T)
        
        return loss

# 간단한 backbone
class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 64)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return x

# 테스트
backbone = SimpleBackbone()
moco = MomentumContrast(backbone, projection_dim=128, queue_size=256)

x_q = torch.randn(32, 3, 224, 224)
x_k = torch.randn(32, 3, 224, 224)

loss = moco(x_q, x_k)
print(f"MoCo Loss: {loss.item():.4f}")
```

---

## 🔗 이론과 실전의 간립

### Pretext Task의 한계

이론: 특정 task 학습 → 일반적 representation.

실전: Task discrepancy 문제.
- Jigsaw: 틀린 순서도 semantic하게 합리적일 수 있음
- Rotation: 회전 불변 이미지(하늘, 텍스처)에서 실패
- 모두 downstream task와 direct 연관 없음

### SimCLR의 메모리 요구

이론: Large batch size (4096)에서 최고 성능.

실전: 메모리 제약이 있는 환경(모바일, edge device).

해결: Gradient accumulation, MoCo, BYOL 등.

### Transfer Learning의 문제

이론: CNN feature는 범용적 (ImageNet pretrain).

실전:
- Domain shift (자율주행 → 의료 이미지)에 취약
- Task 특성에 맞는 augmentation 중요
- Self-supervised feature가 특정 domain에 더 robust

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 같은 이미지의 augmentation = semantic consistent | 극단적 augmentation은 정보 손실 |
| Augmentation은 invariant하지만 discriminative | 모든 downstream task에 맞는 augmentation 없음 |
| Large batch size 필수 (SimCLR) | 메모리 제약 환경에서 어려움 |
| Momentum encoder collapse 방지 (MoCo) | 실제로는 BYOL도 동작 (collapse 원인 미해결) |
| Self-supervised = downstream task에 도움 | 일부 task(매우 simple classification)에선 불필요 |

---

## 📌 핵심 정리

$$\boxed{L_{\text{SimCLR}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)} \text{ — contrastive instance discrimination}}$$

| 개념 | 정의 |
|------|------|
| **Pretext task** | 라벨 없이 supervision signal 생성 |
| **Jigsaw puzzle** | 섞인 patch 순서 복원 |
| **Rotation prediction** | 회전 각도 예측 |
| **Colorization** | 흑백→칼라 변환 |
| **Contrastive learning** | 같은 샘플 가깝게, 다른 샘플 멀게 |
| **SimCLR** | Large batch (4096) + strong augmentation |
| **NT-Xent loss** | Temperature-scaled cross-entropy |
| **MoCo** | Momentum encoder + queue, batch size 독립적 |
| **BYOL** | Predictor + momentum encoder, contrastive loss 제거 |
| **Linear evaluation** | 사전학습 feature quality 측정 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): SimCLR loss에서 배치 크기 $N=256$일 때, positive pair와 negative pair의 개수는?

<details>
<summary>힌트 및 해설</summary>

각 이미지 2개 augmentation → 총 $2N = 512$ 샘플.

각 샘플 $i$에 대해:
- Positive pair: 1개 (같은 이미지의 다른 augmentation)
- Negative pairs: $2N - 2 = 510$ (자신과 positive 제외)

배치 크기가 크수록 negative pair 많아짐 → contrastive signal 강함.

</details>

**문제 2** (심화): MoCo의 momentum encoder에서 $\alpha = 0.999$일 때, 100 step 후 $\theta_k$는 $\theta_q$의 몇 %를 반영하는가?

<details>
<summary>힌트 및 해설</summary>

$$\theta_k^{(t)} = \alpha^t \theta_k^{(0)} + (1-\alpha) \sum_{s=0}^{t-1} \alpha^s \theta_q^{(s)}$$

$\alpha = 0.999$이므로 $\alpha^{100} = 0.999^{100} \approx 0.9048$.

따라서 100 step 후, 초기값의 약 90%는 유지, 10% 정도만 새로운 $\theta_q$ 반영.

$\alpha$가 크면 momentum이 크고, 천천히 업데이트.

</details>

**문제 3** (이론-실전): Self-supervised learning이 ImageNet pretrain을 대체할 수 있는가? 언제는 가능하고, 언제는 불가능한가?

<details>
<summary>힌트 및 해설</summary>

**가능한 경우**:
- 대규모 unlabeled data 있음 (Web-scale images)
- Task가 semantic-level (분류, detection, segmentation)
- 충분한 compute 자원

**불가능한 경우**:
- Labeled data만 있음 (ImageNet-scale은 비용)
- Task가 매우 specific (특정 산업 이미지)
- 계산 자원 제한 (엣지 디바이스)

현대: **최적 조합**: ImageNet pretrain + domain-specific unlabeled data로 self-supervised 추가학습.

</details>

---

<div align="center">

[◀ 이전](./04-segmentation.md) | [📚 README](../README.md) | [다음 ▶](../ch7-limits-vit/01-inductive-bias.md)

</div>
