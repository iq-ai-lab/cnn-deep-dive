# 03. 한 단계 객체 탐지 (One-Stage Detection) — YOLO, RetinaNet과 Focal Loss

## 🎯 핵심 질문

- YOLO (Redmon et al., 2016)는 어떻게 45 FPS의 실시간 탐지를 달성했는가?
- Focal Loss $FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t$는 왜 class imbalance를 해결하며, 특히 $(1-p_t)^\gamma$ 항이 중요한가?
- Easy example($p_t \to 1$)과 hard example($p_t \to 0$)에 대해 focal loss는 어떻게 다르게 작용하는가?
- $\gamma = 2$는 왜 가장 흔히 사용되는 hyperparameter이며, 다른 값들은 어떤 효과를 갖는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Two-stage detection(Ch6-02)은 정확하지만 느립니다(~200ms/이미지). 실시간 응용(자율주행, 로봇, 보안)에는 속도가 필수입니다. **One-stage detection**은 region proposal 단계를 건너뛰고, 직접 grid cell별로 객체를 예측합니다. YOLO의 혁신성은 이를 처음 제시했고, 그 후 **Focal Loss**(RetinaNet, Lin et al., 2017)는 one-stage 탐지의 가장 큰 문제인 **foreground-background class imbalance**를 우아하게 해결했습니다. 이는 detection과 segmentation의 핵심 기법입니다.

---

## 📐 수학적 선행 조건

- [Ch6-01 Classification](./01-classification.md): Cross-entropy loss, softmax
- [Ch6-02 Two-Stage Detection](./02-two-stage-detection.md): Bounding box regression, IoU, NMS
- [Ch3-04 Regularization](../ch3-optimization/04-regularization.md): Loss weighting, class imbalance
- 확률론: 확률 분포, weighted loss

---

## 📖 직관적 이해

### YOLO — You Only Look Once

**핵심 아이디어** (Redmon et al., 2016):
1. 이미지를 $S \times S$ grid로 나눔 (보통 $S=7$)
2. 각 grid cell에서 $B$개 bounding box 예측 (보통 $B=2$)
3. 각 box마다 objectness score + class probability

**출력**: $S \times S \times (B \cdot 5 + C)$ 텐서
- 각 box: $(x, y, w, h, \text{confidence})$ = 5개
- 클래스: $C$개 (PASCAL VOC = 20개)

**속도**: 전체 네트워크가 단일 CNN이므로 45 FPS(당시 GPU에서).

**문제점**:
1. 각 grid cell은 최대 1개 객체만 예측 가능 → 작은 객체 클러스터링 실패
2. Aspect ratio 고정 → 특이한 형태 객체 못 맞춤
3. Localization 오류가 분류 오류보다 더 큼

### YOLOv3, v4, v5, v7의 발전

**YOLOv3** (Redmon & Farhadi, 2018):
- Multi-scale prediction (3개 스케일)
- Darknet-53 backbone
- mAP ~57% on COCO

**YOLOv4** (Bochkovskiy et al., 2020):
- CSPDarknet backbone
- Mosaic augmentation, IoU loss
- mAP ~65%

**YOLOv5** (Ultralytics, 2020):
- PyTorch로 재구현, 더 간편한 배포
- mAP ~70%

**YOLOv7** (Wang et al., 2022):
- Reparameterized convolutions
- Auxiliary head for training
- mAP ~73%

### Focal Loss와 Class Imbalance 문제

**배경**: One-stage detection의 가장 큰 문제는 **class imbalance**.

예: COCO dataset에서:
- 전경(객체): ~100,000개
- 배경(no object): ~9,900,000개
- 비율: **1:100**

표준 cross-entropy loss에서는:
$$L_{\text{CE}} = -\log p_t$$

여기서 $p_t$:
- 정답 클래스면 $p_t = p$ (모델의 확률)
- 오답 클래스면 $p_t = 1 - p$

**문제**: 99%의 쉬운 배경 샘플이 1%의 어려운 전경 샘플을 압도.

### Focal Loss의 우아한 해결책

**Focal Loss** (Lin et al., 2017):
$$FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t$$

두 개의 hyperparameter:
- $\alpha$: class weight (일반적으로 $\alpha=0.25$ for foreground)
- $\gamma$: focusing parameter (일반적으로 $\gamma=2$)

**작동 원리**:

1. **$(1-p_t)^\gamma$ 항**: Easy example($p_t \to 1$)의 기여도를 감쇠.
   - $p_t = 0.9$ (거의 맞음): $(1-0.9)^2 = 0.01$ → loss 1/100로 감소
   - $p_t = 0.5$ (불확실): $(1-0.5)^2 = 0.25$ → loss 1/4로만 감소
   - $p_t = 0.1$ (틀림): $(1-0.1)^2 = 0.81$ → loss는 거의 원래대로

2. **$\alpha$ 항**: 전경 클래스에 추가 가중치.

**효과**: 어려운 샘플(hard negatives, hard positives)에만 집중 학습.

### YOLO vs RetinaNet의 비교

**YOLO**:
- 아이디어: grid cell 기반 직접 예측
- 속도: 빠름 (45-155 FPS)
- 정확도: 보통 (mAP ~50-57%)

**RetinaNet** (Lin et al., 2017):
- 아이디어: anchor-based (Faster R-CNN과 유사) + focal loss
- 속도: 느림 (32 FPS)
- 정확도: 높음 (mAP ~59-61%)

Focal Loss의 도입으로 RetinaNet은 one-stage의 단순함을 유지하면서도 two-stage의 정확도에 근접.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.14 — YOLO 출력 구조

이미지를 $S \times S$ grid로 분할, 각 cell $(i, j)$:
$$y_{ijb} = (x_{ijb}, y_{ijb}, w_{ijb}, h_{ijb}, c_{ijb}, p_{ijb}^1, \ldots, p_{ijb}^C)$$

여기서:
- $(x, y, w, h)$: bounding box (cell 좌표계)
- $c$: objectness confidence (해당 cell에 객체가 있을 확률)
- $p^k$: class probability

총 출력: $S \times S \times (B \cdot 5 + C)$

### 정의 6.15 — Cross-Entropy Loss의 Weighted Version

$$L_{\text{CE}} = \sum_i w_i \ell_i$$

여기서 $w_i$는 샘플 $i$의 가중치. Class imbalance 해결을 위해 자주 사용되지만, 모든 샘플을 동등하게 가중치.

### 정의 6.16 — Focal Loss

진정 확률 $p_t \in (0, 1)$에 대해:
$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log p_t$$

또는 one-hot label $y \in \{0,1\}$에 대해:
$$FL(p, y) = -[y \alpha (1-p)^\gamma \log p + (1-y) (1-\alpha) p^\gamma \log(1-p)]$$

### 정리 6.17 — Focal Loss의 수렴 성질

$\gamma = 0$일 때 focal loss는 standard cross-entropy.
$\gamma \to \infty$일 때 focal loss는 hard negatives만 학습.

### 정의 6.18 — RetinaNet 아키텍처

- Backbone: ResNet-50/101
- Neck: Feature Pyramid Network (FPN)
- Head: 분류 subnet + 회귀 subnet
- Loss: focal loss (분류) + Smooth L1 (회귀)

---

## 🔬 증명 또는 수학적 유도

### Focal Loss의 Direct Gradient Analysis

Cross-entropy에서:
$$L_{\text{CE}}(p) = -\log p \quad \text{(진정 클래스)}$$

Gradient:
$$\frac{\partial L_{\text{CE}}}{\partial p} = -\frac{1}{p}$$

예: $p = 0.9$일 때 gradient = $-1/0.9 \approx -1.11$ (큼)
예: $p = 0.1$일 때 gradient = $-1/0.1 = -10$ (매우 큼)

Focal loss에서:
$$L_{\text{FL}}(p) = -(1-p)^2 \log p \quad (\gamma=2)$$

Gradient (chain rule):
$$\frac{\partial L_{\text{FL}}}{\partial p} = -(1-p)^2 \cdot \frac{1}{p} - 2(1-p)(-1) \log p$$
$$= -\frac{(1-p)^2}{p} + 2(1-p) \log p$$

비교:
- $p = 0.9$: $L_{\text{FL}}' = -\frac{0.01}{0.9} + 2 \cdot 0.1 \cdot (-0.105) \approx -0.032$ (100배 작음)
- $p = 0.1$: $L_{\text{FL}}' = -\frac{0.81}{0.1} + 2 \cdot 0.9 \cdot (-2.303) \approx -4.23$ (약 2배 작음)

즉, **easy example에 대한 gradient는 극적으로 감소하고, hard example에 대한 gradient는 유지**.

### Class Imbalance 처리

배경과 전경의 비율이 100:1이면:
$$L_{\text{total}} = \sum_{i=1}^{100} L_{\text{easy bg}} + \sum_{j=1}^{1} L_{\text{hard fg}}$$

easy background loss들이 hard foreground loss를 압도. Focal loss에서:

$$L_{\text{total}} = \sum_{i=1}^{100} (1-p_i)^2 L_{\text{CE}} + \sum_{j=1}^{1} (1-p_j)^\gamma L_{\text{CE}}$$

$(1-p_i)^2$ term이 easy bg의 loss를 대폭 감소시킴.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Focal Loss vs Cross-Entropy

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cross_entropy_loss(p, gamma=0):
    """
    p: 진정 클래스의 확률 (0 ~ 1)
    """
    epsilon = 1e-7
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return -torch.log(p)

def focal_loss(p, gamma=2.0, alpha=0.25):
    """
    Focal loss
    """
    epsilon = 1e-7
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return -alpha * (1 - p) ** gamma * torch.log(p)

# 다양한 확률에 대해 비교
p_values = torch.linspace(0.01, 0.99, 100)
ce_loss = cross_entropy_loss(p_values)
fl_loss_g0 = focal_loss(p_values, gamma=0)
fl_loss_g1 = focal_loss(p_values, gamma=1)
fl_loss_g2 = focal_loss(p_values, gamma=2)
fl_loss_g5 = focal_loss(p_values, gamma=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss 값
ax1.plot(p_values, ce_loss, 'k-', linewidth=2, label='Cross-Entropy')
ax1.plot(p_values, fl_loss_g0, 'b--', label='FL (γ=0)')
ax1.plot(p_values, fl_loss_g1, 'g--', label='FL (γ=1)')
ax1.plot(p_values, fl_loss_g2, 'r-', linewidth=2, label='FL (γ=2)')
ax1.plot(p_values, fl_loss_g5, 'purple', linestyle='--', label='FL (γ=5)')
ax1.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5)
ax1.text(0.9, 2, 'p=0.9\n(easy)', ha='center')
ax1.axvline(x=0.1, color='gray', linestyle=':', alpha=0.5)
ax1.text(0.1, 2, 'p=0.1\n(hard)', ha='center')
ax1.set_xlabel('Predicted probability $p_t$')
ax1.set_ylabel('Loss')
ax1.set_title('Focal Loss vs Cross-Entropy')
ax1.legend()
ax1.grid()

# Gradient
p_values_grad = torch.linspace(0.01, 0.99, 100, requires_grad=True)
ce_grad = torch.autograd.grad(cross_entropy_loss(p_values_grad).sum(), p_values_grad)[0]
fl_grad = torch.autograd.grad(focal_loss(p_values_grad, gamma=2).sum(), p_values_grad)[0]

ax2.plot(p_values_grad.detach(), ce_grad.detach(), 'k-', linewidth=2, label='CE gradient')
ax2.plot(p_values_grad.detach(), fl_grad.detach(), 'r-', linewidth=2, label='FL (γ=2) gradient')
ax2.set_xlabel('Predicted probability $p_t$')
ax2.set_ylabel('Gradient magnitude')
ax2.set_title('Gradient Comparison')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
```

**예상 출력**: FL은 $p_t > 0.9$에서 gradient가 극적으로 감소.

### 실험 2 — YOLO 손실 함수 구현

```python
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        S: grid size
        B: boxes per cell
        C: number of classes
        lambda_coord: weight for coordinate loss
        lambda_noobj: weight for no-object loss
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, targets):
        """
        predictions: [batch, S*S*(B*5+C)]
        targets: [batch, S, S, B*5+C] or [batch, S, S, 5] (one box per cell)
        """
        batch_size = predictions.shape[0]
        
        # Reshape
        predictions = predictions.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        
        # Extract components
        pred_xy = predictions[..., :2*self.B].view(batch_size, self.S, self.S, self.B, 2)
        pred_wh = predictions[..., 2*self.B:4*self.B].view(batch_size, self.S, self.S, self.B, 2)
        pred_conf = predictions[..., 4*self.B:5*self.B].view(batch_size, self.S, self.S, self.B)
        pred_class = predictions[..., 5*self.B:].view(batch_size, self.S, self.S, self.C)
        
        # targets 처리 (간단히: 각 cell마다 최대 1개 box라고 가정)
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4:5]
        target_class = targets[..., 5:]
        
        # Objectness mask
        obj_mask = (target_conf > 0).float()
        noobj_mask = 1.0 - obj_mask
        
        # Loss 계산 (간단한 버전)
        # 1. Coordinate loss
        coord_loss = self.lambda_coord * torch.sum(
            obj_mask * ((pred_xy - target_xy) ** 2 + (pred_wh - target_wh) ** 2)
        )
        
        # 2. Objectness loss
        obj_loss = torch.sum(
            obj_mask * (pred_conf - target_conf) ** 2
        )
        noobj_loss = self.lambda_noobj * torch.sum(
            noobj_mask * pred_conf ** 2
        )
        
        # 3. Classification loss
        class_loss = torch.sum(
            obj_mask.squeeze(-1, keepdims=True) * (pred_class - target_class) ** 2
        )
        
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        return total_loss / batch_size, {
            'coord': coord_loss / batch_size,
            'obj': obj_loss / batch_size,
            'noobj': noobj_loss / batch_size,
            'class': class_loss / batch_size
        }

# 테스트
S, B, C = 7, 2, 20
loss_fn = YOLOLoss(S=S, B=B, C=C)

batch_size = 4
predictions = torch.randn(batch_size, S * S * (B * 5 + C))
targets = torch.randn(batch_size, S, S, 5 + C)

total_loss, losses = loss_fn(predictions, targets)
print(f"Total loss: {total_loss.item():.4f}")
for key, val in losses.items():
    print(f"  {key}: {val.item():.4f}")
```

### 실험 3 — RetinaNet with torchvision

```python
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn

# 사전학습 모델 로드
model = retinanet_resnet50_fpn(pretrained=True, 
                                num_classes=91)  # COCO
model.eval()

# Focal loss는 모델에 내장되어 있음
# training 중에만 사용

# 추론
image = torch.randn(3, 800, 800)

with torch.no_grad():
    outputs = model([image])

# 결과
for output in outputs:
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    print(f"Detected {len(boxes)} objects")
    for i, (box, score, label) in enumerate(zip(boxes[:5], scores[:5], labels[:5])):
        print(f"  {i}: box={box.tolist()}, score={score:.3f}, label={label}")
```

---

## 🔗 이론과 실전의 간극

### YOLO의 Grid-based 접근의 한계

이론: 각 cell이 1개 box를 예측하므로 간단하고 빠름.

실전: 작은 객체 클러스터(예: 새 떼)에 실패. 이를 해결하기 위해 YOLOv3+는 multi-scale prediction, anchor boxes 도입.

### Focal Loss의 적용 범위

이론: Class imbalance 해결 (1:100 비율).

실전:
- Detection, segmentation, medical imaging에서 매우 효과적
- Balanced dataset에서는 standard CE와 비슷 (또는 약간 더 나쁨)
- $\gamma$ 선택이 중요: 데이터셋에 따라 1-3 범위

### 속도 vs 정확도 트레이드오프

- YOLO: 45-155 FPS, mAP ~50-60%
- RetinaNet: 32 FPS, mAP ~60-62%
- 현대 two-stage (Faster R-CNN): 5-10 FPS, mAP ~65-70%

실전: task 특성에 따라 선택 (실시간 vs 정확도 우선).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Class imbalance는 foreground vs background만 | 클래스 간 imbalance(예: 보행자 vs 자동차)도 존재 |
| Focal loss의 $\gamma$ 고정 | 데이터셋마다 최적값이 다름 |
| NMS는 후처리 (미분 불가) | End-to-end differentiable NMS 필요한 경우 있음 |
| 고정 앵커 설계 | 새로운 객체 크기에 적응 불가 |
| 단일 스케일 feature | 작은 객체 탐지에 약함 |

---

## 📌 핵심 정리

$$\boxed{FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t \text{ — class imbalance의 우아한 해결책}}$$

| 개념 | 정의 |
|------|------|
| **YOLO** | $S \times S$ grid, 직접 예측, 45 FPS |
| **Grid-based output** | $S \times S \times (B \cdot 5 + C)$ 텐서 |
| **One-stage detection** | Region proposal 없이 직접 탐지 |
| **Class imbalance** | Foreground:background = 1:100+ |
| **Focal Loss** | $-\alpha(1-p_t)^\gamma \log p_t$, easy example 감쇠 |
| **$(1-p_t)^\gamma$ 항** | Easy example($p_t \to 1$)에 $\to 0$, hard($p_t \to 0$)에 $\to 1$ |
| **RetinaNet** | Anchor-based + focal loss, 높은 정확도 |
| **YOLOv3-v7** | Multi-scale, backbone 개선, 계속 발전 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Cross-entropy loss $L_{\text{CE}} = -\log p$에서 $p = 0.9$일 때와 $p = 0.1$일 때의 loss 값을 비교하라.

<details>
<summary>힌트 및 해설</summary>

$L_{\text{CE}}(0.9) = -\log(0.9) \approx 0.105$
$L_{\text{CE}}(0.1) = -\log(0.1) \approx 2.303$

약 22배 차이. 모델이 틀렸을 때(0.1) 훨씬 큰 loss.

</details>

**문제 2** (심화): Focal loss $FL = -(1-p)^\gamma \log p$ ($\gamma=2$)에서 $p = 0.9$와 $p = 0.1$일 때를 비교하라.

<details>
<summary>힌트 및 해설</summary>

$FL(0.9) = -(0.1)^2 \log(0.9) = -0.01 \times 0.105 \approx 0.00105$
$FL(0.1) = -(0.9)^2 \log(0.1) = -0.81 \times 2.303 \approx 1.865$

약 1,800배 차이! $(1-p)^2$ 항이 easy example($p=0.9$)의 loss를 극적으로 감소.

</details>

**问题 3** (이론-실전): YOLO에서 $\lambda_{\text{noobj}} = 0.5$를 사용하는 이유는 무엇인가? 이를 1.0이나 0.0으로 바꾸면?

<details>
<summary>힌트 및 해설</summary>

대부분의 예측이 배경(no object)이므로, noobj loss가 전체를 압도.
$\lambda_{\text{noobj}} = 0.5 < 1$로 감소시켜 배경 샘플의 영향력 완화.

- $\lambda = 1.0$: 배경과 객체 동등 가중치, 객체 탐지 성능 떨어짐
- $\lambda = 0.0$: 배경 무시, 모델이 배경 학습 실패, 높은 false positive

최적값은 데이터셋 특성에 따라 다름.

</details>

---

<div align="center">

[◀ 이전](./02-two-stage-detection.md) | [📚 README](../README.md) | [다음 ▶](./04-segmentation.md)

</div>
