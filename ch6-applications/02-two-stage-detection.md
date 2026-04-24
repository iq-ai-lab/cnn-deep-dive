# 02. 두 단계 객체 탐지 (Two-Stage Detection) — Faster R-CNN과 RPN

## 🎯 핵심 질문

- Faster R-CNN (Ren et al., 2015)의 혁신은 무엇이고, Region Proposal Network(RPN)는 왜 필수였는가?
- Anchor box를 3 scale × 3 ratio로 설계하는 이유는 무엇이며, 다른 조합이 성능에 미치는 영향은?
- RoI Pooling과 RoI Align의 차이는 무엇인가? RoI Align이 mask AP를 약 3% 개선하는 이유는?
- Non-Maximum Suppression (NMS)과 soft-NMS의 수학적 원리와 실전적 비교는?

---

## 🔍 왜 이 개념이 CNN에 중요한가

객체 탐지는 단순 분류보다 훨씬 복잡합니다. 이미지에 수백 개의 객체가 있을 수 있고, 정확한 bounding box를 예측해야 합니다. Faster R-CNN은 **두 단계** 접근으로:
1. **RPN**: 가능성 있는 영역(region proposal) 생성
2. **분류·회귀**: 각 영역에 대해 클래스와 정확한 box 예측

이는 one-stage 탐지(Ch6-03)의 기초가 되며, 현대의 Mask R-CNN, Cascade R-CNN, DETR 등도 이 아이디어를 상속합니다.

---

## 📐 수학적 선행 조건

- [Ch6-01 Classification](./01-classification.md): Softmax, cross-entropy, smoothmax의 gradient
- [Ch5-03 Convolutional Networks](../ch5-modern-cnn/03-conv-arithmetic.md): Convolution operation, receptive field
- [Ch3 Optimization](../ch3-optimization/): SGD, momentum
- 선형대수: Bounding box coordinates, Intersection over Union (IoU)

---

## 📖 직관적 이해

### Fast R-CNN에서 Faster R-CNN로

**Fast R-CNN** (Girshick, 2015):
1. CNN backbone으로 image feature map 추출
2. Selective search로 region proposal 생성(~2000개/이미지, 느림)
3. RoI Pooling으로 고정 크기 feature 추출
4. FC layer로 분류·회귀

병목: Selective search는 CNN 밖에서 독립적으로 수행, 시간 낭비.

**Faster R-CNN** (Ren et al., 2015):
- RPN을 추가: CNN feature map 위에서 직접 region proposal 생성
- 전체 네트워크 end-to-end 학습 가능
- 속도: ~200ms/이미지 (당시로는 실시간에 가까움)

### Region Proposal Network (RPN)

RPN은 sliding window 방식:
1. CNN backbone 마지막 feature map (예: 14×14 spatial, 512 channel)
2. 각 spatial location에 K개 anchor box 생성 (보통 K=9)
3. 각 anchor에 대해 2개 점수 출력:
   - **objectness**: 이 anchor가 객체를 포함할 확률
   - **bounding box regression**: 객체의 정확한 위치로의 offset

### Anchor Box의 설계

좋은 anchor 설계 = 모든 크기·형태의 객체를 잘 커버.

**3 scale × 3 ratio = 9 anchors per location**:

Scales: 8, 16, 32 (원본 이미지 기준, stride 16인 경우)
Ratios: 1:1, 1:2, 2:1

예를 들어, base anchor (64×64):
- (1:1): 64×64
- (1:2): 45×90 (세로 긴 객체)
- (2:1): 90×45 (가로 긴 객체)

각 scale마다 반복하여 총 9개.

**효과**: 작은 객체(scale 8), 큰 객체(scale 32), 가로·세로 비율 모두 커버.

### RoI Pooling vs RoI Align

**RoI Pooling** (Fast R-CNN):
- Region proposal을 고정 크기(예: 7×7)로 강제 축약
- 양자화(rounding)로 인한 오차 발생

수식: feature map의 region $[x_1, y_1, x_2, y_2]$를 7×7로 grid 나누고, 각 grid 셀의 max pooling.

**문제**: 원본 좌표와 feature map 좌표 사이의 mismatch
- Bounding box 좌표는 float, 하지만 grid 인덱스는 정수
- 반올림 오류 누적 → mask 생성 시 정렬 오차

**RoI Align** (Mask R-CNN, He et al., 2017):
- Bilinear interpolation으로 sub-pixel 정확도 유지
- 양자화 단계 제거

```
RoI Pooling: [region] → round to grid → max pool
RoI Align:   [region] → bilinear interp at fractional coords → max pool
```

**효과**: Mask AP(instance segmentation) +3%, 특히 small 객체에서 유의미.

### Non-Maximum Suppression (NMS)

문제: RPN과 분류기가 같은 객체에 대해 여러 bounding box 제출 가능.

**NMS 알고리즘**:
1. 모든 box를 objectness score 기준으로 정렬
2. 최고 점수 box 선택, 나머지와 IoU 계산
3. IoU > 임계값(보통 0.5)인 모든 box 제거
4. 반복

```python
boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
keep = []
while boxes:
    current = boxes.pop(0)
    keep.append(current)
    boxes = [b for b in boxes if iou(current, b) < 0.5]
return keep
```

**Soft-NMS** (Bodla et al., 2017):
- Hard removal 대신, score를 linearly 감소
- $s_i := s_i \cdot (1 - \text{IoU}(M, b_i))$
- 또는 gaussian: $s_i := s_i \cdot \exp(-\text{IoU}(M, b_i)^2 / \sigma)$

효과: 부분적으로 겹친 객체도 탐지 가능.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.8 — Bounding Box Parameterization

원본 좌표와 offset:
$$x = w_a \delta_x + x_a, \quad y = h_a \delta_y + y_a$$
$$w = w_a \exp(\delta_w), \quad h = h_a \exp(\delta_h)$$

여기서 $(x_a, y_a, w_a, h_a)$는 anchor, $(\delta_x, \delta_y, \delta_w, \delta_h)$는 네트워크 출력.

### 정의 6.9 — Intersection over Union (IoU)

두 box $A, B$:
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

범위: [0, 1]. 겹치지 않으면 0, 완전히 같으면 1.

### 정리 6.10 — RPN 손실 함수

$$L = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* L_{\text{reg}}(t_i, t_i^*)$$

여기서:
- $p_i$ = objectness score (logit)
- $p_i^*$ = 1 (positive, IoU > 0.7) 또는 0 (negative, IoU < 0.3)
- $t_i$ = bounding box regression target
- $L_{\text{cls}}$ = binary cross-entropy
- $L_{\text{reg}}$ = Smooth L1 loss
- $\lambda = 10$ (균형 계수)

### 정의 6.11 — Smooth L1 Loss

$$L_{\text{smooth L1}}(x) = \begin{cases}
0.5 x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$

L2 loss보다 outlier에 robust.

### 정의 6.12 — RoI Align (Bilinear Interpolation)

Region $[x_1, y_1, x_2, y_2]$를 $H \times W$ grid로 나누고, 각 grid 셀에서 bilinear interpolation:

$$f(x, y) = \sum_{i=0}^1 \sum_{j=0}^1 f(i, j) \max(0, 1-|x-i|) \max(0, 1-|y-j|)$$

### 정의 6.13 — Non-Maximum Suppression

점수 $s$, IoU 임계값 $\tau$:

**Hard NMS**:
$$\text{NMS}(B) = \{\arg\max_i s_i\} \cup \text{NMS}(\{b_j : \text{IoU}(b_j, \arg\max) < \tau\})$$

**Soft NMS**:
$$s_i \gets s_i \cdot \begin{cases}
1 - \text{IoU}(M, b_i) & \text{linear} \\
\exp(-\text{IoU}(M, b_i)^2 / \sigma) & \text{gaussian}
\end{cases}$$

---

## 🔬 증명 또는 수학적 유도

### Bounding Box Regression의 정당성

Anchor $A = (x_a, y_a, w_a, h_a)$에서 target box $T = (x, y, w, h)$로의 변환:

Linear transformation (center, log-scale):
$$\delta_x = \frac{x - x_a}{w_a}, \quad \delta_y = \frac{y - y_a}{h_a}$$
$$\delta_w = \log\frac{w}{w_a}, \quad \delta_h = \log\frac{h}{h_a}$$

역변환:
$$x = w_a \delta_x + x_a, \quad w = w_a \exp(\delta_w)$$

**장점**:
1. **Scale invariance**: $\delta_w, \delta_h$는 log 스케일이므로, 작은 객체와 큰 객체의 offset이 유사한 범위
2. **안정성**: 로그는 양수를 음수 범위로 매핑, 신경망이 학습하기 쉬움

### IoU의 속성

```
두 box A, B에 대해:
- IoU(A, B) ∈ [0, 1]
- IoU(A, B) = 1 ⟺ A = B
- IoU(A, B) = IoU(B, A) (대칭)
- 0 < IoU(A, B) < IoU(A, C) < IoU(A, D)라고 해도, A와 C의 거리가 A와 D보다 작을 수 있음
  (IoU는 거리 메트릭이 아님, 특히 scale 변화에 민감)
```

### Soft-NMS의 수학적 해석

Linear soft-NMS:
$$s_i \gets s_i (1 - \text{IoU}(M, b_i))$$

$\text{IoU} = 0$일 때 $s_i$는 유지, $\text{IoU} \to 1$일 때 $s_i \to 0$.

Gaussian soft-NMS:
$$s_i \gets s_i \exp\left(-\frac{\text{IoU}(M, b_i)^2}{\sigma}\right)$$

이는 Gaussian kernel로 볼 수 있으며, smooth한 score decay 제공.

**효과**: Overlapping box 중 일부는 제거되지 않고 score 감소만 → 부분적 occlusion에 robust.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — IoU 계산 및 NMS

```python
import torch
import numpy as np

def iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def nms(boxes, scores, iou_threshold=0.5):
    """
    Hard NMS
    boxes: [N, 4]
    scores: [N]
    """
    assert len(boxes) == len(scores)
    
    indices = np.argsort(-scores)  # 내림차순
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        remaining = indices[1:]
        
        ious = np.array([iou(current_box, boxes[i]) for i in remaining])
        indices = remaining[ious < iou_threshold]
    
    return keep

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, method='linear'):
    """
    Soft NMS
    """
    indices = np.argsort(-scores)
    keep = []
    scores_out = scores.copy()
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        remaining = indices[1:]
        
        ious = np.array([iou(current_box, boxes[i]) for i in remaining])
        
        if method == 'linear':
            scores_out[remaining] = scores_out[remaining] * (1 - ious)
        elif method == 'gaussian':
            scores_out[remaining] = scores_out[remaining] * np.exp(-(ious**2) / sigma)
        
        # 임계값보다 높은 것만 유지
        indices = remaining[scores_out[remaining] > 0.0001]
    
    return keep, scores_out

# 테스트
boxes = np.array([
    [10, 10, 100, 100],
    [15, 15, 105, 105],
    [200, 200, 300, 300],
    [205, 205, 305, 305],
])
scores = np.array([0.95, 0.8, 0.9, 0.7])

print("Hard NMS:")
keep_hard = nms(boxes, scores, iou_threshold=0.5)
print(f"Kept boxes: {keep_hard}")

print("\nSoft NMS (linear):")
keep_soft, scores_soft = soft_nms(boxes, scores, iou_threshold=0.5, method='linear')
print(f"Kept boxes: {keep_soft}")
print(f"Updated scores: {scores_soft}")
```

**예상 출력**: Hard NMS는 겹친 box들을 제거, soft-NMS는 score만 감소.

### 실험 2 — RPN 손실 함수

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_l1_loss(predictions, targets, beta=1.0):
    """
    Smooth L1 loss for bounding box regression
    """
    diff = torch.abs(predictions - targets)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss.mean()

class RPNLoss(nn.Module):
    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, objectness_logits, bbox_deltas, 
                objectness_targets, bbox_targets):
        """
        objectness_logits: [N, num_anchors]
        bbox_deltas: [N, num_anchors, 4]
        objectness_targets: [N, num_anchors] {0, 1}
        bbox_targets: [N, num_anchors, 4]
        """
        # Classification loss (binary cross-entropy)
        cls_loss = F.binary_cross_entropy_with_logits(
            objectness_logits, 
            objectness_targets,
            reduction='mean'
        )
        
        # Regression loss (Smooth L1)
        # Only positive anchors contribute
        pos_mask = (objectness_targets == 1).unsqueeze(-1)
        bbox_targets_masked = bbox_targets[pos_mask]
        bbox_deltas_masked = bbox_deltas[pos_mask]
        
        if len(bbox_targets_masked) > 0:
            reg_loss = smooth_l1_loss(bbox_deltas_masked, bbox_targets_masked)
        else:
            reg_loss = torch.tensor(0.0, device=bbox_deltas.device)
        
        total_loss = cls_loss + self.lambda_reg * reg_loss
        return total_loss, cls_loss, reg_loss

# 테스트
rpn_loss = RPNLoss(lambda_reg=10.0)

batch_size = 2
num_anchors = 9
num_classes = 1

objectness_logits = torch.randn(batch_size, num_anchors)
bbox_deltas = torch.randn(batch_size, num_anchors, 4)
objectness_targets = torch.randint(0, 2, (batch_size, num_anchors)).float()
bbox_targets = torch.randn(batch_size, num_anchors, 4) * 0.1  # 작은 offset

total, cls, reg = rpn_loss(objectness_logits, bbox_deltas, 
                            objectness_targets, bbox_targets)

print(f"Total loss: {total.item():.4f}")
print(f"Classification loss: {cls.item():.4f}")
print(f"Regression loss: {reg.item():.4f}")
```

### 실험 3 — Faster R-CNN with torchvision

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

# 사전학습 모델 로드 (COCO dataset)
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지 로드 및 전처리
image = Image.open('example.jpg')
image_tensor = F.to_tensor(image)

# 추론
with torch.no_grad():
    outputs = model([image_tensor])

# 결과 처리
for output in outputs:
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    print(f"Detected {len(boxes)} objects")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score > 0.5:  # 신뢰도 필터
            x1, y1, x2, y2 = box
            print(f"  Box {i}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], score={score:.3f}, label={label}")
```

---

## 🔗 이론과 실전의 간극

### Anchor Design의 실전 고려사항

이론: 3 scale × 3 ratio = 고정 9개 anchor.

실전: 
- 데이터셋 특성에 따라 scale/ratio 조정 (예: 의료 이미지는 작은 객체 많음)
- Feature Pyramid Networks (FPN): 여러 해상도 feature map에서 anchor 사용
- Dense object detection: 더 많은 anchor 사용 가능 (메모리 허용 범위 내)

### RoI Align의 성능 개선

이론: Bilinear interpolation으로 sub-pixel 정확도 향상.

실전: 
- Mask R-CNN에서 mask AP +3% (instance segmentation)
- 하지만 detection AP에서는 minimal 개선 (bbox만 필요)
- 계산량 증가: RoI Pooling보다 약 10-20% 느림

### NMS vs Soft-NMS

이론: Soft-NMS는 linear/gaussian score decay로 overlapping 객체 유지.

실전:
- 표준 NMS: 간단하고 빠름, 많은 경우 충분
- Soft-NMS: crowded scene (보행자 탐지 등)에서 +1-2% 성능 개선
- Weighted NMS: 더 정교한 weighting scheme 가능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 고정 anchor ratio/scale | 새로운 객체 크기·형태에 적응 불가 |
| IoU로 positive/negative anchor 판정 (IoU > 0.7 등) | 경계 케이스(IoU = 0.7001)에 민감 |
| 고정 NMS 임계값(보통 0.5) | 다양한 씬에 최적값이 다름 |
| RPN과 분류기 head가 공유 backbone | 특정 task에는 독립적 backbone이 더 나을 수 있음 |
| 후처리(NMS) 비미분 연산 | 강화학습 같은 end-to-end 학습에 부적합 |

---

## 📌 핵심 정리

$$\boxed{L_{\text{RPN}} = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* L_{\text{smooth L1}}(t_i, t_i^*)}$$

| 개념 | 정의 |
|------|------|
| **RPN** | Convolutional region proposal network, anchor-based box 생성 |
| **Anchor box** | 3 scale × 3 ratio = 9 reference boxes per location |
| **RoI Pooling** | Region을 고정 크기로 max pooling (양자화 오차 있음) |
| **RoI Align** | Bilinear interpolation으로 sub-pixel 정확도 유지 |
| **Smooth L1 loss** | L1과 L2의 중간, outlier에 robust |
| **NMS** | Hard removal, IoU 기반 중복 제거 |
| **Soft-NMS** | Score decay, overlapping 객체 부분 유지 |
| **IoU** | $\frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 두 bounding box $A = [10, 10, 100, 100]$, $B = [50, 50, 150, 150]$에 대해 IoU를 계산하라. (좌표는 [x1, y1, x2, y2])

<details>
<summary>힌트 및 해설</summary>

Intersection:
- $x_1^{\text{inter}} = \max(10, 50) = 50$
- $y_1^{\text{inter}} = \max(10, 50) = 50$
- $x_2^{\text{inter}} = \min(100, 150) = 100$
- $y_2^{\text{inter}} = \min(100, 150) = 100$
- Area = $(100-50) \times (100-50) = 2500$

Area of A: $(100-10) \times (100-10) = 8100$
Area of B: $(150-50) \times (150-50) = 10000$
Union: $8100 + 10000 - 2500 = 15600$

IoU = $2500 / 15600 \approx 0.16$

</details>

**문제 2** (심화): Bounding box regression에서 offset $(delta_x, \delta_y, \delta_w, \delta_h) = (0.1, 0.1, 0.2, -0.1)$를 anchor $(x_a, y_a, w_a, h_a) = (50, 50, 100, 100)$에 적용하면 최종 box는?

<details>
<summary>힌트 및 해설</summary>

$$x = w_a \delta_x + x_a = 100 \cdot 0.1 + 50 = 60$$
$$y = h_a \delta_y + y_a = 100 \cdot 0.1 + 50 = 60$$
$$w = w_a \exp(\delta_w) = 100 \cdot \exp(0.2) \approx 100 \cdot 1.221 = 122.1$$
$$h = h_a \exp(\delta_h) = 100 \cdot \exp(-0.1) \approx 100 \cdot 0.905 = 90.5$$

최종 box: center $(60, 60)$, size $(122.1, 90.5)$

</details>

**문제 3** (이론-실전): 현재 NMS가 선택한 box의 objectness score가 0.9이고, 다음 후보 box의 score는 0.85, IoU는 0.6일 때, hard NMS와 linear soft-NMS에서 각각 어떻게 처리되는가?

<details>
<summary>힌트 및 해설</summary>

Hard NMS (IoU 임계값 = 0.5):
- IoU 0.6 > 0.5이므로, 다음 box를 완전히 제거.
- 최종: score 0.85는 사라짐.

Soft-NMS (linear):
- Score를 감소: $0.85 \times (1 - 0.6) = 0.85 \times 0.4 = 0.34$
- 이제 score 0.34로 다음 단계 진행.

결과: Soft-NMS는 다음 box를 완전 제거하지 않고, score 감소만 가함. Crowded scene에서는 soft-NMS가 더 많은 객체를 탐지 가능.

</details>

---

<div align="center">

[◀ 이전](./01-classification.md) | [📚 README](../README.md) | [다음 ▶](./03-one-stage-detection.md)

</div>
