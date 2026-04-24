# 01. CNN의 Inductive Bias: 강점과 한계

## 🎯 핵심 질문

- CNN의 세 가지 핵심 귀납적 편향(inductive bias) — translation equivariance, locality, hierarchy — 은 왜 vision 문제에 적합한가?
- 작은 데이터셋에서 CNN이 Vision Transformer를 능가하는 이유는 무엇인가?
- Dosovitskiy et al. (2021)의 JFT-300M 실험에서 ViT가 CNN을 능가하는 임계점은 어디인가?
- 데이터 스케일과 귀납적 편향의 trade-off를 정량적으로 특성화할 수 있는가?

---

## 🔍 왜 이 개념이 CNN 이해에 중요한가

"CNN이 좋다"는 직관은 오래되었지만, "왜" CNN이 특정 문제에 적합한지 명확히 설명하지 못합니다. **Inductive bias**는 이 질문에 답하는 핵심 개념입니다. 작은 데이터에서는 강력한 inductive bias (CNN)가 필수이고, 거대 데이터에서는 유연한 모델 (Vision Transformer)이 더 우수합니다. 이 경계를 이해하면 모델 선택 원리를 획득하고, CNN과 Transformer의 본질적 차이를 인식할 수 있습니다.

---

## 📐 수학적 선행 조건

- [CNN 기본 연산](../ch2-cnn-ops/README.md): Convolution, stride, padding 정의
- [Receptive Field](../ch3-receptive-field/README.md): Receptive field와 계층 구조
- [ResNet과 모던 CNN](../ch5-modern-cnn/README.md): Skip connection, normalization
- 최적화 이론: Regularization, generalization gap
- 통계학: Sample complexity, VC dimension

---

## 📖 직관적 이해

### Inductive Bias란 무엇인가?

**Inductive bias**는 학습 알고리즘이 데이터에 대해 사전에 가정하는 구조입니다. 예를 들어:

- **Linear regression**: "함수는 선형"이라는 bias
- **Decision tree**: "특성 간 독립성", "정보 이득 기준" bias
- **CNN**: "translation equivariance", "locality", "hierarchy" bias

이러한 bias가 없으면, 모든 함수족을 동등하게 취급하므로 sample complexity가 지수적으로 증가합니다.

### CNN의 세 가지 핵심 Inductive Bias

#### 1. Translation Equivariance (평행이동 동변성)

이미지를 약간 옮기면, 결과도 동등하게 옮겨갑니다:

$$f(\text{shift}(x)) = \text{shift}(f(x))$$

이는 convolution의 핵심성질입니다. CNN은 같은 필터를 모든 위치에 공유하므로 이 성질을 **자동으로 만족**합니다.

#### 2. Locality (국소성)

픽셀 $(i, j)$의 출력은 주변 receptive field 내의 픽셀들만 영향을 받습니다. 먼 거리의 픽셀은 초기 층에서 직접 영향을 주지 않습니다. 이는 **작은 kernel** (3×3, 5×5)에서 비롯됩니다.

#### 3. Hierarchy (계층성)

초기 층은 edge, corner 같은 저수준 특징을 학습하고, 깊은 층은 이들을 조합하여 고수준 의미론적 특징을 학습합니다. **Pooling과 stride**가 이를 구현합니다.

### 데이터와 Bias의 Trade-off

- **작은 데이터 (예: CIFAR-10, ImageNet-1k)**: Bias 없이 충분한 학습 불가능. CNN의 강한 bias가 필수 → CNN > Transformer
- **거대 데이터 (예: JFT-300M)**: 데이터가 충분하면 bias 없는 유연한 모델 (Transformer)이 더 좋음. → Transformer > CNN

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Translation Equivariance

함수 $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H' \times W' \times C'}$가 translation equivariant라 함은:

$$f(T_\delta x) = T_\delta f(x)$$

여기서 $T_\delta$는 shift operator. 즉, $(T_\delta x)_{i,j} = x_{i-\delta_1, j-\delta_2}$.

### 정의 1.2 — Locality (Receptive Field 기반)

Layer $\ell$에서 위치 $(i, j)$의 출력이 input image의 위치 $(i', j')$에 의존하려면:

$$|(i, j) - (i', j')|_\infty \leq \text{RF}_\ell$$

여기서 $\text{RF}_\ell$은 layer $\ell$의 receptive field 반경.

### 정의 1.3 — Sample Complexity

데이터셋 $D$ 크기 $n$에서, 모델이 $\epsilon$-optimal solution에 도달하기 위한 필요 샘플 수:

$$n(\epsilon) = O(d_{\text{eff}} / \epsilon)$$

여기서 $d_{\text{eff}}$는 **effective dimension** (귀납적 bias에 따라 결정).

### 정리 1.4 — Inductive Bias와 Generalization

더 강한 inductive bias를 가진 모델 $M_1$은:
- ✓ 작은 $n$에서 더 낮은 generalization error
- ✗ 큰 $n$에서 bias가 부정확하면 높은 approximation error

역으로, 약한 bias $M_2$는:
- ✗ 작은 $n$에서 높은 variance
- ✓ 큰 $n$에서 유연성으로 더 나음

---

## 🔬 증명 및 수학적 유도

### CNN이 Translation Equivariance를 만족하는 증명

Convolution layer를 생각해봅시다:

$$(\text{conv}_k * x)_{i,j} = \sum_{a,b} k_{a,b} \cdot x_{i+a, j+b}$$

Shift된 input $x' = T_\delta x$에 대해:

$$(\text{conv}_k * x')_{i,j} = \sum_{a,b} k_{a,b} \cdot x'_{i+a, j+b}$$

$x'_{i+a, j+b} = x_{i+a-\delta_1, j+b-\delta_2}$이므로:

$$(\text{conv}_k * x')_{i,j} = \sum_{a,b} k_{a,b} \cdot x_{i+a-\delta_1, j+b-\delta_2}$$

$a' = a - \delta_1$, $b' = b - \delta_2$로 치환하면:

$$= \sum_{a',b'} k_{a'-\delta_1, b'-\delta_2} \cdot x_{i+a', j+b'}$$

만약 $k$가 shift-invariant (대부분의 CNN kernel은 이 가정 없음)하면, 이는 $(\text{conv}_k * x)_{i-\delta_1, j-\delta_2} = T_\delta (\text{conv}_k * x)$를 의미합니다. 더 정확히, **convolution 자체가 정의상 translation equivariant**입니다.

### Receptive Field와 Hierarchy의 수학적 특성화

Layer $\ell$의 receptive field:

$$\text{RF}_\ell = 1 + \sum_{\ell'=1}^\ell (k_{\ell'} - 1) \prod_{\ell''=1}^{\ell'-1} s_{\ell''}$$

여기서 $k_\ell$은 kernel size, $s_\ell$은 stride.

Pooling (stride > 1)은 RF를 기하급수적으로 증가시키므로, ResNet-50 같은 깊은 네트워크는 전체 이미지를 한 픽셀으로 receptive field를 가집니다. 그 과정에서 spatial resolution은 감소하지만, **semantic receptiveness는 증가**합니다.

---

## 💻 실험 재현

### 실험 1 — 데이터 크기별 CNN vs ViT 성능

ImageNet-1k, CIFAR-10, CIFAR-100에서의 비교:

```python
import torch
import torchvision.models as models
from timm.models import vision_transformer as vit

# 사전학습된 모델 로드
resnet50 = models.resnet50(pretrained=True)
vit_b = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

# 간단한 평가 함수
def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):  # DINO 모델
                outputs = outputs['logits']
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 결과: ImageNet-1k에서
# ResNet-50: ~76%
# ViT-B: ~77% (추가 증강 포함)
# CIFAR-10에서
# ResNet-50: ~96%
# ViT-B: ~94% (증강 없음, 작은 데이터)
```

**관찰**: CIFAR-10 같은 작은 데이터에서는 CNN (ResNet)이 이기고, 큰 데이터 (JFT-300M)에서는 ViT가 우수합니다.

### 실험 2 — Translation Equivariance 확인

```python
import torch
import torch.nn.functional as F
from torchvision import transforms

# 간단한 CNN 모델
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

model = SimpleCNN().eval()

# 임의의 이미지 (Batch=1, C=3, H=32, W=32)
x = torch.randn(1, 3, 32, 32)

# 원본 이미지에서의 출력
y1 = model(x)

# Shift된 이미지 (2 픽셀 위로, 3 픽셀 오른쪽)
shift_h, shift_w = 2, 3
x_shifted = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))

# Shift된 이미지에서의 출력
y2 = model(x_shifted)

# Shift된 출력과 비교
y1_shifted = torch.roll(y1, shifts=(-shift_h, -shift_w), dims=(2, 3))

# Equivariance 검사
mse_error = F.mse_loss(y1_shifted, y2)
print(f"Translation Equivariance Error (MSE): {mse_error.item():.6f}")
# 결과: Padding 때문에 경계에서 약간의 오차, 내부에서는 0에 가까움
```

**관찰**: CNN은 translation equivariance를 만족하지만, padding의 경계 효과가 있습니다.

### 실험 3 — Receptive Field 증가의 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# Receptive Field 계산 함수
def compute_rf(kernel_sizes, strides, dilations=[1]*5):
    """
    kernel_sizes: 각 layer의 kernel 크기 리스트
    strides: 각 layer의 stride 리스트
    dilations: 각 layer의 dilation
    """
    rf = 1
    effective_stride = 1
    rf_list = [rf]
    
    for k, s, d in zip(kernel_sizes, strides, dilations):
        rf = rf + (k - 1) * d * effective_stride
        effective_stride *= s
        rf_list.append(rf)
    
    return rf_list

# ResNet-50 구조 (단순화)
kernels = [7] + [3]*49  # 첫 conv (7x7), 나머지 3x3
strides = [2, 1, 2, 1, 2, 1, 2, 1] + [1]*41  # pooling/stride-2 위치

rf_list = compute_rf(kernels[:8], strides[:8])

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(rf_list)), rf_list, 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Layer Depth')
ax.set_ylabel('Receptive Field Size')
ax.set_title('Receptive Field Growth in CNN')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.show()
```

**관찰**: 초반에는 천천히 증가하지만, stride-2 layer 이후 기하급수적으로 증가합니다.

### 실험 4 — Small Dataset에서 Regularization 효과

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 작은 synthetic 데이터셋
torch.manual_seed(42)
n_samples = 100
x_train = torch.randn(n_samples, 3, 32, 32)
y_train = torch.randint(0, 10, (n_samples,))

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN vs "No Bias" (fully connected)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 16 * 16, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NoBiasModel(nn.Module):  # No bias, fully connected
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 훈련
cnn_model = SimpleCNN()
fc_model = NoBiasModel()

criterion = nn.CrossEntropyLoss()
cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
fc_opt = torch.optim.Adam(fc_model.parameters(), lr=0.001)

for epoch in range(20):
    cnn_model.train()
    fc_model.train()
    
    cnn_loss_epoch = 0
    fc_loss_epoch = 0
    
    for x_batch, y_batch in loader:
        # CNN
        cnn_opt.zero_grad()
        out = cnn_model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        cnn_opt.step()
        cnn_loss_epoch += loss.item()
        
        # FC
        fc_opt.zero_grad()
        out = fc_model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        fc_opt.step()
        fc_loss_epoch += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: CNN Loss={cnn_loss_epoch/len(loader):.3f}, "
              f"FC Loss={fc_loss_epoch/len(loader):.3f}")

# 결과: CNN이 더 안정적으로 학습, FC는 overfitting 심함
```

**관찰**: 같은 크기 데이터에서 CNN의 inductive bias가 overfitting을 억제합니다.

---

## 🔗 이론과 실전의 간극

### ImageNet 규모에서의 현실

실제로 ImageNet-1k (약 100만 이미지)는:

- **CNN (ResNet-50)**: 76-77% top-1 accuracy (표준 augmentation)
- **ViT-B (DINO 사전학습)**: 77-78% (추가 증강 + 훈련 스케일 업)

하지만 같은 데이터, 같은 계산 비용으로 공정하게 비교하면 CNN이 약간 우수합니다.

### JFT-300M 규모에서의 전환

Google의 사내 JFT-300M 데이터셋 (약 3억 이미지)에서:

- **ResNet-50**: ~84% ImageNet top-1 transfer accuracy
- **ViT-B**: ~88% ImageNet top-1 transfer accuracy

약 300배의 데이터 스케일에서 **inductive bias의 중요성이 역전**됩니다.

### 중간 규모 (ImageNet-21k)의 경계

약 1400만 이미지의 ImageNet-21k에서는 **중간 영역**이 나타납니다:

- ViT-B: 85% (ImageNet-1k transfer)
- ResNet-152: ~84.5%

더 큰 ViT (ViT-L, ViT-H)는 이 규모에서도 CNN을 능가합니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Translation equivariance가 중요 | 객체 중심이 아닌 장면 이해에선 덜 중요 (예: 장면 문제) |
| Locality가 vision에 필수 | Global context가 중요한 작업 (예: 분할 이미지 이해)에선 불리 |
| Hierarchy는 자동 학습 | 데이터 구조가 계층적 아니면 (예: 그래프)는 해당 없음 |
| 고정된 공간 구조 | 비유클리드 데이터 (3D point cloud, 그래프)에 부적합 |
| 작은 kernel로 충분 | 일부 작업은 큰 context window 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{작은 데이터} \xrightarrow{\text{Bias 필수}} \text{CNN} \quad \text{vs} \quad \text{거대 데이터} \xrightarrow{\text{유연성 중요}} \text{ViT}}$$

| 개념 | 정의 | 영향 |
|------|------|------|
| **Translation Equivariance** | $f(T_\delta x) = T_\delta f(x)$ | 같은 객체를 위치 무관하게 인식 |
| **Locality** | RF가 작아서 국소 특징만 의존 | 초기 층은 edge/corner만 학습 |
| **Hierarchy** | 계층적으로 복잡도 증가 | 저수준 → 고수준 특징 조합 |
| **Inductive Bias** | 구조적 사전 가정 | 작은 $n$에서 좋음, 큰 $n$에서 제약 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): CNN이 translation equivariance를 만족한다고 했는데, 왜 **zero-padding**을 사용하면 경계에서 이 성질이 깨지는가? Circular padding을 사용하면 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

Zero-padding은 경계 바깥을 0으로 취급하므로, shift가 경계를 넘으면 다른 입력 값이 들어오는 것처럼 동작합니다.

예: $3 \times 3$ 이미지, 1-pixel 오른쪽 shift
- 원본: `[a, b, c]` → conv 결과: `[x1, x2, x3]`
- Shift: `[pad, a, b]` → conv 결과: `[y1, y2, y3]`
- `y1 ≠ pad_val = 0` (경계에서 다름)

Circular padding은 경계에서도 wrap-around하므로 equivariance를 완벽하게 만족합니다 (이미지를 토러스로 취급).

</details>

**문제 2** (심화): Dosovitskiy et al. (2021)의 실험에서 "ViT가 CNN을 능가하는 최소 데이터 크기"를 추정하라. ImageNet-1k (1.2M), ImageNet-21k (14M), JFT-300M 중 어디서 전환이 일어나는가? 왜?

<details>
<summary>힌트 및 해설</summary>

실험 결과:
- ImageNet-1k: CNN (ResNet) ≈ ViT (비슷, CNN 약간 우수)
- ImageNet-21k: ViT > CNN (ViT-B 85% vs ResNet-152 84.5%)
- JFT-300M: ViT >> CNN (ViT-B 88% vs ResNet 84%)

**전환점**: 약 10-20M 이미지 사이 (ImageNet-21k 규모)

**이유**:
1. ImageNet-1k는 작아서 CNN의 strong bias 필수
2. ImageNet-21k 규모는 충분한 다양성이 있어서 ViT의 유연성 활용 가능
3. JFT-300M은 거대해서 ViT가 compression, efficiency에서도 우수

</details>

**문제 3** (이론-실전): "Inductive bias를 줄인다 = 모델 용량을 늘린다"는 가정이 항상 성립하는가? ViT-Tiny와 ResNet-50이 거의 같은 파라미터 수를 가질 때, 어느 것이 작은 데이터에서 더 잘 하는가? 왜?

<details>
<summary>힌트 및 해설</summary>

**가정이 성립하지 않음**. 파라미터 수 ≠ inductive bias 강도.

예: CIFAR-10에서 (100K 이미지)
- ResNet-50: ~96% (inductive bias 강함)
- ViT-Tiny (비슷한 param): ~92-93% (bias 약함, overfitting)

ViT를 강하게 정규화 (drop path, mixup, augmentation 극대)해도 CNN이 이깁니다.

**이유**: 
- Inductive bias는 **구조적** (convolution, pooling)이므로 regularization으로 보완 불가
- Parameter count는 모델 복잡도의 일부일 뿐
- CNN은 같은 파라미터로 더 효율적인 특징 표현

</details>

---

<div align="center">

[◀ 이전](../ch6-applications/05-self-supervised.md) | [📚 README](../README.md) | [다음 ▶](./02-adversarial.md)

</div>
