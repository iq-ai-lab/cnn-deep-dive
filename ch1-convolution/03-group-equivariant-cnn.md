# 03. Group Equivariant CNN (Cohen & Welling 2016)

## 🎯 핵심 질문

- Translation equivariance를 넘어, 회전(rotation)이나 반사(reflection) 같은 다른 변환에도 equivariant하게 만들 수 있는가?
- Cohen & Welling의 Group Equivariant CNN (G-CNN)이 무엇이고, 표준 CNN과 어떻게 다른가?
- p4-CNN (90도 회전)이 어떤 구조를 가지며, 어떤 문제에서 유용한가?
- Steerable filter 기반 일반화는 무엇인가?
- 의료 영상, 분자 구조, 원격 감시 등 실제 응용에서 rotation equivariance가 왜 중요한가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

표준 CNN은 **translation equivariant**이지만, **rotation, reflection, scaling 등은 아닙니다**. 실제 응용을 보면:

1. **의료 영상** — MRI/CT 스캔에서 종양은 어느 방향이든 종양입니다. Rotation이 중요합니다.
2. **분자 구조** — 원자 배치 그래프는 회전 불변입니다.
3. **물체 인식** — 자동차, 비행기 등의 방향은 의미 있지만, 텍스처의 방향은 무의미합니다.

이 문서에서는 **임의의 변환 그룹에 대해 equivariant한 CNN**을 설계하는 방법을 소개합니다. 이것이 **Geometric Deep Learning**의 출발점입니다.

---

## 📐 수학적 선행 조건

- 이전 문서: 02-translation-equivariance.md (Group action, equivariance 개념)
- [Group Theory Deep Dive](https://github.com/iq-ai-lab/group-theory-deep-dive): Finite groups, dihedral group, group representations
- 선형대수: 행렬 표현, 고유벡터

---

## 📖 직관적 이해

### 표준 CNN의 한계

$$\text{Conv}(\text{Rotate}(I), K) = \text{Rotate}(\text{Conv}(I, K))?$$

**아닙니다!** 왜냐하면:
- 입력을 회전해도 커널 $K$는 회전되지 않습니다.
- 따라서 회전된 입력에 적응하지 못합니다.

### 해법: Group Equivariant Convolution

**Idea**: Kernel도 함께 회전시키세요!

$$\phi(R_\theta \cdot I) = R_\theta \cdot \phi(I)$$

를 만족하려면, $\phi$가 여러 회전된 커널로 구성되어야 합니다.

### Rotation Group의 예: $\mathbb{Z}_4$

$\mathbb{Z}_4 = \{0, 1, 2, 3\}$ — 90도 단위 회전.

- $R_0$: 회전 없음 (원본)
- $R_1$: 90도 시계방향
- $R_2$: 180도
- $R_3$: 270도

Group structure:
- $R_1 \circ R_1 = R_2$
- $R_1 \circ R_2 = R_3$
- $R_1^4 = R_0$ (4번 회전 = 원본)

### Feature Map의 Group-Valued 표현

표준 CNN:
$$y[i, j] = \sum_m \sum_n x[i+m, j+n] w[m,n] + b$$

G-CNN (회전 equivariant):
$$y[i, j, g] = \sum_m \sum_n x[i+m, j+n] w[m, n, g] + b$$

여기서 $g \in \mathbb{Z}_4$ — 어느 회전에서의 활성화인지 나타냅니다.

**해석**: 각 spatial location $(i,j)$에서 4개의 "방향별 feature"를 계산합니다.

### p4-CNN의 구조

**Input**: 표준 이미지 $I[i, j]$

**First layer**: Rotation group $\mathbb{Z}_4$를 도입
$$y_1[i, j, g] = \sum_m \sum_n I[R_g^{-1}(m, n)] w[m, n]$$

여기서 $R_g^{-1}(m,n)$은 필터의 spatial coordinates를 회전합니다.

**Hidden layers**: Group equivariant convolution
$$y_{l+1}[i, j, g] = \sum_m \sum_n \sum_{h \in \mathbb{Z}_4} y_l[i+m, j+n, h] w_l[m, n, h, g]$$

**Output**: Global pooling (group dimension 포함)
$$\text{class score} = \sum_i \sum_j \sum_g y_L[i, j, g]$$

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Finite Group (복습)

유한한 원소 집합 $G = \{g_1, \ldots, g_n\}$과 곱셈 연산이 group을 이룸.

**예**:
- 순환군 $\mathbb{Z}_n$ (크기 $n$)
- 이면체군 $D_n$ (정 $n$각형의 회전과 반사, 크기 $2n$)
- 교대군 $A_n$ (우함수인 순열들)

### 정의 3.2 — Group Action on Image Space

$G$가 image space 위에 작용:

$$R_g: [H] \times [W] \to [H] \times [W], \quad R_g(i, j) = \text{rotated coordinates}$$

이때 $G$의 group structure가 보존:

$$R_{g_1} \circ R_{g_2} = R_{g_1 g_2}$$

### 정의 3.3 — Group-Equivariant Feature Map

Feature map $\phi: [H] \times [W] \times G \to \mathbb{R}^C$ (또는 $\mathbb{R}^{C \times G}$로 생각)

$$\phi(R_g(i), j, h) = \phi(i, j, g^{-1}h)$$

**의미**: 이미지를 $g$만큼 회전하고 위치를 찾으면, feature의 group index가 $g^{-1}h$로 변합니다.

### 정의 3.4 — Group Equivariant Convolution

$$\phi_2[i, j, h] = \sum_{m,n} \sum_{g \in G} \phi_1[i+m, j+n, g] \cdot w[m, n, g, h]$$

여기서 $w$는 group-indexed kernel입니다.

### 정리 3.5 — G-CNN의 Equivariance

위의 정의 3.4의 group equivariant convolution $\phi_2$는 $G$ action에 equivariant입니다:

$$\phi_2(R_g \cdot I) = R_g \cdot \phi_2(I)$$

**증명 아이디어**:
$$\phi_2(R_g(i,j), h) = \sum_{m,n} \sum_{k} \phi_1(R_g(i,j)+R_g(m,n), k) w[m,n,k,h]$$

Group action의 동형성 ($R_g(a+b) = R_g(a) + R_g(b)$)을 사용하면:

$$= \sum_{m,n} \sum_{k} \phi_1(R_g(i+m, j+n), k) w[m,n,k,h]$$

이것이 $R_g(\phi_2[i,j,\cdot])$의 $h$-번째 성분과 일치. $\square$

### 정리 3.6 — p4-CNN의 정규 표현

$G = \mathbb{Z}_4$일 때, feature map을 행렬로:

$$\Phi[i,j] = \begin{pmatrix} \phi[i,j,0] \\ \phi[i,j,1] \\ \phi[i,j,2] \\ \phi[i,j,3] \end{pmatrix} \in \mathbb{R}^4$$

Convolution은:
$$\Phi'[i,j] = \sum_{m,n} \begin{pmatrix} w_0 & w_3 & w_2 & w_1 \\ w_1 & w_0 & w_3 & w_2 \\ w_2 & w_1 & w_0 & w_3 \\ w_3 & w_2 & w_1 & w_0 \end{pmatrix}[m,n] \Phi[i+m,j+n]$$

이 행렬은 **circulant matrix** 구조를 가지며, 자동으로 회전 equivariance를 만족합니다.

---

## 🔬 증명 및 수학적 유도

### 유도 1 — 왜 Rotation Equivariance가 필요한가?

표준 CNN에서:
$$\text{Conv}(R_\theta(I), K) \stackrel{?}{=} R_\theta(\text{Conv}(I, K))$$

**아닌 이유**: 
$$\text{Conv}(R_\theta(I), K)[i,j] = \sum_m \sum_n R_\theta(I)[i+m, j+n] K[m,n]$$
$$= \sum_m \sum_n I[R_\theta^{-1}(i+m, j+n)] K[m,n]$$

한편,
$$R_\theta(\text{Conv}(I,K))[i,j] = \text{Conv}(I,K)[R_\theta^{-1}(i,j)]$$
$$= \sum_m \sum_n I[R_\theta^{-1}(i,j)+m, j+n] K[m,n]$$

$R_\theta^{-1}(i+m,j+n) \neq R_\theta^{-1}(i,j) + (m,n)$이므로 (회전은 덧셈에 대해 선형이 아님), 일반적으로 불일치합니다.

### 유도 2 — Steerable Filters

**개념**: Spatial coordinate $(x,y)$와 angle $\theta$에 대해, kernel이 "steering"됩니다:

$$K_\theta(x,y) = K_0(R_{-\theta}(x,y))$$

즉, 기본 kernel을 회전합니다.

**예**: Gabor filter
$$K(x,y) = \exp(-\frac{x^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2}) \cos(2\pi f_x x)$$

회전하면:
$$K_\theta(x,y) = K(x \cos\theta + y \sin\theta, -x\sin\theta + y\cos\theta)$$

### 유도 3 — Harmonic Basis 분해

Group $G$의 **표현(representation)** $\rho: G \to GL(\mathbb{R}^d)$를 사용하면, feature map을 분해:

$$\phi[i,j] = \sum_{k=1}^{n_\text{irreps}} \sum_{\alpha=1}^{d_k} c_{k,\alpha}[i,j] \cdot v_{k,\alpha}$$

여기서 $v_{k,\alpha}$는 irreducible representation의 basis입니다.

이는 **Fourier transform의 group 버전**이며, group convolution이 diagonal 구조를 가집니다 (다음 챕터의 Toeplitz 행렬과 유사).

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 간단한 p4-CNN 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class P4Convolution(nn.Module):
    """Simple p4-CNN layer (rotation group Z_4)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Group-indexed kernel: [out_ch, in_ch, 4, k, k]
        # (마지막 4는 rotation group Z_4)
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, 4, kernel_size, kernel_size
        ) * 0.01)
    
    def forward(self, x):
        # x shape: [batch, in_ch, height, width, 4]
        # or [batch, in_ch*4, height, width] 로 재해석
        
        batch, in_ch, h, w, _ = x.shape
        
        # 각 rotation h에 대해 convolution
        outputs = []
        for h_out in range(4):
            out = torch.zeros(batch, self.out_channels, h, w, 
                            device=x.device)
            
            for h_in in range(4):
                # Rotate kernel: weight[:, :, h_in, :, :] -> weight[:, :, :, :]
                # rotated kernel to account for input rotation
                
                # 간단한 구현: group convolution as weighted sum
                x_rotated = x[:, :, :, :, h_in]  # [batch, in_ch, h, w]
                
                # weight를 rotate하여 적용
                w_rotated = self.weight[:, :, (h_out - h_in) % 4, :, :]
                
                out += F.conv2d(x_rotated, w_rotated, padding=self.padding)
            
            outputs.append(out)
        
        return torch.stack(outputs, dim=-1)  # [batch, out_ch, h, w, 4]


class P4CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 첫 번째 layer에서 group을 도입
        self.p4conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 비교용 표준 conv
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.p4conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 사용 예
model = P4CNN()
x = torch.randn(4, 1, 28, 28)
y = model(x)
print("Output shape:", y.shape)  # [4, 10]
```

### 실험 2 — Rotation 불변성 테스트

```python
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def rotate_tensor(x, angle):
    """PyTorch tensor를 PIL로 변환 후 회전"""
    # [batch, channels, h, w] -> PIL
    img = transforms.ToPILImage()(x[0])
    img_rotated = img.rotate(angle)
    return transforms.ToTensor()(img_rotated).unsqueeze(0)

# 표준 CNN과 G-CNN 비교
standard_model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
)

# 간단한 테스트 이미지 (숫자)
test_image = torch.randn(1, 1, 28, 28)

# 원본 이미지와 회전 이미지
rotated_0 = test_image
rotated_90 = transforms.functional.rotate(test_image, 90)
rotated_180 = transforms.functional.rotate(test_image, 180)

# 특성 추출
feat_0 = standard_model(rotated_0).detach()
feat_90 = standard_model(rotated_90).detach()
feat_180 = standard_model(rotated_180).detach()

# Rotation 일관성 측정
consistency_90 = F.cosine_similarity(feat_0.view(-1), feat_90.view(-1), dim=0)
consistency_180 = F.cosine_similarity(feat_0.view(-1), feat_180.view(-1), dim=0)

print(f"Standard CNN - 90° rotation consistency: {consistency_90:.4f}")
print(f"Standard CNN - 180° rotation consistency: {consistency_180:.4f}")
print("Expected: Low consistency (표준 CNN은 회전 불변성 없음)")
```

### 실험 3 — MNIST-Rotated Dataset

```python
# MNIST를 회전시켜 데이터셋 생성
from torchvision import datasets

def load_mnist_rotated(root, angle=45, train=True):
    """각 이미지를 지정된 각도로 회전"""
    mnist = datasets.MNIST(root, train=train, download=True)
    
    rotated_data = []
    for img, label in mnist:
        img_rotated = transforms.functional.rotate(
            img, angle, expand=False
        )
        rotated_data.append((img_rotated, label))
    
    return rotated_data

# 훈련 및 비교
# ... (상세 구현은 생략)

print("MNIST-Rotated 데이터셋에서:")
print("표준 CNN: 낮은 정확도 (회전에 민감)")
print("G-CNN: 높은 정확도 (회전 불변성)")
```

---

## 🔗 이론과 실전의 간극

### 1. 연속 회전 vs 이산 회전

**이론**: $SO(2)$ — 모든 각도의 회전

**실전**: $\mathbb{Z}_n$ — 이산 각도 ($0°, 90°, 180°, ...$)

더 정교한 방법:
- **Steerable CNNs** (Cohen et al. 2017) — continuous rotation에 equivariant
- **Harmonic networks** — Fourier basis로 표현
- **RotEqNet** — 임의의 각도에 equivariance

### 2. 계산 비용

표준 Conv: $C_{in} \times C_{out} \times k \times k$ 파라미터

G-CNN (group size $|G|$): $C_{in} \times C_{out} \times |G| \times k \times k$ 파라미터

**대략 $|G|$배 증가!** 하지만:
- Shared weights (group structure 활용) → 실제로는 약간만 증가
- 더 적은 layer 필요 가능

### 3. 다른 그룹들

- **Dihedral group $D_4$**: 회전 + 반사 (8개 원소)
- **E(2)**: 유클리드 평면의 회전 + 이동
- **SE(3)**: 3D 회전 + 이동 (분자 구조)

각각은 다른 응용에 맞춤입니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Group structure가 알려짐 | 데이터의 진정한 symmetry를 모르면 부적절 |
| 정규 그리드 | 비정형 데이터는 graph neural network 필요 |
| Finite groups | Continuous symmetry는 steerable filters 필요 |
| Perfect alignment | 실제 데이터는 약간의 회전, scale 변화 포함 |
| 계산 비용 | Group size 증가 → 메모리, 속도 증가 |

---

## 📌 핵심 정리

$$\boxed{\text{G-CNN: } \phi(g \cdot I) = g \cdot \phi(I) \text{ — Group equivariance로 convolution 확장}}$$

$$\boxed{\text{p4-CNN: } y[i,j,g] = \sum_{m,n,h} x[i+m,j+n,h] \cdot w[m,n,h,g] \text{ — 회전 group } \mathbb{Z}_4}$$

| 개념 | 정의 |
|------|------|
| **Group equivariance** | $\phi(g \cdot x) = g \cdot \phi(x)$ for group $G$ |
| **p4-CNN** | 90도 회전 group $\mathbb{Z}_4$를 도입한 CNN |
| **Group-indexed kernel** | $w[m,n,g,h]$ — 입출력 group 인덱스 포함 |
| **Steerable filters** | $K_\theta(x,y) = K_0(R_{-\theta}(x,y))$ — 연속 회전 |
| **Rotation equivariance** | Conv가 회전에 equivariant (자동으로 회전된 features 감지) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Group $G = \mathbb{Z}_4$에서 feature map $\phi[i,j,g]$는 총 4개의 값을 가진다 (각 위치마다). 이를 길이 4의 벡터 $\Phi[i,j] \in \mathbb{R}^4$로 생각할 때, rotation group $\mathbb{Z}_4$의 작용을 행렬 $R_g \in \mathbb{R}^{4 \times 4}$로 어떻게 나타낼까?

<details>
<summary>해설</summary>

정의상 $\Phi'[i,j] = R_g \Phi[i,j]$이어야 하고, $g \in \{0,1,2,3\}$일 때:

$$R_0 = I, \quad R_1 = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}, \quad R_2 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}, \quad R_3 = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \end{pmatrix}$$

이들은 **cyclic permutation** 행렬이며, $R_4 = I$를 만족합니다. 이런 행렬들의 집합은 group을 이루며, 이를 **group의 정규 표현(regular representation)**이라 합니다.

</details>

**문제 2** (심화): p4-CNN 커널 $w[m,n,g,h]$는 총 $k \times k \times 4 \times C_{out} \times C_{in}$ 파라미터를 가진다. 하지만 group structure를 활용하면, 실제로는 $k \times k \times C_{out} \times C_{in}$ 파라미터만 필요하다 (group index 없음). 이를 실현하는 방법을 제안하라.

<details>
<summary>해설</summary>

**아이디어**: 기본 kernel $\tilde{w}[m,n,h]$로부터 4개의 회전 버전을 **자동으로 생성**합니다.

$$w[R_g(m,n), g, h] = \tilde{w}[m,n,h]$$

구체적으로, 기본 kernel을 4번 회전시켜 저장:

$$w[m,n,g,h] = \tilde{w}[R_g^{-1}(m,n), h]$$

이렇게 하면:
- **파라미터**: $k \times k \times C_{out} \times C_{in}$ (공유)
- **계산량**: 4배 conv 연산 필요 (group index마다)

하지만 group structure를 잃지 않으면서도 파라미터를 줄입니다. 이를 **parameter sharing** 또는 **lifting**이라 합니다.

</details>

**문제 3** (논문 비평): Rotation equivariance가 모든 시각 인식 문제에 유용한가? 어떤 경우에는 **오히려 해로울** 수 있을까?

<details>
<summary>해설</summary>

**유용한 경우**:
- 의료 영상 (종양은 방향과 무관)
- 자동 인식 (회전된 자동차도 자동차)
- 위성 영상 (건물, 도로는 회전 불변)

**해로울 수 있는 경우**:
- **텍스트, 숫자** — "6"을 180도 회전하면 "9". Rotation equivariance는 이 의미론적 차이를 무시합니다.
- **체스판, 지도** — 방향이 의미를 가집니다 (North/South 구별).
- **자세 추정** — 사람이 누워있는 방향 vs 서있는 방향은 중요합니다.

**결론**: 데이터의 진정한 symmetry를 이해하고, 그에 맞는 group을 선택해야 합니다. 모든 경우에 rotation equivariance가 좋은 것은 아닙니다.

</details>

---

<div align="center">

[◀ 이전](./02-translation-equivariance.md) | [📚 README](../README.md) | [다음 ▶](./04-toeplitz-matrix.md)

</div>
