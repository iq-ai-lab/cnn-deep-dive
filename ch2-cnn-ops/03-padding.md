# 03. Padding 전략과 Boundary 효과

## 🎯 핵심 질문

- Zero padding, reflection padding, replicate padding의 수학적 정의와 실제 영향은?
- 'Same' convolution에서 $p = (k-1)/2$인 이유는 무엇인가?
- Boundary에서 zero padding이 만드는 "dark halo" artifact는 무엇이고, 왜 reflection padding이 이를 줄이는가?
- Boundary information loss와 feature detection 간의 trade-off는?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Convolution은 기본적으로 **경계 처리 문제**를 가집니다. Kernel이 $3 \times 3$이면, 이미지 경계의 픽셀은 완전한 receptive field를 가질 수 없습니다. 이를 어떻게 처리할지에 따라:

1. 출력 크기가 달라집니다.
2. Boundary region의 feature extraction 품질이 달라집니다.
3. 학습된 모델의 translation invariance가 달라집니다.

예를 들어, 의료 영상이나 얼굴 인식에서 경계 픽셀도 중요할 수 있습니다. Zero padding을 사용하면 "가짜" 경계가 생겨 artifact를 만들 수 있습니다. 반면 reflection padding은 이미지 내용을 반사시켜 더 자연스러운 boundary를 만듭니다. 이 문서에서는 **padding 전략의 수학과 실제 영향**을 정확히 이해합니다.

---

## 📐 수학적 선행 조건

- [Ch2-01: Convolution Forward/Backward](./01-conv-forward-backward.md): Convolution의 정의, boundary handling
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix padding, reshaping
- [Signal Processing Deep Dive](https://github.com/iq-ai-lab/signal-processing-deep-dive): Boundary conditions in filtering

---

## 📖 직관적 이해

### Zero Padding: 경계를 0으로 채우기

가장 간단한 padding. 입력 주변을 0으로 채웁니다:

$$X_{\text{padded}}[c, i, j] = \begin{cases} X[c, i-p, j-p] & \text{if } 0 \leq i-p < H, 0 \leq j-p < W \\ 0 & \text{otherwise} \end{cases}$$

**장점**: 구현이 간단, 계산 효율 좋음.

**단점**: Boundary 근처에서 "dark halo"가 생길 수 있습니다. 예를 들어 밝은 이미지 경계를 convolution하면, 0 padding의 영향으로 그 근처 output이 어두워질 수 있습니다.

### Reflection Padding: 경계를 반사시키기

경계를 입상 내용을 대칭으로 반사시킵니다:

$$X_{\text{reflected}}[c, i, j] = \begin{cases} X[c, i-p, j-p] & \text{if } 0 \leq i-p < H, 0 \leq j-p < W \\ X[c, 2(p-1) - (i-p), j-p] & \text{if } i-p < 0 \text{ (위쪽 경계)} \\ \cdots \end{cases}$$

더 정확하게, position $i-p$가 음수면 $X[c, -(i-p)-1, j-p]$를 사용.

**장점**: 자연스러운 boundary, texture synthesis에서 artifact 감소.

**단점**: 구현이 복잡, 약간의 계산 오버헤드.

### Replicate Padding: 경계 픽셀 반복

경계 픽셀을 그대로 반복합니다:

$$X_{\text{replicated}}[c, i, j] = \begin{cases} X[c, \max(0, \min(H-1, i-p)), \max(0, \min(W-1, j-p))] \end{cases}$$

**장점**: Zero padding보다는 자연스럽지만, reflection만큼 복잡하지 않음.

**단점**: Sharp edge가 생길 수 있음.

### 'Same' vs 'Valid' Convolution

- **'Valid'**: Padding 없음. Output size = $\lfloor (H - K) / s \rfloor + 1$. 경계에서 정보 손실.
- **'Same'**: Padding으로 출력 크기 = 입력 크기 유지. $p = (K - 1) / 2$ (stride = 1일 때).

'Same'을 사용하면 layer를 깊게 쌓을 수 있지만, 경계 information이 왜곡됩니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.10 — Zero Padding

Input $X \in \mathbb{R}^{C \times H \times W}$에 padding size $p$를 적용:

$$X^{(p)}[c, i, j] := \begin{cases} X[c, i-p, j-p] & \text{if } p \leq i < H+p, p \leq j < W+p \\ 0 & \text{otherwise} \end{cases}$$

Padded shape: $C \times (H + 2p) \times (W + 2p)$.

### 정의 2.11 — Reflection Padding

$$X^{(r)}[c, i, j] := \begin{cases} X[c, i-p, j-p] & \text{if } p \leq i < H+p, p \leq j < W+p \\ X[c, p-i-1, j-p] & \text{if } 0 \leq i < p, p \leq j < W+p \text{ (top)} \\ X[c, i-p, p-j-1] & \text{if } p \leq i < H+p, 0 \leq j < p \text{ (left)} \\ X[c, 2p-i-1, 2p-j-1] & \text{if } i < p, j < p \text{ (corner)} \end{cases}$$

### 정의 2.12 — 'Same' Convolution

Stride $s = 1$일 때 output size = input size가 되도록 padding을 설정:

$$p_{\text{same}} = \frac{K - 1}{2}$$

(K가 홀수일 때만 정확함. 짝수이면 asymmetric padding 필요)

### 정리 2.13 — Output Size 공식

Padding $p$, kernel size $K$, stride $s$일 때:

$$H_{\text{out}} = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1$$

---

## 🔬 증명 또는 수학적 유도

### Boundary Artifact 분석: Zero Padding vs Reflection

Simple example: 1D signal $x = [1, 1, 1]$, kernel $k = [1, 1]$.

**Zero padding ($p=1$):**
$$x^{(p)} = [0, 1, 1, 1, 0]$$
$$y = [0 \cdot 1 + 1 \cdot 1, 1 \cdot 1 + 1 \cdot 1, 1 \cdot 1 + 1 \cdot 1, 1 \cdot 1 + 0 \cdot 1] = [1, 2, 2, 1]$$

경계에서 값이 작음 (boundary dip).

**Reflection padding ($p=1$):**
$$x^{(r)} = [1, 1, 1, 1, 1]$$ (경계를 반사)
$$y = [1 \cdot 1 + 1 \cdot 1, 1 \cdot 1 + 1 \cdot 1, \ldots] = [2, 2, 2, 2]$$

전체 균일. $\square$

### 'Same' Padding 필요 조건

Output size 공식에서:
$$H_{\text{out}} = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1 = H$$

$s = 1$일 때:
$$H + 2p - K + 1 = H$$
$$p = \frac{K - 1}{2}$$

따라서 $K$가 홀수일 때만 정수 padding이 가능. $\square$

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — 다양한 Padding 방식의 영향

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 간단한 경계 이미지: 흰색 배경에 검은 테두리
X = torch.ones(1, 1, 8, 8)
X[:, :, 1:-1, 1:-1] = 0  # 가운데는 검음
print("Input image (1 = white, 0 = black):")
print(X[0, 0])

# 3x3 edge detection kernel (Sobel-like)
K = torch.tensor([[[[-1., 0., 1.],
                     [-2., 0., 2.],
                     [-1., 0., 1.]]]])

# 다양한 padding 적용
methods = {
    'zero': F.pad(X, (1, 1, 1, 1), mode='constant', value=0),
    'reflect': F.pad(X, (1, 1, 1, 1), mode='reflect'),
    'replicate': F.pad(X, (1, 1, 1, 1), mode='replicate'),
}

results = {}
for name, X_padded in methods.items():
    Y = F.conv2d(X_padded, K, padding=0)
    results[name] = Y
    print(f"\n{name.upper()} padding result:")
    print(Y[0, 0])

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(X[0, 0], cmap='gray')
axes[0, 0].set_title('Input')
for (name, Y), ax in zip(results.items(), axes.flatten()[1:]):
    ax.imshow(Y[0, 0].detach(), cmap='RdBu')
    ax.set_title(f'{name.capitalize()} Padding')
plt.tight_layout()
plt.show()
```

**예상**: Zero padding이 경계에서 artifact를 보이고, reflection/replicate가 더 부드러움.

### 실험 2 — 'Same' Convolution 검증

```python
# 3x3 kernel에서 'same' convolution
X = torch.randn(1, 3, 32, 32)
K = torch.randn(16, 3, 3, 3)

# 'same' padding: p = (3-1)/2 = 1
Y_same = F.conv2d(X, K, padding=1, stride=1)

print("Input shape:", X.shape)
print("Output shape (with 'same' padding):", Y_same.shape)
print("Shapes match:", Y_same.shape[2:] == X.shape[2:])

# 다른 padding으로 비교
Y_valid = F.conv2d(X, K, padding=0, stride=1)
print("Output shape (with 'valid'):", Y_valid.shape)
print("Output size reduced by:", X.shape[2] - Y_valid.shape[2])
```

**예상**: 'Same' padding은 출력 크기를 입력과 동일하게 유지.

### 실험 3 — Boundary Information Loss

```python
# 경계에 중요한 특징이 있는 경우
X_boundary = torch.zeros(1, 1, 8, 8)
X_boundary[:, :, 0, :] = 1  # 위쪽 경계가 1
X_boundary[:, :, -1, :] = 1  # 아래쪽 경계가 1

K = torch.ones(1, 1, 3, 3) / 9  # Average pooling kernel

# Valid convolution (padding 없음)
Y_valid = F.conv2d(X_boundary, K, padding=0)

# Same convolution (padding=1)
Y_same = F.conv2d(X_boundary, K, padding=1)

print("Boundary image (경계가 1):")
print(X_boundary[0, 0])

print("\nValid convolution output (경계 정보 손실):")
print(Y_valid[0, 0])

print("\nSame convolution output (경계 정보 partially 유지):")
print(Y_same[0, 0])

# 수치 비교
print("\nBoundary loss: Valid output의 크기가 (6, 6)이므로 위아래쪽 정보 손실")
print("Same output의 첫줄: ", Y_same[0, 0, 0, :])
print("(경계 근처이므로 0과 1 사이의 값)")
```

**예상**: Valid는 경계를 완전히 놓침, Same은 경계를 부분적으로 유지 (padding 때문).

### 실험 4 — Padding 방식의 에너지

```python
# Gradient 관점에서의 차이
X = torch.randn(1, 1, 4, 4, requires_grad=True)
K = torch.randn(1, 1, 3, 3, requires_grad=True)

# Zero padding backward
Y_zero = F.conv2d(X, K, padding=1)
loss_zero = Y_zero.sum()
loss_zero.backward()
grad_zero = X.grad.clone()
X.grad.zero_()

# Reflection padding
X_refl_padded = F.pad(X, (1, 1, 1, 1), mode='reflect')
Y_refl = F.conv2d(X_refl_padded, K, padding=0)
loss_refl = Y_refl.sum()
loss_refl.backward()
grad_refl = X.grad

print("Gradient magnitude with zero padding:", grad_zero.norm().item())
print("Gradient with reflection:", grad_refl.norm().item())
print("(차이는 boundary handling에서 기인)")
```

---

## 🔗 이론과 실전의 간극

### Real-world Padding 전략

실제 응용에서:

1. **분류 작업 (ImageNet)**: Zero padding 사용 (속도, 간단함). Boundary information이 중요하지 않음.
2. **의료 영상**: Reflection/replicate padding. Boundary artifact가 진단에 영향 줄 수 있음.
3. **Texture synthesis / Style transfer**: Reflection padding. 자연스러운 경계가 중요.
4. **Object detection**: 다양한 padding 사용, 보통 zero도 괜찮음 (RPN이 경계 수정).

### Symmetric vs Asymmetric Padding

Stride가 2 이상일 때나 kernel이 짝수일 때는 **asymmetric** padding이 필요할 수 있습니다:

$$p_{\text{left}} \neq p_{\text{right}}$$

PyTorch의 일부 layer (예: Conv2d)는 이를 자동으로 처리하지 않으므로, `F.pad()` 다음 `conv2d()`를 순차적으로 사용해야 합니다.

---

## ⚖️ 가정과 한계

| 가정 | 설명 | 한계 |
|------|------|------|
| Rectangular image | 정사각형/직사각형 | 원형, 다각형 이미지는 다른 방식 필요 |
| Uniform padding | 모든 방향 같은 padding | Asymmetric padding도 사용 가능 |
| Static padding strategy | 고정된 padding 방식 | Learnable padding (adversarial examples 대응)도 연구 중 |
| 경계 = 이미지 외부 | Boundary 처리 방식만 고려 | Circular padding (cyclic 이미지)도 가능 |

---

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| **Zero Padding** | 경계를 0으로 채움, 간단하지만 artifact 가능 |
| **Reflection Padding** | 경계를 입상 내용으로 대칭 반사 |
| **Replicate Padding** | 경계 픽셀 반복 |
| **'Same' Convolution** | $p = (K-1)/2$로 output size = input size 유지 |
| **Boundary Artifact** | Zero padding이 경계 근처에서 "halo" 효과 생성 |

---

## 🤔 생각해볼 문제

**문제 1 (기초).** $3 \times 3$ kernel에서 'same' convolution을 하려면 padding은 몇일까? 출력 크기는?

<details>
<summary>해설</summary>

$p = (3 - 1) / 2 = 1$.

Output size: $\lfloor (H + 2 \cdot 1 - 3) / 1 \rfloor + 1 = H + 2 - 3 + 1 = H$.

따라서 output size = input size.

</details>

**문제 2 (심화).** Zero padding에서 boundary artifact가 왜 생기는가? 신호 처리 관점에서 설명하라.

<details>
<summary>해설</summary>

Zero padding은 입력 신호에 **급격한 step discontinuity**를 도입합니다. 예를 들어 흰 배경($x=1$)을 0으로 padding하면, 경계에서 $1 \to 0$의 jump가 생깁니다.

Convolution (low-pass filtering)은 이 jump 주변에서 **transition region을 만들**어 경계 근처가 어두워집니다. 이것이 "dark halo"입니다.

Reflection padding은 신호를 연속적으로 확장하므로 discontinuity가 없고, 따라서 artifact도 적습니다.

</details>

**문제 3 (논문 비평).** 최근 Vision Transformer(ViT)는 padding의 개념이 없습니다. 왜 그렇고, 대신 무엇을 사용할까요?

<details>
<summary>해설</summary>

ViT는:
1. 입력을 patch로 나누어 patch 자체를 token으로 사용 → padding이 불필요.
2. Self-attention은 모든 위치가 global context를 볼 수 있음 → local padding의 의미 감소.
3. Positional encoding으로 위치 정보를 직접 인코딩.

따라서 padding strategy가 ViT에서는 거의 관련 없습니다. CNN과의 주요 차이점 중 하나입니다.

</details>

---

<div align="center">

[◀ 이전](./02-pooling.md) | [📚 README](../README.md) | [다음 ▶](./04-stride-dilation-transposed.md)

</div>
