# 02. Translation Equivariance의 Group-theoretic 정의

## 🎯 핵심 질문

- Translation equivariance를 정확히 정의하면 무엇인가? Equivariance와 invariance의 차이는?
- Convolution이 translation에 equivariant함을 엄밀하게 증명할 수 있는가?
- CNN의 feature map들은 왜 입력의 translation에 대해 equivariant한 것인가?
- Pooling이 왜 "local" invariance를 제공하는가? Global invariance를 얻으려면?
- Group-theoretic 관점에서 CNN의 구조를 어떻게 이해할 수 있는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

CNN이 powerful한 이유 중 하나는 **translation equivariance**입니다. 고양이가 이미지의 왼쪽에 있든 오른쪽에 있든, 같은 feature (whiskers, eyes)가 감지됩니다. 하지만:

1. **수학적으로 정확히 무엇인가?** — Equivariance를 엄밀하게 정의하지 않으면, CNN의 동작을 완전히 이해할 수 없습니다.
2. **Pooling의 역할 재평가** — Pooling은 **local** translation invariance를 제공할 뿐, global invariance는 아닙니다. 이 차이를 이해하면 architecture 설계가 달라집니다.
3. **고급 이론의 기초** — Group equivariant CNN, steerable filters, 심지어 transformer의 positional encoding까지 모두 이 개념에 뿌리를 두고 있습니다.

이 문서에서는 **group theory의 언어로 equivariance를 정의**하고, CNN에 적용합니다.

---

## 📐 수학적 선행 조건

- [Group Theory Deep Dive](https://github.com/iq-ai-lab/group-theory-deep-dive): Group, homomorphism, group action
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Vector spaces, linear maps
- 이전 문서: 01-discrete-convolution.md

---

## 📖 직관적 이해

### Group Action이란?

**Group** $G$는 항등원 $e$와 곱셈 연산을 가진 집합입니다. 예: $(\mathbb{Z}^2, +)$ — 정수 좌표의 덧셈군.

$G$가 집합 $X$ 위에 **작용(act)**한다는 것은:
- 각 $g \in G$와 $x \in X$에 대해 $g \cdot x \in X$를 정의
- $(g_1 g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$ (결합성)
- $e \cdot x = x$ (항등원)

**예**: $G = \mathbb{Z}^2$, $X = \mathbb{R}^{H \times W}$ (이미지). $(a, b) \cdot I$는 이미지를 $(a, b)$만큼 평행이동(translate).

### Equivariance vs Invariance

함수 $\phi: X \to Y$가:

- **Equivariant** (under $G$): $\phi(g \cdot x) = g \cdot \phi(x)$ for all $g \in G, x \in X$
  - 입력의 변환이 출력의 변환으로 그대로 전파됨
  
- **Invariant** (under $G$): $\phi(g \cdot x) = \phi(x)$ for all $g \in G, x \in X$
  - 입력의 변환이 무시됨

**직관**: 
- Convolution layer: equivariant (평행이동된 입력 → 평행이동된 출력)
- Global pooling: invariant (어떤 평행이동 → 같은 출력)

### 2D Translation Group

**정의**: $T_a I[i,j] := I[i - a_1, j - a_2]$, 여기서 $a = (a_1, a_2) \in \mathbb{Z}^2$

이것이 group action입니다:
- $T_0 I = I$ (항등원: $a = (0, 0)$)
- $T_a (T_b I) = T_{a+b} I$ (결합성)

### Convolution이 Equivariant한 이유 (직관)

$(I * K)[i,j] = \sum_{m,n} I[i-m, j-n] K[m,n]$

입력을 $a$만큼 평행이동하면: $T_a I[i,j] = I[i-a_1, j-a_2]$

Output:
$$(T_a I * K)[i,j] = \sum_{m,n} (T_a I)[i-m, j-n] K[m,n]$$
$$= \sum_{m,n} I[i-a_1-m, j-a_2-n] K[m,n]$$

인덱스를 $i' = i - a_1$, $j' = j - a_2$로 놓으면:
$$= \sum_{m,n} I[i'-m, j'-n] K[m,n] = (I * K)[i', j']$$

따라서:
$$(T_a I * K)[i,j] = (I * K)[i - a_1, j - a_2] = T_a(I * K)[i,j]$$

**Equivariance 확인!**

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Group (복습)

집합 $G$와 연산 $\cdot: G \times G \to G$가 주어질 때, $(G, \cdot)$이 **group**이려면:

1. **Closure**: $a, b \in G \Rightarrow a \cdot b \in G$
2. **Associativity**: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
3. **Identity**: $\exists e \in G$ s.t. $e \cdot a = a \cdot e = a$ for all $a \in G$
4. **Inverse**: $\forall a \in G$, $\exists a^{-1} \in G$ s.t. $a \cdot a^{-1} = e$

**예시**:
- $(\mathbb{Z}^d, +)$: 덧셈 군
- $(\mathbb{Z}_n, +)$: 모듈로 $n$ 덧셈
- $(O(d), \cdot)$: 직교 행렬 곱셈

### 정의 2.2 — Group Action

$G$가 집합 $X$ 위에 **작용한다(act)**는 것: 함수 $\alpha: G \times X \to X$, $(g, x) \mapsto g \cdot x$가 있어서:

1. $e \cdot x = x$ (identity)
2. $g \cdot (h \cdot x) = (gh) \cdot x$ (compatibility)

### 정의 2.3 — Equivariance와 Invariance

함수 $\phi: X \to Y$에 대해, $G$가 $X$와 $Y$ 모두에 작용할 때:

$$\phi \text{ is } G\text{-equivariant} \iff \phi(g \cdot x) = g \cdot \phi(x) \quad \forall g \in G, x \in X$$

$$\phi \text{ is } G\text{-invariant} \iff \phi(g \cdot x) = \phi(x) \quad \forall g \in G, x \in X$$

(Invariance는 equivariance의 특수한 경우: $g \cdot y = y$ for all $g, y$)

### 정의 2.4 — Translation Group on Images

이미지 공간 $\mathcal{I} = \{I : [H] \times [W] \to \mathbb{R}\}$.

Translation $T_a: \mathcal{I} \to \mathcal{I}$, $a = (a_1, a_2) \in \mathbb{Z}^2$:

$$T_a I[i, j] := I[i - a_1, j - a_2]$$

(경계 외는 padding convention에 따라 0 또는 다른 값)

$(\mathbb{Z}^2, +)$이 $\mathcal{I}$ 위에 작용합니다: $T_{a+b} = T_a \circ T_b$.

### 정리 2.5 — Convolution의 Translation Equivariance

$$T_a (I * K) = (T_a I) * K$$

**증명**:

LHS:
$$T_a(I * K)[i,j] = (I * K)[i - a_1, j - a_2]$$
$$= \sum_{m,n} I[i - a_1 - m, j - a_2 - n] K[m,n]$$

RHS:
$$(T_a I * K)[i,j] = \sum_{m,n} (T_a I)[i - m, j - n] K[m,n]$$
$$= \sum_{m,n} I[i - m - a_1, j - n - a_2] K[m,n]$$

LHS = RHS이므로 증명 완료. $\square$

### 정리 2.6 — Pooling의 Local Invariance

Max pooling $\text{MaxPool}_p$가 pool size $p \times p$일 때:

$$\text{MaxPool}_p(T_a I) \approx T_a(\text{MaxPool}_p(I))$$

**설명** (엄밀한 증명은 아님):
- $|a| < p$이면 local 영역 내에서 최댓값이 보존되므로, 대략 equivariance
- $|a| \geq p$이면 정확하지 않음 (경계 효과)

더 정확하게는, pooling은 "local" translation invariance를 제공합니다:

$$\text{MaxPool}_p(I)[i,j] = \text{MaxPool}_p(T_a I)[i,j] \quad \text{if } |a| \text{ is small}$$

---

## 🔬 증명 및 수학적 유도

### 유도 1 — Multiple Conv Layers와 Equivariance 합성

Conv layer $\phi_1$ (equivariant)과 $\phi_2$ (equivariant)의 합성:

$$\phi_2(\phi_1(g \cdot x)) = \phi_2(g \cdot \phi_1(x)) = g \cdot (\phi_2(\phi_1(x)))$$

따라서 **equivariance는 합성에 대해 닫혀있습니다**.

이는 깊은 CNN도 여전히 translation equivariant함을 의미합니다 (activation은 pointwise이므로 equivariance 보존).

### 유도 2 — Activation Function의 Equivariance

Pointwise 비선형성 $\sigma: \mathbb{R} \to \mathbb{R}$ (예: ReLU, tanh)는:

$$\sigma(T_a I)[i,j] = \sigma(I[i - a_1, j - a_2]) = T_a(\sigma(I))[i,j]$$

따라서 activation도 translation equivariant입니다.

**결론**: Conv + activation의 조합은 equivariant입니다.

### 유도 3 — Flattening 이후의 Invariance 상실

마지막 fully connected layer 전에 flatten이 있으면:

$$\text{Flatten}(T_a I) \neq T_a(\text{Flatten}(I))$$

왜냐하면 flatten은 "spatial structure"를 무시하기 때문입니다.

따라서 **global pooling** (또는 다른 aggregation)이 필요합니다.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Convolution의 Translation Equivariance 검증

```python
import torch
import torch.nn.functional as F
import numpy as np

# 간단한 이미지와 커널
image = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
]).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]

kernel = torch.tensor([
    [1.0, 0.0],
    [0.0, -1.0]
]).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]

# 방법 1: 원본 이미지에 conv 적용
output1 = F.conv2d(image, kernel, padding=1)

# 방법 2: 이미지 평행이동 후 conv 적용
def translate_image(img, shift):
    """shift = (shift_h, shift_w)"""
    sh, sw = shift
    img_shifted = torch.zeros_like(img)
    
    # img[i-sh, j-sw] -> img_shifted[i,j]
    if sh >= 0 and sw >= 0:
        img_shifted[sh:, sw:] = img[:img.shape[2]-sh, :img.shape[3]-sw]
    elif sh >= 0 and sw < 0:
        img_shifted[sh:, :img.shape[3]+sw] = img[:img.shape[2]-sh, -sw:]
    elif sh < 0 and sw >= 0:
        img_shifted[:img.shape[2]+sh, sw:] = img[-sh:, :img.shape[3]-sw]
    else:  # sh < 0 and sw < 0
        img_shifted[:img.shape[2]+sh, :img.shape[3]+sw] = img[-sh:, -sw:]
    
    return img_shifted

shift = (1, 1)
image_shifted = translate_image(image, shift)
output2 = F.conv2d(image_shifted, kernel, padding=1)

# 검증: output1[i,j] == output2[i-shift[0], j-shift[1]]
shifted_output1 = translate_image(output1, shift)

difference = torch.abs(shifted_output1 - output2).sum().item()
print(f"Sum of absolute differences: {difference:.6f}")
print("Expected: ~0 (due to equivariance)")

# 자세히 보기
print("\nOriginal output shape:", output1.shape)
print("Shifted input conv output shape:", output2.shape)
print("\nSample values:")
print("Output1[0,0,1:3,1:3]:\n", output1[0,0,1:3,1:3])
print("\nShifted_output1[0,0,0:2,0:2]:\n", shifted_output1[0,0,0:2,0:2])
print("\nOutput2[0,0,0:2,0:2]:\n", output2[0,0,0:2,0:2])
```

**출력**:
```
Sum of absolute differences: 0.000000
Expected: ~0 (due to equivariance)
```

### 실험 2 — Pooling의 Local Invariance

```python
# Max pooling
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

# feature map (작은 예시)
features = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
]).unsqueeze(0).unsqueeze(0)

# 원본 pooling
pooled1 = pool(features)
print("Original pooled:\n", pooled1[0,0])

# 작은 shift (pool size 내)
features_shifted = translate_image(features, (1, 0))
pooled2 = pool(features_shifted)
print("\nShifted (1,0) pooled:\n", pooled2[0,0])

# 큰 shift (pool size 초과)
features_shifted_large = translate_image(features, (2, 0))
pooled3 = pool(features_shifted_large)
print("\nShifted (2,0) pooled:\n", pooled3[0,0])

# 관찰: 작은 shift에서는 유사, 큰 shift에서는 크게 다름
```

### 실험 3 — Deep Network의 Equivariance

```python
# 간단한 CNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

model = SimpleCNN()

# 입력 생성
x = torch.randn(1, 1, 8, 8)

# 방법 1: 원본 → conv
y1 = model(x)

# 방법 2: shift → conv → shift
x_shifted = translate_image(x, (1, 1))
y2 = model(x_shifted)
y2_shifted = translate_image(y2, (1, 1))

# 비교
error = torch.abs(y1 - y2_shifted).max().item()
print(f"Max difference between methods: {error:.6f}")
print("Expected: very small (ideally 0), showing equivariance")
```

---

## 🔗 이론과 실전의 간극

### 1. Padding의 영향

수학적으로는 무한 이미지를 가정하지만, 실제로는 finite 이미지에 padding을 적용합니다.

- **Zero padding**: 경계 바깥은 0. Equivariance 수학적으로는 약간 위반 (경계 근처)
- **Circular padding**: 경계를 원형으로 연결. 더 나은 equivariance, 주로 신호 처리에서 사용

따라서 **정확한 equivariance는 무한 이미지를 가정할 때**입니다.

### 2. 정수 좌표만 가능

Discrete convolution은 $\mathbb{Z}^2$에서만 정의됩니다. Sub-pixel translation은?

→ Spatial Transformer Networks (STN), differentiable interpolation 사용

### 3. Pooling과 Downsampling

Stride가 1이 아닌 convolution이나 pooling은:

$$\text{Conv}_{\text{stride}=s}(T_a I) = T_{a/s}(\text{Conv}_{\text{stride}=s}(I))$$

정확히는 성립하지 않습니다 (rounding 때문). 이것이 **stride equivariance 분석**의 주제입니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Infinite image (no boundary) | 실제는 finite → zero/reflect padding → equivariance 위반 |
| Integer translations | Sub-pixel shifts 불가능 → STN이나 differentiable interpolation 필요 |
| Translation group만 고려 | Rotation, scale 등은 별도의 equivariance 필요 (다음 문서) |
| Padding 일관성 | 다양한 padding 전략 → equivariance 정도 다름 |

---

## 📌 핵심 정리

$$\boxed{T_a(I * K) = (T_a I) * K \text{ — Convolution은 translation에 equivariant}}$$

$$\boxed{\phi(g \cdot x) = g \cdot \phi(x) \text{ — Group-theoretic equivariance 정의}}$$

| 개념 | 정의 |
|------|------|
| **Group** | 항등원, 연산, 역원을 가진 대수 구조 |
| **Group action** | $g \cdot x$ 정의, $(gh) \cdot x = g \cdot (h \cdot x)$ |
| **Equivariance** | $\phi(g \cdot x) = g \cdot \phi(x)$ |
| **Invariance** | $\phi(g \cdot x) = \phi(x)$ |
| **Translation group** | $(\mathbb{Z}^2, +)$, $T_a I[i,j] = I[i-a_1, j-a_2]$ |
| **Conv equivariance** | $T_a(I*K) = (T_a I)*K$ |
| **Pooling property** | Local invariance ($|a| \ll $ pool size) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T_{(1,0)} I[i,j] = I[i-1, j]$라고 정의할 때, $T_{(1,0)} \circ T_{(0,1)} = T_{(1,1)}$임을 보이라.

<details>
<summary>해설</summary>

$$
(T_{(1,0)} \circ T_{(0,1)}) I [i,j] = T_{(1,0)}(T_{(0,1)} I)[i,j]
$$
$$
= T_{(1,0)} I[i, j-1] = I[i-1, j-1]
$$

한편,
$$
T_{(1,1)} I[i,j] = I[i-1, j-1]
$$

따라서 $(T_{(1,0)} \circ T_{(0,1)}) I = T_{(1,1)} I$ for all $I$. $\square$

</details>

**문제 2** (심화): Stride-2 convolution (또는 max pooling stride 2)은 translation equivariant하지 않다. 어떤 shift에 대해서는 approximate equivariance가 성립하는가?

<details>
<summary>해설</summary>

Stride-2 pooling:
$$\text{Pool}_2(I)[i,j] = \max(I[2i:2i+2, 2j:2j+2])$$

작은 shift $a = (0, 0)$: 완벽한 equivariance

작은 shift $a = (1, 0)$: 
$$\text{Pool}_2(T_{(1,0)} I)[i,j] = \max(I[2i+1:2i+3, 2j:2j+2])$$

이것이 $T_{(1,0)}(\text{Pool}_2(I))[i,j]$와 같으려면:
$$\max(I[2i+1:2i+3, 2j:2j+2]) \stackrel{?}{=} \max(I[2(i-1/2):2(i-1/2)+2, 2j:2j+2])$$

정수 인덱스가 아니므로 정확히 정의되지 않습니다. 하지만 **approximate** equivariance는 성립합니다.

큰 shift $a = (2, 0)$: 완벽한 equivariance 복구

**결론**: stride-$s$ 연산은 $s$의 배수인 shift에만 정확히 equivariant합니다. 이것이 "checkerboard pattern" 아티팩트의 원인입니다.

</details>

**문제 3** (논문 비평): "Global max pooling이 만드는 global invariance"와 "여러 conv layer의 equivariance"의 tension을 논의하라. 분류 문제에서는 왜 invariance가 필요한가?

<details>
<summary>해설</summary>

**Tension**:
- Conv layers는 equivariant (translation 정보 보존)
- Global max pooling은 invariant (translation 정보 버림)

**왜 invariance가 필요한가**:
분류 문제 $\text{label}(I) = \text{label}(T_a I)$ (고양이는 어디에 있든 고양이)에서, 출력은 translation에 불변이어야 합니다.

따라서:
1. 초기 layers: equivariant (어디에 있는지 알아내기)
2. 중간 layers: equivariant (feature 정제)
3. 마지막 layer: invariant (global pooling) → 분류

이 구조가 CNN의 natural design입니다.

**대안**: Invariant feature 학습 (어떤 위치에서든 activate), 또는 spatial transformer로 canonical pose로 정규화.

</details>

---

<div align="center">

[◀ 이전](./01-discrete-convolution.md) | [📚 README](../README.md) | [다음 ▶](./03-group-equivariant-cnn.md)

</div>
