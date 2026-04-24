# 02. Pooling의 수학적 역할

## 🎯 핵심 질문

- Max pooling과 average pooling의 수학적 정의와 차이점은 무엇인가?
- Backward pass에서 max pooling은 왜 argmax 위치로만 gradient를 전달하는가?
- "Local translation invariance"는 무엇이며, pooling이 이를 어떻게 구현하는가?
- Pooling이 receptive field를 어떻게 확대하고, downsampling이 실제 이미지 처리에서 왜 필요한가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Convolution만으로는 deep neural network를 구성하기 어렵습니다. 이유는:

1. **메모리와 계산량**: 입력이 $H \times W$이고 convolution이 공간 크기를 유지하면, 계산량이 quadratic입니다.
2. **Global context 부족**: 각 뉴런의 receptive field가 작아서 큰 패턴을 인식하기 어렵습니다.

Pooling은 이를 해결합니다. 작은 영역에서 최댓값을 취하거나 평균을 내어 **공간 차원을 감소**시키면서, **중요한 정보는 유지**합니다. 또한 pooling은 **약간의 shift에 대해 invariant**하게 만들어 모델의 강건성을 높입니다. 이 문서에서는 pooling의 수학적 정의, backward 계산, translation invariance 성질을 정확히 이해합니다.

---

## 📐 수학적 선행 조건

- [Ch2-01: Convolution Forward/Backward](./01-conv-forward-backward.md): Convolution 기초, chain rule
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Max/min 연산, 부분 미분가능성
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Subgradient, non-smooth optimization

---

## 📖 직관적 이해

### Max Pooling: 중요한 특징만 선택

Max pooling은 작은 슬라이딩 윈도우(예: $2 \times 2$)에서 **최댓값**을 추출합니다. 직관적으로:

$$Y[i, j] = \max_{(m, n) \in R(i, j)} X[m, n]$$

여기서 $R(i, j)$는 $(i, j)$ 위치의 receptive region입니다.

**왜 최댓값일까?** 만약 필터가 어떤 패턴(예: edge)을 탐지하도록 학습했다면, 그 영역에서 가장 강한 활성화값이 그 패턴의 유무를 가장 잘 나타냅니다. 주변 값들은 noise일 가능성이 높으므로, 최댓값만 취하면 중요 정보는 유지하면서 noise를 줄일 수 있습니다.

### Average Pooling: 부드러운 요약

Average pooling은 그냥 평균을 냅니다:

$$Y[i, j] = \frac{1}{|R(i, j)|} \sum_{(m, n) \in R(i, j)} X[m, n]$$

**장점**: Differentiable (미분가능). **단점**: 모든 값이 영향을 주므로, small activation도 output에 기여하여 noise가 남을 수 있습니다.

### Local Translation Invariance: 작은 이동에 견디기

만약 어떤 특징이 이미지에서 약간 shift되면 어떻게 될까요?

Convolution은 이동에 따라 output도 이동합니다 (정확히는 shift in, shift out). 하지만 pooling을 거치면, **small shift는 무시**됩니다.

예: $2 \times 2$ max pooling으로 내려보낸 후, 입력이 1픽셀 이동되었다고 해도, pooling region 내에서 최댓값은 대부분 같을 것입니다. 이것이 **local translation invariance**입니다.

### Receptive Field 확대

Stride가 $s$인 pooling은 공간을 $1/s$로 축소합니다. 따라서 다음 layer의 같은 픽셀이 이전 layer의 $s$배 넓은 영역을 본다는 뜻입니다. 여러 pooling layer를 거치면 exponential하게 receptive field가 커집니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.5 — Max Pooling

Input $X \in \mathbb{R}^{C \times H \times W}$, pool size $p$, stride $s$에 대해:

$$Y[c, i, j] := \max_{0 \leq m < p, 0 \leq n < p} X[c, i \cdot s + m, j \cdot s + n]$$

Output shape: $C \times H_{out} \times W_{out}$, 단 $H_{out} = \lfloor (H - p) / s \rfloor + 1$.

**Argmax 저장**: Backward를 위해 $\text{ArgMax}[c, i, j] = (m^*, n^*)$를 저장합니다.

### 정의 2.6 — Average Pooling

$$Y[c, i, j] := \frac{1}{p^2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} X[c, i \cdot s + m, j \cdot s + n]$$

### 정의 2.7 — Backward Pass

**Max pooling:**

$$\frac{\partial L}{\partial X[c, i \cdot s + m^*, j \cdot s + n^*]} += \frac{\partial L}{\partial Y[c, i, j]}$$

(다른 위치는 0)

**Average pooling:**

$$\frac{\partial L}{\partial X[c, i \cdot s + m, j \cdot s + n]} = \frac{1}{p^2} \frac{\partial L}{\partial Y[c, i, j]}$$

### 정리 2.8 — Local Translation Invariance

Stride $s$와 pool size $p$인 pooling에서, input이 최대 $\|\delta\| < \min(p - 1, s - 1)$ 만큼 shift되면, output은 불변입니다.

증명: Shift된 위치에서도 pooling region이 최댓값을 포함하면, 결과는 동일. $\square$

### 정리 2.9 — Receptive Field 곱셈적 확대

Stride $s$인 pooling 후 receptive field는 이전의 $s$배로 확대됩니다.

증명: 다음 layer의 위치 $(i, j)$는 이전 layer의 영역 $[i \cdot s, i \cdot s + p - 1]$에 대응. $\square$

---

## 🔬 증명 또는 수학적 유도

### Local Translation Invariance 증명

Pool size $p = 2$, stride $s = 2$인 max pooling을 가정합시다.

원래 input의 한 block:
$$X = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad \max(a, b, c, d) = a$$

이제 input이 오른쪽으로 1픽셀 shift되면 (wrap-around 무시):
$$X' = \begin{bmatrix} b & ? \\ d & ? \end{bmatrix}$$

문제: 이동 후 block이 이전 최댓값 $a$를 포함하지 않으면 output이 달라집니다.

**더 정확한 증명**: Shift 크기 $\delta < \min(p, s)$일 때, 원래 pooling region이 새로운 region과 significant overlap을 가지므로 최댓값은 대부분 같습니다. (엄밀한 증명은 spatial distribution에 따라 다름) $\square$

### Receptive Field 계산

$n$ 개의 convolution layer ($k \times k$ kernel, stride 1)와 1개의 pooling layer (stride 2)를 거친 후 receptive field:

Conv chain: RF = $1 + (n-1) \cdot 1 = n$ (stride 1이므로 linear하게 증가)

Pooling (stride 2): 다음 layer의 1픽셀 = 이전 layer의 2픽셀

따라서 총 RF = $n \times 2$ = $2n$.

일반적으로: $\text{RF}_{\text{final}} = \text{RF}_{\text{before pool}} \times \text{stride}_{\text{pool}}$. $\square$

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Max vs Average Pooling

```python
import torch
import torch.nn.functional as F

# 간단한 입력: 한 채널, 작은 공간
X = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
print("Input X:\n", X[0, 0])

# Max pooling: 2x2, stride 2
Y_max = F.max_pool2d(X, kernel_size=2, stride=2)
print("\nMax Pooling (2x2, stride 2):\n", Y_max[0, 0])

# Average pooling
Y_avg = F.avg_pool2d(X, kernel_size=2, stride=2)
print("\nAverage Pooling (2x2, stride 2):\n", Y_avg[0, 0])

# 해석
print("\n--- 해석 ---")
print("위-왼쪽 (0-3):", X[0, 0, 0:2, 0:2], "→ max =", Y_max[0, 0, 0, 0].item(), ", avg =", Y_avg[0, 0, 0, 0].item())
print("위-오른쪽 (4-7):", X[0, 0, 0:2, 2:4], "→ max =", Y_max[0, 0, 0, 1].item(), ", avg =", Y_avg[0, 0, 0, 1].item())
```

**예상 결과**: Max가 더 극값을 강조, average가 부드러운 값을 출력.

### 실험 2 — Backward Pass 검증

```python
# Max pooling backward
X = torch.randn(1, 1, 4, 4, requires_grad=True)
Y = F.max_pool2d(X, kernel_size=2, stride=2)
loss = Y.sum()
loss.backward()

print("Max Pooling Backward:")
print("Input gradient:\n", X.grad)
print("Non-zero entries:", (X.grad != 0).sum().item())

# Average pooling backward
X2 = torch.randn(1, 1, 4, 4, requires_grad=True)
Y2 = F.avg_pool2d(X2, kernel_size=2, stride=2)
loss2 = Y2.sum()
loss2.backward()

print("\nAverage Pooling Backward:")
print("Input gradient:\n", X2.grad)
print("All entries equal?", (X2.grad.std() < 1e-6).item())
```

**예상**: Max pooling gradient는 sparse (argmax 위치만), average는 uniform.

### 실험 3 — Local Translation Invariance 확인

```python
def shift_image(X, shift_x, shift_y):
    """
    Image를 shift (roll)
    """
    return torch.roll(X, (shift_x, shift_y), dims=(2, 3))

# 원본 이미지
X_orig = torch.randn(1, 3, 16, 16)
Y_orig = F.max_pool2d(X_orig, kernel_size=2, stride=2)

# 1픽셀 shift
X_shift = shift_image(X_orig, 1, 0)
Y_shift = F.max_pool2d(X_shift, kernel_size=2, stride=2)

# 비교
diff = (Y_orig - Y_shift).abs().max().item()
print("Max pooling invariance to 1px shift:", diff)
print("(diff는 작아야 함, 완전 0은 아니지만 pool size에 비해 작음)")

# 더 큰 shift
X_shift_big = shift_image(X_orig, 3, 3)
Y_shift_big = F.max_pool2d(X_shift_big, kernel_size=2, stride=2)
diff_big = (Y_orig - Y_shift_big).abs().max().item()
print("Max pooling invariance to 3px shift:", diff_big)
print("(diff는 커질 것 — pool size = 2보다 크므로)")
```

**예상**: 작은 shift (< pool size)에서는 출력이 작은 변화, 큰 shift에서는 큰 변화.

### 실험 4 — Receptive Field 확인

```python
def compute_rf_after_pooling(rf_before, stride, pool_size):
    """
    Pooling 후 receptive field 계산
    """
    return rf_before * stride

# Conv layer: 3x3 kernel, stride 1 → RF = 3
rf_after_conv = 3

# Pooling: 2x2, stride 2 → RF *= 2
rf_after_pool = compute_rf_after_pooling(rf_after_conv, stride=2, pool_size=2)
print("RF after Conv (3x3, stride 1):", rf_after_conv)
print("RF after Pooling (2x2, stride 2):", rf_after_pool)

# 여러 층
rfs = [1]  # 입력
for i in range(3):
    rfs.append(rfs[-1] * 2)  # Conv (stride 1, kernel 3)
    rfs.append(rfs[-1] * 2)  # Pooling (stride 2)

print("\nRF through layers:")
for i, rf in enumerate(rfs):
    print(f"  Layer {i}: RF = {rf}")
```

**예상**: Receptive field가 exponential하게 증가 (stride와 pool size의 곱).

---

## 🔗 이론과 실전의 간극

### Max Pooling의 Differentiability 문제

Max pooling은 non-smooth입니다. 최댓값이 tie될 때 (여러 원소가 동일한 최댓값), gradient는 정의되지 않습니다. 실제로 PyTorch는 이런 경우 **첫 번째 최댓값**으로 선택합니다.

### Pooling의 정보 손실

Pooling은 공간 정보를 감소시킵니다. 따라서:
- 분류(classification) 작업에서는 괜찮습니다 (fine-grained spatial info가 필요 없음).
- 의료 영상이나 객체 탐지에서는 모든 픽셀이 중요하므로, 최근 모델(예: ResNet without pooling)은 **stride를 증가**시켜 pooling을 대체합니다.

### Modern Trends: Pooling의 감소

최근 CNN (예: EfficientNet, Vision Transformer)에서는:
- Pooling을 **완전히 제거**하거나,
- Convolution의 stride를 증가시켜 downsampling을 하거나,
- Adaptive pooling (전체를 1x1로 pool)으로 global context를 직접 사용합니다.

---

## ⚖️ 가정과 한계

| 가정 | 설명 | 한계 |
|------|------|------|
| Non-overlapping pooling | Pool region이 겹치지 않음 | Overlapping pooling도 사용되지만 계산량 증가 |
| Max/Avg만 사용 | 두 종류의 pooling | Stochastic pooling, mixed pooling 등 다른 변형 가능 |
| Spatial dimension만 pool | Channel은 유지 | Channel dimension을 pool하기도 함 |
| Fixed pool size | 입력과 무관 | Adaptive pooling (target size 지정)도 가능 |

---

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| **Max Pooling** | $Y[i,j] = \max_{(m,n) \in R(i,j)} X[m,n]$ — argmax 위치만 gradient |
| **Avg Pooling** | $Y[i,j] = \text{mean}(X \text{ in region})$ — uniform gradient |
| **Local Translation Invariance** | Pool region 내 shift $\|\delta\| < p$에서 output 불변 |
| **Receptive Field** | Stride $s$인 pooling → RF *= $s$ |
| **Downsampling** | 공간 해상도 감소 → 계산량·메모리 감소, RF 확대 |

---

## 🤔 생각해볼 문제

**문제 1 (기초).** $4 \times 4$ 입력에서 $2 \times 2$ max pooling, stride 2 후 output shape은? 그리고 왼쪽-위 $2 \times 2$ 블록의 최댓값은?

$$X = \begin{bmatrix} 1 & 5 & 3 & 2 \\ 4 & 2 & 6 & 1 \\ 7 & 1 & 2 & 4 \\ 3 & 8 & 5 & 2 \end{bmatrix}$$

<details>
<summary>해설</summary>

Output shape: $(4 - 2) / 2 + 1 = 2$, 따라서 $2 \times 2$.

왼쪽-위 블록: $\max(1, 5, 4, 2) = 5$. 오른쪽-위: $\max(3, 2, 6, 1) = 6$. 왼쪽-아래: $\max(7, 1, 3, 8) = 8$. 오른쪽-아래: $\max(2, 4, 5, 2) = 5$.

Output: $\begin{bmatrix} 5 & 6 \\ 8 & 5 \end{bmatrix}$

</details>

**문제 2 (심화).** Average pooling에서 gradient를 계산하면 모든 input 원소가 output에 **동일하게** 기여한다는 것을 증명하라.

<details>
<summary>해설</summary>

Forward: $Y[i, j] = \frac{1}{4} (X[2i, 2j] + X[2i, 2j+1] + X[2i+1, 2j] + X[2i+1, 2j+1])$

Chain rule: $\frac{\partial L}{\partial X[2i, 2j]} = \frac{\partial L}{\partial Y[i,j]} \cdot \frac{\partial Y[i,j]}{\partial X[2i, 2j]} = \frac{\partial L}{\partial Y[i,j]} \cdot \frac{1}{4}$

마찬가지로 다른 세 원소도 $\frac{1}{4}$. 따라서 모두 같은 가중치 $\frac{1}{4}$를 받음.

</details>

**문제 3 (논문 비평).** 최근 Vision Transformer는 pooling을 사용하지 않습니다. 그렇다면 어떻게 receptive field를 늘릴까요?

<details>
<summary>해설</summary>

Vision Transformer (ViT):
1. **Self-attention**: 모든 토큰이 모든 다른 토큰과 직접 연결 → global receptive field in one layer.
2. **Patch embedding**: 입력을 $16 \times 16$ 패치로 나누어 이미 downsampled.
3. **Multi-head attention**: 다양한 scale의 attention pattern을 학습.

따라서 pooling이 불필요 — attention mechanism이 spatial pooling의 역할을 수행.

</details>

---

<div align="center">

[◀ 이전](./01-conv-forward-backward.md) | [📚 README](../README.md) | [다음 ▶](./03-padding.md)

</div>
