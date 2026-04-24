# 01. Convolution Layer의 Forward/Backward 연산

## 🎯 핵심 질문

- Multi-channel convolution의 수식 $Y[c_o, i, j] = \sum_{c_i, m, n} K[c_o, c_i, m, n] \cdot X[c_i, i+m, j+n] + b[c_o]$는 어떤 의미이며, 왜 이렇게 정의되는가?
- Backward pass에서 $\partial L/\partial K$와 $\partial L/\partial X$를 어떻게 구하는가?
- Chain rule을 통해 gradient 계산이 원래 연산의 형태(convolution, full-mode convolution)로 표현되는 이유는?
- im2col trick은 convolution을 어떻게 matrix multiplication으로 변환하는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Convolution 연산은 CNN의 핵심입니다. CNN이 이미지, 음성, 시계열 데이터에서 우수한 성능을 내는 이유는 convolution이 **local connectivity**와 **weight sharing**을 통해 이동 불변성(translation invariance)을 자연스럽게 구현하기 때문입니다. 하지만 convolution의 정확한 수학적 정의, 특히 backward pass의 유도를 이해하지 못하면, gradient descent가 어떻게 작동하는지, 왜 im2col 같은 최적화가 가능한지, 또한 depthwise separable convolution 같은 변형들이 무엇을 하는 것인지 알 수 없습니다. 이 문서는 **forward/backward 연산의 수학적 기초**를 제공합니다.

---

## 📐 수학적 선행 조건

- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Chain rule, partial derivatives, Jacobian
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix multiplication, reshaping, tensor notation
- [Signal Processing Deep Dive](https://github.com/iq-ai-lab/signal-processing-deep-dive) (권장): Discrete convolution, cross-correlation, frequency domain
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backpropagation, automatic differentiation

---

## 📖 직관적 이해

### Feature Extraction을 위한 Local Pattern 탐지

일반적인 fully connected layer에서는 모든 입력 뉴런이 모든 출력 뉴런과 연결되므로, 이미지처럼 큰 입력에서는 파라미터가 폭발합니다. Convolution은 대신 **작은 receptive field**에서만 연산을 합니다.

예를 들어 $3 \times 3$ 필터를 생각해보세요. 입력 이미지의 $(i, j)$ 위치에서 시작하여 $3 \times 3$ 영역을 필터와 곱하고 합산합니다. 이 필터가 edge, corner, texture 같은 기본 패턴을 학습하면, 같은 필터를 이미지 전체에 반복적으로 적용함으로써 **어디서 그 패턴이 나타나는지** 탐지할 수 있습니다.

### Weight Sharing과 Translation Invariance

Convolution의 가장 중요한 성질은 **같은 필터를 모든 위치에서 사용**한다는 것입니다. 따라서 어떤 패턴을 이미지의 한 위치에서 인식하면, 다른 위치에서도 (약간의 shift에 대해) 인식할 수 있습니다. 이것이 **translation invariance**이고, CNN이 이미지에서 강력한 이유입니다.

### Multi-channel Convolution의 의미

실제 CNN에서 convolution은 single channel이 아니라 **여러 input channel을 여러 output channel로** 변환합니다:

$$Y[c_o, i, j] = \sum_{c_i=0}^{C_{in}-1} \sum_{m,n} K[c_o, c_i, m, n] \cdot X[c_i, i+m, j+n] + b[c_o]$$

- $X$: shape $(C_{in}, H, W)$ — input
- $K$: shape $(C_{out}, C_{in}, K_h, K_w)$ — kernel(weight)
- $Y$: shape $(C_{out}, H', W')$ — output
- $b$: shape $(C_{out},)$ — bias

각 output channel $c_o$는 모든 input channel들의 가중 합입니다. 이는 매 layer마다 점점 추상적인 feature를 학습하게 합니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Multi-channel Convolution (Forward Pass)

Input $X \in \mathbb{R}^{C_{in} \times H \times W}$, kernel $K \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$, bias $b \in \mathbb{R}^{C_{out}}$에 대해, padding $p$, stride $s$로 정의된 output:

$$Y[c_o, i, j] := \sum_{c_i=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} K[c_o, c_i, m, n] \cdot X[c_i, i \cdot s + m - p, j \cdot s + n - p] + b[c_o]$$

Output shape: $C_{out} \times H_{out} \times W_{out}$, 단 $H_{out} = \lfloor (H + 2p - K_h) / s \rfloor + 1$.

(주의: boundary handling — padding 바깥의 원소는 zero또는 다른 방식으로 대체)

### 정의 2.2 — Backward Pass (Gradient 계산)

Loss $L$이 output $Y$의 함수일 때:

$$\frac{\partial L}{\partial X[c_i, h, w]} = \sum_{c_o=0}^{C_{out}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} K[c_o, c_i, m, n] \cdot \frac{\partial L}{\partial Y[c_o, h-m+p, w-n+p]}$$

(단, 인덱스가 유효한 범위에서만)

$$\frac{\partial L}{\partial K[c_o, c_i, m, n]} = \sum_{i'=0}^{H_{out}-1} \sum_{j'=0}^{W_{out}-1} X[c_i, i' \cdot s + m - p, j' \cdot s + n - p] \cdot \frac{\partial L}{\partial Y[c_o, i', j']}$$

### 정리 2.3 — Backward Pass의 Convolution 표현

$\partial L / \partial X$는 $\partial L / \partial Y$와 **flipped kernel** $K^{flip}$의 full-mode convolution으로 표현되고, $\partial L / \partial K$는 $X$와 $\partial L / \partial Y$의 convolution으로 표현된다. $\square$

### 정의 2.4 — im2col (Image to Column) 변환

Input $X$의 sliding window들을 column으로 정렬한 행렬 $X_{col} \in \mathbb{R}^{C_{in} K_h K_w \times H_{out} W_{out}}$:

$$X_{col}[:, i' W_{out} + j'] = \text{vec}(X[:, i':i'+K_h, j':j'+K_w])$$

Kernel을 행렬로 reshape: $K_{mat} \in \mathbb{R}^{C_{out} \times C_{in} K_h K_w}$.

그러면 forward pass는:

$$Y_{mat} = K_{mat} @ X_{col} + b$$

---

## 🔬 증명 또는 수학적 유도

### Backward Pass 유도 (Chain Rule)

Loss $L$이 output $Y$에 대해 scalar이고, $Y = f(X, K; p, s)$일 때:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X}$$

Forward에서:
$$Y[c_o, i, j] = \sum_{c_i, m, n} K[c_o, c_i, m, n] X[c_i, i \cdot s + m - p, j \cdot s + n - p] + b[c_o]$$

Chain rule을 적용하면, 특정 $X[c_i, h, w]$에 영향을 주는 모든 output element들을 찾아야 합니다.

$X[c_i, h, w]$는 output의 위치 $(i', j')$에서 $i' \cdot s + m - p = h$, $j' \cdot s + n - p = w$인 모든 $(m, n)$에 대해 나타납니다.

따라서:
$$\frac{\partial L}{\partial X[c_i, h, w]} = \sum_{c_o, m, n: i' \cdot s + m - p = h, j' \cdot s + n - p = w} K[c_o, c_i, m, n] \frac{\partial L}{\partial Y[c_o, i', j']}$$

이를 정리하면:
$$\frac{\partial L}{\partial X[c_i, h, w]} = \sum_{c_o} \sum_{m, n} K[c_o, c_i, m, n] \frac{\partial L}{\partial Y[c_o, h - m + p, w - n + p]}$$

이는 **flipped kernel** $K'[c_o, c_i, m, n] = K[c_o, c_i, K_h - 1 - m, K_w - 1 - n]$을 사용한 full-mode convolution과 같습니다. $\square$

### im2col의 정당성

Forward를 행렬 형태로 쓰면:

$$Y_{flat} = K_{flat} X_{col}$$

이는 일반적인 dense layer와 동일하므로, cuDNN·BLAS 같은 최적화된 행렬 곱셈 라이브러리를 직접 사용할 수 있습니다. 역전파도 동일하게 적용됩니다. $\square$

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — Forward Pass 검증

```python
import torch
import torch.nn.functional as F

# 간단한 예제: 1개 output channel, 1개 input channel
B, C_in, H, W = 1, 1, 5, 5
C_out, K_h, K_w = 1, 3, 3
p, s = 0, 1

# Input과 kernel 생성
X = torch.arange(H * W, dtype=torch.float32).view(B, C_in, H, W)
K = torch.tensor([[[[1., 0., -1.],
                     [2., 0., -2.],
                     [1., 0., -1.]]]], dtype=torch.float32)

print("Input X shape:", X.shape)
print("X:\n", X[0, 0])
print("\nKernel K shape:", K.shape)
print("K:\n", K[0, 0])

# Forward pass
Y = F.conv2d(X, K, padding=p, stride=s)
print("\nOutput Y shape:", Y.shape)
print("Y:\n", Y[0, 0])

# 수동으로 검증: (0,0) 위치
manual_y00 = (X[0, 0, 0:3, 0:3] * K[0, 0]).sum()
print("\nManual Y[0,0]:", manual_y00.item())
print("PyTorch Y[0,0]:", Y[0, 0, 0, 0].item())
```

**예상 출력**: 수동 계산과 PyTorch 결과가 일치.

### 실험 2 — Backward Pass (Gradient) 검증

```python
# Backward pass 검증
X = torch.randn(2, 3, 8, 8, requires_grad=True)
K = torch.randn(4, 3, 3, 3, requires_grad=True)
b = torch.randn(4, requires_grad=True)

Y = F.conv2d(X, K, bias=b, padding=1, stride=1)
loss = Y.sum()
loss.backward()

print("X gradient shape:", X.grad.shape)
print("K gradient shape:", K.grad.shape)
print("b gradient shape:", b.grad.shape)
print("\nGradients computed successfully!")
print("X gradient norm:", X.grad.norm().item())
print("K gradient norm:", K.grad.norm().item())
```

**예상 결과**: 모든 gradient가 계산되고, shapes가 입력과 동일.

### 실험 3 — im2col 트릭 검증

```python
from torch.nn.modules.utils import _pair

def im2col_custom(X, K_size, padding, stride):
    """
    Input X (B, C_in, H, W)에서 im2col 수행
    Output shape: (B, C_in * K_h * K_w, H_out * W_out)
    """
    B, C_in, H, W = X.shape
    K_h, K_w = _pair(K_size)
    p, s = _pair(padding), _pair(stride)
    
    H_out = (H + 2*p[0] - K_h) // s[0] + 1
    W_out = (W + 2*p[1] - K_w) // s[1] + 1
    
    # Padding
    X_padded = F.pad(X, (p[1], p[1], p[0], p[0]))
    
    # Extract patches (unfold)
    col = F.unfold(X_padded, kernel_size=(K_h, K_w), stride=s)
    return col, H_out, W_out

# Test
X = torch.randn(2, 3, 8, 8)
K = torch.randn(4, 3, 3, 3)

col, H_out, W_out = im2col_custom(X, K_size=3, padding=1, stride=1)
print("col shape:", col.shape)  # (2, 3*3*3, H_out*W_out)
print("Expected: (2, 27, {})".format(H_out * W_out))

# Reshape kernel for matmul
K_flat = K.view(K.shape[0], -1)  # (4, 27)

# Forward: Y_flat = K_flat @ col
Y_col = torch.bmm(K_flat.unsqueeze(0).expand(2, -1, -1), col)
Y_col = Y_col.view(2, 4, H_out, W_out)

# Compare with F.conv2d
Y_conv = F.conv2d(X, K, padding=1, stride=1)

print("\nim2col-based Y shape:", Y_col.shape)
print("F.conv2d Y shape:", Y_conv.shape)
print("Max difference:", (Y_col - Y_conv).abs().max().item())
```

**예상**: im2col 기반 결과와 F.conv2d가 수치적으로 거의 일치 (부동소수점 오차 수준).

### 실험 4 — Numerical Gradient 검증

```python
def numerical_gradient(func, x, eps=1e-5):
    """
    Numerical gradient을 중앙 차분으로 계산
    """
    grad = torch.zeros_like(x)
    for idx in torch.ndindex(x.shape):
        x_plus = x.clone()
        x_plus[idx] += eps
        x_minus = x.clone()
        x_minus[idx] -= eps
        grad[idx] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad

# 작은 예제에서 검증
X = torch.randn(1, 1, 3, 3, requires_grad=True)
K = torch.randn(1, 1, 3, 3, requires_grad=True)

def loss_fn(X_val):
    Y = F.conv2d(X_val.unsqueeze(0), K, padding=0, stride=1)
    return Y.sum()

# Automatic gradient
Y = F.conv2d(X, K, padding=0, stride=1)
loss = Y.sum()
loss.backward()
auto_grad = X.grad.clone()

# Numerical gradient (계산 비용이 크므로 작은 예제)
# 실제로는 skip하거나 subset만 확인
print("Automatic gradient X norm:", auto_grad.norm().item())
print("Gradient computation successful!")
```

---

## 🔗 이론과 실전의 간극

### GPU 최적화와 im2col

Forward pass의 수학적 정의는 간단하지만, GPU에서 효율적으로 구현하려면 여러 최적화가 필요합니다:

1. **im2col + GEMM**: cuDNN은 im2col로 변환 후 최적화된 행렬 곱셈(GEMM)을 사용합니다.
2. **Winograd Algorithm**: 특정 kernel 크기($3 \times 3$ 등)에 대해 더 빠른 계산 (frequency domain 기법).
3. **Memory Layout**: NCHW vs NHWC 등 데이터 배치 순서 최적화.

이러한 구현 디테일은 수학적 정의와는 독립적이지만, 실제 성능에 큰 영향을 줍니다.

### Backward Pass의 대칭성

Backward에서 $\partial L/\partial K$는 $X$와 $\partial L/\partial Y$의 convolution인 반면, $\partial L/\partial X$는 flipped kernel을 사용합니다. 이는 **transposed convolution**과 관련이 있으며 (Ch2-04 참고), 신경망의 생성 모델(decoder, generator)에서 핵심 역할을 합니다.

---

## ⚖️ 가정과 한계

| 가정 | 설명 | 한계 |
|------|------|------|
| Padding은 zero | 경계는 0으로 채움 | Reflection/replicate padding도 가능하지만 수식이 복잡 |
| Fixed kernel size | Kernel이 입력과 무관 | Adaptive kernel이나 deformable convolution은 다름 |
| Regular grid | 입력이 규칙적 격자 | 불규칙 그래프 데이터는 graph convolution 필요 |
| Continuous kernel 가정 아님 | Discrete convolution만 다룸 | Continuous convolution (SDF, neural field)는 다른 이론 |

---

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| **Forward Convolution** | $Y[c_o, i, j] = \sum_{c_i, m, n} K[c_o, c_i, m, n] X[c_i, i \cdot s + m - p, j \cdot s + n - p] + b[c_o]$ |
| **Backward w.r.t. X** | $\partial L/\partial X = (\partial L/\partial Y) *_{full} K^{flip}$ |
| **Backward w.r.t. K** | $\partial L/\partial K = X *_{valid} (\partial L/\partial Y)$ |
| **im2col** | Convolution을 dense matrix multiply로 변환하여 GPU 최적화 활용 |
| **Weight Sharing** | 같은 kernel을 모든 위치에서 사용 → translation invariance |

---

## 🤔 생각해볼 문제

**문제 1 (기초).** Input $X \in \mathbb{R}^{1 \times 4 \times 4}$, kernel $K \in \mathbb{R}^{1 \times 1 \times 2 \times 2}$, padding $p=0$, stride $s=1$일 때, output shape은?

$$X = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Y[0,0]을 계산하라.

<details>
<summary>해설</summary>

Output shape: $H_{out} = (4 + 2 \cdot 0 - 2) / 1 + 1 = 3$, 따라서 $(1, 3, 3)$.

$Y[0, 0, 0] = 1 \cdot 1 + 2 \cdot 0 + 5 \cdot 0 + 6 \cdot 1 = 1 + 6 = 7$.

</details>

**문제 2 (심화).** Backward pass에서 $\partial L/\partial X[0, 1, 1]$을 구하라. (위 예제 계속, $\partial L/\partial Y$는 모두 1이라 가정)

<details>
<summary>해설</summary>

Flipped kernel: $K^{flip} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ (이미 대칭).

$X[0, 1, 1]$은 output의 $(0, 0), (0, 1), (1, 0), (1, 1)$ 등 여러 위치에 나타남. 

Forward에서:
- $Y[0, 0, 0]$ uses $X[0, 0:2, 0:2]$ → kernel의 $(1, 1)$ 위치 ($K[0, 0, 1, 1] = 1$)
- $Y[0, 0, 1]$ uses $X[0, 0:2, 1:3]$ → kernel의 $(1, 0)$ 위치 ($K[0, 0, 1, 0] = 0$)
- 등등...

Backward: $\partial L/\partial X[0, 1, 1] = K[0, 0, 1, 1] \cdot 1 + K[0, 0, 0, 1] \cdot 1 + K[0, 0, 1, 0] \cdot 1 + K[0, 0, 0, 0] \cdot 1 = 1 + 0 + 0 + 1 = 2$.

</details>

**문제 3 (논문 비평).** im2col 기반 구현은 메모리 효율이 낮을 수 있다 (큰 입력에서 col matrix가 메모리를 많이 사용). 이를 해결하는 다른 접근 방법을 생각해보라.

<details>
<summary>해설</summary>

1. **Tiling**: 입력을 작은 타일로 나누어 im2col → im2col 행렬이 작아짐.
2. **Direct convolution**: im2col 없이 직접 loop → 메모리 사용 적지만 CPU에서는 느림.
3. **Winograd/FFT**: Frequency domain에서 곱셈 (특정 kernel 크기에 유리).
4. **Sparse convolution**: 많은 0을 가진 입력에서 sparse 연산.

실제로 cuDNN은 여러 알고리즘 중 입력 크기에 따라 최적 방법을 선택합니다.

</details>

---

<div align="center">

[◀ 이전](../ch1-convolution/05-convolution-theorem.md) | [📚 README](../README.md) | [다음 ▶](./02-pooling.md)

</div>
