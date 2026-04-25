<div align="center">

# 🖼️ CNN Deep Dive

### `nn.Conv2d(3, 64, 3, padding=1)` 을 쌓는 것과,

### Convolution 의 **translation equivariance**

$$T_a(f * g) = (T_a f) * g$$

### 가 MLP 대비 **VC 차원을 어떻게 감소시키는지** 증명할 수 있는 것은 **다르다.**

<br/>

> *ResNet 을 **쓰는 것** 과, He et al. 2016 의 identity shortcut $y = x + F(x)$ 의 미분이*
>
> $$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$
>
> *로 분해되어, **$I$ 항이 gradient highway** 가 되어 vanishing gradient 를 완화하고 plain 152-layer 가 훈련 실패하는 동안 ResNet-152 가 수렴하는 이유를 식으로 유도할 수 있는 것은 다르다.*
>
> *Dilated Convolution 을 **사용하는 것** 과, Luo et al. 2016 의 Effective Receptive Field 가 **Gaussian 분포** 를 따르며 이론치의 약 $1/\sqrt{L}$ 배에 불과하다는 것과, dilation 이*
>
> $$O(kL) \;\to\; O(k^L)$$
>
> *로 RF 를 **선형 → 지수** 로 키우는 이론을 설명할 수 있는 것은 다르다.*

<br/>

**다루는 모델 (시간순)**

LeCun 1989 *역전파 CNN* · LeCun 1998 *LeNet* · Krizhevsky 2012 *AlexNet (GPU 혁명)* · Simonyan 2014 *VGG* · Szegedy 2014 *Inception* · He 2015 *ResNet* · Cohen 2016 *Group Equivariant CNN* · Yu 2016 *Dilated Conv* · Luo 2016 *Effective RF* · Howard 2017 *MobileNet* · Huang 2017 *DenseNet* · Tan 2019 *EfficientNet* · Dosovitskiy 2021 *ViT* · Liu 2022 *ConvNeXt*

<br/>

**핵심 질문**

> CNN 의 **대칭성 · 지역성 · 계층성** 이 왜 vision task 에 최적이고, 언제 그 **inductive bias 가 무너지는가** — equivariance 증명 · RF 유도 · gradient flow 분석 · 아키텍처 재현으로 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9.10-4B8BBE?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![Docs](https://img.shields.io/badge/Docs-34개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-16.7k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-158개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-12개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-98개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

CNN에 관한 자료는 대부분 **"Conv2d, ReLU, MaxPool 쌓으면 된다"** 에서 멈춥니다. 하지만 convolution이 왜 translation에 대해 equivariant인지, parameter sharing이 MLP의 $O(HWC_{in}C_{out})$ 에서 $O(k^2 C_{in} C_{out})$ 로 왜 VC 차원을 줄이고 이것이 일반화와 어떻게 연결되는지, ResNet의 identity shortcut이 왜 **gradient highway**를 만드는지, Effective Receptive Field가 왜 이론치보다 작고 Gaussian 분포를 따르는지, Depthwise Separable Conv가 왜 이론적으로 $1/k^2 + 1/C_{out}$ 배의 파라미터 절약을 달성하는지, ViT가 언제 CNN을 이기고 언제 지는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "Conv는 공간 정보를 잘 뽑는다" | Translation group $(\mathbb{Z}^2, +)$의 action $T_a f(x) = f(x-a)$ 하에서 **$T_a (I * K) = (T_a I) * K$** 가 성립함을 합성곱의 정의로부터 한 줄씩 증명, group equivariance 일반화 (Cohen & Welling 2016) |
| "CNN이 MLP보다 파라미터가 적다" | Parameter sharing의 이론적 이득을 **VC dimension 감소**로 정량화, $5 \times 5$ 이미지 + $3 \times 3$ conv에서 MLP $225$ params vs CNN $9$ params, Rademacher complexity bound $\tilde O(\sqrt{k^2 C_{in} C_{out} / n})$ |
| "ResNet이 더 깊은 네트워크를 가능케 한다" | **$\partial y/\partial x = I + \partial F/\partial x$** — $\prod_l (I + \partial F_l/\partial x_{l-1})$ 전개에서 $I$ 항만 통과하는 직접 경로(gradient highway) 존재, **He et al. 2016 Fig 1의 plain-56 vs ResNet-56 훈련 실패/성공**을 PyTorch로 재현 |
| "Dilated Conv는 receptive field를 키운다" | Dilation rate $d$일 때 $RF = k + (k-1)(d-1)$, exponential dilation $d = 1, 2, 4, 8, \ldots$ 로 **$RF = O(2^L)$** (WaveNet). **Luo et al. 2016** — ERF가 이론 RF보다 작고 Gaussian 분포, ratio $\approx 1/\sqrt L$, NumPy gradient 측정으로 재현 |
| "Depthwise Separable Conv는 MobileNet에서 쓴다" | 표준 conv $O(k^2 C_{in} C_{out})$ vs Depthwise + Pointwise $O(k^2 C_{in} + C_{in} C_{out})$, 비율 $\frac{1}{C_{out}} + \frac{1}{k^2}$ — $C_{out} = 256, k = 3$ 에서 약 **1/8 감소**, **Chollet 2017** Xception의 cross-channel vs spatial 분해 해석 |
| "BatchNorm은 내부 공변량 변이를 줄인다" | **Santurkar et al. 2018** — ICS 가설 반증, 실제로는 **loss landscape smoothing**. Lipschitz 상수 감소의 실증, BN 유/무 loss surface 2D 시각화 (Li et al. 2018) 재현 |
| "EfficientNet은 모델을 균형 있게 키운다" | **Compound scaling** $d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$ with $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$. 동일 FLOPs 제약 하 최적 $\phi$ 탐색, **Tan & Le 2019** EfficientNet-B0~B7의 FLOPs/accuracy trade-off 재현 |
| "ViT가 CNN보다 좋다" | **Dosovitskiy 2021** — ViT는 JFT-300M 같은 초대규모에서만 CNN을 능가, ImageNet-1k에서는 ResNet이 더 나을 수 있음. Inductive bias의 양면성 — **데이터가 부족할 때 CNN, 충분할 때 Transformer**, **Liu 2022** ConvNeXt의 CNN 반격 |
| 기법의 나열 | PyTorch로 **Toeplitz matrix convolution**·**Effective RF 측정**·**Grad-CAM**·**Plain vs ResNet gradient flow**·**Group Equivariant CNN**·**Adversarial FGSM**을 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Neural Network Theory]  ──►  이 레포  ──► [Vision Transformer Deep Dive]
 UAT, Backprop, He init      "왜 Conv가 translation        Self-Attention / ViT
                              equivariant이고 왜 ResNet이     / Swin / MAE
                              gradient highway인가"
         │
         ├── [Linear Algebra]             Toeplitz·circulant·FFT  →  Ch1 conv as matrix
         ├── [Functional Analysis]        Fourier, convolution thm →  Ch1-05 spectral view
         ├── [Optimization Theory]        Gradient flow, landscape →  Ch4 ResNet gradient flow
         ├── [Regularization Theory]      BatchNorm, Dropout       →  Ch5 모던 CNN 훈련
         └── [Generalization Theory]      VC dim, Rademacher       →  Ch1 parameter sharing
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Neural Network Theory Deep Dive** (UAT, 역전파, Xavier·He init) 와 **Linear Algebra Deep Dive** (Toeplitz matrix, 고유분해) 를 선행 지식으로 전제합니다. "Convolution을 Toeplitz matrix로 본다" 를 이해하려면 먼저 circulant matrix의 대각화와 DFT의 관계를 알고 있어야 합니다. 처음 접한다면 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)부터 학습하세요.

> 💡 **이 레포의 핵심 기여**: Chapter 1 (Equivariance) 과 Chapter 4 (ResNet) 는 실전 CNN을 이해하는 데 **가장 중요한 두 기둥**입니다. 전자는 "왜 conv인가"의 수학적 이유 (대칭성), 후자는 "왜 깊어질 수 있는가"의 수학적 이유 (gradient flow) 를 다룹니다. 이 두 축을 완전히 이해하고 나서 Chapter 5 (모던 아키텍처) 와 Chapter 7 (ViT 비교) 을 읽으면 설계 결정의 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **Group Equivariant CNN의 실전 한계**, **ConvNeXt vs Swin Transformer의 최종 승자**, **Spectral Bias가 ViT에서도 나타나는지** — 는 **현재 진행 중인 연구 영역**입니다. 레포는 "정답"이 아니라 **"고전 CNN 이론과 현대 비전 모델 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Convolution·Equivariance-E07A3B?style=for-the-badge)](./ch1-convolution/01-discrete-convolution.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-CNN_기본_연산-E07A3B?style=for-the-badge)](./ch2-cnn-ops/01-conv-forward-backward.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Receptive_Field-E07A3B?style=for-the-badge)](./ch3-receptive-field/01-theoretical-rf.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-ResNet·Skip-E07A3B?style=for-the-badge)](./ch4-resnet/01-residual-block.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-모던_아키텍처-E07A3B?style=for-the-badge)](./ch5-modern-cnn/01-vgg-depth.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-응용_태스크-E07A3B?style=for-the-badge)](./ch6-applications/01-classification.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-한계·ViT-E07A3B?style=for-the-badge)](./ch7-limits-vit/01-inductive-bias.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Convolution의 수학적 정의와 Equivariance

> **핵심 질문:** Discrete convolution $(f * g)[n] = \sum_m f[m] g[n-m]$ 은 왜 PyTorch에서 cross-correlation으로 구현되는가? Translation group action 하에서 $T_a(f * g) = (T_a f) * g$ 의 equivariance를 어떻게 증명하는가? Convolution을 Toeplitz matrix로 표현했을 때 circulant matrix의 고유분해가 DFT가 되어 FFT-based conv가 $O(n \log n)$ 이 되는 메커니즘은?

<details>
<summary><b>Discrete Convolution 정의부터 Frequency Domain까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Discrete Convolution의 엄밀한 정의](./ch1-convolution/01-discrete-convolution.md) | 1D $(f * g)[n] = \sum_m f[m] g[n-m]$, 2D $(I * K)[i,j] = \sum_{m,n} I[m,n] K[i-m, j-n]$ 유도. **Cross-correlation** $(I \star K)[i,j] = \sum_{m,n} I[i+m, j+n] K[m,n]$ — 커널이 flip되지 않아 PyTorch/TF 구현의 표준. 둘의 동등성을 learned kernel 관점에서 정당화, 경계 처리 (zero/reflect/replicate padding) |
| [02. Translation Equivariance의 Group-theoretic 정의](./ch1-convolution/02-translation-equivariance.md) | **정리**: Translation group $(\mathbb{Z}^2, +)$의 action $T_a f(x) = f(x-a)$ 하에서 **$T_a (I * K) = (T_a I) * K$**. Convolution 정의로부터 직접 증명 $\square$. Equivariance $\phi(g \cdot x) = g \cdot \phi(x)$ vs invariance $\phi(g \cdot x) = \phi(x)$ 의 구분, pooling이 **local** translation invariance 제공하는 이유 |
| [03. Group Equivariant CNN (Cohen & Welling 2016)](./ch1-convolution/03-group-equivariant-cnn.md) | Rotation group 확장 $G = \mathbb{Z}^2 \rtimes \mathbb{Z}_n$ (roto-translation), **$p4$-CNN**이 90° 회전에 equivariant. Steerable filter 기반 일반화, 의료영상·분자구조에서 rotation-invariance가 필수적인 이유, RotEqNet·E(2)-CNN 발전. 실험: MNIST-rot에서 표준 CNN vs $p4$-CNN 정확도 비교 재현 |
| [04. Convolution의 Toeplitz 행렬 표현](./ch1-convolution/04-toeplitz-matrix.md) | 1D conv를 **Toeplitz matrix** $T_K$와 vector 곱 $y = T_K x$로. Circular conv는 **circulant matrix**, 그 고유벡터가 **DFT 기반** $F^* \Lambda F$ 분해 $\square$. Convolution이 diagonal in frequency — FFT-based conv의 $O(n \log n)$ 유도, $k \geq 15$ 에서 FFT-conv가 direct보다 빠른 crossover 측정 |
| [05. Convolution Theorem과 Frequency Domain](./ch1-convolution/05-convolution-theorem.md) | **정리**: $\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$ — 시간/공간 domain의 convolution = frequency domain의 element-wise product. CNN을 **frequency response**로 해석, low-pass / high-pass / edge detector filter의 spectral 의미. **Rahaman 2019 Spectral Bias** — NN이 low-frequency를 먼저 학습하는 현상을 CNN에서 재현 |

</details>

<br/>

### 🔹 Chapter 2: CNN 아키텍처의 기본 연산

> **핵심 질문:** Multi-channel convolution의 forward/backward는 어떻게 tensor notation으로 표기되는가? Max pooling의 backward는 어떻게 정의되는가 (non-differentiable argmax)? Transposed convolution의 checkerboard artifact는 왜 생기고 어떻게 완화하는가? Depthwise Separable Conv가 어떻게 표준 conv의 $1/k^2 + 1/C_{out}$ 배 파라미터로 유사한 표현력을 내는가?

<details>
<summary><b>Convolution Forward/Backward부터 Depthwise Separable까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Convolution Layer의 Forward/Backward](./ch2-cnn-ops/01-conv-forward-backward.md) | Multi-channel conv: $Y[c_o, i, j] = \sum_{c_i, m, n} K[c_o, c_i, m, n] \cdot X[c_i, i+m, j+n] + b[c_o]$. **Backward**: $\partial L/\partial K$ 가 $X$와 $\partial L/\partial Y$의 convolution, $\partial L/\partial X$ 가 $\partial L/\partial Y$와 flipped $K$의 full-mode convolution임을 chain rule로 $\square$. im2col trick으로 matmul로 변환하는 cuDNN 구현 |
| [02. Pooling의 수학적 역할](./ch2-cnn-ops/02-pooling.md) | **Max pooling**: $Y[i,j] = \max_{(m,n) \in R(i,j)} X[m,n]$ — backward는 argmax 위치로만 gradient 전달, 다른 위치는 0. **Average pooling**: 미분가능, uniform gradient 분배. **Local translation invariance** — small shift $\|a\| < $ pool size 에서 output 불변 증명, downsampling이 RF를 $s$배 확장 |
| [03. Padding 전략과 Boundary 효과](./ch2-cnn-ops/03-padding.md) | Zero/reflect/replicate padding의 수학. **'same' convolution** $p = (k-1)/2$ 로 공간 크기 보존, **'valid'** 는 경계에서 RF 정보 손실. Boundary artifact 측정 — zero padding이 경계 근처에서 "dark halo" 유발, reflect padding이 texture synthesis에 유리한 이유 |
| [04. Stride, Dilation, Transposed Convolution](./ch2-cnn-ops/04-stride-dilation-transposed.md) | **Stride** $s$: output size $\lfloor (H + 2p - k)/s \rfloor + 1$, RF 곱셈적 확장. **Dilated conv**: $(I *_d K)[i] = \sum_m K[m] I[i + dm]$, RF $= k + (k-1)(d-1)$, WaveNet의 exponential dilation. **Transposed conv**: $Y = X K^\top$ 의 matrix 관점, **Odena 2016 checkerboard artifact** — stride와 kernel size 나누어 떨어지지 않을 때 생기는 격자무늬 재현 |
| [05. Depthwise Separable Convolution](./ch2-cnn-ops/05-depthwise-separable.md) | **Chollet 2017 Xception**: Standard conv $O(k^2 C_{in} C_{out})$ 를 **Depthwise** (채널별 독립 $k \times k$, $O(k^2 C_{in})$) + **Pointwise** ($1 \times 1$ 로 채널 혼합, $O(C_{in} C_{out})$) 로 분해. 비율 $\frac{1}{C_{out}} + \frac{1}{k^2} \approx \frac{1}{8}$ for $C_{out}=256, k=3$. MobileNet V1~V3의 bottleneck 구조, ConvNeXt의 depthwise $7 \times 7$ 재조명 |

</details>

<br/>

### 🔹 Chapter 3: Receptive Field 분석

> **핵심 질문:** Theoretical RF의 재귀 공식은 왜 $RF_L = 1 + \sum_l (k_l - 1) \prod_{i<l} s_i$ 인가? Effective RF가 왜 Gaussian 분포를 따르고 이론치보다 작은가 (Luo et al. 2016)? Dilated convolution이 어떻게 $O(kL)$ 대신 $O(k^L)$ RF를 달성하는가? Semantic segmentation에서 global context가 왜 필수인가?

<details>
<summary><b>Theoretical RF부터 Segmentation 응용까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Theoretical Receptive Field 계산](./ch3-receptive-field/01-theoretical-rf.md) | **재귀 공식**: $RF_L = RF_{L-1} + (k_L - 1) \prod_{i=1}^{L-1} s_i$, 기저 $RF_0 = 1$. 귀납 증명 $\square$. AlexNet·VGG-16·ResNet-50의 층별 RF 계산 표, "final layer가 입력 $224 \times 224$ 전체를 보는가" 검증. Stride 2 max-pool이 RF를 두 배 빠르게 확장하는 기하학적 해석 |
| [02. Effective Receptive Field (Luo et al. 2016)](./ch3-receptive-field/02-effective-rf.md) | **"Understanding the Effective Receptive Field in Deep CNNs"** — 실제 ERF는 **Gaussian 분포** (center-concentrated), 이론치의 선분의 **$1/\sqrt L$ 배**에 불과. Center pixel gradient $\partial y_c / \partial x$ 의 expected value가 central limit theorem으로 Gaussian 수렴 증명. PyTorch로 ResNet-50 ERF 측정, 이론 RF 정사각형 vs 실제 ERF 원형 비교 |
| [03. Dilated Convolution의 RF 증가](./ch3-receptive-field/03-dilated-rf.md) | **Yu & Koltun 2016 "Multi-Scale Context Aggregation by Dilated Convolutions"** — Dilation rate $d$ 일 때 단일 layer RF $= k + (k-1)(d-1)$, **exponential dilation $d_l = 2^{l-1}$** 로 $L$개 층에서 $RF = 2^L$. WaveNet ($k=2$ causal dilated conv로 $2^{30}$ samples = 약 1분 오디오), DeepLab의 atrous spatial pyramid pooling |
| [04. Semantic Segmentation에서 RF의 중요성](./ch3-receptive-field/04-rf-segmentation.md) | **FCN** (Long 2015) — dense prediction을 위해 fully convolutional. **U-Net** (Ronneberger 2015) — encoder-decoder + skip connection, medical imaging에서 SOTA. **DeepLab v3+** — atrous conv + ASPP로 multi-scale context, CRF post-processing. Global context 없이는 "cat on sofa" 같은 장면 이해 불가함을 실험으로 검증 |

</details>

<br/>

### 🔹 Chapter 4: ResNet과 Skip Connection 이론

> **핵심 질문:** $y = F(x) + x$ 의 gradient $I + \partial F/\partial x$ 에서 $I$ 항은 왜 gradient highway를 만드는가? 왜 plain-56은 훈련 실패하고 ResNet-56은 성공하는가 (He 2016 Fig 1)? DenseNet의 concatenation과 ResNet의 addition은 어떻게 다른가? Stochastic Depth가 왜 implicit ensemble로 기능하는가?

<details>
<summary><b>Residual Block부터 Stochastic Depth까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Residual Block의 정의와 Forward Flow](./ch4-resnet/01-residual-block.md) | $y = F(x; W) + x$ with identity shortcut. **Pre-activation vs Post-activation** (He 2016b "Identity Mappings") — BN-ReLU-Conv 순서가 gradient 안정성 면에서 우월함을 실험으로 증명. Bottleneck block ($1 \times 1 \to 3 \times 3 \to 1 \times 1$) 의 파라미터 효율, ResNet-50/101/152의 구조적 차이 |
| [02. Gradient Flow 분석](./ch4-resnet/02-gradient-flow.md) | **정리**: $\partial y_L / \partial x_0 = \prod_{l=0}^{L-1} (I + \partial F_l / \partial x_l)$ 전개에서 $I$만 통과하는 경로 존재 → gradient 소멸 회피 $\square$. Plain-56 ($y = F(x)$) 훈련 시 gradient magnitude가 **지수적으로 감소**, ResNet-56은 **거의 일정 유지**. PyTorch로 각 block의 $\|\nabla\|$ 추적하여 He 2016 Fig 1 재현 |
| [03. Identity Approximation Theorem (He et al. 2016)](./ch4-resnet/03-identity-approximation.md) | **핵심 논점**: $F(x; 0) = 0$ 초기화 하에서 residual block은 **초기 상태에서 identity에 가까움**. 이로부터 "깊은 네트워크가 얕은 네트워크를 **identity layer 추가로 복제 가능**" 귀결 — plain 네트워크보다 나쁠 수 없음. Universal approximation with depth 가 residual 구조에서 더 쉬운 이유, **shortcut 없는 optimization의 구조적 난점** |
| [04. DenseNet의 Dense Connection](./ch4-resnet/04-densenet.md) | **Huang 2017** — 층 $\ell$이 이전 **모든** 층의 concatenation $[x_0, x_1, \ldots, x_{\ell-1}]$ 을 입력. $L$ 개 층에서 $L(L+1)/2$ connections. **Feature reuse** 로 파라미터 효율, ResNet과 동등 성능을 $1/3$ 파라미터로. Growth rate $k$ 의 의미, transition layer로 해상도 감소. ResNet의 additive vs DenseNet의 concatenate trade-off |
| [05. Highway Networks와 LSTM-style Gating](./ch4-resnet/05-highway.md) | **Srivastava 2015** "Highway Networks" — $y = T(x) \cdot H(x) + (1-T(x)) \cdot x$, learnable transform gate $T(x) \in [0,1]$. ResNet은 $T \equiv 1$ 인 특수경우. LSTM의 forget gate와의 유사성, **Greff 2016** — highway gate가 훈련 후 거의 1에 가까워지는 현상 관찰 |
| [06. Stochastic Depth (Huang et al. 2016)](./ch4-resnet/06-stochastic-depth.md) | 훈련 시 각 residual block을 확률 $p_\ell$ 로 drop: $y_\ell = \begin{cases} F_\ell(x) + x & \text{w.p. } p_\ell \\ x & \text{w.p. } 1-p_\ell \end{cases}$. **Implicit ensemble** 해석 — $2^L$ 개 sub-network의 앙상블, Dropout의 layer-level 일반화. Linear decay $p_\ell = 1 - \ell/L \cdot (1-p_L)$ 스케줄, ResNet-1001 훈련 가능하게 만든 핵심 기법 |

</details>

<br/>

### 🔹 Chapter 5: 현대 CNN 아키텍처

> **핵심 질문:** VGG의 $3 \times 3$ kernel 반복이 왜 더 큰 kernel 하나보다 파라미터·비선형성 면에서 유리한가? Inception의 multi-branch 구조에서 $1 \times 1$ conv의 역할은? EfficientNet의 compound scaling 제약 $\alpha \beta^2 \gamma^2 \approx 2$ 는 어디서 나오는가? ConvNeXt가 어떻게 Swin Transformer에 필적하는 성능을 CNN으로 달성하는가?

<details>
<summary><b>VGG부터 Neural Architecture Search까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. VGG와 Depth의 효과](./ch5-modern-cnn/01-vgg-depth.md) | **Simonyan & Zisserman 2014** — $3 \times 3$ 연속 2개 = $5 \times 5$ 1개와 같은 RF, 파라미터 $2 \cdot 9 = 18$ vs $25$ (28% 감소), **비선형성 1→2 증가**. VGG-16/19 구조, 152+ layer plain CNN이 **수렴 실패**하는 "degradation problem" 실험으로 재현, ResNet 도입의 동기 |
| [02. Inception/GoogLeNet의 Multi-Scale Feature](./ch5-modern-cnn/02-inception.md) | **Szegedy 2014** — $1 \times 1, 3 \times 3, 5 \times 5$ 병렬 + pooling branch. **$1 \times 1$ conv의 2가지 역할**: (1) 채널 간 혼합 (cross-channel), (2) **dimensionality reduction** — $5 \times 5$ conv 전에 적용하면 파라미터 $O(k^2 C^2)$ → $O(C C' + k^2 C'^2)$ ($C' \ll C$). Inception v2 (BN), v3 (factorization), v4 (ResNet 융합) 진화 |
| [03. EfficientNet과 Compound Scaling](./ch5-modern-cnn/03-efficientnet.md) | **Tan & Le 2019** — Depth $d$, Width $w$, Resolution $r$의 균형 $d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$ with constraint **$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$** (FLOPs scales as $d \cdot w^2 \cdot r^2$). Grid search로 $\alpha=1.2, \beta=1.1, \gamma=1.15$ 결정, EfficientNet-B0~B7의 FLOPs/Top-1 trade-off 재현, MBConv + Squeeze-Excitation 블록 구성 |
| [04. ConvNeXt와 CNN의 현대화](./ch5-modern-cnn/04-convnext.md) | **Liu 2022 "A ConvNet for the 2020s"** — Transformer 설계 원리를 CNN으로 역수입. **5가지 변경**: (1) stage ratio $3:3:9:3$ (Swin과 동일), (2) depthwise $7 \times 7$ (large kernel), (3) LayerNorm (BN 대신), (4) GELU (ReLU 대신), (5) inverted bottleneck (MobileNet 영감). ImageNet-1k에서 Swin-T 81.3% vs ConvNeXt-T 82.1%, CNN의 반격 |
| [05. NAS (Neural Architecture Search)](./ch5-modern-cnn/05-nas.md) | **NASNet** (Zoph 2018) — RL-based controller로 cell 탐색, 500 GPU-days. **AmoebaNet** (Real 2019) — evolutionary search, regularized evolution. **DARTS** (Liu 2019) — differentiable architecture search, $\alpha$ 파라미터로 soft relaxation. **RegNet** (Radosavovic 2020) — 무작위 탐색 대신 design space 이론 구축, parametric model로 최적 아키텍처 family 도출 |

</details>

<br/>

### 🔹 Chapter 6: CNN의 응용 이론

> **핵심 질문:** Two-stage detection의 RoI pooling과 RoI Align의 차이는 왜 mAP 차이로 이어지는가? YOLO의 grid-based prediction이 왜 속도에 유리한가? Focal Loss $-\alpha(1-p_t)^\gamma \log p_t$ 의 $\gamma$ 는 어떤 class imbalance 문제를 해결하는가? U-Net의 skip connection이 medical imaging에 왜 특히 효과적인가? SimCLR의 contrastive loss가 어떻게 label 없이 representation을 학습하는가?

<details>
<summary><b>Classification부터 Self-Supervised까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Image Classification의 전형적 Pipeline](./ch6-applications/01-classification.md) | ImageNet ILSVRC benchmark, Top-1/Top-5 accuracy, **Cross-entropy loss** $L = -\sum_c y_c \log p_c$ 에서 softmax gradient가 $p - y$ 로 단순해지는 유도, label smoothing (Szegedy 2016) $y_c' = (1-\epsilon)y_c + \epsilon/K$ 의 calibration 효과, MixUp·CutMix data augmentation의 regularization 관점 |
| [02. Two-Stage Detection (Faster R-CNN)](./ch6-applications/02-two-stage-detection.md) | **Ren 2015** — **RPN** (Region Proposal Network) + 분류·회귀 head. Anchor box 디자인 (3 scale × 3 ratio = 9 anchors). **RoI Pooling** (Fast R-CNN) → **RoI Align** (Mask R-CNN, He 2017) — bilinear interpolation으로 sub-pixel precision, mask AP 약 +3. Non-Maximum Suppression (NMS) 알고리즘과 soft-NMS 개선 |
| [03. One-Stage Detection — YOLO, RetinaNet](./ch6-applications/03-one-stage-detection.md) | **YOLO** (Redmon 2016) — $S \times S$ grid cell별 bbox 예측, 실시간 45 FPS. YOLOv3/v4/v5/v7의 발전. **Focal Loss** (Lin 2017) $FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t$ — easy example ($p_t \to 1$) 의 기여를 $(1-p_t)^\gamma$ 로 감쇠, class imbalance (1:1000 전경:배경) 해결. $\gamma = 2$ 의 실전적 기원 |
| [04. Semantic Segmentation — FCN, U-Net, DeepLab](./ch6-applications/04-segmentation.md) | **FCN** (Long 2015) — 최초 end-to-end dense prediction. **U-Net** (Ronneberger 2015) — symmetric encoder-decoder + skip connection, 적은 데이터에서도 학습 가능하여 medical imaging 표준. **DeepLab v3+** (Chen 2018) — atrous conv + ASPP + decoder, Cityscapes SOTA. IoU metric과 Dice loss의 불균형 처리 |
| [05. Self-Supervised Learning with CNNs](./ch6-applications/05-self-supervised.md) | **Pretext tasks**: Jigsaw (Noroozi 2016), Rotation prediction (Gidaris 2018), Colorization (Zhang 2016). **SimCLR** (Chen 2020) — contrastive loss $L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$, strong augmentation이 핵심. **MoCo** (He 2019) — momentum encoder + queue, batch size 의존성 제거. Transfer learning을 위한 CNN feature의 범용성 |

</details>

<br/>

### 🔹 Chapter 7: CNN의 이론적 한계와 Vision Transformer

> **핵심 질문:** CNN의 inductive bias (translation equivariance, locality, hierarchy) 는 언제 강점이고 언제 약점인가? Adversarial example $\|x - x'\| < \epsilon$ 이 prediction을 바꾸는 현상의 수학적 메커니즘은? Spectral bias가 high-frequency 학습을 어떻게 방해하는가? ViT의 patch embedding은 CNN의 첫 conv와 구조적으로 동일한가 다른가?

<details>
<summary><b>Inductive Bias부터 CNN-Transformer 수렴까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. CNN의 Inductive Bias 장단점](./ch7-limits-vit/01-inductive-bias.md) | Translation equivariance · locality (small kernel) · hierarchy (pooling) 가 **vision 에 적합한 prior**. 데이터가 부족할 때 강점 — ImageNet-1k에서 ResNet-50 76% vs ViT-B 77% (추가 증강 필요). **Dosovitskiy 2021** — JFT-300M 에서 ViT가 CNN 능가, 하지만 작은 dataset에서는 inductive bias 없는 ViT가 불리. 데이터-prior trade-off의 정량적 경계 |
| [02. Adversarial Examples와 CNN의 취약성](./ch7-limits-vit/02-adversarial.md) | **Szegedy 2014** 의 발견, **Goodfellow 2014 FGSM** — $x' = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))$ 로 지각 불가능한 perturbation이 prediction 완전 변경. $\ell_\infty$ ball 내 공격, **PGD** (Madry 2018) iterative 공격. Robustness vs accuracy trade-off (Tsipras 2019), adversarial training의 비용. PyTorch로 pretrained ResNet-50에 FGSM 공격 재현 |
| [03. Spectral Bias와 CNN](./ch7-limits-vit/03-spectral-bias.md) | **Rahaman 2019** "On the Spectral Bias of Neural Networks" — NN이 **low-frequency 를 먼저 학습**, high-frequency 학습에 오래 걸림. Target function의 Fourier 분해로 frequency 별 수렴 속도 측정. CNN에서도 동일 현상, **NeRF의 Fourier feature** $\gamma(x) = [\sin(2^k \pi x), \cos(2^k \pi x)]_{k=0}^L$ 로 high-frequency 학습 촉진 |
| [04. Vision Transformer와 하이브리드](./ch7-limits-vit/04-vit.md) | **Dosovitskiy 2021** — 이미지를 $16 \times 16$ patch로 split, linear embed 후 Transformer encoder에 투입. **ViT의 patch embedding = stride $16$의 첫 conv** 로 해석 가능. **Swin Transformer** (Liu 2021) — window-based self-attention + hierarchical (CNN-like), SOTA. **ConvNeXt** (Liu 2022) 의 반격, MAE (He 2022)의 masked image modeling. "CNN vs Transformer" → 설계 원리의 수렴으로 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-02의 translation equivariance 증명에 group action 공리 체크를 보강했고, Ch3-02 Effective RF 실험에서 **power iteration으로 Hessian-vector product** 대신 **input에 대한 gradient의 spatial 분산**을 직접 측정하도록 개선했으며, Ch4-02 gradient flow 실험이 plain-56 / ResNet-56 50k step까지 확장되었습니다. Ch5-04 ConvNeXt 전체 구현이 `timm==0.9.10` 기반으로 리팩토링되었고, 11-섹션 문서 골격이 전체 34개 문서에서 일관됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현**을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다. (전체 79개 정리 중 핵심만 발췌)

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Translation Equivariance** | $(I * K)$ 가 $T_a (I * K) = (T_a I) * K$ 만족 — convolution 정의로 증명 | [Ch1-02](./ch1-convolution/02-translation-equivariance.md) |
| **Convolution Theorem** | $\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$ — FFT-based conv $O(n \log n)$ | [Ch1-05](./ch1-convolution/05-convolution-theorem.md) |
| **Circulant Matrix 대각화** | Circulant matrix의 고유벡터 = DFT basis → $T_K = F^* \Lambda F$ | [Ch1-04](./ch1-convolution/04-toeplitz-matrix.md) |
| **Group Equivariant CNN** | $G$-equivariant conv가 weight sharing을 group orbit에 확장 | [Ch1-03](./ch1-convolution/03-group-equivariant-cnn.md) |
| **Conv Backward Rule** | $\partial L/\partial K = X * (\partial L/\partial Y)$, $\partial L/\partial X = (\partial L/\partial Y) *_{\text{full}} \tilde K$ | [Ch2-01](./ch2-cnn-ops/01-conv-forward-backward.md) |
| **Depthwise Separable 파라미터 비율** | $\frac{1}{C_{out}} + \frac{1}{k^2}$ — MobileNet의 8배 절약 | [Ch2-05](./ch2-cnn-ops/05-depthwise-separable.md) |
| **Theoretical RF 재귀 공식** | $RF_L = 1 + \sum_l (k_l - 1) \prod_{i<l} s_i$ | [Ch3-01](./ch3-receptive-field/01-theoretical-rf.md) |
| **Effective RF Gaussian (Luo 2016)** | ERF $\approx$ Gaussian, ratio $ERF/RF \sim 1/\sqrt L$ | [Ch3-02](./ch3-receptive-field/02-effective-rf.md) |
| **Dilated Exponential RF (Yu–Koltun 2016)** | Exponential dilation $d_l = 2^{l-1}$ 로 $RF = 2^L$ | [Ch3-03](./ch3-receptive-field/03-dilated-rf.md) |
| **ResNet Gradient Identity** | $\partial y/\partial x = I + \partial F/\partial x$ — gradient highway | [Ch4-02](./ch4-resnet/02-gradient-flow.md) |
| **He 2016 Plain vs ResNet Degradation** | Plain-56의 수렴 실패 · ResNet-56의 수렴 재현 | [Ch4-02](./ch4-resnet/02-gradient-flow.md) |
| **Identity Approximation (He 2016)** | Deep residual net이 shallow net을 id-layer 추가로 복제 가능 | [Ch4-03](./ch4-resnet/03-identity-approximation.md) |
| **DenseNet Feature Reuse** | $L$ 층에서 $L(L+1)/2$ connections, 파라미터 $1/3$ 절약 | [Ch4-04](./ch4-resnet/04-densenet.md) |
| **VGG 3×3 Stack 이득** | $3\!\times\!3$ ×2 $\Leftrightarrow$ $5\!\times\!5$ RF, 파라미터 28%↓ + 비선형성 2× | [Ch5-01](./ch5-modern-cnn/01-vgg-depth.md) |
| **Inception 1×1 Reduction** | $O(k^2 C^2) \to O(CC' + k^2 C'^2)$ dimensionality reduction | [Ch5-02](./ch5-modern-cnn/02-inception.md) |
| **EfficientNet Compound Scaling** | $d \cdot w^2 \cdot r^2 \approx 2$ constraint 하 optimal scaling | [Ch5-03](./ch5-modern-cnn/03-efficientnet.md) |
| **Focal Loss (Lin 2017)** | $FL = -\alpha(1-p_t)^\gamma \log p_t$ — class imbalance 해결 | [Ch6-03](./ch6-applications/03-one-stage-detection.md) |
| **SimCLR Contrastive Loss** | $L = -\log \exp(sim/\tau)/\sum \exp(sim/\tau)$ — label 없이 representation | [Ch6-05](./ch6-applications/05-self-supervised.md) |
| **FGSM Adversarial (Goodfellow 2014)** | $x' = x + \epsilon \cdot \text{sign}(\nabla_x L)$ 로 prediction 역전 | [Ch7-02](./ch7-limits-vit/02-adversarial.md) |
| **Rahaman 2019 Spectral Bias** | NN이 low-frequency를 먼저 학습하는 현상의 Fourier 측정 | [Ch7-03](./ch7-limits-vit/03-spectral-bias.md) |
| **ViT Data-Prior Trade-off** | JFT-300M에서 ViT > CNN, ImageNet-1k에서는 역전 | [Ch7-04](./ch7-limits-vit/04-vit.md) |

> 💡 **챕터별 문서·정리/정의 수**: Ch1(5문서, 30 정리·정의) · Ch2(5문서, 23) · Ch3(4문서, 12) · Ch4(6문서, 23) · Ch5(5문서, 20) · Ch6(5문서, 31) · Ch7(4문서, 19) — 합계 **34문서 + 158 정리·정의 + 38 엄밀한 $\square$ 증명 + 119개 PyTorch 실험**, 약 **16,717 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
tqdm==4.66.0
torch==2.1.0
torchvision==0.16.0
timm==0.9.10          # 현대 아키텍처 모음 (ConvNeXt, EfficientNet, ViT)
scikit-image==0.22.0  # 이미지 처리 / FFT 비교
Pillow==10.0.0
jupyter==1.0.0
# 선택 사항
einops==0.7.0         # Ch7 ViT 구현
grad-cam==1.4.8       # Ch3 Grad-CAM 시각화
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 tqdm==4.66.0 \
            torch==2.1.0 torchvision==0.16.0 timm==0.9.10 \
            scikit-image==0.22.0 Pillow==10.0.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Plain-56 vs ResNet-56 gradient flow (Ch4-02, He 2016 Fig 1 재현)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PlainBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.conv(x) + x)   # identity shortcut

def build_net(block_type, depth=20, c=16):
    blocks = [block_type(c) for _ in range(depth)]
    return nn.Sequential(
        nn.Conv2d(3, c, 3, padding=1),
        *blocks,
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10),
    )

# Plain-20과 ResNet-20을 CIFAR-10에서 훈련 후
# 각 block의 gradient norm 분포 비교
# → Plain은 깊어질수록 norm이 기하급수적으로 감소 (vanishing)
# → ResNet은 거의 일정 유지 (identity highway)

# 대표 실험 ② — Effective Receptive Field 측정 (Ch3-02, Luo 2016 재현)
def measure_erf(model, input_size=(3, 224, 224), n_samples=100):
    """center pixel gradient로 input에 대한 기울기 측정 → ERF"""
    erf = torch.zeros(input_size[1:])
    for _ in range(n_samples):
        x = torch.randn(1, *input_size, requires_grad=True)
        y = model(x)
        H, W = y.shape[-2:]
        y[0, :, H // 2, W // 2].sum().backward()
        erf += x.grad[0].abs().sum(0).detach()
    return erf / n_samples

erf = measure_erf(build_net(ResBlock, depth=20))
plt.imshow(erf.cpu(), cmap='hot'); plt.title('Effective RF (≈ Gaussian, Luo 2016)')
plt.colorbar(); plt.show()
# → 이론 RF는 꽉 찬 정사각형이지만 실제 ERF는 중심 집중 Gaussian

# 대표 실험 ③ — Translation Equivariance 수치 검증 (Ch1-02)
def test_translation_equivariance(conv, x, shift=(5, 7)):
    """T_a (conv x) == conv (T_a x) 를 수치로 확인"""
    y1 = conv(x)                             # conv then shift
    y1_shifted = torch.roll(y1, shifts=shift, dims=(-2, -1))
    x_shifted = torch.roll(x, shifts=shift, dims=(-2, -1))
    y2 = conv(x_shifted)                     # shift then conv
    return (y1_shifted - y2).abs().max().item()  # ≈ 0 (boundary 제외)

# 대표 실험 ④ — Reddi-free FGSM adversarial (Ch7-02, Goodfellow 2014)
def fgsm_attack(model, x, y, epsilon=0.03):
    x = x.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    return (x + epsilon * x.grad.sign()).clamp(0, 1)
# → pretrained ResNet-50에 적용해 ε=8/255 에서 분류 정확도 75% → 6% 급락
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 설계가 CNN에 필수인가** | Equivariance·RF·gradient flow·parameter 효율과의 연결 |
| 3 | 📐 **수학적 선행 조건** | LA · NN Theory · FA · Opt · Reg 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | 시각적 매개체 (filter 시각화·RF 경로·gradient flow·feature map) |
| 5 | ✏️ **엄밀한 정의·정리** | Convolution·equivariance·RF 공식·residual block |
| 6 | 🔬 **증명 또는 수학적 유도** | Equivariance 증명 · RF 공식 유도 · ResNet gradient flow · Focal Loss |
| 7 | 💻 **실험 재현** | Toeplitz conv · ERF 측정 · Plain vs ResNet · Grad-CAM · FGSM |
| 8 | 🔗 **이론과 실전의 간극** | 이 결과가 실제 ImageNet·COCO·Segmentation을 얼마나 설명하는가 |
| 9 | ⚖️ **가정과 한계** | Inductive bias의 양면성 · adversarial 취약성 · data 크기 의존성 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현·논문 비평 문제 |

> 📚 **연습문제 총 98개**: 대부분 문서가 3문제(기초/심화/논문 비평), 일부(Ch4-05, Ch4-06, Ch6-03, Ch7-02)는 2문제. 모든 문제에 `<details>` 펼침 해설 포함. Translation equivariance 손 증명부터 RF 공식 재유도, ResNet gradient 추적 재구현, ViT patch embedding 대 CNN 첫 conv 비교까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 492줄(정의·증명·코드·연습문제 포함) 기준 **약 55분~1시간 15분**. 전체 34문서는 약 **31~42시간** 상당 (증명 재구성·실험 재현 포함 시 50시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Conv2d는 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Discrete Convolution 정의
       Ch1-02  Translation Equivariance 증명
Day 2  Ch2-01  Conv Forward/Backward
       Ch2-02  Pooling의 역할
Day 3  Ch3-01  Theoretical RF 계산
       Ch3-02  Effective RF (Luo 2016)
Day 4  Ch4-01  Residual Block 정의
       Ch4-02  Gradient Flow 분석
Day 5  Ch5-01  VGG와 Depth
       Ch5-03  EfficientNet Compound Scaling
Day 6  Ch6-01  Classification Pipeline
       Ch6-03  Focal Loss
Day 7  Ch7-01  CNN Inductive Bias
       Ch7-04  ViT 비교
```

</details>

<details>
<summary><b>🟡 "Equivariance와 ResNet의 수학을 완전히 정복한다" — 이론 집중 (2주, 약 20~24시간)</b></summary>

<br/>

```
1주차 — Convolution의 대칭성
  Day 1    Ch1-01~02   Discrete conv + equivariance 증명 꼼꼼히
  Day 2    Ch1-03      Group Equivariant CNN (Cohen 2016)
  Day 3    Ch1-04~05   Toeplitz matrix + Convolution Theorem + FFT
  Day 4    Ch2-01~02   Backward rule 손 유도
  Day 5    Ch2-04~05   Dilated conv + Depthwise Separable 파라미터 분석
  Day 6-7  Ch3-01~04   RF 이론 + ERF 측정 + Dilated RF + Segmentation

2주차 — ResNet과 깊이의 수학
  Day 1    Ch4-01      Residual block forward
  Day 2-3  Ch4-02~03   Gradient flow + Identity approximation
  Day 4    Ch4-04~05   DenseNet + Highway
  Day 5    Ch4-06      Stochastic Depth
  Day 6    Ch5-01~02   VGG depth + Inception 1×1
  Day 7    Ch5-03~04   EfficientNet scaling + ConvNeXt
```

</details>

<details>
<summary><b>🔴 "CNN의 수학을 완전 정복한다" — 전체 정복 (10주, 약 30~40시간 + 실험 재현 10~15시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — Convolution 수학 기초
         → Equivariance 손 증명, Toeplitz 분해 구현, FFT vs direct conv

2주차   Chapter 2 전체 — CNN 기본 연산
         → Multi-channel backward 손 유도
         → Checkerboard artifact 재현
         → Depthwise Separable로 MobileNet 재구현

3주차   Chapter 3 전체 — Receptive Field
         → Theoretical RF 재귀 공식 손 계산
         → ERF Gaussian 분포 확인 (Luo 2016 재현)
         → Dilated conv로 WaveNet-like 구조

4주차   Chapter 4 (1~3) — ResNet 핵심
         → Gradient highway 수식 유도
         → Plain-56 vs ResNet-56 gradient norm 측정
         → He 2016 Fig 1 재현

5주차   Chapter 4 (4~6) — 심화
         → DenseNet concat vs ResNet add 비교
         → Stochastic Depth로 1001-layer 훈련 시도

6주차   Chapter 5 전체 — 현대 아키텍처
         → VGG vs ResNet 파라미터·FLOPs 비교
         → Inception 1×1 효과 측정
         → EfficientNet-B0~B3 재현
         → ConvNeXt로 Swin Transformer에 도전

7주차   Chapter 6 (1~3) — Classification·Detection
         → Label smoothing·MixUp 효과
         → Faster R-CNN 구현 / YOLO 재현
         → Focal Loss γ 실험

8주차   Chapter 6 (4~5) — Segmentation·Self-Supervised
         → U-Net으로 medical dataset
         → SimCLR 사전훈련 → linear probing

9주차   Chapter 7 (1~3) — CNN 한계
         → Inductive bias 양면성 실험
         → FGSM/PGD 공격 재현
         → Spectral bias 측정

10주차  Chapter 7-04 + 종합 — CNN vs ViT
         → ViT-B 구현
         → ConvNeXt로 CNN 반격
         → "CNN vs Transformer" 설계 원리 수렴 정리
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Toeplitz·circulant·고유분해·SVD | **Ch1-04** (Convolution as Toeplitz), Ch1-05 (DFT 대각화) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | Fourier transform, convolution theorem, Sobolev space | **Ch1-05** (Convolution Theorem), Ch7-03 (Spectral Bias) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | UAT, Backprop, Xavier·He init, 아키텍처 기초 | **전체 레포의 전제**, Ch2-01 (Backprop), Ch4-03 (UAT) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | GD·SGD·Adam·gradient flow·landscape | **Ch4-02** (Gradient flow), Ch5 (현대 아키텍처 훈련) |
| [regularization-theory-deep-dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive) | L2·Dropout·BatchNorm·Early Stopping | **Ch4-01** (pre-activation), Ch5 (BN in ConvNeXt·Inception) |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | VC dim·Rademacher·NTK·Double Descent | **Ch1** (parameter sharing↔VC), Ch7-01 (inductive bias) |
| [vision-transformer-deep-dive](https://github.com/iq-ai-lab/vision-transformer-deep-dive) *(다음)* | Self-Attention · ViT · Swin · MAE · DINOv2 | **Ch7-04** 이후 직접 연결 |

> 💡 이 레포는 **"CNN이 왜 vision task에 적합하고 언제 그 bias가 무너지는가"** 에 집중합니다. Linear Algebra에서 Toeplitz matrix와 circulant 대각화를 익히고, NN Theory에서 backpropagation과 초기화를 익힌 후 오면 Chapter 1 (Equivariance) 과 Chapter 4 (ResNet) 의 증명이 훨씬 자연스럽습니다. Vision Transformer (다음 레포) 는 이 레포 Chapter 7-04 (ViT) 을 전제로 시작합니다.

---

## 📖 Reference

### 🏛️ CNN 고전·기초
- **Gradient-Based Learning Applied to Document Recognition** (LeCun, Bottou, Bengio, Haffner, 1998) — **LeNet 원전**
- **ImageNet Classification with Deep Convolutional Neural Networks** (Krizhevsky, Sutskever, Hinton, 2012) — **AlexNet 원전**
- **Very Deep Convolutional Networks for Large-Scale Image Recognition** (Simonyan & Zisserman, 2014) — **VGG**
- **Going Deeper with Convolutions** (Szegedy et al., 2014) — **GoogLeNet / Inception v1**
- **Rethinking the Inception Architecture for Computer Vision** (Szegedy et al., 2016) — Inception v2/v3
- **Deep Learning** (Goodfellow, Bengio, Courville, 2016) — Chapter 9 CNN 표준

### 🎨 Equivariance · Group CNN · Spectral
- **Group Equivariant Convolutional Networks** (Cohen & Welling, 2016) — **G-CNN 원전**
- **Steerable CNNs** (Cohen & Welling, 2017)
- **General E(2)-Equivariant Steerable CNNs** (Weiler & Cesa, 2019)
- **A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction** (Wiatowski & Bölcskei, 2018)
- **On the Spectral Bias of Neural Networks** (Rahaman et al., 2019)

### 🏔️ ResNet · Skip Connection
- **Deep Residual Learning for Image Recognition** (He, Zhang, Ren, Sun, 2016) — **ResNet 원전**
- **Identity Mappings in Deep Residual Networks** (He, Zhang, Ren, Sun, 2016b) — Pre-activation
- **Densely Connected Convolutional Networks** (Huang, Liu, Van Der Maaten, Weinberger, 2017) — **DenseNet**
- **Highway Networks** (Srivastava, Greff, Schmidhuber, 2015)
- **Deep Networks with Stochastic Depth** (Huang, Sun, Liu, Sedra, Weinberger, 2016)
- **Wide Residual Networks** (Zagoruyko & Komodakis, 2016)

### 🔭 Receptive Field
- **Understanding the Effective Receptive Field in Deep Convolutional Neural Networks** (Luo, Li, Urtasun, Zemel, 2016) — **ERF 원전**
- **Multi-Scale Context Aggregation by Dilated Convolutions** (Yu & Koltun, 2016) — **Dilated Conv 원전**
- **WaveNet: A Generative Model for Raw Audio** (van den Oord et al., 2016)

### 🏗️ 효율적 아키텍처
- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** (Howard et al., 2017)
- **Xception: Deep Learning with Depthwise Separable Convolutions** (Chollet, 2017)
- **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices** (Zhang et al., 2018)
- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** (Tan & Le, 2019) — **Compound Scaling**
- **Designing Network Design Spaces** (Radosavovic et al., 2020) — **RegNet**
- **A ConvNet for the 2020s** (Liu et al., 2022) — **ConvNeXt**

### 🔍 Object Detection · Segmentation
- **Rich Feature Hierarchies for Accurate Object Detection** (Girshick et al., 2014) — R-CNN
- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** (Ren, He, Girshick, Sun, 2015)
- **You Only Look Once: Unified, Real-Time Object Detection** (Redmon et al., 2016) — **YOLO**
- **Focal Loss for Dense Object Detection** (Lin, Goyal, Girshick, He, Dollar, 2017) — **RetinaNet + Focal Loss**
- **Fully Convolutional Networks for Semantic Segmentation** (Long, Shelhamer, Darrell, 2015) — **FCN**
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger, Fischer, Brox, 2015)
- **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation** (Chen et al., 2018) — **DeepLab v3+**
- **Mask R-CNN** (He, Gkioxari, Dollar, Girshick, 2017)

### 🌱 Self-Supervised
- **Unsupervised Representation Learning by Predicting Image Rotations** (Gidaris, Singh, Komodakis, 2018)
- **A Simple Framework for Contrastive Learning of Visual Representations** (Chen, Kornblith, Norouzi, Hinton, 2020) — **SimCLR**
- **Momentum Contrast for Unsupervised Visual Representation Learning** (He, Fan, Wu, Xie, Girshick, 2019) — **MoCo**
- **Masked Autoencoders Are Scalable Vision Learners** (He et al., 2022) — **MAE**

### ⚔️ Adversarial · Robustness · ViT
- **Explaining and Harnessing Adversarial Examples** (Goodfellow, Shlens, Szegedy, 2014) — **FGSM**
- **Towards Deep Learning Models Resistant to Adversarial Attacks** (Madry et al., 2018) — **PGD**
- **Robustness May Be at Odds with Accuracy** (Tsipras et al., 2019)
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2021) — **ViT**
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., 2021)

### 🧩 기타 설계 · 이론
- **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** (Ioffe & Szegedy, 2015)
- **How Does Batch Normalization Help Optimization?** (Santurkar, Tsipras, Ilyas, Madry, 2018)
- **Deconvolution and Checkerboard Artifacts** (Odena, Dumoulin, Olah, 2016)
- **Visualizing and Understanding Convolutional Networks** (Zeiler & Fergus, 2014)
- **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization** (Selvaraju et al., 2017)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`nn.Conv2d(3, 64, 3, padding=1)`을 쌓는 것과 — group action으로 translation equivariance $T_a(f*g)=(T_a f)*g$ 를 증명 · Luo 2016으로 ERF가 왜 Gaussian인지 · He 2016으로 $\partial y/\partial x = I + \partial F/\partial x$ 가 어떻게 gradient highway를 만드는지 · Tan & Le 2019로 compound scaling 제약 $\alpha\beta^2\gamma^2 \approx 2$ 유도 · Dosovitskiy 2021로 ViT가 언제 CNN을 이기는지 — 이 모든 '왜'를 직접 유도할 수 있는 것은 다르다"*

</div>
